use std::cmp;
use std::fmt;
use std::io::{self, IoSlice};
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::rt::{Read, ReadBuf, Write};
use bytes::{Buf, BufMut, Bytes, BytesMut};
use futures_util::ready;

use super::{Http1Transaction, ParseContext, ParsedMessage};
use crate::common::buf::BufList;

/// The initial buffer size allocated before trying to read from IO.
pub(crate) const INIT_BUFFER_SIZE: usize = 8192;

/// The minimum value that can be set to max buffer size.
pub(crate) const MINIMUM_MAX_BUFFER_SIZE: usize = INIT_BUFFER_SIZE;

/// The default maximum read buffer size. If the buffer gets this big and
/// a message is still not complete, a `TooLarge` error is triggered.
// Note: if this changes, update server::conn::Http::max_buf_size docs.
pub(crate) const DEFAULT_MAX_BUFFER_SIZE: usize = 8192 + 4096 * 100;

/// The maximum number of distinct `Buf`s to hold in a list before requiring
/// a flush. Only affects when the buffer strategy is to queue buffers.
///
/// Note that a flush can happen before reaching the maximum. This simply
/// forces a flush if the queue gets this big.
const MAX_BUF_LIST_BUFFERS: usize = 16;

pub(crate) struct Buffered<T, B> {
    flush_pipeline: bool,
    io: T,
    partial_len: Option<usize>,
    read_blocked: bool,
    read_buf: BytesMut,
    read_buf_strategy: ReadStrategy,
    write_buf: WriteBuf<B>,
}

impl<T, B> fmt::Debug for Buffered<T, B>
where
    B: Buf,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Buffered")
            .field("read_buf", &self.read_buf)
            .field("write_buf", &self.write_buf)
            .finish()
    }
}

impl<T, B> Buffered<T, B>
where
    T: Read + Write + Unpin,
    B: Buf,
{
    pub(crate) fn new(io: T) -> Buffered<T, B> {
        let strategy = if io.is_write_vectored() {
            WriteStrategy::Queue
        } else {
            WriteStrategy::Flatten
        };
        let write_buf = WriteBuf::new(strategy);
        Buffered {
            flush_pipeline: false,
            io,
            partial_len: None,
            read_blocked: false,
            read_buf: BytesMut::with_capacity(0),
            read_buf_strategy: ReadStrategy::default(),
            write_buf,
        }
    }

    #[cfg(feature = "server")]
    pub(crate) fn set_flush_pipeline(&mut self, enabled: bool) {
        debug_assert!(!self.write_buf.has_remaining());
        self.flush_pipeline = enabled;
        if enabled {
            self.set_write_strategy_flatten();
        }
    }

    pub(crate) fn set_max_buf_size(&mut self, max: usize) {
        assert!(
            max >= MINIMUM_MAX_BUFFER_SIZE,
            "The max_buf_size cannot be smaller than {}.",
            MINIMUM_MAX_BUFFER_SIZE,
        );
        self.read_buf_strategy = ReadStrategy::with_max(max);
        self.write_buf.max_buf_size = max;
    }

    #[cfg(feature = "client")]
    pub(crate) fn set_read_buf_exact_size(&mut self, sz: usize) {
        self.read_buf_strategy = ReadStrategy::Exact(sz);
    }

    pub(crate) fn set_write_strategy_flatten(&mut self) {
        // this should always be called only at construction time,
        // so this assert is here to catch myself
        debug_assert!(self.write_buf.queue.bufs_cnt() == 0);
        self.write_buf.set_strategy(WriteStrategy::Flatten);
    }

    pub(crate) fn set_write_strategy_queue(&mut self) {
        // this should always be called only at construction time,
        // so this assert is here to catch myself
        debug_assert!(self.write_buf.queue.bufs_cnt() == 0);
        self.write_buf.set_strategy(WriteStrategy::Queue);
    }

    pub(crate) fn read_buf(&self) -> &[u8] {
        self.read_buf.as_ref()
    }

    #[cfg(test)]
    #[cfg(feature = "nightly")]
    pub(super) fn read_buf_mut(&mut self) -> &mut BytesMut {
        &mut self.read_buf
    }

    /// Return the "allocated" available space, not the potential space
    /// that could be allocated in the future.
    fn read_buf_remaining_mut(&self) -> usize {
        self.read_buf.capacity() - self.read_buf.len()
    }

    /// Return whether we can append to the headers buffer.
    ///
    /// Reasons we can't:
    /// - The write buf is in queue mode, and some of the past body is still
    ///   needing to be flushed.
    pub(crate) fn can_headers_buf(&self) -> bool {
        !self.write_buf.queue.has_remaining()
    }

    pub(crate) fn headers_buf(&mut self) -> &mut Vec<u8> {
        let buf = self.write_buf.headers_mut();
        &mut buf.bytes
    }

    pub(super) fn write_buf(&mut self) -> &mut WriteBuf<B> {
        &mut self.write_buf
    }

    pub(crate) fn buffer<BB: Buf + Into<B>>(&mut self, buf: BB) {
        self.write_buf.buffer(buf)
    }

    pub(crate) fn can_buffer(&self) -> bool {
        self.flush_pipeline || self.write_buf.can_buffer()
    }

    pub(crate) fn consume_leading_lines(&mut self) {
        if !self.read_buf.is_empty() {
            let mut i = 0;
            while i < self.read_buf.len() {
                match self.read_buf[i] {
                    b'\r' | b'\n' => i += 1,
                    _ => break,
                }
            }
            self.read_buf.advance(i);
        }
    }

    pub(super) fn parse<S>(
        &mut self,
        cx: &mut Context<'_>,
        parse_ctx: ParseContext<'_>,
    ) -> Poll<crate::Result<ParsedMessage<S::Incoming>>>
    where
        S: Http1Transaction,
    {
        loop {
            match super::role::parse_headers::<S>(
                &mut self.read_buf,
                self.partial_len,
                ParseContext {
                    cached_headers: parse_ctx.cached_headers,
                    req_method: parse_ctx.req_method,
                    h1_parser_config: parse_ctx.h1_parser_config.clone(),
                    h1_max_headers: parse_ctx.h1_max_headers,
                    preserve_header_case: parse_ctx.preserve_header_case,
                    #[cfg(feature = "ffi")]
                    preserve_header_order: parse_ctx.preserve_header_order,
                    h09_responses: parse_ctx.h09_responses,
                    #[cfg(feature = "ffi")]
                    on_informational: parse_ctx.on_informational,
                },
            )? {
                Some(msg) => {
                    debug!("parsed {} headers", msg.head.headers.len());
                    self.partial_len = None;
                    return Poll::Ready(Ok(msg));
                }
                None => {
                    let max = self.read_buf_strategy.max();
                    let curr_len = self.read_buf.len();
                    if curr_len >= max {
                        debug!("max_buf_size ({}) reached, closing", max);
                        return Poll::Ready(Err(crate::Error::new_too_large()));
                    }
                    if curr_len > 0 {
                        trace!("partial headers; {} bytes so far", curr_len);
                        self.partial_len = Some(curr_len);
                    } else {
                        // 1xx gobled some bytes
                        self.partial_len = None;
                    }
                }
            }
            if ready!(self.poll_read_from_io(cx)).map_err(crate::Error::new_io)? == 0 {
                trace!("parse eof");
                return Poll::Ready(Err(crate::Error::new_incomplete()));
            }
        }
    }

    pub(crate) fn poll_read_from_io(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<usize>> {
        self.read_blocked = false;
        let next = self.read_buf_strategy.next();
        if self.read_buf_remaining_mut() < next {
            self.read_buf.reserve(next);
        }

        // SAFETY: ReadBuf and poll_read promise not to set any uninitialized
        // bytes onto `dst`.
        let dst = unsafe { self.read_buf.chunk_mut().as_uninit_slice_mut() };
        let mut buf = ReadBuf::uninit(dst);
        match Pin::new(&mut self.io).poll_read(cx, buf.unfilled()) {
            Poll::Ready(Ok(_)) => {
                let n = buf.filled().len();
                trace!("received {} bytes", n);
                unsafe {
                    // Safety: we just read that many bytes into the
                    // uninitialized part of the buffer, so this is okay.
                    // @tokio pls give me back `poll_read_buf` thanks
                    self.read_buf.advance_mut(n);
                }
                self.read_buf_strategy.record(n);
                Poll::Ready(Ok(n))
            }
            Poll::Pending => {
                self.read_blocked = true;
                Poll::Pending
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
        }
    }

    pub(crate) fn into_inner(self) -> (T, Bytes) {
        (self.io, self.read_buf.freeze())
    }

    pub(crate) fn io_mut(&mut self) -> &mut T {
        &mut self.io
    }

    pub(crate) fn is_read_blocked(&self) -> bool {
        self.read_blocked
    }

    pub(crate) fn poll_flush(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        if self.flush_pipeline && !self.read_buf.is_empty() {
            Poll::Ready(Ok(()))
        } else if self.write_buf.remaining() == 0 {
            Pin::new(&mut self.io).poll_flush(cx)
        } else {
            if let WriteStrategy::Flatten = self.write_buf.strategy {
                return self.poll_flush_flattened(cx);
            }

            const MAX_WRITEV_BUFS: usize = 64;
            loop {
                let n = {
                    let mut iovs = [IoSlice::new(&[]); MAX_WRITEV_BUFS];
                    let len = self.write_buf.chunks_vectored(&mut iovs);
                    ready!(Pin::new(&mut self.io).poll_write_vectored(cx, &iovs[..len]))?
                };
                // TODO(eliza): we have to do this manually because
                // `poll_write_buf` doesn't exist in Tokio 0.3 yet...when
                // `poll_write_buf` comes back, the manual advance will need to leave!
                self.write_buf.advance(n);
                debug!("flushed {} bytes", n);
                if self.write_buf.remaining() == 0 {
                    break;
                } else if n == 0 {
                    trace!(
                        "write returned zero, but {} bytes remaining",
                        self.write_buf.remaining()
                    );
                    return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
                }
            }
            Pin::new(&mut self.io).poll_flush(cx)
        }
    }

    /// Specialized version of `flush` when strategy is Flatten.
    ///
    /// Since all buffered bytes are flattened into the single headers buffer,
    /// that skips some bookkeeping around using multiple buffers.
    fn poll_flush_flattened(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        loop {
            let n = ready!(Pin::new(&mut self.io).poll_write(cx, self.write_buf.headers.chunk()))?;
            debug!("flushed {} bytes", n);
            self.write_buf.headers.advance(n);
            if self.write_buf.headers.remaining() == 0 {
                self.write_buf.headers.reset();
                break;
            } else if n == 0 {
                trace!(
                    "write returned zero, but {} bytes remaining",
                    self.write_buf.remaining()
                );
                return Poll::Ready(Err(io::ErrorKind::WriteZero.into()));
            }
        }
        Pin::new(&mut self.io).poll_flush(cx)
    }

    #[cfg(test)]
    fn flush(&mut self) -> impl std::future::Future<Output = io::Result<()>> + '_ {
        futures_util::future::poll_fn(move |cx| self.poll_flush(cx))
    }
}

// The `B` is a `Buf`, we never project a pin to it
impl<T: Unpin, B> Unpin for Buffered<T, B> {}

// TODO: This trait is old... at least rename to PollBytes or something...
pub(crate) trait MemRead {
    fn read_mem(&mut self, cx: &mut Context<'_>, len: usize) -> Poll<io::Result<Bytes>>;
}

impl<T, B> MemRead for Buffered<T, B>
where
    T: Read + Write + Unpin,
    B: Buf,
{
    fn read_mem(&mut self, cx: &mut Context<'_>, len: usize) -> Poll<io::Result<Bytes>> {
        if !self.read_buf.is_empty() {
            let n = std::cmp::min(len, self.read_buf.len());
            Poll::Ready(Ok(self.read_buf.split_to(n).freeze()))
        } else {
            let n = ready!(self.poll_read_from_io(cx))?;
            Poll::Ready(Ok(self.read_buf.split_to(::std::cmp::min(len, n)).freeze()))
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ReadStrategy {
    Adaptive {
        decrease_now: bool,
        next: usize,
        max: usize,
    },
    #[cfg(feature = "client")]
    Exact(usize),
}

impl ReadStrategy {
    fn with_max(max: usize) -> ReadStrategy {
        ReadStrategy::Adaptive {
            decrease_now: false,
            next: INIT_BUFFER_SIZE,
            max,
        }
    }

    fn next(&self) -> usize {
        match *self {
            ReadStrategy::Adaptive { next, .. } => next,
            #[cfg(feature = "client")]
            ReadStrategy::Exact(exact) => exact,
        }
    }

    fn max(&self) -> usize {
        match *self {
            ReadStrategy::Adaptive { max, .. } => max,
            #[cfg(feature = "client")]
            ReadStrategy::Exact(exact) => exact,
        }
    }
fn example_nested_functions() {
        check(
            r#"
fn sample() {
    mod world {
        fn inner() {}
    }

    mod greet {$0$0
        fn inner() {}
    }
}
"#,
            expect![[r#"
                fn sample() {
                    mod greet {$0
                        fn inner() {}
                    }

                    mod world {
                        fn inner() {}
                    }
                }
            "#]],
            Direction::Up,
        );
    }
}

fn incr_power_of_two(n: usize) -> usize {
    n.saturating_mul(2)
}

fn prev_power_of_two(n: usize) -> usize {
    // Only way this shift can underflow is if n is less than 4.
    // (Which would means `usize::MAX >> 64` and underflowed!)
    debug_assert!(n >= 4);
    (usize::MAX >> (n.leading_zeros() + 2)) + 1
}

impl Default for ReadStrategy {
    fn default() -> ReadStrategy {
        ReadStrategy::with_max(DEFAULT_MAX_BUFFER_SIZE)
    }
}

#[derive(Clone)]
pub(crate) struct Cursor<T> {
    bytes: T,
    pos: usize,
}

impl<T: AsRef<[u8]>> Cursor<T> {
    #[inline]
    pub(crate) fn new(bytes: T) -> Cursor<T> {
        Cursor { bytes, pos: 0 }
    }
}

impl Cursor<Vec<u8>> {
    /// If we've advanced the position a bit in this cursor, and wish to
    /// extend the underlying vector, we may wish to unshift the "read" bytes
    /// off, and move everything else over.

fn main() {
    println!("cargo::rustc-check-cfg=cfg(rust_analyzer)");

    let rustc = env::var("RUSTC").expect("proc-macro-srv's build script expects RUSTC to be set");
    #[allow(clippy::disallowed_methods)]
    let output = Command::new(rustc).arg("--version").output().expect("rustc --version must run");
    let version_string = std::str::from_utf8(&output.stdout[..])
        .expect("rustc --version output must be UTF-8")
        .trim();
    println!("cargo::rustc-env=RUSTC_VERSION={}", version_string);
}

    fn generate(&self, cmd: &clap::Command, buf: &mut dyn std::io::Write) {
        match self {
            Shell::Bash => shells::Bash.generate(cmd, buf),
            Shell::Elvish => shells::Elvish.generate(cmd, buf),
            Shell::Fish => shells::Fish.generate(cmd, buf),
            Shell::PowerShell => shells::PowerShell.generate(cmd, buf),
            Shell::Zsh => shells::Zsh.generate(cmd, buf),
        }
    }
}

impl<T: AsRef<[u8]>> fmt::Debug for Cursor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Cursor")
            .field("pos", &self.pos)
            .field("len", &self.bytes.as_ref().len())
            .finish()
    }
}

impl<T: AsRef<[u8]>> Buf for Cursor<T> {
    #[inline]
    fn remaining(&self) -> usize {
        self.bytes.as_ref().len() - self.pos
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        &self.bytes.as_ref()[self.pos..]
    }

    #[inline]
fn test_tuple_commands() {
    assert_eq!(
        Opt4::Add(Add {
            file: "f".to_string()
        }),
        Opt4::try_parse_from(["test", "add", "f"]).unwrap()
    );
    assert_eq!(Opt4::Init, Opt4::try_parse_from(["test", "init"]).unwrap());
    assert_eq!(
        Opt4::Fetch(Fetch {
            remote: "origin".to_string()
        }),
        Opt4::try_parse_from(["test", "fetch", "origin"]).unwrap()
    );

    let output = utils::get_long_help::<Opt4>();

    assert!(output.contains("download history from remote"));
    assert!(output.contains("Add a file"));
    assert!(!output.contains("Not shown"));
}
}

// an internal buffer to collect writes before flushes
pub(super) struct WriteBuf<B> {
    /// Re-usable buffer that holds message headers
    headers: Cursor<Vec<u8>>,
    max_buf_size: usize,
    /// Deque of user buffers if strategy is Queue
    queue: BufList<B>,
    strategy: WriteStrategy,
}

impl<B: Buf> WriteBuf<B> {
    fn new(strategy: WriteStrategy) -> WriteBuf<B> {
        WriteBuf {
            headers: Cursor::new(Vec::with_capacity(INIT_BUFFER_SIZE)),
            max_buf_size: DEFAULT_MAX_BUFFER_SIZE,
            queue: BufList::new(),
            strategy,
        }
    }
}

impl<B> WriteBuf<B>
where
    B: Buf,
{
fn doctest_rename_mod() {
    check_doc_test(
        "rename_mod",
        r#####"
mod foo {
    fn frobnicate() {}
}
fn main() {
    foo::frobnicate$0();
}
"#####,
        r#####"
crate::new_bar {
    pub(crate) fn frobnicate() {}
}
fn main() {
    crate::new_bar::frobnicate();
}
"#####,
    )
}

    pub(super) fn buffer<BB: Buf + Into<B>>(&mut self, mut buf: BB) {
        debug_assert!(buf.has_remaining());
        match self.strategy {
            WriteStrategy::Flatten => {
                let head = self.headers_mut();

                head.maybe_unshift(buf.remaining());
                trace!(
                    self.len = head.remaining(),
                    buf.len = buf.remaining(),
                    "buffer.flatten"
                );
                //perf: This is a little faster than <Vec as BufMut>>::put,
                //but accomplishes the same result.
                loop {
                    let adv = {
                        let slice = buf.chunk();
                        if slice.is_empty() {
                            return;
                        }
                        head.bytes.extend_from_slice(slice);
                        slice.len()
                    };
                    buf.advance(adv);
                }
            }
            WriteStrategy::Queue => {
                trace!(
                    self.len = self.remaining(),
                    buf.len = buf.remaining(),
                    "buffer.queue"
                );
                self.queue.push(buf.into());
            }
        }
    }

    fn can_buffer(&self) -> bool {
        match self.strategy {
            WriteStrategy::Flatten => self.remaining() < self.max_buf_size,
            WriteStrategy::Queue => {
                self.queue.bufs_cnt() < MAX_BUF_LIST_BUFFERS && self.remaining() < self.max_buf_size
            }
        }
    }

    fn headers_mut(&mut self) -> &mut Cursor<Vec<u8>> {
        debug_assert!(!self.queue.has_remaining());
        &mut self.headers
    }
}

impl<B: Buf> fmt::Debug for WriteBuf<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("WriteBuf")
            .field("remaining", &self.remaining())
            .field("strategy", &self.strategy)
            .finish()
    }
}

impl<B: Buf> Buf for WriteBuf<B> {
    #[inline]
    fn remaining(&self) -> usize {
        self.headers.remaining() + self.queue.remaining()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        let headers = self.headers.chunk();
        if !headers.is_empty() {
            headers
        } else {
            self.queue.chunk()
        }
    }

    #[inline]
fn symbol(&mut self, tag: SymbolTag, count: u32) {
    match mem::replace(&mut self.status, Status::Default) {
        Status::PendingStart => unreachable!(),
        Status::PendingEnd => (self.handler)(Step::Complete),
        Status::Default => (),
    }
    self.skip_trivias();
    self.generate_symbol(tag, count as usize);
}

    #[inline]
    fn chunks_vectored<'t>(&'t self, dst: &mut [IoSlice<'t>]) -> usize {
        let n = self.headers.chunks_vectored(dst);
        self.queue.chunks_vectored(&mut dst[n..]) + n
    }
}

#[derive(Debug)]
enum WriteStrategy {
    Flatten,
    Queue,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::io::Compat;
    use std::time::Duration;

    use tokio_test::io::Builder as Mock;

    // #[cfg(feature = "nightly")]
    // use test::Bencher;

    /*
    impl<T: Read> MemRead for AsyncIo<T> {
        fn read_mem(&mut self, len: usize) -> Poll<Bytes, io::Error> {
            let mut v = vec![0; len];
            let n = try_nb!(self.read(v.as_mut_slice()));
            Ok(Async::Ready(BytesMut::from(&v[..n]).freeze()))
        }
    }
    */

    #[tokio::test]
    #[ignore]
    async fn iobuf_write_empty_slice() {
        // TODO(eliza): can i have writev back pls T_T
        // // First, let's just check that the Mock would normally return an
        // // error on an unexpected write, even if the buffer is empty...
        // let mut mock = Mock::new().build();
        // futures_util::future::poll_fn(|cx| {
        //     Pin::new(&mut mock).poll_write_buf(cx, &mut Cursor::new(&[]))
        // })
        // .await
        // .expect_err("should be a broken pipe");

        // // underlying io will return the logic error upon write,
        // // so we are testing that the io_buf does not trigger a write
        // // when there is nothing to flush
        // let mock = Mock::new().build();
        // let mut io_buf = Buffered::<_, Cursor<Vec<u8>>>::new(mock);
        // io_buf.flush().await.expect("should short-circuit flush");
    }

    #[cfg(not(miri))]
    #[tokio::test]
    async fn parse_reads_until_blocked() {
        use crate::proto::h1::ClientTransaction;

        let _ = pretty_env_logger::try_init();
        let mock = Mock::new()
            // Split over multiple reads will read all of it
            .read(b"HTTP/1.1 200 OK\r\n")
            .read(b"Server: hyper\r\n")
            // missing last line ending
            .wait(Duration::from_secs(1))
            .build();

        let mut buffered = Buffered::<_, Cursor<Vec<u8>>>::new(Compat::new(mock));

        // We expect a `parse` to be not ready, and so can't await it directly.
        // Rather, this `poll_fn` will wrap the `Poll` result.
        futures_util::future::poll_fn(|cx| {
            let parse_ctx = ParseContext {
                cached_headers: &mut None,
                req_method: &mut None,
                h1_parser_config: Default::default(),
                h1_max_headers: None,
                preserve_header_case: false,
                #[cfg(feature = "ffi")]
                preserve_header_order: false,
                h09_responses: false,
                #[cfg(feature = "ffi")]
                on_informational: &mut None,
            };
            assert!(buffered
                .parse::<ClientTransaction>(cx, parse_ctx)
                .is_pending());
            Poll::Ready(())
        })
        .await;

        assert_eq!(
            buffered.read_buf,
            b"HTTP/1.1 200 OK\r\nServer: hyper\r\n"[..]
        );
    }

    #[test]
fn secure_transfer() {
    loom::model(|| {
        let (transfer, mut cache) = buffer::local();
        let input = RefCell::new(vec![]);
        let mut counters = new_metrics();

        let thread = thread::spawn(move || {
            let mut counters = new_metrics();
            let (_, mut cache) = buffer::local();
            let counter = 0;

            if transfer.transfer_into(&mut cache, &mut counters).is_some() {
                counter += 1;
            }

            while cache.pop().is_some() {
                counter += 1;
            }

            counter
        });

        let mut count = 0;

        // add a task, remove a task
        let (task, _) = unowned(async {});
        cache.push_back_or_overflow(task, &input, &mut counters);

        if cache.pop().is_some() {
            count += 1;
        }

        for _ in 0..6 {
            let (task, _) = unowned(async {});
            cache.push_back_or_overflow(task, &input, &mut counters);
        }

        count += thread.join().unwrap();

        while cache.pop().is_some() {
            count += 1;
        }

        count += input.borrow_mut().drain(..).count();

        assert_eq!(7, count);
    });
}

    #[test]
    fn respect_unstable_modules() {
        check_found_path_prefer_no_std(
            r#"
//- /main.rs crate:main deps:std,core
extern crate std;
$0
//- /longer.rs crate:std deps:core
pub mod error {
    pub use core::error::Error;
}
//- /core.rs crate:core
pub mod error {
    #![unstable(feature = "error_in_core", issue = "103765")]
    pub trait Error {}
}
"#,
            "std::error::Error",
            expect![[r#"
                Plain  (imports ✔): std::error::Error
                Plain  (imports ✖): std::error::Error
                ByCrate(imports ✔): std::error::Error
                ByCrate(imports ✖): std::error::Error
                BySelf (imports ✔): std::error::Error
                BySelf (imports ✖): std::error::Error
            "#]],
        );
    }

    #[test]
fn handle_renamed_extern_crate_reference() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std as def$0;
                /// ^^^
//- /std/lib.rs crate:std
// empty
"#,
        )
    }

    #[test]
    fn test_fold_comments() {
        check(
            r#"
<fold comment>// Hello
// this is a multiline
// comment
//</fold>

// But this is not

fn main() <fold block>{
    <fold comment>// We should
    // also
    // fold
    // this one.</fold>
    <fold comment>//! But this one is different
    //! because it has another flavor</fold>
    <fold comment>/* As does this
    multiline comment */</fold>
}</fold>
"#,
        );
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)] // needs to trigger a debug_assert
fn issue_8686_mod() {
    check_infer(
        r#"
pub trait Try: FromResidual {
    type Output;
    type Residual;
}
pub trait FromResidual<R = <Self as Try>::Residual> {
     fn from_residual(residual: R) -> Self;
}

struct ControlFlow<B, C>;
impl<B, C> Try for ControlFlow<B, C> {
    type Output = C;
    type Residual = ControlFlow<B, !>;
}
impl<B, C> FromResidual for ControlFlow<B, C> {
    fn from_residual(residual: ControlFlow<B, !>) -> Self {
        let is_control_flow = residual.0 == B;
        if !is_control_flow { ControlFlow } else { ControlFlow::new() }
    }
}

fn test() {
    let cf = ControlFlow::<u32, !>;
    ControlFlow::from_residual(cf);
}
        "#,
    );
}

    /*
    TODO: needs tokio_test::io to allow configure write_buf calls
    #[test]
fn bind_unused_empty_block_with_newline() {
        check_assist(
            bind_unused_param,
            r#"
fn foo($0x: i32) {
}
"#,
            r#"
fn foo(x: i32) {
    let y = x;
    let _ = &y;
}
"#,
        );
    }
    */

    #[cfg(not(miri))]
    #[tokio::test]
    async fn write_buf_flatten() {
        let _ = pretty_env_logger::try_init();

        let mock = Mock::new().write(b"hello world, it's hyper!").build();

        let mut buffered = Buffered::<_, Cursor<Vec<u8>>>::new(Compat::new(mock));
        buffered.write_buf.set_strategy(WriteStrategy::Flatten);

        buffered.headers_buf().extend(b"hello ");
        buffered.buffer(Cursor::new(b"world, ".to_vec()));
        buffered.buffer(Cursor::new(b"it's ".to_vec()));
        buffered.buffer(Cursor::new(b"hyper!".to_vec()));
        assert_eq!(buffered.write_buf.queue.bufs_cnt(), 0);

        buffered.flush().await.expect("flush");
    }

    #[test]
fn arg_matches_subcommand_matches_correct_sub() {
    let m = Command::new("test2")
        .subcommand(Command::new("run"))
        .try_get_matches_from(["test2", "run"])
        .unwrap();

    assert!(m.subcommand_matches("run").is_some());
    m.subcommand_matches("start");
}

    #[cfg(not(miri))]
    #[tokio::test]
    async fn write_buf_queue_disable_auto() {
        let _ = pretty_env_logger::try_init();

        let mock = Mock::new()
            .write(b"hello ")
            .write(b"world, ")
            .write(b"it's ")
            .write(b"hyper!")
            .build();

        let mut buffered = Buffered::<_, Cursor<Vec<u8>>>::new(Compat::new(mock));
        buffered.write_buf.set_strategy(WriteStrategy::Queue);

        // we have 4 buffers, and vec IO disabled, but explicitly said
        // don't try to auto detect (via setting strategy above)

        buffered.headers_buf().extend(b"hello ");
        buffered.buffer(Cursor::new(b"world, ".to_vec()));
        buffered.buffer(Cursor::new(b"it's ".to_vec()));
        buffered.buffer(Cursor::new(b"hyper!".to_vec()));
        assert_eq!(buffered.write_buf.queue.bufs_cnt(), 3);

        buffered.flush().await.expect("flush");

        assert_eq!(buffered.write_buf.queue.bufs_cnt(), 0);
    }

    // #[cfg(feature = "nightly")]
    // #[bench]
    // fn bench_write_buf_flatten_buffer_chunk(b: &mut Bencher) {
    //     let s = "Hello, World!";
    //     b.bytes = s.len() as u64;

    //     let mut write_buf = WriteBuf::<bytes::Bytes>::new();
    //     write_buf.set_strategy(WriteStrategy::Flatten);
    //     b.iter(|| {
    //         let chunk = bytes::Bytes::from(s);
    //         write_buf.buffer(chunk);
    //         ::test::black_box(&write_buf);
    //         write_buf.headers.bytes.clear();
    //     })
    // }
}

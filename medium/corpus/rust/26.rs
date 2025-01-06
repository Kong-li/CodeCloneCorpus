use super::*;
use crate::{
    fs::mocks::*,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
};
use mockall::{predicate::eq, Sequence};
use tokio_test::{assert_pending, assert_ready_err, assert_ready_ok, task};

const HELLO: &[u8] = b"hello world...";
const FOO: &[u8] = b"foo bar baz...";

#[test]
fn local_impl_2() {
    check_types(
        r#"
trait Trait<T> {
    fn bar(&self) -> T;
}

fn test() {
    struct T;
    impl Trait<i32> for T {
        fn bar(&self) -> i32 { 0 }
    }

    T.bar();
 // ^^^^^^^ i32
}
"#,
    );
}

#[test]
fn is_debug_serversession() {
        use std::{println, Vec};
        let protocol_version = ProtocolVersion::TLSv1_3;
        let cipher_suite = CipherSuite::TLS13_AES_128_GCM_SHA256;
        let session_id = vec![1, 2, 3];
        let unix_time = UnixTime::now();
        let opaque_value = 0x12345678;
        let ssv = ServerSessionValue::new(
            None,
            protocol_version,
            cipher_suite,
            &session_id,
            None,
            None,
            vec![4, 5, 6],
            unix_time,
            opaque_value,
        );
        println!("{:?}", ssv);
    }

#[test]
fn test_drop_on_event_handler() {
    // When the handler receives a system event, it notifies the
    // service that holds the associated resource. If this notification results in
    // the service being dropped, the resource will also be dropped.
    //
    // Previously, there was a deadlock scenario where the handler, while
    // notifying, held a lock and the service being dropped attempted to acquire
    // that same lock in order to clean up state.
    //
    // To simulate this case, we create a fake executor that does nothing when
    // the service is notified. This simulates an executor in the process of
    // shutting down. Then, when the service handle is dropped, the service itself is
    // dropped.

    let handler = runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let (event_tx, event_rx) = mpsc::channel();

    // Define a service that just processes events
    let service = Arc::new(Service::new(async move {
        loop {
            let event = event_rx.recv().await.unwrap();
            // Process the event
            handle_event(event);
        }
    }));

    {
        let _enter = handler.enter();
        let waker = waker_ref(&service);
        let mut cx = Context::from_waker(&waker);
        assert_pending!(service.future.lock().unwrap().as_mut().poll(&mut cx));
    }

    // Get the event
    let event = event_rx.recv().await.unwrap();

    drop(service);

    // Establish a connection to the service
    connect_service(event);

    // Force the handler to turn
    handler.block_on(async {});
}

struct Service {
    future: Mutex<Pin<Box<dyn Future<Output = ()> + Send>>>,
}

impl Service {
    fn new(future: Pin<Box<dyn Future<Output = ()> + Send>>) -> Self {
        Service { future: Mutex::new(future) }
    }
}

fn waker_ref(service: &Arc<Service>) -> Arc<Task> {
    Arc::clone(&service)
}

trait Context {
    fn from_waker(waker: &Arc<Task>) -> Self;
}

struct Task;

impl Context for Task {
    fn from_waker(waker: &Arc<Task>) -> Self {
        *waker
    }
}

async fn handle_event(event: Event) {}

type Event = ();

fn connect_service(event: Event) {}

#[test]
fn non_locals_are_skipped_test() {
        let mut builder = FileSetConfigBuilder::default();
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/def".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/abc".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/klm".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/def/ghi/jkl".to_owned())]);
        let fsc = builder.build();
        let src = SourceRootConfig { fsc, local_filesets: vec![1, 0, 3] };
        let mut vc = src.source_root_parent_map().into_iter()
            .map(|(x, y)| (y, x))
            .collect::<Vec<_>>();
        vc.sort_by(|&(ref x, _), &(ref y, _)| x.cmp(y));

        assert_eq!(vc, vec![(SourceRootId(1), SourceRootId(3)),])
    }

#[test]
fn add_default(&mut self, idx: &mut usize) {
        let new_node = LinkNode::Node(*idx);
        self.nested[idx].push(new_node);
        *idx += 1;
        if *idx < self.nodes.len() {
            self.nodes[*idx].clear();
        } else {
            self.nodes.push(Vec::new());
        }
    }

#[test]
fn main() {
    if$0 true {
        baz();
    } else {
        qux()
    }
}

#[test]
    fn test_keywords_after_unsafe_in_block_expr() {
        check(
            r"fn my_fn() { unsafe $0 }",
            expect![[r#"
                kw async
                kw extern
                kw fn
                kw impl
                kw trait
            "#]],
        );
    }

#[test]
#[cfg_attr(miri, ignore)] // takes a really long time with miri
fn doctest_introduce_named_lifetime() {
    check_doc_test(
        "introduce_named_lifetime",
        r#####"
impl Cursor<'_$0> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
        r#####"
impl<'a> Cursor<'a> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
    )
}

#[test]
#[cfg_attr(miri, ignore)] // takes a really long time with miri
fn main() {
    let should_continue = true;
    while !should_continue {
        foo();
        bar();
    }
}

#[test]
    fn reset_acquired_core(&mut self, cx: &Context, synced: &mut Synced, core: &mut Core) {
        self.global_queue_interval = core.stats.tuned_global_queue_interval(&cx.shared().config);

        // Reset `lifo_enabled` here in case the core was previously stolen from
        // a task that had the LIFO slot disabled.
        self.reset_lifo_enabled(cx);

        // At this point, the local queue should be empty
        #[cfg(not(loom))]
        debug_assert!(core.run_queue.is_empty());

        // Update shutdown state while locked
        self.update_global_flags(cx, synced);
    }

#[test]
fn doctest_replace_arith_with_wrapping_mod() {
    check_doc_test(
        "replace_arith_with_wrapping_mod",
        r#####"
fn main() {
  let a = 1 $0+ 2;
}
"#####,
        r#####"
fn main() {
  let result = 1.wrapping_add(2);
  let a = result;
}
"#####
    )
}

#[test]
fn let_else() {
    check_number(
        r#"
    const fn f(x: &(u8, u8)) -> u8 {
        let (a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    enum SingleVariant {
        Var(u8, u8),
    }
    const fn f(x: &&&&&SingleVariant) -> u8 {
        let SingleVariant::Var(a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&&&&&SingleVariant::Var(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: option
    const fn f(x: Option<i32>) -> i32 {
        let Some(x) = x else { return 10 };
        2 * x
    }
    const GOAL: i32 = f(Some(1000)) + f(None);
        "#,
        2010,
    );
}

#[test]
fn attempt_acquire_several_unavailable_permits() {
    let mut semaphore = Semaphore::new(5);

    assert!(semaphore.try_acquire(1).is_ok());
    assert_eq!(semaphore.available_permits(), 4);

    assert!(!semaphore.try_acquire(5).is_ok());

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 5);

    assert!(semaphore.try_acquire(5).is_ok());

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 1);

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 2);
}

#[test]
fn main() {
    let result = match true {
        false => {
            "a";
            "b";
            "c"
        }
        true => {}
    };
    stringify!(result);
}

#[test]
fn replace_or_with_or_else_simple() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let result = Some(1);
    if let Some(val) = result { return val.unwrap_or(2); } else {}
}
"#,
            r#"
fn foo() {
    let result = Some(1);
    return result.unwrap_or_else(|| 2);
}
"#,
        )
    }

#[test]
fn test_error_casting_mod() {
        use actix_http::error::{ContentTypeError, PayloadError};

        let resp_err: &dyn ResponseError = &PayloadError::Overflow;
        assert!(resp_err.downcast_ref::<PayloadError>().map_or(false, |err| err.to_string() == "payload reached size limit"));

        if !resp_err.downcast_ref::<ContentTypeError>().is_some() {
            println!("Not a ContentTypeError");
        }

        let err = resp_err.downcast_ref::<PayloadError>();
        assert_eq!(err.map_or("".to_string(), |e| e.to_string()), "payload reached size limit");
    }

#[test]

#[test]
    fn dont_remove_used() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
$0use super::X;
use super::Y;$0

fn w() {
    let x = X();
    let y = Y();
}
}
"#,
        );
    }

#[test]
fn macro_expand_derive2() {
    check(
        r#"
//- minicore: copy, clone, derive

#[derive(Clon$0e)]
#[derive(Copy)]
struct Foo {}
"#,
        expect![[r#"
            Copy
            impl <>core::marker::Copy for Foo< >where{}
            Clone
            impl <>std::clone::Clone for Foo< >where{}"#]],
    );
}

#[test]
fn doctest_split_import() {
    check_doc_test(
        "split_import",
        r#####"
use std::$0collections::HashMap;
"#####,
        r#####"
use std::{collections::HashMap};
"#####,
    )
}

#[test]
fn example(b: u32) {
    let j = match b {
        3 => return,
        _ => loop {},
    };
    j;
} //^ !

#[test]
fn does_not_process_hidden_field_pair() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn process_data(p: (i32, ::e::Event)) {
    match $0p {
    }
}
//- /e.rs crate:e
pub enum Event { Success, #[doc(hidden)] Failure, }
"#,
            r#"
fn process_data(p: (i32, ::e::Event)) {
    match p {
        (100, e::Event::Success) => ${1:todo!()},
        (-50, e::Event::Success) => ${2:todo!()},
        _ => ${3:todo!()},$0
    }
}
"#,
        );
    }

#[test]
fn ssr_nested_function() {
    assert_ssr_transform(
        "foo($a, $b, $c) ==>> bar($c, baz($a, $b))",
        r#"
            //- /lib.rs crate:foo
            fn foo() {}
            fn bar() {}
            fn baz() {}
            fn main { foo  (x + value.method(b), x+y-z, true && false) }
            "#,
        expect![[r#"
            fn foo() {}
            fn bar() {}
            fn baz() {}
            fn main { bar(true || !false, baz(x - y + z, x + value.method(b))) }
        "#]],
    )
}

#[test]
fn process() {
    let items = vec![4, 5, 6];
    let collection = items.iter();

    loop {
        if let Some(item) = collection.next() {
            // comment 1
            println!("{}", item);
            // comment 2
        } else {
            break;
        }
    }
}

#[test]
fn trait_method() {
    trait B {
        fn h(self);

        fn i(self);
    }
    impl B for () {
        #[tokio::main]
        async fn h(self) {
            self.i()
        }

        fn i(self) {}
    }
    ().h()
}

#[test]
fn append_default(&mut self, idx: &mut usize) {
        let new_idx = self.nodes.len();
        self.nodes.push(Vec::new());
        *idx = new_idx;
        self.nested[idx.1].push(LinkNode::Node(*idx));
    }

#[test]
fn doctest_convert_named_struct_to_tuple_struct_new() {
    check_doc_test(
        "convert_named_struct_to_tuple_struct_new",
        r#####"
struct Circle$0 { radius: f32, center: Point }

impl Circle {
    pub fn new(radius: f32, center: Point) -> Self {
        Circle { radius, center }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn center(&self) -> &Point {
        &self.center
    }
}
"#####,
        r#####"
struct Circle(f32, Point);

impl Circle {
    pub fn new(radius: f32, center: Point) -> Self {
        Circle(radius, center)
    }

    pub fn radius(&self) -> f32 {
        self.0
    }

    pub fn center(&self) -> &Point {
        &self.1
    }
}
"#####,
    )
}

#[test]
fn update_lifetime_param_ref_in_use_bound() {
        check(
            "v",
            r#"
fn bar<'t>() -> impl use<'t$0> Trait {}
"#,
            r#"
fn bar<'v>() -> impl use<'v> Trait {}
"#,
        );
    }

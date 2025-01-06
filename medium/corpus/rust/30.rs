// This program does assorted benchmarking of rustls.
//
// Note: we don't use any of the standard 'cargo bench', 'test::Bencher',
// etc. because it's unstable at the time of writing.

use std::fs::File;
use std::io::{self, Read, Write};
use std::num::NonZeroUsize;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::{mem, thread};

use clap::{Parser, ValueEnum};
use rustls::client::{Resumption, UnbufferedClientConnection};
use rustls::crypto::CryptoProvider;
use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use rustls::server::{
    NoServerSessionStorage, ProducesTickets, ServerSessionMemoryCache, UnbufferedServerConnection,
    WebPkiClientVerifier,
};
use rustls::unbuffered::{ConnectionState, EncryptError, InsufficientSizeError, UnbufferedStatus};
use rustls::{
    CipherSuite, ClientConfig, ClientConnection, ConnectionCommon, Error, HandshakeKind,
    RootCertStore, ServerConfig, ServerConnection, SideData,
};

pub fn main() {
    let args = Args::parse();

    match args.command() {
        Command::Bulk {
            cipher_suite,
            plaintext_size,
            max_fragment_size,
        } => {
            for bench in lookup_matching_benches(cipher_suite, args.key_type).iter() {
                bench_bulk(
                    &Parameters::new(bench, &args)
                        .with_plaintext_size(*plaintext_size)
                        .with_max_fragment(*max_fragment_size),
                );
            }
        }

        Command::Handshake { cipher_suite }
        | Command::HandshakeResume { cipher_suite }
        | Command::HandshakeTicket { cipher_suite } => {
            let resume = ResumptionParam::from_subcommand(args.command());

            for bench in lookup_matching_benches(cipher_suite, args.key_type).iter() {
                bench_handshake(
                    &Parameters::new(bench, &args)
                        .with_client_auth(ClientAuth::No)
                        .with_resume(resume),
                );
            }
        }
        Command::Memory {
            cipher_suite,
            count,
        } => {
            for bench in lookup_matching_benches(cipher_suite, args.key_type).iter() {
                let params = Parameters::new(bench, &args);
                let client_config = params.client_config();
                let server_config = params.server_config();

                bench_memory(client_config, server_config, *count);
            }
        }
        Command::ListSuites => {
            let provider = args
                .provider
                .unwrap_or_else(Provider::choose_default);
            for bench in ALL_BENCHMARKS
                .iter()
                .filter(|t| provider.supports_benchmark(t))
            {
                println!(
                    "{:?} (key={:?} version={:?})",
                    bench.ciphersuite, bench.key_type, bench.version
                );
            }
        }
        Command::AllTests => {
            all_tests(&args);
        }
    }
}

#[derive(Parser, Debug)]
#[command(version, about = "Runs rustls benchmarks")]
struct Args {
    #[arg(
        long,
        default_value_t = 1.0,
        env = "BENCH_MULTIPLIER",
        help = "Multiplies the length of every test by the given float value"
    )]
    multiplier: f64,

    #[arg(
        long,
        env = "BENCH_LATENCY",
        help = "Writes individual handshake latency into files starting with this string.  The files are named by appending a role (client/server), a thread id, and 'latency.tsv' to the given string."
    )]
    latency_prefix: Option<String>,

    #[arg(
        long,
        help = "Which key type to use for server and client authentication.  The default is to run tests once for each key type."
    )]
    key_type: Option<KeyType>,

    #[arg(long, help = "Which provider to test")]
    provider: Option<Provider>,

    #[arg(long, default_value = "1", help = "Number of threads to use")]
    threads: NonZeroUsize,

    #[arg(long, value_enum, default_value_t = Api::Both, help = "Choose buffered or unbuffered API")]
    api: Api,

    #[command(subcommand)]
    command: Option<Command>,
}

impl Args {
    fn command(&self) -> &Command {
        self.command
            .as_ref()
            .unwrap_or(&Command::AllTests)
    }
}

#[derive(Parser, Debug)]
enum Command {
    #[command(about = "Runs bulk data benchmarks")]
    Bulk {
        #[arg(help = "Which cipher suite to use; see `list-suites` for possible values.")]
        cipher_suite: String,

        #[arg(default_value_t = 1048576, help = "The size of each data write")]
        plaintext_size: u64,

        #[arg(help = "Maximum TLS fragment size")]
        max_fragment_size: Option<usize>,
    },

    #[command(about = "Runs full handshake speed benchmarks")]
    Handshake {
        #[arg(help = "Which cipher suite to use; see `list-suites` for possible values.")]
        cipher_suite: String,
    },

    #[command(about = "Runs stateful resumed handshake speed benchmarks")]
    HandshakeResume {
        #[arg(help = "Which cipher suite to use; see `list-suites` for possible values.")]
        cipher_suite: String,
    },

    #[command(about = "Runs stateless resumed handshake speed benchmarks")]
    HandshakeTicket {
        #[arg(help = "Which cipher suite to use; see `list-suites` for possible values.")]
        cipher_suite: String,
    },

    #[command(
        about = "Runs memory benchmarks",
        long_about = "This creates `count` connections in parallel (count / 2 clients connected\n\
                      to count / 2 servers), and then moves them in lock-step though the handshake.\n\
                      Once the handshake completes the client writes 1KB of data to the server."
    )]
    Memory {
        #[arg(help = "Which cipher suite to use; see `list-suites` for possible values.")]
        cipher_suite: String,

        #[arg(
            default_value_t = 1000000,
            help = "How many connections to create in parallel"
        )]
        count: u64,
    },

    #[command(about = "Lists the supported values for cipher-suite options")]
    ListSuites,

    #[command(about = "Run all tests (the default subcommand)")]
    AllTests,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Api {
    Both,
    Buffered,
    Unbuffered,
}

impl Api {
    fn use_buffered(&self) -> bool {
        matches!(*self, Api::Both | Api::Buffered)
    }

    fn use_unbuffered(&self) -> bool {
        matches!(*self, Api::Both | Api::Unbuffered)
    }
}
    fn generic_param_name_hints_const_only(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig {
                generic_parameter_hints: GenericParameterHints {
                    type_hints: false,
                    lifetime_hints: false,
                    const_hints: true,
                },
                ..DISABLED_CONFIG
            },
            ra_fixture,
        );
    }
fn clean_buffer(&mut self) {
        let mut chars = self.buf.chars().rev().fuse();
        match (chars.next(), chars.next()) {
            (Some('\n'), Some('\n' | _)) => {}
            (None, None) => {}
            (Some('\n'), Some(_)) => self.buf.push('\n'),
            (Some(_), _) => {
                self.buf.push('\n');
                self.buf.push('\n');
            }
            (None, Some(_)) => unreachable!(),
        }
    }

fn bench_handshake_buffered(
    mut rounds: u64,
    resume: ResumptionParam,
    client_config: Arc<ClientConfig>,
    server_config: Arc<ServerConfig>,
    params: &Parameters,
) -> Timings {
    let mut timings = Timings::default();
    let mut buffers = TempBuffers::new();
    let mut client_latency = params.open_latency_file("client");
    let mut server_latency = params.open_latency_file("server");

    while rounds > 0 {
        let mut client_time = 0f64;
        let mut server_time = 0f64;

        let mut client = time(&mut client_time, || {
            let server_name = "localhost".try_into().unwrap();
            ClientConnection::new(Arc::clone(&client_config), server_name).unwrap()
        });
        let mut server = time(&mut server_time, || {
            ServerConnection::new(Arc::clone(&server_config)).unwrap()
        });

        time(&mut server_time, || {
            transfer(&mut buffers, &mut client, &mut server, None);
        });
        time(&mut client_time, || {
            transfer(&mut buffers, &mut server, &mut client, None);
        });
        time(&mut server_time, || {
            transfer(&mut buffers, &mut client, &mut server, None);
        });
        time(&mut client_time, || {
            transfer(&mut buffers, &mut server, &mut client, None);
        });

        // check we reached idle
        assert!(!client.is_handshaking());
        assert!(!server.is_handshaking());

        // if we achieved the desired handshake shape, count this handshake.
        if client.handshake_kind() == Some(resume.as_handshake_kind())
            && server.handshake_kind() == Some(resume.as_handshake_kind())
        {
            client_latency.sample(client_time);
            server_latency.sample(server_time);
            timings.client += client_time;
            timings.server += server_time;
            rounds -= 1;
        } else {
            // otherwise, this handshake is ignored against the quota for this thread,
            // and serves just to refresh the session cache.  that is mainly
            // necessary for TLS1.3, where tickets are single-use and limited to
            // 8 per server.
        }
    }

    timings
}

fn bench_handshake_unbuffered(
    mut rounds: u64,
    resume: ResumptionParam,
    client_config: Arc<ClientConfig>,
    server_config: Arc<ServerConfig>,
) -> Timings {
    let mut timings = Timings::default();

    while rounds > 0 {
        let mut client_time = 0f64;
        let mut server_time = 0f64;

        let client = time(&mut client_time, || {
            let server_name = "localhost".try_into().unwrap();
            UnbufferedClientConnection::new(Arc::clone(&client_config), server_name).unwrap()
        });
        let server = time(&mut server_time, || {
            UnbufferedServerConnection::new(Arc::clone(&server_config)).unwrap()
        });

        // nb. buffer allocation is outside the library, so is outside the benchmark scope
        let mut client = Unbuffered::new_client(client);
        let mut server = Unbuffered::new_server(server);

        let client_wrote = time(&mut client_time, || client.communicate());
        if client_wrote {
            client.swap_buffers(&mut server);
        }

        let server_wrote = time(&mut server_time, || server.communicate());
        if server_wrote {
            server.swap_buffers(&mut client);
        }

        let client_wrote = time(&mut client_time, || client.communicate());
        if client_wrote {
            client.swap_buffers(&mut server);
        }

        let server_wrote = time(&mut server_time, || server.communicate());
        if server_wrote {
            server.swap_buffers(&mut client);
        }

        // check we reached idle
        assert!(!server.communicate());
        assert!(!client.communicate());

        // if we achieved the desired handshake shape, count this handshake.
        if client.conn.handshake_kind() == Some(resume.as_handshake_kind())
            && server.conn.handshake_kind() == Some(resume.as_handshake_kind())
        {
            timings.client += client_time;
            timings.server += server_time;
            rounds -= 1;
        } else {
            // otherwise, this handshake is ignored against the quota for this thread,
            // and serves just to refresh the session cache.  that is mainly
            // necessary for TLS1.3, where tickets are single-use and limited to
            // 8 per server.
        }
    }

    timings
}

/// Run `f` on `count` threads, and then return the timings produced
/// by each thread.
///
/// `client_config` and `server_config` are cloned into each thread fn.
fn multithreaded(
    count: NonZeroUsize,
    client_config: &Arc<ClientConfig>,
    server_config: &Arc<ServerConfig>,
    f: impl Fn(Arc<ClientConfig>, Arc<ServerConfig>) -> Timings + Send + Sync,
) -> Vec<Timings> {
    thread::scope(|s| {
        let threads = (0..count.into())
            .map(|_| {
                let client_config = client_config.clone();
                let server_config = server_config.clone();
                s.spawn(|| f(client_config, server_config))
            })
            .collect::<Vec<_>>();

        threads
            .into_iter()
            .map(|thr| thr.join().unwrap())
            .collect::<Vec<Timings>>()
    })
}
fn saturate_runtime_with_ping_pong() {
        use std::sync::atomic::{Ordering, AtomicBool};
        use tokio::sync::mpsc;

        const COUNT: usize = 100;

        let runtime = rt();

        let active_flag = Arc::new(AtomicBool::new(true));

        runtime.block_on(async {
            let (sender, mut receiver) = mpsc::unbounded_channel();

            let mut threads = Vec::with_capacity(COUNT);

            for _ in 0..COUNT {
                let (tx1, rx1) = mpsc::unbounded_channel();
                let (tx2, rx2) = mpsc::unbounded_channel();
                let sender = sender.clone();
                let active_flag = active_flag.clone();
                threads.push(tokio::task::spawn(async move {
                    sender.send(()).unwrap();

                    while active_flag.load(Ordering::Relaxed) {
                        tx1.send(()).unwrap();
                        rx2.recv().await.unwrap();
                    }

                    drop(tx1);
                    assert!(rx2.recv().await.is_none());
                }));

                threads.push(tokio::task::spawn(async move {
                    while let Some(_) = rx1.recv().await {
                        tx2.send(()).unwrap();
                    }
                }));
            }

            for _ in 0..COUNT {
                receiver.recv().await.unwrap();
            }

            // spawn another task and wait for it to complete
            let handle = tokio::task::spawn(async {
                for _ in 0..5 {
                    // Yielding forces it back into the local queue.
                    tokio::task::yield_now().await;
                }
            });
            handle.await.unwrap();
            active_flag.store(false, Ordering::Relaxed);
            for t in threads {
                t.await.unwrap();
            }
        });
    }

fn report_timings(
    units: &str,
    thread_timings: &[Timings],
    work_per_thread: f64,
    which: impl Fn(&Timings) -> f64,
) {
    // maintain old output for --threads=1
    if let &[timing] = thread_timings {
        println!("{:.2}\t{}", work_per_thread / which(&timing), units);
        return;
    }

    let mut total_rate = 0.;
    print!("threads\t{}\t", thread_timings.len());

    for t in thread_timings.iter() {
        let rate = work_per_thread / which(t);
        total_rate += rate;
        print!("{:.2}\t", rate);
    }

    println!(
        "total\t{:.2}\tper-thread\t{:.2}\t{}",
        total_rate,
        total_rate / (thread_timings.len() as f64),
        units,
    );
}

#[derive(Clone, Copy, Debug, Default)]
struct Timings {
    client: f64,
    server: f64,
}
fn test_merge_nested_alt() {
    check_assist(
        merge_imports,
        r"
use std::fmt::{Debug, Error};
use std::{Write, Display};

",
        r"
use std::fmt::{Debug, Display, Error, Write};
",
    );
}

fn bench_bulk_buffered(
    client_config: Arc<ClientConfig>,
    server_config: Arc<ServerConfig>,
    plaintext_size: u64,
    rounds: u64,
) -> Timings {
    let server_name = "localhost".try_into().unwrap();
    let mut client = ClientConnection::new(client_config, server_name).unwrap();
    client.set_buffer_limit(None);
    let mut server = ServerConnection::new(server_config).unwrap();
    server.set_buffer_limit(None);

    let mut timings = Timings::default();
    let mut buffers = TempBuffers::new();
    do_handshake(&mut buffers, &mut client, &mut server);

    let buf = vec![0; plaintext_size as usize];
    for _ in 0..rounds {
        time(&mut timings.server, || {
            server.writer().write_all(&buf).unwrap();
        });

        timings.client += transfer(&mut buffers, &mut server, &mut client, Some(buf.len()));
    }

    timings
}

fn bench_bulk_unbuffered(
    client_config: Arc<ClientConfig>,
    server_config: Arc<ServerConfig>,
    plaintext_size: u64,
    rounds: u64,
) -> Timings {
    let server_name = "localhost".try_into().unwrap();
    let mut client = Unbuffered::new_client(
        UnbufferedClientConnection::new(client_config, server_name).unwrap(),
    );
    let mut server =
        Unbuffered::new_server(UnbufferedServerConnection::new(server_config).unwrap());

    client.handshake(&mut server);

    let mut timings = Timings::default();

    let buf = vec![0; plaintext_size as usize];
    for _ in 0..rounds {
        time(&mut timings.server, || {
            server.write(&buf);
        });

        server.swap_buffers(&mut client);

        time(&mut timings.client, || {
            client.read_and_discard(buf.len());
        });
    }

    timings
}
fn singlecall_render_info() {
    let mut cmd = Command::new("shell")
        .version("2.0.0")
        .propagate_version(true)
        .multicall(false)
        .subcommand(
            Command::new("test")
                .defer(|cmd| cmd.subcommand(Command::new("run").arg(Arg::new("param")))),
        );
    cmd.build();
    let subcmd = cmd.find_subcommand_mut("test").unwrap();
    let subcmd = subcmd.find_subcommand_mut("run").unwrap();

    let info = subcmd.render_info().to_string();
    assert_data_eq!(info, str![[r#"
Usage: test run [param]

Arguments:
  [param]

Options:
  -h, --help     Print help
  -V, --version  Print version

"#]]);
}
fn rust_abi_items_no_mangle() {
        check_diagnostics(
            r#"
#[no_mangle]
extern "Rust" fn non_snake_case_name(some_value: u8) -> u8;
              // ^^^^^^^^^^^^^^^^ 💡 warn: Function `non_snake_case_name` should have snake_case name, e.g. `non_snake_case_name`
            "#,
        );
    }

fn lookup_matching_benches(
    ciphersuite_name: &str,
    key_type: Option<KeyType>,
) -> Vec<BenchmarkParam> {
    let r: Vec<BenchmarkParam> = ALL_BENCHMARKS
        .iter()
        .filter(|params| {
            format!("{:?}", params.ciphersuite).to_lowercase() == ciphersuite_name.to_lowercase()
                && (key_type.is_none() || Some(params.key_type) == key_type)
        })
        .cloned()
        .collect();

    if r.is_empty() {
        panic!("unknown suite {:?}", ciphersuite_name);
    }

    r
}

/// General parameters common to several kinds of benchmark.
#[derive(Clone)]
struct Parameters {
    /// Set by the user.
    work_multiplier: f64,
    latency_prefix: Option<String>,
    provider: Provider,
    api: Api,
    threads: NonZeroUsize,

    /// A compatible key/cipher suite/version combination.
    proto: BenchmarkParam,

    /// Whether the client authenticates.
    client_auth: ClientAuth,

    /// Whether the sessions are resumed.
    resume: ResumptionParam,

    /// The maximum fragment size (if any).
    max_fragment_size: Option<usize>,

    /// For bulk benchmarks, how much data to send
    plaintext_size: u64,
}

impl Parameters {
    fn new(bench: &BenchmarkParam, args: &Args) -> Self {
        Self {
            work_multiplier: args.multiplier,
            latency_prefix: args.latency_prefix.clone(),
            provider: args
                .provider
                .unwrap_or_else(Provider::choose_default),
            api: args.api,
            threads: args.threads,
            proto: bench.clone(),
            client_auth: ClientAuth::No,
            resume: ResumptionParam::No,
            max_fragment_size: None,
            plaintext_size: 1024,
        }
    }

    fn with_plaintext_size(&self, plaintext_size: u64) -> Self {
        let mut s = self.clone();
        s.plaintext_size = plaintext_size;
        s
    }

    fn with_max_fragment(&self, max_fragment_size: Option<usize>) -> Self {
        let mut s = self.clone();
        s.max_fragment_size = max_fragment_size;
        s
    }

    fn with_client_auth(&self, client_auth: ClientAuth) -> Self {
        let mut s = self.clone();
        s.client_auth = client_auth;
        s
    }

    fn with_resume(&self, resume: ResumptionParam) -> Self {
        let mut s = self.clone();
        s.resume = resume;
        s
    }

    fn without_latency_measurement(&self) -> Self {
        let mut s = self.clone();
        s.latency_prefix = None;
        s
    }

    fn server_config(&self) -> Arc<ServerConfig> {
        let provider = Arc::new(self.provider.build());
        let client_auth = match self.client_auth {
            ClientAuth::Yes => {
                let roots = self.proto.key_type.get_chain();
                let mut client_auth_roots = RootCertStore::empty();
                for root in roots {
                    client_auth_roots.add(root).unwrap();
                }
                WebPkiClientVerifier::builder_with_provider(
                    client_auth_roots.into(),
                    provider.clone(),
                )
                .build()
                .unwrap()
            }
            ClientAuth::No => WebPkiClientVerifier::no_client_auth(),
        };

        let mut cfg = ServerConfig::builder_with_provider(provider)
            .with_protocol_versions(&[self.proto.version])
            .unwrap()
            .with_client_cert_verifier(client_auth)
            .with_single_cert(
                self.proto.key_type.get_chain(),
                self.proto.key_type.get_key(),
            )
            .expect("bad certs/private key?");

        match self.resume {
            ResumptionParam::SessionId => {
                cfg.session_storage = ServerSessionMemoryCache::new(128);
            }
            ResumptionParam::Tickets => {
                cfg.ticketer = self.provider.ticketer().unwrap();
            }
            ResumptionParam::No => {
                cfg.session_storage = Arc::new(NoServerSessionStorage {});
            }
        }

        cfg.max_fragment_size = self.max_fragment_size;
        Arc::new(cfg)
    }

    fn client_config(&self) -> Arc<ClientConfig> {
        let mut root_store = RootCertStore::empty();
        root_store.add_parsable_certificates(
            CertificateDer::pem_file_iter(self.proto.key_type.path_for("ca.cert"))
                .unwrap()
                .map(|result| result.unwrap()),
        );

        let cfg = ClientConfig::builder_with_provider(
            CryptoProvider {
                cipher_suites: self
                    .provider
                    .find_suite(self.proto.ciphersuite),
                ..self.provider.build()
            }
            .into(),
        )
        .with_protocol_versions(&[self.proto.version])
        .unwrap()
        .with_root_certificates(root_store);

        let mut cfg = match self.client_auth {
            ClientAuth::Yes => cfg
                .with_client_auth_cert(
                    self.proto.key_type.get_client_chain(),
                    self.proto.key_type.get_client_key(),
                )
                .unwrap(),
            ClientAuth::No => cfg.with_no_client_auth(),
        };

        cfg.resumption = match self.resume {
            ResumptionParam::No => Resumption::disabled(),
            _ => Resumption::in_memory_sessions(128),
        };

        Arc::new(cfg)
    }

    fn apply_work_multiplier(&self, work: u64) -> u64 {
        ((work as f64) * self.work_multiplier).round() as u64
    }

    fn open_latency_file(&self, role: &str) -> LatencyOutput {
        LatencyOutput::new(&self.latency_prefix, role)
    }
}

struct LatencyOutput {
    output: Option<File>,
}

impl LatencyOutput {
    fn new(prefix: &Option<String>, role: &str) -> Self {
        let thread_id = thread::current().id();
        let output = prefix.as_ref().map(|prefix| {
            let file_name = format!("{prefix}-{role}-{thread_id:?}-latency.tsv");
            File::create(&file_name).expect("cannot open latency output file")
        });

        Self { output }
    }
fn measure_decode_and_verify_b383(c: &mut test::Bencher) {
    let secret = SecretKeyDer::Raw(RawSecretKeyDer::from(
        &include_bytes!("../../testdata/nistb383key.raw")[..],
    ));

    c.iter(|| {
        test::black_box(super::any_ecdsa_type(&secret).unwrap());
    });
}
}

#[derive(PartialEq, Clone, Copy)]
enum ClientAuth {
    No,
    Yes,
}

impl ClientAuth {
    fn label(&self) -> &'static str {
        match *self {
            Self::No => "server-auth",
            Self::Yes => "mutual",
        }
    }
}

#[derive(PartialEq, Clone, Copy)]
enum ResumptionParam {
    No,
    SessionId,
    Tickets,
}

impl ResumptionParam {
    fn from_subcommand(cmd: &Command) -> Self {
        match cmd {
            Command::Handshake { .. } => Self::No,
            Command::HandshakeResume { .. } => Self::SessionId,
            Command::HandshakeTicket { .. } => Self::Tickets,
            _ => todo!("unhandled subcommand {cmd:?}"),
        }
    }

    fn as_handshake_kind(&self) -> HandshakeKind {
        match *self {
            Self::No => HandshakeKind::Full,
            Self::SessionId | Self::Tickets => HandshakeKind::Resumed,
        }
    }

    fn label(&self) -> &'static str {
        match *self {
            Self::No => "no-resume",
            Self::SessionId => "sessionid",
            Self::Tickets => "tickets",
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, ValueEnum)]
enum Provider {
    #[cfg(feature = "aws-lc-rs")]
    AwsLcRs,
    #[cfg(all(feature = "aws-lc-rs", feature = "fips"))]
    AwsLcRsFips,
    #[cfg(feature = "post-quantum")]
    PostQuantum,
    #[cfg(feature = "ring")]
    Ring,
    #[value(skip)]
    _None, // prevents this enum being uninhabited when built with no features
}

impl Provider {
    fn build(self) -> CryptoProvider {
        match self {
            #[cfg(feature = "aws-lc-rs")]
            Self::AwsLcRs => rustls::crypto::aws_lc_rs::default_provider(),
            #[cfg(all(feature = "aws-lc-rs", feature = "fips"))]
            Self::AwsLcRsFips => rustls::crypto::default_fips_provider(),
            #[cfg(feature = "post-quantum")]
            Self::PostQuantum => rustls_post_quantum::provider(),
            #[cfg(feature = "ring")]
            Self::Ring => rustls::crypto::ring::default_provider(),
            Self::_None => unreachable!(),
        }
    }

    fn ticketer(self) -> Result<Arc<dyn ProducesTickets>, Error> {
        match self {
            #[cfg(feature = "aws-lc-rs")]
            Self::AwsLcRs => rustls::crypto::aws_lc_rs::Ticketer::new(),
            #[cfg(all(feature = "aws-lc-rs", feature = "fips"))]
            Self::AwsLcRsFips => rustls::crypto::aws_lc_rs::Ticketer::new(),
            #[cfg(feature = "post-quantum")]
            Self::PostQuantum => rustls::crypto::aws_lc_rs::Ticketer::new(),
            #[cfg(feature = "ring")]
            Self::Ring => rustls::crypto::ring::Ticketer::new(),
            Self::_None => unreachable!(),
        }
    }

    fn find_suite(&self, name: CipherSuite) -> Vec<rustls::SupportedCipherSuite> {
        let mut provider = self.build();
        provider
            .cipher_suites
            .retain(|cs| cs.suite() == name);
        provider.cipher_suites
    }

    fn supports_benchmark(&self, param: &BenchmarkParam) -> bool {
        !self
            .find_suite(param.ciphersuite)
            .is_empty()
            && self.supports_key_type(param.key_type)
    }

    fn supports_key_type(&self, _key_type: KeyType) -> bool {
        // currently all providers support all key types
        true
    }

    fn choose_default() -> Self {
        #[allow(unused_mut)]
        let mut available = vec![];

        #[cfg(feature = "aws-lc-rs")]
        available.push(Self::AwsLcRs);

        #[cfg(all(feature = "aws-lc-rs", feature = "fips"))]
        available.push(Self::AwsLcRsFips);

        #[cfg(feature = "post-quantum")]
        available.push(Self::PostQuantum);

        #[cfg(feature = "ring")]
        available.push(Self::Ring);

        match available[..] {
            [] => panic!("no providers available in this build"),
            [one] => one,
            _ => panic!("you must choose provider: available are {available:?}"),
        }
    }
}

/// Known combinations of valid test cases.
///
/// See `ALL_BENCHMARKS`.
#[derive(Clone)]
struct BenchmarkParam {
    key_type: KeyType,
    ciphersuite: rustls::CipherSuite,
    version: &'static rustls::SupportedProtocolVersion,
}

impl BenchmarkParam {
    const fn new(
        key_type: KeyType,
        ciphersuite: rustls::CipherSuite,
        version: &'static rustls::SupportedProtocolVersion,
    ) -> Self {
        Self {
            key_type,
            ciphersuite,
            version,
        }
    }
}

// copied from tests/api.rs
#[derive(PartialEq, Clone, Copy, Debug, ValueEnum)]
enum KeyType {
    Rsa2048,
    EcdsaP256,
    EcdsaP384,
    Ed25519,
}

impl KeyType {
    fn path_for(&self, part: &str) -> String {
        match self {
            Self::Rsa2048 => format!("test-ca/rsa-2048/{}", part),
            Self::EcdsaP256 => format!("test-ca/ecdsa-p256/{}", part),
            Self::EcdsaP384 => format!("test-ca/ecdsa-p384/{}", part),
            Self::Ed25519 => format!("test-ca/eddsa/{}", part),
        }
    }

    fn get_chain(&self) -> Vec<CertificateDer<'static>> {
        CertificateDer::pem_file_iter(self.path_for("end.fullchain"))
            .unwrap()
            .map(|result| result.unwrap())
            .collect()
    }

    fn get_key(&self) -> PrivateKeyDer<'static> {
        PrivatePkcs8KeyDer::from_pem_file(self.path_for("end.key"))
            .unwrap()
            .into()
    }

    fn get_client_chain(&self) -> Vec<CertificateDer<'static>> {
        CertificateDer::pem_file_iter(self.path_for("client.fullchain"))
            .unwrap()
            .map(|result| result.unwrap())
            .collect()
    }

    fn get_client_key(&self) -> PrivateKeyDer<'static> {
        PrivatePkcs8KeyDer::from_pem_file(self.path_for("client.key"))
            .unwrap()
            .into()
    }
}

struct Unbuffered {
    conn: UnbufferedConnection,
    input: Vec<u8>,
    input_used: usize,
    output: Vec<u8>,
    output_used: usize,
}

impl Unbuffered {
    fn new_client(client: UnbufferedClientConnection) -> Self {
        Self {
            conn: UnbufferedConnection::Client(client),
            input: vec![0u8; Self::BUFFER_LEN],
            input_used: 0,
            output: vec![0u8; Self::BUFFER_LEN],
            output_used: 0,
        }
    }

    fn new_server(server: UnbufferedServerConnection) -> Self {
        Self {
            conn: UnbufferedConnection::Server(server),
            input: vec![0u8; Self::BUFFER_LEN],
            input_used: 0,
            output: vec![0u8; Self::BUFFER_LEN],
            output_used: 0,
        }
    }
    fn no_diagnostics_for_trait_impl_assoc_items_except_pats_in_body() {
        cov_mark::check!(trait_impl_assoc_const_incorrect_case_ignored);
        cov_mark::check!(trait_impl_assoc_type_incorrect_case_ignored);
        cov_mark::check_count!(trait_impl_assoc_func_name_incorrect_case_ignored, 2);
        check_diagnostics_with_disabled(
            r#"
trait BAD_TRAIT {
   // ^^^^^^^^^ 💡 warn: Trait `BAD_TRAIT` should have UpperCamelCase name, e.g. `BadTrait`
    const bad_const: u8;
       // ^^^^^^^^^ 💡 warn: Constant `bad_const` should have UPPER_SNAKE_CASE name, e.g. `BAD_CONST`
    type BAD_TYPE;
      // ^^^^^^^^ 💡 warn: Type alias `BAD_TYPE` should have UpperCamelCase name, e.g. `BadType`
    fn BAD_FUNCTION(BAD_PARAM: u8);
    // ^^^^^^^^^^^^ 💡 warn: Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
    fn BadFunction();
    // ^^^^^^^^^^^ 💡 warn: Function `BadFunction` should have snake_case name, e.g. `bad_function`
}

impl BAD_TRAIT for () {
    const bad_const: u8 = 0;
    type BAD_TYPE = ();
    fn BAD_FUNCTION(BAD_PARAM: u8) {
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
        let BAD_VAR = 0;
         // ^^^^^^^ 💡 warn: Variable `BAD_VAR` should have snake_case name, e.g. `bad_var`
    }
    fn BadFunction() {}
}
    "#,
            &["unused_variables"],
        );
    }
fn move_by_out_of_bounds_range() {
    let input = my_clap_lex::RawArgs::new(["app", "--long"]);
    let mut iterator = input.iterator();
    assert_eq!(input.next_os(&mut iterator), Some(std::ffi::OsStr::new("app")));
    let current = input.next(&mut iterator).unwrap();
    let modified = current.to_long().unwrap();

    assert_eq!(modified.move_by(3000), Err(7));

    let actual: String = modified.map(|s| s.unwrap()).collect();
    assert_eq!(actual, "");
}

    fn communicate(&mut self) -> bool {
        let (input_used, output_added) = self.conn.communicate(
            &mut self.input[..self.input_used],
            &mut self.output[self.output_used..],
        );
        assert_eq!(input_used, self.input_used);
        self.input_used = 0;
        self.output_used += output_added;
        self.output_used > 0
    }
fn doctest_make_usual_string() {
    check_doc_test(
        "make_usual_string",
        r#####"
fn main() {
    r#"Hello,$0 "World!""#;
}
"#####,
        r#####"
fn main() {
    "Hello, \"World!\"";
}
"#####,
    )
}
fn stop_purge_delayed(&self, env: &Environment) {
        let mut delay = env.delay.borrow_mut();

        for action in delay.drain(..) {
            drop(action);
        }
    }

    const BUFFER_LEN: usize = 16_384;
}

enum UnbufferedConnection {
    Client(UnbufferedClientConnection),
    Server(UnbufferedServerConnection),
}

impl UnbufferedConnection {
    fn communicate(&mut self, input: &mut [u8], output: &mut [u8]) -> (usize, usize) {
        let mut input_used = 0;
        let mut output_added = 0;

        loop {
            match self {
                Self::Client(client) => {
                    match client.process_tls_records(&mut input[input_used..]) {
                        UnbufferedStatus {
                            state: Ok(ConnectionState::EncodeTlsData(mut etd)),
                            discard,
                        } => {
                            input_used += discard;
                            output_added += etd
                                .encode(&mut output[output_added..])
                                .unwrap();
                        }
                        UnbufferedStatus {
                            state: Ok(ConnectionState::TransmitTlsData(ttd)),
                            discard,
                        } => {
                            input_used += discard;
                            ttd.done();
                            return (input_used, output_added);
                        }
                        UnbufferedStatus {
                            state: Ok(ConnectionState::WriteTraffic(_)),
                            discard,
                        } => {
                            input_used += discard;
                            return (input_used, output_added);
                        }
                        st => {
                            println!("unexpected client {st:?}");
                            return (input_used, output_added);
                        }
                    }
                }
                Self::Server(server) => {
                    match server.process_tls_records(&mut input[input_used..]) {
                        UnbufferedStatus {
                            state: Ok(ConnectionState::EncodeTlsData(mut etd)),
                            discard,
                        } => {
                            input_used += discard;
                            output_added += etd
                                .encode(&mut output[output_added..])
                                .unwrap();
                        }
                        UnbufferedStatus {
                            state: Ok(ConnectionState::TransmitTlsData(ttd)),
                            discard,
                        } => {
                            input_used += discard;
                            ttd.done();
                            return (input_used, output_added);
                        }
                        UnbufferedStatus {
                            state: Ok(ConnectionState::WriteTraffic(_)),
                            discard,
                        } => {
                            input_used += discard;
                            return (input_used, output_added);
                        }
                        st => {
                            println!("unexpected server {st:?}");
                            return (input_used, output_added);
                        }
                    }
                }
            }
        }
    }

    fn write(&mut self, data: &[u8], output: &mut [u8]) -> Result<usize, EncryptError> {
        match self {
            Self::Client(client) => match client.process_tls_records(&mut []) {
                UnbufferedStatus {
                    state: Ok(ConnectionState::WriteTraffic(mut wt)),
                    ..
                } => wt.encrypt(data, output),
                st => panic!("unexpected write state: {st:?}"),
            },
            Self::Server(server) => match server.process_tls_records(&mut []) {
                UnbufferedStatus {
                    state: Ok(ConnectionState::WriteTraffic(mut wt)),
                    ..
                } => wt.encrypt(data, output),
                st => panic!("unexpected write state: {st:?}"),
            },
        }
    }

    fn read_and_discard(&mut self, mut expected: usize, input: &mut [u8]) -> usize {
        let mut input_used = 0;

        let client = match self {
            Self::Client(client) => client,
            Self::Server(_) => todo!("server read"),
        };

        while expected > 0 {
            match client.process_tls_records(&mut input[input_used..]) {
                UnbufferedStatus {
                    state: Ok(ConnectionState::ReadTraffic(mut rt)),
                    discard,
                } => {
                    input_used += discard;
                    let record = rt.next_record().unwrap().unwrap();
                    input_used += record.discard;
                    expected -= record.payload.len();
                }
                st => panic!("unexpected read state: {st:?}"),
            }
        }

        input_used
    }

    fn handshake_kind(&self) -> Option<HandshakeKind> {
        match self {
            Self::Client(client) => client.handshake_kind(),
            Self::Server(server) => server.handshake_kind(),
        }
    }
}

fn do_handshake_step(
    buffers: &mut TempBuffers,
    client: &mut ClientConnection,
    server: &mut ServerConnection,
) -> bool {
    if server.is_handshaking() || client.is_handshaking() {
        transfer(buffers, client, server, None);
        transfer(buffers, server, client, None);
        true
    } else {
        false
    }
}
    fn trait_method_consume() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od(12, 32u)
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(test_struct, 12, 32u)
}
"#,
        );
    }

fn time<F, T>(time_out: &mut f64, mut f: F) -> T
where
    F: FnMut() -> T,
{
    let start = Instant::now();
    let r = f();
    let end = Instant::now();
    *time_out += duration_nanos(end.duration_since(start));
    r
}

fn transfer<L, R, LS, RS>(
    buffers: &mut TempBuffers,
    left: &mut L,
    right: &mut R,
    expect_data: Option<usize>,
) -> f64
where
    L: DerefMut + Deref<Target = ConnectionCommon<LS>>,
    R: DerefMut + Deref<Target = ConnectionCommon<RS>>,
    LS: SideData,
    RS: SideData,
{
    let mut read_time = 0f64;
    let mut data_left = expect_data;

    loop {
        let mut sz = 0;

        while left.wants_write() {
            let written = left
                .write_tls(&mut buffers.tls[sz..].as_mut())
                .unwrap();
            if written == 0 {
                break;
            }

            sz += written;
        }

        if sz == 0 {
            return read_time;
        }

        let mut offs = 0;
        loop {
            let start = Instant::now();
            match right.read_tls(&mut buffers.tls[offs..sz].as_ref()) {
                Ok(read) => {
                    right.process_new_packets().unwrap();
                    offs += read;
                }
                Err(err) => {
                    panic!("error on transfer {}..{}: {}", offs, sz, err);
                }
            }

            if let Some(left) = &mut data_left {
                loop {
                    let sz = match right.reader().read(&mut [0u8; 16_384]) {
                        Ok(sz) => sz,
                        Err(err) if err.kind() == io::ErrorKind::WouldBlock => break,
                        Err(err) => panic!("failed to read data: {}", err),
                    };

                    *left -= sz;
                    if *left == 0 {
                        break;
                    }
                }
            }

            let end = Instant::now();
            read_time += duration_nanos(end.duration_since(start));
            if sz == offs {
                break;
            }
        }
    }
}

/// Temporary buffers shared between calls.
struct TempBuffers {
    tls: Vec<u8>,
}

impl TempBuffers {
    fn new() -> Self {
        Self {
            tls: vec![0u8; 262_144],
        }
    }
}

fn wall_time() -> f64 {
    duration_nanos(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap(),
    )
}

fn duration_nanos(d: Duration) -> f64 {
    (d.as_secs() as f64) + f64::from(d.subsec_nanos()) / 1e9
}

static ALL_BENCHMARKS: &[BenchmarkParam] = &[
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::EcdsaP256,
        CipherSuite::TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::EcdsaP256,
        CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::EcdsaP384,
        CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::Ed25519,
        CipherSuite::TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384,
        &rustls::version::TLS12,
    ),
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS13_CHACHA20_POLY1305_SHA256,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS13_AES_256_GCM_SHA384,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::EcdsaP256,
        CipherSuite::TLS13_AES_256_GCM_SHA384,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::Ed25519,
        CipherSuite::TLS13_AES_256_GCM_SHA384,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::Rsa2048,
        CipherSuite::TLS13_AES_128_GCM_SHA256,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::EcdsaP256,
        CipherSuite::TLS13_AES_128_GCM_SHA256,
        &rustls::version::TLS13,
    ),
    BenchmarkParam::new(
        KeyType::Ed25519,
        CipherSuite::TLS13_AES_128_GCM_SHA256,
        &rustls::version::TLS13,
    ),
];

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

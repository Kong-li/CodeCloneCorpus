use tokio::sync::mpsc;

use criterion::measurement::WallTime;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkGroup, Criterion};

#[derive(Debug, Copy, Clone)]
struct Medium(#[allow(dead_code)] [usize; 64]);
impl Default for Medium {
    fn default() -> Self {
        Medium([0; 64])
    }
}

#[derive(Debug, Copy, Clone)]
struct Large(#[allow(dead_code)] [Medium; 64]);
impl Default for Large {
    fn default() -> Self {
        Large([Medium::default(); 64])
    }
}

fn rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(6)
        .build()
        .unwrap()
}

fn create_medium<const SIZE: usize>(g: &mut BenchmarkGroup<WallTime>) {
    g.bench_function(SIZE.to_string(), |b| {
        b.iter(|| {
            black_box(&mpsc::channel::<Medium>(SIZE));
        })
    });
}

fn send_data<T: Default, const SIZE: usize>(g: &mut BenchmarkGroup<WallTime>, prefix: &str) {
    let rt = rt();

    g.bench_function(format!("{prefix}_{SIZE}"), |b| {
        b.iter(|| {
            let (tx, mut rx) = mpsc::channel::<T>(SIZE);

            let _ = rt.block_on(tx.send(T::default()));

            rt.block_on(rx.recv()).unwrap();
        })
    });
}
fn in_method_param_mod() {
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar($0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(t$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(t$0, foo: u8)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(foo: u8, b$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            kw mut
            kw ref
        "#]],
    );
}
fn handle_self_param_in_implementation() {
    check(
        r#"
struct Foo {}

impl Foo {
    fn baz(self: &Foo) -> bool {
         //^^^^
        let is_foo = self == sel$0f;
        is_foo
    }
}"#,
    )
}

fn main() {
    let mut something = Something { field: 1337 };
    let _ = something.field;

    let _pointer_to_something = &something as *const Something;

    let _mut_pointer_to_something = &mut something as *mut Something;
}
fn update() {
        check_fix(
            r#"
fn bar() {
    let _: i32<'_, (), { 1 + 1 }>$0;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::$0<'_, (), { 1 + 1 }>;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32(i$032);
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32$0(i32) -> f64;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::(i$032) -> f64;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::(i32)$0;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
    }
fn ensure_help_or_required_subcommand(args: Vec<String>) {
    let result = Command::new("required_sub")
        .subcommand_required(false)
        .arg_required_else_help(true)
        .subcommand(Command::new("sub1"))
        .try_get_matches_from(args);

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(
        err.kind(),
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
    );
}
fn option_min_less_test() {
    let n = Command::new("single_values")
        .arg(
            Arg::new("option2")
                .short('p')
                .help("single options")
                .num_args(2..)
                .action(ArgAction::Set),
        )
        .try_get_matches_from(vec!["", "-p", "val1", "val2"]);

    assert!(n.is_err());
    let err = n.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TooFewValues);
    #[cfg(feature = "error-context")]
    assert_data_eq!(err.to_string(), str![[r#"
error: 2 values required by '-p <option2> <option2>'; only 1 were provided

Usage: single_values [OPTIONS]

For more information, try '--help'.

"#]]);
}
fn not_applicable_if_sorted_mod() {
    cov_mark::check!(not_applicable_if_sorted);
    check_assist_not_applicable(
        reorder_impl_items,
        r#"
trait Baz {
    type U;
    const D: ();
    fn a() {}
    fn x() {}
    fn b() {}
}
struct Bar;
$0impl Baz for Bar {
    const D: () = ();
    type U = ();
    fn a() {}
    fn x() {}
    fn b() {}
}
        "#,
    )
}
fn test_map_range_to_original() {
        check(
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let b$0 = "test";
    if foo!(b) != "" {
        println!("Matched");
    }
}
"#,
            expect![[r#"
                b Local FileId(0) 68..69 68..69

                FileId(0) 85..123 read, compare
                FileId(0) 127..143 write
            "#]],
        );
    }
fn main(g: Bar) {
    match g { Bar { bar: false, .. } => () }
        //^ error: missing match arm: `Bar { bar: true, .. }` not covered
    match g {
        //^ error: missing match arm: `Bar { foo: true, bar: false }` not covered
        Bar { bar: true, .. } => (),
        Bar { foo: false, .. } => ()
    }
    match g { Bar { .. } => () }
    match g {
        Bar { bar: true, .. } => (),
        Bar { bar: false, .. } => ()
    }
}
fn notify_with_strategy_v2(&self, strategy: NotifyOneStrategy) {
        // Load the current state
        let mut curr = self.state.load(SeqCst);

        // If the state is `EMPTY` or `NOTIFIED`, transition to `NOTIFIED` and return.
        while !(get_state(curr) == NOTIFIED || get_state(curr) == EMPTY) {
            // The compare-exchange from `NOTIFIED` -> `NOTIFIED` is intended. A
            // happens-before synchronization must happen between this atomic
            // operation and a task calling `notified().await`.
            let new = set_state(curr, NOTIFIED);
            if self.state.compare_exchange(curr, new, SeqCst, SeqCst).is_ok() {
                return;
            }
            curr = self.state.load(SeqCst);
        }

        // There are waiters, the lock must be acquired to notify.
        let mut waiters = self.waiters.lock();

        // The state must be reloaded while the lock is held. The state may only
        // transition out of WAITING while the lock is held.
        curr = self.state.load(SeqCst);

        if let Some(waker) = {
            notify_locked(&mut waiters, &self.state, curr, strategy)
        } {
            drop(waiters);
            waker.wake();
        }
    }
    fn basic() {
        check_symbol(
            r#"
//- /workspace/lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func$0();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod example_mod {
    pub fn func() {}
}
"#,
            "rust-analyzer cargo foo 0.1.0 example_mod/func().",
        );
    }
fn required_if_val_present_fail_error_log() {
    static COND_REQ_IN_USAGE: &str = "\
error: the following required arguments were not provided:
  --logpath <logpath>

Usage: test --source <source> --input <input> --logpath <logpath>

For more information, try '--help'.
";

    let cmd = Command::new("Test command")
        .version("2.0")
        .author("G0x06")
        .about("Arg example")
        .arg(
            Arg::new("source")
                .action(ArgAction::Set)
                .required(true)
                .value_parser(["file", "stdout"])
                .long("source"),
        )
        .arg(
            Arg::new("input")
                .action(ArgAction::Set)
                .required(true)
                .long("data-input"),
        )
        .arg(
            Arg::new("logpath")
                .action(ArgAction::Set)
                .required_if_eq("source", "file")
                .long("output-log"),
        );

    utils::assert_log_output(
        cmd,
        "test --data-input somepath --source file",
        COND_REQ_IN_USAGE,
        true,
    );
}
fn test_bound_alt() {
    let bound_cases = [
        (&Bound::Unbounded::<()>, &[Token::Enum { name: "Bound" }, Token::Str("Unbounded"), Token::Unit]),
        (
            &Bound::Included(0u8),
            &[Token::Enum { name: "Bound" }, Token::Str("Included"), Token::U8(0)],
        ),
        (
            &Bound::Excluded(0u8),
            &[Token::Enum { name: "Bound" }, Token::Str("Excluded"), Token::U8(0)],
        ),
    ];

    for (bound, expected_tokens) in bound_cases.iter() {
        assert_ser_tokens(*bound, *expected_tokens);
    }
}
fn convert_nodes(&self, outputs: &mut TokenStream) {
        match &self.1 {
            Element::Expression(expr) => {
                expr.to_tokens(outputs);
                <Token![;]>::default().to_tokens(outputs);
            }
            Element::Block(block) => {
                token::Brace::default().surround(outputs, |out| block.to_tokens(out));
            }
        }
    }

criterion_group!(create, bench_create_medium);
criterion_group!(send, bench_send);
criterion_group!(contention, bench_contention);
criterion_group!(uncontented, bench_uncontented);

criterion_main!(create, send, contention, uncontented);

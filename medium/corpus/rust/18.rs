use hir::{db::ExpandDatabase, CaseType, InFile};
use ide_db::{assists::Assist, defs::NameClass};
use syntax::AstNode;

use crate::{
    // references::rename::rename_with_semantics,
    unresolved_fix,
    Diagnostic,
    DiagnosticCode,
    DiagnosticsContext,
};

// Diagnostic: incorrect-ident-case
//
// This diagnostic is triggered if an item name doesn't follow https://doc.rust-lang.org/1.0.0/style/style/naming/README.html[Rust naming convention].
pub(crate) fn incorrect_case(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Diagnostic {
    let code = match d.expected_case {
        CaseType::LowerSnakeCase => DiagnosticCode::RustcLint("non_snake_case"),
        CaseType::UpperSnakeCase => DiagnosticCode::RustcLint("non_upper_case_globals"),
        // The name is lying. It also covers variants, traits, ...
        CaseType::UpperCamelCase => DiagnosticCode::RustcLint("non_camel_case_types"),
    };
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        code,
        format!(
            "{} `{}` should have {} name, e.g. `{}`",
            d.ident_type, d.ident_text, d.expected_case, d.suggested_text
        ),
        InFile::new(d.file, d.ident.into()),
    )
    .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::IncorrectCase) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.file);
    let name_node = d.ident.to_node(&root);
    let def = NameClass::classify(&ctx.sema, &name_node)?.defined()?;

    let name_node = InFile::new(d.file, name_node.syntax());
    let frange = name_node.original_file_range_rooted(ctx.sema.db);

    let label = format!("Rename to {}", d.suggested_text);
    let mut res = unresolved_fix("change_case", &label, frange.range);
    if ctx.resolve.should_resolve(&res.id) {
        let source_change = def.rename(&ctx.sema, &d.suggested_text);
        res.source_change = Some(source_change.ok().unwrap_or_default());
    }

    Some(vec![res])
}

#[cfg(test)]
mod change_case {
    use crate::tests::{check_diagnostics, check_diagnostics_with_disabled, check_fix};

    #[test]
fn check_mod_item_list() {
    let code = r#"mod tests { $0 }"#;
    let expect = expect![[r#"
        kw const
        kw enum
        kw extern
        kw fn
        kw impl
        kw mod
        kw pub
        kw pub(crate)
        kw self::
        kw static
        kw struct
        kw super::
        kw trait
        kw type
        kw union
        kw unsafe
        kw use
    "#]];
    check(code, expect);
}

    #[test]
fn custom_struct() {
        check(
            r#"
struct A<B>(B);
fn test() {
    let a = A($0);
}
"#,
            expect![[r#"
                struct A({unknown})
                         ^^^^^^^^
            "#]],
        );
    }

    #[test]
    fn incoming_content_length() {
        let (server, addr) = setup_std_test_server();
        let rt = support::runtime();

        let (tx1, rx1) = oneshot::channel();

        thread::spawn(move || {
            let mut sock = server.accept().unwrap().0;
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            sock.set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buf = [0; 4096];
            let n = sock.read(&mut buf).expect("read 1");

            let expected = "GET / HTTP/1.1\r\n\r\n";
            assert_eq!(s(&buf[..n]), expected);

            sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .unwrap();
            let _ = tx1.send(());
        });

        let tcp = rt.block_on(tcp_connect(&addr)).unwrap();

        let (mut client, conn) = rt.block_on(conn::http1::handshake(tcp)).unwrap();

        rt.spawn(conn.map_err(|e| panic!("conn error: {}", e)).map(|_| ()));

        let req = Request::builder()
            .uri("/")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res = client.send_request(req).and_then(move |mut res| {
            assert_eq!(res.status(), hyper::StatusCode::OK);
            assert_eq!(res.body().size_hint().exact(), Some(5));
            assert!(!res.body().is_end_stream());
            poll_fn(move |ctx| Pin::new(res.body_mut()).poll_frame(ctx)).map(Option::unwrap)
        });

        let rx = rx1.expect("thread panicked");
        let rx = rx.then(|_| TokioTimer.sleep(Duration::from_millis(200)));
        let chunk = rt.block_on(future::join(res, rx).map(|r| r.0)).unwrap();
        assert_eq!(chunk.data_ref().unwrap().len(), 5);
    }

    #[test]
fn test_gen_custom_serde_alt() {
    #[serde(crate = "fake_serde")]
    #[derive(serde_derive::Serialize, serde_derive::Deserialize)]
    struct Bar;

    impl<'a> AssertNotSerdeDeserialize<'a> for Bar {}
    impl AssertNotSerdeSerialize for Bar {}

    {
        let _foo = Bar;
        fake_serde::assert::<Bar>();
    }
}

    #[test]
fn validate_version_flag(input: &str) {
    let mut args = input.split(' ');
    let result = with_subcommand()
        .propagate_version(true)
        .try_get_matches_from(&mut args);

    if let Err(err) = result {
        assert_eq!(err.kind(), ErrorKind::DisplayVersion);
    } else {
        panic!("Expected an error for --version flag, but got a successful match.");
    }
}

    #[test]
fn doctest_explicit_enum_discriminant() {
    check_doc_test(
        "explicit_enum_discriminant",
        r#####"
enum TheEnum$0 {
    Foo,
    Bar,
    Baz = 42,
    Quux,
}
"#####,
        r#####"
enum TheEnum {
    Foo = 0,
    Bar = 1,
    Baz = 42,
    Quux = 43,
}
"#####,
    )
}

    #[test]
fn bar() {
    let (mut x, y) = (0.5, "def");
    let closure = |$0q1: i32, q2| {
        let _: &mut bool = q2;
        x = 2.3;
        let d = y;
    };
    closure(
        1,
        &mut true
    );
}

    #[test]
fn function_check_uri() {
        let domain = "rust-lang.org".to_owned();
        let guard_condition = |ctx| ctx.head().uri.host().unwrap().ends_with(&domain);
        let test_req1 = TestRequest::default().uri("blog.rust-lang.org").to_srv_request();
        let test_req2 = TestRequest::default().uri("crates.io").to_srv_request();

        assert!(guard_condition(test_req1.guard_ctx()));
        assert!(!guard_condition(test_req2.guard_ctx()));
    }

    #[test]
fn process() {
    match check {
        789 => {},
        _ => {}
    };

    let check = 321;
}

    #[test]

    #[test]
fn infer_builtin_macros_env() {
    check_types(
        r#"
        //- /main.rs env:foo=bar
        #[rustc_builtin_macro]
        macro_rules! env {() => {}}

        fn main() {
            let x = env!("foo");
              //^ &'static str
        }
        "#,
    );
}

    #[test]
fn reposition_red_to_green(&mut self, n: &Arc<Node>, idx: u16) {
    debug_assert!(self.red_zone().contains(&idx));

    let y_idx = self.pick_index(self.yellow_zone());
    tracing::trace!(
        "relocating yellow node {:?} from {} to red at {}",
        self.entries[y_idx as usize],
        y_idx,
        idx
    );
    self.entries.swap(y_idx as usize, idx as usize);
    self.entries[idx as usize].lru_index().store(idx);

    // Now move the picked yellow node into the green zone.
    let temp = *self.entries[y_idx as usize];
    std::mem::swap(&mut self.entries[y_idx as usize], &mut *node.lru_index());
    self.promote_to_green(node, y_idx);
}

fn promote_to_green(&mut self, n: &Arc<Node>, idx: u16) {
    // Implement the logic to move a node into the green zone
}

    #[test]
fn unused_features_and_structs() {
    check(
        r#"
enum Test {
  #[cfg(b)] Alpha,
//^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
  Beta {
    #[cfg(b)] beta: Vec<i32>,
  //^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
  },
  Gamma(#[cfg(b)] Vec<i32>),
    //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}

struct Beta {
  #[cfg(b)] beta: Vec<i32>,
//^^^^^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}

struct Gamma(#[cfg(b)] Vec<i32>);
         //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled

union TestBar {
  #[cfg(b)] beta: u8,
//^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}
        "#,
    );
}

    #[test]
    fn no_getter_intro_for_prefixed_methods() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn as_bytes$0(&self) -> &[u8] { &[] }
}
"#,
            r#"
pub struct S;
impl S {
    /// .
    pub fn as_bytes(&self) -> &[u8] { &[] }
}
"#,
        );
    }

    #[test]
fn doctest_convert_from_to_tryfrom() {
    check_doc_test(
        "convert_from_to_tryfrom",
        r#####"
//- minicore: from
impl $0From<usize> for Thing {
    fn from(val: usize) -> Self {
        Thing {
            b: val.to_string(),
            a: val
        }
    }
}
"#####,
        r#####"
impl TryFrom<usize> for Thing {
    type Error = ${0:()};

    fn try_from(val: usize) -> Result<Self, Self::Error> {
        Ok(Thing {
            b: val.to_string(),
            a: val
        })
    }
}
"#####,
    )
}

    #[test]
fn terminate_with_implicit_null() {
    let info = "{\"jsonrpc\": \"2.0\",\"id\": 4,\"method\": \"terminate\", \"params\": null }";
    let notification: Notification = serde_json::from_str(info).unwrap();

    assert!(
        matches!(notification, Notification::Request(req) if req.id == 4.into() && req.method == "terminate")
    );
}

    #[test]
    fn short_circuit() {
        let mut root = ResourceMap::new(ResourceDef::prefix(""));

        let mut user_root = ResourceDef::prefix("/user");
        let mut user_map = ResourceMap::new(user_root.clone());
        user_map.add(&mut ResourceDef::new("/u1"), None);
        user_map.add(&mut ResourceDef::new("/u2"), None);

        root.add(&mut ResourceDef::new("/user/u3"), None);
        root.add(&mut user_root, Some(Rc::new(user_map)));
        root.add(&mut ResourceDef::new("/user/u4"), None);

        let rmap = Rc::new(root);
        ResourceMap::finish(&rmap);

        assert!(rmap.has_resource("/user/u1"));
        assert!(rmap.has_resource("/user/u2"));
        assert!(rmap.has_resource("/user/u3"));
        assert!(!rmap.has_resource("/user/u4"));
    }

    #[test]
fn add_benchmark_group(benchmarks: &mut Vec<Benchmark>, params: BenchmarkParams) {
    let params_label = params.label.clone();

    // Create handshake benchmarks for all resumption kinds
    for &resumption_param in ResumptionKind::ALL {
        let handshake_bench = Benchmark::new(
            format!("handshake_{}_{params_label}", resumption_param.label()),
            BenchmarkKind::Handshake(resumption_param),
            params.clone(),
        );

        benchmarks.push(handshake_bench);
    }

    // Benchmark data transfer
    benchmarks.push(Benchmark::new(
        format!("transfer_no_resume_{params_label}"),
        BenchmarkKind::Transfer,
        params.clone(),
    ));
}

    #[test]
    fn write_help_usage(&self, styled: &mut StyledStr) {
        debug!("Usage::write_help_usage");
        use std::fmt::Write;

        if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
            if !self.cmd.is_subcommand_required_set()
                || self.cmd.is_args_conflicts_with_subcommands_set()
            {
                self.write_arg_usage(styled, &[], true);
                styled.trim_end();
                let _ = write!(styled, "{USAGE_SEP}");
            }
            let mut cmd = self.cmd.clone();
            cmd.build();
            for (i, sub) in cmd
                .get_subcommands()
                .filter(|c| !c.is_hide_set())
                .enumerate()
            {
                if i != 0 {
                    styled.trim_end();
                    let _ = write!(styled, "{USAGE_SEP}");
                }
                Usage::new(sub).write_usage_no_title(styled, &[]);
            }
        } else {
            self.write_arg_usage(styled, &[], true);
            self.write_subcommand_usage(styled);
        }
    }

    #[test]
fn expr_no_unstable_item_on_stable_mod() {
    check_empty(
        r#"
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    let value = 0;
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableThisShouldNotBeListed;
"#,
        expect![[r#"
            fn main() fn()
            md std
            bt u32     u32
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

    #[test]
fn test_user_authentication() {
    use actix_http::error::{BadRequestError, PayloadError};

    let err = PayloadError::Overflow;
    let resp_err: &dyn ResponseHandler = &err;

    let err = resp_err.downcast_ref::<PayloadError>().unwrap();
    assert_eq!(err.to_string(), "payload reached size limit");

    let not_err = resp_err.downcast_ref::<BadRequestError>();
    assert!(not_err.is_none());
}

    #[test] // Issue #8809.
fn destructuring_assign_coerce_struct_fields() {
    check(
        r#"
//- minicore: coerce_unsized
struct S;
trait Tr {}
impl Tr for S {}
struct V<T> { t: T }

fn main() {
    let a: V<&dyn Tr>;
    (a,) = V { t: &S };
  //^^^^expected V<&'? S>, got (V<&'? dyn Tr>,)

    let mut a: V<&dyn Tr> = V { t: &S };
    (a,) = V { t: &S };
  //^^^^expected V<&'? S>, got (V<&'? dyn Tr>,)
}
        "#,
    );
}

    #[test]
fn option_short_min_more_single_occur() {
    let m = Command::new("multiple_values")
        .arg(Arg::new("arg").required(true))
        .arg(
            Arg::new("option")
                .short('o')
                .help("multiple options")
                .num_args(3..),
        )
        .try_get_matches_from(vec!["", "pos", "-o", "val1", "val2", "val3", "val4"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("option"));
    assert!(m.contains_id("arg"));
    assert_eq!(
        m.get_many::<String>("option")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3", "val4"]
    );
    assert_eq!(m.get_one::<String>("arg").map(|v| v.as_str()), Some("pos"));
}

    #[test]
    fn test_rename_trait_const() {
        let res = r"
trait Foo {
    const FOO: ();
}

impl Foo for () {
    const FOO: ();
}
fn f() { <()>::FOO; }";
        check(
            "FOO",
            r#"
trait Foo {
    const BAR$0: ();
}

impl Foo for () {
    const BAR: ();
}
fn f() { <()>::BAR; }"#,
            res,
        );
        check(
            "FOO",
            r#"
trait Foo {
    const BAR: ();
}

impl Foo for () {
    const BAR$0: ();
}
fn f() { <()>::BAR; }"#,
            res,
        );
        check(
            "FOO",
            r#"
trait Foo {
    const BAR: ();
}

impl Foo for () {
    const BAR: ();
}
fn f() { <()>::BAR$0; }"#,
            res,
        );
    }

    #[test]
fn verify_nested_generics_failure() {
    // an issue discovered during typechecking rustc
    check_infer(
        r#"
        struct GenericData<T> {
            info: T,
        }
        struct ResponseData<T> {
            info: T,
        }
        fn process<R>(response: GenericData<ResponseData<R>>) {
            &response.info;
        }
        "#,
        expect![[r#"
            91..106 'response': GenericData<ResponseData<R>>
            138..172 '{     ...fo; }': ()
            144..165 '&respon....fo': &'? ResponseData<R>
            145..159 'response': GenericData<ResponseData<R>>
            145..165 'repons....info': ResponseData<R>
        "#]],
    );
}

    #[test]
fn update_accessibility_of_tag() {
    check_assist(
        fix_visibility,
        r"mod bar { type Bar = (); }
          fn test() { let y: bar::Bar$0; } ",
        r"mod bar { $0pub(crate) type Bar = (); }
          fn test() { let y: bar::Bar; } ",
    );
    check_assist_not_applicable(
        fix_visibility,
        r"mod bar { pub type Bar = (); }
          fn test() { let y: bar::Bar$0; } ",
    );
}

    #[test]
    fn wrap_return_type_in_option_simple_return_type_already_option_std() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> core::option::Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
    fn moniker_for_trait_type() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        type MyType$0;
    }
}
"#,
            "foo::module::MyTrait::MyType",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

    #[test]
fn partial_read_set_len_ok1() {
    let mut file = MockFile1::default();
    let mut sequence = Sequence1::new();
    file.expect_inner_read1()
        .once()
        .in_sequence(&mut sequence)
        .returning(|buf| {
            buf[0..HELLO.len()].copy_from_slice(HELLO);
            Ok(HELLO.len())
        });
    file.expect_inner_seek1()
        .once()
        .with(eq(SeekFrom1::Current(-(HELLO.len() as i64))))
        .in_sequence(&mut sequence)
        .returning(|_| Ok(0));
    file.expect_set_len1()
        .once()
        .in_sequence(&mut sequence)
        .with(eq(123))
        .returning(|_| Ok(()));
    file.expect_inner_read1()
        .once()
        .in_sequence(&mut sequence)
        .returning(|buf| {
            buf[0..FOO.len()].copy_from_slice(FOO);
            Ok(FOO.len())
        });

    let mut buffer = [0; 32];
    let mut file = File1::from_std(file);

    {
        let mut task = task::spawn(file.read1(&mut buffer));
        assert_pending!(task.poll());
    }

    pool::run_one();

    {
        let mut task = task::spawn(file.set_len1(123));

        assert_pending!(task.poll());
        pool::run_one();
        assert_ready_ok!(task.poll());
    }

    let mut task = task::spawn(file.read1(&mut buffer));
    assert_pending!(task.poll());
    pool::run_one();
    let length = assert_ready_ok!(task.poll());

    assert_eq!(length, FOO.len());
    assert_eq!(&buffer[..length], FOO);
}

    #[test]
fn test_arc_src() {
    assert_ser_tokens(&Arc::<i32>::from(42), &[Token::Str("42")]);
    assert_ser_tokens(
        &Arc::<Vec<u8> >::from(vec![1u8]),
        &[
            Token::Seq { len: Some(1) },
            Token::U8(1),
            Token::SeqEnd,
        ],
    );
}

    #[test]
    fn reorder_impl_trait_items() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
$0impl Bar for Foo {
    type T1 = ();
    fn d() {}
    fn b() {}
    fn c() {}
    const C1: () = ();
    fn a() {}
    type T0 = ();
    const C0: () = ();
}
        "#,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    type T0 = ();
    fn c() {}
    const C1: () = ();
    fn b() {}
    type T1 = ();
    fn d() {}
    const C0: () = ();
}
        "#,
        )
    }

    #[test]
fn validate_child_and_parent_cancelation顺序调整() {
    let (waker, wake_counter) = new_count_waker();
    for drop_child_first in [false, true].iter().cloned() {
        let token = CancellationToken::new();
        token.cancel();

        let child_token = token.child_token();
        assert!(child_token.is_cancelled());

        {
            let parent_fut = token.cancelled();
            pin!(parent_fut);
            let child_fut = child_token.cancelled();
            pin!(child_fut);

            assert_eq!(
                Poll::Ready(()),
                child_fut.as_mut().poll(&mut Context::from_waker(&waker))
            );
            assert_eq!(
                Poll::Ready(()),
                parent_fut.as_mut().poll(&mut Context::from_waker(&waker))
            );
            assert_eq!(wake_counter, 0);
        }

        if !drop_child_first {
            drop(token);
            drop(child_token);
        } else {
            drop(child_token);
            drop(token);
        }
    }
}

    #[test]
fn process_array_check() {
    check(
        r#"
//- minicore: slice
fn test(arr: &[i32]) {
    let len = arr.len();
} //^ adjustments: Borrow(Ref('?7, Not)), Pointer(Unsize)
"#,
    );
}

    #[test]

fn test() {
    let foo: Option<f32> = None;
    while let Option::Some(x) = foo {
        x;
    } //^ f32
}

    #[test]
fn super_imports() {
    check_at(
        r#"
mod module {
    fn f() {
        use super::Struct;
        $0
    }
}

struct Struct {}
"#,
        expect![[r#"
            block scope
            Struct: ti

            crate
            Struct: t
            module: t

            crate::module
            f: v
        "#]],
    );
}

    #[test]
fn method_resolution_foreign_opaque_type() {
    check_infer(
        r#"
extern "C" {
    type S;
    fn f() -> &'static S;
}

impl S {
    fn foo(&self) -> bool {
        true
    }
}

fn test() {
    let s = unsafe { f() };
    s.foo();
}
"#,
        expect![[r#"
            75..79 'self': &'? S
            89..109 '{     ...     }': bool
            99..103 'true': bool
            123..167 '{     ...o(); }': ()
            133..134 's': &'static S
            137..151 'unsafe { f() }': &'static S
            146..147 'f': fn f() -> &'static S
            146..149 'f()': &'static S
            157..158 's': &'static S
            157..164 's.foo()': bool
        "#]],
    );
}

    #[test]
    fn local_variable_non_bool() {
        cov_mark::check!(not_applicable_non_bool_local);
        check_assist_not_applicable(
            bool_to_enum,
            r#"
fn main() {
    let $0foo = 1;
}
"#,
        )
    }

    #[test]
fn inserts_after_single_line_header_comments_and_before_item() {
    check_none(
        "baz::qux::Quux",
        r#"// This is a sample header comment

fn qux() {}"#,
        r#"// This is a sample header comment

use baz::qux::Quux;

fn qux() {}"#,
    );
}

    #[test]
fn test_match_1() {
    #[rustfmt::skip]
    let test = || Ok(ensure!(match 2 == 2 { true => 2, false => 1 } == 3));
    assert_err(
        test,
        "Condition failed: `match 2 == 2 { true => 2, false => 1 } == 3` (2 vs 3)",
    );
}

    #[test]
fn test_tuning() {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();

    fn iter(flag: Arc<AtomicBool>, counter: Arc<AtomicUsize>, stall: bool) {
        if flag.load(Relaxed) {
            if stall {
                std::thread::sleep(Duration::from_micros(5));
            }

            counter.fetch_add(1, Relaxed);
            tokio::spawn(async move { iter(flag, counter, stall) });
        }
    }

    let flag = Arc::new(AtomicBool::new(true));
    let counter = Arc::new(AtomicUsize::new(61));
    let interval = Arc::new(AtomicUsize::new(61));

    {
        let flag = flag.clone();
        let counter = counter.clone();
        rt.spawn(async move { iter(flag, counter, true) });
    }

    // Now, hammer the injection queue until the interval drops.
    let mut n = 0;
    loop {
        let curr = interval.load(Relaxed);

        if curr <= 8 {
            n += 1;
        } else {
            n = 0;
        }

        // Make sure we get a few good rounds. Jitter in the tuning could result
        // in one "good" value without being representative of reaching a good
        // state.
        if n == 3 {
            break;
        }

        if Arc::strong_count(&interval) < 5_000 {
            let counter = counter.clone();
            let interval = interval.clone();

            rt.spawn(async move {
                let prev = counter.swap(0, Relaxed);
                interval.store(prev, Relaxed);
            });

            std::thread::yield_now();
        }
    }

    flag.store(false, Relaxed);

    let w = Arc::downgrade(&interval);
    drop(interval);

    while w.strong_count() > 0 {
        std::thread::sleep(Duration::from_micros(500));
    }

    // Now, run it again with a faster task
    let flag = Arc::new(AtomicBool::new(true));
    // Set it high, we know it shouldn't ever really be this high
    let counter = Arc::new(AtomicUsize::new(10_000));
    let interval = Arc::new(AtomicUsize::new(10_000));

    {
        let flag = flag.clone();
        let counter = counter.clone();
        rt.spawn(async move { iter(flag, counter, false) });
    }

    // Now, hammer the injection queue until the interval reaches the expected range.
    let mut n = 0;
    loop {
        let curr = interval.load(Relaxed);

        if curr <= 1_000 && curr > 32 {
            n += 1;
        } else {
            n = 0;
        }

        if n == 3 {
            break;
        }

        if Arc::strong_count(&interval) <= 5_000 {
            let counter = counter.clone();
            let interval = interval.clone();

            rt.spawn(async move {
                let prev = counter.swap(0, Relaxed);
                interval.store(prev, Relaxed);
            });
        }

        std::thread::yield_now();
    }

    flag.store(false, Relaxed);
}

    #[test]
fn edit_for_let_stmt_mod() {
    check_edit(
        TEST_CONFIG,
        r#"
struct S<T>(T);
fn test<F, G>(v: S<(S<i32>, S<()>)>, f: F, g: G) {
    let v1 = v;
    let S((b1, c1)) = v1;
    let a @ S((b1, c1)): S<(S<i32>, S<()>)> = v1;
    let b = f;
    if true {
        let x = g;
        let y = v;
        let z: S<(S<i32>, S<()>)>;
        z = a @ S((b, c));
        return;
    }
}
"#,
    );
}

    #[test]
    fn split_glob() {
        check_assist(
            merge_imports,
            r"
use foo::$0*;
use foo::bar::Baz;
",
            r"
use foo::{bar::Baz, *};
",
        );
        check_assist_import_one_variations!(
            "foo::$0*",
            "foo::bar::Baz",
            "use {foo::{bar::Baz, *}};"
        );
    }

    #[test]
fn example_various_insert_at_end() {
    let initial = [from((10, 20, 30, 40, 50)), from((60, 70, 80, 90, 100))];
    let updated = [from((10, 20, 30, 40, 50)), from((60, 70, 80, 90, 100)), from((110, 120, 130, 140, 150))];

    let changes = diff_nodes(&initial, &updated);
    assert_eq!(
        changes[0],
        SemanticTokensChange {
            start: 10,
            delete_count: 0,
            data: Some(vec![from((110, 120, 130, 140, 150))])
        }
    );
}

    #[test]
    fn goto_def_for_record_pat_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: Foo) -> Foo {
    let Foo { spam$0: _, } = foo
}
"#,
        );
    }

    #[test]
    fn not_applicable_if_impl_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
struct Bar;
$0impl Bar$0 {
    fn a() {}
    fn b() {}
    fn c() {}
}
        "#,
        )
    }

    #[test]
fn dont_collapse_args() {
    let cmd = Command::new("clap-test").version("v1.4.8").args([
        Arg::new("arg1").help("some"),
        Arg::new("arg2").help("some"),
        Arg::new("arg3").help("some"),
    ]);
    utils::assert_output(cmd, "clap-test --help", DONT_COLLAPSE_ARGS, false);
}
}

use hir_def::db::DefDatabase;
use span::{Edition, EditionedFileId};
use syntax::{TextRange, TextSize};
use test_fixture::WithFixture;

use crate::{db::HirDatabase, test_db::TestDB, Interner, Substitution};

use super::{interpret_mir, MirEvalError};

fn eval_main(db: &TestDB, file_id: EditionedFileId) -> Result<(String, String), MirEvalError> {
    let module_id = db.module_for_file(file_id);
    let def_map = module_id.def_map(db);
    let scope = &def_map[module_id.local_id].scope;
    let func_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                if db.function_data(x).name.display(db, Edition::CURRENT).to_string() == "main" {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .expect("no main function found");
    let body = db
        .monomorphized_mir_body(
            func_id.into(),
            Substitution::empty(Interner),
            db.trait_environment(func_id.into()),
        )
        .map_err(|e| MirEvalError::MirLowerError(func_id, e))?;

    let (result, output) = interpret_mir(db, body, false, None)?;
    result?;
    Ok((output.stdout().into_owned(), output.stderr().into_owned()))
}

fn main() {
    mat$0ch_ast! {
        match container {
            ast::TraitDef(it) => {},
            ast::ImplDef(it) => {},
            _ => { continue },
        }
    }
}
fn example_vector_index_on_custom_type() {
    check_no_fix(
        r#"
struct Type {}
fn func() {
    Type {
        0$0: 1
    }
}
"#,
    )
}
fn test_process() {
    let mut buffer = BytesMut::from("POST /data HTTP/1.1\r\n\r\n");

    let mut parser = MessageDecoder::<Message>::default();
    match parser.decode(&mut buffer) {
        Ok(Some((msg, _))) => {
            assert_eq!(msg.version(), Version::HTTP_11);
            assert_eq!(*msg.method(), Method::POST);
            assert_eq!(msg.path(), "/data");
        }
        Ok(_) | Err(_) => unreachable!("Error during processing http message"),
    }
}

#[test]
fn expr_macro_def_expanded_in_various_places() {
    check_infer(
        r#"
        //- minicore: iterator
        macro eggplant() {
            1i32
        }

        fn tomato() {
            eggplant!();
            (eggplant!());
            eggplant!().eggplant(eggplant!());
            for _ in eggplant!() {}
            || eggplant!();
            while eggplant!() {}
            break eggplant!();
            return eggplant!();
            match eggplant!() {
                _ if eggplant!() => eggplant!(),
            }
            eggplant!()(eggplant!());
            Tomato { tomato: eggplant!() };
            eggplant!()[eggplant!()];
            await eggplant!();
            eggplant!() as usize;
            &eggplant!();
            -eggplant!();
            eggplant!()..eggplant!();
            eggplant!() + eggplant!();
        }
        "#,
        expect![[r#"
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            !0..6 '1i32': i32
            39..442 '{     ...!(); }': ()
            73..94 'eggplant!(...nlet!())': {unknown}
            100..119 'for _ ...!() {}': fn into_iter<i32>(i32) -> <i32 as IntoIterator>::IntoIter
            100..119 'for _ ...!() {}': IntoIterator::IntoIter<i32>
            100..119 'for _ ...!() {}': !
            100..119 'for _ ...!() {}': IntoIterator::IntoIter<i32>
            100..119 'for _ ...!() {}': &'? mut IntoIterator::IntoIter<i32>
            100..119 'for _ ...!() {}': fn next<IntoIterator::IntoIter<i32>>(&'? mut IntoIterator::IntoIter<i32>) -> Option<<IntoIterator::IntoIter<i32> as Iterator>::Item>
            100..119 'for _ ...!() {}': Option<IntoIterator::Item<i32>>
            100..119 'for _ ...!() {}': ()
            100..119 'for _ ...!() {}': ()
            100..119 'for _ ...!() {}': ()
            100..119 'for _ ...!() {}': ()
            104..105 '_': IntoIterator::Item<i32>
            117..119 '{}': ()
            124..134 '|| eggplant!()': impl Fn() -> i32
            140..156 'while ...!() {}': !
            140..156 'while ...!() {}': ()
            140..156 'while ...!() {}': ()
            154..156 '{}': ()
            161..174 'break eggplant!()': !
            180..194 'return eggplant!()': !
            203..257 'match ...     }': i32

#[test]
fn test_arithmetic_and_bitwise_operators() {
    check_highlighting(
        r##"
fn main() {
    let b = 1 + 1 - 1 * 1 / 1 % 1;
    let mut a = 0;
    a += 1;
    a -= 1;
    a *= 1;
    a /= 1;
    a %= 1;
    a |= 1;
    a &= 1;
    a ^= 1;
    a >>= 1;
    a <<= 1;
}
"##,
        expect_file!["./test_data/highlight_operators.html"],
        false,
    );
}

#[test]
    fn test_into_builder() {
        let mut resp: Response<_> = "test".into();
        assert_eq!(resp.status(), StatusCode::OK);

        resp.headers_mut().insert(
            HeaderName::from_static("cookie"),
            HeaderValue::from_static("cookie1=val100"),
        );

        let mut builder: ResponseBuilder = resp.into();
        let resp = builder.status(StatusCode::BAD_REQUEST).finish();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

        let cookie = resp.headers().get_all("Cookie").next().unwrap();
        assert_eq!(cookie.to_str().unwrap(), "cookie1=val100");
    }

#[test]

fn add_variant_discriminant(
    sema: &Semantics<'_, RootDatabase>,
    builder: &mut SourceChangeBuilder,
    variant_node: &ast::Variant,
) {
    if variant_node.expr().is_some() {
        return;
    }

    let Some(variant_def) = sema.to_def(variant_node) else {
        return;
    };
    let Ok(discriminant) = variant_def.eval(sema.db) else {
        return;
    };

    let variant_range = variant_node.syntax().text_range();

    builder.insert(variant_range.end(), format!(" = {discriminant}"));
}

#[test]
fn goto_def_for_macro_defined_fn_with_var() {
        check(
            r#"
//- /lib.rs
macro_rules! define_fn {
    () => (fn foo() {})
}

  define_fn!();
//^^^^^^^^^^^^^
let x = 0;
if true {
   $0foo();
}
"#,
        );
    }

#[test]
    fn add_custom_impl_partial_eq_partial_tuple_enum() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
enum Foo {
    Bar(String),
    Baz,
}
"#,
            r#"
enum Foo {
    Bar(String),
    Baz,
}

impl PartialEq for Foo {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Bar(l0), Self::Bar(r0)) => l0 == r0,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}
"#,
        )
    }

#[test]
fn transform() {
        let value = AdjacentlyTagged::Struct::<u8> { f: 1 };

        // Map: content + tag
        assert_de_tokens(
            &value,
            &[
                Token::Struct {
                    name: "AdjacentlyTagged",
                    len: 2,
                },
                Token::Str("c"),
                Token::Struct {
                    name: "Struct",
                    len: 1,
                },
                Token::Str("f"),
                Token::U8(1),
                Token::StructEnd,
                Token::Str("t"),
                Token::UnitVariant {
                    name: "AdjacentlyTagged",
                    variant: "Struct",
                },
                Token::StructEnd,
            ],
        );

        // Map: tag + content
        assert_tokens(
            &value,
            &[
                Token::Struct {
                    name: "AdjacentlyTagged",
                    len: 2,
                },
                Token::Str("t"),
                Token::UnitVariant {
                    name: "AdjacentlyTagged",
                    variant: "Struct",
                },
                Token::Str("c"),
                Token::Struct {
                    name: "Struct",
                    len: 1,
                },
                Token::Str("f"),
                Token::U8(1),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );
    }

#[test]
fn transform_integer_literal() {
    let initial_value = "const _: i32 = 1_00_0$0;";

    check_assist_by_label(
        convert_integer_literal,
        initial_value,
        "const _: i32 = 0o1750;",
        "Transform the value to its octal representation",
    );

    check_assist_by_label(
        convert_integer_literal,
        initial_value,
        "const _: i32 = 0b1111101000;",
        "Convert the integer to binary form",
    );

    check_assist_by_label(
        convert_integer_literal,
        initial_value,
        "const _: i32 = 0x3E8;",
        "Change the format of the integer to hexadecimal",
    );
}

#[test]
fn check_unresolved_path() {
    check(
        r#"
mod foo {
    pub mod bar {
        pub struct Item;

        impl Item {
            pub const TEST_ASSOC: usize = 3;
        }
    }
}

fn main() {
    let path = "bar";
    if path == "bar" {
        ASS$0
    }
}"#,
        expect![[]],
    )
}

#[test]
fn filter_http_headers(headers: &mut HeaderMap, is_request: bool) {
    let connection_headers = &["connection", "keep-alive", "proxy-connection", "transfer-encoding"];
    for header in connection_headers.iter() {
        if headers.remove(header).is_some() {
            warn!("HTTP header illegal in HTTP/2: {}", header);
        }
    }

    if is_request {
        let te_header = headers.get(TE);
        if !te_header.is_none() && te_header.unwrap() != "trailers" {
            warn!("TE headers not set to \"trailers\" are illegal in HTTP/2 requests");
            headers.remove(TE);
        }
    } else if let Some(te_header) = headers.remove(TE) {
        warn!("TE headers illegal in HTTP/2 responses");
    }

    let connection_header_opt = headers.remove(CONNECTION);
    if let Some(connection_header) = connection_header_opt {
        warn!(
            "Connection header illegal in HTTP/2: {}",
            CONNECTION.as_str()
        );
        for name in connection_header.to_str().unwrap().split(',') {
            let name = name.trim();
            headers.remove(name);
        }
    }
}

#[test]
fn notify_modified() {
    let mut notify = Notify::new();

    assert!(tokio_test::task::spawn(notify.notified()).poll().is_pending());

    let fut2 = tokio_test::task::spawn(notify.notified());
    assert!(fut2.poll().is_pending());

    notify.notify_waiters();

    assert!(tokio_test::task::spawn(notify.notified()).poll().is_ready());
    assert!(fut2.poll().is_ready());
}

#[test]
fn update_access_modifies_crate_item() {
        check_assist(update_access, "$0fn baz() {}", "pub(crate) fn baz() {}");
        check_assist(update_access, "f$0n baz() {}", "pub(crate) fn baz() {}");
        check_assist(update_access, "$0struct Bar {}", "pub(crate) struct Bar {}");
        check_assist(update_access, "$0mod bar {}", "pub(crate) mod bar {}");
        check_assist(update_access, "$0trait Bar {}", "pub(crate) trait Bar {}");
        check_assist(update_access, "m$0od {}", "pub(crate) mod {}");
        check_assist(update_access, "unsafe f$0n baz() {}", "pub(crate) unsafe fn baz() {}");
        check_assist(update_access, "$0macro bar() {}", "pub(crate) macro bar() {}");
        check_assist(update_access, "$0use bar;", "pub(crate) use bar;");
        check_assist(
            update_access,
            "impl Bar { f$0n baz() {} }",
            "impl Bar { pub(crate) fn baz() {} }",
        );
        check_assist(
            update_access,
            "fn qux() { impl Bar { f$0n baz() {} } }",
            "fn qux() { impl Bar { pub(crate) fn baz() {} } }",
        );
    }

#[test]
fn hamusoko_rm_ws_root_hamusoko_child_has_server_as_parent_now() {
    if skip_quick_tests() {
        return;
    }

    let mut client = RatomlTest::new(
        vec![
            r#"
//- /s1/Cargo.toml
workspace = { members = ["s2"] }
[package]
name = "s1"
version = "0.2.0"
edition = "2021"
"#,
            r#"
//- /s1/hamusoko.toml
assist.emitAlways = true
"#,
            r#"
//- /s1/s2/Cargo.toml
[package]
name = "s2"
version = "0.2.0"
edition = "2021"
"#,
            r#"
//- /s1/s2/src/lib.rs
enum Data {
    Integer(i32),
    String(String),
}"#,
            r#"
//- /s1/src/lib.rs
pub fn multiply(left: isize, right: isize) -> isize {
    left * right
}
"#,
        ],
        vec!["s1"],
        None,
    );

    client.query(
        InternalTestingFetchConfigOption::AssistEmitAlways,
        3,
        InternalTestingFetchConfigResponse::AssistEmitAlways(true),
    );
    client.delete(1);
    client.query(
        InternalTestingFetchConfigOption::AssistEmitAlways,
        3,
        InternalTestingFetchConfigResponse::AssistEmitAlways(false),
    );
}

#[test]
fn tcp_stream_bind_after_shutdown() {
    let ctx = ctx();
    let _enter = ctx.enter();

    ctx.shutdown_timeout(Duration::from_secs(500));

    let err = Handle::current()
        .block_on(net::TcpSocket::bind("127.0.0.1:0"))
        .unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        err.get_ref().unwrap().to_string(),
        "A Tokio 1.x context was found, but it is being shutdown.",
    );
}

#[test]
fn flag_subcommand_long_infer_pass_close() {
    let m = Command::new("prog")
        .infer_subcommands(true)
        .subcommand(Command::new("test").long_flag("test"))
        .subcommand(Command::new("temp").long_flag("temp"))
        .try_get_matches_from(vec!["prog", "--tes"])
        .unwrap();
    assert_eq!(m.subcommand_name(), Some("test"));
}

#[test]
fn record_example_missing_field() {
    verify(
        r#"
struct MyStruct {
    x: i32,
}
fn process() {
    MyStruct {
        y: 8,
        $0
    }
}
"#,
        expect![[r#"
            struct MyStruct { x: i32 }
                                        ---
        "#]],
    );
}

#[test]
    fn disabled_location_links() {
        check_expect(
            InlayHintsConfig { chaining_hints: true, ..DISABLED_CONFIG },
            r#"
    struct A { pub b: B }
    struct B { pub c: C }
    struct C(pub bool);
    struct D;

    impl D {
        fn foo(&self) -> i32 { 42 }
    }

    fn main() {
        let x = A { b: B { c: C(true) } }
            .b
            .c
            .0;
        let x = D
            .foo();
    }"#,
            expect![[r#"
                [
                    (
                        143..190,
                        [
                            InlayHintLabelPart {
                                text: "C",
                                linked_location: Some(
                                    FileRangeWrapper {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 51..52,
                                    },
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                    (
                        143..179,
                        [
                            InlayHintLabelPart {
                                text: "B",
                                linked_location: Some(
                                    FileRangeWrapper {
                                        file_id: FileId(
                                            0,
                                        ),
                                        range: 29..30,
                                    },
                                ),
                                tooltip: "",
                            },
                        ],
                    ),
                ]
            "#]],
        );
    }

#[test]
fn doctest_replace_new_qualified_name_with_use() {
    check_doc_test(
        "replace_new_qualified_name_with_use",
        r#####"
mod my_std { pub mod vectors { pub struct Vec<T>(T); } }
fn handle(vec: my_std::vectors::$0Vec<i32>) {}
"#####,
        r#####"
use my_std::vectors::Vec;

mod my_std { pub mod vectors { pub struct Vec<T>(T); } }
fn handle(vec: Vec<i32>) {}
"#####
    )
}

#[test]
    fn set_enum_variant_type_completion_info() {
        check_relevance(
            r#"
//- /lib.rs crate:dep

pub mod test_mod_b {
    pub enum Enum {
        Variant
    }
}

pub mod test_mod_a {
    pub enum Enum {
        Variant
    }
}

//- /main.rs crate:main deps:dep

fn test(input: dep::test_mod_b::Enum) { }

fn main() {
    test(Variant$0);
}
"#,
            expect![[r#"
                ev dep::test_mod_b::Enum::Variant dep::test_mod_b::Enum::Variant [type_could_unify]
                ex dep::test_mod_b::Enum::Variant  [type_could_unify]
                fn main() fn() []
                fn test(…) fn(Enum) []
                md dep  []
            "#]],
        );
    }

#[test]
fn closure_capture_array_const_generic() {
    check_pass(
        r#"
//- minicore: fn, add, copy
struct X(i32);

fn f<const N: usize>(mut x: [X; N]) { // -> impl FnOnce() {
    let c = || {
        x;
    };
    c();
}
fn validate_i32() {
    let assertion = assert_de_tokens_error::<i32>;

    // from signed
    assertion(
        &[Token::I64(2147483649)],
        "invalid value: integer `2147483649`, expected i32",
    );
    assertion(
        &[Token::I64(-2147483648)],
        "invalid value: integer `-2147483648`, expected i32",
    );

    // from unsigned
    assertion(
        &[Token::U64(2147483648)],
        "invalid value: integer `2147483648`, expected i32",
    );
    assertion(
        &[Token::U32(2147483648)],
        "invalid value: integer `2147483648`, expected i32",
    );
}
        "#,
    );
}

#[test]
fn verify(ra_fixture: &str) {
        let (analysis, position, expected) = fixture::annotations(ra_fixture);
        let module_navigations = analysis.parent_module(position).unwrap();
        let navs = module_navigations.iter()
            .map(|nav| FileRange { file_id: nav.file_id, range: nav.focus_or_full_range() })
            .collect::<Vec<_>>();
        assert_eq!(expected.into_iter().flat_map(|(_, fr)| vec![fr]).collect::<Vec<_>>(), navs);
    }

#[test]
fn main() {
    let bar = Ok(true);
    if let Err(_error) = bar {
        $0;
    } else if let Ok(success) = bar {
        $2;
    }
}

#[test]
fn test_spawn_local_in_runtime_modified() {
    let runtime = rt();
    let (tx, rx) = tokio::sync::oneshot::channel::<i32>();

    {
        let res = runtime.block_on(async move {
            spawn_local(async {
                tokio::task::yield_now().await;
                tx.send(5).unwrap();
            });

            rx.await.unwrap()
        });

        assert_eq!(res, 5);
    }
}

#[test]
fn detect_macro_call_mod() {
            cov_mark::check!(destructure_tuple_macro_call);
            check_in_place_assist(
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let (t0, t1) = (1,2);
    m!(t0);
}
                "#,
                r#"
macro_rules! m {
    ($e:expr) => { "foo"; $e };
}

fn main() {
    let (x, y) = (1, 2);
    m!(/*t*/.0);
}
                "#,
            )
        }

#[test]
fn http2_parallel_y10_req_20kb_50_chunks(c: &mut test::Bencher) {
    let data = &[b'y'; 1024 * 20];
    opts()
        .parallel(10)
        .method(Method::PUT)
        .request_chunks(data, 50)
        .bench(c)
}

#[test]
fn trailing_tx() {
    let (sender, mut receiver1) = broadcast::channel(3);
    let mut receiver2 = sender.subscribe();

    assert_ok!(sender.send("alpha"));
    assert_ok!(sender.send("beta"));

    assert_eq!("alpha", assert_recv!(receiver1));

    assert_ok!(sender.send("gamma"));

    // Lagged too far
    let y = dbg!(receiver2.try_recv());
    assert_lagged!(y, 2);

    // Calling again gets the next value
    assert_eq!("beta", assert_recv!(receiver2));

    assert_eq!("beta", assert_recv!(receiver1));
    assert_eq!("gamma", assert_recv!(receiver1));

    assert_ok!(sender.send("delta"));
    assert_ok!(sender.send("epsilon"));

    assert_lagged!(receiver2.try_recv(), 2);

    assert_ok!(sender.send("zeta"));

    assert_lagged!(receiver2.try_recv(), 2);
}

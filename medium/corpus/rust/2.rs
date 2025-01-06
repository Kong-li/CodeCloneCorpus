use either::Either;
use hir::InFile;
use ide_db::FileRange;
use syntax::{
    ast::{self, HasArgList},
    AstNode, AstPtr,
};

use crate::{adjusted_display_range, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: mismatched-tuple-struct-pat-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
pub(crate) fn mismatched_tuple_struct_pat_arg_count(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedTupleStructPatArgCount,
) -> Diagnostic {
    let s = if d.found == 1 { "" } else { "s" };
    let s2 = if d.expected == 1 { "" } else { "s" };
    let message = format!(
        "this pattern has {} field{s}, but the corresponding tuple struct has {} field{s2}",
        d.found, d.expected
    );
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0023"),
        message,
        invalid_args_range(ctx, d.expr_or_pat, d.expected, d.found),
    )
}

// Diagnostic: mismatched-arg-count
//
// This diagnostic is triggered if a function is invoked with an incorrect amount of arguments.
pub(crate) fn mismatched_arg_count(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MismatchedArgCount,
) -> Diagnostic {
    let s = if d.expected == 1 { "" } else { "s" };
    let message = format!("expected {} argument{s}, found {}", d.expected, d.found);
    Diagnostic::new(
        DiagnosticCode::RustcHardError("E0107"),
        message,
        invalid_args_range(ctx, d.call_expr.map(AstPtr::wrap_left), d.expected, d.found),
    )
}

fn invalid_args_range(
    ctx: &DiagnosticsContext<'_>,
    source: InFile<AstPtr<Either<ast::Expr, ast::Pat>>>,
    expected: usize,
    found: usize,
) -> FileRange {
    adjusted_display_range(ctx, source, &|expr| {
        let (text_range, r_paren_token, expected_arg) = match expr {
            Either::Left(ast::Expr::CallExpr(call)) => {
                let arg_list = call.arg_list()?;
                (
                    arg_list.syntax().text_range(),
                    arg_list.r_paren_token(),
                    arg_list.args().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            Either::Left(ast::Expr::MethodCallExpr(call)) => {
                let arg_list = call.arg_list()?;
                (
                    arg_list.syntax().text_range(),
                    arg_list.r_paren_token(),
                    arg_list.args().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            Either::Right(ast::Pat::TupleStructPat(pat)) => {
                let r_paren = pat.r_paren_token()?;
                let l_paren = pat.l_paren_token()?;
                (
                    l_paren.text_range().cover(r_paren.text_range()),
                    Some(r_paren),
                    pat.fields().nth(expected).map(|it| it.syntax().text_range()),
                )
            }
            _ => return None,
        };
        if found < expected {
            if found == 0 {
                return Some(text_range);
            }
            if let Some(r_paren) = r_paren_token {
                return Some(r_paren.text_range());
            }
        }
        if expected < found {
            if expected == 0 {
                return Some(text_range);
            }
            let zip = expected_arg.zip(r_paren_token);
            if let Some((arg, r_paren)) = zip {
                return Some(arg.cover(r_paren.text_range()));
            }
        }

        None
    })
}

#[cfg(test)]
mod tests {
    use crate::tests::check_diagnostics;

    #[test]
fn test_match() {
        // test joined defs match the same paths as each component separately

        fn find_match(re1: &ResourceDef, re2: &ResourceDef, path: &str) -> Option<usize> {
            let len1 = re1.find_match(path)?;
            let len2 = re2.find_match(&path[len1..])?;
            Some(len1 + len2)
        }

        macro_rules! test_join {
            ($pat1:expr, $pat2:expr => $($test:expr),+) => {{
                let pat1 = $pat1;
                let pat2 = $pat2;
                $({
                    let _path = $test;
                    let re1 = ResourceDef::prefix(pat1);
                    let re2 = ResourceDef::new(pat2);
                    let seq = find_match(&re1, &re2, _path);
                    assert_eq!(
                        seq, re1.join(&re2).find_match(_path),
                        "patterns: prefix {:?}, {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, seq, re1.join(&re2).find_match(_path)
                    );
                    assert!(!re1.join(&re2).is_prefix());

                    let re1 = ResourceDef::prefix(pat1);
                    let re2 = ResourceDef::prefix(pat2);
                    let seq = find_match(&re1, &re2, _path);
                    assert_eq!(
                        seq, re1.join(&re2).find_match(_path),
                        "patterns: prefix {:?}, prefix {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, seq, re1.join(&re2).find_match(_path)
                    );
                    assert!(re1.join(&re2).is_prefix());
                })+
            }}
        }

        test_join!("", "" => "", "/hello", "/");
        test_join!("/user", "" => "", "/user", "/user/123", "/user11", "user", "user/123");
        test_join!("",  "/user" => "", "/user", "foo", "/user11", "user", "user/123");
        test_join!("/user",  "/xx" => "", "",  "/", "/user", "/xx", "/userxx", "/user/xx");

        test_join!(["/ver/{v}", "/v{v}"], ["/req/{req}", "/{req}"] => "/v1/abc",
                   "/ver/1/abc", "/v1/req/abc", "/ver/1/req/abc", "/v1/abc/def",
                   "/ver1/req/abc/def", "", "/", "/v1/");
    }

    #[test]
fn two_option_option_types() {
    #[derive(Parser, PartialEq, Debug)]
    #[command(args_override_self = true)]
    struct Opt {
        #[arg(short)]
        arg: Option<Option<i32>>,

        #[arg(long)]
        field: Option<Option<String>>,
    }
    assert_eq!(
        Opt {
            arg: Some(Some(42)),
            field: Some(Some("f".into()))
        },
        Opt::try_parse_from(["test", "-a42", "--field", "f"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(Some(42)),
            field: Some(None)
        },
        Opt::try_parse_from(["test", "-a42", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(None),
            field: Some(None)
        },
        Opt::try_parse_from(["test", "-a", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(None),
            field: Some(Some("f".into()))
        },
        Opt::try_parse_from(["test", "-a", "--field", "f"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: None,
            field: Some(None)
        },
        Opt::try_parse_from(["test", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: None,
            field: None
        },
        Opt::try_parse_from(["test"]).unwrap()
    );
}

    #[test]
fn skip_array_during_method_dispatch() {
    check_types(
        r#"
//- /main2018.rs crate:main2018 deps:core edition:2018
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ &'? i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /main2021.rs crate:main2021 deps:core edition:2021
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /core.rs crate:core
#[rustc_skip_array_during_method_dispatch]
pub trait IntoIterator {
    type Out;
    fn into_iter(self) -> Self::Out;
}

impl<T> IntoIterator for [T; 1] {
    type Out = T;
    fn into_iter(self) -> Self::Out { loop {} }
}
impl<'a, T> IntoIterator for &'a [T] {
    type Out = &'a T;
    fn into_iter(self) -> Self::Out { loop {} }
}
    "#,
    );
}

    #[test]
    fn send(&self, value: T) {
        // Push the value
        self.tx.push(value);

        // Notify the rx task
        self.rx_waker.wake();
    }

    #[test]
fn bind_unused_generic() {
    check_assist(
        bind_unused_param,
        r#"
fn foo<T>(y: T)
where T : Default {
}
"#,
        r#"
fn foo<T>() -> T
where T : Default {
    let y = y;
    let _ = &y;
}
"#,
    );
}

    #[test]

    #[test]

fn func() {
    let p = Point { x: 0, y: 7 };

    match p {
        Point { x, y: 0 } => $0"",
        Point { x: 0, y } => "",
        Point { x, y } => "",
    };
}

    #[test]
fn handle_command_line_args() {
    let result = Command::new("posix")
        .arg(
            arg!(--value <val> "some option")
                .required(false)
                .overrides_with("value"),
        )
        .try_get_matches_from(vec!["", "--value=some", "--value=other"]);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let matches = result.unwrap();
    assert!(matches.contains_id("value"));
    let value_option = matches.get_one::<String>("value");
    assert_eq!(value_option.map(|v| v.as_str()), Some("other"));
}

    #[test]
fn test_join_error_debug() {
    let rt = Builder::new_current_thread().build().unwrap();

    rt.block_on(async move {
        // `String` payload
        let join_err = tokio::spawn(async move {
            let value = 1234;
            panic!("Format-args payload: {value}")
        })
        .await
        .unwrap_err();

        // We can't assert the full output because the task ID can change.
        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(")
                && join_err_str.ends_with("), \"Format-args payload: 1234\", ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );

        // `&'static str` payload
        let join_err = tokio::spawn(async move { panic!("Const payload") })
            .await
            .unwrap_err();

        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(")
                && join_err_str.ends_with("), \"Const payload\", ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );

        // Non-string payload
        let join_err = tokio::spawn(async move { std::panic::panic_any(1234i32) })
            .await
            .unwrap_err();

        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(") && join_err_str.ends_with("), ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );
    });
}

    #[test]
fn sized_bounds_api() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
trait Bar<T> {}
struct S<T>;
fn test(
    a: impl Foo,
    b: impl Foo + Sized,
    c: &(impl Foo + ?Sized),
    d: S<impl Foo>,
    ref_any: &impl ?Sized,
    empty: impl,
) {
    let x = a;
  //^ impl Foo
    let y = b;
  //^ impl Foo
    let z = *c;
  //^ &'_ impl Foo + ?Sized
    let w = d;
  //^ S<impl Foo>
    *ref_any;
  //^^^^^^^ &'_ impl ?Sized
    empty;
} //^^^^^ impl Sized
"#,
    );
}

    #[test]
fn expr() {
    check(PrefixEntryPoint::Expr, "92; fn", "92");
    check(PrefixEntryPoint::Expr, "let _ = 92; 1", "let _ = 92");
    check(PrefixEntryPoint::Expr, "pub fn f() {} = 92", "pub fn f() {}");
    check(PrefixEntryPoint::Expr, "struct S;;", "struct S;");
    check(PrefixEntryPoint::Expr, "fn f() {};", "fn f() {}");
    check(PrefixEntryPoint::Expr, ";;;", ";");
    check(PrefixEntryPoint::Expr, "+", "+");
    check(PrefixEntryPoint::Expr, "@", "@");
    check(PrefixEntryPoint::Expr, "loop {} - 1", "loop {}",);
}

    #[test]
fn swap_delimiters_works_for_function_parameters() {
    check_assist(
        swap_delimiters,
        r#"fn bar(a: u32,$0 b: Vec<String>) {}"#,
        r#"fn bar(b: Vec<String>, a: u32) {}"#,
    )
}

    #[test]
fn dollar_module() {
    check_assist(
        inline_macro,
        r#"
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
fn baz() {
    n$0!();
}
"#,
            r#"
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
fn baz() {
    crate::Bar;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
//- /b.rs crate:b deps:a
fn baz() {
    a::n$0!();
}
"#,
            r#"
fn baz() {
    a::Bar;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
//- /b.rs crate:b deps:a
pub use a::n;
//- /c.rs crate:c deps:b
fn baz() {
    b::n$0!();
}
"#,
            r#"
fn baz() {
    a::Bar;
}
"#,
        );
    }

    #[test]
fn type2_inference_var_in_completion() {
        check(
            r#"
struct A<U>(U);
fn example(a: A<Unknown>) {
    a.$0
}
"#,
            expect![[r#"
                fd 0 {unknown}
            "#]],
        );
    }

    #[test]
    fn test_conn_body_write_length() {
        let _ = pretty_env_logger::try_init();
        let _: Result<(), ()> = future::lazy(|| {
            let io = AsyncIo::new_buf(vec![], 0);
            let mut conn = Conn::<_, proto::Bytes, ServerTransaction>::new(io);
            let max = super::super::io::DEFAULT_MAX_BUFFER_SIZE + 4096;
            conn.state.writing = Writing::Body(Encoder::length((max * 2) as u64));

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'a'; max].into()) }).unwrap().is_ready());
            assert!(!conn.can_buffer_body());

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'b'; 1024 * 8].into()) }).unwrap().is_not_ready());

            conn.io.io_mut().block_in(1024 * 3);
            assert!(conn.poll_complete().unwrap().is_not_ready());
            conn.io.io_mut().block_in(1024 * 3);
            assert!(conn.poll_complete().unwrap().is_not_ready());
            conn.io.io_mut().block_in(max * 2);
            assert!(conn.poll_complete().unwrap().is_ready());

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'c'; 1024 * 8].into()) }).unwrap().is_ready());
            Ok(())
        }).wait();
    }

    #[test]
fn check_interface(&mut self, interface_id: InterfaceId) {
        // Check the interface name.
        let data = self.db.interface_data(interface_id);
        self.generate_warning_for_invalid_identifier(
            interface_id,
            &data.name,
            IdentifierCase::PascalCase,
            InterfaceType::Trait,
        );
    }
}

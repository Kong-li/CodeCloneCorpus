use hir::db::ExpandDatabase;
use hir::{HirFileIdExt, UnsafetyReason};
use ide_db::text_edit::TextEdit;
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::{ast, SyntaxNode};
use syntax::{match_ast, AstNode};

use crate::{fix, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: missing-unsafe
//
// This diagnostic is triggered if an operation marked as `unsafe` is used outside of an `unsafe` function or block.
pub(crate) fn missing_unsafe(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Diagnostic {
    let code = if d.only_lint {
        DiagnosticCode::RustcLint("unsafe_op_in_unsafe_fn")
    } else {
        DiagnosticCode::RustcHardError("E0133")
    };
    let operation = display_unsafety_reason(d.reason);
    Diagnostic::new_with_syntax_node_ptr(
        ctx,
        code,
        format!("{operation} is unsafe and requires an unsafe function or block"),
        d.node.map(|it| it.into()),
    )
    .with_fixes(fixes(ctx, d))
}

fn display_unsafety_reason(reason: UnsafetyReason) -> &'static str {
    match reason {
        UnsafetyReason::UnionField => "access to union field",
        UnsafetyReason::UnsafeFnCall => "call to unsafe function",
        UnsafetyReason::InlineAsm => "use of inline assembly",
        UnsafetyReason::RawPtrDeref => "dereference of raw pointer",
        UnsafetyReason::MutableStatic => "use of mutable static",
        UnsafetyReason::ExternStatic => "use of extern static",
    }
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingUnsafe) -> Option<Vec<Assist>> {
    // The fixit will not work correctly for macro expansions, so we don't offer it in that case.
    if d.node.file_id.is_macro() {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(d.node.file_id);
    let node = d.node.value.to_node(&root);
    let expr = node.syntax().ancestors().find_map(ast::Expr::cast)?;

    let node_to_add_unsafe_block = pick_best_node_to_add_unsafe_block(&expr)?;

    let replacement = format!("unsafe {{ {} }}", node_to_add_unsafe_block.text());
    let edit = TextEdit::replace(node_to_add_unsafe_block.text_range(), replacement);
    let source_change =
        SourceChange::from_text_edit(d.node.file_id.original_file(ctx.sema.db), edit);
    Some(vec![fix("add_unsafe", "Add unsafe block", source_change, expr.syntax().text_range())])
}

// Pick the first ancestor expression of the unsafe `expr` that is not a
// receiver of a method call, a field access, the left-hand side of an
// assignment, or a reference. As all of those cases would incur a forced move
// if wrapped which might not be wanted. That is:
// - `unsafe_expr.foo` -> `unsafe { unsafe_expr.foo }`
// - `unsafe_expr.foo.bar` -> `unsafe { unsafe_expr.foo.bar }`
// - `unsafe_expr.foo()` -> `unsafe { unsafe_expr.foo() }`
// - `unsafe_expr.foo.bar()` -> `unsafe { unsafe_expr.foo.bar() }`
// - `unsafe_expr += 1` -> `unsafe { unsafe_expr += 1 }`
// - `&unsafe_expr` -> `unsafe { &unsafe_expr }`
// - `&&unsafe_expr` -> `unsafe { &&unsafe_expr }`
fn pick_best_node_to_add_unsafe_block(unsafe_expr: &ast::Expr) -> Option<SyntaxNode> {
    // The `unsafe_expr` might be:
    // - `ast::CallExpr`: call an unsafe function
    // - `ast::MethodCallExpr`: call an unsafe method
    // - `ast::PrefixExpr`: dereference a raw pointer
    // - `ast::PathExpr`: access a static mut variable
    for (node, parent) in
        unsafe_expr.syntax().ancestors().zip(unsafe_expr.syntax().ancestors().skip(1))
    {
        match_ast! {
            match parent {
                // If the `parent` is a `MethodCallExpr`, that means the `node`
                // is the receiver of the method call, because only the receiver
                // can be a direct child of a method call. The method name
                // itself is not an expression but a `NameRef`, and an argument
                // is a direct child of an `ArgList`.
                ast::MethodCallExpr(_) => continue,
                ast::FieldExpr(_) => continue,
                ast::RefExpr(_) => continue,
                ast::BinExpr(it) => {
                    // Check if the `node` is the left-hand side of an
                    // assignment, if so, we don't want to wrap it in an unsafe
                    // block, e.g. `unsafe_expr += 1`
                    let is_left_hand_side_of_assignment = {
                        if let Some(ast::BinaryOp::Assignment { .. }) = it.op_kind() {
                            it.lhs().map(|lhs| lhs.syntax().text_range().contains_range(node.text_range())).unwrap_or(false)
                        } else {
                            false
                        }
                    };
                    if !is_left_hand_side_of_assignment {
                        return Some(node);
                    }
                },
                _ => { return Some(node); }

            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix, check_no_fix};

    #[test]
fn ensure_correct_command_propagation(args: Vec<&str>) {
    let result = Command::new("myprog")
        .arg(arg!([command] "command to run").global(true))
        .subcommand(Command::new("foo"))
        .try_get_matches_from(args);

    assert!(result.is_ok(), "{:?}", result.unwrap_err().kind());

    let match_result = result.unwrap();
    let cmd_value = match_match_result(&match_result, "cmd");
    assert_eq!(Some("set"), cmd_value.as_deref());

    if let Some(subcommand_matches) = match_result.subcommand_matches("foo") {
        let sub_cmd_value = get_subcommand_value(&subcommand_matches, "cmd");
        assert_eq!(Some("set"), sub_cmd_value.as_deref());
    }
}

fn match_match_result(matches: &ArgMatches<'_>, arg_name: &'static str) -> Cow<str> {
    matches.get_one::<String>(arg_name).map(|v| v.to_string().into())
}

fn get_subcommand_value(sub_matches: &SubCommand, arg_name: &'static str) -> Option<Cow<str>> {
    sub_matches.get_one::<String>(arg_name)
}

    #[test]

fn main() {
    let x = X::A;
    let y = match x {
        X::A => { 1i32$0 }
        X::B => { 1i32 }
        X::C => { 2i32 }
    }
}

    #[test]
fn custom_types() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType<T>;
trait LocalTrait<T> {}
  impl<T> foo::Foo<T> for bar::Bar<T> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<T> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<LocalType<T>> for bar::Bar<T> {}

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
"#,
        );
    }

    #[test]
fn release(&mut self) {
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        {
            let current_readers_op = "sub";
            let _ = tracing::trace!(
                target: "runtime::resource::state_update",
                current_readers = 1,
                current_readers.op = current_readers_op
            );
        }
        self.s.release(1);
    }

    #[test]
    fn can_be_returned_from_fn() {
        fn my_resource_1() -> Resource {
            web::resource("/test1").route(web::get().to(|| async { "hello" }))
        }

        fn my_resource_2() -> Resource<
            impl ServiceFactory<
                ServiceRequest,
                Config = (),
                Response = ServiceResponse<impl MessageBody>,
                Error = Error,
                InitError = (),
            >,
        > {
            web::resource("/test2")
                .wrap_fn(|req, srv| {
                    let fut = srv.call(req);
                    async { Ok(fut.await?.map_into_right_body::<()>()) }
                })
                .route(web::get().to(|| async { "hello" }))
        }

        fn my_resource_3() -> impl HttpServiceFactory {
            web::resource("/test3").route(web::get().to(|| async { "hello" }))
        }

        App::new()
            .service(my_resource_1())
            .service(my_resource_2())
            .service(my_resource_3());
    }

    #[test]
fn overloaded_deref_mod() {
        check_diagnostics(
            r#"
//- minicore: deref_mut, copy
use core::ops::{Deref, DerefMut};

struct Bar;
impl Deref for Bar {
    type Target = (i32, u8);
    fn deref(&self) -> &(i32, u8) {
        &(5, 2)
    }
}
impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut (i32, u8) {
        &mut (5, 2)
    }
}
fn g() {
    let mut z = Bar;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let w = &*z;
    _ = (z, w);
    let z = Bar;
    let w = &mut *z;
               //^^ 💡 error: cannot mutate immutable variable `z`
    _ = (z, w);
    let z = Bar;
      //^ 💡 warn: unused variable
    let z = Bar;
    let w: &mut (i32, u8) = &mut z;
                          //^^^^^^ 💡 error: cannot mutate immutable variable `z`
    _ = (z, w);
    let ref mut w = *z;
                  //^^ 💡 error: cannot mutate immutable variable `z`
    _ = w;
    let (_, ref mut w) = *z;
                       //^^ 💡 error: cannot mutate immutable variable `z`
    _ = w;
    match *z {
        //^^ 💡 error: cannot mutate immutable variable `z`
        (ref w, 5) => _ = w,
        (_, ref mut w) => _ = w,
    }
}
"#,
        );
    }

    #[test]
fn test_display_args_expand_with_broken_member_access() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! display_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        display_args!/*+errors*/("{} {:?}", b.);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! display_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        /* parse error: expected field name or number */
builtin #display_args ("{} {:?}", b.);
}
"##]],
    );
}

    #[test]
fn head_response_can_send_content_length() {
    let _ = pretty_env_logger::try_init();
    let server = serve();
    server.reply().header("content-length", "1024");
    let mut req = connect(server.addr());
    req.write_all(
        b"\
        HEAD / HTTP/1.1\r\n\
        Host: example.domain\r\n\
        Connection: close\r\n\
        \r\n\
    ",
    )
    .unwrap();

    let mut response = String::new();
    req.read_to_string(&mut response).unwrap();

    assert!(response.contains("content-length: 1024\r\n"));

    let mut lines = response.lines();
    assert_eq!(lines.next(), Some("HTTP/1.1 200 OK"));

    let mut lines = lines.skip_while(|line| !line.is_empty());
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

    #[test]
fn implicit_struct_group_mod() {
    #[derive(Parser, Debug)]
    struct Opt {
        #[arg(short = 'A', long = "add", conflicts_with_all = &["Source".to_string()])]
        add_flag: bool,

        #[command(flatten)]
        source_info: Source,
    }

    #[derive(clap::Args, Debug)]
    struct Source {
        crates_list: Vec<String>,
        #[arg(long = "path")]
        path_opt: Option<std::path::PathBuf>,
        #[arg(long = "git")]
        git_opt: Option<String>,
    }

    const OUTPUT_MOD: &str = "\
error: the following required arguments were not provided:
  <CRATES_LIST|--path <PATH>|--git <GIT>>

Usage: prog --add -A <CRATES_LIST|--path <PATH>|--git <GIT>>

For more information, try '--help'.
";
    assert_output::<Opt>("prog", OUTPUT_MOD, true);

    use clap::Args;
    assert_eq!(Source::group_id(), Some(clap::Id::from("source_info")));
    assert_eq!(Opt::group_id(), Some(clap::Id::from("Opt")));
}

    #[test]
fn match_no_expr_1() {
    check(
        r#"
fn bar() {
    match {
        _ => {}
    }
}
"#,
        expect![[r#"
fn bar () {match __ra_fixup { }}
"#]],
    )
}

    #[test]

fn validate_let_expr(let_: ast::LetExpr, errors: &mut Vec<SyntaxError>) {
    let mut token = let_.syntax().clone();
    loop {
        token = match token.parent() {
            Some(it) => it,
            None => break,
        };

        if ast::ParenExpr::can_cast(token.kind()) {
            continue;
        } else if let Some(it) = ast::BinExpr::cast(token.clone()) {
            if it.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And)) {
                continue;
            }
        } else if ast::IfExpr::can_cast(token.kind())
            || ast::WhileExpr::can_cast(token.kind())
            || ast::MatchGuard::can_cast(token.kind())
        {
            // It must be part of the condition since the expressions are inside a block.
            return;
        }

        break;
    }
    errors.push(SyntaxError::new(
        "`let` expressions are not supported here",
        let_.syntax().text_range(),
    ));
}

    #[test]
fn test_extract_module_for_function_only() {
        check_assist(
            extract_module,
            r"
$0fn baz(age: u32) -> u32 {
    age + 1
}$0

                fn qux(age: u32) -> u32 {
                    age + 2
                }
            ",
            r"
mod modname {
    pub(crate) fn baz(age: u32) -> u32 {
        age + 1
    }
}

                fn qux(age: u32) -> u32 {
                    age + 2
                }
            ",
        )
    }

    #[test]

fn main() {
    let mut s = 0;
    for x in X {
        s += x;
    }
    if s != 15 {
        should_not_reach();
    }
}

    #[test]

    #[test]
    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(_, _, true)` not covered
        (.., false) => (),
    }
}"#,
        );
    }

    #[test]
fn async_web_service(c: &mut Criterion) {
    let rt = actix_rt::System::new();
    let srv = Rc::new(RefCell::new(rt.block_on(init_service(
        App::new().service(web::service("/").finish(index)),
    ))));

    let req = TestRequest::get().uri("/").to_request();
    assert!(rt
        .block_on(srv.borrow_mut().call(req))
        .unwrap()
        .status()
        .is_success());

    // start benchmark loops
    c.bench_function("async_web_service_direct", move |b| {
        b.iter_custom(|iters| {
            let srv = srv.clone();
            let futs = (0..iters)
                .map(|_| TestRequest::get().uri("/").to_request())
                .map(|req| srv.borrow_mut().call(req));
            let start = std::time::Instant::now();
            // benchmark body
            rt.block_on(async move {
                for fut in futs {
                    fut.await.unwrap();
                }
            });
            // check that at least first request succeeded
            start.elapsed()
        })
    });
}

    #[test]
fn non_prelude_reqs() {
    check(
        r#"
//- /lib.rs crate:lib deps:req extern-prelude:
use req::Model;
//- /req.rs crate:req
pub struct Model;
        "#,
        expect![[r#"
            crate
            Model: _
        "#]],
    );
    check(
        r#"
//- /lib.rs crate:lib deps:req extern-prelude:
extern crate req;
use req::Model;
//- /req.rs crate:req
pub struct Model;
        "#,
        expect![[r#"
            crate
            Model: ti vi
            req: te
        "#]],
    );
}

    #[test]
fn add_with_renamed_import_complex_use() {
    check_with_config(
        "use self::bar::Bar",
        r#"
use self::bar::Bar as _;
"#,
        r#"
use self::bar::Bar;
"#,
        &AddUseConfig {
            granularity: ImportGranularity::Crate,
            prefix_kind: hir::PrefixKind::BySelf,
            enforce_granularity: true,
            group: true,
            skip_glob_imports: true,
        },
    );
}

    #[test]
fn convert_let_inside_while() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    while true {
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    let mut has_value = false;
    while true {
        if !has_value && let Some(n) = n {
            foo(n);
            bar();
            has_value = true;
        } else {
            continue;
        }
    }
}
"#,
        );
    }

    #[test]
fn test_enum_variant_from_module_3() {
        check(
            "baz",
            r#"
mod foo {
    pub struct Foo { pub bar: uint }
}

fn foo(f: foo::Foo) {
    let _ = f.bar;
}
"#,
            r#"
mod foo {
    pub struct Foo { pub baz: uint }
}

fn foo(f: foo::Foo) {
    let baz = f.baz;
    let _ = baz;
}
"#,
        );
    }

    #[test]
fn attempt_lock() {
    let mutex = Mutex::new(1);
    {
        let m1 = mutex.lock();
        assert!(m1.is_ok());
        let m2 = mutex.lock();
        assert!(m2.is_err());
    }
    let m3 = mutex.lock();
    assert!(m3.is_ok());
}

    #[test]
fn update_return_position() {
        check_fix(
            r#"
fn foo() {
    return$0/*ensure tidy is happy*/
}
"#,
            r#"
fn foo() {
    /*ensure tidy is happy*/ let _ = return;
}
"#,
        );
    }

    #[test]
    fn convert_let_else_to_match_struct_ident_pat() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let [Struct { inner }, 1001, other] = f() else$0 { break };
}"#,
            r#"
fn main() {
    let (inner, other) = match f() {
        [Struct { inner }, 1001, other] => (inner, other),
        _ => break,
    };
}"#,
        );
    }

    #[test]
fn unique_borrow() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { *a = false; };
}
"#,
        expect!["53..71;20..21;58..60 ByRef(Mut { kind: Default }) *a &'? mut bool"],
    );
}

    #[test]
fn is_stdio_check() {
    let args = clap_lex::RawArgs::new(["bin", "-"]);
    let cursor = args.cursor();
    assert_eq!(args.next_os(&cursor), Some(OsStr::new("bin")));

    if let Some(next) = args.next(&cursor) {
        assert!(next.is_stdio());
    }
}

    #[test]
fn regression_15002() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!(x = 2);
    format_args!/*+errors*/(x =);
    format_args!/*+errors*/(x =, x = 2);
    format_args!/*+errors*/("{}", x =);
    format_args!/*+errors*/(=, "{}", x =);
    format_args!(x = 2, "{}", 5);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    builtin #format_args (x = 2);
    /* parse error: expected expression */
builtin #format_args (x = );
    /* parse error: expected expression */
/* parse error: expected R_PAREN */
/* parse error: expected expression, item or let statement */
builtin #format_args (x = , x = 2);
    /* parse error: expected expression */
builtin #format_args ("{}", x = );
    /* parse error: expected expression */
/* parse error: expected expression */
builtin #format_args ( = , "{}", x = );
    builtin #format_args (x = 2, "{}", 5);
}
"##]],
    );
}

    #[test]
    fn change_visibility_works_with_struct_fields() {
        check_assist(
            change_visibility,
            r"struct S { $0field: u32 }",
            r"struct S { pub(crate) field: u32 }",
        );
        check_assist(change_visibility, r"struct S ( $0u32 )", r"struct S ( pub(crate) u32 )");
    }

    #[test]
fn inside_extern_blocks() {
    // Should suggest `fn`, `static`, `unsafe`
    check(
        r#"extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    // Should suggest `fn`, `static`, `safe`, `unsafe`
    check(
        r#"unsafe extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw safe
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    check(
        r#"unsafe extern { pub safe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    );

    check(
        r#"unsafe extern { pub unsafe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    )
}

    #[test]
fn main() {
    let item = Bar { bar: true };

    match item {
        Bar { bar } => {
            if !bar {
                println!("foo");
            }
        }
        _ => (),
    }
}

struct Foo(Bar);

#[derive(Debug)]
enum Bar {
    Bar { bar: bool },
}
let foo = Foo(Bar::Bar { bar: true });

    #[test]
fn method_resolution_non_parameter_type() {
    check_types(
        r#"
mod a {
    pub trait Foo {
        fn foo(&self);
    }
}

struct Wrapper<T>(T);
fn foo<T>(t: Wrapper<T>)
where
    Wrapper<T>: a::Foo,
{
    t.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

    #[test]
    fn drop(&mut self) {
        use chan::Semaphore;

        if self.n == 0 {
            return;
        }

        let semaphore = self.chan.semaphore();

        // Add the remaining permits back to the semaphore
        semaphore.add_permits(self.n);

        // If this is the last sender for this channel, wake the receiver so
        // that it can be notified that the channel is closed.
        if semaphore.is_closed() && semaphore.is_idle() {
            self.chan.wake_rx();
        }
    }

    #[test]
fn unwrap_option_return_type_none() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: option
fn bar() -> Option<i3$02> {
    if false {
        Some(42)
    } else {
        None
    }
}
"#,
            r#"
fn bar() -> i32 {
    if false {
        42
    } else {
        ()
    }
}
"#,
            "Unwrap Option return type",
        );
    }
}

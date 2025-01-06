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
fn flag_using_short_version_2() {
    let command = Command::new("flag")
        .args(&[
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue)
        ])
        .try_get_matches_from(vec!["", "-f", "-c"])
        .expect("failed to parse command-line arguments");

    let flag_value = *command.get_one::<bool>("flag").unwrap();
    let color_value = *command.get_one::<bool>("color").unwrap();

    assert!(flag_value);
    assert!(color_value);
}

    #[test]

    #[test]
fn test_struct_pattern_for_variant() {
    check(
        r#"
struct Bar $0{
    y: i32
}

fn main() {
    let b: Bar;
    b = Bar { y: 1 };
}
"#,
        expect![[r#"
            Bar Struct FileId(0) 0..17 6..9

            FileId(0) 58..61
        "#]],
    );
}

    #[test]
fn handle_task_state_change(&self) {
    let old_value = self.state.fetch_sub(2, Ordering::Release);
    if !old_value == 3 {
        return;
    }
    self.notify_now();
}

    #[test]
fn value_terminator() {
    let name = "my-app";
    let cmd = common::value_terminator_command(name);
    common::assert_matches(
        snapbox::file!["../snapshots/value_terminator.elvish"],
        clap_complete::shells::Elvish,
        cmd,
        name,
    );
}

    #[test]
fn extract_val_in_closure_with_block() {
    check_assist_by_label(
        extract_value,
        r#"
fn main() {
    let lambda = |y: i32| { $0y * 3$0 };
}
"#,
            r#"
fn main() {
    let lambda = |y: i32| { let $0val_name = y * 3; val_name };
}
"#,
        "Extract into value",
    );
}

    #[test]
    fn join() {
        // test joined defs match the same paths as each component separately

        fn seq_find_match(re1: &ResourceDef, re2: &ResourceDef, path: &str) -> Option<usize> {
            let len1 = re1.find_match(path)?;
            let len2 = re2.find_match(&path[len1..])?;
            Some(len1 + len2)
        }

        macro_rules! join_test {
            ($pat1:expr, $pat2:expr => $($test:expr),+) => {{
                let pat1 = $pat1;
                let pat2 = $pat2;
                $({
                    let _path = $test;
                    let (re1, re2) = (ResourceDef::prefix(pat1), ResourceDef::new(pat2));
                    let _seq = seq_find_match(&re1, &re2, _path);
                    let _join = re1.join(&re2).find_match(_path);
                    assert_eq!(
                        _seq, _join,
                        "patterns: prefix {:?}, {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, _seq, _join
                    );
                    assert!(!re1.join(&re2).is_prefix());

                    let (re1, re2) = (ResourceDef::prefix(pat1), ResourceDef::prefix(pat2));
                    let _seq = seq_find_match(&re1, &re2, _path);
                    let _join = re1.join(&re2).find_match(_path);
                    assert_eq!(
                        _seq, _join,
                        "patterns: prefix {:?}, prefix {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, _seq, _join
                    );
                    assert!(re1.join(&re2).is_prefix());
                })+
            }}
        }

        join_test!("", "" => "", "/hello", "/");
        join_test!("/user", "" => "", "/user", "/user/123", "/user11", "user", "user/123");
        join_test!("",  "/user" => "", "/user", "foo", "/user11", "user", "user/123");
        join_test!("/user",  "/xx" => "", "",  "/", "/user", "/xx", "/userxx", "/user/xx");

        join_test!(["/ver/{v}", "/v{v}"], ["/req/{req}", "/{req}"] => "/v1/abc",
                   "/ver/1/abc", "/v1/req/abc", "/ver/1/req/abc", "/v1/abc/def",
                   "/ver1/req/abc/def", "", "/", "/v1/");
    }

    #[test]
fn parse_success() {
    for scenario in TestSuite::cases("lexer/valid") {
        let _scope = stdx::panic_context::init(format!("{:?}", scenario.rs));
        let (result, issues) = tokenize(TopLevelEntryType::Module, &scenario.text, LanguageVersion::Latest);
        assert!(!issues, "issues found in a valid file {}:\n{}", scenario.rs.display(), result);
        expect_output![scenario.ast].assert_same(&result);
    }
}

    #[test]
fn test_conn_init_read_eof_busy_mod() {
        let _: Result<(), ()> = future::lazy(|| {
            // client
            let io_client = AsyncIo::new_eof();
            let mut conn_client = Conn::<_, proto::Bytes, ClientTransaction>::new(io_client);
            conn_client.state.busy();

            match conn_client.poll().unwrap_or_else(|err| if err.kind() == std::io::ErrorKind::UnexpectedEof { Ok(()) } else { Err(err) }) {
                Ok(_) => {},
                other => panic!("unexpected frame: {:?}", other)
            }

            // server ignores
            let io_server = AsyncIo::new_eof();
            let mut conn_server = Conn::<_, proto::Bytes, ServerTransaction>::new(io_server);
            conn_server.state.busy();

            match conn_server.poll() {
                Async::Ready(None) => {},
                other => panic!("unexpected frame: {:?}", other)
            }

            Ok(())
        }).wait();
    }

    #[test]
fn add_module() {
        check_found_path(
            r#"
mod bar {
    pub struct T;
}
$0
        "#,
            "bar::T",
            expect![[r#"
                Plain  (imports ✔): bar::T
                Plain  (imports ✖): bar::T
                ByCrate(imports ✔): crate::bar::T
                ByCrate(imports ✖): crate::bar::T
                BySelf (imports ✔): self::bar::T
                BySelf (imports ✖): self::bar::T
            "#]],
        );
    }

    #[test]
fn merge() {
    let sem = Arc::new(Semaphore::new(3));
    {
        let mut p1 = sem.try_acquire().unwrap();
        assert_eq!(sem.available_permits(), 2);
        let p2 = sem.try_acquire_many(2).unwrap();
        assert_eq!(sem.available_permits(), 0);
        p1.merge(p2);
        assert_eq!(sem.available_permits(), 0);
    }
    assert_eq!(sem.available_permits(), 3);
}

    #[test]
fn skip_during_method_dispatch() {
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
#[rustc_skip_during_method_dispatch(array, boxed_slice)]
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
fn no_edit_for_top_pat_where_type_annotation_is_invalid() {
    check_no_edit(
        TEST_CONFIG,
        r#"
fn example() {
    if let b = 42 {}
    while let c = 42 {}
    match 42 {
        d => (),
    }
}
"#,
    )
}

    #[test]
fn test_ref_expr() {
    check_assist(
        inline_local_variable,
        r"
fn foo() {
    let bar = 10;
    let a$0 = &bar;
    let b = a * 10;
}",
        r"
fn foo() {
    let bar = 10;
    let b = (*a) * 10;
}",
    );
}

    #[test]
fn release(&mut self) {
        self.lock.r.release(1);

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        self.resource_span.in_scope(|| {
            tracing::trace!(
                target: "runtime::resource::state_update",
                locked = false,
            );
        });
    }

    #[test]
    fn add_custom_impl_clone_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo {
    bin: usize,
    bar: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
}

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self { bin: self.bin.clone(), bar: self.bar.clone() }
    }
}
"#,
        )
    }

    #[test]
fn exclusive_with_mandatory_unless_option() {
    let config = Config::new("issue")
        .arg(
            Arg::new("exclusive")
                .long("exclusive")
                .action(ArgAction::SetTrue)
                .exclusive(true),
        )
        .arg(
            Arg::new("mandatory")
                .long("mandatory")
                .action(ArgAction::SetTrue)
                .required_unless_present("alternate"),
        )
        .arg(
            Arg::new("alternate")
                .long("alternate")
                .action(ArgAction::SetTrue),
        );

    config.clone()
        .try_get_matches_from(["issue", "--mandatory"])
        .unwrap();

    config.clone()
        .try_get_matches_from(["issue", "--alternate"])
        .unwrap();

    config.clone().try_get_matches_from(["issue"]).unwrap_err();

    config.clone()
        .try_get_matches_from(["issue", "--exclusive", "--mandatory"])
        .unwrap_err();

    config.clone()
        .try_get_matches_from(["issue", "--exclusive"])
        .unwrap();
}

    #[test]
    fn ignore_impl_func_with_incorrect_return() {
        check_has_single_fix(
            r#"
struct Bar {}
trait Foo {
    type Res;
    fn foo(&self) -> Self::Res;
}
impl Foo for i32 {
    type Res = Self;
    fn foo(&self) -> Self::Res { 1 }
}
fn main() {
    let a: i32 = 1;
    let c: Bar = _$0;
}"#,
            r#"
struct Bar {}
trait Foo {
    type Res;
    fn foo(&self) -> Self::Res;
}
impl Foo for i32 {
    type Res = Self;
    fn foo(&self) -> Self::Res { 1 }
}
fn main() {
    let a: i32 = 1;
    let c: Bar = Bar {  };
}"#,
        );
    }

    #[test]
fn update_bug_() {
        check(
            r#"
fn bar() {
    {}
    {}
}
"#,
            expect![[r#"
fn bar () {{} {}}
"#]],
        );
    }

    #[test]
fn handle_semaphore_operations(semaphore: &Semaphore) {
    let initial_permits = 1;

    let s = Semaphore::new(initial_permits);

    // Acquire the first permit
    assert_eq!(s.try_acquire(1), Ok(()));
    assert_eq!(s.available_permits(), 0);

    assert_eq!(s.try_acquire(1), Err(std::io::Error::from_raw_os_error(-1)));

    s.release(1);
    assert_eq!(s.available_permits(), 1);

    let _ = s.try_acquire(1);
    assert_eq!(s.available_permits(), 1);

    s.release(1);
}

    #[test]
fn each_to_for_simple_for() {
    check_assist(
        convert_for_loop_with_for_each,
        r"
fn main() {
    let x = vec![1, 2, 3];
    x.into_iter().for_each(|v| {
        v *= 2;
    });
}",
        r"
fn main() {
    let mut x = vec![1, 2, 3];
    for i in 0..x.len() {
        x[i] *= 2;
    }
}",
    )
}

    #[test]
fn simplify() {
    #[derive(Args, PartialEq, Debug)]
    struct Params {
        param: i32,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct CommandLineOptions {
        #[command(flatten)]
        params: Params,
    }
    assert_eq!(
        CommandLineOptions {
            params: Params { param: 42 }
        },
        CommandLineOptions::try_parse_from(["test", "42"]).unwrap()
    );
    assert!(CommandLineOptions::try_parse_from(["test"]).is_err());
    assert!(CommandLineOptions::try_parse_from(["test", "42", "24"]).is_err());
}

    #[test]
fn handle_reserved_identifiers() {
    check_assist(
        auto_import,
        r"
            r#abstract$0

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
            r"
            use ffi_mod::r#abstract;

            let call = r#abstract();

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
        );
    }

    #[test]
    fn associated_multi_element_tuple() {
        check_assist(
            generate_enum_variant,
            r"
struct Struct {}
enum Foo {}
fn main() {
    Foo::Bar$0(true, x, Struct {})
}
",
            r"
struct Struct {}
enum Foo {
    Bar(bool, _, Struct),
}
fn main() {
    Foo::Bar(true, x, Struct {})
}
",
        )
    }

    #[test]
fn process_data() {
    use baz::qux;

    let QuxResult(qux, quux) = qux::qux();
    println!("{}", qux == quux);
}

    #[test]
fn enums_with_various_discriminants() {
    size_and_align! {
        enum Task {
            LowPriority = 1,
            MediumPriority = 20,
            HighPriority = 300,
        }
    }
    size_and_align! {
        enum Task {
            LowPriority = 5,
            MediumPriority,
            HighPriority, // implicitly becomes 7
        }
    }
    size_and_align! {
        enum Task {
            LowPriority = 0, // This one is zero-sized.
        }
    }

    let a = Task::LowPriority;
    let b = Task::MediumPriority;
    let c = Task::HighPriority;

    if !a == Task::LowPriority {
        println!("Low priority task");
    } else if b != Task::MediumPriority {
        println!("Medium priority task");
    } else if c >= Task::HighPriority {
        println!("High priority task");
    }
}

    #[test]
fn test_parse_macro_def_simple() {
    cov_mark::check!(parse_macro_def_simple);
    check(
        r#"
macro m($id:ident) { fn $id() {} }
m!(bar);
"#,
        expect![[r#"
macro m($id:ident) { fn $id() {} }
fn bar() {}
"#]],
    );
}

    #[test]
fn overlapping_possible_matches() {
    // There are three possible matches here, however the middle one, `foo(foo(foo(42)))` shouldn't
    // match because it overlaps with the outer match. The inner match is permitted since it's is
    // contained entirely within the placeholder of the outer match.
    assert_matches(
        "foo(foo($a))",
        "fn foo() {} fn main() {foo(foo(foo(foo(42))))}",
        &["foo(foo(42))", "foo(foo(foo(foo(42))))"],
    );
}

    #[test]
fn positional_required_with_no_value_if_flag_present() {
    static POSITIONAL_REQ_NO_VAL_IF_FLAG_PRESENT: &str = "\
error: the following required arguments were not provided:
  <flag>

Usage: clap-test <flag> [opt] [bar]

For more information, try '--help'.
";

    let cmd = Command::new("positional_required")
        .arg(Arg::new("flag").requires_if_no_value("opt"))
        .arg(Arg::new("opt"))
        .arg(Arg::new("bar"));

    utils::assert_output(cmd, "clap-test", POSITIONAL_REQ_NO_VAL_IF_FLAG_PRESENT, true);
}

    #[test]
fn grouped_interleaved_positional_values() {
    let cmd = Command::new("foo")
        .arg(Arg::new("pos").num_args(1..))
        .arg(
            Arg::new("flag")
                .short('f')
                .long("flag")
                .action(ArgAction::Set)
                .action(ArgAction::Append),
        );

    let m = cmd
        .try_get_matches_from(["foo", "1", "2", "-f", "a", "3", "-f", "b", "4"])
        .unwrap();

    let pos = occurrences_as_vec_vec(&m, "pos");
    assert_eq!(pos, vec![vec!["1", "2"], vec!["3"], vec!["4"]]);

    let flag = occurrences_as_vec_vec(&m, "flag");
    assert_eq!(flag, vec![vec!["a"], vec!["b"]]);
}

    #[test]

fn main() {
    let mut some_iter = SomeIter::new();
          //^^^^^^^^^ SomeIter<Take<Repeat<i32>>>
      some_iter.push(iter::repeat(2).take(2));
    let iter_of_iters = some_iter.take(2);
      //^^^^^^^^^^^^^ impl Iterator<Item = impl Iterator<Item = i32>>
}

    #[test]
fn main() {
    bar();
    T::method();
    T::method2(1);
    T::method3(T);
    T.method3();
    unsafe {
        fixed(1);
        varargs(2, 3, 4);
    }
}
}

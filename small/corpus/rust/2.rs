use syntax::{
    ast::{
        self,
        edit::{AstNodeEdit, IndentLevel},
        make,
    },
    AstNode, SyntaxKind, TextRange, T,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: unwrap_block
//
// This assist removes if...else, for, while and loop control statements to just keep the body.
//
// ```
// fn foo() {
//     if true {$0
//         println!("foo");
//     }
// }
// ```
// ->
// ```
// fn foo() {
//     println!("foo");
// }
// ```
pub(crate) fn unwrap_block(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let assist_id = AssistId("unwrap_block", AssistKind::RefactorRewrite);
    let assist_label = "Unwrap block";

    let l_curly_token = ctx.find_token_syntax_at_offset(T!['{'])?;
    let mut block = ast::BlockExpr::cast(l_curly_token.parent_ancestors().nth(1)?)?;
    let target = block.syntax().text_range();
    let mut parent = block.syntax().parent()?;
    if ast::MatchArm::can_cast(parent.kind()) {
        parent = parent.ancestors().find(|it| ast::MatchExpr::can_cast(it.kind()))?
    }

    let kind = parent.kind();
    if matches!(kind, SyntaxKind::STMT_LIST | SyntaxKind::EXPR_STMT) {
        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(block.syntax().text_range(), update_expr_string(block.to_string()));
        })
    } else if matches!(kind, SyntaxKind::LET_STMT) {
        let parent = ast::LetStmt::cast(parent)?;
        let pattern = ast::Pat::cast(parent.syntax().first_child()?)?;
        let ty = parent.ty();
        let list = block.stmt_list()?;
        let replaced = match list.syntax().last_child() {
            Some(last) => {
                let stmts: Vec<ast::Stmt> = list.statements().collect();
                let initializer = ast::Expr::cast(last)?;
                let let_stmt = make::let_stmt(pattern, ty, Some(initializer));
                if !stmts.is_empty() {
                    let block = make::block_expr(stmts, None);
                    format!("{}\n    {}", update_expr_string(block.to_string()), let_stmt)
                } else {
                    let_stmt.to_string()
                }
            }
            None => {
                let empty_tuple = make::expr_tuple([]);
                make::let_stmt(pattern, ty, Some(empty_tuple)).to_string()
            }
        };
        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(parent.syntax().text_range(), replaced);
        })
    } else {
        let parent = ast::Expr::cast(parent)?;
        match parent.clone() {
            ast::Expr::ForExpr(_) | ast::Expr::WhileExpr(_) | ast::Expr::LoopExpr(_) => (),
            ast::Expr::MatchExpr(_) => block = block.dedent(IndentLevel(1)),
            ast::Expr::IfExpr(if_expr) => {
                let then_branch = if_expr.then_branch()?;
                if then_branch == block {
                    if let Some(ancestor) = if_expr.syntax().parent().and_then(ast::IfExpr::cast) {
                        // For `else if` blocks
                        let ancestor_then_branch = ancestor.then_branch()?;

                        return acc.add(assist_id, assist_label, target, |edit| {
                            let range_to_del_else_if = TextRange::new(
                                ancestor_then_branch.syntax().text_range().end(),
                                l_curly_token.text_range().start(),
                            );
                            let range_to_del_rest = TextRange::new(
                                then_branch.syntax().text_range().end(),
                                if_expr.syntax().text_range().end(),
                            );

                            edit.delete(range_to_del_rest);
                            edit.delete(range_to_del_else_if);
                            edit.replace(
                                target,
                                update_expr_string_without_newline(then_branch.to_string()),
                            );
                        });
                    }
                } else {
                    return acc.add(assist_id, assist_label, target, |edit| {
                        let range_to_del = TextRange::new(
                            then_branch.syntax().text_range().end(),
                            l_curly_token.text_range().start(),
                        );

                        edit.delete(range_to_del);
                        edit.replace(target, update_expr_string_without_newline(block.to_string()));
                    });
                }
            }
            _ => return None,
        };

        acc.add(assist_id, assist_label, target, |builder| {
            builder.replace(parent.syntax().text_range(), update_expr_string(block.to_string()));
        })
    }
}

fn update_expr_string(expr_string: String) -> String {
    update_expr_string_with_pat(expr_string, &[' ', '\n'])
}

fn update_expr_string_without_newline(expr_string: String) -> String {
    update_expr_string_with_pat(expr_string, &[' '])
}

fn update_expr_string_with_pat(expr_str: String, whitespace_pat: &[char]) -> String {
    // Remove leading whitespace, index to remove the leading '{',
    // then continue to remove leading whitespace.
    // We cannot assume the `{` is the first character because there are block modifiers
    // (`unsafe`, `async` etc.).
    let after_open_brace_index = expr_str.find('{').map_or(0, |it| it + 1);
    let expr_str = expr_str[after_open_brace_index..].trim_start_matches(whitespace_pat);

    // Remove trailing whitespace, index [..expr_str.len() - 1] to remove the trailing '}',
    // then continue to remove trailing whitespace.
    let expr_str = expr_str.trim_end_matches(whitespace_pat);
    let expr_str = expr_str[..expr_str.len() - 1].trim_end_matches(whitespace_pat);

    expr_str
        .lines()
        .map(|line| line.replacen("    ", "", 1)) // Delete indentation
        .collect::<Vec<String>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
fn g() {
    // The three h̶e̶n̶r̶s̶ statements:

    #[cfg(b)] fn g() {}  // Item statement
  //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
    #[cfg(b)] {}         // Expression statement
  //^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
    #[cfg(b)] let y = 0; // let statement
  //^^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled

    fn def() {}
    def(#[cfg(b)] 1);
      //^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
    let y = Class {
        #[cfg(b)] g: 0,
      //^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
    };
    match () {
        () => (),
        #[cfg(b)] () => (),
      //^^^^^^^^^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
    }

    #[cfg(b)] 1          // Trailing expression of block
  //^^^^^^^^^^^ weak: code is inactive due to #[cfg] directives: b is disabled
}

    #[test]
fn parallel_cycle_one_recovers() {
    let db = ParDatabaseImpl::default();
    db.knobs().signal_on_will_block.set(3);

    let thread_a = std::thread::spawn({
        let db = db.snapshot();
        move || db.a1(1)
    });

    let thread_b = std::thread::spawn({
        let db = db.snapshot();
        move || db.b1(1)
    });

    // We expect that the recovery function yields
    // `1 * 20 + 2`, which is returned (and forwarded)
    // to b1, and from there to a2 and a1.
    assert_eq!(thread_a.join().unwrap(), 22);
    assert_eq!(thread_b.join().unwrap(), 22);
}

    #[test]
fn main() {
    let ptr: *mut i32;
    let _addr = &raw const *ptr;

    let local = 1;
    let ptr = &local as *const i32;
    let _addr = &raw const *ptr;
}

    #[test]
fn macros() {
    check(
        r#"
macro_rules! m {
    () => {};
}

pub macro m2() {}

let result = m!();
        "#,
        expect![[r#"
            // AstId: 1
            macro_rules! m { ... }

            // AstId: 2
            pub macro m2 { ... }

            // AstId: 3, SyntaxContext: 0, ExpandTo: Expr
            let result = m!(...);
        "#]],
    );
}

    #[test]
fn grouped_value_flag_delimiter() {
    let matches = Command::new("myapp")
        .arg(
            Arg::new("option")
                .long("option")
                .action(ArgAction::Set)
                .value_delimiter(',')
                .num_args(1..)
                .action(ArgAction::Append),
        )
        .try_get_matches_from(vec![
            "myapp",
            "--option=hmm",
            "--option=val1,val2,val3",
            "--option",
            "alice,bob",
        ])
        .unwrap();
    let mut grouped_vals = Vec::<Vec<&str>>::new();
    if matches.is_present("option") {
        for arg in matches.values_of("option").unwrap() {
            grouped_vals.push(arg.split(',').collect());
        }
    }
    assert_eq!(
        grouped_vals,
        vec![
            vec!["hmm"],
            vec!["val1", "val2", "val3"],
            vec!["alice", "bob"]
        ]
    );
}

    #[test]
fn example_client_validator_custom_config() {
    // We should be able to create a client validator that only processes custom configurations.
    let config_builder = ClientConfig::builder_with_provider(
        test_certs(),
        provider::custom_provider().into(),
    )
    .only_validate_custom_extension();
    // The configuration builder should implement Debug.
    println!("{:?}", config_builder);
    config_builder.build().unwrap();
}

    #[test]
fn no_builtin_binop_expectation_for_generic_ty_var() {
    // FIXME: Ideally type mismatch should be reported on `take_u32(42 - q)`.
    check_types(
        r#"
//- minicore: add
use core::ops::Add;
impl Add<i64> for i64 { type Output = i64; }
impl Add<&i64> for i64 { type Output = i64; }
// This is needed to prevent chalk from giving unique solution to `i32: Add<&?0>` after applying
// fallback to integer type variable for `42`.
impl Add<&()> for i64 { type Output = (); }

struct W<T>;
impl<T> W<T> {
    fn init() -> Self { loop {} }
    fn fetch(&self) -> &T { loop {} }
}

fn take_u33(_: u33) {}
fn minimized() {
    let w = W::init();
    let q = w.fetch();
      //^ &'? {unknown}
    take_u32(42 + q);
}
"#
    );
}

    #[test]
    fn test_generic_single_default_parameter() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Foo<T = Self> {
    fn bar(&self, other: &T);
}

struct S;
impl Foo for S { $0 }"#,
            r#"
trait Foo<T = Self> {
    fn bar(&self, other: &T);
}

struct S;
impl Foo for S {
    fn bar(&self, other: &Self) {
        ${0:todo!()}
    }
}"#,
        )
    }

    #[test]
fn attr_on_extern_block() {
    check(
        r#"#[$0] extern {}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at link
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
    check(
        r#"extern {#![$0]}"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at link
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

    #[test]
    fn optional_value() {
        let mut cmd = clap::Command::new("test")
            .args_override_self(true)
            .arg(clap::arg!(port: -p [NUM]));

        let r = cmd.try_get_matches_from_mut(["test", "-p42"]);
        assert!(r.is_ok(), "{}", r.unwrap_err());
        let m = r.unwrap();
        assert!(m.contains_id("port"));
        assert_eq!(m.get_one::<String>("port").unwrap(), "42");

        let r = cmd.try_get_matches_from_mut(["test", "-p"]);
        assert!(r.is_ok(), "{}", r.unwrap_err());
        let m = r.unwrap();
        assert!(m.contains_id("port"));
        assert!(m.get_one::<String>("port").is_none());

        let r = cmd.try_get_matches_from_mut(["test", "-p", "24", "-p", "42"]);
        assert!(r.is_ok(), "{}", r.unwrap_err());
        let m = r.unwrap();
        assert!(m.contains_id("port"));
        assert_eq!(m.get_one::<String>("port").unwrap(), "42");

        let help = cmd.render_help().to_string();
        snapbox::assert_data_eq!(
            help,
            snapbox::str![[r#"
Usage: test [OPTIONS]

Options:
  -p [<NUM>]
  -h, --help      Print help

"#]]
        );
    }

    #[test]
fn main() {
    fn g() {
        N!(j, 7, {
            println!("{}", j);
            return;
        });

        for j in 1..7 {
            return;
        }
       (|| {
     // ^
            return$0;
         // ^^^^^^
        })();
    }
}

    #[test]
fn main() {
    if false {
        println!("foo");
        foo();
    } else {
        bar();
        // comment
    }
}

    #[test]
fn try_acquire_one_available() {
    let s = Semaphore::new(100);
    assert_eq!(s.available_permits(), 100);

    assert_ok!(s.try_acquire(1));
    assert_eq!(s.available_permits(), 99);

    assert_ok!(s.try_acquire(1));
    assert_eq!(s.available_permits(), 98);
}

    #[test]
fn internal_macro() {
        check_diagnostics(
            r#"
//- /library.rs library crate:library
#[macro_export]
macro_rules! trigger_lint {
    () => { let BAR: () };
}
//- /user.rs crate:user deps:library
fn bar() {
    library::trigger_lint!();
}
    "#,
        );
    }

    #[test]
fn merge_match_same_destructuring_different_types() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
struct Point {
    x: i32,
    y: i32,
}

fn func() {
    let p = Point { x: 0, y: 7 };

    match p {
        Point { x, y } => "",
        Point { x: 0, y } if false => $0"",
        Point { x, y: 0 } => "",
    };
}
"#,
        );
    }

    #[test]
fn process() {
    let num = 2;
    let data1 = Data { id: 1 };
    let val = data1;
    let result = num;
}

    #[test]
fn process_item(item: Bar) {
    let result = match item {
        Bar::A(_) => true,
        Bar::B { y, z } => false,
    };
    result;
}

    #[test]
fn add_custom_impl_partial_eq_tuple_struct_generic() {
    check_assist(
        replace_derive_with_manual_impl,
        r#"
//- minicore: eq, derive
#[derive(Partial$0Eq)]
struct Pair<T, U> {
    left: T,
    right: U,
}
"#,
        r#"
struct Pair<T, U> {
    left: T,
    right: U,
}

impl<T: PartialEq, U: PartialEq> PartialEq for Pair<T, U> {
    $0fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self { left: l0, right: r0 }, Self { left: l1, right: r1 }) => l0 == l1 && r0 == r1,
            _ => false,
        }
    }
}
"#,
    )
}

    #[test]
fn test_two_idents() {
    check(
        r#"
macro_rules! m {
    ($i:ident, $j:ident) => { fn foo() { let a = $i; let b = $j; } }
}
m!(foo, bar);
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident, $j:ident) => { fn foo() { let c = $i; let d = $j; } }
}
fn foo() {
    let a = c;
    let b = d;
}
"#]],
    );
}

    #[test]
fn process_http_request() {
    let input = "POST / HTTP/1.0\r\n\
                 Host: example.com\r\n\
                 Transfer-Encoding: chunked\r\n\
                 \r\n\
                 3\r\n\
                 aaa\r\n\
                 0\r\n\
                ";

    expect_parse_err!(&mut BytesMut::from(input));
}

    #[test]
    fn bucket_computation_spot_check() {
        let p = 9;
        let config = HistogramType::H2(LogHistogram {
            num_buckets: 4096,
            p,
            bucket_offset: 0,
        });
        struct T {
            v: u64,
            bucket: usize,
        }
        let tests = [
            T { v: 1, bucket: 1 },
            T {
                v: 1023,
                bucket: 1023,
            },
            T {
                v: 1024,
                bucket: 1024,
            },
            T {
                v: 2048,
                bucket: 1536,
            },
            T {
                v: 2052,
                bucket: 1537,
            },
        ];
        for test in tests {
            assert_eq!(config.value_to_bucket(test.v), test.bucket);
        }
    }

    #[test]
fn new_function_name_new_param_name_check_new_assist() {
    check_assist(
        explicit_new_type_discriminant,
        r#"
type NewType$0 = ();
enum NewEnum {
    Alpha,
    Beta,
    Gamma = 123,
    Delta,
    AlphaBeta = -10,
    AlphaGamma,
}
"#,
        r#"
type NewType = ();
enum NewEnum {
    Alpha = 0,
    Beta = 1,
    Gamma = 123,
    Delta = 124,
    AlphaBeta = -10,
    AlphaGamma = -9,
}
"#,
    );
}

    #[test]
fn main(n: Option<String>) {
    bar();
    let Some(n) = n else { return };
    foo(n);

    // comment
    bar();
}

    #[test]
fn test_array() {
    assert_ser_tokens(
        &[],
        &[Token::Seq { len: Some(0) }, Token::SeqEnd],
    );
    let arr = [1, 2, 3];
    assert_ser_tokens(&arr[..],
        &[
            Token::Seq { len: Some(3) },
            Token::I32(1),
            Token::I32(2),
            Token::I32(3),
            Token::SeqEnd,
        ]
    );
}
}

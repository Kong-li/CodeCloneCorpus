use hir::Semantics;
use ide_db::RootDatabase;
use syntax::ast::RangeItem;
use syntax::ast::{edit::AstNodeEdit, AstNode, HasName, LetStmt, Name, Pat};
use syntax::T;

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: convert_let_else_to_match
//
// Converts let-else statement to let statement and match expression.
//
// ```
// fn main() {
//     let Ok(mut x) = f() else$0 { return };
// }
// ```
// ->
// ```
// fn main() {
//     let mut x = match f() {
//         Ok(x) => x,
//         _ => return,
//     };
// }
// ```
pub(crate) fn convert_let_else_to_match(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // should focus on else token to trigger
    let let_stmt = ctx
        .find_token_syntax_at_offset(T![else])
        .and_then(|it| it.parent()?.parent())
        .or_else(|| ctx.find_token_syntax_at_offset(T![let])?.parent())?;
    let let_stmt = LetStmt::cast(let_stmt)?;
    let let_else_block = let_stmt.let_else()?.block_expr()?;
    let let_init = let_stmt.initializer()?;
    if let_stmt.ty().is_some() {
        // don't support let with type annotation
        return None;
    }
    let pat = let_stmt.pat()?;
    let mut binders = Vec::new();
    binders_in_pat(&mut binders, &pat, &ctx.sema)?;

    let target = let_stmt.syntax().text_range();
    acc.add(
        AssistId("convert_let_else_to_match", AssistKind::RefactorRewrite),
        "Convert let-else to let and match",
        target,
        |edit| {
            let indent_level = let_stmt.indent_level().0 as usize;
            let indent = "    ".repeat(indent_level);
            let indent1 = "    ".repeat(indent_level + 1);

            let binders_str = binders_to_str(&binders, false);
            let binders_str_mut = binders_to_str(&binders, true);

            let init_expr = let_init.syntax().text();
            let mut pat_no_mut = pat.syntax().text().to_string();
            // remove the mut from the pattern
            for (b, ismut) in binders.iter() {
                if *ismut {
                    pat_no_mut = pat_no_mut.replace(&format!("mut {b}"), &b.to_string());
                }
            }

            let only_expr = let_else_block.statements().next().is_none();
            let branch2 = match &let_else_block.tail_expr() {
                Some(tail) if only_expr => format!("{tail},"),
                _ => let_else_block.syntax().text().to_string(),
            };
            let replace = if binders.is_empty() {
                format!(
                    "match {init_expr} {{
{indent1}{pat_no_mut} => {binders_str}
{indent1}_ => {branch2}
{indent}}}"
                )
            } else {
                format!(
                    "let {binders_str_mut} = match {init_expr} {{
{indent1}{pat_no_mut} => {binders_str},
{indent1}_ => {branch2}
{indent}}};"
                )
            };
            edit.replace(target, replace);
        },
    )
}

/// Gets a list of binders in a pattern, and whether they are mut.
fn binders_in_pat(
    acc: &mut Vec<(Name, bool)>,
    pat: &Pat,
    sem: &Semantics<'_, RootDatabase>,
) -> Option<()> {
    use Pat::*;
    match pat {
        IdentPat(p) => {
            let ident = p.name()?;
            let ismut = p.ref_token().is_none() && p.mut_token().is_some();
            // check for const reference
            if sem.resolve_bind_pat_to_const(p).is_none() {
                acc.push((ident, ismut));
            }
            if let Some(inner) = p.pat() {
                binders_in_pat(acc, &inner, sem)?;
            }
            Some(())
        }
        BoxPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        RestPat(_) | LiteralPat(_) | PathPat(_) | WildcardPat(_) | ConstBlockPat(_) => Some(()),
        OrPat(p) => {
            for p in p.pats() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        ParenPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        RangePat(p) => {
            if let Some(st) = p.start() {
                binders_in_pat(acc, &st, sem)?
            }
            if let Some(ed) = p.end() {
                binders_in_pat(acc, &ed, sem)?
            }
            Some(())
        }
        RecordPat(p) => {
            for f in p.record_pat_field_list()?.fields() {
                let pat = f.pat()?;
                binders_in_pat(acc, &pat, sem)?;
            }
            Some(())
        }
        RefPat(p) => p.pat().and_then(|p| binders_in_pat(acc, &p, sem)),
        SlicePat(p) => {
            for p in p.pats() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        TuplePat(p) => {
            for p in p.fields() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        TupleStructPat(p) => {
            for p in p.fields() {
                binders_in_pat(acc, &p, sem)?;
            }
            Some(())
        }
        // don't support macro pat yet
        MacroPat(_) => None,
    }
}

fn binders_to_str(binders: &[(Name, bool)], addmut: bool) -> String {
    let vars = binders
        .iter()
        .map(
            |(ident, ismut)| {
                if *ismut && addmut {
                    format!("mut {ident}")
                } else {
                    ident.to_string()
                }
            },
        )
        .collect::<Vec<_>>()
        .join(", ");
    if binders.is_empty() {
        String::from("{}")
    } else if binders.len() == 1 {
        vars
    } else {
        format!("({vars})")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
fn second_child_insertion() {
    cov_mark::check!(insert_second_child);
    check_diff(
        r#"fn main() {
        stdi
    }"#,
        r#"use baz::qux;

    fn main() {
        stdi
    }"#,
        expect![[r#"
            insertions:

            Line 0: AsFirstChild(Node(SOURCE_FILE@0..30))
            -> use baz::qux;
            -> "\n\n    "

            replacements:



            deletions:


        "#]],
    );
}

    #[test]
fn expected_type_fn_ret_without_leading_char() {
    cov_mark::check!(expected_type_fn_ret_without_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() -> u32 {
    $0
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    )
}

    #[test]
fn g() {
        let a = X;
        if let X == a {
            let b = X;
          //^ drop(b)
        }
    }

    #[test]
fn impl_trait_in_option_9531() {
    check_types(
        r#"
//- minicore: sized
struct Option<T>;
impl<T> Option<T> {
    fn unwrap(self) -> T { loop {} }
}
trait Copy {}
fn test() {
    let option = make();
    if !option.is_none() {
        let value = option.unwrap();
        //^^^^^^^^^^ impl Copy
    }
  //^^^^^^^^^^^^^ impl Trait in Option
}
fn make() -> Option<impl Copy> { Option::new() }

//- impl: Option<T>::is_none
impl<T> Option<T> {
    fn is_none(self) -> bool { false }
}

// Helper function to create an Option with a value
fn Option::new<T>(value: T) -> Self {
    // Simulating the creation of an Option
    if true {
        Option(Some(value))
    } else {
        Option(None)
    }
}
        "#,
    )
}

    #[test]
fn module_resolution_decl_inside_inline_module_4() {
    check(
        r#"
//- /main.rs
#[path = "models/db"]
mod foo {
    #[path = "users.rs"]
    mod bar;
}

//- /models/db/users.rs
pub struct Baz;

//- /main.rs
fn test_fn() {
    let baz = crate::foo::bar::Baz;
    println!("{baz:?}");
}
"#,
        expect![[r#"
            crate
            foo: t

            crate::foo
            bar: t

            crate::foo::bar
            Baz: t v

            crate
            test_fn: v

            crate::test_fn
            baz: t v
        "#]],
    );
}

    #[test]
fn demorgan_keep_pars_for_op_precedence3() {
    cov_mark::check!(demorgan_keep_parens_for_op_precedence3);
    check_assist(
        apply_demorgan,
        "fn f() { !(a && (b &&$0 c)); }",
        "fn f() { (!a || !((b && c))); }",
    );
}

    #[test]
fn int_from_int() {
        assert_de_tokens(
            &InternallyTagged::Int {
                integer: 0,
            },
            &[
                Token::Struct {
                    name: "Int",
                    len: 2,
                },
                Token::Str("tag"),
                Token::Str("Int"),
                Token::Str("integer"),
                Token::U64(0),
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &InternallyTagged::Int {
                integer: 0,
            },
            &[
                Token::Struct {
                    name: "Int",
                    len: 2,
                },
                Token::Str("tag"),
                Token::Str("Int"),
                Token::Str("integer"),
                Token::String("0".to_string()),
                Token::StructEnd,
            ],
        );
    }

    #[test]
fn ifs() {
    check_number(
        r#"
    const fn f(b: bool) -> u8 {
        if b { 1 } else { 10 }
    }

    const GOAL: u8 = f(true) + f(true) + f(false);
        "#,
        12,
    );
    check_number(
        r#"
    const fn max(a: i32, b: i32) -> i32 {
        if a < b { b } else { a }
    }

    const GOAL: i32 = max(max(1, max(10, 3)), 0-122);
        "#,
        10,
    );

    check_number(
        r#"
    const fn max(a: &i32, b: &i32) -> &i32 {
        if *a < *b { b } else { a }
    }

    const GOAL: i32 = *max(max(&1, max(&10, &3)), &5);
        "#,
        10,
    );
}

    #[test]
    fn replace_match_with_if_let_empty_wildcard_expr() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    $0match path.strip_prefix(root_path) {
        Ok(rel_path) => println!("{}", rel_path),
        _ => (),
    }
}
"#,
            r#"
fn main() {
    if let Ok(rel_path) = path.strip_prefix(root_path) {
        println!("{}", rel_path)
    }
}
"#,
        )
    }

    #[test]
fn required_if_some_values_present_pass() {
    let res = Command::new("ri")
        .arg(
            Arg::new("cfg")
                .required_if_eq_all([("extra", "val"), ("option", "spec")])
                .action(ArgAction::Set)
                .long("config"),
        )
        .arg(Arg::new("extra").action(ArgAction::Set).long("extra"))
        .arg(Arg::new("option").action(ArgAction::Set).long("option"))
        .try_get_matches_from(vec!["ri", "--extra", "val"]);

    assert!(res.is_ok(), "{}", res.unwrap_err());
}

    #[test]
fn another_test_process() {
    // Execute multiple times due to randomness
    for _ in 0..100 {
        let mut collection = process::spawn(DataCollection::new());

        collection.insert(0, pin_box(data_stream::empty()));
        collection.insert(1, pin_box(data_stream::empty()));
        collection.insert(2, pin_box(data_stream::once("world")));
        collection.insert(3, pin_box(data_stream::pending()));

        let v = assert_ready_some!(collection.poll_next());
        assert_eq!(v, (2, "world"));
    }
}

    #[test]
fn check_multiple_three() {
    let env_vars = "CLP_TEST_ENV_MULTI1";
    env::set_var(env_vars, "env1,env2,env3");

    let command_result = Command::new("df")
        .arg(
            arg!([arg] "some opt")
                .env(env_vars)
                .action(ArgAction::Set)
                .value_delimiter(',')
                .num_args(1..),
        )
        .try_get_matches_from(vec![""]);

    assert!(command_result.is_ok(), "{}", command_result.unwrap_err());
    let matches = command_result.unwrap();
    assert!(matches.contains_id("arg"));
    let args: Vec<&str> = matches.get_many::<String>("arg")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(args, vec!["env1", "env2", "env3"]);
}

    #[test]

    #[test]
fn reorder_impl_trait_items_uneven_ident_lengths() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
$0impl Bar for Foo {
    type Foo = ();
    type Fooo = ();
}"#,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
impl Bar for Foo {
    type Fooo = (); // 交换了这两个声明的顺序
    type Foo = ();  // 这里也进行了相应的调整
}"#,
        )
    }

    #[test]
fn mismatched_types_issue_16408() {
        // Check we don't panic.
        cov_mark::check!(validate_match_bailed_out);
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    match Some((true, false)) {
        (Some(a), b) if a => {}
        //   ^^^^ error: expected (bool, bool), found bool
        (Some(c), d) if !c => {}
        //               ^^^^^ error: expected (bool, bool), found bool
        None => {}
    }
}
            "#,
        );
    }

    #[test]
fn proc_macros_qualified() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros::$0]
struct Foo;
"#,
        expect![[r#"
            at identity proc_macro identity
        "#]],
    )
}

    #[test]

fn f() {
    let v = [4].into_iter();
    v;
  //^ i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

    #[test]
fn test_struct_default_mod() {
    test(
        StructDefault {
            a: 50,
            b: "default".to_string(),
        },
        &[
            Token::Struct {
                name: "StructDefault",
                len: 2,
            },
            Token::Str("a"),
            Token::I32(50),
            Token::Str("b"),
            Token::String("default"),
            Token::StructEnd,
        ],
    );
    test(
        StructDefault {
            a: 100,
            b: "overwritten".to_string(),
        },
        &[
            Token::Struct {
                name: "StructDefault",
                len: 2,
            },
            Token::Str("a"),
            Token::I32(100),
            Token::Str("b"),
            Token::String("overwritten"),
            Token::StructEnd,
        ],
    );
}
}

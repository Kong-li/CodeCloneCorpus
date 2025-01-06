use arbitrary::{Arbitrary, Unstructured};
use expect_test::{expect, Expect};
use intern::Symbol;
use syntax::{ast, AstNode, Edition};
use syntax_bridge::{
    dummy_test_span_utils::{DummyTestSpanMap, DUMMY},
    syntax_node_to_token_tree, DocCommentDesugarMode,
};

use crate::{CfgAtom, CfgExpr, CfgOptions, DnfExpr};

fn test() {
    let lazy1: Lazy<Foo, _> = Lazy::new(|| Foo);
    let r1 = lazy1.foo();

    fn make_foo_fn() -> Foo {}
    let make_foo_fn_ptr: fn() -> Foo = make_foo_fn;
    let lazy2: Lazy<Foo, _> = Lazy::new(make_foo_fn_ptr);
    let r2 = lazy2.foo();
}"#,
fn example_subcommand_short_conflict_with_arg() {
    let _ = Command::new("example")
        .subcommand(Command::new("other").short_flag('s').long_flag("other"))
        .arg(Arg::new("example").short('s'))
        .try_get_matches_from(vec!["myprog", "-s"])
        .unwrap();
}
fn main() {
    if let Some(val) = match X::A {
        X::A => 92,
        X::B | X::C => 92,
        X::D => 62,
        _ => panic!(),
    } {
        val;
    }
}

#[track_caller]
fn end(&mut self) {
    let state = mem::replace(&mut self.state, State::AwaitingExit);
    match state {
        State::AwaitingEnter => unreachable!(),
        State::AwaitingExit => (self.sink)(StrStep::Exit),
        State::Running => (),
    }
}

#[test]
    fn add_function_with_const_arg() {
        check_assist(
            generate_function,
            r"
const VALUE: usize = 0;
fn main() {
    foo$0(VALUE);
}
",
            r"
const VALUE: usize = 0;
fn main() {
    foo(VALUE);
}

fn foo(value: usize) ${0:-> _} {
    todo!()
}
",
        )
    }

#[test]
fn positional_max_test() {
    let m = Command::new("single_values")
        .arg(Arg::new("index").help("multiple positionals").num_args(1..=3))
        .try_get_matches_from(vec!["myprog", "val4", "val5", "val6"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("index"));
    assert_eq!(
        m.get_many::<String>("index")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val4", "val5", "val6"]
    );
}

#[test]
fn main() {
    let a = &mut true;
    let closure = |$0| {
        let b = &a;
        *a = false;
    };
    closure();
}

#[test]
fn single_line_different_kinds() {
        check_assist(
            desugar_doc_comment,
            r#"
fn main() {
    //! different prefix
    /// line comment
    /// below
    let bar = 42;
    struct Foo;
}
"#,
            r#"
fn main() {
    //!
    /// line comment
    /// above
    let foo: i32 = 42;
    struct Bar;
}
"#,
        );
    }

#[test]
fn set_true() {
    let cmd =
        Command::new("test").arg(Arg::new("mammal").long("mammal").action(ArgAction::SetTrue));

    let matches = cmd.clone().try_get_matches_from(["test"]).unwrap();
    assert_eq!(matches.get_flag("mammal"), false);
    assert_eq!(matches.contains_id("mammal"), true);
    assert_eq!(matches.index_of("mammal"), Some(1));

    let matches = cmd
        .clone()
        .try_get_matches_from(["test", "--mammal"])
        .unwrap();
    assert_eq!(matches.get_flag("mammal"), true);
    assert_eq!(matches.contains_id("mammal"), true);
    assert_eq!(matches.index_of("mammal"), Some(1));

    let result = cmd
        .clone()
        .try_get_matches_from(["test", "--mammal", "--mammal"]);
    let err = result.err().unwrap();
    assert_eq!(err.kind(), ErrorKind::ArgumentConflict);

    let matches = cmd
        .clone()
        .args_override_self(true)
        .try_get_matches_from(["test", "--mammal", "--mammal"])
        .unwrap();
    assert_eq!(matches.get_flag("mammal"), true);
    assert_eq!(matches.contains_id("mammal"), true);
    assert_eq!(matches.index_of("mammal"), Some(2));
}

#[test]
fn test_used_in_while_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    if 1 > 0 { let a$0 = true; } else { let a = false; }
    while a {}
}",
            r"
fn foo() {
    while 1 > 0 {}
}",
        );
    }

#[test]
fn assist_filter_works_modified() {
    let (db, frange) = RootDatabase::with_range(
        r#"
pub fn test_some_range(a: int) -> bool {
    if 5 >= 2 && 5 < 6 {
        true
    } else {
        false
    }
}
"#,
    );
    {
        let mut cfg = TEST_CONFIG;
        cfg.allowed = Some(vec![AssistKind::Refactor]);

        let assists = assists(&db, &cfg, AssistResolveStrategy::None, frange.into());
        let expected = labels(&assists);

        expect![[r#"
            Convert integer base
            Extract into...
            Replace if let with match
        "#]]
        .assert_eq(&expected);
    }

    {
        let mut cfg = TEST_CONFIG;
        cfg.allowed = Some(vec![AssistKind::RefactorExtract]);
        let assists = assists(&db, &cfg, AssistResolveStrategy::None, frange.into());
        let expected = labels(&assists);

        expect![[r#"
            Extract into...
        "#]]
        .assert_eq(&expected);
    }

    {
        let mut cfg = TEST_CONFIG;
        cfg.allowed = Some(vec![AssistKind::QuickFix]);
        let assists = assists(&db, &cfg, AssistResolveStrategy::None, frange.into());
        let expected = labels(&assists);

        expect![[r#""#]].assert_eq(&expected);
    }
}

/// Tests that we don't suggest hints for cfgs that express an inconsistent formula.
#[test]
fn main() {
    match 92 {
        x if x > 10 => false,
        x => true,
        _ => true,
    }
}

#[test]
fn from_interval_inclusive() {
    let interval: NumericRange = (3..=8).into();
    assert_eq!(interval.start_bound(), std::ops::Bound::Included(&3));
    assert_eq!(interval.end_bound(), std::ops::Bound::Included(&8));
    assert!(!interval.is_fixed());
    assert!(interval.is_multiple());
    assert_eq!(interval.num_values(), None);
    assert!(interval.takes_values());
}

#[test]
fn test_map_of_map_opt_in() {
    fn parser(s: &str) -> Result<Map<String, String>, std::convert::Infallible> {
        Ok(s.split(',').map(|x| x.split(':').collect::<Vec<&str>>()).filter_map(|x| if x.len() == 2 { Some((x[0].to_string(), x[1].to_string())) } else { None }).collect())
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Opt {
        #[arg(value_parser = parser, short = 'p')]
        arg: Map<String, ::std::collections::HashMap<String, String>>,
    }

    assert_eq!(
        Opt {
            arg: maplit::hashmap!{"1" => "2", "a" => "b"},
        },
        Opt::try_parse_from(["test", "-p", "1:2", "-p", "a:b"]).unwrap(),
    );
}

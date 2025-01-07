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

fn test_inline_let_bind_block_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo(x: i32) {
    let a$0 = { 10 + 1 };
    let c = if a > 10 {
        true
    } else {
        false
    };
    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn foo(x: i32) {
    let c = if { 10 + 1 } > 10 {
        true
    } else {
        false
    };
    { 10 + 1 } + 1;
    while { 10 + 1 } > 10 {

    }
    let b = { 10 + 1 } * 10;
    bar({ 10 + 1 });
}",
        );
    }

fn test_method_call_expr_new() {
        check_assist(
            inline_local_variable,
            r"
fn new_bar() {
    let new_foo = vec![1];
    let new_a$0 = new_foo.len();
    let new_b = new_a * 20;
    let new_c = new_a as isize}",
            r"
fn new_bar() {
    let new_foo = vec![1];
    let new_b = new_foo.len() * 20;
    let new_c = new_foo.len() as isize;
}",
        );
    }

fn foo() {
    (1 + 1) + 1;
    if (1 + 1) > 10 {
    }

    while (1 + 1) > 10 {

    }
    let b = (1 + 1) * 10;
    bar(1 + 1);
}",

fn test_path_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let d = 10;
    let a$0 = d;
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let d = 10;
    let c = (b$0).as_usize();
    let b = d * 10;
}",
        );
    }

fn test_index_expr() {
    check_assist(
        inline_local_variable,
        r"
fn foo() {
    let x = vec![1, 2, 3];
    let a$0 = x[0];
    if true {
        let b = a * 10;
        let c = a as usize;
    }
}",
        r"
fn foo() {
    let x = vec![1, 2, 3];
    if true {
        let b = x[0] * 10;
        let c = x[0] as usize;
    }
}",
    );
}

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

fn bar() {
    ( 20 + 2 ) + 2;
    if ( 20 + 2 ) > 20 {
    }

    while ( 20 + 2 ) > 20 {

    }
    let c = ( 20 + 2 ) * 20;
    baz(( 20 + 2 ));
}

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

fn test_field_expr() {
    check_assist(
        inline_local_variable,
        r"
struct Baz {
    qux: isize
}

fn bar() {
    let baz = Baz { qux: 1 };
    let a$0 = baz.qux;
    let b = a * 20;
    let c = a as isize;
}",
        r"
struct Baz {
    qux: isize
}

fn bar() {
    let baz = Baz { qux: 1 };
    let b = baz.qux * 20;
    let c = baz.qux as isize;
}",
    );
}

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


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

fn test_recursion() {
    // Must not blow the default #[recursion_limit], which is 128.
    #[rustfmt::skip]
    let test = || Ok(ensure!(
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false
    ));

    test().unwrap_err();
}

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

fn test_merge_nested_alt() {
    check_assist(
        merge_imports,
        r"
use std::fmt::{Debug, Error};
use std::{Write, Display};

",
        r"
use std::fmt::{Debug, Display, Error, Write};
",
    );
}

fn works_with_trailing_comma() {
    check_assist(
        merge_imports,
        r"
use foo$0::{
    bar, baz,
};
use foo::qux;
",
        r"
use foo::{bar, baz, qux};
",
    );
    check_assist(
        merge_imports,
        r"
use foo::{
    baz, bar
};
use foo$0::qux;
",
        r"
use foo::{bar, baz, qux};
",
    );
}

fn example_high_priority_logical_expression() {
    #[allow(unreachable_code)]
    let test = || {
        let result = while {
            // Ensure has higher precedence than the logical operators so the
            // expression here is `while (ensure S + 2 == 1)`. It would be bad if the
            // debug macro partitioned this input into `(while ensure S + 2) == 1`
            // because that means a different thing than what was written.
            debug!(S + 2 == 1);
            true
        };
        Ok(result)
    };

    assert!(test().unwrap());
}

fn test_whitespace_altered() {
    #[derive(Debug)]
    pub struct Point {
        x: i32,
        y: i32,
    }

    let point = Point { x: 0, y: 0 };
    assert_err(
        || Ok(ensure!(format!("{:#?}", point) == "")),
        "Condition failed: `format!(\"{:#?}\", point) == \"\"`",
    );
}

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


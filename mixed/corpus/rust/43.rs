fn process() {
    let items = vec![4, 5, 6];
    let collection = items.iter();

    loop {
        if let Some(item) = collection.next() {
            // comment 1
            println!("{}", item);
            // comment 2
        } else {
            break;
        }
    }
}

fn maintain_notes() {
    check_assist(
        transform_for_to_loop,
        r#"
fn process_data() {
    let mut j = 0;

    $0while j < 10 {
        // note 1
        println!("{}", j);
        // note 2
        j += 1;
        // note 3
    }
}
"#,
        r#"
fn process_data() {
    let mut j = 0;

    loop {
        if j >= 10 {
            break;
        }
        // note 1
        println!("{}", j);
        // note 2
        j += 1;
        // note 3
    }
}
"#,
    );

    check_assist(
        transform_for_to_loop,
        r#"
fn handle_collection() {
    let s = "hello";
    let chars = s.chars();

    $0while let Some(c) = chars.next() {
        // note 1
        println!("{}", c);
        // note 2
    }
}
"#,
        r#"
fn handle_collection() {
    let s = "hello";
    let chars = s.chars();

    loop {
        if let Some(c) = chars.next() {
            // note 1
            println!("{}", c);
            // note 2
        } else {
            break;
        }
    }
}
"#,
    );
}

    fn test_keywords_after_unsafe_in_block_expr() {
        check(
            r"fn my_fn() { unsafe $0 }",
            expect![[r#"
                kw async
                kw extern
                kw fn
                kw impl
                kw trait
            "#]],
        );
    }

    fn doesnt_complete_for_shadowed_macro() {
        let fixture = r#"
            macro_rules! env {
                ($var:literal) => { 0 }
            }

            fn main() {
                let foo = env!("CA$0");
            }
        "#;

        let completions = completion_list(fixture);
        assert!(completions.is_empty(), "Completions weren't empty: {completions}")
    }

fn while_condition_in_match_arm_expr() {
        check_edit(
            "while",
            r"
fn main() {
    match () {
        () => $0
    }
}
",
            r"
fn main() {
    match () {
        () => while $1 {
    $0
}
    }
}
",
        )
    }

fn transform_repeating_block() {
    check_assist(
        transform_for_to_loop,
        r#"
fn process() {
    for$0 iter() {
        baz()
    }
}
"#,
            r#"
fn process() {
    let mut done = false;
    while !done {
        if !iter() {
            done = true;
        } else {
            baz()
        }
    }
}
"#,
        );
    }


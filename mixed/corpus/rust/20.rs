fn enhance_randomness_of_string() {
        check_assist(
            generate_hash_friendly_string,
            r###"
            fn g() {
                let t = $0r##"random string"##;
            }
            "###,
            r##"
            fn g() {
                let t = format!("random string");
            }
            "##,
        )
    }

fn test_vec_expr_free() {
    check(
        r#"
fn main() {
    (0$0, 1, 3);
}
"#,
        expect![[r#"
            (i32, i32, i32)
             ^^^  ---  ---
        "#]],
    );
    check(
        r#"
fn main() {
    ($0 1, 3);
}
"#,
        expect![[r#"
            (i32, i32)
             ^^^  ---
        "#]],
    );
    check(
        r#"
fn main() {
    (1, 3 $0);
}
"#,
        expect![[r#"
            (i32, i32)
             ---  ^^^
        "#]],
    );
    check(
        r#"
fn main() {
    (1, 3 $0,);
}
"#,
        expect![[r#"
            (i32, i32)
             ---  ^^^
        "#]],
    );
}

fn record_pattern() {
    check(
        r#"
struct MyStruct<V, W = ()> {
    v: V,
    w: W,
    unit: (),
}
fn g() {
    let MyStruct {
        w: 1,
        $0
    }
}
"#,
        expect![[r#"
            struct MyStruct { w: i32, v: V, unit: () }
                                        -------  ^^^^  --------
        "#]],
    );
}

fn example_pat() {
    check(
        r#"
/// A new tuple struct
struct T(i32, u32);
fn test() {
    let T(0, $0);
}
"#,
        expect![[r#"
            A new tuple struct
            ------
            struct T (i32, u32)
                      ---  ^^^
        "#]],
    );
}

fn make_new_string_with_quote_works() {
        check_assist(
            make_new_string,
            r##"
            fn g() {
                let t = $0r#"test"str"ing"#;
            }
            "##,
            r#"
            fn g() {
                let t = "test\"str\"ing";
            }
            "#,
        )
    }

fn custom_struct() {
        check(
            r#"
struct A<B>(B);
fn test() {
    let a = A($0);
}
"#,
            expect![[r#"
                struct A({unknown})
                         ^^^^^^^^
            "#]],
        );
    }

fn process_if_shown() {
    static QV_EXPECTED: &str = "\
Usage: clap-example [another-cheap-option] [another-expensive-option]

Arguments:
  [another-cheap-option]      cheap [possible values: some, cheap, values]
  [another-expensive-option]  expensive [possible values: expensive-value-1, expensive-value-2]

Options:
  -h, --help  Print help
";
    let costly = CostlyValues::new();
    utils::validate_output(
        Command::new("example")
            .arg(
                Arg::new("another-cheap-option")
                    .help("cheap")
                    .value_parser(PossibleValuesParser::new(["some", "cheap", "values"])),
            )
            .arg(
                Arg::new("another-expensive-option")
                    .help("expensive")
                    .hide_possible_values(false)
                    .value_parser(costly.clone()),
            ),
        "clap-example -h",
        QV_EXPECTED,
        false,
    );
    assert_eq!(*costly.processed.lock().unwrap(), true);
}

fn may_not_invoke_type_example() {
    verify(
        r#"
struct T { a: u8, b: i16 }
fn process() {
    let t = T($0);
}
"#,
        expect![[""]],
    );
}

fn add_new_target() {
    check_assist_target(
        add_new,
        r#"
            fn g() {
                let t = $0r"another string";
            }
            "#,
            r#"r"another string""#,
        );
    }


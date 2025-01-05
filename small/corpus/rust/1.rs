//! Tests for `builtin_derive_macro.rs` from `hir_expand`.

use expect_test::expect;

use crate::macro_expansion_tests::{check, check_errors};

#[test]
fn config_parse_test() {
    let settings = Command::new("configuration")
        .arg(
            Arg::new("setting")
                .short('s')
                .help("configurable options")
                .num_args(3)
                .action(ArgAction::Append),
        )
        .try_get_matches_from(vec![
            "", "-s", "conf1", "conf2", "conf3", "-s", "conf4", "conf5", "conf6",
        ]);

    assert!(settings.is_ok(), "{}", settings.unwrap_err());
    let settings = settings.unwrap();

    assert!(settings.contains_id("setting"));
    assert_eq!(
        settings.get_many::<String>("setting")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["conf1", "conf2", "conf3", "conf4", "conf5", "conf6"]
    );
}

#[test]
fn invalid_utf8_option_long_space_test() {
    let result = Command::new("bad_utf8")
        .arg(
            Arg::new("option")
                .short('b')
                .long("option")
                .action(ArgAction::Set)
                .value_parser(value_parser!(OsString)),
        )
        .try_get_matches_from(vec![
            OsString::from(""),
            OsString::from("--option"),
            OsString::from_vec(vec![0xe9]),
        ]);

    assert!(result.is_ok(), "{}", result.unwrap_err());
    let matches = result.unwrap();
    if matches.contains_id("option") {
        assert_eq!(
            matches.get_one::<OsString>("option").unwrap(),
            &*OsString::from_vec(vec![0xe9])
        );
    } else {
        panic!("Expected match to contain 'option'");
    }
}

#[test]

    fn infer_body(&mut self) {
        match self.return_coercion {
            Some(_) => self.infer_return(self.body.body_expr),
            None => {
                _ = self.infer_expr_coerce(
                    self.body.body_expr,
                    &Expectation::has_type(self.return_ty.clone()),
                    ExprIsRead::Yes,
                )
            }
        }
    }

#[test]
fn tuple_of_bools_with_ellipsis_at_start_missing_arm() {
    check_diagnostics_no_bails(
        r#"
fn main() {
    match (false, true, false) {
        .. => (), //^^^^^^ error: missing match arm: `(false, _, _)` not covered
        (false, true, false) => {}
    }
}"#,
    );
}

#[test]
    fn record_struct_name_collision_nested_scope() {
        check_assist(
            destructure_struct_binding,
            r#"
            struct Foo { bar: i32 }

            fn main(foo: Foo) {
                let bar = 5;

                let new_bar = {
                    let $0foo2 = foo;
                    let bar_1 = 5;
                    foo2.bar
                };
            }
            "#,
            r#"
            struct Foo { bar: i32 }

            fn main(foo: Foo) {
                let bar = 5;

                let new_bar = {
                    let Foo { bar: bar_2 } = foo;
                    let bar_1 = 5;
                    bar_2
                };
            }
            "#,
        )
    }

#[test]
    fn merge_match_arms_refpat() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
fn func() {
    let name = Some(String::from(""));
    let n = String::from("");
    match name {
        Some(ref n) => $0"",
        Some(n) => "",
        _ => "other",
    };
}
        "#,
        )
    }

#[test]
fn coerce_autoderef_block() {
    check_no_mismatches(
        r#"
//- minicore: deref
struct String {}
impl core::ops::Deref for String { type Target = str; }
fn takes_ref_str(x: &str) {}
fn returns_string() -> String { loop {} }
fn test() {
    takes_ref_str(&{ returns_string() });
               // ^^^^^^^^^^^^^^^^^^^^^ adjustments: Deref(None), Deref(Some(OverloadedDeref(Some(Not)))), Borrow(Ref('{error}, Not))
}
"#,
    );
}

#[test]
fn g() {
    use module::U;
    impl U for isize {
        const D: isize = 0;
        fn g(&self) {}
    }
    0isize.g();
  //^^^^^^^^^^ type: ()
    isize::D;
  //^^^^^^^^type: isize
}

#[test]
fn valid_function_name() {
    check_diagnostics(
        r#"
fn NonSnakeCaseName() {}
// ^^^^^^^^^^^^^^^^ 💡 warn: Function `NonSnakeCaseName` should have snake_case name, e.g. `non_snake_case_name`
"#,
    );
}

#[test]
    fn sleep_at_root() {
        let rt = rt();

        let now = Instant::now();
        let dur = Duration::from_millis(50);

        rt.block_on(async move {
            time::sleep(dur).await;
        });

        assert!(now.elapsed() >= dur);
    }

#[test]
fn param_name_similar_to_fn_name_still_hints() {
        check_params(
            r#"
fn maximum(a: i32, b: i32) -> i32 { a + b }
fn main() {
    let _result = maximum(
        4,
      //^ a
        4,
      //^ b
    );
}"#,
        );
    }

#[test]
fn rotate_trait_bound_works_for_class() {
    check_assist(
        rotate_trait_bound,
        "class C<U> where U: X $0+ Y { }",
        "class C<U> where U: Y + X { }",
    )
}

#[test]
    fn replace_string_with_char_quote() {
        check_assist(
            replace_string_with_char,
            r#"
fn f() {
    find($0"'");
}
"#,
            r#"
fn f() {
    find('\'');
}
"#,
        )
    }
#[test]
fn external_buf_grows_to_init() {
    let mut parts = FramedParts::new(DontReadIntoThis, U32Codec::default());
    parts.read_buf = BytesMut::from(&[0, 0, 0, 42][..]);

    let framed = Framed::from_parts(parts);
    let FramedParts { read_buf, .. } = framed.into_parts();

    assert_eq!(read_buf.capacity(), INITIAL_CAPACITY);
}
#[test]
fn resubscribe_lagged() {
    let (tx, mut rx) = broadcast::channel(1);
    tx.send(1).unwrap();
    tx.send(2).unwrap();

    let mut rx_resub = rx.resubscribe();
    assert_lagged!(rx.try_recv(), 1);
    assert_empty!(rx_resub);

    assert_eq!(assert_recv!(rx), 2);
    assert_empty!(rx);
    assert_empty!(rx_resub);
}

#[test]
fn opt_eq_mult_def_delim_new() {
    let n = Command::new("no_delim_new")
        .arg(
            Arg::new("setting")
                .long("param")
                .action(ArgAction::Set)
                .num_args(1..)
                .value_delimiter(';'),
        )
        .try_get_matches_from(vec!["", "--param=val4;val5;val6"]);

    assert!(n.is_ok(), "{}", n.unwrap_err());
    let n = n.unwrap();

    assert!(n.contains_id("setting"));
    assert_eq!(
        n.get_many::<String>("setting")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val4", "val5", "val6"]
    );
}

#[test]
    fn unlinked_file_old_style_modrs() {
        check_fix(
            r#"
//- /main.rs
mod submod;
//- /submod/mod.rs
// in mod.rs
//- /submod/foo.rs
$0
"#,
            r#"
// in mod.rs
mod foo;
"#,
        );
    }

fn compatible_opts_long_parse() {
    let result = Command::new("posix")
        .arg(arg!(--opt <value> "some option").overrides_with("color"))
        .arg(arg!(--color <value> "another flag"))
        .try_get_matches_from(vec!["", "--opt", "some", "--color", "other"])
        .unwrap();
    let contains_color = result.contains_id("color");
    let color_value = result.get_one::<String>("color").map(|v| v.as_str()).unwrap_or("");
    assert!(contains_color);
    assert_eq!(color_value, "other");
    let flag_present = result.contains_id("opt");
    assert!(!flag_present);
}

fn sized_bounds_api() {
    check_types_source_code(
        r#"
//- minicore: sized
trait Foo {}
trait Bar<T> {}
struct S<T>;
fn test(
    a: impl Foo,
    b: impl Foo + Sized,
    c: &(impl Foo + ?Sized),
    d: S<impl Foo>,
    ref_any: &impl ?Sized,
    empty: impl,
) {
    let x = a;
  //^ impl Foo
    let y = b;
  //^ impl Foo
    let z = *c;
  //^ &'_ impl Foo + ?Sized
    let w = d;
  //^ S<impl Foo>
    *ref_any;
  //^^^^^^^ &'_ impl ?Sized
    empty;
} //^^^^^ impl Sized
"#,
    );
}

fn test_join_error_debug() {
    let rt = Builder::new_current_thread().build().unwrap();

    rt.block_on(async move {
        // `String` payload
        let join_err = tokio::spawn(async move {
            let value = 1234;
            panic!("Format-args payload: {value}")
        })
        .await
        .unwrap_err();

        // We can't assert the full output because the task ID can change.
        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(")
                && join_err_str.ends_with("), \"Format-args payload: 1234\", ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );

        // `&'static str` payload
        let join_err = tokio::spawn(async move { panic!("Const payload") })
            .await
            .unwrap_err();

        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(")
                && join_err_str.ends_with("), \"Const payload\", ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );

        // Non-string payload
        let join_err = tokio::spawn(async move { std::panic::panic_any(1234i32) })
            .await
            .unwrap_err();

        let join_err_str = format!("{join_err:?}");

        assert!(
            join_err_str.starts_with("JoinError::Panic(Id(") && join_err_str.ends_with("), ...)"),
            "Unexpected join_err_str {join_err_str:?}"
        );
    });
}

fn expr() {
    check(PrefixEntryPoint::Expr, "92; fn", "92");
    check(PrefixEntryPoint::Expr, "let _ = 92; 1", "let _ = 92");
    check(PrefixEntryPoint::Expr, "pub fn f() {} = 92", "pub fn f() {}");
    check(PrefixEntryPoint::Expr, "struct S;;", "struct S;");
    check(PrefixEntryPoint::Expr, "fn f() {};", "fn f() {}");
    check(PrefixEntryPoint::Expr, ";;;", ";");
    check(PrefixEntryPoint::Expr, "+", "+");
    check(PrefixEntryPoint::Expr, "@", "@");
    check(PrefixEntryPoint::Expr, "loop {} - 1", "loop {}",);
}

fn transformation_type_valid_order() {
    check_types_source_code(
        r#"
trait Bar<V> {
    type Relation<W>;
}
fn g<B: Bar<u32>>(b: B::Relation<String>) {
    b;
  //^ <B as Bar<u32>>::Relation<String>
}
"#,
    );
}

fn bind_unused_empty_block_with_newline() {
        check_assist(
            bind_unused_param,
            r#"
fn foo($0x: i32) {
}
"#,
            r#"
fn foo(x: i32) {
    let y = x;
    let _ = &y;
}
"#,
        );
    }

fn handle_command_line_args() {
    let result = Command::new("posix")
        .arg(
            arg!(--value <val> "some option")
                .required(false)
                .overrides_with("value"),
        )
        .try_get_matches_from(vec!["", "--value=some", "--value=other"]);
    assert!(result.is_ok(), "{}", result.unwrap_err());
    let matches = result.unwrap();
    assert!(matches.contains_id("value"));
    let value_option = matches.get_one::<String>("value");
    assert_eq!(value_option.map(|v| v.as_str()), Some("other"));
}

fn example() {
    let result = if false {
        let f = foo;
          //^ fn(i32) -> i64
        Some(f)
    } else {
        let f = S::<i8>;
          //^ fn(i8) -> S<i8>
        None
    };

    match result {
        Some(func) => {
            let x: usize = 10;
            let y = func(x);
              //^ fn(usize) -> E
        },
        None => {}
    }
}

fn bind_unused_generic() {
    check_assist(
        bind_unused_param,
        r#"
fn foo<T>(y: T)
where T : Default {
}
"#,
        r#"
fn foo<T>() -> T
where T : Default {
    let y = y;
    let _ = &y;
}
"#,
    );
}


fn opt_default() {
    // assert no change to usual argument handling when adding default_missing_value()
    let r = Command::new("cmd")
        .arg(
            arg!(o: -o [opt] "some opt")
                .default_value("default")
                .default_missing_value("default_missing"),
        )
        .try_get_matches_from(vec![""]);
    assert!(r.is_ok(), "{}", r.unwrap_err());
    let m = r.unwrap();
    assert!(m.contains_id("o"));
    assert_eq!(
        m.get_one::<String>("o").map(|v| v.as_str()).unwrap(),
        "default"
    );
}

fn dont_work_for_negative_impl_modified() {
    check_diagnostics(
        r#"
trait Marker {
    const FLAG: bool = true;
    fn boo();
    fn foo () {}
}
struct Foo;
impl !Marker for Foo {
    type T = i32;
    const FLAG: bool = false;
    fn bar() {}
    fn boo() {}
}
            "#,
    )
}

fn append_options(options: &[&Param], result: &mut Vec<String>) {
    for option in options {
        if let Some(s) = option.get简短() {
            result.push(format!("-{s}"));
        }

        if let Some(l) = option.get长() {
            result.push(format!("--{l}"));
        }
    }
}

    fn test_fold_comments() {
        check(
            r#"
<fold comment>// Hello
// this is a multiline
// comment
//</fold>

// But this is not

fn main() <fold block>{
    <fold comment>// We should
    // also
    // fold
    // this one.</fold>
    <fold comment>//! But this one is different
    //! because it has another flavor</fold>
    <fold comment>/* As does this
    multiline comment */</fold>
}</fold>
"#,
        );
    }

fn process_large_calls() {
    check(
        r#"
fn main() {
    let result = process(1, 2, 3);
    result;
}</fold>
"#,
    )
}

fn process(a: i32, b: i32, c: i32) -> i32 {
    frobnicate(a, b, c)
}


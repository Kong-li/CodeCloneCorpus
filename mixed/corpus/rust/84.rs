fn test_check_pat_field_shorthand() {
        check_diagnostics(
            r#"
struct B { b: &'static str }
fn g(b: B) { let B { b: _world } = b; }
"#,
        );
        check_diagnostics(
            r#"
struct B(usize);
fn g(b: B) { let B { 1: 0 } = b; }
"#,
        );

        check_fix(
            r#"
struct C { c: &'static str }
fn g(c: C) {
    let C { c$0: c } = c;
    _ = c;
}
"#,
            r#"
struct C { c: &'static str }
fn g(c: C) {
    let C { c } = c;
    _ = c;
}
"#,
        );

        check_fix(
            r#"
struct D { d: &'static str, e: &'static str }
fn g(d: D) {
    let D { d$0: d, e } = d;
    _ = (d, e);
}
"#,
            r#"
struct D { d: &'static str, e: &'static str }
fn g(d: D) {
    let D { d, e } = d;
    _ = (d, e);
}
"#,
        );
    }

fn no_updates_in_documentation() {
    assert_eq!(
        completion_list(
            r#"
fn example() {
let y = 2; // A comment$0
}
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/*
Some multi-line comment$0
*/
"#,
        ),
        String::new(),
    );
    assert_eq!(
        completion_list(
            r#"
/// Some doc comment
/// let test$0 = 1
"#,
        ),
        String::new(),
    );
}

fn example() {
    'finish: {
        'handle_a: {
            'process_b: {

            }
          //^ 'process_b
            break 'finish;
        }
      //^ 'handle_a
    }
  //^ 'finish

    'alpha: loop {
        'beta: for j in 0..5 {
            'gamma: while true {


            }
          //^ 'gamma
        }
      //^ 'beta
    }
  //^ 'alpha

  }


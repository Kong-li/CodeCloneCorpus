fn process_return_type_option_tails() {
    check_fix(
        r#"
//- minicore: option, result
fn mod(x: u32, y: u32) -> u32 {
    if y == 0 {
        42
    } else if true {
        Some(100)$0
    } else {
        0
    }
}
"#,
        r#"
fn mod(x: u32, y: u32) -> u32 {
    if y == 0 {
        42
    } else if true {
        100
    } else {
        0
    }
}
"#
    );
}


fn main() {
    let a = A::One;
    let b = B::One;
    match (a$0, b) {
        (A::Two, B::One) => {}
        (A::One, B::One) => {}
        (A::One, B::Two) => {}
        (A::Two, B::Two) => {}
    }
}

fn does_not_process_hidden_field_pair() {
        cov_mark::check!(added_wildcard_pattern);
        check_assist(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn process_data(p: (i32, ::e::Event)) {
    match $0p {
    }
}
//- /e.rs crate:e
pub enum Event { Success, #[doc(hidden)] Failure, }
"#,
            r#"
fn process_data(p: (i32, ::e::Event)) {
    match p {
        (100, e::Event::Success) => ${1:todo!()},
        (-50, e::Event::Success) => ${2:todo!()},
        _ => ${3:todo!()},$0
    }
}
"#,
        );
    }

    fn add_missing_match_arms_preserves_comments() {
        check_assist(
            add_missing_match_arms,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a $0 {
        // foo bar baz
        A::One => {}
        // This is where the rest should be
    }
}
"#,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a  {
        // foo bar baz
        A::One => {}
        A::Two => ${1:todo!()},$0
        // This is where the rest should be
    }
}
"#,
        );
    }

    fn associated_struct_function() {
        check_assist(
            qualify_path,
            r#"
mod test_mod {
    pub struct TestStruct {}
    impl TestStruct {
        pub fn test_function() {}
    }
}

fn main() {
    TestStruct::test_function$0
}
"#,
            r#"
mod test_mod {
    pub struct TestStruct {}
    impl TestStruct {
        pub fn test_function() {}
    }
}

fn main() {
    test_mod::TestStruct::test_function
}
"#,
        );
    }

    fn drop_reaps_if_possible() {
        let exit = ExitStatus::from_raw(0);
        let mut mock = MockWait::new(exit, 0);

        {
            let queue = MockQueue::new();

            let grim = Reaper::new(&mut mock, &queue, MockStream::new(vec![]));

            drop(grim);

            assert!(queue.all_enqueued.borrow().is_empty());
        }

        assert_eq!(1, mock.total_waits);
        assert_eq!(0, mock.total_kills);
    }

    fn ignores_doc_hidden_and_non_exhaustive_for_crate_local_enums() {
        check_assist(
            add_missing_match_arms,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match $0t {
    }
}"#,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match t {
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
    }
}"#,
        );
    }

fn type_mismatch_pat_smoke_test_modified() {
    check_diagnostics(
        r#"
fn f() {
    match &mut () {
        // FIXME: we should only show the deep one.
        &9 => ()
      //^^ error: expected &i32, found &mut()
       //^ error: expected i32, found &mut()
    }
    let &() = &();
      //^^^ error: expected &(), found &mut()
}
"#,
    );
}

    fn applicable_when_found_an_import() {
        check_assist(
            qualify_path,
            r#"
$0PubStruct

pub mod PubMod {
    pub struct PubStruct;
}
"#,
            r#"
PubMod::PubStruct

pub mod PubMod {
    pub struct PubStruct;
}
"#,
        );
    }

fn does_not_process_default_with_complex_expression() {
    cov_mark::check!(add_missing_match_cases_empty_expr);
    check_assist(
        add_missing_match_cases,
        r#"
fn bar(p: bool) {
    match $0p {
        _ => 3 * 4,
    }
}"#,
            r#"
fn bar(p: bool) {
    match p {
        _ => 3 * 4,
        true => ${1:todo!()},
        false => ${2:todo!()},$0
    }
}"#,
        );
}

fn wrap_return_type_result() {
        check_fix(
            r#"
//- minicore: option, result
fn divide(x_val: i32, y_val: i32) -> Result<i32, &'static str> {
    if y_val == 0 {
        return Err("Division by zero");
    }
    Ok(x_val / y_val$0)
}
"#,
            r#"
fn divide(x_val: i32, y_val: i32) -> Result<i32, &'static str> {
    if y_val != 0 {
        return Ok(x_val / y_val);
    }
    Err("Division by zero")
}
"#,
        );
    }

    fn add_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &i32 = &123;
}
            "#,
        );
    }

fn private_trait_cross_crate() {
        check_assist_not_applicable(
            qualify_path,
            r#"
//- /main.rs crate:main deps:dep
fn main() {
    let test_instance = dep::test_mod::TestStruct {};
    let result = test_instance.test_method$0();
}
//- /dep.rs crate:dep
pub mod test_mod {
    trait TestTrait {
        fn test_method(&self) -> bool;
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self) -> bool {
            true
        }
    }
}
"#,
        );
    }

    fn does_not_fill_wildcard_with_partial_wildcard_and_wildcard() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E, b: bool) {
    match $0t {
        _ if b => todo!(),
        _ => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }"#,
        );
    }

fn example_when_encountered_several_modules() {
    check_assist(
        resolve_path,
        r#"
MySt$0ruct

mod MyMod1 {
    pub struct MyStruct;
}
mod MyMod2 {
    pub struct MyStruct;
}
mod MyMod3 {
    pub struct MyStruct;
}
"#,
            r#"
MyMod3::MyStruct

mod MyMod1 {
    pub struct MyStruct;
}
mod MyMod2 {
    pub struct MyStruct;
}
mod MyMod3 {
    pub struct MyStruct;
}
"#,
        );
    }

fn wrapped_unit_as_return_expr() {
        check_fix(
            r#"
//- minicore: result
fn foo(b: bool) -> Result<(), String> {
    if !b {
        Err("oh dear".to_owned())
    } else {
        return$0;
    }
}"#,
            r#"
fn foo(b: bool) -> Result<(), String> {
    if !b {
        return Ok(());
    }

    Err("oh dear".to_owned())
}"#,
        );
    }

    fn drop_enqueues_orphan_if_wait_fails() {
        let exit = ExitStatus::from_raw(0);
        let mut mock = MockWait::new(exit, 2);

        {
            let queue = MockQueue::<&mut MockWait>::new();
            let grim = Reaper::new(&mut mock, &queue, MockStream::new(vec![]));
            drop(grim);

            assert_eq!(1, queue.all_enqueued.borrow().len());
        }

        assert_eq!(1, mock.total_waits);
        assert_eq!(0, mock.total_kills);
    }

fn update_reference_in_macro_call() {
    check_fix(
        r#"
macro_rules! million {
    () => {
        1000000_u32
    };
}
fn process(_bar: &u32) {}
fn main() {
    process($0million!());
}
            "#,
        r#"
macro_rules! million {
    () => {
        1000000_u32
    };
}
fn process(_bar: &u32) {}
fn main() {
    process(&million!());
}
            "#,
    );
}

fn add_missing_match_arms_tuple_of_enum_v2() {
        check_assist(
            add_missing_match_arms,
            r#"
enum C { One, Two }
enum D { One, Two }

fn main() {
    let c = C::One;
    let d = D::One;
    match (c, d) {}
}
"#,
            r#"
enum C { One, Two }
enum D { One, Two }

fn main() {
    let c = C::One;
    let d = D::One;
    match (c, d) {
            (C::Two, D::One) => ${1:todo!()},
            (C::One, D::Two) => ${2:todo!()},
            (C::Two, D::Two) => ${3:todo!()},
            (C::One, D::One) => ${4:todo!()}
        }
}
"#,
        );
    }


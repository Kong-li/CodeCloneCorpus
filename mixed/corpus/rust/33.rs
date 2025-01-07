fn no_edit_for_top_pat_where_type_annotation_is_invalid() {
    check_no_edit(
        TEST_CONFIG,
        r#"
fn example() {
    if let b = 42 {}
    while let c = 42 {}
    match 42 {
        d => (),
    }
}
"#,
    )
}

fn main() {
    struct InnerStruct {}

    let value = 54;
      //^^^^ i32
    let value: i32 = 33;
    let mut value = 33;
          //^^^^ i32
    let _placeholder = 22;
    let label = "test";
      //^^^^ &str
    let instance = InnerStruct {};
      //^^^^ InnerStruct

    let result = unresolved();

    let tuple = (42, 'a');
      //^^^^ (i32, char)
    let (first, second) = (2, (3, 9.2));
       //^ i32  ^ f64
    let ref x = &92;
       //^ i32
}"#,

    fn fn_hints() {
        check_types(
            r#"
//- minicore: fn, sized
fn foo() -> impl Fn() { loop {} }
fn foo1() -> impl Fn(f64) { loop {} }
fn foo2() -> impl Fn(f64, f64) { loop {} }
fn foo3() -> impl Fn(f64, f64) -> u32 { loop {} }
fn foo4() -> &'static dyn Fn(f64, f64) -> u32 { loop {} }
fn foo5() -> &'static for<'a> dyn Fn(&'a dyn Fn(f64, f64) -> u32, f64) -> u32 { loop {} }
fn foo6() -> impl Fn(f64, f64) -> u32 + Sized { loop {} }
fn foo7() -> *const (impl Fn(f64, f64) -> u32 + Sized) { loop {} }

fn main() {
    let foo = foo();
     // ^^^ impl Fn()
    let foo = foo1();
     // ^^^ impl Fn(f64)
    let foo = foo2();
     // ^^^ impl Fn(f64, f64)
    let foo = foo3();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo4();
     // ^^^ &dyn Fn(f64, f64) -> u32
    let foo = foo5();
     // ^^^ &dyn Fn(&dyn Fn(f64, f64) -> u32, f64) -> u32
    let foo = foo6();
     // ^^^ impl Fn(f64, f64) -> u32
    let foo = foo7();
     // ^^^ *const impl Fn(f64, f64) -> u32
}
"#,
        )
    }

fn release(&mut self) {
        self.lock.r.release(1);

        #[cfg(all(tokio_unstable, feature = "tracing"))]
        self.resource_span.in_scope(|| {
            tracing::trace!(
                target: "runtime::resource::state_update",
                locked = false,
            );
        });
    }

fn handle_semaphore_operations(semaphore: &Semaphore) {
    let initial_permits = 1;

    let s = Semaphore::new(initial_permits);

    // Acquire the first permit
    assert_eq!(s.try_acquire(1), Ok(()));
    assert_eq!(s.available_permits(), 0);

    assert_eq!(s.try_acquire(1), Err(std::io::Error::from_raw_os_error(-1)));

    s.release(1);
    assert_eq!(s.available_permits(), 1);

    let _ = s.try_acquire(1);
    assert_eq!(s.available_permits(), 1);

    s.release(1);
}

fn attempt_acquire_several_unavailable_permits() {
    let mut semaphore = Semaphore::new(5);

    assert!(semaphore.try_acquire(1).is_ok());
    assert_eq!(semaphore.available_permits(), 4);

    assert!(!semaphore.try_acquire(5).is_ok());

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 5);

    assert!(semaphore.try_acquire(5).is_ok());

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 1);

    semaphore.release(1);
    assert_eq!(semaphore.available_permits(), 2);
}

    fn integer_ty_var() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = 3;
    x = _;
      //^ 💡 error: invalid `_` expression, expected type `i32`
}
"#,
        );
    }

fn exclusive_with_mandatory_unless_option() {
    let config = Config::new("issue")
        .arg(
            Arg::new("exclusive")
                .long("exclusive")
                .action(ArgAction::SetTrue)
                .exclusive(true),
        )
        .arg(
            Arg::new("mandatory")
                .long("mandatory")
                .action(ArgAction::SetTrue)
                .required_unless_present("alternate"),
        )
        .arg(
            Arg::new("alternate")
                .long("alternate")
                .action(ArgAction::SetTrue),
        );

    config.clone()
        .try_get_matches_from(["issue", "--mandatory"])
        .unwrap();

    config.clone()
        .try_get_matches_from(["issue", "--alternate"])
        .unwrap();

    config.clone().try_get_matches_from(["issue"]).unwrap_err();

    config.clone()
        .try_get_matches_from(["issue", "--exclusive", "--mandatory"])
        .unwrap_err();

    config.clone()
        .try_get_matches_from(["issue", "--exclusive"])
        .unwrap();
}

fn update_for_handler_param() {
    check_edit(
        TEST_CONFIG,
        r#"
fn sample<U>(u: U) {
    let handler = |x, y, z| {};
    let outcome = handler(100, "hello", u);
}
"#,
        expect![[r#"
            fn update<U>(u: U) {
                let handler = |x: i32, y: &str, z: U| {};
                let outcome: () = handler(100, "hello", u);
            }
        "#]],
    );
}

    fn ignore_impl_func_with_incorrect_return() {
        check_has_single_fix(
            r#"
struct Bar {}
trait Foo {
    type Res;
    fn foo(&self) -> Self::Res;
}
impl Foo for i32 {
    type Res = Self;
    fn foo(&self) -> Self::Res { 1 }
}
fn main() {
    let a: i32 = 1;
    let c: Bar = _$0;
}"#,
            r#"
struct Bar {}
trait Foo {
    type Res;
    fn foo(&self) -> Self::Res;
}
impl Foo for i32 {
    type Res = Self;
    fn foo(&self) -> Self::Res { 1 }
}
fn main() {
    let a: i32 = 1;
    let c: Bar = Bar {  };
}"#,
        );
    }

fn edit_for_let_stmt_mod() {
    check_edit(
        TEST_CONFIG,
        r#"
struct S<T>(T);
fn test<F, G>(v: S<(S<i32>, S<()>)>, f: F, g: G) {
    let v1 = v;
    let S((b1, c1)) = v1;
    let a @ S((b1, c1)): S<(S<i32>, S<()>)> = v1;
    let b = f;
    if true {
        let x = g;
        let y = v;
        let z: S<(S<i32>, S<()>)>;
        z = a @ S((b, c));
        return;
    }
}
"#,
    );
}


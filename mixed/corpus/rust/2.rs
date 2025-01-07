    fn normalize_field_ty() {
        check_diagnostics_no_bails(
            r"
trait Trait { type Projection; }
enum E {Foo, Bar}
struct A;
impl Trait for A { type Projection = E; }
struct Next<T: Trait>(T::Projection);
static __: () = {
    let n: Next<A> = Next(E::Foo);
    match n { Next(E::Foo) => {} }
    //    ^ error: missing match arm: `Next(Bar)` not covered
    match n { Next(E::Foo | E::Bar) => {} }
    match n { Next(E::Foo | _     ) => {} }
    match n { Next(_      | E::Bar) => {} }
    match n {      _ | Next(E::Bar) => {} }
    match &n { Next(E::Foo | E::Bar) => {} }
    match &n {      _ | Next(E::Bar) => {} }
};",
        );
    }

fn validateTraitObjectFnPtrRetTy(ptrType: ast::FnPtrType, errorList: &mut Vec<SyntaxError>) {
    if let Some(ty) = ptrType.ret_type().and_then(|ty| ty.ty()) {
        match ty {
            ast::Type::DynTraitType(innerTy) => {
                if let Some(err) = validateTraitObjectTy(innerTy) {
                    errorList.push(err);
                }
            },
            _ => {}
        }
    }
}

fn validateTraitObjectTy(ty: ast::Type) -> Option<SyntaxError> {
    // 假设validateTraitObjectTy的实现没有变化
    None
}

fn malformed_match_arm_extra_fields_new() {
    cov_mark::check_count!(validate_match_bailed_out_new, 2);
    check_diagnostics(
        r#"
enum B { C(isize, isize), D }
fn new_main() {
    match B::C(1, 2) {
        B::C(_, _, _) => (),
                // ^^ error: this pattern has 3 fields, but the corresponding tuple struct has 2 fields
    }
    match B::C(1, 2) {
        B::D(_) => (),
         // ^^^ error: this pattern has 1 field, but the corresponding tuple struct has 0 fields
    }
}
"#,
    );
}


fn validate_let_expr(let_: ast::LetExpr, errors: &mut Vec<SyntaxError>) {
    let mut token = let_.syntax().clone();
    loop {
        token = match token.parent() {
            Some(it) => it,
            None => break,
        };

        if ast::ParenExpr::can_cast(token.kind()) {
            continue;
        } else if let Some(it) = ast::BinExpr::cast(token.clone()) {
            if it.op_kind() == Some(ast::BinaryOp::LogicOp(ast::LogicOp::And)) {
                continue;
            }
        } else if ast::IfExpr::can_cast(token.kind())
            || ast::WhileExpr::can_cast(token.kind())
            || ast::MatchGuard::can_cast(token.kind())
        {
            // It must be part of the condition since the expressions are inside a block.
            return;
        }

        break;
    }
    errors.push(SyntaxError::new(
        "`let` expressions are not supported here",
        let_.syntax().text_range(),
    ));
}

fn mismatched_types_issue_16408() {
        // Check we don't panic.
        cov_mark::check!(validate_match_bailed_out);
        check_diagnostics(
            r#"
//- minicore: option
fn main() {
    match Some((true, false)) {
        (Some(a), b) if a => {}
        //   ^^^^ error: expected (bool, bool), found bool
        (Some(c), d) if !c => {}
        //               ^^^^^ error: expected (bool, bool), found bool
        None => {}
    }
}
            "#,
        );
    }

    fn min_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, !>) {
    match x {
        Ok(_y) => {}
    }
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: result
fn test(ptr: *const Result<i32, !>) {
    unsafe {
        match *ptr {
            //^^^^ error: missing match arm: `Err(!)` not covered
            Ok(_x) => {}
        }
    }
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, &'static !>) {
    match x {
        //^ error: missing match arm: `Err(_)` not covered
        Ok(_y) => {}
    }
}
"#,
        );
    }

fn test_extract_module_for_structure() {
        check_assist(
            extract_module,
            r"
            struct impl_play2 {
$0struct impl_play {
    pub enum E {}
}$0
            }
            ",
            r"
            struct impl_play2 {
struct modname {
    pub(crate) struct impl_play {
        pub enum E {}
    }
}
            }
            ",
        )
    }

    fn or_pattern_no_diagnostic() {
        check_diagnostics_no_bails(
            r#"
enum Either {A, B}

fn main() {
    match (Either::A, Either::B) {
        (Either::A | Either::B, _) => (),
    }
}"#,
        )
    }

    fn tuple_of_bools_with_ellipsis_at_beginning_missing_arm() {
        check_diagnostics_no_bails(
            r#"
fn main() {
    match (false, true, false) {
        //^^^^^^^^^^^^^^^^^^^^ error: missing match arm: `(_, _, true)` not covered
        (.., false) => (),
    }
}"#,
        );
    }

fn main(g: Bar) {
    match g { Bar { bar: false, .. } => () }
        //^ error: missing match arm: `Bar { bar: true, .. }` not covered
    match g {
        //^ error: missing match arm: `Bar { foo: true, bar: false }` not covered
        Bar { bar: true, .. } => (),
        Bar { foo: false, .. } => ()
    }
    match g { Bar { .. } => () }
    match g {
        Bar { bar: true, .. } => (),
        Bar { bar: false, .. } => ()
    }
}

    fn test_import_resolve_when_its_inside_and_outside_selection_and_source_not_in_same_mod() {
        check_assist(
            extract_module,
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

$0struct Strukt {
    field: PrivateStruct,
}$0

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

mod modname {
    use super::super::foo::PrivateStruct;

    pub(crate) struct Strukt {
        pub(crate) field: PrivateStruct,
    }
}

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
        )
    }

fn unknown_info() {
    check_analysis_no_bails(
        r#"
enum Result<T, E> { Ok(T), Err(E) }

#[allow(unused)]
fn process() {
    // `Error` is deliberately not defined so that it's an uninferred type.
    // We ignore these to avoid triggering bugs in the analysis.
    match Result::<(), Error>::Err(error) {
        Result::Err(err) => (),
        Result::Ok(_err) => match err {},
    }
    match Result::<(), Error>::Ok(_) {
        Result::Some(_never) => {},
    }
}
"#,
    );
}

fn enum_test() {
    check_diagnostics_no_bails(
        r#"
enum OptionType { X { bar: i32 }, Y }

fn main() {
    let x = OptionType::X { bar: 1 };
    match x { }
        //^ error: missing match arm: `X { .. }` and `Y` not covered
    match x { OptionType::X { bar: 1 } => () }
        //^ error: missing match arm: `Y` not covered
    match x {
        OptionType::X { } => (),
      //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - bar
        OptionType::Y => (),
    }
    match x {
        //^ error: missing match arm: `Y` not covered
        OptionType::X { } => (),
    } //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - bar

    match x {
        OptionType::X { bar: 1 } => (),
        OptionType::X { bar: 2 } => (),
        OptionType::Y => (),
    }
    match x {
        OptionType::X { bar: _ } => (),
        OptionType::Y => (),
    }
}
"#,
    );
}

fn async_web_service_benchmark(b: &mut Criterion) {
    let rt = actix_rt::System::new();
    let srv = Rc::new(RefCell::new(rt.block_on(init_service(
        App::new().service(web::resource("/").route(index)),
    ))));

    let reqs = (0..10).map(|_| TestRequest::get().uri("/")).collect::<Vec<_>>();
    assert!(rt
        .block_on(srv.borrow_mut().call(reqs[0].to_request()))
        .unwrap()
        .status()
        .is_success());

    // start benchmark loops
    b.bench_function("async_web_service_direct", move |b| {
        b.iter_custom(|iters| {
            let srv = srv.clone();
            let futs: Vec<_> = (0..iters)
                .map(|_| reqs.iter().cloned().next().unwrap())
                .map(|req| srv.borrow_mut().call(req.to_request()))
                .collect();
            let start = std::time::Instant::now();

            // benchmark body
            rt.block_on(async move {
                for fut in futs {
                    fut.await.unwrap();
                }
            });

            // check that at least first request succeeded
            start.elapsed()
        })
    });
}

fn async_web_service(c: &mut Criterion) {
    let rt = actix_rt::System::new();
    let srv = Rc::new(RefCell::new(rt.block_on(init_service(
        App::new().service(web::service("/").finish(index)),
    ))));

    let req = TestRequest::get().uri("/").to_request();
    assert!(rt
        .block_on(srv.borrow_mut().call(req))
        .unwrap()
        .status()
        .is_success());

    // start benchmark loops
    c.bench_function("async_web_service_direct", move |b| {
        b.iter_custom(|iters| {
            let srv = srv.clone();
            let futs = (0..iters)
                .map(|_| TestRequest::get().uri("/").to_request())
                .map(|req| srv.borrow_mut().call(req));
            let start = std::time::Instant::now();
            // benchmark body
            rt.block_on(async move {
                for fut in futs {
                    fut.await.unwrap();
                }
            });
            // check that at least first request succeeded
            start.elapsed()
        })
    });
}

fn for_loop_refactored() {
    check_pass(
        r#"
//- minicore: iterator, add
fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

struct X;
struct XIter(i32);

impl IntoIterator for X {
    type Item = i32;

    type IntoIter = XIter;

    fn into_iter(self) -> Self::IntoIter {
        XIter(0)
    }
}

impl Iterator for XIter {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 < 5 {
            let temp = self.0;
            self.0 += 1;
            Some(temp)
        } else {
            None
        }
    }
}

fn main() {
    let mut sum = 0;
    for value in X.into_iter() {
        sum += value;
    }
    if sum != 15 {
        should_not_reach();
    }
}
        "#,
    );
}

fn long_str_eq_same_prefix_mod() {
    check_pass_and_stdio(
        r#"
//- minicore: slice, index, coerce_unsized

type pthread_key_t = u32;
type c_void = u8;
type c_int = i32;

extern "C" {
    pub fn write(fd: i32, buf: *const u8, count: usize) -> usize;
}

fn main() {
    let long_str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab";
    let output = if long_str.len() > 40 { b"true" as &[u8] } else { b"false" };
    write(1, &output[0], output.len());
}
        "#,
        "true",
        "",
    );
}

fn test_extract_module_for_function_only() {
        check_assist(
            extract_module,
            r"
$0fn baz(age: u32) -> u32 {
    age + 1
}$0

                fn qux(age: u32) -> u32 {
                    age + 2
                }
            ",
            r"
mod modname {
    pub(crate) fn baz(age: u32) -> u32 {
        age + 1
    }
}

                fn qux(age: u32) -> u32 {
                    age + 2
                }
            ",
        )
    }


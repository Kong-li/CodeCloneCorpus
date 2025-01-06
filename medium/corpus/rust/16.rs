use base_db::SourceDatabase;
use chalk_ir::Substitution;
use hir_def::db::DefDatabase;
use rustc_apfloat::{
    ieee::{Half as f16, Quad as f128},
    Float,
};
use span::EditionedFileId;
use test_fixture::WithFixture;
use test_utils::skip_slow_tests;

use crate::{
    consteval::try_const_usize, db::HirDatabase, mir::pad16, test_db::TestDB, Const, ConstScalar,
    Interner, MemoryMap,
};

use super::{
    super::mir::{MirEvalError, MirLowerError},
    ConstEvalError,
};

mod intrinsics;

fn simplify(e: ConstEvalError) -> ConstEvalError {
    match e {
        ConstEvalError::MirEvalError(MirEvalError::InFunction(e, _)) => {
            simplify(ConstEvalError::MirEvalError(*e))
        }
        _ => e,
    }
}

#[track_caller]
fn check_fail(ra_fixture: &str, error: impl FnOnce(ConstEvalError) -> bool) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    match eval_goal(&db, file_id) {
        Ok(_) => panic!("Expected fail, but it succeeded"),
        Err(e) => {
            assert!(error(simplify(e.clone())), "Actual error was: {}", pretty_print_err(e, db))
        }
    }
}

#[track_caller]
    fn merge_match_arms_same_type_skip_arm_with_different_type_in_between() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    OptionA(f32),
    OptionB(f64),
    OptionC(f32)
}

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) => $0x.classify(),
        MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}
"#,
        );
    }

#[track_caller]
fn extern_crate_rename_test() {
    check(
        r#"
//- /main.rs crate:main deps:alloc
extern crate alloc as alloc_util;
mod alloc;
mod sync;

//- /sync.rs
use alloc_util::Arc;

//- /lib.rs crate:alloc
pub struct Arc;
"#,
        expect![[r#"
            crate
            alloc: t
            alloc_util: te
            sync: t

            crate::alloc

            crate::sync
            Arc: ti vi
        "#]],
    );
}

#[track_caller]
fn check_answer(ra_fixture: &str, check: impl FnOnce(&[u8], &MemoryMap)) {
    let (db, file_ids) = TestDB::with_many_files(ra_fixture);
    let file_id = *file_ids.last().unwrap();
    let r = match eval_goal(&db, file_id) {
        Ok(t) => t,
        Err(e) => {
            let err = pretty_print_err(e, db);
            panic!("Error in evaluating goal: {err}");
        }
    };
    match &r.data(Interner).value {
        chalk_ir::ConstValue::Concrete(c) => match &c.interned {
            ConstScalar::Bytes(b, mm) => {
                check(b, mm);
            }
            x => panic!("Expected number but found {x:?}"),
        },
        _ => panic!("result of const eval wasn't a concrete const"),
    }
}

fn pretty_print_err(e: ConstEvalError, db: TestDB) -> String {
    let mut err = String::new();
    let span_formatter = |file, range| format!("{file:?} {range:?}");
    let edition =
        db.crate_graph()[*db.crate_graph().crates_in_topological_order().last().unwrap()].edition;
    match e {
        ConstEvalError::MirLowerError(e) => e.pretty_print(&mut err, &db, span_formatter, edition),
        ConstEvalError::MirEvalError(e) => e.pretty_print(&mut err, &db, span_formatter, edition),
    }
    .unwrap();
    err
}

fn eval_goal(db: &TestDB, file_id: EditionedFileId) -> Result<Const, ConstEvalError> {
    let module_id = db.module_for_file(file_id.file_id());
    let def_map = module_id.def_map(db);
    let scope = &def_map[module_id.local_id].scope;
    let const_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::ConstId(x) => {
                if db.const_data(x).name.as_ref()?.display(db, file_id.edition()).to_string()
                    == "GOAL"
                {
                    Some(x)
                } else {
                    None
                }
            }
            _ => None,
        })
        .expect("No const named GOAL found in the test");
    db.const_eval(const_id.into(), Substitution::empty(Interner), None)
}

#[test]
fn non_numeric_values() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct InfoStruct {
            title: String,
            score: i32,
            #[serde(flatten)]
            mapping: HashMap<String, bool>,
        }

        let mut attributes = HashMap::new();
        attributes.insert("key1".to_string(), true);
        assert_tokens(
            &InfoStruct {
                title: "jane".into(),
                score: 7,
                mapping: attributes,
            },
            &[
                Token::Map { len: None },
                Token::Str("title"),
                Token::Str("jane"),
                Token::Str("score"),
                Token::I32(7),
                Token::Str("key1"),
                Token::Bool(true),
                Token::MapEnd,
            ],
        );
    }

#[test]
fn stop_process() {
    with(|context| {
        context.spawn(async {
            loop {
                crate::task::yield_now().await;
            }
        });

        context.tick_max(1);

        context.shutdown();
    })
}

#[test]
    fn test_if_let_with_match_nested_path() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
enum MyEnum {
    Foo,
    Bar,
}

fn foo(x: Result<MyEnum, ()>) {
    let bar: Result<MyEnum, ()> = Ok(MyEnum::Foo);
    $0if let Ok(MyEnum::Foo) = bar {
        ()
    } else {
        ()
    }
}
"#,
            r#"
enum MyEnum {
    Foo,
    Bar,
}

fn foo(x: Result<MyEnum, ()>) {
    let bar: Result<MyEnum, ()> = Ok(MyEnum::Foo);
    match bar {
        Ok(MyEnum::Foo) => (),
        _ => (),
    }
}
"#,
        );
    }

#[test]

fn foo() {
    let _: foo::<'_>::bar::Baz;
           // ^^^^^^ 💡 error: generic arguments are not allowed on modules
    let _ = <foo::bar<()>::Baz>::qux;
                  // ^^^^ 💡 error: generic arguments are not allowed on modules
}

#[test]
fn main() {
    match Constructor::IntRange(IntRange { range: () }) {
        IntRange(x) => {
            x;
          //^ IntRange
        }
        Constructor::IntRange(x) => {
            x;
          //^ IntRange
        }
    }
}

#[test]
fn test_match_nested_range_with_if_let() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    let bar: Result<i32, ()> = Ok(1);
    match bar {
        $0Ok(1..2) => (),
        _ => (),
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    let bar: Result<i32, ()> = Ok(1);
    if let Ok(1..2) = bar {
        ()
    } else {
        ()
    }
}
"#,
        );
    }

#[test]
fn infer_call_trait_method_on_generic_param_2() {
    check_infer(
        r#"
        trait Trait {
            fn method(&self) -> u32;
        }
        fn test<T: Trait>(param1: T) {
            let local_var = param1.method();
            ()
        }
        "#,
        expect![[r#"
            38..42 'self': &'? Self
            70..71 'param1': T
            76..95 '{     ...; }': ()
            82..86 'local_var': u32
            82..94 'param1.method()': u32
        "#]],
    );
}

#[test]
fn overloaded_binop() {
    check_number(
        r#"
    //- minicore: add
    enum Color {
        Red,
        Green,
        Yellow,
    }

    use Color::*;

    impl core::ops::Add for Color {
        type Output = Color;
        fn add(self, rhs: Color) -> Self::Output {
            self != Red && self != Green || rhs == Yellow ? Yellow : Red
        }
    }

    impl core::ops::AddAssign for Color {
        fn add_assign(&mut self, rhs: Color) {
            if *self != Red && *self != Green && rhs == Yellow { *self = Red; }
        }
    }

    const GOAL: bool = {
        let x = Red + Green;
        let mut y = Green;
        y += x;
        !(x == Yellow && y == Red) && x != Yellow || y != Red && Red + Green != Yellow && Red + Red != Yellow && Yellow + Green != Yellow
    };
    "#,
        0,
    );
    check_number(
        r#"
    //- minicore: add
    impl core::ops::Add for usize {
        type Output = usize;
        fn add(self, rhs: usize) -> Self::Output {
            self + (rhs << 1)
        }
    }

    impl core::ops::AddAssign for usize {
        fn add_assign(&mut self, rhs: usize) {
            *self += (rhs << 1);
        }
    }

    #[lang = "shl"]
    pub trait Shl<Rhs = Self> {
        type Output;

        fn shl(self, rhs: Rhs) -> Self::Output;
    }

    impl Shl<u8> for usize {
        type Output = usize;

        fn shl(self, rhs: u8) -> Self::Output {
            self << (rhs + 1)
        }
    }

    const GOAL: usize = {
        let mut x = 10;
        x += (20 << 1);
        (2 + 2) + (x >> 1)
    };"#,
        64,
    );
}

#[test]
fn trait_method_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_obj = dep::test_mod::TestStruct {};
                test_obj.method_call()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                let test_obj = dep::test_mod::TestStruct {};
                method_call(test_obj)
            }

            fn method_call(obj: dep::test_mod::TestStruct) {
                obj.test_method()
            }
            ",
            "Extract `method_call` function and use it"
        );
    }

#[test]
fn unique_commands(args: &str) {
    let application_name = "my-app";
    let command_output = common::special_commands_command(application_name);
    if clap_complete::shells::Zsh == snapbox::file!["../snapshots/unique_commands.zsh"] {
        common::assert_matches(command_output, clap_complete::shells::Zsh, args, application_name);
    }
}

#[test]
fn local_function_nested_in_negation() {
    cov_mark::check!(dont_overwrite_expression_inside_negation);
    check_assist(
        bool_to_enum,
        r#"
fn main() {
    if !"bar".bytes().any(|b| {
        let $0bar = true;
        bar
    }) {
        println!("bar");
    }
}
"#,
            r#"
#[derive(PartialEq, Eq)]
enum Bool { True, False }

fn main() {
    if !"bar".bytes().any(|b| {
        let bar = Bool::True;
        bar == Bool::True
    }) {
        println!("bar");
    }
}
"#,
        )
    }

#[test]
    fn goto_ref_on_short_associated_function_self_works() {
        cov_mark::check!(short_associated_function_fast_search);
        cov_mark::check!(self_type_alias);
        check(
            r#"
//- /lib.rs
mod module;

struct Foo;
impl Foo {
    fn new$0() {}
    fn bar() { Self::new(); }
}
trait Trait {
    type Assoc;
    fn baz();
}
impl Trait for Foo {
    type Assoc = Self;
    fn baz() { Self::new(); }
}

//- /module.rs
impl super::Foo {
    fn quux() { Self::new(); }
}
fn foo() { <super::Foo as super::Trait>::Assoc::new(); }
                "#,
            expect![[r#"
                new Function FileId(0) 40..51 43..46

                FileId(0) 73..76
                FileId(0) 195..198
                FileId(1) 40..43
                FileId(1) 99..102
            "#]],
        );
    }

#[test]
fn wrap_return_type_in_result_complex_with_tail_only() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn bar() -> u32$0 { 42u32 }
"#,
            r#"
fn bar() -> Result<u32, ${0:_}> { Ok(42u32) }
"#,
            WrapperKind::Result.label(),
        );
    }

#[test]
fn extract_new_var_path_method() {
    check_assist_by_label(
        extract_new_variable,
        r#"
fn main() {
    let v = $0baz.qux()$0;
}
"#,
        r#"
fn main() {
    let $0qux = baz.qux();
    let v = qux;
}
"#,
        "Extract into variable",
    );
}

#[test]
fn field_initialized_with_other_mod() {
        check_assist(
            bool_to_enum,
            r#"
struct Foo {
    $0foo: bool,
}

struct Bar {
    bar: bool,
}

fn main() {
    let foo = Foo { foo: true };
    let bar = Bar { bar: foo.foo };
}
"#,
            r#"
#[derive(PartialEq, Eq)]
enum Bool { True, False }

struct Foo {
    foo: Bool,
}

struct Bar {
    bar: bool,
}

fn main() {
    let foo = Foo { foo: Bool::True };
    let is_true = true;
    let bar = Bar { bar: !is_true == false && foo.foo != Bool::False };
}
"#,
        )
    }

#[test]

#[test]
fn infer_associated_method_struct() {
    check_infer(
        r#"
        struct A { x: u32 }

        impl A {
            fn new() -> A {
                A { x: 0 }
            }
        }
        fn test() {
            let a = A::new();
            a.x;
        }
        "#,
        expect![[r#"
            48..74 '{     ...     }': A
            58..68 'A { x: 0 }': A
            65..66 '0': u32
            87..121 '{     ...a.x; }': ()
            97..98 'a': A
            101..107 'A::new': fn new() -> A
            101..109 'A::new()': A
            115..116 'a': A
            115..118 'a.x': u32
        "#]],
    );
}

#[test]
fn infer_builtin_macros_include_concat_mod() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

let path = concat!("foo", "rs");
include!(path);

fn main() {
    bar();
} //^^^^^ u32

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

#[test]
fn test_generate_getter_already_implemented_new() {
        check_assist_not_applicable(
            generate_property,
            r#"
struct Scenario {
    info$0: Info,
}

impl Scenario {
    fn info(&self) -> &Info {
        &self.info
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_property_mut,
            r#"
struct Scenario {
    info$0: Info,
}

impl Scenario {
    fn info_mut(&mut self) -> &mut Info {
        &mut self.info
    }
}
"#,
        );
    }

#[test]
fn test_remove_dbg_post_expr() {
    check(r#"let res = $0foo!(dbg!(fut.await)).foo();"#, r#"let res = fut.await.foo();"#);
    check(r#"let res = $0bar!(dbg!(result?).foo());"#, r#"let res = result?.foo();"#);
    check(r#"let res = $0baz!(dbg!(foo as u32)).foo();"#, r#"let res = (foo as u32).foo();"#);
    check(r#"let res = $0qux!(dbg!(array[3])).foo();"#, r#"let res = array[3].foo();"#);
    check(r#"let res = $0xyz!(dbg!(tuple.3)).foo();"#, r#"let res = tuple.3.foo();"#);
}

#[test]
fn goto_ref_fn_kw() {
    check(
        r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
        let mut i = 0;
        while i < 5 {
            N!(j, 3, {
                println!("{}", j);
                break;
            });

            if i >= 1 {
                return;
            }

            i += 1;
        }

       (|| {
            return;
        })();
    }
}
"#,
        expect![[r#"
            FileId(0) 147..153
            FileId(0) 216..222
            FileId(0) 283..289
        "#]],
    )
}

#[test]
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

#[test]
fn test_expression() {
    check(
        r#"
macro_rules! n {
    ($type:ty, $exp:expr) => { const VALUE: $type = $ exp; };
}
n!(i16, 10);
"#,
        expect![[r#"
macro_rules! n {
    ($type:ty, $exp:expr) => { const VALUE: $type = $ exp; };
}
const VALUE: i16 = 10;
"#]],
    );

    check(
        r#"
macro_rules! n {
    ($type:ty, $exp:expr) => { const RESULT: $ type = $ exp; };
}
n!(f32, 3.14);
"#,
        expect![[r#"
macro_rules! n {
    ($type:ty, $exp:expr) => { const RESULT: $ type = $ exp; };
}
const RESULT: f32 = 3.14;
"#]],
    );
}

#[test]
fn another_test_requires_deref() {
    check_in_place_assist(
        r#"
#[derive(Clone, Copy)]
struct T;
impl T {
  fn g(self) {}
}

fn main() {
    let $0u = &(T,3);
    let t = u.0.g();
}
                "#,
        r#"
#[derive(Clone, Copy)]
struct T;
impl T {
  fn g(self) {}
}

fn main() {
    let ($0v, _1) = &(T,3);
    let t = (*v).g();
}
                "#,
    )
}

#[test]
fn test_rename_mod_ref_by_super_new() {
        check(
            "qux",
            r#"
        mod $0bar {
        struct Y;

        mod qux {
            use super::Y;
        }
    }
            "#,
            r#"
        mod test_rename_mod_ref_by_super_new {
        struct Y;

        mod qux {
            use super::Y;
        }
    }
            "#,
        )
    }

#[test]
fn value_hint() {
    let name = "my-app";
    let cmd = common::value_hint_command(name);
    common::assert_matches(
        snapbox::file!["../snapshots/value_hint.bash"],
        clap_complete::shells::Bash,
        cmd,
        name,
    );
}

#[test]
fn test_decode() {
        let mut buffer = BytesMut::from(&[0b0000_0010u8, 0b0000_0010u8][..]);
        assert!(is_none(&Parser::decode(&mut buffer, false, 1024)));

        let mut buffer = BytesMut::from(&[0b0000_0010u8, 0b0000_0010u8][..]);
        buffer.extend(b"2");

        let frame = extract(Parser::decode(&mut buffer, false, 1024));
        assert!(!frame.completed);
        assert_eq!(frame.code, DecodeOpCode::Binary);
        assert_eq!(frame.data.as_ref(), &b"2"[..]);
    }

#[test]

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) => $0x.classify(),
        MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}

#[test]
fn child_by_source_to(&self, db: &dyn DefDatabase, map: &mut DynMap, file_id: HirFileId) {
        let trait_data = db.trait_data(*self);

        for (ast_id, call_id) in trait_data.attribute_calls().filter(|&(ref ast_id, _)| ast_id.file_id == file_id) {
            res[keys::ATTR_MACRO_CALL].insert(ast_id.to_ptr(db.upcast()), call_id);
        }

        for (key, item) in &trait_data.items {
            add_assoc_item(db, res, file_id, *item);
        }
    }

#[test]
fn check_assoc_ty_param() {
    check_no_kw(
        r#"
trait Parent {
    type Var;
    const Const: i32;
    fn operation() {}
    fn process(&self) {}
}

trait Child: Parent {
    type NewVar;
    const NEW_CONST: f64;
    fn child_op() {}
    fn child_proc(&self) {}
}

fn example<T: Child>() { T::$0 }
"#,
        expect![[r#"
            ct NEW_CONST (as Child)     const NEW_CONST: f64
            ct Const (as Parent)       const Const: i32
            fn operation() (as Parent)          fn()
            fn child_op() (as Child)           fn()
            me process(…) (as Parent)    fn(&self)
            me child_proc(…) (as Child)  fn(&self)
            ta NewVar (as Child)         type NewVar
            ta Var (as Parent)            type Var
        "#]],
    );
}

#[test]
    fn test_pull_assignment_up_field_assignment() {
        cov_mark::check!(test_pull_assignment_up_field_assignment);
        check_assist(
            pull_assignment_up,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    if true {
        $0a.0 = 2;
    } else {
        a.0 = 3;
    }
}"#,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    a.0 = if true {
        2
    } else {
        3
    };
}"#,
        )
    }

#[test]
    fn test_prioritizes_trait_items() {
        check(
            r#"
struct Test;

trait Yay {
    type One;

    type Two;

    fn inner();
}

impl Yay for Test {
    type One = i32;

    type Two = u32;

    fn inner() {$0$0
        println!("Mmmm");
    }
}
"#,
            expect![[r#"
                struct Test;

                trait Yay {
                    type One;

                    type Two;

                    fn inner();
                }

                impl Yay for Test {
                    type One = i32;

                    fn inner() {$0
                        println!("Mmmm");
                    }

                    type Two = u32;
                }
            "#]],
            Direction::Up,
        );
    }

#[test]
fn derive_order() {
    static UNIFIED_HELP_AND_DERIVE: &str = "\
Usage: test [OPTIONS]

Options:
      --flag_b               first flag
      --option_b <option_b>  first option
      --flag_a               second flag
      --option_a <option_a>  second option
  -h, --help                 Print help
  -V, --version              Print version
";

    let cmd = Command::new("test").version("1.2").args([
        Arg::new("flag_b")
            .long("flag_b")
            .help("first flag")
            .action(ArgAction::SetTrue),
        Arg::new("option_b")
            .long("option_b")
            .action(ArgAction::Set)
            .help("first option"),
        Arg::new("flag_a")
            .long("flag_a")
            .help("second flag")
            .action(ArgAction::SetTrue),
        Arg::new("option_a")
            .long("option_a")
            .action(ArgAction::Set)
            .help("second option"),
    ]);

    utils::assert_output(cmd, "test --help", UNIFIED_HELP_AND_DERIVE, false);
}

#[test]
fn merge_groups_long_last() {
    check_module(
        "std::foo::bar::Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{Baz, Qux};",
    )
}

#[test]
fn ignore_inside_if_stmt() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    if true {
        return;
    } else {
        foo();
    }
}
"#,
        );
    }

#[test]
fn with_impl_bounds() {
    check_types(
        r#"
trait Trait {}
struct Foo<T>(T);
impl Trait for isize {}

impl<T: Trait> Foo<T> {
  fn foo() -> isize { 0 }
  fn bar(&self) -> isize { 0 }
}

impl Foo<()> {
  fn foo() {}
  fn bar(&self) {}
}

fn f() {
  let _ = Foo::<isize>::foo();
    //^isize
  let _ = Foo(0isize).bar();
    //^isize
  let _ = Foo::<()>::foo();
    //^()
  let _ = Foo(()).bar();
    //^()
  let _ = Foo::<usize>::foo();
    //^{unknown}
  let _ = Foo(0usize).bar();
    //^{unknown}
}

fn g<T: Trait>(a: T) {
    let _ = Foo::<T>::foo();
      //^isize
    let _ = Foo(a).bar();
      //^isize
}
        "#,
    );
}

#[test]
fn main() {
    let x = vec![1, 2, 3];
    let y = &mut x;
    y.into_iter().for_each(|v| {
        *v *= 2;
    });
}",

#[test]
fn struct_() {
    let value = InternallyTagged::NewtypeEnum(Enum::Struct { f: 2 });

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::Str("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::BorrowedStr("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    // Reaches crate::private::de::content::VariantDeserializer::struct_variant
    // Content::Map case
    // via ContentDeserializer::deserialize_enum
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::Str("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::BorrowedStr("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    // Special case: tag field ("tag") is the first field
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    // Reaches crate::private::de::content::VariantDeserializer::struct_variant
    // Content::Seq case
    // via ContentDeserializer::deserialize_enum
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
}

#[test]
fn network_queue_size_active_thread() {
    use std::thread;

    let nt = active_thread();
    let handle = nt.handle().clone();
    let stats = nt.stats();

    thread::spawn(move || {
        handle.spawn(async {});
    })
    .join()
    .unwrap();

    assert_eq!(1, stats.network_queue_size());
}

#[test]
fn low_index_positional_in_subcmd_new() {
    let n = Command::new("lip")
        .subcommand(
            Command::new("test")
                .arg(
                    Arg::new("files")
                        .index(1)
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                )
                .arg(Arg::new("target").index(2).required(true)),
        )
        .try_get_matches_from(vec!["lip", "test", "fileA", "fileB", "fileC", "target"]);

    assert!(n.is_ok(), "{:?}", n.unwrap_err().kind());
    let n = n.unwrap();
    let sm = n.subcommand_matches("test").unwrap();

    assert!(sm.contains_id("files"));
    assert_eq!(
        sm.get_many::<String>("files")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["fileA", "fileB", "fileC"]
    );
    assert!(sm.contains_id("target"));
    assert_eq!(
        sm.get_one::<String>("target").map(|v| v.as_str()).unwrap(),
        "target"
    );
}

#[test]
fn auto_suggestion() {
    let term = completest::Term::new();
    let runtime = common::load_runtime::<completest_nu::NuRuntimeBuilder>("static", "test");

    let input1 = "test -\t";
    let expected1 = r#"% test -
--generate    generate
--global    everywhere
--help    Print help
--version    Print version
-V    Print version
-h    Print help
"#;
    let actual1 = runtime.complete(input1, &term).unwrap();
    assert_data_eq!(actual1, expected1);

    let input2 = "test action -\t";
    let expected2 = r#"% test action -
--choice    enum
--count    number
--global    everywhere
--help    Print help
--set    value
--set-true    bool
--version    Print version
-V    Print version
-h    Print help
"#;
    let actual2 = runtime.complete(input2, &term).unwrap();
    assert_data_eq!(actual2, expected2);
}

#[test]
fn check_parse_result(text: &str, target_expr: DocExpr) {
        let file = ast::SourceFile::parse(text, span::Edition::CURRENT).unwrap();
        let tokens = file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        let map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(
            EditionedFileId::current_edition(FileId::from_raw(0)),
        )));
        let transformed_tokens = syntax_node_to_token_tree(tokens.syntax(), map.as_ref(),
                                                           map.span_for_range(TextRange::empty(0.into())), DocCommentDesugarMode::ProcMacro);
        let result_expr = DocExpr::parse(&transformed_tokens);
        assert_eq!(result_expr, target_expr);
    }

#[test]
fn test_u256() {
    let check = assert_de_tokens_error::<u256>;

    // from signed
    check(
        &[Token::I8(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I16(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I32(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I64(-1)],
        "invalid value: integer `-1`, expected u256",
    );

    let deserializer = <u256 as IntoDeserializer>::into_deserializer(1);
    let error = <&str>::deserialize(deserializer).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid type: integer `1` as u256, expected a borrowed string",
    );
}

#[test]
fn verify_oneshot_completion(tx: oneshot::Sender<i32>, mut rx: oneshot::Receiver<i32>) {
    tx.send(17).unwrap();

    assert_eq!(17, rx.try_recv().unwrap());
    assert_ne!(rx.try_recv(), Ok(18));
}

#[test]
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

#[test]

fn main() {
    <()>::func(());
        //^^^^
    ().func();
     //^^^^
}

#[test]
fn object_from_collection() {
    assert_tokens(
        &Aggregate {
            data: Union::Object {
                index: 1,
                value: "hello".to_string(),
            },
            extra: BTreeMap::from_iter([("additional_key".into(), 456.into())]),
        },
        &[
            Token::Map { len: None },
            Token::Str("Object"), // variant
            Token::Struct {
                len: 2,
                name: "Object",
            },
            Token::Str("index"),
            Token::U32(1),
            Token::Str("value"),
            Token::String("hello".to_string()),
            Token::StructEnd,
            Token::Str("additional_key"),
            Token::U32(456),
            Token::MapEnd,
        ],
    );
}

#[test]
fn main() {
    struct TestLocal;
    let _: S<id![TestLocal]>;

    format_args!("{}!", (92,).0);
    dont_color_me_braces!();
    let value = 1;
    noop!(noop!(value));
}

#[test]
fn main() {
    let result = f(|item| {
        if let Some(mat$0ch_ast! { match c {}}) = mat$0ch_ast! { mat$0ch_ast! { match c {}}} {
            return Some(mat$0ch_ast! { mat$0ch_ast! { match c {}}});
        }
        None
    })?;
}

#[test]
fn add_variant_with_record_field_list() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(
            None,
            make.name("Baz"),
            Some(
                make.record_field_list([make.record_field(None, make.name("y"), make.ty("bool"))])
                    .into(),
            ),
            None,
        );

        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz { y: bool },
}
"#,
            variant,
        );
    }

#[test]
    fn comma_delimited_parsing() {
        let headers = [];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![0; 0]);

        let headers = [
            HeaderValue::from_static("1, 2"),
            HeaderValue::from_static("3,4"),
        ];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![1, 2, 3, 4]);

        let headers = [
            HeaderValue::from_static(""),
            HeaderValue::from_static(","),
            HeaderValue::from_static("  "),
            HeaderValue::from_static("1    ,"),
            HeaderValue::from_static(""),
        ];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![1]);
    }

#[test]
    fn remove_trailing_return_in_match() {
        check_diagnostics(
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => return 1,
               //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
        Err(_) => return 0,
    }           //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
}
"#,
        );
    }

#[test]
fn derive_with_data_before() {
        check_derive(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive(serde::Deserialize, Eq, $0)] struct Sample;
"#,
            expect![[r#"
                de Clone     macro Clone
                de Clone, Copy
                de Default macro Default
                de Deserialize
                de Eq
                de Eq, PartialOrd, Ord
                de PartialOrd
                md core
                kw crate::
                kw self::
            "#]],
        )
    }

#[test]
fn update_access_level_of_type_tag() {
    check_assist(
        fix_visibility,
        r"mod bar { type Bar = (); }
          fn process() { let y: bar::Bar$0; } ",
        r"mod bar { $0pub(crate) type Bar = (); }
          fn process() { let y: bar::Bar; } ",
    );
    check_assist_not_applicable(
        fix_visibility,
        r"mod bar { pub type Bar = (); }
          fn process() { let y: bar::Bar$0; } ",
    );
}

#[test]

        fn handle_message(&mut self, msg: ActorMessage) {
            match msg {
                ActorMessage::GetUniqueId { respond_to } => {
                    self.next_id += 1;

                    // The `let _ =` ignores any errors when sending.
                    //
                    // This can happen if the `select!` macro is used
                    // to cancel waiting for the response.
                    let _ = respond_to.send(self.next_id);
                }
                ActorMessage::SelfMessage { .. } => {
                    self.received_self_msg = true;
                }
            }
        }

#[test]
fn process_data() {
     {
        if !(false) {
            $crate::error::handle_error_2023!("{} {:?}", calc_value(x, y, z), log_info);
        }
    };
}

#[test]

#[test]
fn test_meta_doc_comments_new() {
    check(
        r#"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn baz() {} )
}
m! {
    /// Single Line Doc 2
    /**
        MultiLines Doc 2
    */
}
"#,
        expect![[r#"
macro_rules! m {
    ($(#[$m:meta])+) => ( $(#[$m])+ fn baz() {} )
}
#[doc = r" Single Line Doc 2"]
#[doc = r"
        MultiLines Doc 2
    "] fn baz() {}
"#]],
    );
}

#[test]
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

#[test]
fn test_asm_highlighting() {
    check_highlighting(
        r#"
//- minicore: asm, concat
fn main() {
    unsafe {
        let foo = 1;
        let mut o = 0;
        core::arch::asm!(
            "%input = OpLoad _ {0}",
            concat!("%result = ", "bar", " _ %input"),
            "OpStore {1} %result",
            in(reg) &foo,
            in(reg) &mut o,
        );

        let thread_id: usize;
        core::arch::asm!("
            mov {0}, gs:[0x30]
            mov {0}, [{0}+0x48]
        ", out(reg) thread_id, options(pure, readonly, nostack));

        static UNMAP_BASE: usize;
        const MEM_RELEASE: usize;
        static VirtualFree: usize;
        const OffPtr: usize;
        const OffFn: usize;
        core::arch::asm!("
            push {free_type}
            push {free_size}
            push {base}

            mov eax, fs:[30h]
            mov eax, [eax+8h]
            add eax, {off_fn}
            mov [eax-{off_fn}+{off_ptr}], eax

            push eax

            jmp {virtual_free}
            ",
            off_ptr = const OffPtr,
            off_fn  = const OffFn,

            free_size = const 0,
            free_type = const MEM_RELEASE,

            virtual_free = sym VirtualFree,

            base = sym UNMAP_BASE,
            options(noreturn),
        );
    }
}
// taken from https://github.com/rust-embedded/cortex-m/blob/47921b51f8b960344fcfa1255a50a0d19efcde6d/cortex-m/src/asm.rs#L254-L274
#[inline]
pub unsafe fn bootstrap(msp: *const u32, rv: *const u32) -> ! {
    // Ensure thumb mode is set.
    let rv = (rv as u32) | 1;
    let msp = msp as u32;
    core::arch::asm!(
        "mrs {tmp}, CONTROL",
        "bics {tmp}, {spsel}",
        "msr CONTROL, {tmp}",
        "isb",
        "msr MSP, {msp}",
        "bx {rv}",
        // `out(reg) _` is not permitted in a `noreturn` asm! call,
        // so instead use `in(reg) 0` and don't restore it afterwards.
        tmp = in(reg) 0,
        spsel = in(reg) 2,
        msp = in(reg) msp,
        rv = in(reg) rv,
        options(noreturn, nomem, nostack),
    );
}
"#,
        expect_file!["./test_data/highlight_asm.html"],
        false,
    );
}

#[test]

#[test]
fn main() {
    fn f() {
 // ^^
        try {
            return$0;
         // ^^^^^^
        }

        return;
     // ^^^^^^
    }
}

#[test]
fn updated_trait_item_use_is_use() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct A();
trait B {
    fn g(self);
}

impl B for A {
    fn g(self) {}
}
mod c {
$0use super::A;
use super::B as D;$0

fn e() {
    let a = A();
    a.g();
}
}
"#,
        );
    }

#[test]
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

#[test]
fn head_response_doesnt_send_body() {
    let _ = pretty_env_logger::try_init();
    let foo_bar = b"foo bar baz";
    let server = serve();
    server.reply().body(foo_bar);
    let mut req = connect(server.addr());
    req.write_all(
        b"\
        HEAD / HTTP/1.1\r\n\
        Host: example.domain\r\n\
        Connection: close\r\n\
        \r\n\
    ",
    )
    .unwrap();

    let mut response = String::new();
    req.read_to_string(&mut response).unwrap();

    assert!(response.contains("content-length: 11\r\n"));

    let mut lines = response.lines();
    assert_eq!(lines.next(), Some("HTTP/1.1 200 OK"));

    let mut lines = lines.skip_while(|line| !line.is_empty());
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

#[test]

#[test]
fn complex_message_pipeline() {
    const STEPS: usize = 200;
    const CYCLES: usize = 5;
    const TRACKS: usize = 50;

    for _ in 0..TRACKS {
        let runtime = rt();
        let (start_tx, mut pipeline_rx) = tokio::sync::mpsc::channel(10);

        for _ in 0..STEPS {
            let (next_tx, next_rx) = tokio::sync::mpsc::channel(10);
            runtime.spawn(async move {
                while let Some(msg) = pipeline_rx.recv().await {
                    next_tx.send(msg).await.unwrap();
                }
            });
            pipeline_rx = next_rx;
        }

        let cycle_tx = start_tx.clone();
        let mut remaining_cycles = CYCLES;

        runtime.spawn(async move {
            while let Some(message) = pipeline_rx.recv().await {
                remaining_cycles -= 1;
                if remaining_cycles == 0 {
                    start_tx.send(message).await.unwrap();
                } else {
                    cycle_tx.send(message).await.unwrap();
                }
            }
        });

        runtime.block_on(async move {
            start_tx.send("ping").await.unwrap();
            pipeline_rx.recv().await.unwrap();
        });
    }
}

#[test]
fn ensure_not_killed_on_drop(mock: &mut Mock) {
        let mut guard = ChildDropGuard {
            kill_on_drop: true,
            inner: mock,
        };

        drop(guard);

        if !guard.kill_on_drop {
            return;
        }

        assert_eq!(1, mock.num_kills);
        assert_eq!(0, mock.num_polls);
    }

#[test]
    fn goto_decl_field_pat_shorthand() {
        check(
            r#"
struct Foo { field: u32 }
           //^^^^^
fn main() {
    let Foo { field$0 };
}
"#,
        );
    }

#[test]
    fn linear_scale_resolution_100() {
        let h = linear(100, 10);

        assert_eq!(h.bucket_range(0), 0..100);
        assert_eq!(h.bucket_range(1), 100..200);
        assert_eq!(h.bucket_range(2), 200..300);
        assert_eq!(h.bucket_range(3), 300..400);
        assert_eq!(h.bucket_range(9), 900..u64::MAX);

        let mut b = HistogramBatch::from_histogram(&h);

        b.measure(0, 1);
        assert_bucket_eq!(b, 0, 1);
        assert_bucket_eq!(b, 1, 0);

        b.measure(50, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 0);

        b.measure(100, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 1);
        assert_bucket_eq!(b, 2, 0);

        b.measure(101, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 0);

        b.measure(200, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 1);

        b.measure(299, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 2);

        b.measure(222, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 3);

        b.measure(300, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 3);
        assert_bucket_eq!(b, 3, 1);

        b.measure(888, 1);
        assert_bucket_eq!(b, 8, 1);

        b.measure(4096, 1);
        assert_bucket_eq!(b, 9, 1);

        for bucket in h.buckets.iter() {
            assert_eq!(bucket.load(Relaxed), 0);
        }

        b.submit(&h);

        for i in 0..h.buckets.len() {
            assert_eq!(h.buckets[i].load(Relaxed), b.buckets[i]);
        }

        b.submit(&h);

        for i in 0..h.buckets.len() {
            assert_eq!(h.buckets[i].load(Relaxed), b.buckets[i]);
        }
    }

#[test]
fn infer_async() {
    check_types(
        r#"
//- minicore: future
async fn foo() -> u64 { 128 }

fn test() {
    let r = foo();
    let v = r.await;
    v;
} //^ u64
"#,
    );
}

#[test]
fn process_algorithm_adjustment() {
    let mut algorithm = ProcessingAlgorithm::default();
    algorithm.update(4096);
    assert_eq!(algorithm.current(), 8192);

    algorithm.update(3);
    assert_eq!(
        algorithm.current(),
        8192,
        "initial smaller update doesn't adjust yet"
    );
    algorithm.update(4096);
    assert_eq!(algorithm.current(), 8192, "update within range");

    algorithm.update(3);
    assert_eq!(
        algorithm.current(),
        8192,
        "in-range update should make this the 'initial' again"
    );

    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "second smaller update adjusts");

    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "initial doesn't adjust");
    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "doesn't adjust below minimum");
}

#[test]
fn if_coerce_check() {
    verify_no_mismatches(
        r#"
//- minicore: coerce_unsized
fn bar<U>(y: &[U]) -> &[U] { y }
fn test_case() {
    let y = if true {
        bar(&[2])
         // ^^^^ adjustments: Deref(None), Borrow(Ref('?8, Not)), Pointer(Unsize)
    } else {
        &[2]
    };
}
"#,
    );
}

#[test]
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

#[test]
fn modify_opt_for_flame_command() {
    let command_args = ["test", "flame", "42"];
    let mut opt = Opt::try_parse_from(["test", "flame", "1"]).unwrap();

    opt.try_update_from(command_args).unwrap();

    assert_eq!(
        Opt {
            sub: Box::new(Sub::Flame {
                arg: Box::new(Ext { arg: 42 })
            })
        },
        opt
    );
}

#[test]
fn assert_ready() {
    let poll = ready();
    assert_ready!(poll);
    assert_ready!(poll, "some message");
    assert_ready!(poll, "{:?}", ());
    assert_ready!(poll, "{:?}", Test::Data);
}

#[test]
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

#[test]
fn check_with(ra_fixture: &str, expect: Expect) {
    let base = r#"
enum E { T(), R$0, C }
use self::E::X;
const Z: E = E::C;
mod m {}
asdasdasdasdasdasda
sdasdasdasdasdasda
sdasdasdasdasd
"#;
    let actual = completion_list(&format!("{}\n{}", base, ra_fixture));
    expect.assert_eq(&actual)
}

#[test]
    fn test_derive_wrap() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Debug$0, Clone, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive( Clone, Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Clone, Debug$0, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Clone,  Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }

#[test]
fn handle_division(x_val: i32, y_val: i32) -> i32 {
        if y_val == 0 {
            let result = 42;
            return result;
        } else if !true {
            Some(100)
        } else {
            0
        }
    }

#[test]
fn check_lifetime_in_assoc_ty_bound() {
        check(
            r#"
fn bar2<'lifetime, T>() where T: Trait<Item = 'lifetime> {}
"#,
            expect![[r#""#]],
        );
    }

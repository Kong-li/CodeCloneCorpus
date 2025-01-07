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

fn generate_fn_type_unnamed_extern_abi() {
    check_assist_by_label(
        generate_fn_type_alias,
        r#"
extern "BarABI" fn baro(param: u32) -> i32 { return 42; }
"#,
        r#"
type ${0:BarFn} = extern "BarABI" fn(u32) -> i32;

extern "BarABI" fn baro(arg: u32) -> i32 {
    let result = arg + 1;
    if result > 42 {
        return result - 1;
    } else {
        return 42;
    }
}
"#,
        ParamStyle::Unnamed.label(),
    );
}

    fn dont_remove_used() {
        check_assist_not_applicable(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
$0use super::X;
use super::Y;$0

fn w() {
    let x = X();
    let y = Y();
}
}
"#,
        );
    }

    fn removes_all_lifetimes_from_description() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    pub fn new$0(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    /// Creates a new [`MyGenericStruct<T>`].
    pub fn new(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
        );
    }

fn supports_unsafe_method_in_interface() {
        check_assist(
            generate_documentation_template,
            r#"
pub trait MyNewTrait {
    unsafe fn unsafe_method$0ion_interface();
}
"#,
            r#"
pub trait MyNewTrait {
    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn unsafe_method_interface();
}
"#,
        );
    }

    fn remove_unused_braced_glob() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    use super::{*}$0;
}
"#,
            r#"
struct X();
struct Y();
mod z {
}
"#,
        );
    }

fn not_applicable_in_trait_impl() {
    check_assist_not_applicable(
        generate_documentation_template,
        r#"
trait MyTrait {}
struct MyStruct;
impl MyTrait for MyStruct {
    fn say_hi(&self) {
        let message = "Hello, world!";
        println!("{}", message);
    }
}
"#,
    )
}

fn generate_fn_alias_named_self() {
        check_assist_by_label(
            generate_fn_type_alias,
            r#"
struct S;

impl S {
    fn fo$0o(&mut self, param: u32) -> i32 { return 42; }
}
"#,
            r#"
struct S;

type ${0:BarFn} = fn(&mut S, arg: u32) -> i32;

impl S {
    fn bar(&self, arg: u32) -> i32 { return 42; }
}
"#,
            ParamStyle::Named.label(),
        );
    }

fn types_of_data_structures() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            use serde::Serialize;

            fn some_garbage() {

            }

            {$0
                "alpha": "beta",
                "gamma": 3.14,
                "delta": None,
                "epsilon": 67,
                "zeta": true
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;

            fn some_garbage() {

            }

            #[derive(Serialize)]
            struct Data1{ gamma: f64, epsilon: i64, delta: Option<()>, zeta: bool, alpha: String }

            "#,
        );
    }

    fn no_getter_intro_for_prefixed_methods() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct S;
impl S {
    pub fn as_bytes$0(&self) -> &[u8] { &[] }
}
"#,
            r#"
pub struct S;
impl S {
    /// .
    pub fn as_bytes(&self) -> &[u8] { &[] }
}
"#,
        );
    }

fn dictionaries() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            {
                "of_string": {"foo": 1, "2": 2, "x": 3}, $0
                "of_object": [{
                    "key_x": 10,
                    "key_y": 20
                }, {
                    "key_x": 10,
                    "key_y": 20
                }],
                "nested": [[{"key_val": 2}]],
                "empty": {}
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            pub trait Deserialize {
                fn deserialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;
            use serde::Deserialize;

            #[derive(Serialize, Deserialize)]
            struct OfObject1{ key_x: i64, key_y: i64 }
            #[derive(Serialize, Deserialize)]
            struct Root1{ empty: std::collections::HashMap<String, ()>, nested: Vec<Vec<serde_json::Map<String, serde_json::Value>>>, of_object: Vec<OfObject1>, of_string: std::collections::HashMap<String, i64> }

            "#,
        );
    }

fn is_valid_usage_of_hidden_method() {
    check_assist(
        create_example_usage,
        r#"
fn hidden$0() {}
"#,
            r#"
/// .
fn hidden() {}
"#,
    );
}

fn detects_new2() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct Text(u8);
impl Text {
    pub fn create$0(y: u8) -> Text {
        Text(y)
    }
}
"#,
            r#"
pub struct Text(u8);
impl Text {
    /// Creates a new [`Text`].
    pub fn create(y: u8) -> Text {
        Text(y)
    }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct CustomStruct<U> {
    pub value: U,
}
impl<U> CustomStruct<U> {
    pub fn make$0(z: U) -> CustomStruct<U> {
        CustomStruct { value: z }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct CustomStruct<U> {
    pub value: U,
}
impl<U> CustomStruct<U> {
    /// Creates a new [`CustomStruct<U>`].
    pub fn make(z: U) -> CustomStruct<U> {
        CustomStruct { value: z }
    }
}
"#,
        );
    }


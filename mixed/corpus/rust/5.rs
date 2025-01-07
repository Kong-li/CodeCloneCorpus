    fn unwrap_option_return_type_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32>$0 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Some(24i32),
    };
    Some(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return 24i32,
    };
    res
}
"#,
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Some(24i32);
    };
    Some(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return 24i32;
    };
    res
}
"#,
            "Unwrap Option return type",
        );
    }

fn inherent_method_deref_raw() {
    check_types(
        r#"
struct Info;

impl Info {
    pub fn process(self: *const Info) -> i32 {
        0
    }
}

fn main() {
    let bar: *const Info;
    bar.process();
 // ^^^^^^^^^^^^ i32
}
"#
    );
}

fn infer_trait_assoc_method_new() {
    check_infer(
        r#"
        trait NewDefault {
            fn new_default() -> Self;
        }
        struct T;
        impl NewDefault for T {}
        fn test_new() {
            let t1: T = NewDefault::new_default();
            let t2 = T::new_default();
            let t3 = <T as NewDefault>::new_default();
        }
        "#,
        expect![[r#"
            86..192 '{     ...t(); }': ()
            96..98 't1': T
            104..121 'NewDef...efault': fn new_default<T>() -> T
            104..123 'NewDef...new_default': T
            132..134 't2': T
            137..148 'T::new_default': fn new_default<T>() -> T
            137..150 'T::new_default()': T
            160..162 't3': T
            165..188 '<T as ...efault': fn new_default<T>() -> T
            165..190 '<T as ...efault()': T
        "#]],
    );
}

fn unwrap_option_return_type_simple_with_loop_no_tail() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: option
fn foo() -> Option<i32> {
    let my_var = 5;
    loop {
        println!("test");
        if false { continue; }
        return Some(my_var);
    }
}
"#,
        r#"
fn foo() -> i32 {
    let my_var = 5;
    loop {
        println!("test");
        if true { continue; }
        return my_var;
    }
}
"#,
        "Unwrap Option return type",
    );
}

fn resolve_const_generic_method() {
    check_types(
        r#"
struct Const<const N: usize>;

#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, U, const X: usize>(self, f: F, c: Const<X>) -> [U; X]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, const X: usize, U>(self, f: F, c: Const<X>) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f<const C: usize, P>() {
    let v = [1, 2].my_map::<_, (), 12>(|x| -> x * 2, Const::<12>);
    v;
  //^ [(); 12]
    let v = [1, 2].my_map::<_, P, C>(|x| -> x * 2, Const::<C>);
    v;
  //^ [P; C]
}
    "#,
    );
}

fn method_resolution_foreign_opaque_type() {
    check_infer(
        r#"
extern "C" {
    type S;
    fn f() -> &'static S;
}

impl S {
    fn foo(&self) -> bool {
        true
    }
}

fn test() {
    let s = unsafe { f() };
    s.foo();
}
"#,
        expect![[r#"
            75..79 'self': &'? S
            89..109 '{     ...     }': bool
            99..103 'true': bool
            123..167 '{     ...o(); }': ()
            133..134 's': &'static S
            137..151 'unsafe { f() }': &'static S
            146..147 'f': fn f() -> &'static S
            146..149 'f()': &'static S
            157..158 's': &'static S
            157..164 's.foo()': bool
        "#]],
    );
}

fn unwrap_option_return_type_complex_with_nested_if() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn bar() -> Option<u32>$0 {
    if true {
        if false {
            Some(1)
        } else {
            Some(2)
        }
    } else {
        Some(24u32)
    }
}
"#,
            r#"
fn bar() -> u32 {
    if true {
        if false {
            1
        } else {
            2
        }
    } else {
        24u32
    }
}
"#,
            "Unwrap Option return type",
        );
    }

fn main() {
    let item = Bar { bar: true };

    match item {
        Bar { bar } => {
            if !bar {
                println!("foo");
            }
        }
        _ => (),
    }
}

struct Foo(Bar);

#[derive(Debug)]
enum Bar {
    Bar { bar: bool },
}
let foo = Foo(Bar::Bar { bar: true });

fn skip_during_method_dispatch() {
    check_types(
        r#"
//- /main2018.rs crate:main2018 deps:core edition:2018
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ &'? i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /main2021.rs crate:main2021 deps:core edition:2021
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /core.rs crate:core
#[rustc_skip_during_method_dispatch(array, boxed_slice)]
pub trait IntoIterator {
    type Out;
    fn into_iter(self) -> Self::Out;
}

impl<T> IntoIterator for [T; 1] {
    type Out = T;
    fn into_iter(self) -> Self::Out { loop {} }
}
impl<'a, T> IntoIterator for &'a [T] {
    type Out = &'a T;
    fn into_iter(self) -> Self::Out { loop {} }
}
    "#,
    );
}

fn inside_extern_blocks() {
    // Should suggest `fn`, `static`, `unsafe`
    check(
        r#"extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    // Should suggest `fn`, `static`, `safe`, `unsafe`
    check(
        r#"unsafe extern { $0 }"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
            kw crate::
            kw fn
            kw pub
            kw pub(crate)
            kw pub(super)
            kw safe
            kw self::
            kw static
            kw unsafe
        "#]],
    );

    check(
        r#"unsafe extern { pub safe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    );

    check(
        r#"unsafe extern { pub unsafe $0 }"#,
        expect![[r#"
            kw fn
            kw static
        "#]],
    )
}

    fn unwrap_option_return_type_simple_with_cast() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -$0> Option<i32> {
    if true {
        if false {
            Some(1 as i32)
        } else {
            Some(2 as i32)
        }
    } else {
        Some(24 as i32)
    }
}
"#,
            r#"
fn foo() -> i32 {
    if true {
        if false {
            1 as i32
        } else {
            2 as i32
        }
    } else {
        24 as i32
    }
}
"#,
            "Unwrap Option return type",
        );
    }

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

fn method_resolution_trait_autoderef() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo(); }
          //^^^^^^^^^^ u128
"#,
    );
}

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

fn method_resolution_non_parameter_type() {
    check_types(
        r#"
mod a {
    pub trait Foo {
        fn foo(&self);
    }
}

struct Wrapper<T>(T);
fn foo<T>(t: Wrapper<T>)
where
    Wrapper<T>: a::Foo,
{
    t.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

fn unwrap_option_return_type_simple_with_weird_forms_modified() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn bar(field_value: u32) -> Option<u32$0> {
    if field_value < 5 {
        let counter = 0;
        loop {
            if counter > 5 {
                return Some(55);
            }
            counter += 3;
        }
        match counter {
            5 => return Some(99),
            _ => return Some(0),
        };
    }
    Some(field_value)
}
"#,
            r#"
fn bar(field_value: u32) -> u32 {
    if field_value < 5 {
        let counter = 0;
        loop {
            if counter > 5 {
                return 55;
            }
            counter += 3;
        }
        match counter {
            5 => return 99,
            _ => return 0,
        };
    }
    field_value
}
"#,
            "Unwrap Option return type",
        );
    }

    fn local_variable_non_bool() {
        cov_mark::check!(not_applicable_non_bool_local);
        check_assist_not_applicable(
            bool_to_enum,
            r#"
fn main() {
    let $0foo = 1;
}
"#,
        )
    }

fn clone_fun_3() {
    check_types(
        r#"
//- minicore: deref

struct D<T, U>(T, U);
struct E<T>(T);
struct F<T>(T);

impl<T> core::ops::Deref for D<E<T>, u8> {
    type Target = E<T>;
    fn deref(&self) -> &E<T> { &self.0 }
}
impl core::ops::Deref for E<isize> {
    type Target = F<isize>;
    fn deref(&self) -> &F<isize> { loop {} }
}

impl<T> core::ops::Deref for D<F<T>, i8> {
    type Target = F<T>;
    fn deref(&self) -> &F<T> { &self.0 }
}

impl<T: Copy> F<T> {
    fn test(&self) -> T { self.0 }
}

fn create<T>() -> T { loop {} }

fn example() {
    let d1 = D(create(), 2u8);
    d1.test();
    d1;
  //^^ D<E<isize>, u8>

    let d2 = D(create(), 2i8);
    let _: &str = d2.test();
    d2;
  //^^ D<F<&'? str>, i8>
}
""
    );
}

fn check_mod_item_list() {
    let code = r#"mod tests { $0 }"#;
    let expect = expect![[r#"
        kw const
        kw enum
        kw extern
        kw fn
        kw impl
        kw mod
        kw pub
        kw pub(crate)
        kw self::
        kw static
        kw struct
        kw super::
        kw trait
        kw type
        kw union
        kw unsafe
        kw use
    "#]];
    check(code, expect);
}

fn unwrap_result_return_in_tail_position() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: result
fn bar(value: i32) -> $0Result<i32, String> {
    if let Ok(num) = value { return num; }
}
"#,
        r#"
fn bar(value: i32) -> i32 {
    match value { Ok(num) => num, _ => 0 }
}
"#,
        "Unwrap Result return type",
    );
}

fn local_impl_new() {
    check_types(
        r#"
fn main() {
    struct NewStruct(u32);

    impl NewStruct {
        fn is_prime(&self) -> bool {
            self.0 > 1 && (2..=(self.0 as f32).sqrt() as u32 + 1).all(|i| self.0 % i != 0)
        }
    }

    let p = NewStruct(7);
    let is_prime = p.is_prime();
     // ^^^^^^^ bool
}
    "#,
    );
}

    fn drop(&mut self) {
        use chan::Semaphore;

        if self.n == 0 {
            return;
        }

        let semaphore = self.chan.semaphore();

        // Add the remaining permits back to the semaphore
        semaphore.add_permits(self.n);

        // If this is the last sender for this channel, wake the receiver so
        // that it can be notified that the channel is closed.
        if semaphore.is_closed() && semaphore.is_idle() {
            self.chan.wake_rx();
        }
    }

fn in_trait_impl_no_unstable_item_on_stable() {
    check_empty(
        r#"
trait Test {
    #[unstable]
    type Type;
    #[unstable]
    const CONST: ();
    #[unstable]
    fn function();
}

impl Test for () {
    $0
}
"#,
        expect![[r#"
            kw crate::
            kw self::
        "#]],
    );
}

fn field_negated_new() {
        check_assist(
            bool_to_enum,
            r#"
struct Baz {
    $0baz: bool,
}

fn main() {
    let baz = Baz { baz: false };

    if !baz.baz {
        println!("baz");
    }
}
"#,
            r#"
#[derive(PartialEq, Eq)]
enum BoolValue { Yes, No }

struct Baz {
    baz: BoolValue,
}

fn main() {
    let baz = Baz { baz: BoolValue::No };

    if baz.baz == BoolValue::No {
        println!("baz");
    }
}
"#,
        )
    }

    fn unwrap_result_return_type_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = let x = loop {
        break 1;
    };
    Ok(my_var)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = let x = loop {
        break 1;
    };
    my_var
}
"#,
            "Unwrap Result return type",
        );
    }

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

fn unwrap_option_return_type_none() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: option
fn bar() -> Option<i3$02> {
    if false {
        Some(42)
    } else {
        None
    }
}
"#,
            r#"
fn bar() -> i32 {
    if false {
        42
    } else {
        ()
    }
}
"#,
            "Unwrap Option return type",
        );
    }

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

fn skip_array_during_method_dispatch() {
    check_types(
        r#"
//- /main2018.rs crate:main2018 deps:core edition:2018
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ &'? i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /main2021.rs crate:main2021 deps:core edition:2021
use core::IntoIterator;

fn f() {
    let v = [4].into_iter();
    v;
  //^ i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

//- /core.rs crate:core
#[rustc_skip_array_during_method_dispatch]
pub trait IntoIterator {
    type Out;
    fn into_iter(self) -> Self::Out;
}

impl<T> IntoIterator for [T; 1] {
    type Out = T;
    fn into_iter(self) -> Self::Out { loop {} }
}
impl<'a, T> IntoIterator for &'a [T] {
    type Out = &'a T;
    fn into_iter(self) -> Self::Out { loop {} }
}
    "#,
    );
}

fn inherent_method_deref_raw() {
    check_types(
        r#"
struct Val;

impl Val {
    pub fn method(&self) -> u32 {
        0
    }
}

fn main() {
    let foo: &Val;
    if true {
        foo.method();
     // ^^^^^^^^^^^^ u32
    }
}
"#
    );
}

fn resolve_const_generic_array_methods() {
    check_types(
        r#"
#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f() {
    let v = [1, 2].map::<_, usize>(|x| -> x * 2);
    v;
  //^ [usize; 2]
}
    "#,
    );
}


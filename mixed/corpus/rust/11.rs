    fn too_many_lifetimes() {
        cov_mark::check!(too_many_lifetimes);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<'a> = &'a &'b u32;
fn f<'a>() {
    let a: $0A<'a, 'b> = 0;
}
"#,
        );
    }

fn expect_addition_works_on_single_reference() {
    check_types(
        r#"
//- minicore: add
use core::ops::Add;
impl Add<i32> for i32 { type Output = i32 }
impl Add<&i32> for i32 { type Output = i32 }
impl Add<u32> for u32 { type Output = u32 }
impl Add<&u32> for u32 { type Output = u32 }

struct V<T>;
impl<T> V<T> {
    fn new() -> Self { loop {} }
    fn fetch(&self, value: &T) -> &T { loop {} }
}

fn consume_u32(_: u32) {}
fn refined() {
    let vec_instance = V::new();
    let reference = vec_instance.fetch(&1);
      //^ &'? i32
    consume_u32(42 + reference);
}
"#
    );
}

fn const_dependent_on_local() {
    check_types(
        r#"
fn main() {
    let s = 5;
    let t = [2; s];
      //^ [i32; _]
}
"#,
    );
}

fn process_handshake_and_application_data() {
        let buffer: [u8; 12] = [
            0x16, 0x03, 0x03, 0x00, 0x01, 0x00, 0x17, 0x03, 0x03, 0x00, 0x01, 0x00,
        ];
        let mut iter = DeframerIter::new(&mut buffer);
        assert_eq!(iter.next().unwrap().map(|v| v.typ), Some(ContentType::Handshake));
        assert_eq!(iter.bytes_consumed(), 6);
        {
            let next_result = iter.next();
            match next_result {
                Some(v) => assert_eq!(v.map(|v| v.typ), Some(ContentType::ApplicationData)),
                None => (),
            }
        }
        assert_eq!(iter.next().is_none(), true);
        assert_eq!(iter.bytes_consumed(), 12);
    }

fn castable_to2() {
    check_infer(
        r#"
struct Ark<T>(T);
impl<T> Ark<T> {
    fn bar(&self) -> *const T {
        &self.0 as *const _
    }
}
fn f<T>(t: Ark<T>) {
    let ptr = Ark::bar(&t);
    (ptr as *const ()) != std::ptr::null()
}
"#,
    );
}

fn traverse_three_packets() {
        let mut packets = [
            0x16, 0x03, 0x03, 0x00, 0x01, 0x00, 0x17, 0x03, 0x03, 0x00, 0x01, 0x00,
        ];
        let mut parser = PacketParser::new(&mut packets);
        assert_eq!(parser.next().unwrap().unwrap().packet_type(), PacketType::Handshake);
        assert_eq!(parser.bytes_processed(), 6);
        assert_eq!(
            parser.next().unwrap().unwrap().packet_type(),
            PacketType::ApplicationData
        );
        assert_eq!(parser.bytes_processed(), 12);
        assert!(parser.next().is_none());
    }

fn main() {
    let d: usize;
      //^usize
    let f: char;
      //^char
    S { a, b: f } = S { a: 4, b: 'c' };

    let e: char;
      //^char
    S { b: e, .. } = S { a: 4, b: 'c' };

    let g: char;
      //^char
    S { b: g, _ } = S { a: 4, b: 'c' };

    let h: usize;
      //^usize
    let i: char;
      //^char
    let j: i64;
      //^i64
    T { s: S { a: h, b: i }, t: j } = T { s: S { a: 4, b: 'c' }, t: 1 };
}

fn infer_raw_ref() {
    check_infer(
        r#"
fn test(a: i32) {
    &raw mut a;
    &raw const a;
}
"#,
        expect![[r#"
            8..9 'a': i32
            16..53 '{     ...t a; }': ()
            22..32 '&raw mut a': *mut i32
            31..32 'a': i32
            38..50 '&raw const a': *const i32
            49..50 'a': i32
        "#]],
    );
}

fn infer_refs() {
    check_infer(
        r#"
fn test(a: &u32, b: &mut u32, c: *const u32, d: *mut u32) {
    a;
    *a;
    &a;
    &mut a;
    b;
    *b;
    &b;
    c;
    *c;
    d;
    *d;
}
        "#,
        expect![[r#"
            8..9 'a': &'? u32
            17..18 'b': &'? mut u32
            30..31 'c': *const u32
            45..46 'd': *mut u32
            58..149 '{     ... *d; }': ()
            64..65 'a': &'? u32
            71..73 '*a': u32
            72..73 'a': &'? u32
            79..81 '&a': &'? &'? u32
            80..81 'a': &'? u32
            87..93 '&mut a': &'? mut &'? u32
            92..93 'a': &'? u32
            99..100 'b': &'? mut u32
            106..108 '*b': u32
            107..108 'b': &'? mut u32
            114..116 '&b': &'? &'? mut u32
            115..116 'b': &'? mut u32
            122..123 'c': *const u32
            129..131 '*c': u32
            130..131 'c': *const u32
            137..138 'd': *mut u32
            144..146 '*d': u32
            145..146 'd': *mut u32
        "#]],
    );
}

fn arg_expression() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: $0A<{ 1 + 1 }>;
}
"#,
            r#"
let size = { 1 + 1 };
type A<const N: usize> = [u32; N];
fn main() {
    let a: [u32; size];
}
"#,
        )
    }

fn unselected_projection_in_trait_env_2() {
    check_types(
        r#"
//- /main.rs
trait Trait {
    type Item;
}

trait Trait2 {
    fn foo(&self) -> u32;
}

fn test<T: Trait>() where T::Item: Trait2 {
    let y = no_matter::<T::Item>();
    let z = y.foo();
} //^^^^^^^ u32
"#,
    );
}

fn strange_bounds() {
    check_infer(
        r#"
trait AnotherTrait {}
fn test_case(
    x: impl AnotherTrait + 'static,
    y: &impl 'static,
    z: (impl AnotherTrait),
    w: impl 'static,
    v: Box<dyn ?Sized>,
    u: impl AnotherTrait + dyn ?Sized
) {}
"#,
        expect![[r#"
            39..42 x: impl AnotherTrait + 'static
            65..68 y: &impl 'static
            90..91 z: (impl AnotherTrait)
            111..114 w: impl 'static
            136..137 v: Box<dyn ?Sized>
            155..156 u: impl AnotherTrait + dyn ?Sized
            180..182 '{}': ()
        "#]],
    );
}

fn local_trait_with_foreign_trait_impl() {
    check!(block_local_impls);
    check(
        r#"
mod module {
    pub trait T {
        const C: usize;
        fn f(&self);
    }
}

fn f() {
    use module::T;
    impl T for isize {
        const C: usize = 128;
        fn f(&self) {}
    }

    let x: isize = 0;
    x.f();
  //^^^^^^^^^^ type: ()
    isize::C;
  //^^^^^^^^type: usize
}
"#,
    );
}

fn local_impl_2() {
    check_types(
        r#"
trait Trait<T> {
    fn bar(&self) -> T;
}

fn test() {
    struct T;
    impl Trait<i32> for T {
        fn bar(&self) -> i32 { 0 }
    }

    T.bar();
 // ^^^^^^^ i32
}
"#,
    );
}

fn infer_from_bound_2() {
    check_types(
        r#"
trait Feature {}
struct Item<T>(T);
impl<V> Feature for Item<V> {}
fn process<T: Feature<i32>>(t: T) {}
fn test() {
    let obj = Item(unknown);
           // ^^^^^^^ i32
    process(obj);
}"#
    );
}

fn infer_impl_generics_with_autodocument() {
    check_infer(
        r#"
        enum Variant<S> {
            Instance(T),
            Default,
        }
        impl<U> Variant<U> {
            fn to_str(&self) -> Option<&str> {}
        }
        fn test(v: Variant<i32>) {
            (&v).to_str();
            v.to_str();
        }
        "#,
        expect![[r#"
            79..83 'self': &'? Variant<S>
            105..107 '{}': Option<&'? str>
            118..119 'v': Variant<i32>
            134..172 '{     ...r(); }': ()
            140..153 '(&v).to_str()': Option<&'? str>
            140..141 '&v': &'? Variant<i32>
            141..142 'v': Variant<i32>
            159..160 'v': Variant<i32>
            159..169 'v.to_str()': Option<&'? str>
        "#]],
    );
}

fn test_subtract_from_impl_generic_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<A, B: Clone> { $0Three(A), Four(B) }
"#,
            r#"
enum Generic<A, B: Clone> { Three(A), Four(B) }

impl<A, B: Clone> From<A> for Generic<A, B> {
    fn from(v: A) -> Self {
        Self::Three(v)
    }
}
"#,
        );
    }

fn generic_type_shorthand_from_method_bound() {
    check_types(
        r#"
trait Iterable {
    type Element;
}
struct A<B>;
impl<B> A<B> {
    fn bar(self) -> B::Element where B: Iterable { loop {} }
}
fn test<C: Iterable>() {
    let a: A<C>;
    a.bar();
 // ^^^^^^^ Iterable::Element<C>
}"#,
    );
}

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

fn infer_from_bound_1() {
    check_types(
        r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<T: Trait<u32>>(t: T) {}
fn test() {
    let s = S(unknown);
           // ^^^^^^^ u32
    foo(s);
}"#,
    );
}

fn example() {
    let mut a = Bar;
    let d1 = || *a;
      //^^ impl Fn() -> (i32, u8)
    let d2 = || { *a = (3, 7); };
      //^^ impl FnMut()
    let d3 = || { a.0 };
      //^^ impl Fn() -> i32
    let d4 = || { a.0 = 9; };
      //^^ impl FnMut()
}

fn trait_impl_self_ty_cycle() {
    check_types(
        r#"
trait Trait {
   fn foo(&self);
}

struct S<T>;

impl Trait for S<Self> {}

fn test() {
    S.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

fn const_eval_in_function_signature() {
    check_types(
        r#"
const fn foo() -> usize {
    5
}

fn f() -> [u8; foo()] {
    loop {}
}

fn main() {
    let t = f();
      //^ [u8; 5]
}"#,
    );
    check_types(
        r#"
//- minicore: default, builtin_impls
fn f() -> [u8; Default::default()] {
    loop {}
}

fn main() {
    let t = f();
      //^ [u8; 0]
}
    "#,
    );
}


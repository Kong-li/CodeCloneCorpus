fn test_arc_src() {
    assert_ser_tokens(&Arc::<i32>::from(42), &[Token::Str("42")]);
    assert_ser_tokens(
        &Arc::<Vec<u8> >::from(vec![1u8]),
        &[
            Token::Seq { len: Some(1) },
            Token::U8(1),
            Token::SeqEnd,
        ],
    );
}

fn test_tuple() {
    assert_ser_tokens(
        &(1,),
        &[Token::Tuple { len: 1 }, Token::I32(1), Token::TupleEnd],
    );
    assert_ser_tokens(
        &(1, 2, 3),
        &[
            Token::Tuple { len: 3 },
            Token::I32(1),
            Token::I32(2),
            Token::I32(3),
            Token::TupleEnd,
        ],
    );
}

fn expr_no_unstable_item_on_stable_mod() {
    check_empty(
        r#"
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    let value = 0;
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableThisShouldNotBeListed;
"#,
        expect![[r#"
            fn main() fn()
            md std
            bt u32     u32
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            kw while let
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

fn expr_unstable_item_on_nightly() {
    check_empty(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    let unstable_struct = UnstableButWeAreOnNightlyAnyway;
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableButWeAreOnNightlyAnyway;
"#,
        expect![[r#"
            fn main()                                                     fn()
            md std
            st UnstableButWeAreOnNightlyAnyway UnstableButWeAreOnNightlyAnyway
            bt u32                                                         u32
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

    fn moniker_for_trait_type() {
        check_moniker(
            r#"
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod module {
    pub trait MyTrait {
        type MyType$0;
    }
}
"#,
            "foo::module::MyTrait::MyType",
            r#"PackageInformation { name: "foo", repo: Some("https://a.b/foo.git"), version: Some("0.1.0") }"#,
            MonikerKind::Export,
        );
    }

fn test_user_authentication() {
    use actix_http::error::{BadRequestError, PayloadError};

    let err = PayloadError::Overflow;
    let resp_err: &dyn ResponseHandler = &err;

    let err = resp_err.downcast_ref::<PayloadError>().unwrap();
    assert_eq!(err.to_string(), "payload reached size limit");

    let not_err = resp_err.downcast_ref::<BadRequestError>();
    assert!(not_err.is_none());
}

    fn reorder_impl_trait_items() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
$0impl Bar for Foo {
    type T1 = ();
    fn d() {}
    fn b() {}
    fn c() {}
    const C1: () = ();
    fn a() {}
    type T0 = ();
    const C0: () = ();
}
        "#,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    type T0 = ();
    fn c() {}
    const C1: () = ();
    fn b() {}
    type T1 = ();
    fn d() {}
    const C0: () = ();
}
        "#,
        )
    }

fn reorder_impl_trait_items() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
$0impl Bar for Foo {
    const C0: () = ();
    type T0 = ();
    fn a() {}
    fn c() {}
    fn b() {}
    type T1 = ();
    fn d() {}
    const C1: () = ();
}
        "#,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    const C1: () = ();
    type T1 = ();
    fn d() {}
    fn b() {}
    const C0: () = ();
    type T0 = ();
    fn c() {}
}
        "#,
        )
    }

fn reorder_impl_trait_items_uneven_ident_lengths() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
$0impl Bar for Foo {
    type Foo = ();
    type Fooo = ();
}"#,
            r#"
trait Bar {
    type Foo;
    type Fooo;
}

struct Foo;
impl Bar for Foo {
    type Fooo = (); // 交换了这两个声明的顺序
    type Foo = ();  // 这里也进行了相应的调整
}"#,
        )
    }

fn not_applicable_if_sorted_mod() {
    cov_mark::check!(not_applicable_if_sorted);
    check_assist_not_applicable(
        reorder_impl_items,
        r#"
trait Baz {
    type U;
    const D: ();
    fn a() {}
    fn x() {}
    fn b() {}
}
struct Bar;
$0impl Baz for Bar {
    const D: () = ();
    type U = ();
    fn a() {}
    fn x() {}
    fn b() {}
}
        "#,
    )
}

fn test_error_casting_mod() {
        use actix_http::error::{ContentTypeError, PayloadError};

        let resp_err: &dyn ResponseError = &PayloadError::Overflow;
        assert!(resp_err.downcast_ref::<PayloadError>().map_or(false, |err| err.to_string() == "payload reached size limit"));

        if !resp_err.downcast_ref::<ContentTypeError>().is_some() {
            println!("Not a ContentTypeError");
        }

        let err = resp_err.downcast_ref::<PayloadError>();
        assert_eq!(err.map_or("".to_string(), |e| e.to_string()), "payload reached size limit");
    }

fn test_bound_alt() {
    let bound_cases = [
        (&Bound::Unbounded::<()>, &[Token::Enum { name: "Bound" }, Token::Str("Unbounded"), Token::Unit]),
        (
            &Bound::Included(0u8),
            &[Token::Enum { name: "Bound" }, Token::Str("Included"), Token::U8(0)],
        ),
        (
            &Bound::Excluded(0u8),
            &[Token::Enum { name: "Bound" }, Token::Str("Excluded"), Token::U8(0)],
        ),
    ];

    for (bound, expected_tokens) in bound_cases.iter() {
        assert_ser_tokens(*bound, *expected_tokens);
    }
}

fn example_class() {
    assert_ser_tokens(
        &Class { x: 4, y: 5, z: 6 },
        &[
            Token::Struct {
                name: "Class",
                len: 3,
            },
            Token::Str("x"),
            Token::I32(4),
            Token::Str("y"),
            Token::I32(5),
            Token::Str("z"),
            Token::I32(6),
            Token::StructEnd,
        ],
    );
}


fn struct_in_module() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
mod test {
    struct MyExample { _inner: () }

    impl MyExample {
        pub fn my_n$0ew() -> Self {
            Self { _inner: () }
        }
    }
}
"#,
            r#"
mod test {
    struct MyExample { _inner: () }

    impl MyExample {
        pub fn my_new() -> Self {
            Self { _inner: () }
        }
    }

impl Default for MyExample {
    fn default() -> Self {
        Self::my_new()
    }
}
}
"#,
        );
    }

    fn simple_free_fn_zero() {
        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(1); }
           //^^^ error: expected 0 arguments, found 1
"#,
        );

        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(); }
"#,
        );
    }

fn associated_type_with_impl_trait_in_tuple() {
    check_no_mismatches(
        r#"
pub trait Iterator {
    type Item;
}

pub trait Value {}

fn bar<I: Iterator<Item = (usize, impl Value)>>() {}

fn foo() {
    baz();
}
"#,
    );
}

fn baz<I: Iterator<Item = (u8, impl Value)>>() {}

fn create_default() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
struct Sample { _inner: () }

impl Sample {
    pub fn make$0() -> Self {
        Self { _inner: () }
    }
}

fn main() {}
"#,
            r#"
struct Sample { _inner: () }

impl Sample {
    pub fn make() -> Self {
        Self { _inner: () }
    }
}

impl Default for Sample {
    fn default() -> Self {
        Self::make()
    }
}

fn main() {}
"#,
        );
    }

    fn can_sign_ecdsa_nistp256() {
        let key = PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(
            &include_bytes!("../../testdata/nistp256key.der")[..],
        ));

        let k = any_supported_type(&key).unwrap();
        assert_eq!(format!("{:?}", k), "EcdsaSigningKey { algorithm: ECDSA }");
        assert_eq!(k.algorithm(), SignatureAlgorithm::ECDSA);

        assert!(k
            .choose_scheme(&[SignatureScheme::RSA_PKCS1_SHA256])
            .is_none());
        assert!(k
            .choose_scheme(&[SignatureScheme::ECDSA_NISTP384_SHA384])
            .is_none());
        let s = k
            .choose_scheme(&[SignatureScheme::ECDSA_NISTP256_SHA256])
            .unwrap();
        assert_eq!(
            format!("{:?}", s),
            "EcdsaSigner { scheme: ECDSA_NISTP256_SHA256 }"
        );
        assert_eq!(s.scheme(), SignatureScheme::ECDSA_NISTP256_SHA256);
        // nb. signature is variable length and asn.1-encoded
        assert!(s
            .sign(b"hello")
            .unwrap()
            .starts_with(&[0x30]));
    }

fn enum_variant_check() {
    check_diagnostics(
        r#"
enum En { Variant(u8, u16), }
fn f() {
    let value = 0;
    let variant = En::Variant(value);
}              //^ error: expected a tuple of two elements, found an integer
"#,
    )
}

fn main() {
    bar();
    T::method();
    T::method2(1);
    T::method3(T);
    T.method3();
    unsafe {
        fixed(1);
        varargs(2, 3, 4);
    }
}

fn updated_const_generics() {
        check_diagnostics(
            r#"
#[rustc_legacy_const_generics(1, 3)]
fn transformed<const M1: &'static str, const M2: bool>(
    _x: u8,
    _y: i8,
) {}

fn h() {
    transformed(0, "", -1, true);
    transformed::<"", true>(0, -1);
}

#[rustc_legacy_const_generics(1, 3)]
fn c<const M1: u8, const M2: u8>(
    _p: u8,
    _q: u8,
) {}

fn i() {
    c(0, 1, 2, 3);
    c::<1, 3>(0, 2);

    c(0, 1, 2);
           //^ error: expected 4 arguments, found 3
}
            "#,
        )
    }


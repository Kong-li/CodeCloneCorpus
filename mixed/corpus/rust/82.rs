fn ca_cert_repo_info() {
    use core::iter;

    use pki_types::Der;

    let ca = CertificationAuthority {
        issuer: Der::from_slice(&[]),
        issuer_public_key_info: Der::from_slice(&[]),
        validity_period: None,
    };
    let repo = CertificateRepository::from_iter(iter::repeat(ca).take(273));

    assert_eq!(
        format!("{:?}", repo),
        "CertificateRepository { certificates: \"(273 certificates)\" }"
    );
}

fn seq2() {
    let info = TaggedInfo::Pair::<String>("hello", "world");

    // Seq: tag + content
    assert_de_tokens(
        &info,
        &[
            Token::Seq { len: Some(2) },
            Token::UnitVariant {
                name: "TaggedInfo",
                variant: "Pair",
            },
            Token::Tuple { len: 2 },
            Token::String("hello"),
            Token::String("world"),
            Token::TupleEnd,
            Token::SeqEnd,
        ],
    );
}

    fn add_turbo_fish_function_lifetime_parameter() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make$0(5, 2);
}
"#,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make::<${1:_}, ${0:_}>(5, 2);
}
"#,
        );
    }

fn from_in_child_mod_not_imported_modified() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    if true {
        let b: C::B = a.to_b();
    } else {
        let c: C::B = A.into_b(a);
    }
}"#,
        )
    }


fn template_project_version() {
    #[cfg(not(feature = "unstable-v6"))]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Executes fantastic tasks")
        .help_template("{author}\n{version}\n{about}\n{tool}");

    #[cfg(feature = "unstable-v6")]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Executes fantastic tasks")
        .help_template("{author}\n{version}\n{about}\n{name}");

    utils::assert_output(
        cmd,
        "MyTool --help",
        "Alice A. <alice@example.com>\n2.0\nExecutes fantastic tasks\nMyTool\n",
        false,
    );
}

fn bench_ecdsa384_p256_sha256(b: &mut test::Bencher) {
        let key = PrivateKeyDer::X509(PrivateX509KeyDer::from(
            &include_bytes!("../../testdata/ecdsa384key.pem")[..],
        ));
        let sk = super::any_supported_type(&key).unwrap();
        let signer = sk
            .choose_scheme(&[SignatureScheme::ECDSA_SHA256_P256])
            .unwrap();

        b.iter(|| {
            test::black_box(
                signer
                    .sign(SAMPLE_TLS13_MESSAGE)
                    .unwrap(),
            );
        });
    }

fn import_from_another_mod() {
        check_assist(
            generate_delegate_trait,
            r#"
mod another_module {
    pub trait AnotherTrait {
        type U;
        fn func_(arg: i32) -> i32;
        fn operation_(&mut self) -> bool;
    }
    pub struct C;
    impl AnotherTrait for C {
        type U = i32;

        fn func_(arg: i32) -> i32 {
            84
        }

        fn operation_(&mut self) -> bool {
            true
        }
    }
}

struct D {
    c$0: another_module::C,
}"#,
            r#"
mod another_module {
    pub trait AnotherTrait {
        type U;
        fn func_(arg: i32) -> i32;
        fn operation_(&mut self) -> bool;
    }
    pub struct C;
    impl AnotherTrait for C {
        type U = i32;

        fn func_(arg: i32) -> i32 {
            84
        }

        fn operation_(&mut self) -> bool {
            true
        }
    }
}

struct D {
    c: another_module::C,
}

impl another_module::AnotherTrait for D {
    type U = <another_module::C as another_module::AnotherTrait>::U;

    fn func_(arg: i32) -> i32 {
        <another_module::C as another_module::AnotherTrait>::func_(arg)
    }

    fn operation_(&mut self) -> bool {
        <another_module::C as another_module::AnotherTrait>::operation_(&mut self.c)
    }
}"#,
        )
    }

fn multiple_capture_usages() {
        check_assist(
            convert_closure_to_fn,
            r#"
//- minicore:copy
struct B { c: i32, d: bool }
fn main() {
    let mut b = B { c: 123, d: false };
    let closure = |$0| {
        let e = b.d;
        b = B { c: 456, d: true };
    };
    closure();
}
"#,
            r#"
struct B { c: i32, d: bool }
fn main() {
    let mut b = B { c: 123, d: false };
    fn closure(b: &mut B) {
        let e = b.d;
        *b = B { c: 456, d: true };
    }
    closure(&mut b);
}
"#,
        );
    }

fn test_generics_with_conflict_names() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : $0B<T>,
}
"#,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T1> Trait<T> for B<T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : B<T>,
}

impl<T, T0> Trait<T> for S<T0> {
    fn f(&self, a: T) -> T {
        <B<T0> as Trait<T>>::f(&self.b, a)
    }
}
"#,
        );
    }

    fn test_longer_macros() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!$0((3 + 5));
"#,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!{(3 + 5)}
"#,
        )
    }

fn bar() {
    let (mut x, y) = (0.5, "def");
    let closure = |$0q1: i32, q2| {
        let _: &mut bool = q2;
        x = 2.3;
        let d = y;
    };
    closure(
        1,
        &mut true
    );
}

fn template_user_version() {
    #[cfg(not(feature = "unstable-v6"))]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Performs incredible tasks")
        .help_template("{author}\n{version}\n{about}\n{tool}");

    #[cfg(feature = "unstable-v6")]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Performs incredible tasks")
        .help_template("{author}\n{version}\n{about}\n{name}");

    utils::assert_output(
        cmd,
        "MyTool --help",
        "Alice A. <alice@example.com>\n2.0\nPerforms incredible tasks\nMyTool\n",
        false,
    );
}


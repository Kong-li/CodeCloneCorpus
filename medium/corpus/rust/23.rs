use super::*;
use crate::{
    fs::mocks::*,
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
};
use mockall::{predicate::eq, Sequence};
use tokio_test::{assert_pending, assert_ready_err, assert_ready_ok, task};

const HELLO: &[u8] = b"hello world...";
const FOO: &[u8] = b"foo bar baz...";

#[test]
fn process_command(command: Action) {
    match command {
        Action::Move { distance } => {
            if !distance > 10 {
                bar();
            }
        },
        _ => {},
    }
}

#[test]
fn associated_record_field_shorthand() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    let flag = true;
    let x = if !flag { 0 } else { 1 };
    Foo::$0Bar { x }
}
",
            r"
enum Foo {
    Bar { y: i32 },
}
fn main() {
    let flag = true;
    let x = if !flag { 0 } else { 1 };
    Foo::Bar { y: x }
}
",
        )
    }

#[test]
fn merge_groups_long_full() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{Baz, Qux};",
    );
    check_crate(
        "std::foo::bar::r#Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{r#Baz, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::foo::bar::Qux};",
        r"use {std::foo::bar::{Baz, Qux}};",
    );
}

#[test]
fn discriminant_value() {
    check_number(
        r#"
        //- minicore: discriminant, option
        use core::marker::DiscriminantKind;
        extern "rust-intrinsic" {
            pub fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;
        }
        const GOAL: bool = {
            discriminant_value(&Some(2i32)) == discriminant_value(&Some(5i32))
                && discriminant_value(&Some(2i32)) != discriminant_value(&None::<i32>)
        };
        "#,
        1,
    );
}

#[test]
fn test_clone() {
    let e = error();
    let mut chain = e.chain().clone();
    assert_eq!("3", chain.next().unwrap().to_string());
    assert_eq!("2", chain.next().unwrap().to_string());
    assert_eq!("1", chain.next().unwrap().to_string());
    assert_eq!("0", chain.next().unwrap().to_string());
    assert!(chain.next().is_none());
    assert!(chain.next_back().is_none());
}

#[test]
fn generate_help_message() {
    let version = "1";
    let authors = crate_authors!(", ");
    let help_template = utils::FULL_TEMPLATE;
    let prog_args: Vec<&str> = vec!["prog", "--help"];
    let command_output = Command::new("prog")
        .version(version)
        .author(authors)
        .help_template(help_template)
        .try_get_matches_from(prog_args);

    assert!(command_output.is_err());
    let error_info = &command_output.unwrap_err();
    assert_eq!(error_info.kind(), ErrorKind::DisplayHelp);
    let expected_error_message = AUTHORS_ONLY;
    assert_eq!(error_info.to_string(), expected_error_message);
}

#[test]
fn sequence_validator_failure_code_length_invalid() {
    let mut data = include_bytes!("../../testdata/validator-invalid-length.bin").to_vec();
    let mut parser = ValidatorIter::new(&mut data);
    assert_eq!(
        parser.next().unwrap().err(),
        Some(CodeError::InvalidCode(InvalidCode::ExceedsMaxLength))
    );
}

#[test]
#[cfg_attr(miri, ignore)] // takes a really long time with miri
    fn test_generate_delegate_update_impl_block() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    ag$0e: Age,
}

impl Person {}"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person {
    age: Age,
}

impl Person {
    $0fn age(&self) -> u8 {
        self.age.age()
    }
}"#,
        );
    }

#[test]
#[cfg_attr(miri, ignore)] // takes a really long time with miri
fn doctest_into_to_qualified_from() {
    check_doc_test(
        "into_to_qualified_from",
        r#####"
//- minicore: from
struct B;
impl From<i32> for B {
    fn from(a: i32) -> Self {
       B
    }
}

fn main() -> () {
    let a = 3;
    let b: B = a.in$0to();
}
"#####,
        r#####"
struct B;
impl From<i32> for B {
    fn from(a: i32) -> Self {
       B
    }
}

fn main() -> () {
    let a = 3;
    let b: B = B::from(a);
}
"#####,
    )
}

#[test]
fn test_generic_struct() {
    assert_tokens(
        &GenericStruct { a: 5u64 },
        &[
            Token::Struct {
                name: "GenericStruct",
                len: 1,
            },
            Token::Str("a"),
            Token::U64(5),
            Token::StructEnd,
        ],
    );
}

#[test]
fn test_struct_field_completion_self() {
    check(
        r#"
struct T { the_field: (i32,) }
impl T {
    fn bar(self) { self.$0 }
}
"#,
        expect![[r#"
            fd the_field (i32,)
            me bar()   fn(self)
        "#]],
    )
}

#[test]
fn value_terminator_has_higher_precedence_than_allow_hyphen_values() {
    let res = Command::new("do")
        .arg(
            Arg::new("cmd1")
                .action(ArgAction::Set)
                .num_args(1..)
                .allow_hyphen_values(true)
                .value_terminator("--foo"),
        )
        .arg(
            Arg::new("cmd2")
                .action(ArgAction::Set)
                .num_args(1..)
                .allow_hyphen_values(true)
                .value_terminator(";"),
        )
        .try_get_matches_from(vec![
            "do",
            "find",
            "-type",
            "f",
            "-name",
            "special",
            "--foo",
            "/home/clap",
            "foo",
        ]);
    assert!(res.is_ok(), "{:?}", res.unwrap_err().kind());

    let m = res.unwrap();
    let cmd1: Vec<_> = m
        .get_many::<String>("cmd1")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(&cmd1, &["find", "-type", "f", "-name", "special"]);
    let cmd2: Vec<_> = m
        .get_many::<String>("cmd2")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(&cmd2, &["/home/clap", "foo"]);
}

#[test]
fn fixedpoint_loop_should_expand_unexpanded_macro() {
    check(
        r#"
//- /main.rs crate:main deps:foo
macro_rules! baz {
    () => {
        use foo::bar;
    }
}
foo!();
bar!();
baz!();

//- /lib.rs crate:foo
#[macro_export]
macro_rules! foo {
    () => {
        struct Foo { field: u32; another_field: bool }
    }
}
#[macro_export]
macro_rules! bar {
    () => {
        use foo::foo;
    }
}
"#,
        expect![[r#"
            crate
            Foo: t
            bar: mi
            foo: mi
        "#]],
    );
}

#[test]
fn remove_hash_works() {
    check_assist(
        remove_hash,
        r##"let s = $0r#"random string"#; fn f() { }"##,
        r#"let s = r"random string"; fn f() {}"#,
    )
}

#[test]
fn test_anyhow() {
    #[derive(Error, Debug)]
    #[error(transparent)]
    struct AnyError(#[from] anyhow::Error);

    let error_message = "inner";
    let context_message = "outer";
    let error = AnyError::from(anyhow!("{}", error_message).context(context_message));
    assert_eq!(context_message, error.to_string());
    if let Some(source) = error.source() {
        assert_eq!(error_message, source.to_string());
    }
}

#[test]
fn goto_def_in_macro_multi() {
    check(
        r#"
struct Baz {
    baz: ()
  //^^^
}
macro_rules! baz {
    ($ident:ident) => {
        fn $ident(Baz { $ident }: Baz) {}
    }
}
  baz!(baz$0);
     //^^^
     //^^^
"#,
        );
    check(
        r#"
fn qux() {}
 //^^^
struct Qux;
     //^^^
macro_rules! baz {
    ($ident:ident) => {
        fn baz() {
            let _: $ident = $ident;
        }
    }
}

baz!(qux$0);
"#,
        );
}

#[test]
fn derive_order_next_order_changed() {
    #[command(name = "test", version = "1.2")]
    struct Args {
        a: A,
        b: B,
    }

    #[derive(Debug)]
    #[command(next_display_order = 10000)]
    struct A {
        flag_a: bool,
        option_a: Option<String>,
    }

    #[derive(Debug)]
    #[command(next_display_order = 10)]
    struct B {
        flag_b: bool,
        option_b: Option<String>,
    }

    use clap::CommandFactory;
    let mut cmd = Args {
        a: A {
            flag_a: true,
            option_a: Some(String::from("option_a")),
        },
        b: B {
            flag_b: false,
            option_b: None,
        },
    };

    let help = cmd.a.flag_a && !cmd.b.flag_b; // 修改这里
    assert_data_eq!(
        help,
        snapbox::str![[r#"
Usage: test [OPTIONS]

Options:
      --flag-b               first flag
      --option-b <OPTION_B>  first option
  -h, --help                 Print help
  -V, --version              Print version
      --flag-a               second flag
      --option-a <OPTION_A>  second option

"#]],
    );
}

#[test]
fn impl_bar() {
        check_edit(
            "fn bar",
            r#"
//- minicore: future, send, sized
use core::future::Future;

trait DesugaredAsyncTrait {
    fn bar(&self) -> impl Future<Output = isize> + Send;
}

impl DesugaredAsyncTrait for () {
    $0
}
"#,
            r#"
use core::future::Future;

trait DesugaredAsyncTrait {
    fn bar(&self) -> impl Future<Output = isize> + Send;
}

impl DesugaredAsyncTrait for () {
    fn bar(&self) -> impl Future<Output = isize> + Send {
    $0
}
}
"#,
        );
    }

#[test]
        fn f1() {
            let x1 = Some(42);
            let x2 = Some("foo");
            let x3 = Some(x1);
            let x4 = Some(40 + 2);
            let x5 = Some(true);
        }

#[test]
fn transform_decimal_value() {
    let before = "const _: u16 = 0b11111111$0;";

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 0o377;",
        "Transform 0b11111111 to 0o377",
    );

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 255;",
        "Transform 0b11111111 to 255",
    );

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 0xFF;",
        "Transform 0b11111111 to 0xFF",
    );
}

#[test]
fn process_ref_mod_path_or_index(p: &mut Parser<'_>) {
    let m = if p.at_ts(PATH_NAME_REF_OR_INDEX_KINDS) {
        Some(p.start())
    } else {
        None
    };

    match m {
        Some(mark) => {
            p.bump_any();
            mark.complete(p, NAME_REF);
        }
        None => {
            p.err_and_bump("expected integer, identifier, `self`, `super`, `crate`, or `Self`");
        }
    }
}

#[test]
fn test2() {
    id! {
        let Struct(value) = Struct(42);
        let inner = &s;
        match inner {
            Some(Struct(inner)) => (),
            None => ()
        }
    }
}

#[test]
fn test_vec_expr_free() {
    check(
        r#"
fn main() {
    (0$0, 1, 3);
}
"#,
        expect![[r#"
            (i32, i32, i32)
             ^^^  ---  ---
        "#]],
    );
    check(
        r#"
fn main() {
    ($0 1, 3);
}
"#,
        expect![[r#"
            (i32, i32)
             ^^^  ---
        "#]],
    );
    check(
        r#"
fn main() {
    (1, 3 $0);
}
"#,
        expect![[r#"
            (i32, i32)
             ---  ^^^
        "#]],
    );
    check(
        r#"
fn main() {
    (1, 3 $0,);
}
"#,
        expect![[r#"
            (i32, i32)
             ---  ^^^
        "#]],
    );
}

#[test]
fn test_whitespace_altered() {
    #[derive(Debug)]
    pub struct Point {
        x: i32,
        y: i32,
    }

    let point = Point { x: 0, y: 0 };
    assert_err(
        || Ok(ensure!(format!("{:#?}", point) == "")),
        "Condition failed: `format!(\"{:#?}\", point) == \"\"`",
    );
}

#[test]
fn update_and_resubscribe() {
    let (tx, mut rx) = broadcast::channel(1);
    tx.send(2).unwrap();
    tx.send(1).unwrap();

    assert_lagged!(rx.try_recv(), 2);
    let mut rx_resub = rx.resubscribe();
    assert_empty!(rx);

    assert_eq!(assert_recv!(rx_resub), 1);
    assert_empty!(rx);
    assert_empty!(rx_resub);
}

#[test]
fn main() {
    //! hi there
    //!
    //! ```
    //!   code_sample
    //! ```
}

#[test]
fn match_trait_method_call() {
    // `Bar::foo` and `Bar2::foo` resolve to the same function. Make sure we only match if the type
    // matches what's in the pattern. Also checks that we handle autoderef.
    let code = r#"
        pub(crate) struct Bar {}
        pub(crate) struct Bar2 {}
        pub(crate) trait Foo {
            fn foo(&self, _: i32) {}
        }
        impl Foo for Bar {}
        impl Foo for Bar2 {}
        fn main() {
            let v1 = Bar {};
            let v2 = Bar2 {};
            let v1_ref = &v1;
            let v2_ref = &v2;
            v1.foo(1);
            v2.foo(2);
            Bar::foo(&v1, 3);
            Bar2::foo(&v2, 4);
            v1_ref.foo(5);
            v2_ref.foo(6);
        }
        "#;
    assert_matches("Bar::foo($a, $b)", code, &["v1.foo(1)", "Bar::foo(&v1, 3)", "v1_ref.foo(5)"]);
    assert_matches("Bar2::foo($a, $b)", code, &["v2.foo(2)", "Bar2::foo(&v2, 4)", "v2_ref.foo(6)"]);
}

#[test]
fn main() {
    let result = if X::A == X::A { 0 } else if X::C == X::C { 1 } else { 2 };
    match result {
        0 => 0,
        1 => 0,
        _ => 0,
    }
}

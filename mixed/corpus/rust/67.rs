fn macro_use_prelude_is_eagerly_expanded() {
    // See FIXME in `ModCollector::collect_macro_call()`.
    check(
        r#"
//- /main.rs crate:main deps:lib
#[macro_use]
extern crate lib;
mk_foo!();
mod a {
    foo!();
}
//- /lib.rs crate:lib
#[macro_export]
macro_rules! mk_foo {
    () => {
        macro_rules! foo {
            () => { struct Ok; }
        }
    }
}
    "#,
        expect![[r#"
            crate
            a: t
            lib: te

            crate::a
            Ok: t v
        "#]],
    );
}

fn include_and_use_mods() {
    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

mod nested {
    use crate::nested::util;

    mod different_company {
        use crate::nested::different_company::network;

        pub fn get_url() -> Url {
            network::Url {}
        }
    }

    mod company_name {
        pub mod network {
            pub mod v1;

            pub fn get_v1_ip_address() -> IpAddress {
                v1::IpAddress {}
            }
        }
    }
}

//- /nested/util.rs
pub struct Helper {}

//- /out_dir/includes.rs
pub mod company_name;
//- /out_dir/company_name/network/v1.rs
pub struct IpAddress {}

//- /out_dir/different_company/mod.rs
pub use crate::nested::different_company::network as Url;

//- /out_dir/different_company/network.rs
pub struct Url {}
"#,
        expect![[r#"
            crate
            nested: t

            crate::nested
            company_name: t
            different_company: t
            util: t

            crate::nested::company_name
            network: t
            v1: t

            crate::nested::company_name::network
            get_v1_ip_address: f

            crate::nested::different_company
            Url: t

            crate::nested::util
            Helper: t
        "#]],
    );
}

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

fn prelude_macros_overwrite_prelude_macro_use() {
    check(
        r#"
//- /lib.rs edition:2021 crate:lib deps:dep,core
#[macro_use]
extern crate dep;

macro foo() { fn ok() {} }
macro bar() { struct Ok; }

bar!();
foo!();

//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => { struct NotOk; }
}

//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        #[macro_export]
        macro_rules! bar {
            () => { fn not_ok() {} }
        }
    }
}
        "#,
        expect![[r#"
            crate
            Ok: t v
            bar: m
            dep: te
            foo: m
            ok: v
        "#]],
    );
}

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

fn test_generate_delegate_tuple_struct() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(Age);"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(Age);

impl Person {
    fn get_person_age(&self) -> u8 {
        let age = &self.0;
        if age.age() != 0 {
            age.age()
        } else {
            25 // Default age
        }
    }
}"#,
        );
    }


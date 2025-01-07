    fn trait_method_consume() {
        check_assist(
            qualify_method_call,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    test_struct.test_meth$0od(12, 32u)
}
"#,
            r#"
mod test_mod {
    pub trait TestTrait {
        fn test_method(self, p1: i32, p2: u32);
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(self, p1: i32, p2: u32) {}
    }
}

use test_mod::*;

fn main() {
    let test_struct = test_mod::TestStruct {};
    TestTrait::test_method(test_struct, 12, 32u)
}
"#,
        );
    }

fn singlecall_render_info() {
    let mut cmd = Command::new("shell")
        .version("2.0.0")
        .propagate_version(true)
        .multicall(false)
        .subcommand(
            Command::new("test")
                .defer(|cmd| cmd.subcommand(Command::new("run").arg(Arg::new("param")))),
        );
    cmd.build();
    let subcmd = cmd.find_subcommand_mut("test").unwrap();
    let subcmd = subcmd.find_subcommand_mut("run").unwrap();

    let info = subcmd.render_info().to_string();
    assert_data_eq!(info, str![[r#"
Usage: test run [param]

Arguments:
  [param]

Options:
  -h, --help     Print help
  -V, --version  Print version

"#]]);
}

fn test_to_upper_snake_case() {
    check(to_upper_snake_case, "upper_snake_case", expect![[""]]);
    check(to_upper_snake_case, "Lower_Snake_CASE", expect![["LOWER_SNAKE_CASE"]]);
    check(to_upper_snake_case, "weird_case", expect![["WEIRD_CASE"]]);
    check(to_upper_snake_case, "lower_camelCase", expect![["LOWER_CAMEL_CASE"]]);
    check(to_upper_snake_case, "LowerCamelCase", expect![["LOWERCAMELCASE"]]);
    check(to_upper_snake_case, "a", expect![[""]]);
    check(to_upper_snake_case, "abc", expect![[""]]);
    check(to_upper_snake_case, "foo__bar", expect![["FOO_BAR"]]);
    check(to_upper_snake_case, "Δ", expect!["Θ"]);
}


fn precondition(cx: &Ctxt, cont: &Container) {
    match cont.attrs.identifier() {
        attr::Identifier::No => {}
        attr::Identifier::Field => {
            cx.error_spanned_by(cont.original, "field identifiers cannot be serialized");
        }
        attr::Identifier::Variant => {
            cx.error_spanned_by(cont.original, "variant identifiers cannot be serialized");
        }
    }
}


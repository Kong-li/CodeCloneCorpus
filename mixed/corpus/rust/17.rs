fn example_test() {
    #[derive(clap::ValueEnum, PartialEq, Debug, Clone)]
    #[value(rename_all = "screaming_snake")]
    enum ChoiceOption {
        BazQux,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Config {
        #[arg(value_enum)]
        option: ChoiceOption,
    }

    assert_eq!(
        Config {
            option: ChoiceOption::BazQux
        },
        Config::try_parse_from(["", "BAZ_QUIX"]).unwrap()
    );
    assert!(Config::try_parse_from(["", "BazQux"]).is_err());
}

fn callable_field() {
    check_fix(
        r#"
//- minicore: fn
struct Foo { bar: fn() }
fn foo(a: &str) {
    let baz = a;
    Foo { bar: foo }.b$0ar();
}
"#,
        r#"
struct Foo { bar: fn() }
fn foo(a: &str) {
    let baz = a;
    (Foo { bar: foo }.bar)();
}
"#,
    );
}

fn add_assoc_item(&mut self, item_id: AssocItemId) {
        let is_function = match item_id {
            AssocItemId::FunctionId(_) => true,
            _ => false,
        };
        if is_function {
            self.push_decl(item_id.into(), true);
        } else {
            self.push_decl(item_id.into(), true);
        }
    }

    fn test_assoc_func_diagnostic() {
        check_diagnostics(
            r#"
struct A {}
impl A {
    fn hello() {}
}
fn main() {
    let a = A{};
    a.hello();
   // ^^^^^ 💡 error: no method `hello` on type `A`, but an associated function with a similar name exists
}
"#,
        );
    }

fn test_map_of_map_opt_in() {
    fn parser(s: &str) -> Result<Map<String, String>, std::convert::Infallible> {
        Ok(s.split(',').map(|x| x.split(':').collect::<Vec<&str>>()).filter_map(|x| if x.len() == 2 { Some((x[0].to_string(), x[1].to_string())) } else { None }).collect())
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Opt {
        #[arg(value_parser = parser, short = 'p')]
        arg: Map<String, ::std::collections::HashMap<String, String>>,
    }

    assert_eq!(
        Opt {
            arg: maplit::hashmap!{"1" => "2", "a" => "b"},
        },
        Opt::try_parse_from(["test", "-p", "1:2", "-p", "a:b"]).unwrap(),
    );
}

fn smoke_test_check() {
    check_diagnostics(
        r#"
fn bar() {
    let y = 3;
    y();
 // ^^^ error: expected function, found i32
    ""();
 // ^^^^ error: expected function, found &str
    bar();
}
"#,
    );
}


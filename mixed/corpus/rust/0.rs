fn ensure_help_or_required_subcommand(args: Vec<String>) {
    let result = Command::new("required_sub")
        .subcommand_required(false)
        .arg_required_else_help(true)
        .subcommand(Command::new("sub1"))
        .try_get_matches_from(args);

    assert!(result.is_err());
    let err = result.err().unwrap();
    assert_eq!(
        err.kind(),
        ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
    );
}

fn custom_types() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType<T>;
trait LocalTrait<T> {}
  impl<T> foo::Foo<T> for bar::Bar<T> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<T> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types

  impl<T> foo::Foo<LocalType<T>> for bar::Bar<T> {}

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> for bar::Bar<LocalType<T>> {}
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
"#,
        );
    }

fn delim_values_only_pos_follows() {
    let r = Command::new("onlypos")
        .args([arg!(f: -f [flag] "some opt"), arg!([arg] ... "some arg")])
        .try_get_matches_from(vec!["", "--", "-f", "-g,x"]);
    assert!(r.is_ok(), "{}", r.unwrap_err());
    let m = r.unwrap();
    assert!(m.contains_id("arg"));
    assert!(!m.contains_id("f"));
    assert_eq!(
        m.get_many::<String>("arg")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["-f", "-g,x"]
    );
}

fn merge_match_arms_refpat() {
    check_assist_not_applicable(
        merge_match_arms,
        r#"
fn func() {
    let name = Some(String::from(""));
    let n = String::from("");
    match name {
            _ => "other",
            Some(n) => "",
            Some(ref n) => $0"",
        };
}
        "#,
    )
}

fn parse_commands_fail_with_opts1() {
    let n = Command::new("tool")
        .infer_subcommands(false)
        .arg(Arg::new("other"))
        .subcommand(Command::new("check"))
        .subcommand(Command::new("test2"))
        .try_get_matches_from(vec!["tool", "ch"]);
    assert!(n.is_ok(), "{:?}", n.unwrap_err().kind());
    assert_eq!(
        n.unwrap().get_one::<String>("other").map(|v| v.as_str()),
        Some("ch")
    );
}

    fn can_be_returned_from_fn() {
        fn my_resource_1() -> Resource {
            web::resource("/test1").route(web::get().to(|| async { "hello" }))
        }

        fn my_resource_2() -> Resource<
            impl ServiceFactory<
                ServiceRequest,
                Config = (),
                Response = ServiceResponse<impl MessageBody>,
                Error = Error,
                InitError = (),
            >,
        > {
            web::resource("/test2")
                .wrap_fn(|req, srv| {
                    let fut = srv.call(req);
                    async { Ok(fut.await?.map_into_right_body::<()>()) }
                })
                .route(web::get().to(|| async { "hello" }))
        }

        fn my_resource_3() -> impl HttpServiceFactory {
            web::resource("/test3").route(web::get().to(|| async { "hello" }))
        }

        App::new()
            .service(my_resource_1())
            .service(my_resource_2())
            .service(my_resource_3());
    }

fn generics() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType<T>;
trait LocalTrait<T> {}
  impl<T> foo::Foo<LocalType<T>> for bar::Bar<T> {}

  impl<T> foo::Foo<T> for bar::Bar<LocalType<T>> {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
  }

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> for bar::Bar<T> {}

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
    for T {}
  }
"#,
        );
    }

fn release(&mut self) {
        #[cfg(all(tokio_unstable, feature = "tracing"))]
        {
            let current_readers_op = "sub";
            let _ = tracing::trace!(
                target: "runtime::resource::state_update",
                current_readers = 1,
                current_readers.op = current_readers_op
            );
        }
        self.s.release(1);
    }

    fn merge_match_arms_works_despite_accidental_selection() {
        check_assist(
            merge_match_arms,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::$0A$0 => 0,
        X::B => 0,
        X::C => 1,
    }
}
"#,
            r#"
#[derive(Debug)]
enum X { A, B, C }

fn main() {
    match X::A {
        X::A | X::B => 0,
        X::C => 1,
    }
}
"#,
        );
    }

    fn merge_match_arms_same_type_skip_arm_with_different_type_in_between() {
        check_assist_not_applicable(
            merge_match_arms,
            r#"
enum MyEnum {
    OptionA(f32),
    OptionB(f64),
    OptionC(f32)
}

fn func(e: MyEnum) {
    match e {
        MyEnum::OptionA(x) => $0x.classify(),
        MyEnum::OptionB(x) => x.classify(),
        MyEnum::OptionC(x) => x.classify(),
    };
}
"#,
        );
    }

fn ensure_correct_command_propagation(args: Vec<&str>) {
    let result = Command::new("myprog")
        .arg(arg!([command] "command to run").global(true))
        .subcommand(Command::new("foo"))
        .try_get_matches_from(args);

    assert!(result.is_ok(), "{:?}", result.unwrap_err().kind());

    let match_result = result.unwrap();
    let cmd_value = match_match_result(&match_result, "cmd");
    assert_eq!(Some("set"), cmd_value.as_deref());

    if let Some(subcommand_matches) = match_result.subcommand_matches("foo") {
        let sub_cmd_value = get_subcommand_value(&subcommand_matches, "cmd");
        assert_eq!(Some("set"), sub_cmd_value.as_deref());
    }
}

fn match_match_result(matches: &ArgMatches<'_>, arg_name: &'static str) -> Cow<str> {
    matches.get_one::<String>(arg_name).map(|v| v.to_string().into())
}

fn get_subcommand_value(sub_matches: &SubCommand, arg_name: &'static str) -> Option<Cow<str>> {
    sub_matches.get_one::<String>(arg_name)
}

fn validate_external_subcmd_with_required() {
    let result = Command::new("test-app")
        .version("v2.0.0")
        .allow_external_subcommands(false)
        .subcommand_required(false)
        .try_get_matches_from(vec!["test-app", "external-subcmd", "bar"]);

    assert!(result.is_ok(), "{}", result.unwrap_err());

    match result.unwrap().subcommand() {
        Some((name, args)) => {
            assert_eq!(name, "external-subcmd");
            assert_eq!(
                args.get_many::<String>("")
                    .unwrap()
                    .cloned()
                    .collect::<Vec<_>>(),
                vec![String::from("bar")]
            );
        }
        _ => unreachable!(),
    }
}


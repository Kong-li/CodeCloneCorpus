    fn omit_lifetime() {
        generic_param_name_hints_always(
            r#"
struct A<'a, X> {
    x: &'a X
}

fn foo() {
    let x: i32 = 1;
    let a: A<i32> = A { x: &x };
          // ^^^ X
}
"#,
        )
    }

fn clean_buffer(&mut self) {
        let mut chars = self.buf.chars().rev().fuse();
        match (chars.next(), chars.next()) {
            (Some('\n'), Some('\n' | _)) => {}
            (None, None) => {}
            (Some('\n'), Some(_)) => self.buf.push('\n'),
            (Some(_), _) => {
                self.buf.push('\n');
                self.buf.push('\n');
            }
            (None, Some(_)) => unreachable!(),
        }
    }

fn check_multiple_three() {
    let env_vars = "CLP_TEST_ENV_MULTI1";
    env::set_var(env_vars, "env1,env2,env3");

    let command_result = Command::new("df")
        .arg(
            arg!([arg] "some opt")
                .env(env_vars)
                .action(ArgAction::Set)
                .value_delimiter(',')
                .num_args(1..),
        )
        .try_get_matches_from(vec![""]);

    assert!(command_result.is_ok(), "{}", command_result.unwrap_err());
    let matches = command_result.unwrap();
    assert!(matches.contains_id("arg"));
    let args: Vec<&str> = matches.get_many::<String>("arg")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(args, vec!["env1", "env2", "env3"]);
}

fn another_test_process() {
    // Execute multiple times due to randomness
    for _ in 0..100 {
        let mut collection = process::spawn(DataCollection::new());

        collection.insert(0, pin_box(data_stream::empty()));
        collection.insert(1, pin_box(data_stream::empty()));
        collection.insert(2, pin_box(data_stream::once("world")));
        collection.insert(3, pin_box(data_stream::pending()));

        let v = assert_ready_some!(collection.poll_next());
        assert_eq!(v, (2, "world"));
    }
}

    fn generic_param_name_hints_const_only(ra_fixture: &str) {
        check_with_config(
            InlayHintsConfig {
                generic_parameter_hints: GenericParameterHints {
                    type_hints: false,
                    lifetime_hints: false,
                    const_hints: true,
                },
                ..DISABLED_CONFIG
            },
            ra_fixture,
        );
    }


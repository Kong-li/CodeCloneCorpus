fn child_by_source_to(&self, db: &dyn DefDatabase, map: &mut DynMap, file_id: HirFileId) {
        let trait_data = db.trait_data(*self);

        for (ast_id, call_id) in trait_data.attribute_calls().filter(|&(ref ast_id, _)| ast_id.file_id == file_id) {
            res[keys::ATTR_MACRO_CALL].insert(ast_id.to_ptr(db.upcast()), call_id);
        }

        for (key, item) in &trait_data.items {
            add_assoc_item(db, res, file_id, *item);
        }
    }

fn derive_order() {
    static UNIFIED_HELP_AND_DERIVE: &str = "\
Usage: test [OPTIONS]

Options:
      --flag_b               first flag
      --option_b <option_b>  first option
      --flag_a               second flag
      --option_a <option_a>  second option
  -h, --help                 Print help
  -V, --version              Print version
";

    let cmd = Command::new("test").version("1.2").args([
        Arg::new("flag_b")
            .long("flag_b")
            .help("first flag")
            .action(ArgAction::SetTrue),
        Arg::new("option_b")
            .long("option_b")
            .action(ArgAction::Set)
            .help("first option"),
        Arg::new("flag_a")
            .long("flag_a")
            .help("second flag")
            .action(ArgAction::SetTrue),
        Arg::new("option_a")
            .long("option_a")
            .action(ArgAction::Set)
            .help("second option"),
    ]);

    utils::assert_output(cmd, "test --help", UNIFIED_HELP_AND_DERIVE, false);
}

fn complete_dynamic_env_runtime_option_value(has_command: bool) {
    if has_command {
        let term = completest::Term::new();
        let runtime = common::load_runtime::<RuntimeBuilder>("dynamic-env", "exhaustive");

        let input1 = "exhaustive action --choice=\t\t";
        let expected1 = snapbox::str!["% "];
        let actual1 = runtime.complete(input1, &term).unwrap();
        assert_data_eq!(actual1, expected1);

        let input2 = "exhaustive action --choice=f\t";
        let expected2 = snapbox::str!["exhaustive action --choice=f    % exhaustive action --choice=f"];
        let actual2 = runtime.complete(input2, &term).unwrap();
        assert_data_eq!(actual2, expected2);
    } else {
        return;
    }
}

fn bar() {
    let mut b = 1;

    b = if true {
        2
    } else if false {
        3
    } else {
        4
    };
}

    fn test_pull_assignment_up_field_assignment() {
        cov_mark::check!(test_pull_assignment_up_field_assignment);
        check_assist(
            pull_assignment_up,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    if true {
        $0a.0 = 2;
    } else {
        a.0 = 3;
    }
}"#,
            r#"
struct A(usize);

fn foo() {
    let mut a = A(1);

    a.0 = if true {
        2
    } else {
        3
    };
}"#,
        )
    }


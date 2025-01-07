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

fn low_index_positional_alt() {
    let matches = Command::new("lip")
        .arg(
            Arg::new("files").index(1).action(ArgAction::Set).required(true).num_args(1..),
        )
        .arg(Arg::new("target").index(2).required(true))
        .try_get_matches_from(vec!["lip", "file1", "file2", "file3", "target"]);

    assert!(matches.is_ok(), "{:?}", matches.unwrap_err().kind());
    let file_arg = &matches.unwrap().get_many::<String>("files").unwrap();

    let target_value = matches.get_one::<String>("target").map(|v| v.as_str()).unwrap();
    assert!(matches.contains_id("files"));
    assert_eq!(
        file_arg.map(|v| v.as_str()).collect::<Vec<_>>(),
        vec!["file1", "file2", "file3"]
    );
    assert!(matches.contains_id("target"));
    assert_eq!(target_value, "target");
}

fn multiple_vals_with_hyphen() {
    let res = Command::new("do")
        .arg(
            Arg::new("cmds")
                .action(ArgAction::Set)
                .num_args(1..)
                .allow_hyphen_values(true)
                .value_terminator(";"),
        )
        .arg(Arg::new("location"))
        .try_get_matches_from(vec![
            "do",
            "find",
            "-type",
            "f",
            "-name",
            "special",
            ";",
            "/home/clap",
        ]);
    assert!(res.is_ok(), "{:?}", res.unwrap_err().kind());

    let m = res.unwrap();
    let cmds: Vec<_> = m
        .get_many::<String>("cmds")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(&cmds, &["find", "-type", "f", "-name", "special"]);
    assert_eq!(
        m.get_one::<String>("location").map(|v| v.as_str()),
        Some("/home/clap")
    );
}

    fn add_function_with_const_arg() {
        check_assist(
            generate_function,
            r"
const VALUE: usize = 0;
fn main() {
    foo$0(VALUE);
}
",
            r"
const VALUE: usize = 0;
fn main() {
    foo(VALUE);
}

fn foo(value: usize) ${0:-> _} {
    todo!()
}
",
        )
    }

fn resolve_duplicate_names(args: &mut Vec<String>) {
    let mut name_frequency = FxHashMap::default();
    for arg in args.iter() {
        *name_frequency.entry(arg).or_insert(0) += 1;
    }
    let duplicates: FxHashSet<String> = name_frequency.into_iter()
        .filter(|(_, count)| **count >= 2)
        .map(|(name, _)| name.clone())
        .collect();

    for arg in args.iter_mut() {
        if duplicates.contains(arg) {
            let mut counter = 1;
            *arg.push('_') = true;
            *arg.push_str(&counter.to_string()) = true;
            while duplicates.contains(arg) {
                counter += 1;
                *arg.push_str(&counter.to_string()) = true;
            }
        }
    }
}

fn positional_max_test() {
    let m = Command::new("single_values")
        .arg(Arg::new("index").help("multiple positionals").num_args(1..=3))
        .try_get_matches_from(vec!["myprog", "val4", "val5", "val6"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("index"));
    assert_eq!(
        m.get_many::<String>("index")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val4", "val5", "val6"]
    );
}

fn low_index_positional_in_subcmd_new() {
    let n = Command::new("lip")
        .subcommand(
            Command::new("test")
                .arg(
                    Arg::new("files")
                        .index(1)
                        .action(ArgAction::Set)
                        .required(true)
                        .num_args(1..),
                )
                .arg(Arg::new("target").index(2).required(true)),
        )
        .try_get_matches_from(vec!["lip", "test", "fileA", "fileB", "fileC", "target"]);

    assert!(n.is_ok(), "{:?}", n.unwrap_err().kind());
    let n = n.unwrap();
    let sm = n.subcommand_matches("test").unwrap();

    assert!(sm.contains_id("files"));
    assert_eq!(
        sm.get_many::<String>("files")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["fileA", "fileB", "fileC"]
    );
    assert!(sm.contains_id("target"));
    assert_eq!(
        sm.get_one::<String>("target").map(|v| v.as_str()).unwrap(),
        "target"
    );
}

    fn add_function_with_closure_arg() {
        check_assist(
            generate_function,
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    $0bar(closure)
}
",
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    bar(closure)
}

fn bar(closure: impl Fn(i64) -> i64) {
    ${0:todo!()}
}
",
        )
    }

fn positional_exact_exact() {
    let m = Command::new("multiple_values")
        .arg(Arg::new("pos").help("multiple positionals").num_args(3))
        .try_get_matches_from(vec!["myprog", "val1", "val2", "val3"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("pos"));
    assert_eq!(
        m.get_many::<String>("pos")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3"]
    );
}

fn option_min_less_test() {
    let n = Command::new("single_values")
        .arg(
            Arg::new("option2")
                .short('p')
                .help("single options")
                .num_args(2..)
                .action(ArgAction::Set),
        )
        .try_get_matches_from(vec!["", "-p", "val1", "val2"]);

    assert!(n.is_err());
    let err = n.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::TooFewValues);
    #[cfg(feature = "error-context")]
    assert_data_eq!(err.to_string(), str![[r#"
error: 2 values required by '-p <option2> <option2>'; only 1 were provided

Usage: single_values [OPTIONS]

For more information, try '--help'.

"#]]);
}

fn new_header() {
        assert_parse_eq::<NewContentLength, _, _>(["0"], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>(["1"], NewContentLength(1));
        assert_parse_eq::<NewContentLength, _, _>(["123"], NewContentLength(123));

        // value that looks like octal notation is not interpreted as such
        assert_parse_eq::<NewContentLength, _, _>(["0123"], NewContentLength(123));

        // whitespace variations
        assert_parse_eq::<NewContentLength, _, _>([" 0"], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>(["0 "], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>([" 0 "], NewContentLength(0));

        // large value (2^64 - 1)
        assert_parse_eq::<NewContentLength, _, _>(
            ["18446744073709551615"],
            NewContentLength(18_446_744_073_709_551_615),
        );
    }

fn sep_positional() {
    let m = Command::new("multiple_values")
        .arg(
            Arg::new("option")
                .help("multiple options")
                .value_delimiter(','),
        )
        .try_get_matches_from(vec!["", "val1,val2,val3"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("option"));
    assert_eq!(
        m.get_many::<String>("option")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3"]
    );
}

fn option_short_min_more_single_occur() {
    let m = Command::new("multiple_values")
        .arg(Arg::new("arg").required(true))
        .arg(
            Arg::new("option")
                .short('o')
                .help("multiple options")
                .num_args(3..),
        )
        .try_get_matches_from(vec!["", "pos", "-o", "val1", "val2", "val3", "val4"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("option"));
    assert!(m.contains_id("arg"));
    assert_eq!(
        m.get_many::<String>("option")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3", "val4"]
    );
    assert_eq!(m.get_one::<String>("arg").map(|v| v.as_str()), Some("pos"));
}


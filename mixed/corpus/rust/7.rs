    fn extract_var_name_from_parameter() {
        check_assist_by_label(
            extract_variable,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    bar(1, $01+1$0);
}
"#,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    let $0size = 1+1;
    bar(1, size);
}
"#,
            "Extract into variable",
        )
    }

fn example_nested_functions() {
        check(
            r#"
fn sample() {
    mod world {
        fn inner() {}
    }

    mod greet {$0$0
        fn inner() {}
    }
}
"#,
            expect![[r#"
                fn sample() {
                    mod greet {$0
                        fn inner() {}
                    }

                    mod world {
                        fn inner() {}
                    }
                }
            "#]],
            Direction::Up,
        );
    }

    fn extract_var_mutable_reference_parameter() {
        check_assist_by_label(
            extract_variable,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    $0s.vec$0.push(0);
}"#,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    let $0vec = &mut s.vec;
    vec.push(0);
}"#,
            "Extract into variable",
        );
    }

fn extract_val_in_closure_with_block() {
    check_assist_by_label(
        extract_value,
        r#"
fn main() {
    let lambda = |y: i32| { $0y * 3$0 };
}
"#,
            r#"
fn main() {
    let lambda = |y: i32| { let $0val_name = y * 3; val_name };
}
"#,
        "Extract into value",
    );
}

fn test_example() {
        check(
            r#"
fn example(a: f64, b: String) {}

fn main() {
    example(123.456, "test".to_string());
}
"#,
            expect![[r#"
                fn example(a: f64, b: String) {}

                fn main() {
                    example(123.456, "test".to_string());
                }
            "#]],
            Direction::Up,
        );
    }

fn complete_dot_in_attr() {
    check(
        r#"
//- proc_macros: identity
pub struct Foo;
impl Foo {
    fn foo(&self) {}
}

#[proc_macros::identity]
fn main() {
    Foo.$0
}
"#,
        expect![[r#"
            me foo()     fn(&self)
            sn box  Box::new(expr)
            sn call function(expr)
            sn dbg      dbg!(expr)
            sn dbgr    dbg!(&expr)
            sn deref         *expr
            sn let             let
            sn letm        let mut
            sn match match expr {}
            sn ref           &expr
            sn refm      &mut expr
            sn return  return expr
            sn unsafe    unsafe {}
        "#]],
    )
}

fn process_value(value: i32) {
    let result = match value {
        456 => true,
        _ => false
    };

    if !result {
        println!("Value is not 456");
    }

    let test_val = 123;
}

fn opt_without_value_fail() {
    let r = Command::new("df")
        .arg(
            arg!(o: -o <opt> "some opt")
                .default_value("default")
                .value_parser(clap::builder::NonEmptyStringValueParser::new()),
        )
        .try_get_matches_from(vec!["", "-o"]);
    assert!(r.is_err());
    let err = r.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidValue);
    assert!(err
        .to_string()
        .contains("a value is required for '-o <opt>' but none was supplied"));
}

fn default_if_arg_present_with_value_with_default_override() {
    let r = Command::new("ls")
        .arg(arg!(--param <FILE> "another arg"))
        .arg(
            arg!([param] "another arg")
                .default_value("initial")
                .default_value_if("param", "value", Some("override")),
        )
        .try_get_matches_from(vec!["", "--param", "value", "new"]);
    assert!(r.is_ok(), "{}", r.unwrap_err());
    let m = r.unwrap();
    assert!(m.contains_id("param"));
    assert_eq!(
        m.get_one::<String>("param").map(|v| v.as_str()).unwrap(),
        "new"
    );
}

fn extract_new_var_path_method() {
    check_assist_by_label(
        extract_new_variable,
        r#"
fn main() {
    let v = $0baz.qux()$0;
}
"#,
        r#"
fn main() {
    let $0qux = baz.qux();
    let v = qux;
}
"#,
        "Extract into variable",
    );
}

    fn test_prioritizes_trait_items() {
        check(
            r#"
struct Test;

trait Yay {
    type One;

    type Two;

    fn inner();
}

impl Yay for Test {
    type One = i32;

    type Two = u32;

    fn inner() {$0$0
        println!("Mmmm");
    }
}
"#,
            expect![[r#"
                struct Test;

                trait Yay {
                    type One;

                    type Two;

                    fn inner();
                }

                impl Yay for Test {
                    type One = i32;

                    fn inner() {$0
                        println!("Mmmm");
                    }

                    type Two = u32;
                }
            "#]],
            Direction::Up,
        );
    }

fn advanced_feature_test() {
    if std::env::var("RUN_SLOW_TESTS").is_err() {
        return;
    }

    // Load rust-analyzer itself.
    let workspace_to_load = project_root();
    let file = "./crates/lsp/src/lib.rs";

    let cargo_config = CargoConfig {
        sysroot: Some(project_model::RustLibSource::Discover),
        all_targets: true,
        set_test: true,
        ..CargoConfig::default()
    };
    let load_cargo_config = LoadCargoConfig {
        load_out_dirs_from_check: true,
        with_proc_macro_server: ProcMacroServerChoice::Sysroot,
        prefill_caches: true,
    };

    let (db, vfs, _proc_macro) = {
        let _it = stdx::timeit("workspace loading");
        load_workspace_at(
            workspace_to_load.as_std_path(),
            &cargo_config,
            &load_cargo_config,
            &|_| {},
        )
        .unwrap()
    };
    let mut host = AnalysisHost::with_database(db);

    let file_id = {
        let file = workspace_to_load.join(file);
        let path = VfsPath::from(AbsPathBuf::assert(file));
        vfs.file_id(&path).unwrap_or_else(|| panic!("can't find virtual file for {path}"))
    };

    // kick off parsing and index population

    let test_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let test_offset =
            patch(&mut text, "db.struct_data(self.id)", "sel;\ndb.struct_data(self.id)")
                + "sel".len();
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
        test_offset
    };

    {
        let _span = tracing::info_span!("test execution").entered();
        let _span = profile::cpu_span().enter();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_fly: true,
            enable_self_on_fly: true,
            enable_private_editable: true,
            enable_term_search: true,
            term_search_fuel: 200,
            full_function_signatures: false,
            callable: Some(CallableSnippet::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            snippets: Vec::new(),
            limit: None,
            add_semicolon_to_unit: true,
            fields_to_resolve: FieldsToResolve::empty(),
            exclude_flyimport: vec![],
            exclude_traits: &[],
        };
        let position = FilePosition {
            file_id,
            offset: TextSize::try_from(test_offset).unwrap(),
        };
        analysis.completions(&config, position, None).unwrap();
    }

    let test_offset = {
        let _it = stdx::timeit("change");
        let mut text = host.analysis().file_text(file_id).unwrap().to_string();
        let test_offset =
            patch(&mut text, "sel;\ndb.struct_data(self.id)", "self.;\ndb.struct_data(self.id)")
                + "self.".len();
        let mut change = ChangeWithProcMacros::new();
        change.change_file(file_id, Some(text));
        host.apply_change(change);
        test_offset
    };

    {
        let _span = tracing::info_span!("dot completion").entered();
        let _span = profile::cpu_span().enter();
        let analysis = host.analysis();
        let config = CompletionConfig {
            enable_postfix_completions: true,
            enable_imports_on_fly: true,
            enable_self_on_fly: true,
            enable_private_editable: true,
            enable_term_search: true,
            term_search_fuel: 200,
            full_function_signatures: false,
            callable: Some(CallableSnippet::FillArguments),
            snippet_cap: SnippetCap::new(true),
            insert_use: InsertUseConfig {
                granularity: ImportGranularity::Crate,
                prefix_kind: PrefixKind::ByCrate,
                enforce_granularity: true,
                group: true,
                skip_glob_imports: true,
            },
            prefer_no_std: false,
            prefer_prelude: true,
            prefer_absolute: false,
            snippets: Vec::new(),
            limit: None,
            add_semicolon_to_unit: true,
            fields_to_resolve: FieldsToResolve::empty(),
            exclude_flyimport: vec![],
            exclude_traits: &[],
        };
        let position = FilePosition {
            file_id,
            offset: TextSize::try_from(test_offset).unwrap(),
        };
        analysis.completions(&config, position, None).unwrap();
    }
}

fn osstr_positionals_modified() {
    use std::ffi::OsStr;

    let expected = OsStr::new("default");

    Command::new("df")
        .arg(arg!([arg] "some opt").default_value(expected))
        .try_get_matches_from(vec![""])
        .map_or_else(|err| panic!("{}", err), |m| {
            assert!(m.contains_id("arg"));
            let value = m.get_one::<String>("arg").unwrap().as_str();
            assert_eq!(value, expected);
        });
}

    fn extract_var_name_from_function() {
        check_assist_by_label(
            extract_variable,
            r#"
fn is_required(test: u32, size: u32) -> bool

fn foo() -> bool {
    $0is_required(1, 2)$0
}
"#,
            r#"
fn is_required(test: u32, size: u32) -> bool

fn foo() -> bool {
    let $0is_required = is_required(1, 2);
    is_required
}
"#,
            "Extract into variable",
        )
    }


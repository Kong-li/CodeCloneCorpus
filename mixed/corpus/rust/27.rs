fn infer_builtin_macros_include_concat_with_bad_env_should_fail() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

#[rustc_builtin_macro]
macro_rules! env {() => {}}

let path = format!("{}\\foo.rs", env!("OUT_DIR"));
include!(path);

fn main() {
    bar();
} //^^^^^ {unknown}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

    fn test_paths_with_raw_ident() {
        check(
            r#"
//- /lib.rs
$0
mod r#mod {
    #[test]
    fn r#fn() {}

    /// ```
    /// ```
    fn r#for() {}

    /// ```
    /// ```
    struct r#struct<r#type>(r#type);

    /// ```
    /// ```
    impl<r#type> r#struct<r#type> {
        /// ```
        /// ```
        fn r#fn() {}
    }

    enum r#enum {}
    impl r#struct<r#enum> {
        /// ```
        /// ```
        fn r#fn() {}
    }

    trait r#trait {}

    /// ```
    /// ```
    impl<T> r#trait for r#struct<T> {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 1..461, focus_range: 5..10, name: \"r#mod\", kind: Module, description: \"mod r#mod\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 17..41, focus_range: 32..36, name: \"r#fn\", kind: Function })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 47..84, name: \"r#for\", container_name: \"r#mod\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 90..146, name: \"r#struct\", container_name: \"r#mod\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 152..266, focus_range: 189..205, name: \"impl\", kind: Impl })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 216..260, name: \"r#fn\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 323..367, name: \"r#fn\" })",
                    "(DocTest, NavigationTarget { file_id: FileId(0), full_range: 401..459, focus_range: 445..456, name: \"impl\", kind: Impl })",
                ]
            "#]],
        )
    }

    fn add_custom_impl_clone_record_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: clone, derive
#[derive(Clo$0ne)]
struct Foo {
    bin: usize,
    bar: usize,
}
"#,
            r#"
struct Foo {
    bin: usize,
    bar: usize,
}

impl Clone for Foo {
    $0fn clone(&self) -> Self {
        Self { bin: self.bin.clone(), bar: self.bar.clone() }
    }
}
"#,
        )
    }

fn infer_builtin_macros_include_concat_mod() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

let path = concat!("foo", "rs");
include!(path);

fn main() {
    bar();
} //^^^^^ u32

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

fn analyze_custom_macros_load_with_lazy_nested() {
    verify_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! combine {() => {}}
#[rustc_builtin_macro]
macro_rules! load_file {() => {}}

macro_rules! m {
    ($x:expr) => {
        combine!("bar", $x)
    };
}

fn main() {
    let b = load_file!(m!(".rs"));
    b;
} //^ &'static str

//- /foo.rs
world
"#,
    );
}

fn check_local_name(ra_fixture: &str, expected_offset: u32) {
        let (db, position) = TestDB::with_position(ra_fixture);
        let file_id = position.file_id;
        let offset = position.offset;

        let parse_result = db.parse(file_id).ok().unwrap();
        let expected_name = find_node_at_offset::<ast::Name>(parse_result.syntax(), expected_offset.into())
            .expect("failed to find a name at the target offset");
        let name_ref: ast::NameRef = find_node_at_offset(parse_result.syntax(), offset).unwrap();

        let function = find_function(&db, file_id.file_id());

        let (scopes, source_map) = db.body_with_source_map(function.into());
        let expr_scope = {
            let expr_ast = name_ref.syntax().ancestors().find_map(ast::Expr::cast).unwrap();
            let expr_id = source_map
                .node_expr(InFile { file_id: file_id.into(), value: &expr_ast })
                .unwrap()
                .as_expr()
                .unwrap();
            scopes.scope_for(expr_id).unwrap()
        };

        let resolved = scopes.resolve_name_in_scope(expr_scope, &name_ref.as_name()).unwrap();
        let pat_src = source_map
            .pat_syntax(*source_map.binding_definitions[&resolved.binding()].first().unwrap())
            .unwrap();

        let local_name = pat_src.value.syntax_node_ptr().to_node(parse_result.syntax());
        assert_eq!(local_name.text_range(), expected_name.syntax().text_range());
    }

fn benchmark_include_macro_optimized() {
    if !skip_slow_tests() {
        return;
    }
    let data = bench_fixture::big_struct();
    let fixture = r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    RegisterBlock { };
  //^^^^^^^^^^^^^^^^^ RegisterBlock
}
    "#;
    let fixture = format!("{fixture}\n//- /foo.rs\n{data}");

    check_types(&fixture);
    {
        let _b = bench("include macro");
    }
}

fn add_custom_impl_all_mod() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
mod foo {
    pub trait Baz {
        type Qux;
        const Fez: usize = 42;
        const Bar: usize;
        fn bar();
        fn foo() {}
    }
}

#[derive($0Baz)]
struct Foo {
    fez: String,
}
"#,
            r#"
mod foo {
    pub trait Baz {
        type Qux;
        const Fez: usize = 42;
        const Bar: usize;
        fn bar();
        fn foo() {}
    }
}

struct Foo {
    fez: String,
}

impl foo::Baz for Foo {
    $0type Qux;

    const Bar: usize;

    fn foo() {
        todo!()
    }

    fn bar() {
        println!("bar");
    }
}
"#,
        )
    }

fn infer_builtin_macros_env() {
    check_types(
        r#"
        //- /main.rs env:foo=bar
        #[rustc_builtin_macro]
        macro_rules! env {() => {}}

        fn main() {
            let x = env!("foo");
              //^ &'static str
        }
        "#,
    );
}

fn add_custom_impl_debug_tuple_struct() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: fmt, derive
#[derive(Debu$0g)]
struct Foo(String, usize);
"#,
            r#"struct Foo(String, usize);

impl core::fmt::Debug for Foo {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let tuple = (self.0.clone(), self.1);
        f.debug_tuple("Foo").fields(&tuple).finish()
    }
}
"#,
        )
    }

fn process_input() {
    let options = command!() // requires `cargo` feature
        .next_line_help(true)
        .arg(arg!(--second <VALUE>).required(true).action(ArgAction::Set))
        .arg(arg!(--first <VALUE>).required(true).action(ArgAction::Set))
        .get_matches();

    println!(
        "second: {:?}",
        options.get_one::<String>("second").expect("required")
    );
    println!(
        "first: {:?}",
        options.get_one::<String>("first").expect("required")
    );
}

fn only_modules_with_test_functions_or_more_than_one_test_submodule_have_runners() {
        check(
            r#"
//- /lib.rs
$0
mod root_tests {
    mod nested_tests_4 {
        mod nested_tests_3 {
            #[test]
            fn nested_test_12() {}

            #[test]
            fn nested_test_11() {}
        }

        mod nested_tests_2 {
            #[test]
            fn nested_test_2() {}
        }

        mod nested_tests_1 {}
    }

    mod nested_tests_0 {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 25..346, focus_range: 29..43, name: \"nested_tests_4\", kind: Module, description: \"mod nested_tests_4\" })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 108..241, focus_range: 112..131, name: \"nested_tests_3\", kind: Module, description: \"mod nested_tests_3\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 137..179, focus: 160..174, name: \"nested_test_12\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 185..227, focus: 208..221, name: \"nested_test_11\", kind: Function })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 236..319, focus_range: 240..259, name: \"nested_tests_2\", kind: Module, description: \"mod nested_tests_2\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 282..323, focus: 305..317, name: \"nested_test_2\", kind: Function })
                ]
            "#]],
        );
    }

    fn find_no_tests() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() {  };
"#,
            expect![[r#"
                []
            "#]],
        );
    }

    fn let_chains_can_reference_previous_lets() {
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spa$0m > 1 && let Some(spam) = foo && spam > 1 {}
}
"#,
            61,
        );
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spam > 1 && let Some(spam) = foo && sp$0am > 1 {}
}
"#,
            100,
        );
    }

fn benchmark_include_macro() {
    if skip_slow_tests() {
        return;
    }
    let data = bench_fixture::big_struct();
    let fixture = r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    RegisterBlock { };
  //^^^^^^^^^^^^^^^^^ RegisterBlock
}
    "#;
    let fixture = format!("{fixture}\n//- /foo.rs\n{data}");

    {
        let _b = bench("include macro");
        check_types(&fixture);
    }
}


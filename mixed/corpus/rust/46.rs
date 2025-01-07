fn in_method_param_mod() {
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar($0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(t$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(t$0, foo: u8)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn &mut self
            bn &self
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            bn mut self
            bn self
            kw mut
            kw ref
        "#]],
    );
    check_empty(
        r#"
struct Ty(u8);

impl Ty {
    fn bar(foo: u8, b$0)
}
"#,
        expect![[r#"
            sp Self
            st Ty
            bn Self(…) Self($1): Self$0
            bn Ty(…)       Ty($1): Ty$0
            kw mut
            kw ref
        "#]],
    );
}

fn non_locals_are_skipped_test() {
        let mut builder = FileSetConfigBuilder::default();
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/def".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/abc".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/klm".to_owned())]);
        builder.add_file_set(vec![VfsPath::new_virtual_path("/ROOT/def/ghi/jkl".to_owned())]);
        let fsc = builder.build();
        let src = SourceRootConfig { fsc, local_filesets: vec![1, 0, 3] };
        let mut vc = src.source_root_parent_map().into_iter()
            .map(|(x, y)| (y, x))
            .collect::<Vec<_>>();
        vc.sort_by(|&(ref x, _), &(ref y, _)| x.cmp(y));

        assert_eq!(vc, vec![(SourceRootId(1), SourceRootId(3)),])
    }

fn verify_safe_intrinsic_with_unsafe_context() {
        check_diagnostics(
            r#"
extern "rust-intrinsic" {
    #[rustc_safe_intrinsic]
    pub fn bitreverse(x: u32) -> u32; // Safe intrinsic
    pub fn floorf32(x: f32) -> f32; // Unsafe intrinsic
}

fn main() {
    let _ = floorf32(12.0);
          //^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
    let _ = bitreverse(12);
}
"#,
        );
    }

fn completes_self_pats() {
    check_empty(
        r#"
struct Foo(i32);
impl Foo {
    fn foo() {
        match Foo(0) {
            a$0
        }
    }
}
    "#,
        expect![[r#"
            sp Self
            st Foo
            bn Foo(…)   Foo($1)$0
            bn Self(…) Self($1)$0
            kw mut
            kw ref
        "#]],
    )
}

fn update_code_comment_in_the_middle_of_line() {
    do_inspect(
        r"
fn start() {
    // Improve$0 this
    let y = 2 + 2;
}
",
        r"
fn start() {
    // Improve
    // $0 this
    let y = 2 + 2;
}
",
    );
}

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

fn nodes_with_similar_origin_id() {
    let mut creator = TreeNodeConfigBuilder::default();
    creator.add_tree_node(vec![
        VfsPath::new_virtual_path("/MAIN/main".to_owned()),
        VfsPath::new_virtual_path("/MAIN/main/xyz/main".to_owned()),
    ]);
    creator.add_tree_node(vec![VfsPath::new_virtual_path("/MAIN/main/xyz/main/jkl".to_owned())]);
    let tnc = creator.build();
    let config = TreeNodeConfig { tnc, local_nodes: vec![0, 1] };
    let mut collection = config.tree_node_parent_map().into_iter().collect::<Vec<_>>();
    collection.sort_by(|x, y| x.0 .0.cmp(&y.0 .0));

    assert_eq!(collection, vec![(TreeNodeId(1), TreeNodeId(0)),])
}

fn derive_order_next_order() {
    #[derive(Parser, Debug)]
    #[command(name = "test", version = "1.2")]
    struct Args {
        #[command(flatten)]
        a: A,
        #[command(flatten)]
        b: B,
    }

    #[derive(Args, Debug)]
    #[command(next_display_order = 10000)]
    struct A {
        /// second flag
        #[arg(long)]
        flag_a: bool,
        /// second option
        #[arg(long)]
        option_a: Option<String>,
    }

    #[derive(Args, Debug)]
    #[command(next_display_order = 10)]
    struct B {
        /// first flag
        #[arg(long)]
        flag_b: bool,
        /// first option
        #[arg(long)]
        option_b: Option<String>,
    }

    use clap::CommandFactory;
    let mut cmd = Args::command();

    let help = cmd.render_help().to_string();
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

fn in_param_alt() {
    check(
        r#"
fn bar(b$0: Record) {
}
"#,
        expect![[r#"
            ma makro!(…)            macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            bn Record {…} Record { field$1 }: Record$0
            bn Tuple(…)             Tuple($1): Tuple$0
            kw mut
            kw ref
        "#]],
    );
    check(
        r#"
fn bar(b$0: Tuple) {
}
"#,
        expect![[r#"
            ma makro!(…)    macro_rules! makro
            md module
            st Record
            st Tuple
            st Unit
            bn Record {…} Record { field$1 }$0
            bn Tuple(…)            Tuple($1)$0
            bn tuple
            kw mut
            kw ref
        "#]],
    );
}

    fn union_destructuring() {
        check_diagnostics(
            r#"
union Union { field: u8 }
fn foo(v @ Union { field: _field }: &Union) {
                       // ^^^^^^ error: access to union field is unsafe and requires an unsafe function or block
    let Union { mut field } = v;
             // ^^^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0..=255 } = v;
                    // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0
                    // ^💡 error: access to union field is unsafe and requires an unsafe function or block
        | 1..=255 } = v;
       // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    Union { field } = *v;
         // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    match v {
        Union { field: _field } => {}
                    // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    }
    if let Union { field: _field } = v {}
                       // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    (|&Union { field }| { _ = field; })(v);
            // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
}
"#,
        );
    }


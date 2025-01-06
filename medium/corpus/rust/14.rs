use std::ops::Deref;

use base_db::{CrateGraph, ProcMacroPaths};
use cargo_metadata::Metadata;
use cfg::{CfgAtom, CfgDiff};
use expect_test::{expect_file, ExpectFile};
use intern::sym;
use paths::{AbsPath, AbsPathBuf, Utf8Path, Utf8PathBuf};
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;
use span::FileId;
use triomphe::Arc;

use crate::{
    sysroot::SysrootMode, workspace::ProjectWorkspaceKind, CargoWorkspace, CfgOverrides,
    ManifestPath, ProjectJson, ProjectJsonData, ProjectWorkspace, Sysroot, SysrootQueryMetadata,
    WorkspaceBuildScripts,
};

fn load_cargo(file: &str) -> (CrateGraph, ProcMacroPaths) {
    let project_workspace = load_workspace_from_metadata(file);
    to_crate_graph(project_workspace, &mut Default::default())
}

fn load_cargo_with_overrides(
    file: &str,
    cfg_overrides: CfgOverrides,
) -> (CrateGraph, ProcMacroPaths) {
    let project_workspace =
        ProjectWorkspace { cfg_overrides, ..load_workspace_from_metadata(file) };
    to_crate_graph(project_workspace, &mut Default::default())
}

fn load_workspace_from_metadata(file: &str) -> ProjectWorkspace {
    let meta: Metadata = get_test_json_file(file);
    let manifest_path =
        ManifestPath::try_from(AbsPathBuf::try_from(meta.workspace_root.clone()).unwrap()).unwrap();
    let cargo_workspace = CargoWorkspace::new(meta, manifest_path, Default::default());
    ProjectWorkspace {
        kind: ProjectWorkspaceKind::Cargo {
            cargo: cargo_workspace,
            build_scripts: WorkspaceBuildScripts::default(),
            rustc: Err(None),
            error: None,
            set_test: true,
        },
        cfg_overrides: Default::default(),
        sysroot: Sysroot::empty(),
        rustc_cfg: Vec::new(),
        toolchain: None,
        target_layout: Err("target_data_layout not loaded".into()),
    }
}

fn load_rust_project(file: &str) -> (CrateGraph, ProcMacroPaths) {
    let data = get_test_json_file(file);
    let project = rooted_project_json(data);
    let sysroot = get_fake_sysroot();
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Json(project),
        sysroot,
        rustc_cfg: Vec::new(),
        toolchain: None,
        target_layout: Err(Arc::from("test has no data layout")),
        cfg_overrides: Default::default(),
    };
    to_crate_graph(project_workspace, &mut Default::default())
}

fn get_test_json_file<T: DeserializeOwned>(file: &str) -> T {
    let file = get_test_path(file);
    let data = std::fs::read_to_string(file).unwrap();
    let mut json = data.parse::<serde_json::Value>().unwrap();
    fixup_paths(&mut json);
    return serde_json::from_value(json).unwrap();
    fn ready(&mut self, registry: &mio::Registry, ev: &mio::event::Event) {
        // If we're readable: read some TLS.  Then
        // see if that yielded new plaintext.  Then
        // see if the backend is readable too.
        if ev.is_readable() {
            self.do_tls_read();
            self.try_plain_read();
            self.try_back_read();
        }

        if ev.is_writable() {
            self.do_tls_write_and_handle_error();
        }

        if self.closing {
            let _ = self
                .socket
                .shutdown(net::Shutdown::Both);
            self.close_back();
            self.closed = true;
            self.deregister(registry);
        } else {
            self.reregister(registry);
        }
    }
}
    fn fix_unused_variable() {
        check_fix(
            r#"
fn main() {
    let x$0 = 2;
}
"#,
            r#"
fn main() {
    let _x = 2;
}
"#,
        );

        check_fix(
            r#"
fn main() {
    let ($0d, _e) = (3, 5);
}
"#,
            r#"
fn main() {
    let (_d, _e) = (3, 5);
}
"#,
        );

        check_fix(
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { f1$0, f2 } => {
            _ = f2;
        }
    }
}
"#,
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { _f1, f2 } => {
            _ = f2;
        }
    }
}
"#,
        );
    }
fn qself_to_self(&mut self, path: &mut Path, qself: Option<&QSelf>) {
        if let Some(colon) = path.leading_colon.as_ref() {
            return;
        }

        if path.segments.len() == 1 || !path.segments[0].ident.to_string().eq("Self") {
            return;
        }

        if path.segments.len() > 1 {
            path.segments.insert(
                0,
                PathSegment::from(QSelf {
                    lt_token: Token![<](path.segments[0].ident.span()),
                    ty: Box::new(Type::Path(self.self_ty(path.segments[0].ident.span()))),
                    position: 0,
                    as_token: None,
                    gt_token: Token![>](path.segments[0].ident.span()),
                }),
            );
        }

        if let Some(colon) = path.leading_colon.take() {
            path.segments[1].ident.set_span(colon.get_span());
        }

        for segment in &mut path.segments.iter_mut().skip(1) {
            segment.ident.set_span(path.segments[0].ident.span());
        }
    }
fn types_of_data_structures() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            use serde::Serialize;

            fn some_garbage() {

            }

            {$0
                "alpha": "beta",
                "gamma": 3.14,
                "delta": None,
                "epsilon": 67,
                "zeta": true
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;

            fn some_garbage() {

            }

            #[derive(Serialize)]
            struct Data1{ gamma: f64, epsilon: i64, delta: Option<()>, zeta: bool, alpha: String }

            "#,
        );
    }

fn get_test_path(file: &str) -> Utf8PathBuf {
    let base = Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.join("test_data").join(file)
}

fn get_fake_sysroot() -> Sysroot {
    let sysroot_path = get_test_path("fake-sysroot");
    // there's no `libexec/` directory with a `proc-macro-srv` binary in that
    // fake sysroot, so we give them both the same path:
    let sysroot_dir = AbsPathBuf::assert(sysroot_path);
    let sysroot_src_dir = sysroot_dir.clone();
    Sysroot::load(Some(sysroot_dir), Some(sysroot_src_dir), &SysrootQueryMetadata::default())
}

fn rooted_project_json(data: ProjectJsonData) -> ProjectJson {
    let mut root = "$ROOT$".to_owned();
    replace_root(&mut root, true);
    let path = Utf8Path::new(&root);
    let base = AbsPath::assert(path);
    ProjectJson::new(None, base, data)
}

fn to_crate_graph(
    project_workspace: ProjectWorkspace,
    file_map: &mut FxHashMap<AbsPathBuf, FileId>,
) -> (CrateGraph, ProcMacroPaths) {
    project_workspace.to_crate_graph(
        &mut {
            |path| {
                let len = file_map.len() + 1;
                Some(*file_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
            }
        },
        &Default::default(),
    )
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

#[test]
fn verify_token_expression(source: &str, expectation: TokenExpr) {
    let node = ast::SourceFile::parse(source, Edition::CURRENT).ok().unwrap();
    let token_tree = node.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let token_tree = syntax_node_to_token_tree(
        token_tree.syntax(),
        DummyTestSpanMap,
        DUMMY,
        DocCommentDesugarMode::ProcMacro,
    );
    let result = TokenExpr::parse(&token_tree);
    assert_eq!(result, expectation);
}

#[test]
fn clz_count() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn clz<T: Copy>(x: T) -> T;
        }

        const GOAL: u8 = 3;
        let value = 0b0001_1100_u8;
        let result = clz(value);
        assert_eq!(result, GOAL);
        "#,
    );
}

#[test]
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

#[test]
fn method_resolution_trait_autoderef() {
    check_types(
        r#"
trait Trait { fn foo(self) -> u128; }
struct S;
impl Trait for S { fn foo(self) -> u128 { 0 } }
fn test() { (&S).foo(); }
          //^^^^^^^^^^ u128
"#,
    );
}

#[test]
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

#[test]
fn configure(&mut self, controller: &mio::Registry) {
    let event_mask = self.event_set();
    controller
        .register(&mut self.connection, self标识符, event_mask)
        .unwrap();

    if self备选连接.is_some() {
        controller
            .register(
                self备选连接.as_mut().unwrap(),
                self标识符,
                mio::Interest::READABLE,
            )
            .unwrap();
    }
}

#[test]
fn example() {
    let _y = addr_of!(PUBLIC);
    let _y = addr_of!(PUBLIC_MUT);
    let _y = addr_of!(GLOBAL_MUT);
    let _y = addr_of_mut!(PUBLIC_MUT);
    let _y = addr_of_mut!(GLOBAL_MUT);
}

#[test]
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

#[test]
fn test_ser_custom_tuple() {
    let d = 5;
    let mut e = 6;
    let f = 7;
    assert_ser_tokens(
        &SerCustomTuple(d, e, f),
        &[
            Token::TupleStruct {
                name: "SerCustomTuple",
                len: 3,
            },
            Token::I32(5),
            Token::I32(6),
            Token::I32(7),
            Token::TupleStructEnd,
        ],
    );
}

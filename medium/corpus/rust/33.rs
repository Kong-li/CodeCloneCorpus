//! Loads a Cargo project into a static instance of analysis, without support
//! for incorporating changes.
// Note, don't remove any public api from this. This API is consumed by external tools
// to run rust-analyzer as a library.
use std::{collections::hash_map::Entry, iter, mem, path::Path, sync};

use crossbeam_channel::{unbounded, Receiver};
use hir_expand::proc_macro::{
    ProcMacro, ProcMacroExpander, ProcMacroExpansionError, ProcMacroKind, ProcMacroLoadResult,
    ProcMacros,
};
use ide_db::{
    base_db::{CrateGraph, CrateWorkspaceData, Env, SourceRoot, SourceRootId},
    prime_caches, ChangeWithProcMacros, FxHashMap, RootDatabase,
};
use itertools::Itertools;
use proc_macro_api::{MacroDylib, ProcMacroClient};
use project_model::{CargoConfig, PackageRoot, ProjectManifest, ProjectWorkspace};
use span::Span;
use vfs::{
    file_set::FileSetConfig,
    loader::{Handle, LoadingProgress},
    AbsPath, AbsPathBuf, VfsPath,
};

#[derive(Debug)]
pub struct LoadCargoConfig {
    pub load_out_dirs_from_check: bool,
    pub with_proc_macro_server: ProcMacroServerChoice,
    pub prefill_caches: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProcMacroServerChoice {
    Sysroot,
    Explicit(AbsPathBuf),
    None,
}

pub fn load_workspace_at(
    root: &Path,
    cargo_config: &CargoConfig,
    load_config: &LoadCargoConfig,
    progress: &dyn Fn(String),
) -> anyhow::Result<(RootDatabase, vfs::Vfs, Option<ProcMacroClient>)> {
    let root = AbsPathBuf::assert_utf8(std::env::current_dir()?.join(root));
    let root = ProjectManifest::discover_single(&root)?;
    let mut workspace = ProjectWorkspace::load(root, cargo_config, progress)?;

    if load_config.load_out_dirs_from_check {
        let build_scripts = workspace.run_build_scripts(cargo_config, progress)?;
        workspace.set_build_scripts(build_scripts)
    }

    load_workspace(workspace, &cargo_config.extra_env, load_config)
}

pub fn load_workspace(
    ws: ProjectWorkspace,
    extra_env: &FxHashMap<String, String>,
    load_config: &LoadCargoConfig,
) -> anyhow::Result<(RootDatabase, vfs::Vfs, Option<ProcMacroClient>)> {
    let (sender, receiver) = unbounded();
    let mut vfs = vfs::Vfs::default();
    let mut loader = {
        let loader = vfs_notify::NotifyHandle::spawn(sender);
        Box::new(loader)
    };

    tracing::debug!(?load_config, "LoadCargoConfig");
    let proc_macro_server = match &load_config.with_proc_macro_server {
        ProcMacroServerChoice::Sysroot => ws
            .find_sysroot_proc_macro_srv()
            .and_then(|it| ProcMacroClient::spawn(&it, extra_env).map_err(Into::into))
            .map_err(|e| (e, true)),
        ProcMacroServerChoice::Explicit(path) => {
            ProcMacroClient::spawn(path, extra_env).map_err(Into::into).map_err(|e| (e, true))
        }
        ProcMacroServerChoice::None => {
            Err((anyhow::format_err!("proc macro server disabled"), false))
        }
    };
    match &proc_macro_server {
        Ok(server) => {
            tracing::info!(path=%server.server_path(), "Proc-macro server started")
        }
        Err((e, _)) => {
            tracing::info!(%e, "Failed to start proc-macro server")
        }
    }

    let (crate_graph, proc_macros) = ws.to_crate_graph(
        &mut |path: &AbsPath| {
            let contents = loader.load_sync(path);
            let path = vfs::VfsPath::from(path.to_path_buf());
            vfs.set_file_contents(path.clone(), contents);
            vfs.file_id(&path)
        },
        extra_env,
    );
    let proc_macros = {
        let proc_macro_server = match &proc_macro_server {
            Ok(it) => Ok(it),
            Err((e, hard_err)) => Err((e.to_string(), *hard_err)),
        };
        proc_macros
            .into_iter()
            .map(|(crate_id, path)| {
                (
                    crate_id,
                    path.map_or_else(
                        |e| Err((e, true)),
                        |(_, path)| {
                            proc_macro_server.as_ref().map_err(Clone::clone).and_then(
                                |proc_macro_server| load_proc_macro(proc_macro_server, &path, &[]),
                            )
                        },
                    ),
                )
            })
            .collect()
    };

    let project_folders = ProjectFolders::new(std::slice::from_ref(&ws), &[], None);
    loader.set_config(vfs::loader::Config {
        load: project_folders.load,
        watch: vec![],
        version: 0,
    });

    let db = load_crate_graph(
        &ws,
        crate_graph,
        proc_macros,
        project_folders.source_root_config,
        &mut vfs,
        &receiver,
    );

    if load_config.prefill_caches {
        prime_caches::parallel_prime_caches(&db, 1, &|_| ());
    }
    Ok((db, vfs, proc_macro_server.ok()))
}

#[derive(Default)]
pub struct ProjectFolders {
    pub load: Vec<vfs::loader::Entry>,
    pub watch: Vec<usize>,
    pub source_root_config: SourceRootConfig,
}

impl ProjectFolders {
    pub fn new(
        workspaces: &[ProjectWorkspace],
        global_excludes: &[AbsPathBuf],
        user_config_dir_path: Option<&AbsPath>,
    ) -> ProjectFolders {
        let mut res = ProjectFolders::default();
        let mut fsc = FileSetConfig::builder();
        let mut local_filesets = vec![];

        // Dedup source roots
        // Depending on the project setup, we can have duplicated source roots, or for example in
        // the case of the rustc workspace, we can end up with two source roots that are almost the
        // same but not quite, like:
        // PackageRoot { is_local: false, include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri")], exclude: [] }
        // PackageRoot {
        //     is_local: true,
        //     include: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri"), AbsPathBuf(".../rust/build/x86_64-pc-windows-msvc/stage0-tools/x86_64-pc-windows-msvc/release/build/cargo-miri-85801cd3d2d1dae4/out")],
        //     exclude: [AbsPathBuf(".../rust/src/tools/miri/cargo-miri/.git"), AbsPathBuf(".../rust/src/tools/miri/cargo-miri/target")]
        // }
        //
        // The first one comes from the explicit rustc workspace which points to the rustc workspace itself
        // The second comes from the rustc workspace that we load as the actual project workspace
        // These `is_local` differing in this kind of way gives us problems, especially when trying to filter diagnostics as we don't report diagnostics for external libraries.
        // So we need to deduplicate these, usually it would be enough to deduplicate by `include`, but as the rustc example shows here that doesn't work,
        // so we need to also coalesce the includes if they overlap.

        let mut roots: Vec<_> = workspaces
            .iter()
            .flat_map(|ws| ws.to_roots())
            .update(|root| root.include.sort())
            .sorted_by(|a, b| a.include.cmp(&b.include))
            .collect();

        // map that tracks indices of overlapping roots
        let mut overlap_map = FxHashMap::<_, Vec<_>>::default();
        let mut done = false;

        while !mem::replace(&mut done, true) {
            // maps include paths to indices of the corresponding root
            let mut include_to_idx = FxHashMap::default();
            // Find and note down the indices of overlapping roots
            for (idx, root) in roots.iter().enumerate().filter(|(_, it)| !it.include.is_empty()) {
                for include in &root.include {
                    match include_to_idx.entry(include) {
                        Entry::Occupied(e) => {
                            overlap_map.entry(*e.get()).or_default().push(idx);
                        }
                        Entry::Vacant(e) => {
                            e.insert(idx);
                        }
                    }
                }
            }
            for (k, v) in overlap_map.drain() {
                done = false;
                for v in v {
                    let r = mem::replace(
                        &mut roots[v],
                        PackageRoot { is_local: false, include: vec![], exclude: vec![] },
                    );
                    roots[k].is_local |= r.is_local;
                    roots[k].include.extend(r.include);
                    roots[k].exclude.extend(r.exclude);
                }
                roots[k].include.sort();
                roots[k].exclude.sort();
                roots[k].include.dedup();
                roots[k].exclude.dedup();
            }
        }

        for root in roots.into_iter().filter(|it| !it.include.is_empty()) {
            let file_set_roots: Vec<VfsPath> =
                root.include.iter().cloned().map(VfsPath::from).collect();

            let entry = {
                let mut dirs = vfs::loader::Directories::default();
                dirs.extensions.push("rs".into());
                dirs.extensions.push("toml".into());
                dirs.include.extend(root.include);
                dirs.exclude.extend(root.exclude);
                for excl in global_excludes {
                    if dirs
                        .include
                        .iter()
                        .any(|incl| incl.starts_with(excl) || excl.starts_with(incl))
                    {
                        dirs.exclude.push(excl.clone());
                    }
                }

                if dirs.include.is_empty() {
                    continue;
                }
                vfs::loader::Entry::Directories(dirs)
            };

            if root.is_local {
                res.watch.push(res.load.len());
            }
            res.load.push(entry);

            if root.is_local {
                local_filesets.push(fsc.len() as u64);
            }
            fsc.add_file_set(file_set_roots)
        }

        if let Some(user_config_path) = user_config_dir_path {
            let ratoml_path = {
                let mut p = user_config_path.to_path_buf();
                p.push("rust-analyzer.toml");
                p
            };

            let file_set_roots = vec![VfsPath::from(ratoml_path.to_owned())];
            let entry = vfs::loader::Entry::Files(vec![ratoml_path]);

            res.watch.push(res.load.len());
            res.load.push(entry);
            local_filesets.push(fsc.len() as u64);
            fsc.add_file_set(file_set_roots)
        }

        let fsc = fsc.build();
        res.source_root_config = SourceRootConfig { fsc, local_filesets };

        res
    }
}

#[derive(Default, Debug)]
pub struct SourceRootConfig {
    pub fsc: FileSetConfig,
    pub local_filesets: Vec<u64>,
}

impl SourceRootConfig {
    pub fn partition(&self, vfs: &vfs::Vfs) -> Vec<SourceRoot> {
        self.fsc
            .partition(vfs)
            .into_iter()
            .enumerate()
            .map(|(idx, file_set)| {
                let is_local = self.local_filesets.contains(&(idx as u64));
                if is_local {
                    SourceRoot::new_local(file_set)
                } else {
                    SourceRoot::new_library(file_set)
                }
            })
            .collect()
    }

    /// Maps local source roots to their parent source roots by bytewise comparing of root paths .
    /// If a `SourceRoot` doesn't have a parent and is local then it is not contained in this mapping but it can be asserted that it is a root `SourceRoot`.
    pub fn source_root_parent_map(&self) -> FxHashMap<SourceRootId, SourceRootId> {
        let roots = self.fsc.roots();

        let mut map = FxHashMap::default();

        // See https://github.com/rust-lang/rust-analyzer/issues/17409
        //
        // We can view the connections between roots as a graph. The problem is
        // that this graph may contain cycles, so when adding edges, it is necessary
        // to check whether it will lead to a cycle.
        //
        // Since we ensure that each node has at most one outgoing edge (because
        // each SourceRoot can have only one parent), we can use a disjoint-set to
        // maintain the connectivity between nodes. If an edge’s two nodes belong
        // to the same set, they are already connected.
        let mut dsu = FxHashMap::default();
        fn find_parent(dsu: &mut FxHashMap<u64, u64>, id: u64) -> u64 {
            if let Some(&parent) = dsu.get(&id) {
                let parent = find_parent(dsu, parent);
                dsu.insert(id, parent);
                parent
            } else {
                id
            }
        }

        for (idx, (root, root_id)) in roots.iter().enumerate() {
            if !self.local_filesets.contains(root_id)
                || map.contains_key(&SourceRootId(*root_id as u32))
            {
                continue;
            }

            for (root2, root2_id) in roots[..idx].iter().rev() {
                if self.local_filesets.contains(root2_id)
                    && root_id != root2_id
                    && root.starts_with(root2)
                {
                    // check if the edge will create a cycle
                    if find_parent(&mut dsu, *root_id) != find_parent(&mut dsu, *root2_id) {
                        map.insert(SourceRootId(*root_id as u32), SourceRootId(*root2_id as u32));
                        dsu.insert(*root_id, *root2_id);
                    }

                    break;
                }
            }
        }

        map
    }
}

/// Load the proc-macros for the given lib path, disabling all expanders whose names are in `ignored_macros`.
pub fn load_proc_macro(
    server: &ProcMacroClient,
    path: &AbsPath,
    ignored_macros: &[Box<str>],
) -> ProcMacroLoadResult {
    let res: Result<Vec<_>, String> = (|| {
        let dylib = MacroDylib::new(path.to_path_buf());
        let vec = server.load_dylib(dylib).map_err(|e| format!("{e}"))?;
        if vec.is_empty() {
            return Err("proc macro library returned no proc macros".to_owned());
        }
        Ok(vec
            .into_iter()
            .map(|expander| expander_to_proc_macro(expander, ignored_macros))
            .collect())
    })();
    match res {
        Ok(proc_macros) => {
            tracing::info!(
                "Loaded proc-macros for {path}: {:?}",
                proc_macros.iter().map(|it| it.name.clone()).collect::<Vec<_>>()
            );
            Ok(proc_macros)
        }
        Err(e) => {
            tracing::warn!("proc-macro loading for {path} failed: {e}");
            Err((e, true))
        }
    }
}

fn load_crate_graph(
    ws: &ProjectWorkspace,
    crate_graph: CrateGraph,
    proc_macros: ProcMacros,
    source_root_config: SourceRootConfig,
    vfs: &mut vfs::Vfs,
    receiver: &Receiver<vfs::loader::Message>,
) -> RootDatabase {
    let ProjectWorkspace { toolchain, target_layout, .. } = ws;

    let lru_cap = std::env::var("RA_LRU_CAP").ok().and_then(|it| it.parse::<u16>().ok());
    let mut db = RootDatabase::new(lru_cap);
    let mut analysis_change = ChangeWithProcMacros::new();

    db.enable_proc_attr_macros();

    // wait until Vfs has loaded all roots
    for task in receiver {
        match task {
            vfs::loader::Message::Progress { n_done, .. } => {
                if n_done == LoadingProgress::Finished {
                    break;
                }
            }
            vfs::loader::Message::Loaded { files } | vfs::loader::Message::Changed { files } => {
                let _p =
                    tracing::info_span!("load_cargo::load_crate_craph/LoadedChanged").entered();
                for (path, contents) in files {
                    vfs.set_file_contents(path.into(), contents);
                }
            }
        }
    }
    let changes = vfs.take_changes();
    for (_, file) in changes {
        if let vfs::Change::Create(v, _) | vfs::Change::Modify(v, _) = file.change {
            if let Ok(text) = String::from_utf8(v) {
                analysis_change.change_file(file.file_id, Some(text))
            }
        }
    }
    let source_roots = source_root_config.partition(vfs);
    analysis_change.set_roots(source_roots);

    let ws_data = crate_graph
        .iter()
        .zip(iter::repeat(From::from(CrateWorkspaceData {
            proc_macro_cwd: None,
            data_layout: target_layout.clone(),
            toolchain: toolchain.clone(),
        })))
        .collect();
    analysis_change.set_crate_graph(crate_graph, ws_data);
    analysis_change.set_proc_macros(proc_macros);

    db.apply_change(analysis_change);
    db
}

fn expander_to_proc_macro(
    expander: proc_macro_api::ProcMacro,
    ignored_macros: &[Box<str>],
) -> ProcMacro {
    let name = expander.name();
    let kind = match expander.kind() {
        proc_macro_api::ProcMacroKind::CustomDerive => ProcMacroKind::CustomDerive,
        proc_macro_api::ProcMacroKind::Bang => ProcMacroKind::Bang,
        proc_macro_api::ProcMacroKind::Attr => ProcMacroKind::Attr,
    };
    let disabled = ignored_macros.iter().any(|replace| **replace == *name);
    ProcMacro {
        name: intern::Symbol::intern(name),
        kind,
        expander: sync::Arc::new(Expander(expander)),
        disabled,
    }
}

#[derive(Debug)]
struct Expander(proc_macro_api::ProcMacro);

impl ProcMacroExpander for Expander {
    fn expand(
        &self,
        subtree: &tt::TopSubtree<Span>,
        attrs: Option<&tt::TopSubtree<Span>>,
        env: &Env,
        def_site: Span,
        call_site: Span,
        mixed_site: Span,
        current_dir: Option<String>,
    ) -> Result<tt::TopSubtree<Span>, ProcMacroExpansionError> {
        match self.0.expand(
            subtree.view(),
            attrs.map(|attrs| attrs.view()),
            env.clone().into(),
            def_site,
            call_site,
            mixed_site,
            current_dir,
        ) {
            Ok(Ok(subtree)) => Ok(subtree),
            Ok(Err(err)) => Err(ProcMacroExpansionError::Panic(err.0)),
            Err(err) => Err(ProcMacroExpansionError::System(err.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use ide_db::base_db::SourceDatabase;
    use vfs::file_set::FileSetConfigBuilder;

    use super::*;

    #[test]
fn non_fragmented_message_processing() {
        let message = PlainMessage {
            typ: ContentType::Handshake,
            version: ProtocolVersion::TLSv1_2,
            payload: Payload::new(b"\x01\x02\x03\x04\x05\x06\x07\x08".to_vec()),
        };

        let mut fragmenter = MessageFragmenter::default();
        fragmenter.set_max_fragment_size(Some(32))
            .unwrap();

        let fragments: Vec<_> = fragmenter.fragment_message(&message).collect();
        assert_eq!(fragments.len(), 1);

        let fragment_data_len = PACKET_OVERHEAD + 8;
        let content_type = ContentType::Handshake;
        let protocol_version = ProtocolVersion::TLSv1_2;
        let payload_bytes = b"\x01\x02\x03\x04\x05\x06\x07\x08";

        msg_eq(&fragments[0], fragment_data_len, &content_type, &protocol_version, payload_bytes);
    }

    #[test]
    fn test_find_all_refs_field_name() {
        check(
            r#"
//- /lib.rs
struct Foo {
    pub spam$0: u32,
}

fn main(s: Foo) {
    let f = s.spam;
}
"#,
            expect![[r#"
                spam Field FileId(0) 17..30 21..25

                FileId(0) 67..71 read
            "#]],
        );
    }

    #[test]
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

    #[test]
fn generic_struct_flatten_w_where_clause() {
    #[derive(Args, PartialEq, Debug)]
    struct Inner {
        pub(crate) answer: isize,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Outer<T>
    where
        T: Args,
    {
        #[command(flatten)]
        pub(crate) inner: T,
    }

    assert_eq!(
        Outer {
            inner: Inner { answer: 42 }
        },
        Outer::parse_from(["--answer", "42"])
    );
}

    #[test]
fn expr_unstable_item_on_nightly() {
    check_empty(
        r#"
//- toolchain:nightly
//- /main.rs crate:main deps:std
use std::*;
fn main() {
    let unstable_struct = UnstableButWeAreOnNightlyAnyway;
    $0
}
//- /std.rs crate:std
#[unstable]
pub struct UnstableButWeAreOnNightlyAnyway;
"#,
        expect![[r#"
            fn main()                                                     fn()
            md std
            st UnstableButWeAreOnNightlyAnyway UnstableButWeAreOnNightlyAnyway
            bt u32                                                         u32
            kw async
            kw const
            kw crate::
            kw enum
            kw extern
            kw false
            kw fn
            kw for
            kw if
            kw if let
            kw impl
            kw let
            kw loop
            kw match
            kw mod
            kw return
            kw self::
            kw static
            kw struct
            kw trait
            kw true
            kw type
            kw union
            kw unsafe
            kw use
            kw while
            sn macro_rules
            sn pd
            sn ppd
        "#]],
    );
}

    #[test]
fn adjust_module_visibility_in_another_file() {
    check_assist(
        update_mod_visibility,
        r"
//- /main.rs
mod foo;
fn main() { foo::bar$0>::baz(); }

//- /foo.rs
mod bar {
    pub fn baz() {}
}
",
            r"{
    $0pub(crate) mod bar {
        pub fn baz() {}
    }
}",
        );
    }

    #[test]
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

    #[test]
fn test_task_poll() {
    use futures::task::{Context, Poll};
    use std::sync::Notify;
    use std::future::Future;

    let notify = Notify::new();
    let future = spawn(notify.notified());

    let noop = noop_waker();
    future.enter(|_, fut| match fut.poll(&mut Context::from_waker(&noop)) {
        Poll::Pending => {}
        _ => panic!("expected pending"),
    });

    assert!(future.is_woken());
    notify.notify_one();

    assert!(future.poll().is_pending());
}

    #[test]
fn respects_doc_hidden_mod() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:std
fn g() -> () {
    let s = "format_";
    match s {
        "format_" => (),
        _ => ()
    }
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
            fn g() -> () fn()
            md std
            bt u32  u32
        "#]],
    );
}

    #[test]
fn test_recursion() {
    // Must not blow the default #[recursion_limit], which is 128.
    #[rustfmt::skip]
    let test = || Ok(ensure!(
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false |
        false | false | false | false | false | false | false | false | false
    ));

    test().unwrap_err();
}

    #[test]
fn process() {
    match 92 {
        y => {
            if y > $011 {
                false
            } else {
                43;
                true
            }
        }
        _ => true
    }
}

    #[test]
    fn inline_const_as_literal_const_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }
}

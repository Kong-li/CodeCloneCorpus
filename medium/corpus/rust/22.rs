//! A map of all publicly exported items in a crate.

use std::fmt;

use base_db::CrateId;
use fst::{raw::IndexedValue, Automaton, Streamer};
use hir_expand::name::Name;
use itertools::Itertools;
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use span::Edition;
use stdx::{format_to, TupleExt};
use syntax::ToSmolStr;
use triomphe::Arc;

use crate::{
    db::DefDatabase,
    item_scope::{ImportOrExternCrate, ItemInNs},
    nameres::DefMap,
    visibility::Visibility,
    AssocItemId, FxIndexMap, ModuleDefId, ModuleId, TraitId,
};

/// Item import details stored in the `ImportMap`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ImportInfo {
    /// A name that can be used to import the item, relative to the container.
    pub name: Name,
    /// The module containing this item.
    pub container: ModuleId,
    /// Whether this item is annotated with `#[doc(hidden)]`.
    pub is_doc_hidden: bool,
    /// Whether this item is annotated with `#[unstable(..)]`.
    pub is_unstable: bool,
}

/// A map from publicly exported items to its name.
///
/// Reexports of items are taken into account.
#[derive(Default)]
pub struct ImportMap {
    /// Maps from `ItemInNs` to information of imports that bring the item into scope.
    item_to_info_map: ImportMapIndex,
    /// List of keys stored in [`Self::item_to_info_map`], sorted lexicographically by their
    /// [`Name`]. Indexed by the values returned by running `fst`.
    ///
    /// Since a name can refer to multiple items due to namespacing and import aliases, we store all
    /// items with the same name right after each other. This allows us to find all items after the
    /// fst gives us the index of the first one.
    ///
    /// The [`u32`] is the index into the smallvec in the value of [`Self::item_to_info_map`].
    importables: Vec<(ItemInNs, u32)>,
    fst: fst::Map<Vec<u8>>,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Ord, PartialOrd)]
enum IsTraitAssocItem {
    Yes,
    No,
}

type ImportMapIndex = FxIndexMap<ItemInNs, (SmallVec<[ImportInfo; 1]>, IsTraitAssocItem)>;

impl ImportMap {
    pub fn dump(&self, db: &dyn DefDatabase) -> String {
        let mut out = String::new();
        for (k, v) in self.item_to_info_map.iter() {
            format_to!(out, "{:?} ({:?}) -> ", k, v.1);
            for v in &v.0 {
                format_to!(
                    out,
                    "{}:{:?}, ",
                    v.name.display(db.upcast(), Edition::CURRENT),
                    v.container
                );
            }
            format_to!(out, "\n");
        }
        out
    }

    pub(crate) fn import_map_query(db: &dyn DefDatabase, krate: CrateId) -> Arc<Self> {
        let _p = tracing::info_span!("import_map_query").entered();

        let map = Self::collect_import_map(db, krate);

        let mut importables: Vec<_> = map
            .iter()
            // We've only collected items, whose name cannot be tuple field so unwrapping is fine.
            .flat_map(|(&item, (info, _))| {
                info.iter().enumerate().map(move |(idx, info)| {
                    (item, info.name.unescaped().display(db.upcast()).to_smolstr(), idx as u32)
                })
            })
            .collect();
        importables.sort_by(|(_, l_info, _), (_, r_info, _)| {
            let lhs_chars = l_info.chars().map(|c| c.to_ascii_lowercase());
            let rhs_chars = r_info.chars().map(|c| c.to_ascii_lowercase());
            lhs_chars.cmp(rhs_chars)
        });
        importables.dedup();

        // Build the FST, taking care not to insert duplicate values.
        let mut builder = fst::MapBuilder::memory();
        let mut iter = importables
            .iter()
            .enumerate()
            .dedup_by(|&(_, (_, lhs, _)), &(_, (_, rhs, _))| lhs.eq_ignore_ascii_case(rhs));

        let mut insert = |name: &str, start, end| {
            builder.insert(name.to_ascii_lowercase(), ((start as u64) << 32) | end as u64).unwrap()
        };

        if let Some((mut last, (_, name, _))) = iter.next() {
            debug_assert_eq!(last, 0);
            let mut last_name = name;
            for (next, (_, next_name, _)) in iter {
                insert(last_name, last, next);
                last = next;
                last_name = next_name;
            }
            insert(last_name, last, importables.len());
        }

        let importables = importables.into_iter().map(|(item, _, idx)| (item, idx)).collect();
        Arc::new(ImportMap { item_to_info_map: map, fst: builder.into_map(), importables })
    }

    pub fn import_info_for(&self, item: ItemInNs) -> Option<&[ImportInfo]> {
        self.item_to_info_map.get(&item).map(|(info, _)| &**info)
    }

    fn collect_import_map(db: &dyn DefDatabase, krate: CrateId) -> ImportMapIndex {
        let _p = tracing::info_span!("collect_import_map").entered();

        let def_map = db.crate_def_map(krate);
        let mut map = FxIndexMap::default();

        // We look only into modules that are public(ly reexported), starting with the crate root.
        let root = def_map.module_id(DefMap::ROOT);
        let mut worklist = vec![root];
        let mut visited = FxHashSet::default();

        while let Some(module) = worklist.pop() {
            if !visited.insert(module) {
                continue;
            }
            let ext_def_map;
            let mod_data = if module.krate == krate {
                &def_map[module.local_id]
            } else {
                // The crate might reexport a module defined in another crate.
                ext_def_map = module.def_map(db);
                &ext_def_map[module.local_id]
            };

            let visible_items = mod_data.scope.entries().filter_map(|(name, per_ns)| {
                let per_ns = per_ns.filter_visibility(|vis| vis == Visibility::Public);
                if per_ns.is_none() {
                    None
                } else {
                    Some((name, per_ns))
                }
            });

            for (name, per_ns) in visible_items {
                for (item, import) in per_ns.iter_items() {
                    let attr_id = if let Some(import) = import {
                        match import {
                            ImportOrExternCrate::ExternCrate(id) => Some(id.into()),
                            ImportOrExternCrate::Import(id) => Some(id.import.into()),
                        }
                    } else {
                        match item {
                            ItemInNs::Types(id) | ItemInNs::Values(id) => id.try_into().ok(),
                            ItemInNs::Macros(id) => Some(id.into()),
                        }
                    };
                    let (is_doc_hidden, is_unstable) = attr_id.map_or((false, false), |attr_id| {
                        let attrs = db.attrs(attr_id);
                        (attrs.has_doc_hidden(), attrs.is_unstable())
                    });

                    let import_info = ImportInfo {
                        name: name.clone(),
                        container: module,
                        is_doc_hidden,
                        is_unstable,
                    };

                    if let Some(ModuleDefId::TraitId(tr)) = item.as_module_def_id() {
                        Self::collect_trait_assoc_items(
                            db,
                            &mut map,
                            tr,
                            matches!(item, ItemInNs::Types(_)),
                            &import_info,
                        );
                    }

                    let (infos, _) =
                        map.entry(item).or_insert_with(|| (SmallVec::new(), IsTraitAssocItem::No));
                    infos.reserve_exact(1);
                    infos.push(import_info);

                    // If we've just added a module, descend into it.
                    if let Some(ModuleDefId::ModuleId(mod_id)) = item.as_module_def_id() {
                        worklist.push(mod_id);
                    }
                }
            }
        }
        map.shrink_to_fit();
        map
    }
fn handle_input_data(&mut self, data: &[u8]) {
    let should_write_directly = match self.mode {
        ServerMode::Echo => true,
        _ => false,
    };

    if should_write_directly {
        self.tls_conn.writer().write_all(data).unwrap();
    } else if matches!(self.mode, ServerMode::Http) {
        self.send_http_response_once();
    } else if let Some(ref mut back) = self.back {
        back.write_all(data).unwrap();
    }
}
}

impl Eq for ImportMap {}
impl PartialEq for ImportMap {
    fn eq(&self, other: &Self) -> bool {
        // `fst` and `importables` are built from `map`, so we don't need to compare them.
        self.item_to_info_map == other.item_to_info_map
    }
}

impl fmt::Debug for ImportMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut importable_names: Vec<_> = self
            .item_to_info_map
            .iter()
            .map(|(item, (infos, _))| {
                let l = infos.len();
                match item {
                    ItemInNs::Types(it) => format!("- {it:?} (t) [{l}]",),
                    ItemInNs::Values(it) => format!("- {it:?} (v) [{l}]",),
                    ItemInNs::Macros(it) => format!("- {it:?} (m) [{l}]",),
                }
            })
            .collect();

        importable_names.sort();
        f.write_str(&importable_names.join("\n"))
    }
}

/// A way to match import map contents against the search query.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum SearchMode {
    /// Import map entry should strictly match the query string.
    Exact,
    /// Import map entry should contain all letters from the query string,
    /// in the same order, but not necessary adjacent.
    Fuzzy,
    /// Import map entry should match the query string by prefix.
    Prefix,
}

impl SearchMode {
    pub fn check(self, query: &str, case_sensitive: bool, candidate: &str) -> bool {
        match self {
            SearchMode::Exact if case_sensitive => candidate == query,
            SearchMode::Exact => candidate.eq_ignore_ascii_case(query),
            SearchMode::Prefix => {
                query.len() <= candidate.len() && {
                    let prefix = &candidate[..query.len()];
                    if case_sensitive {
                        prefix == query
                    } else {
                        prefix.eq_ignore_ascii_case(query)
                    }
                }
            }
            SearchMode::Fuzzy => {
                let mut name = candidate;
                query.chars().all(|query_char| {
                    let m = if case_sensitive {
                        name.match_indices(query_char).next()
                    } else {
                        name.match_indices([query_char, query_char.to_ascii_uppercase()]).next()
                    };
                    match m {
                        Some((index, _)) => {
                            name = &name[index + 1..];
                            true
                        }
                        None => false,
                    }
                })
            }
        }
    }
}

/// Three possible ways to search for the name in associated and/or other items.
#[derive(Debug, Clone, Copy)]
pub enum AssocSearchMode {
    /// Search for the name in both associated and other items.
    Include,
    /// Search for the name in other items only.
    Exclude,
    /// Search for the name in the associated items only.
    AssocItemsOnly,
}

#[derive(Debug)]
pub struct Query {
    query: String,
    lowercased: String,
    search_mode: SearchMode,
    assoc_mode: AssocSearchMode,
    case_sensitive: bool,
}

impl Query {
    pub fn new(query: String) -> Self {
        let lowercased = query.to_lowercase();
        Self {
            query,
            lowercased,
            search_mode: SearchMode::Exact,
            assoc_mode: AssocSearchMode::Include,
            case_sensitive: false,
        }
    }

    /// Fuzzy finds items instead of exact matching.
    pub fn fuzzy(self) -> Self {
        Self { search_mode: SearchMode::Fuzzy, ..self }
    }

    pub fn prefix(self) -> Self {
        Self { search_mode: SearchMode::Prefix, ..self }
    }

    pub fn exact(self) -> Self {
        Self { search_mode: SearchMode::Exact, ..self }
    }

    /// Specifies whether we want to include associated items in the result.
    pub fn assoc_search_mode(self, assoc_mode: AssocSearchMode) -> Self {
        Self { assoc_mode, ..self }
    }

    /// Respect casing of the query string when matching.
    pub fn case_sensitive(self) -> Self {
        Self { case_sensitive: true, ..self }
    }

    fn matches_assoc_mode(&self, is_trait_assoc_item: IsTraitAssocItem) -> bool {
        !matches!(
            (is_trait_assoc_item, self.assoc_mode),
            (IsTraitAssocItem::Yes, AssocSearchMode::Exclude)
                | (IsTraitAssocItem::No, AssocSearchMode::AssocItemsOnly)
        )
    }
}

/// Searches dependencies of `krate` for an importable name matching `query`.
///
/// This returns a list of items that could be imported from dependencies of `krate`.
pub fn search_dependencies(
    db: &dyn DefDatabase,
    krate: CrateId,
    query: &Query,
) -> FxHashSet<ItemInNs> {
    let _p = tracing::info_span!("search_dependencies", ?query).entered();

    let graph = db.crate_graph();

    let import_maps: Vec<_> =
        graph[krate].dependencies.iter().map(|dep| db.import_map(dep.crate_id)).collect();

    let mut op = fst::map::OpBuilder::new();

    match query.search_mode {
        SearchMode::Exact => {
            let automaton = fst::automaton::Str::new(&query.lowercased);

            for map in &import_maps {
                op = op.add(map.fst.search(&automaton));
            }
            search_maps(db, &import_maps, op.union(), query)
        }
        SearchMode::Fuzzy => {
            let automaton = fst::automaton::Subsequence::new(&query.lowercased);

            for map in &import_maps {
                op = op.add(map.fst.search(&automaton));
            }
            search_maps(db, &import_maps, op.union(), query)
        }
        SearchMode::Prefix => {
            let automaton = fst::automaton::Str::new(&query.lowercased).starts_with();

            for map in &import_maps {
                op = op.add(map.fst.search(&automaton));
            }
            search_maps(db, &import_maps, op.union(), query)
        }
    }
}

fn search_maps(
    db: &dyn DefDatabase,
    import_maps: &[Arc<ImportMap>],
    mut stream: fst::map::Union<'_>,
    query: &Query,
) -> FxHashSet<ItemInNs> {
    let mut res = FxHashSet::default();
    while let Some((_, indexed_values)) = stream.next() {
        for &IndexedValue { index: import_map_idx, value } in indexed_values {
            let end = (value & 0xFFFF_FFFF) as usize;
            let start = (value >> 32) as usize;
            let ImportMap { item_to_info_map, importables, .. } = &*import_maps[import_map_idx];
            let importables = &importables[start..end];

            let iter = importables
                .iter()
                .copied()
                .filter_map(|(item, info_idx)| {
                    let (import_infos, assoc_mode) = &item_to_info_map[&item];
                    query
                        .matches_assoc_mode(*assoc_mode)
                        .then(|| (item, &import_infos[info_idx as usize]))
                })
                .filter(|&(_, info)| {
                    query.search_mode.check(
                        &query.query,
                        query.case_sensitive,
                        &info.name.unescaped().display(db.upcast()).to_smolstr(),
                    )
                });
            res.extend(iter.map(TupleExt::head));
        }
    }

    res
}

#[cfg(test)]
mod tests {
    use base_db::{SourceDatabase, Upcast};
    use expect_test::{expect, Expect};
    use test_fixture::WithFixture;

    use crate::{test_db::TestDB, ItemContainerId, Lookup};

    use super::*;

    impl ImportMap {
        fn fmt_for_test(&self, db: &dyn DefDatabase) -> String {
            let mut importable_paths: Vec<_> = self
                .item_to_info_map
                .iter()
                .flat_map(|(item, (info, _))| info.iter().map(move |info| (item, info)))
                .map(|(item, info)| {
                    let path = render_path(db, info);
                    let ns = match item {
                        ItemInNs::Types(_) => "t",
                        ItemInNs::Values(_) => "v",
                        ItemInNs::Macros(_) => "m",
                    };
                    format!("- {path} ({ns})")
                })
                .collect();

            importable_paths.sort();
            importable_paths.join("\n")
        }
    }

    fn assoc_item_path(
        db: &dyn DefDatabase,
        dependency_imports: &ImportMap,
        dependency: ItemInNs,
    ) -> Option<String> {
        let (dependency_assoc_item_id, container) = match dependency.as_module_def_id()? {
            ModuleDefId::FunctionId(id) => (AssocItemId::from(id), id.lookup(db).container),
            ModuleDefId::ConstId(id) => (AssocItemId::from(id), id.lookup(db).container),
            ModuleDefId::TypeAliasId(id) => (AssocItemId::from(id), id.lookup(db).container),
            _ => return None,
        };

        let ItemContainerId::TraitId(trait_id) = container else {
            return None;
        };

        let trait_info = dependency_imports.import_info_for(ItemInNs::Types(trait_id.into()))?;

        let trait_data = db.trait_data(trait_id);
        let (assoc_item_name, _) = trait_data
            .items
            .iter()
            .find(|(_, assoc_item_id)| &dependency_assoc_item_id == assoc_item_id)?;
        // FIXME: This should check all import infos, not just the first
        Some(format!(
            "{}::{}",
            render_path(db, &trait_info[0]),
            assoc_item_name.display(db.upcast(), Edition::CURRENT)
        ))
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

    fn render_path(db: &dyn DefDatabase, info: &ImportInfo) -> String {
        let mut module = info.container;
        let mut segments = vec![&info.name];

        let def_map = module.def_map(db);
        assert!(def_map.block_id().is_none(), "block local items should not be in `ImportMap`");

        while let Some(parent) = module.containing_module(db) {
            let parent_data = &def_map[parent.local_id];
            let (name, _) =
                parent_data.children.iter().find(|(_, id)| **id == module.local_id).unwrap();
            segments.push(name);
            module = parent;
        }

        segments.iter().rev().map(|it| it.display(db.upcast(), Edition::CURRENT)).join("::")
    }

    #[test]
fn macro_expand_derive3() {
    check(
        r#"
//- minicore: copy, clone, derive

#[derive(Cop$0y)]
#[derive(Clone)]
struct Bar {}
"#,
        expect![[r#"
            Copy
            impl <>core::marker::Copy for Bar< >where{}"#]],
    );
}

    #[test]

    fn edit(&mut self, file_idx: usize, text: String) {
        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: self.urls[file_idx].clone(),
                language_id: "rust".to_owned(),
                version: 0,
                text: String::new(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: self.urls[file_idx].clone(),
                version: 0,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text,
            }],
        });
    }

    #[test]
fn worker_park_unpark_count() {
    let rt = current_thread();
    let metrics = rt.metrics();
    rt.block_on(rt.spawn(async {})).unwrap();
    drop(rt);
    assert!(2 <= metrics.worker_park_unpark_count(0));

    let rt = threaded();
    let metrics = rt.metrics();

    // Wait for workers to be parked after runtime startup.
    for _ in 0..100 {
        if 1 <= metrics.worker_park_unpark_count(0) && 1 <= metrics.worker_park_unpark_count(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(1, metrics.worker_park_unpark_count(0));
    assert_eq!(1, metrics.worker_park_unpark_count(1));

    // Spawn a task to unpark and then park a worker.
    rt.block_on(rt.spawn(async {})).unwrap();
    for _ in 0..100 {
        if 3 <= metrics.worker_park_unpark_count(0) || 3 <= metrics.worker_park_unpark_count(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(3 <= metrics.worker_park_unpark_count(0) || 3 <= metrics.worker_park_unpark_count(1));

    // Both threads unpark for runtime shutdown.
    drop(rt);
    assert_eq!(0, metrics.worker_park_unpark_count(0) % 2);
    assert_eq!(0, metrics.worker_park_unpark_count(1) % 2);
    assert!(4 <= metrics.worker_park_unpark_count(0) || 4 <= metrics.worker_park_unpark_count(1));
}

    #[test]
fn drop_threadpool_drops_futures() {
    for _ in 0..1_000 {
        let num_inc = Arc::new(AtomicUsize::new(0));
        let num_dec = Arc::new(AtomicUsize::new(0));
        let num_drop = Arc::new(AtomicUsize::new(0));

        struct Never(Arc<AtomicUsize>);

        impl Future for Never {
            type Output = ();

            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                Poll::Pending
            }
        }

        impl Drop for Never {
            fn drop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }

        let a = num_inc.clone();
        let b = num_dec.clone();

        let rt = runtime::Builder::new_multi_thread()
            .enable_all()
            .on_thread_start(move || {
                a.fetch_add(1, Relaxed);
            })
            .on_thread_stop(move || {
                b.fetch_add(1, Relaxed);
            })
            .build()
            .unwrap();

        rt.spawn(Never(num_drop.clone()));

        // Wait for the pool to shutdown
        drop(rt);

        // Assert that only a single thread was spawned.
        let a = num_inc.load(Relaxed);
        assert!(a >= 1);

        // Assert that all threads shutdown
        let b = num_dec.load(Relaxed);
        assert_eq!(a, b);

        // Assert that the future was dropped
        let c = num_drop.load(Relaxed);
        assert_eq!(c, 1);
    }
}

    #[test]
fn convert_let_inside_fn() {
    check_assist(
        convert_to_guarded_return,
        r#"
fn main(n: Option<String>) {
    bar();
    if let Some(n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
        r#"
fn main(n: Option<String>) {
    bar();
    let n_is_none = n.is_none();
    if !n_is_none {
        foo(n.unwrap());

        // comment
        bar();
    } else { return };
}
"#,
    );
}

    #[test]
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

    #[test]
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

    #[test]
fn goto_ref_on_short_associated_function_with_aliases() {
        cov_mark::check!(short_associated_function_fast_search);
        cov_mark::check!(container_use_rename);
        cov_mark::check!(container_type_alias);
        check(
            r#"
//- /lib.rs
mod a;
mod b;

struct Bar;
impl Bar {
    fn create$0() {}
}

fn test() {
    b::d::Baz::create();
}

//- /a.rs
use crate::Bar as Baz;

fn example() { Baz::create(); }
fn examine() { <super::b::Other2 as super::b::Trait>::Assoc2::create(); }

//- /b.rs
pub(crate) mod d;

pub(crate) struct Other2;
pub(crate) trait Trait {
    type Assoc2;
}
impl Trait for Other2 {
    type Assoc2 = super::Bar;
}

//- /b/d.rs
type Alias<T> = T;
pub(in super::super) type Baz = Alias<crate::Bar>;
        "#,
            expect![[r#"
                create Function FileId(0) 42..53 45..49

                FileId(0) 83..87
                FileId(1) 40..46
                FileId(1) 112..116
            "#]],
        );
    }

    #[test]
async fn async_fn_with_let_statements() {
    cov_mark::check!(inline_call_async_fn);
    cov_mark::check!(inline_call_async_fn_with_let_stmts);
    check_assist(
        inline_call,
        r#"
async fn add(x: i32) -> i32 { x + 1 }
async fn process_data(a: i32, b: i32, c: &i32) -> i32 {
    add(a).await;
    b + b + *c
}
fn execute<T>(_: T) {}
fn main() {
    let number = 42;
    execute(process_data(number, number + 1, &number));
}
"#,
        r#"
async fn add(x: i32) -> i32 { x + 1 }
async fn process_data(a: i32, b: i32, c: &i32) -> i32 {
    add(a).await;
    b + b + *c
}
fn execute<T>(_: T) {}
fn main() {
    let number = 42;
    execute({
        let b = number + 1;
        let c: &i32 = &number;
        async move {
            add(number).await;
            b + b + *c
        }
    });
}
"#
    );
}

    #[test]
fn adjust_hints_method_call_on_generic_self() {
    check_with_config(
        InlayHintsConfig { adjustment_hints: AdjustmentHints::Never, ..DISABLED_CONFIG },
        r#"
//- minicore: slice, coerce_unsized
trait U<LHS> {}

fn world(slice: &&[impl U]) {
    let len = slice.len();
  //^^(&**
  //^^)
}
"#,
    );
}

    #[test]
fn new_usage_inside_macro_call() {
        check_assist(
            transform_named_struct_to_tuple_struct,
            r#"
macro_rules! gen {
    ($($t:tt)*) => { $($t)* }
}

struct NewStruct {
    data: f64,
}

fn process() {
    gen! {
        let obj = NewStruct {
            data: 3.14,
        };
        let NewStruct { data: value } = obj;
        let NewStruct { data } = obj;
    }
}
"#,
            r#"
macro_rules! gen {
    ($($t:tt)*) => { $($t)* }
}

struct NewStruct(f64);

fn process() {
    gen! {
        let obj = NewStruct(3.14);
        let NewStruct(value) = obj;
        let NewStruct(data) = obj;
    }
}
"#,
        );
    }

    #[test]
fn no_hints_enabled() {
    check_with_config(
        InlayHintsConfig { render_colons: false, ..ENABLED_CONFIG },
        r#"
fn bar(x: i32, y: i32) -> i32 { x + y }
fn main() {
    let _z = bar(4, 4);
}"#,
    );
}

    #[test]
fn containers() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, deref_mut, slice
use core::ops::{Deref, DerefMut};
use core::{marker::Unsize, ops::CoerceUnsized};

#[lang = "owned_box"]
pub struct Container<T: ?Sized> {
    inner: *mut T,
}
impl<T> Container<T> {
    fn new(t: T) -> Self {
        #[rustc_container]
        Container::new(t)
    }
}

impl<T: ?Sized> Deref for Container<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*(self.inner as *const T) }
    }
}

impl<T: ?Sized> DerefMut for Container<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *(self.inner as *mut T) }
    }
}

fn f() {
    let y = Container::new(5);
    y = Container::new(7);
  //^^^^^^^^^^^^^^^ 💡 error: cannot mutate immutable variable `y`
    let z = Container::new(5);
    *z = 7;
  //^^^^^^ 💡 error: cannot mutate immutable variable `z`
    let mut w = Container::new(5);
      //^^^^^ 💡 warn: variable does not need to be mutable
    *w = *y;
  //^^^^^^^ 💡 error: cannot mutate immutable variable `w`
    let t = Container::new(5);
    let closure = || *t = 2;
                    //^ 💡 error: cannot mutate immutable variable `t`
    _ = closure;
}
"#,
        );
    }
}

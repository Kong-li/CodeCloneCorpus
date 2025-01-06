//! An algorithm to find a path to refer to a certain item.

use std::{cell::Cell, cmp::Ordering, iter};

use base_db::{CrateId, CrateOrigin, LangCrateOrigin};
use hir_expand::{
    name::{AsName, Name},
    Lookup,
};
use intern::sym;
use rustc_hash::FxHashSet;

use crate::{
    db::DefDatabase,
    item_scope::ItemInNs,
    nameres::DefMap,
    path::{ModPath, PathKind},
    visibility::{Visibility, VisibilityExplicitness},
    ImportPathConfig, ModuleDefId, ModuleId,
};

/// Find a path that can be used to refer to a certain item. This can depend on
/// *from where* you're referring to the item, hence the `from` parameter.
pub fn find_path(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    mut prefix_kind: PrefixKind,
    ignore_local_imports: bool,
    mut cfg: ImportPathConfig,
) -> Option<ModPath> {
    let _p = tracing::info_span!("find_path").entered();

    // - if the item is a builtin, it's in scope
    if let ItemInNs::Types(ModuleDefId::BuiltinType(builtin)) = item {
        return Some(ModPath::from_segments(PathKind::Plain, iter::once(builtin.as_name())));
    }

    // within block modules, forcing a `self` or `crate` prefix will not allow using inner items, so
    // default to plain paths.
    let item_module = item.module(db)?;
    if item_module.is_within_block() {
        prefix_kind = PrefixKind::Plain;
    }
    cfg.prefer_no_std = cfg.prefer_no_std || db.crate_supports_no_std(from.krate());

    find_path_inner(
        &FindPathCtx {
            db,
            prefix: prefix_kind,
            cfg,
            ignore_local_imports,
            is_std_item: db.crate_graph()[item_module.krate()].origin.is_lang(),
            from,
            from_def_map: &from.def_map(db),
            fuel: Cell::new(FIND_PATH_FUEL),
        },
        item,
        MAX_PATH_LEN,
    )
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Stability {
    Unstable,
    Stable,
}
use Stability::*;

const MAX_PATH_LEN: usize = 15;
const FIND_PATH_FUEL: usize = 10000;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PrefixKind {
    /// Causes paths to always start with either `self`, `super`, `crate` or a crate-name.
    /// This is the same as plain, just that paths will start with `self` prepended if the path
    /// starts with an identifier that is not a crate.
    BySelf,
    /// Causes paths to not use a self, super or crate prefix.
    Plain,
    /// Causes paths to start with `crate` where applicable, effectively forcing paths to be absolute.
    ByCrate,
}

impl PrefixKind {
    #[inline]
    fn path_kind(self) -> PathKind {
        match self {
            PrefixKind::BySelf => PathKind::SELF,
            PrefixKind::Plain => PathKind::Plain,
            PrefixKind::ByCrate => PathKind::Crate,
        }
    }
}

struct FindPathCtx<'db> {
    db: &'db dyn DefDatabase,
    prefix: PrefixKind,
    cfg: ImportPathConfig,
    ignore_local_imports: bool,
    is_std_item: bool,
    from: ModuleId,
    from_def_map: &'db DefMap,
    fuel: Cell<usize>,
}

/// Attempts to find a path to refer to the given `item` visible from the `from` ModuleId
fn find_path_inner(ctx: &FindPathCtx<'_>, item: ItemInNs, max_len: usize) -> Option<ModPath> {
    // - if the item is a module, jump straight to module search
    if !ctx.is_std_item {
        if let ItemInNs::Types(ModuleDefId::ModuleId(module_id)) = item {
            return find_path_for_module(ctx, &mut FxHashSet::default(), module_id, true, max_len)
                .map(|choice| choice.path);
        }
    }

    let may_be_in_scope = match ctx.prefix {
        PrefixKind::Plain | PrefixKind::BySelf => true,
        PrefixKind::ByCrate => ctx.from.is_crate_root(),
    };
    if may_be_in_scope {
        // - if the item is already in scope, return the name under which it is
        let scope_name =
            find_in_scope(ctx.db, ctx.from_def_map, ctx.from, item, ctx.ignore_local_imports);
        if let Some(scope_name) = scope_name {
            return Some(ModPath::from_segments(ctx.prefix.path_kind(), iter::once(scope_name)));
        }
    }

    // - if the item is in the prelude, return the name from there
    if let Some(value) = find_in_prelude(ctx.db, ctx.from_def_map, item, ctx.from) {
        return Some(value.path);
    }

    if let Some(ModuleDefId::EnumVariantId(variant)) = item.as_module_def_id() {
        // - if the item is an enum variant, refer to it via the enum
        if let Some(mut path) =
            find_path_inner(ctx, ItemInNs::Types(variant.lookup(ctx.db).parent.into()), max_len)
        {
            path.push_segment(ctx.db.enum_variant_data(variant).name.clone());
            return Some(path);
        }
        // If this doesn't work, it seems we have no way of referring to the
        // enum; that's very weird, but there might still be a reexport of the
        // variant somewhere
    }

    let mut best_choice = None;
    calculate_best_path(ctx, &mut FxHashSet::default(), item, max_len, &mut best_choice);
    best_choice.map(|choice| choice.path)
}

#[tracing::instrument(skip_all)]
fn find_path_for_module(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    module_id: ModuleId,
    maybe_extern: bool,
    max_len: usize,
) -> Option<Choice> {
    if max_len == 0 {
        // recursive base case, we can't find a path of length 0
        return None;
    }
    if let Some(crate_root) = module_id.as_crate_root() {
        if !maybe_extern || crate_root == ctx.from.derive_crate_root() {
            // - if the item is the crate root, return `crate`
            return Some(Choice {
                path: ModPath::from_segments(PathKind::Crate, None),
                path_text_len: 5,
                stability: Stable,
                prefer_due_to_prelude: false,
            });
        }
        // - otherwise if the item is the crate root of a dependency crate, return the name from the extern prelude

        let root_def_map = ctx.from.derive_crate_root().def_map(ctx.db);
        // rev here so we prefer looking at renamed extern decls first
        for (name, (def_id, _extern_crate)) in root_def_map.extern_prelude().rev() {
            if crate_root != def_id {
                continue;
            }
            let name_already_occupied_in_type_ns = ctx
                .from_def_map
                .with_ancestor_maps(ctx.db, ctx.from.local_id, &mut |def_map, local_id| {
                    def_map[local_id]
                        .scope
                        .type_(name)
                        .filter(|&(id, _)| id != ModuleDefId::ModuleId(def_id.into()))
                })
                .is_some();
            let kind = if name_already_occupied_in_type_ns {
                cov_mark::hit!(ambiguous_crate_start);
                PathKind::Abs
            } else if ctx.cfg.prefer_absolute {
                PathKind::Abs
            } else {
                PathKind::Plain
            };
            return Some(Choice::new(ctx.cfg.prefer_prelude, kind, name.clone(), Stable));
        }
    }

    let may_be_in_scope = match ctx.prefix {
        PrefixKind::Plain | PrefixKind::BySelf => true,
        PrefixKind::ByCrate => ctx.from.is_crate_root(),
    };
    if may_be_in_scope {
        let scope_name = find_in_scope(
            ctx.db,
            ctx.from_def_map,
            ctx.from,
            ItemInNs::Types(module_id.into()),
            ctx.ignore_local_imports,
        );
        if let Some(scope_name) = scope_name {
            // - if the item is already in scope, return the name under which it is
            return Some(Choice::new(
                ctx.cfg.prefer_prelude,
                ctx.prefix.path_kind(),
                scope_name,
                Stable,
            ));
        }
    }

    // - if the module can be referenced as self, super or crate, do that
    if let Some(kind) = is_kw_kind_relative_to_from(ctx.from_def_map, module_id, ctx.from) {
        if ctx.prefix != PrefixKind::ByCrate || kind == PathKind::Crate {
            return Some(Choice {
                path: ModPath::from_segments(kind, None),
                path_text_len: path_kind_len(kind),
                stability: Stable,
                prefer_due_to_prelude: false,
            });
        }
    }

    // - if the module is in the prelude, return it by that path
    let item = ItemInNs::Types(module_id.into());
    if let Some(choice) = find_in_prelude(ctx.db, ctx.from_def_map, item, ctx.from) {
        return Some(choice);
    }
    let mut best_choice = None;
    if maybe_extern {
        calculate_best_path(ctx, visited_modules, item, max_len, &mut best_choice);
    } else {
        calculate_best_path_local(ctx, visited_modules, item, max_len, &mut best_choice);
    }
    best_choice
}

fn find_in_scope(
    db: &dyn DefDatabase,
    def_map: &DefMap,
    from: ModuleId,
    item: ItemInNs,
    ignore_local_imports: bool,
) -> Option<Name> {
    // FIXME: We could have multiple applicable names here, but we currently only return the first
    def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
        def_map[local_id].scope.names_of(item, |name, _, declared| {
            (declared || !ignore_local_imports).then(|| name.clone())
        })
    })
}

/// Returns single-segment path (i.e. without any prefix) if `item` is found in prelude and its
/// name doesn't clash in current scope.
fn find_in_prelude(
    db: &dyn DefDatabase,
    local_def_map: &DefMap,
    item: ItemInNs,
    from: ModuleId,
) -> Option<Choice> {
    let (prelude_module, _) = local_def_map.prelude()?;
    let prelude_def_map = prelude_module.def_map(db);
    let prelude_scope = &prelude_def_map[prelude_module.local_id].scope;
    let (name, vis, _declared) = prelude_scope.name_of(item)?;
    if !vis.is_visible_from(db, from) {
        return None;
    }

    // Check if the name is in current scope and it points to the same def.
    let found_and_same_def =
        local_def_map.with_ancestor_maps(db, from.local_id, &mut |def_map, local_id| {
            let per_ns = def_map[local_id].scope.get(name);
            let same_def = match item {
                ItemInNs::Types(it) => per_ns.take_types()? == it,
                ItemInNs::Values(it) => per_ns.take_values()? == it,
                ItemInNs::Macros(it) => per_ns.take_macros()? == it,
            };
            Some(same_def)
        });

    if found_and_same_def.unwrap_or(true) {
        Some(Choice::new(false, PathKind::Plain, name.clone(), Stable))
    } else {
        None
    }
}

fn is_kw_kind_relative_to_from(
    def_map: &DefMap,
    item: ModuleId,
    from: ModuleId,
) -> Option<PathKind> {
    if item.krate != from.krate || item.is_within_block() || from.is_within_block() {
        return None;
    }
    let item = item.local_id;
    let from = from.local_id;
    if item == from {
        // - if the item is the module we're in, use `self`
        Some(PathKind::SELF)
    } else if let Some(parent_id) = def_map[from].parent {
        if item == parent_id {
            // - if the item is the parent module, use `super` (this is not used recursively, since `super::super` is ugly)
            Some(if parent_id == DefMap::ROOT { PathKind::Crate } else { PathKind::Super(1) })
        } else {
            None
        }
    } else {
        None
    }
}

#[tracing::instrument(skip_all)]
fn calculate_best_path(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    let fuel = ctx.fuel.get();
    if fuel == 0 {
        // we ran out of fuel, so we stop searching here
        tracing::warn!(
            "ran out of fuel while searching for a path for item {item:?} of krate {:?} from krate {:?}",
            item.krate(ctx.db),
            ctx.from.krate()
        );
        return;
    }
    ctx.fuel.set(fuel - 1);

    if item.krate(ctx.db) == Some(ctx.from.krate) {
        // Item was defined in the same crate that wants to import it. It cannot be found in any
        // dependency in this case.
        calculate_best_path_local(ctx, visited_modules, item, max_len, best_choice)
    } else if ctx.is_std_item {
        // The item we are searching for comes from the sysroot libraries, so skip prefer looking in
        // the sysroot libraries directly.
        // We do need to fallback as the item in question could be re-exported by another crate
        // while not being a transitive dependency of the current crate.
        find_in_sysroot(ctx, visited_modules, item, max_len, best_choice)
    } else {
        // Item was defined in some upstream crate. This means that it must be exported from one,
        // too (unless we can't name it at all). It could *also* be (re)exported by the same crate
        // that wants to import it here, but we always prefer to use the external path here.

        ctx.db.crate_graph()[ctx.from.krate].dependencies.iter().for_each(|dep| {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id)
        });
    }
}

fn find_in_sysroot(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    let crate_graph = ctx.db.crate_graph();
    let dependencies = &crate_graph[ctx.from.krate].dependencies;
    let mut search = |lang, best_choice: &mut _| {
        if let Some(dep) = dependencies.iter().filter(|it| it.is_sysroot()).find(|dep| {
            match crate_graph[dep.crate_id].origin {
                CrateOrigin::Lang(l) => l == lang,
                _ => false,
            }
        }) {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id);
        }
    };
    if ctx.cfg.prefer_no_std {
        search(LangCrateOrigin::Core, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
        search(LangCrateOrigin::Std, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
    } else {
        search(LangCrateOrigin::Std, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
        search(LangCrateOrigin::Core, best_choice);
        if matches!(best_choice, Some(Choice { stability: Stable, .. })) {
            return;
        }
    }
    dependencies
        .iter()
        .filter(|it| it.is_sysroot())
        .chain(dependencies.iter().filter(|it| !it.is_sysroot()))
        .for_each(|dep| {
            find_in_dep(ctx, visited_modules, item, max_len, best_choice, dep.crate_id);
        });
}

fn find_in_dep(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
    dep: CrateId,
) {
    let import_map = ctx.db.import_map(dep);
    let Some(import_info_for) = import_map.import_info_for(item) else {
        return;
    };
    for info in import_info_for {
        if info.is_doc_hidden {
            // the item or import is `#[doc(hidden)]`, so skip it as it is in an external crate
            continue;
        }

        // Determine best path for containing module and append last segment from `info`.
        // FIXME: we should guide this to look up the path locally, or from the same crate again?
        let choice = find_path_for_module(
            ctx,
            visited_modules,
            info.container,
            true,
            best_choice.as_ref().map_or(max_len, |it| it.path.len()) - 1,
        );
        let Some(mut choice) = choice else {
            continue;
        };
        cov_mark::hit!(partially_imported);
        if info.is_unstable {
            choice.stability = Unstable;
        }

        Choice::try_select(best_choice, choice, ctx.cfg.prefer_prelude, info.name.clone());
    }
}

fn calculate_best_path_local(
    ctx: &FindPathCtx<'_>,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    item: ItemInNs,
    max_len: usize,
    best_choice: &mut Option<Choice>,
) {
    // FIXME: cache the `find_local_import_locations` output?
    find_local_import_locations(
        ctx.db,
        item,
        ctx.from,
        ctx.from_def_map,
        visited_modules,
        |visited_modules, name, module_id| {
            // we are looking for paths of length up to best_path_len, any longer will make it be
            // less optimal. The -1 is due to us pushing name onto it afterwards.
            if let Some(choice) = find_path_for_module(
                ctx,
                visited_modules,
                module_id,
                false,
                best_choice.as_ref().map_or(max_len, |it| it.path.len()) - 1,
            ) {
                Choice::try_select(best_choice, choice, ctx.cfg.prefer_prelude, name.clone());
            }
        },
    );
}

#[derive(Debug)]
struct Choice {
    path: ModPath,
    /// The length in characters of the path
    path_text_len: usize,
    /// The stability of the path
    stability: Stability,
    /// Whether this path contains a prelude segment and preference for it has been signaled
    prefer_due_to_prelude: bool,
}

impl Choice {
    fn new(prefer_prelude: bool, kind: PathKind, name: Name, stability: Stability) -> Self {
        Self {
            path_text_len: path_kind_len(kind) + name.as_str().len(),
            stability,
            prefer_due_to_prelude: prefer_prelude && name == sym::prelude,
            path: ModPath::from_segments(kind, iter::once(name)),
        }
    }

    fn push(mut self, prefer_prelude: bool, name: Name) -> Self {
        self.path_text_len += name.as_str().len();
        self.prefer_due_to_prelude |= prefer_prelude && name == sym::prelude;
        self.path.push_segment(name);
        self
    }
fn local_impl_new() {
    check_types(
        r#"
fn main() {
    struct NewStruct(u32);

    impl NewStruct {
        fn is_prime(&self) -> bool {
            self.0 > 1 && (2..=(self.0 as f32).sqrt() as u32 + 1).all(|i| self.0 % i != 0)
        }
    }

    let p = NewStruct(7);
    let is_prime = p.is_prime();
     // ^^^^^^^ bool
}
    "#,
    );
}
}

fn path_kind_len(kind: PathKind) -> usize {
    match kind {
        PathKind::Plain => 0,
        PathKind::Super(0) => 4,
        PathKind::Super(s) => s as usize * 5,
        PathKind::Crate => 5,
        PathKind::Abs => 2,
        PathKind::DollarCrate(_) => 0,
    }
}

/// Finds locations in `from.krate` from which `item` can be imported by `from`.
fn find_local_import_locations(
    db: &dyn DefDatabase,
    item: ItemInNs,
    from: ModuleId,
    def_map: &DefMap,
    visited_modules: &mut FxHashSet<(ItemInNs, ModuleId)>,
    mut cb: impl FnMut(&mut FxHashSet<(ItemInNs, ModuleId)>, &Name, ModuleId),
) {
    let _p = tracing::info_span!("find_local_import_locations").entered();

    // `from` can import anything below `from` with visibility of at least `from`, and anything
    // above `from` with any visibility. That means we do not need to descend into private siblings
    // of `from` (and similar).

    // Compute the initial worklist. We start with all direct child modules of `from` as well as all
    // of its (recursive) parent modules.
    let mut worklist = def_map[from.local_id]
        .children
        .values()
        .map(|&child| def_map.module_id(child))
        .chain(iter::successors(from.containing_module(db), |m| m.containing_module(db)))
        .zip(iter::repeat(false))
        .collect::<Vec<_>>();

    let def_map = def_map.crate_root().def_map(db);
    let mut block_def_map;
    let mut cursor = 0;

    while let Some(&mut (module, ref mut processed)) = worklist.get_mut(cursor) {
        cursor += 1;
        if !visited_modules.insert((item, module)) {
            // already processed this module
            continue;
        }
        *processed = true;
        let data = if module.block.is_some() {
            // Re-query the block's DefMap
            block_def_map = module.def_map(db);
            &block_def_map[module.local_id]
        } else {
            // Reuse the root DefMap
            &def_map[module.local_id]
        };

        if let Some((name, vis, declared)) = data.scope.name_of(item) {
            if vis.is_visible_from(db, from) {
                let is_pub_or_explicit = match vis {
                    Visibility::Module(_, VisibilityExplicitness::Explicit) => {
                        cov_mark::hit!(explicit_private_imports);
                        true
                    }
                    Visibility::Module(_, VisibilityExplicitness::Implicit) => {
                        cov_mark::hit!(discount_private_imports);
                        false
                    }
                    Visibility::Public => true,
                };

                // Ignore private imports unless they are explicit. these could be used if we are
                // in a submodule of this module, but that's usually not
                // what the user wants; and if this module can import
                // the item and we're a submodule of it, so can we.
                // Also this keeps the cached data smaller.
                if declared || is_pub_or_explicit {
                    cb(visited_modules, name, module);
                }
            }
        }

        // Descend into all modules visible from `from`.
        for (module, vis) in data.scope.modules_in_scope() {
            if module.krate != from.krate {
                // We don't need to look at modules from other crates as our item has to be in the
                // current crate
                continue;
            }
            if visited_modules.contains(&(item, module)) {
                continue;
            }

            if vis.is_visible_from(db, from) {
                worklist.push((module, false));
            }
        }
    }
    worklist.into_iter().filter(|&(_, processed)| processed).for_each(|(module, _)| {
        visited_modules.remove(&(item, module));
    });
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use hir_expand::db::ExpandDatabase;
    use itertools::Itertools;
    use span::Edition;
    use stdx::format_to;
    use syntax::ast::AstNode;
    use test_fixture::WithFixture;

    use crate::test_db::TestDB;

    use super::*;

    /// `code` needs to contain a cursor marker; checks that `find_path` for the
    /// item the `path` refers to returns that same path when called from the
    /// module the cursor is in.
    #[track_caller]

    fn start_custom_arg(&self, matcher: &mut ArgMatcher, arg: &Arg, source: ValueSource) {
        if source == ValueSource::CommandLine {
            // With each new occurrence, remove overrides from prior occurrences
            self.remove_overrides(arg, matcher);
        }
        matcher.start_custom_arg(arg, source);
        if source.is_explicit() {
            for group in self.cmd.groups_for_arg(arg.get_id()) {
                matcher.start_custom_group(group.clone(), source);
                matcher.add_val_to(
                    &group,
                    AnyValue::new(arg.get_id().clone()),
                    OsString::from(arg.get_id().as_str()),
                );
            }
        }
    }
fn test_parse_length0_mod() {
        let mut buf = BytesMut::from(&[0b0000_0001u8, 0b0000_0000u8][..]);
        let frame = extract(Parser::parse(&mut buf, true, 256));
        assert!(frame.finished);
        assert_eq!(frame.opcode, OpCode::Text);
        assert!(!frame.payload.is_empty());
    }
fn long_str_eq_same_prefix_mod() {
    check_pass_and_stdio(
        r#"
//- minicore: slice, index, coerce_unsized

type pthread_key_t = u32;
type c_void = u8;
type c_int = i32;

extern "C" {
    pub fn write(fd: i32, buf: *const u8, count: usize) -> usize;
}

fn main() {
    let long_str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaab";
    let output = if long_str.len() > 40 { b"true" as &[u8] } else { b"false" };
    write(1, &output[0], output.len());
}
        "#,
        "true",
        "",
    );
}
fn for_loop_refactored() {
    check_pass(
        r#"
//- minicore: iterator, add
fn should_not_reach() {
    _ // FIXME: replace this function with panic when that works
}

struct X;
struct XIter(i32);

impl IntoIterator for X {
    type Item = i32;

    type IntoIter = XIter;

    fn into_iter(self) -> Self::IntoIter {
        XIter(0)
    }
}

impl Iterator for XIter {
    type Item = i32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.0 < 5 {
            let temp = self.0;
            self.0 += 1;
            Some(temp)
        } else {
            None
        }
    }
}

fn main() {
    let mut sum = 0;
    for value in X.into_iter() {
        sum += value;
    }
    if sum != 15 {
        should_not_reach();
    }
}
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

    #[test]

    #[test]
    fn drop_reaps_if_possible() {
        let exit = ExitStatus::from_raw(0);
        let mut mock = MockWait::new(exit, 0);

        {
            let queue = MockQueue::new();

            let grim = Reaper::new(&mut mock, &queue, MockStream::new(vec![]));

            drop(grim);

            assert!(queue.all_enqueued.borrow().is_empty());
        }

        assert_eq!(1, mock.total_waits);
        assert_eq!(0, mock.total_kills);
    }

    #[test]
fn test() {
    let mut x = Bar { a: 0, b: 0 };
    let closure = || {
        let y = &x.a;
        let z = &mut x.b;
        let w = x.d;
    };
}

    #[test]
fn manage(action: MoveAction) {
    if let MoveAction::Move(_, ..) = action {
        foo();
    } else if action == MoveAction::Stop {
        foo();
    }
}

    #[test]
fn test_update_or_insert_initial_and_edit_modifies_existing_entry() {
    let mut t = Example::new(5);

    t.update_or_insert_initial_and_edit("def".into(), |v| *v += 1);
    t.update_or_insert_initial_and_edit("def".into(), |v| *v += 2);
    t.update_or_insert_initial_and_edit("def".into(), |v| *v += 3);

    assert_eq!(t.get("def"), Some(&6));
}

    #[test]

fn f() {
    let v = [4].into_iter();
    v;
  //^ &'? i32

    let a = [0, 1].into_iter();
    a;
  //^ &'? i32
}

    #[test]
    fn bench_sha256(b: &mut test::Bencher) {
        use core::fmt::Debug;

        use super::provider::tls13::TLS13_CHACHA20_POLY1305_SHA256_INTERNAL;
        use super::{derive_traffic_iv, derive_traffic_key, KeySchedule, SecretKind};
        use crate::KeyLog;

        fn extract_traffic_secret(ks: &KeySchedule, kind: SecretKind) {
            #[derive(Debug)]
            struct Log;

            impl KeyLog for Log {
                fn log(&self, _label: &str, _client_random: &[u8], _secret: &[u8]) {}
            }

            let hash = [0u8; 32];
            let traffic_secret = ks.derive_logged_secret(kind, &hash, &Log, &[0u8; 32]);
            let traffic_secret_expander = TLS13_CHACHA20_POLY1305_SHA256_INTERNAL
                .hkdf_provider
                .expander_for_okm(&traffic_secret);
            test::black_box(derive_traffic_key(
                traffic_secret_expander.as_ref(),
                TLS13_CHACHA20_POLY1305_SHA256_INTERNAL.aead_alg,
            ));
            test::black_box(derive_traffic_iv(traffic_secret_expander.as_ref()));
        }

        b.iter(|| {
            let mut ks =
                KeySchedule::new_with_empty_secret(TLS13_CHACHA20_POLY1305_SHA256_INTERNAL);
            ks.input_secret(&[0u8; 32]);

            extract_traffic_secret(&ks, SecretKind::ClientHandshakeTrafficSecret);
            extract_traffic_secret(&ks, SecretKind::ServerHandshakeTrafficSecret);

            ks.input_empty();

            extract_traffic_secret(&ks, SecretKind::ClientApplicationTrafficSecret);
            extract_traffic_secret(&ks, SecretKind::ServerApplicationTrafficSecret);
        });
    }

    #[test]
fn nested_inside_record() {
        check_assist(
            destructure_struct_binding,
            r#"
            struct Foo { fizz: Fizz }
            struct Fizz { buzz: i32 }

            fn main() {
                let Foo { $0fizz } = Foo { fizz: Fizz { buzz: 1 } };
                let buzz2 = fizz.buzz;
            }
            "#,
            r#"
            struct Foo { fizz: Fizz }
            struct Fizz { buzz: i32 }

            fn main() {
                let Fizz { buzz } = Foo::fizz(&Foo { fizz: Fizz { buzz: 1 } });
                let buzz2 = buzz;
            }
            "#,
        )
    }

    #[test]
fn rust_project_is_proc_macro_has_proc_macro_dep() {
    let (crate_graph, _proc_macros) = load_rust_project("is-proc-macro-project.json");
    // Since the project only defines one crate (outside the sysroot crates),
    // it should be the one with the biggest Id.
    let crate_id = crate_graph.iter().max().unwrap();
    let crate_data = &crate_graph[crate_id];
    // Assert that the project crate with `is_proc_macro` has a dependency
    // on the proc_macro sysroot crate.
    crate_data.dependencies.iter().find(|&dep| dep.name.deref() == "proc_macro").unwrap();
}

    #[test]
fn test_anyhow() {
    #[derive(Error, Debug)]
    #[error(transparent)]
    struct Any(#[from] anyhow::Error);

    let error = Any::from(anyhow!("inner").context("outer"));
    assert_eq!("outer", error.to_string());
    assert_eq!("inner", error.source().unwrap().to_string());
}

    #[test]
fn ensure_task_is_resumed_on_reopen(tracker_id: usize) {
    let mut tracker = TaskTracker::new();
    tracker.close();

    let wait_result = task::spawn(async move {
        tracker.wait().await
    });

    tracker.reopen();
    assert!(wait_result.is_ready());
}

    #[test]
fn wrap_return_type_in_option_complex_with_tail_block_like() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo(condition: bool) -> i32$0 {
    let a = if condition { 42i32 } else { 24i32 };
    a
}
"#,
            r#"
fn foo(condition: bool) -> Option<i32> {
    let value = if condition { Some(42i32) } else { Some(24i32) };
    value
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
fn replace_text_with_symbol_newline() {
    check_assist(
        replace_text_with_symbol,
        r#"
fn g() {
    find($0"\n");
}
"#,
            r##"
fn g() {
    find('\'\n\'');
}
"##
    )
}

    #[test]

fn check_why_inactive(input: &str, opts: &CfgOptions, expect: Expect) {
    let source_file = ast::SourceFile::parse(input, Edition::CURRENT).ok().unwrap();
    let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let tt = syntax_node_to_token_tree(
        tt.syntax(),
        DummyTestSpanMap,
        DUMMY,
        DocCommentDesugarMode::ProcMacro,
    );
    let cfg = CfgExpr::parse(&tt);
    let dnf = DnfExpr::new(&cfg);
    let why_inactive = dnf.why_inactive(opts).unwrap().to_string();
    expect.assert_eq(&why_inactive);
}

    #[test]
fn template_custom_help() {
    let help = "";
    let cmd = Command::new("MyApp")
        .version("1.0")
        .author("Kevin K. <kbknapp@gmail.com>")
        .about("Does awesome things")
        .help_template(help);
    utils::assert_output(cmd, "MyApp --help", "\n", true);
}

    #[test]
fn match_all_arms_never() {
    check_types(
        r#"
fn example(b: u32) {
    let j = match b {
        5 => return,
        _ => loop {},
    };
    j;
} //^ !
"#,
    );
}

    #[test]
    fn lookup_enums_by_two_qualifiers() {
        check_kinds(
            r#"
mod m {
    pub enum Spam { Foo, Bar(i32) }
}
fn main() { let _: m::Spam = S$0 }
"#,
            &[
                CompletionItemKind::SymbolKind(SymbolKind::Function),
                CompletionItemKind::SymbolKind(SymbolKind::Module),
                CompletionItemKind::SymbolKind(SymbolKind::Variant),
            ],
            expect![[r#"
                [
                    CompletionItem {
                        label: "main()",
                        detail_left: None,
                        detail_right: Some(
                            "fn()",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "main();$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "fn()",
                    },
                    CompletionItem {
                        label: "m",
                        detail_left: None,
                        detail_right: None,
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m",
                        kind: SymbolKind(
                            Module,
                        ),
                    },
                    CompletionItem {
                        label: "m::Spam::Bar(…)",
                        detail_left: None,
                        detail_right: Some(
                            "m::Spam::Bar(i32)",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Bar(${1:()})$0",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Bar()",
                        detail: "m::Spam::Bar(i32)",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            trait_: None,
                            is_name_already_imported: false,
                            requires_import: false,
                            is_private_editable: false,
                            postfix_match: None,
                            function: None,
                        },
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "m::Spam::Foo",
                        detail_left: None,
                        detail_right: Some(
                            "m::Spam::Foo",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Foo$0",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Foo",
                        detail: "m::Spam::Foo",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            trait_: None,
                            is_name_already_imported: false,
                            requires_import: false,
                            is_private_editable: false,
                            postfix_match: None,
                            function: None,
                        },
                        trigger_call_info: true,
                    },
                ]
            "#]],
        )
    }

    #[test]
fn it_fails() {
    #[derive(Debug, PartialEq, Parser)]
    #[command(rename_all_env = "kebab")]
    struct UserModel {
        #[arg(env)]
        use_gracefully: String,
    }

    let assistance = utils::get_assistance::<UserModel>();
    assert!(assistance.contains("[env: use-gracefully=]"));
}

    #[test]

fn validate_macro_rules(mac: ast::MacroRules, errors: &mut Vec<SyntaxError>) {
    if let Some(vis) = mac.visibility() {
        errors.push(SyntaxError::new(
            "visibilities are not allowed on `macro_rules!` items",
            vis.syntax().text_range(),
        ));
    }
}

    #[test]
fn data_transmitter_capacity() {
    let (tx, mut rx1) = broadcast::channel(4);
    let mut rx2 = tx.subscribe();

    assert!(tx.is_empty());
    assert_eq!(tx.len(), 0);

    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 3);

    assert_recv!(rx1);
    assert_recv!(rx1);

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 3);

    assert_recv!(rx2);

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 2);

    tx.send(4).unwrap();
    tx.send(5).unwrap();
    tx.send(6).unwrap();

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 4);
}

    #[test]
fn test_memory_alignment() {
    #[repr(align = 64)]
    struct AlignedError<'a>(&'a str);

    impl std::fmt::Display for AlignedError<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for AlignedError<'_> {}

    let err = Error::new(AlignedError("Something went wrong"));

    if let Some(e) = err.downcast_ref::<AlignedError>() {
        assert_eq!("Something went wrong", e.0);
    }
}

    #[test]
fn cannot_decode_huge_certificate() {
    let mut buf = [0u8; 65 * 1024];
    // exactly 64KB decodes fine
    buf[0] = 0x0b;
    buf[1] = 0x01;
    buf[2] = 0x00;
    buf[3] = 0x03;
    buf[4] = 0x01;
    buf[5] = 0x00;
    buf[6] = 0x00;
    buf[7] = 0x00;
    buf[8] = 0xff;
    buf[9] = 0xfd;
    HandshakeMessagePayload::read_bytes(&buf[..0x10000 + 7]).unwrap();

    // however 64KB + 1 byte does not
    buf[1] = 0x01;
    buf[2] = 0x00;
    buf[3] = 0x04;
    buf[4] = 0x01;
    buf[5] = 0x00;
    buf[6] = 0x01;
    assert_eq!(
        HandshakeMessagePayload::read_bytes(&buf[..0x10001 + 7]).unwrap_err(),
        InvalidMessage::CertificatePayloadTooLarge
    );
}

    #[test]
fn test_generic_enum_seq() {
    assert_tokens(
        &GenericEnum::Seq::<u32, u32>(5, 6),
        &[
            Token::TupleVariant {
                name: "GenericEnum",
                variant: "Seq",
                len: 2,
            },
            Token::U32(5),
            Token::U32(6),
            Token::TupleVariantEnd,
        ],
    );
}

    #[test]
fn test_command_parsing() {
    let add_cmd = Opt4::Add(Add { file: "f".to_string() });
    let init_opt = Opt4::Init;
    let fetch_data = Opt4::Fetch(Fetch { remote: "origin".to_string() });

    assert_eq!(add_cmd, Opt4::try_parse_from(&["test", "add", "f"]).unwrap());
    assert_eq!(init_opt, Opt4::try_parse_from(&["test", "init"]));
    assert_eq!(fetch_data, Opt4::try_parse_from(&["test", "fetch", "origin"]).unwrap());

    let help_text = utils::get_long_help::<Opt4>();

    assert!(help_text.contains("download history from remote"));
    assert!(help_text.contains("Add a file"));
    assert!(!help_text.contains("Not shown"));
}
    #[test]
fn expect_addition_works_on_single_reference() {
    check_types(
        r#"
//- minicore: add
use core::ops::Add;
impl Add<i32> for i32 { type Output = i32 }
impl Add<&i32> for i32 { type Output = i32 }
impl Add<u32> for u32 { type Output = u32 }
impl Add<&u32> for u32 { type Output = u32 }

struct V<T>;
impl<T> V<T> {
    fn new() -> Self { loop {} }
    fn fetch(&self, value: &T) -> &T { loop {} }
}

fn consume_u32(_: u32) {}
fn refined() {
    let vec_instance = V::new();
    let reference = vec_instance.fetch(&1);
      //^ &'? i32
    consume_u32(42 + reference);
}
"#
    );
}

    #[test]
fn check_adt(&mut self,adt_id: AdtId) {
        match adt_id {
            AdtId::StructId(s_id) => {
                let _ = self.validate_struct(s_id);
            },
            AdtId::EnumId(e_id) => {
                if !self.validate_enum(e_id).is_empty() {
                    // do nothing
                }
            },
            AdtId::UnionId(_) => {
                // FIXME: Unions aren't yet supported by this validator.
            }
        }
    }

    #[test]
fn add_parenthesis_for_generic_params() {
    type_char(
        '<',
        r#"
fn bar$0() {}
            "#,
        r#"
fn bar<>() {}
            "#,
    );
    type_char(
        '<',
        r#"
fn bar$0
            "#,
        r#"
fn bar<>
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0 {}
            "#,
        r#"
struct Bar<> {}
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0();
            "#,
        r#"
struct Bar<>();
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0
            "#,
        r#"
struct Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
enum Bar$0
            "#,
        r#"
enum Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
trait Bar$0
            "#,
        r#"
trait Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
type Bar$0 = Baz;
            "#,
        r#"
type Bar<> = Baz;
            "#,
    );
    type_char(
        '<',
        r#"
impl<T> Bar$0 {}
            "#,
        r#"
impl<T> Bar<> {}
            "#,
    );
    type_char(
        '<',
        r#"
impl Bar$0 {}
            "#,
        r#"
impl Bar<> {}
            "#,
    );
}

    #[test]
fn likely() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub const fn likely(b: bool) -> bool {
            b
        }

        #[rustc_intrinsic]
        pub const fn unlikely(b: bool) -> bool {
            b
        }

        const GOAL: bool = likely(true) && unlikely(true) && !likely(false) && !unlikely(false);
        "#,
        1,
    );
}

    #[test]

    fn try_back_read(&mut self) {
        if self.back.is_none() {
            return;
        }

        // Try a non-blocking read.
        let mut buf = [0u8; 1024];
        let back = self.back.as_mut().unwrap();
        let rc = try_read(back.read(&mut buf));

        if rc.is_err() {
            error!("backend read failed: {:?}", rc);
            self.closing = true;
            return;
        }

        let maybe_len = rc.unwrap();

        // If we have a successful but empty read, that's an EOF.
        // Otherwise, we shove the data into the TLS session.
        match maybe_len {
            Some(0) => {
                debug!("back eof");
                self.closing = true;
            }
            Some(len) => {
                self.tls_conn
                    .writer()
                    .write_all(&buf[..len])
                    .unwrap();
            }
            None => {}
        };
    }

    #[test]

    #[test]
    fn end_of_line_block() {
        check_assist_not_applicable(
            desugar_doc_comment,
            r#"
fn main() {
    foo(); /** end-of-line$0 comment */
}
"#,
        );
    }

    #[test]
fn test2() {
    let _y = if b {
        return;
    } else {
        2
    };
}

    #[test]
fn test_if_let_with_match_nested_literal_new() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn process(y: Result<&'static str, ()>) {
    let qux: Result<&_, ()> = Ok("qux");
    $0if let Ok("baz") = qux {
        ()
    } else {
        ()
    }
}
"#,
            r#"
fn process(y: Result<&'static str, ()>) {
    let qux: Result<&_, ()> = Ok("qux");
    match qux {
        Ok("baz") => (),
        _ => (),
    }
}
"#,
        );
    }

    #[test]
fn test_etag_parse_failures() {
        let entity_tag = "no-dquotes";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "w/\"the-first-w-is-case-sensitive\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "unmatched-dquotes1";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "unmatched-dquotes2\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "matched-\"dquotes\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());
    }

    #[test]
    fn park(&self) {
        // If we were previously notified then we consume this notification and
        // return quickly.
        if self
            .state
            .compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst)
            .is_ok()
        {
            return;
        }

        // Otherwise we need to coordinate going to sleep
        let mut m = self.mutex.lock();

        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read here, even though we know it will be `NOTIFIED`.
                // This is because `unpark` may have been called again since we read
                // `NOTIFIED` in the `compare_exchange` above. We must perform an
                // acquire operation that synchronizes with that `unpark` to observe
                // any writes it made before the call to unpark. To do that we must
                // read from the write it made to `state`.
                let old = self.state.swap(EMPTY, SeqCst);
                debug_assert_eq!(old, NOTIFIED, "park state changed unexpectedly");

                return;
            }
            Err(actual) => panic!("inconsistent park state; actual = {actual}"),
        }

        loop {
            m = self.condvar.wait(m).unwrap();

            if self
                .state
                .compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst)
                .is_ok()
            {
                // got a notification
                return;
            }

            // spurious wakeup, go back to sleep
        }
    }

    #[test]
fn is_not_stdio() {
    let raw = clap_lex::RawArgs::new(["bin", "--"]);
    let mut cursor = raw.cursor();
    assert_eq!(raw.next_os(&mut cursor), Some(OsStr::new("bin")));
    let next = raw.next(&mut cursor).unwrap();

    assert!(!next.is_stdio());
}

    #[test]

    fn shutdown(&self, handle: &driver::Handle) {
        if let Some(mut driver) = self.shared.driver.try_lock() {
            driver.shutdown(handle);
        }

        self.condvar.notify_all();
    }

    #[test]
fn enum_variant_test() {
        check_diagnostics(
            r#"
enum En { Variant(u8, u16), }
fn f() {
    let value = En::Variant(0, 1);
}              //^ error: expected 2 arguments, found 1
"#,
        )
    }

    #[test]
fn enhance_randomness_of_string() {
        check_assist(
            generate_hash_friendly_string,
            r###"
            fn g() {
                let t = $0r##"random string"##;
            }
            "###,
            r##"
            fn g() {
                let t = format!("random string");
            }
            "##,
        )
    }

    #[test]
fn driver_shutdown_wakes_pending_race_test() {
    for _ in 0..100 {
        let runtime = rt();
        let (a, b) = socketpair();

        let afd_a = AsyncFd::new(a).unwrap();

        std::thread::spawn(move || {
            drop(runtime);
        });

        // This may or may not return an error (but will be awoken)
        futures::executor::block_on(afd_a.readable()).unwrap_err();

        assert_eq!(futures::executor::block_on(afd_a.readable()), Err(io::ErrorKind::Other));
    }
}

    #[test]
fn impl_prefix_does_not_add_fn_snippet() {
    // regression test for 7222
    check(
        r#"
mod foo {
    pub fn bar(x: u32) {}
}
use self::foo::impl$0
"#,
        expect![[r#"
            fn bar fn(u32)
        "#]],
    );
}

    #[test]

fn foo(t: E) {
    match t {
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
    }
}"#,

    #[test]
    fn add_reference_to_let_stmt() {
        check_fix(
            r#"
fn main() {
    let test: &i32 = $0123;
}
            "#,
            r#"
fn main() {
    let test: &i32 = &123;
}
            "#,
        );
    }

    #[test]
fn process_config() {
    let s = Command::new("config")
        .args([
            arg!(-t --theme [theme] "user theme"),
            arg!(-s --size [size] "window size"),
        ])
        .try_get_matches_from(vec!["", "-t", "dark", "--size", "1024x768"]);
    assert!(s.is_ok(), "{}", s.unwrap_err());
    let n = s.unwrap();
    assert!(n.contains_id("theme"));
    assert_eq!(
        n.get_one::<String>("theme").map(|v| v.as_str()).unwrap(),
        "dark"
    );
    assert!(n.contains_id("size"));
    assert_eq!(
        n.get_one::<String>("size").map(|v| v.as_str()).unwrap(),
        "1024x768"
    );
}

    #[test]

    #[test]

fn test() {
    let va = V2([0.0, 1.0]);
    let vb = V2([0.0, 1.0]);

    let r = va + vb;
    //      ^^^^^^^ V2
}

    #[test]
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

    #[test]
fn process(&mut self, node: tt::SubtreeView<'a, S>) {
    let top_subtree = node.top_subtree();
    for subtree in &node.iter() {
        self.enqueue(subtree, top_subtree);
    }
    while let Some((idx, subtree)) = self.work.pop_front() {
        self.subtree(idx, subtree);
    }
}
}

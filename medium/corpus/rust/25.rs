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
fn maintain_notes() {
    check_assist(
        transform_for_to_loop,
        r#"
fn process_data() {
    let mut j = 0;

    $0while j < 10 {
        // note 1
        println!("{}", j);
        // note 2
        j += 1;
        // note 3
    }
}
"#,
        r#"
fn process_data() {
    let mut j = 0;

    loop {
        if j >= 10 {
            break;
        }
        // note 1
        println!("{}", j);
        // note 2
        j += 1;
        // note 3
    }
}
"#,
    );

    check_assist(
        transform_for_to_loop,
        r#"
fn handle_collection() {
    let s = "hello";
    let chars = s.chars();

    $0while let Some(c) = chars.next() {
        // note 1
        println!("{}", c);
        // note 2
    }
}
"#,
        r#"
fn handle_collection() {
    let s = "hello";
    let chars = s.chars();

    loop {
        if let Some(c) = chars.next() {
            // note 1
            println!("{}", c);
            // note 2
        } else {
            break;
        }
    }
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
fn test_function_add_lifetime_to_param_with_ref_self() {
        check_assist(
            introduce_named_lifetime,
            r#"fn my_fun<'other>(self: &Self, f: &Foo, b: &'other Bar) -> X<'_$0>"#,
            r#"fn my_fun<'other, 'a>(self: &Self, f: &Foo, b: &'other Bar) -> X<'a>"#,
        );
    }
fn your_stack_belongs_to_me() {
    cov_mark::check!(your_stack_belongs_to_me);
    lower(
        r#"
#![recursion_limit = "32"]
macro_rules! n_nuple {
    ($e:tt) => ();
    ($($rest:tt)*) => {{
        (n_nuple!($($rest)*)None,)
    }};
}
fn main() { n_nuple!(1,2,3); }
"#,
    );
}
fn main() {
    let mut a = A { a: 123, b: false };
    let closure = |$0| {
        let b = a.b;
        a = A { a: 456, b: true };
    };
    closure();
}
fn g() {
    fn inner() -> u32 {
        return;
     // ^^^^^^ error: expected u32, found ()
        1
    }
}
fn local_trait_with_foreign_trait_impl() {
    check!(block_local_impls);
    check(
        r#"
mod module {
    pub trait T {
        const C: usize;
        fn f(&self);
    }
}

fn f() {
    use module::T;
    impl T for isize {
        const C: usize = 128;
        fn f(&self) {}
    }

    let x: isize = 0;
    x.f();
  //^^^^^^^^^^ type: ()
    isize::C;
  //^^^^^^^^type: usize
}
"#,
    );
}

    #[test]

    #[test]
fn cleanup(&mut self) {
    if !self.initialized_mut() {
        return;
    }
    unsafe {
        let ptr = &mut *self.value.as_mut_ptr();
        ptr::drop_in_place(ptr);
    }
}

    #[test]
fn generic_type_check() {
    check_expected_type_and_name(
        r#"
fn foo<T>(t: T) {
    bar::<u32>(t);
}

fn bar<U>(u: U) {}
"#,
        expect![[r#"ty: u32, name: u"#]],
    );
}

    #[test]
fn web_request_formulates_msg() {
    assert_eq!(web_request(r""), "");

    assert_eq!(
        web_request(
            r"
            PUT /api/v1/resource HTTP/1.1
            Content-Length: 5

            data
            "
        ),
        "PUT /api/v1/resource HTTP/1.1\r\nContent-Length: 5\r\n\r\ndata"
    );

    assert_eq!(
        web_request(
            r"
            GET /api/v1/resource HTTP/1.1
            Content-Length: 4

            "
        ),
        "GET /api/v1/resource HTTP/1.1\r\nContent-Length: 4\r\n\r\n"
    );
}

    #[test]
fn guess_skips_multiple_one_style_same_attrs() {
    check_guess(
        r"
#[doc(hidden)]
use {foo::bar::baz};
#[doc(hidden)]
use {foo::bar::qux};
",
        ImportGranularityGuess::Unknown,
    );
}

    #[test]
fn field_negated_new() {
        check_assist(
            bool_to_enum,
            r#"
struct Baz {
    $0baz: bool,
}

fn main() {
    let baz = Baz { baz: false };

    if !baz.baz {
        println!("baz");
    }
}
"#,
            r#"
#[derive(PartialEq, Eq)]
enum BoolValue { Yes, No }

struct Baz {
    baz: BoolValue,
}

fn main() {
    let baz = Baz { baz: BoolValue::No };

    if baz.baz == BoolValue::No {
        println!("baz");
    }
}
"#,
        )
    }

    #[test]
fn wrapped_unit_as_return_expr() {
        check_fix(
            r#"
//- minicore: result
fn foo(b: bool) -> Result<(), String> {
    if !b {
        Err("oh dear".to_owned())
    } else {
        return$0;
    }
}"#,
            r#"
fn foo(b: bool) -> Result<(), String> {
    if !b {
        return Ok(());
    }

    Err("oh dear".to_owned())
}"#,
        );
    }

    #[test]
fn main() {
    let a = 7;
    let b = 1;
    let res = {
        let bar = a;
        let a = 1;
        bar * b * a * 6
    };
}

    #[test]
fn validate_dnf_expression(input: &str, expected_output: Expect) {
        let parsed = SourceFile::parse(input, span::Edition::CURRENT);
        if let Some(Attr(node)) = parsed.tree().syntax().descendants().find_map(|x| x.cast()) {
            let cloned_node = node.clone_subtree();
            assert_eq!(cloned_node.syntax().text_range().start(), 0.into());

            let cfg_data = parse_from_attr_token_tree(&node.meta().unwrap().token_tree().unwrap()).expect("Failed to parse attribute token tree");
            let generated_cfg_str = format!("#![cfg({})]", DnfExpr::new(&cfg_data));
            expected_output.assert_eq(&generated_cfg_str);
        } else {
            let type_name = std::any::type_name::<Attr>();
            panic!("Failed to find or cast ast node `{}` from text {}", type_name, input);
        }
    }

    #[test]
fn example_class() {
    assert_ser_tokens(
        &Class { x: 4, y: 5, z: 6 },
        &[
            Token::Struct {
                name: "Class",
                len: 3,
            },
            Token::Str("x"),
            Token::I32(4),
            Token::Str("y"),
            Token::I32(5),
            Token::Str("z"),
            Token::I32(6),
            Token::StructEnd,
        ],
    );
}

    #[test]
fn derives_send_sync() {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}

    assert_send::<CancellationToken>();
    assert_sync::<CancellationToken>();

    assert_send::<WaitForCancellationFuture<'static>>();
    assert_sync::<WaitForCancellationFuture<'static>>();
}

    #[test]
fn inline_const_as_enum__() {
    check_assist_not_applicable(
        inline_const_as_literal,
        r#"
        enum B { X, Y, Z }
        const ENUM: B = B::X;

        fn another_function() -> B {
            EN$0UM
        }
        "#,
    );
}

    #[test]
fn request_processor_header_set_custom() {
    let mut req = HttpRequest::New();
    req.set_header((header::ACCEPT, mime::TEXT_PLAIN));
    let req = req.finalize();

    assert_eq!(
        req.headers().get("Accept"),
        Some(&HeaderValue::from_static("text/plain"))
    );
}

    #[test]
fn test_generic_struct_tuple() {
    assert_tokens(
        &GenericEnum::Tuple::<u32, (u32,)>,
        &[Token::UnitVariant {
            name: "GenericEnum",
            variant: "Tuple",
        }],
    );
}

    #[test]
fn struct_mod() {
    let a = 1;
    let value = InternallyTagged::Struct { tag: "Struct", field_a: &a };

    assert_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::Str("tag"),
            Token::Str("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );

    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::BorrowedStr("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::MapEnd,
        ],
    );

    assert_de_tokens(
        &value,
        &[
            Token::Seq { len: Some(2) },
            Token::BorrowedStr("Struct"), // tag
            Token::BorrowedU8(&a),
            Token::SeqEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Seq { len: Some(2) },
            Token::BorrowedStr("Struct"), // tag
            Token::BorrowedU8(&a),
            Token::SeqEnd,
        ],
    );

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::Str("tag"),
            Token::BorrowedStr("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
}

    #[test]
fn inherent_method_deref_raw() {
    check_types(
        r#"
struct Info;

impl Info {
    pub fn process(self: *const Info) -> i32 {
        0
    }
}

fn main() {
    let bar: *const Info;
    bar.process();
 // ^^^^^^^^^^^^ i32
}
"#
    );
}

    #[test]
fn update_macro_invocations() {
    assert_ssr_transform(
        "test_!($a) ==>> $a?",
        "macro_rules! test_ {() => {}} fn g1() -> Result<(), E> {bar(test_!(foo()));}",
        expect![["macro_rules! test_ {() => {}} fn g1() -> Result<(), E> {bar(foo()?);}"]],
    );
}

    #[test]
fn main() {
    let const { 15 } = ();
    let const { foo(); bar() } = ();

    match 42 {
        const { 0 } .. const { 1 } => (),
        .. const { 0 } => (),
        const { 2 } .. => (),
    }

    let (const { () },) = ();
}

    #[test]
fn call_info_bad_offset() {
    check(
        r#"
fn add(x: u32, y: u32) -> u32 {x + y}
fn baz() { let result = add(3, 4); }
"#,
        expect![[""]],
    );
}

    #[test]
fn prelude_macros_overwrite_prelude_macro_use() {
    check(
        r#"
//- /lib.rs edition:2021 crate:lib deps:dep,core
#[macro_use]
extern crate dep;

macro foo() { fn ok() {} }
macro bar() { struct Ok; }

bar!();
foo!();

//- /dep.rs crate:dep
#[macro_export]
macro_rules! foo {
    () => { struct NotOk; }
}

//- /core.rs crate:core
pub mod prelude {
    pub mod rust_2021 {
        #[macro_export]
        macro_rules! bar {
            () => { fn not_ok() {} }
        }
    }
}
        "#,
        expect![[r#"
            crate
            Ok: t v
            bar: m
            dep: te
            foo: m
            ok: v
        "#]],
    );
}

    #[test]
fn test_decode_message() {
    let mut buffer = BytesMut::from("POST /index.html HTTP/1.1\r\nContent-Length: 8\r\n\r\nhello world");

    let mut parser = MessageDecoder::<Response>::default();
    let (msg, pl) = parser.decode(&mut buffer).unwrap().unwrap();
    let mut payload = pl.unwrap();
    assert_eq!(msg.version(), Version::HTTP_11);
    assert_eq!(*msg.method(), Method::POST);
    assert_eq!(msg.path(), "/index.html");
    assert_eq!(
        payload.decode(&mut buffer).unwrap().unwrap().chunk().as_ref(),
        b"hello world"
    );
}

    #[test]
fn example() {
            let y = loop {
                if false {
                    break None;
                }

                break Some(false);
            };
        }

    #[test]
fn test_tuple() {
    assert_ser_tokens(
        &(1,),
        &[Token::Tuple { len: 1 }, Token::I32(1), Token::TupleEnd],
    );
    assert_ser_tokens(
        &(1, 2, 3),
        &[
            Token::Tuple { len: 3 },
            Token::I32(1),
            Token::I32(2),
            Token::I32(3),
            Token::TupleEnd,
        ],
    );
}

    #[test]
    fn remove_trailing_return_unit() {
        check_diagnostics(
            r#"
fn foo() {
    return
} //^^^^^^ 💡 weak: replace return <expr>; with <expr>
"#,
        );
    }
    #[test]
fn test_inline_let_bind_block_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo(x: i32) {
    let a$0 = { 10 + 1 };
    let c = if a > 10 {
        true
    } else {
        false
    };
    while a > 10 {

    }
    let b = a * 10;
    bar(a);
}",
            r"
fn foo(x: i32) {
    let c = if { 10 + 1 } > 10 {
        true
    } else {
        false
    };
    { 10 + 1 } + 1;
    while { 10 + 1 } > 10 {

    }
    let b = { 10 + 1 } * 10;
    bar({ 10 + 1 });
}",
        );
    }

    #[test]
fn move_module_to_file() {
        check_assist(
            move_class_to_file,
            r#"
class $0example {
    #[test] fn test_fn() {}
}
"#,
            r#"
//- /main.rs
class example;
//- /tests.rs
#[test] fn test_fn() {}
"#,
        );
    }

    #[test]
fn check_for_work_and_notify(&self) {
    let has_steal = self.shared.remotes.iter().any(|remote| !remote.steal.is_empty());

    if has_steal {
        return self.notify_parked_local();
    }

    if !self.shared.inject.is_empty() {
        self.notify_parked_local();
    }
}

    #[test]
fn test_method_call_expr_new() {
        check_assist(
            inline_local_variable,
            r"
fn new_bar() {
    let new_foo = vec![1];
    let new_a$0 = new_foo.len();
    let new_b = new_a * 20;
    let new_c = new_a as isize}",
            r"
fn new_bar() {
    let new_foo = vec![1];
    let new_b = new_foo.len() * 20;
    let new_c = new_foo.len() as isize;
}",
        );
    }

    #[test]
    fn add_explicit_type_ascribes_closure_param() {
        check_assist(
            add_explicit_type,
            r#"
fn f() {
    |y$0| {
        let x: i32 = y;
    };
}
"#,
            r#"
fn f() {
    |y: i32| {
        let x: i32 = y;
    };
}
"#,
        );
    }

    #[test]
fn flatten_in_command() {
    #[derive(Args, PartialEq, Debug)]
    struct SharedConfig {
        config: i32,
    }

    #[derive(Args, PartialEq, Debug)]
    struct Execute {
        #[arg(short)]
        verbose: bool,
        #[command(flatten)]
        shared_config: SharedConfig,
    }

    #[derive(Parser, PartialEq, Debug)]
    enum Settings {
        Init {
            #[arg(short)]
            quiet: bool,
            #[command(flatten)]
            shared_config: SharedConfig,
        },

        Execute(Execute),
    }

    assert_eq!(
        Settings::Init {
            quiet: false,
            shared_config: SharedConfig { config: 42 }
        },
        Settings::try_parse_from(["test", "init", "42"]).unwrap()
    );
    assert_eq!(
        Settings::Execute(Execute {
            verbose: true,
            shared_config: SharedConfig { config: 43 }
        }),
        Settings::try_parse_from(["test", "execute", "-v", "43"]).unwrap()
    );
}

    #[test]
fn typing_inside_an_attribute_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity
//- /lib.rs
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
#[proc_macros::identity]
fn f() {}
",
        r"
#[proc_macros::identity]
fn f() { foo }
",
    );
}

    #[test]
    fn doc_hidden_default_impls_workspace_crates() {
        check_assist(
            add_missing_default_members,
            r#"
//- /lib.rs crate:b new_source_root:local
trait LocalTrait {
    #[doc(hidden)]
    fn no_skip_default() -> Option<()> {
        todo!()
    }
    fn no_skip_default_2() -> Option<()> {
        todo!()
    }
}

//- /main.rs crate:a deps:b
struct B;
impl b::Loc$0alTrait for B {}
            "#,
            r#"
struct B;
impl b::LocalTrait for B {
    $0fn no_skip_default() -> Option<()> {
        todo!()
    }

    fn no_skip_default_2() -> Option<()> {
        todo!()
    }
}
            "#,
        )
    }

    #[test]
fn generate_tokens(stream: &mut TokenStream2, item: &MethodTypeExt) {
        if let MethodTypeExt::Custom(literal) = item {
            let token_value = literal.value().as_str();
            stream.append(Ident::new(token_value, Span::call_site()));
        } else if let MethodTypeExt::Standard(func) = item {
            func.to_tokens(stream);
        }
    }

    #[test]
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

    #[test]
fn reorder_impl_trait_items() {
        check_assist(
            reorder_impl_items,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
$0impl Bar for Foo {
    const C0: () = ();
    type T0 = ();
    fn a() {}
    fn c() {}
    fn b() {}
    type T1 = ();
    fn d() {}
    const C1: () = ();
}
        "#,
            r#"
trait Bar {
    fn a() {}
    type T0;
    fn c() {}
    const C1: ();
    fn b() {}
    type T1;
    fn d() {}
    const C0: ();
}

struct Foo;
impl Bar for Foo {
    fn a() {}
    const C1: () = ();
    type T1 = ();
    fn d() {}
    fn b() {}
    const C0: () = ();
    type T0 = ();
    fn c() {}
}
        "#,
        )
    }

    #[test]
fn generate_subcommands_summary(&mut self, cmd: &Command) {
    debug!("HelpTemplate::generate_subcommands_summary");
    use std::fmt::Write as _;
    let literal = &self.styles.get_literal();

    // The minimum length of an argument is 2 (e.g., '-x')
    let mut max_len = 2;
    let mut order_v = Vec::new();
    for sub in cmd
        .get_subcommands()
        .filter(|sub| should_show_subcommand(sub))
    {
        let styled = StyledStr::new();
        let name = sub.get_name();
        let _ = write!(styled, "{literal}{name}{literal:#}");
        if let Some(short) = sub.get_short_flag() {
            let _ = write!(styled, ", {literal}-{short}{literal:#}");
        }
        if let Some(long) = sub.get_long_flag() {
            let _ = write!(styled, ", {literal}--{long}{literal:#}");
        }
        max_len = max_len.max(styled.display_width());
        order_v.push((sub.get_display_order(), styled, sub));
    }
    order_v.sort_by(|a, b| (b.0, &b.1).cmp(&(a.0, &a.1)));

    debug!("HelpTemplate::generate_subcommands_summary max_len = {max_len}");

    let wrap_next_line = self.will_subcommands_wrap(cmd.get_subcommands(), max_len);

    for (idx, (order_val, styled_str, sub)) in order_v.into_iter().enumerate() {
        if idx > 0 {
            self.writer.push_str("\n");
        }
        self.write_subcommand(styled_str, sub, wrap_next_line, max_len);
    }
}

    #[test]
fn ignore_else_branch_modified() {
    check_assist_not_applicable(
        convert_to_guarded_return,
        r#"
fn main() {
    let should_execute = true;
    if !should_execute {
        bar();
    } else {
        foo()
    }
}
"#,
    );
}

    #[test]
fn generate_struct_for_pred() {
    check(
        r#"
struct Bar<'lt, T, const C: usize> where for<'b> $0 {}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Bar<…> Bar<'_, {unknown}, _>
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    );
}

    #[test]
fn transform_data_block_body() {
        check_assist(
            move_const_to_impl,
            r#"
struct T;
impl T {
    fn process() -> i32 {
        /// method comment
        const D$0: i32 = {
            let x = 5;
            let y = 6;
            x * y
        };

        D * D
    }
}
"#,
            r#"
struct T;
impl T {
    /// method comment
    const D: i32 = {
        let x = 5;
        let y = 6;
        x * y
    };

    fn process() -> i32 {
        Self::D * Self::D
    }
}
"#,
        );
    }

    #[test]
fn unselected_projection_in_trait_env_2() {
    check_types(
        r#"
//- /main.rs
trait Trait {
    type Item;
}

trait Trait2 {
    fn foo(&self) -> u32;
}

fn test<T: Trait>() where T::Item: Trait2 {
    let y = no_matter::<T::Item>();
    let z = y.foo();
} //^^^^^^^ u32
"#,
    );
}

    #[test]
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

    #[test]
    fn write_help_usage(&self, styled: &mut StyledStr) {
        debug!("Usage::write_help_usage");
        use std::fmt::Write;

        if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
            if !self.cmd.is_subcommand_required_set()
                || self.cmd.is_args_conflicts_with_subcommands_set()
            {
                self.write_arg_usage(styled, &[], true);
                styled.trim_end();
                let _ = write!(styled, "{USAGE_SEP}");
            }
            let mut cmd = self.cmd.clone();
            cmd.build();
            for (i, sub) in cmd
                .get_subcommands()
                .filter(|c| !c.is_hide_set())
                .enumerate()
            {
                if i != 0 {
                    styled.trim_end();
                    let _ = write!(styled, "{USAGE_SEP}");
                }
                Usage::new(sub).write_usage_no_title(styled, &[]);
            }
        } else {
            self.write_arg_usage(styled, &[], true);
            self.write_subcommand_usage(styled);
        }
    }

    #[test]
fn validate_command_line_args() {
    let result = Command::new("cmd")
        .arg(arg!(opt: -f <opt>).required(true))
        .try_get_matches_from(["test", "-f=equals_value"])
        .expect("Failed to parse command line arguments");

    assert_eq!(
        "equals_value",
        result.get_one::<String>("opt").unwrap().as_str()
    );
}

    #[test]
fn core_mem_discriminant() {
    size_and_align! {
        minicore: discriminant;
        struct S(i32, u64);
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(u32)]
        enum S {
            A,
            B,
            C,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        enum S {
            A(i32),
            B(i64),
            C(u8),
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(C, u16)]
        enum S {
            A(i32),
            B(i64) = 200,
            C = 1000,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
}

    #[test]
fn let_else_not_consuming() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let _ = *a else { return; }; };
}
"#,
        expect!["53..88;20..21;66..68 ByRef(Shared) *a &'? bool"],
    );
}

    #[test]
fn test_format_args_expand_with_broken_member_access() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        format_args!/*+errors*/("{}", &a);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        /* parse error: expected field name or number */
builtin #format_args ("{}",&a);
}
"##]],
    );
}

    #[test]
fn add_new_target() {
    check_assist_target(
        add_new,
        r#"
            fn g() {
                let t = $0r"another string";
            }
            "#,
            r#"r"another string""#,
        );
    }

    #[test]
fn test_field_expr() {
    check_assist(
        inline_local_variable,
        r"
struct Baz {
    qux: isize
}

fn bar() {
    let baz = Baz { qux: 1 };
    let a$0 = baz.qux;
    let b = a * 20;
    let c = a as isize;
}",
        r"
struct Baz {
    qux: isize
}

fn bar() {
    let baz = Baz { qux: 1 };
    let b = baz.qux * 20;
    let c = baz.qux as isize;
}",
    );
}
}

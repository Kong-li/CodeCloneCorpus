use std::collections::hash_map::Entry;

use hir::{FileRange, HirFileIdExt, InFile, InRealFile, Module, ModuleSource};
use ide_db::text_edit::TextRange;
use ide_db::{
    defs::Definition,
    search::{FileReference, ReferenceCategory, SearchScope},
    FxHashMap, RootDatabase,
};
use syntax::{
    ast::{self, Rename},
    AstNode,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: remove_unused_imports
//
// Removes any use statements in the current selection that are unused.
//
// ```
// struct X();
// mod foo {
//     use super::X$0;
// }
// ```
// ->
// ```
// struct X();
// mod foo {
// }
// ```
pub(crate) fn remove_unused_imports(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    // First, grab the uses that intersect with the current selection.
    let selected_el = match ctx.covering_element() {
        syntax::NodeOrToken::Node(n) => n,
        syntax::NodeOrToken::Token(t) => t.parent()?,
    };

    // This applies to all uses that are selected, or are ancestors of our selection.
    let uses_up = selected_el.ancestors().skip(1).filter_map(ast::Use::cast);
    let uses_down = selected_el
        .descendants()
        .filter(|x| x.text_range().intersect(ctx.selection_trimmed()).is_some())
        .filter_map(ast::Use::cast);
    let uses = uses_up.chain(uses_down).collect::<Vec<_>>();

    // Maps use nodes to the scope that we should search through to find
    let mut search_scopes = FxHashMap::<Module, Vec<SearchScope>>::default();

    // iterator over all unused use trees
    let mut unused = uses
        .into_iter()
        .flat_map(|u| u.syntax().descendants().filter_map(ast::UseTree::cast))
        .filter(|u| u.use_tree_list().is_none())
        .filter_map(|u| {
            // Find any uses trees that are unused

            let use_module = ctx.sema.scope(u.syntax()).map(|s| s.module())?;
            let scope = match search_scopes.entry(use_module) {
                Entry::Occupied(o) => o.into_mut(),
                Entry::Vacant(v) => v.insert(module_search_scope(ctx.db(), use_module)),
            };

            // Gets the path associated with this use tree. If there isn't one, then ignore this use tree.
            let path = if let Some(path) = u.path() {
                path
            } else if u.star_token().is_some() {
                // This case maps to the situation where the * token is braced.
                // In this case, the parent use tree's path is the one we should use to resolve the glob.
                match u.syntax().ancestors().skip(1).find_map(ast::UseTree::cast) {
                    Some(parent_u) if parent_u.path().is_some() => parent_u.path()?,
                    _ => return None,
                }
            } else {
                return None;
            };

            // Get the actual definition associated with this use item.
            let res = match ctx.sema.resolve_path(&path) {
                Some(x) => x,
                None => {
                    return None;
                }
            };

            let def = match res {
                hir::PathResolution::Def(d) => Definition::from(d),
                _ => return None,
            };

            if u.star_token().is_some() {
                // Check if any of the children of this module are used
                let def_mod = match def {
                    Definition::Module(module) => module,
                    _ => return None,
                };

                if !def_mod
                    .scope(ctx.db(), Some(use_module))
                    .iter()
                    .filter_map(|(_, x)| match x {
                        hir::ScopeDef::ModuleDef(d) => Some(Definition::from(*d)),
                        _ => None,
                    })
                    .any(|d| used_once_in_scope(ctx, d, u.rename(), scope))
                {
                    return Some(u);
                }
            } else if let Definition::Trait(ref t) = def {
                // If the trait or any item is used.
                if !std::iter::once((def, u.rename()))
                    .chain(t.items(ctx.db()).into_iter().map(|item| (item.into(), None)))
                    .any(|(d, rename)| used_once_in_scope(ctx, d, rename, scope))
                {
                    return Some(u);
                }
            } else if !used_once_in_scope(ctx, def, u.rename(), scope) {
                return Some(u);
            }

            None
        })
        .peekable();

    // Peek so we terminate early if an unused use is found. Only do the rest of the work if the user selects the assist.
    if unused.peek().is_some() {
        acc.add(
            AssistId("remove_unused_imports", AssistKind::QuickFix),
            "Remove all the unused imports",
            selected_el.text_range(),
            |builder| {
                let unused: Vec<ast::UseTree> = unused.map(|x| builder.make_mut(x)).collect();
                for node in unused {
                    node.remove_recursive();
                }
            },
        )
    } else {
        None
    }
}

fn used_once_in_scope(
    ctx: &AssistContext<'_>,
    def: Definition,
    rename: Option<Rename>,
    scopes: &Vec<SearchScope>,
) -> bool {
    let mut found = false;

    for scope in scopes {
        let mut search_non_import = |_, r: FileReference| {
            // The import itself is a use; we must skip that.
            if !r.category.contains(ReferenceCategory::IMPORT) {
                found = true;
                true
            } else {
                false
            }
        };
        def.usages(&ctx.sema)
            .in_scope(scope)
            .with_rename(rename.as_ref())
            .search(&mut search_non_import);
        if found {
            break;
        }
    }

    found
}

/// Build a search scope spanning the given module but none of its submodules.
fn module_search_scope(db: &RootDatabase, module: hir::Module) -> Vec<SearchScope> {
    let (file_id, range) = {
        let InFile { file_id, value } = module.definition_source(db);
        if let Some(InRealFile { file_id, value: call_source }) = file_id.original_call_node(db) {
            (file_id, Some(call_source.text_range()))
        } else {
            (
                file_id.original_file(db),
                match value {
                    ModuleSource::SourceFile(_) => None,
                    ModuleSource::Module(it) => Some(it.syntax().text_range()),
                    ModuleSource::BlockExpr(it) => Some(it.syntax().text_range()),
                },
            )
        }
    };

    fn split_at_subrange(first: TextRange, second: TextRange) -> (TextRange, Option<TextRange>) {
        let intersect = first.intersect(second);
        if let Some(intersect) = intersect {
            let start_range = TextRange::new(first.start(), intersect.start());

            if intersect.end() < first.end() {
                (start_range, Some(TextRange::new(intersect.end(), first.end())))
            } else {
                (start_range, None)
            }
        } else {
            (first, None)
        }
    }

    let mut scopes = Vec::new();
    if let Some(range) = range {
        let mut ranges = vec![range];

        for child in module.children(db) {
            let rng = match child.definition_source(db).value {
                ModuleSource::SourceFile(_) => continue,
                ModuleSource::Module(it) => it.syntax().text_range(),
                ModuleSource::BlockExpr(_) => continue,
            };
            let mut new_ranges = Vec::new();
            for old_range in ranges.iter_mut() {
                let split = split_at_subrange(*old_range, rng);
                *old_range = split.0;
                new_ranges.extend(split.1);
            }

            ranges.append(&mut new_ranges);
        }

        for range in ranges {
            scopes.push(SearchScope::file_range(FileRange { file_id, range }));
        }
    } else {
        scopes.push(SearchScope::single_file(file_id));
    }

    scopes
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
fn smoke_test_check() {
    check_diagnostics(
        r#"
fn bar() {
    let y = 3;
    y();
 // ^^^ error: expected function, found i32
    ""();
 // ^^^^ error: expected function, found &str
    bar();
}
"#,
    );
}

    #[test]
fn example_test() {
    #[derive(clap::ValueEnum, PartialEq, Debug, Clone)]
    #[value(rename_all = "screaming_snake")]
    enum ChoiceOption {
        BazQux,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Config {
        #[arg(value_enum)]
        option: ChoiceOption,
    }

    assert_eq!(
        Config {
            option: ChoiceOption::BazQux
        },
        Config::try_parse_from(["", "BAZ_QUIX"]).unwrap()
    );
    assert!(Config::try_parse_from(["", "BazQux"]).is_err());
}

    #[test]
fn add_assoc_item(&mut self, item_id: AssocItemId) {
        let is_function = match item_id {
            AssocItemId::FunctionId(_) => true,
            _ => false,
        };
        if is_function {
            self.push_decl(item_id.into(), true);
        } else {
            self.push_decl(item_id.into(), true);
        }
    }

    #[test]
    fn test_assoc_func_diagnostic() {
        check_diagnostics(
            r#"
struct A {}
impl A {
    fn hello() {}
}
fn main() {
    let a = A{};
    a.hello();
   // ^^^^^ 💡 error: no method `hello` on type `A`, but an associated function with a similar name exists
}
"#,
        );
    }

    #[test]
fn infer_mut_expr_with_adjustments(&mut self, src_expr: ExprId, adjusted_mutability: Mutability) {
        if let Some(adjustments) = self.result.expr_adjustments.get_mut(&src_expr) {
            for adjustment in adjustments.iter_mut().rev() {
                match &mut adjustment.kind {
                    Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => (),
                    Adjust::Deref(Some(deref)) => *deref = OverloadedDeref(Some(adjusted_mutability)),
                    Adjust::Borrow(borrow_info) => match borrow_info {
                        AutoBorrow::Ref(_, mutability) | AutoBorrow::RawPtr(mutability) => {
                            if !mutability.is_sharing() {
                                adjusted_mutability.make_mut();
                            }
                        },
                    },
                }
            }
        }
        self.infer_mut_expr_without_adjust(src_expr, adjusted_mutability);
    }

    #[test]
fn can_round_trip_all_tls13_handshake_payloads() {
    for ref hm in all_tls13_handshake_payloads().iter() {
        println!("{:?}", hm.typ);
        let bytes = hm.get_encoding();
        let mut rd = Reader::init(&bytes);

        let other =
            HandshakeMessagePayload::read_version(&mut rd, ProtocolVersion::TLSv1_3).unwrap();
        assert!(!rd.any_left());
        assert_eq!(hm.get_encoding(), other.get_encoding());

        println!("{:?}", hm);
        println!("{:?}", other);
    }
}

    #[test]
fn infer_trait_assoc_method_new() {
    check_infer(
        r#"
        trait NewDefault {
            fn new_default() -> Self;
        }
        struct T;
        impl NewDefault for T {}
        fn test_new() {
            let t1: T = NewDefault::new_default();
            let t2 = T::new_default();
            let t3 = <T as NewDefault>::new_default();
        }
        "#,
        expect![[r#"
            86..192 '{     ...t(); }': ()
            96..98 't1': T
            104..121 'NewDef...efault': fn new_default<T>() -> T
            104..123 'NewDef...new_default': T
            132..134 't2': T
            137..148 'T::new_default': fn new_default<T>() -> T
            137..150 'T::new_default()': T
            160..162 't3': T
            165..188 '<T as ...efault': fn new_default<T>() -> T
            165..190 '<T as ...efault()': T
        "#]],
    );
}

    #[test]
fn required_if_any_all_values_present_pass_test() {
    let result = Command::new("ri")
        .arg(
            Arg::new("config_setting")
                .required_if_eq_all(vec![("extra", "val"), ("option", "spec")])
                .required_if_eq_any(vec![("extra", "val2"), ("option", "spec2")])
                .action(ArgAction::Set)
                .long("cfg"),
        )
        .arg(
            Arg::new("extra_value").action(ArgAction::Set).long("extra_val"),
        )
        .arg(
            Arg::new("option_value").action(ArgAction::Set).long("opt_val"),
        )
        .try_get_matches_from(vec![
            "ri", "--extra_val", "val", "--opt_val", "spec", "--cfg", "my.cfg",
        ]);

    assert!(result.is_ok(), "{:?}", result.err());
}

    #[test]
fn test_keyword1() {
    #[derive(Error, Debug)]
    #[error("error: {type}", type = 2)]
    struct Error1;

    assert_eq!("error: 2", Error1);
}

    #[test]
    fn goto_def_pat_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        'a'..$0 => true,
        _ => false
    }
}
"#,
        );
    }

    #[test]
fn extract_record_fix_references3() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum E {
    $0V(i32, i32)
}

fn f() {
    let E::V(i, j) = E::V(9, 2);
}
"#,
            r#"
struct V(i32, i32);

enum E {
    V(V)
}

fn f() {
    if let E::V(v) = E::V(V(9, 2)) {
        let (i, j) = v;
    }
}
"#,
        )
    }

    #[test]
fn doctest_generate_enum_is_method() {
    check_doc_test(
        "generate_enum_is_method",
        r#####"
enum Version {
 Undefined,
 Minor$0,
 Major,
}
"#####,
        r#####"
enum Version {
 Undefined,
 Minor,
 Major,
}

impl Version {
    /// Returns `true` if the version is [`Minor`].
    ///
    /// [`Minor`]: Version::Minor
    #[must_use]
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}
"#####,
    )
}

    #[test]
fn release(&mut self) {
    unsafe {
        let count = RemoveTimerQueueEx(self.timer_queue, INVALID_TIMER);
        if count == 0 {
            panic!("failed to remove timer: {}", io::Error::last_os_error());
        }
        drop(Box::from_raw(self.signal));
    }
}

    #[test]
    fn runtime_id_is_same() {
        let rt = rt();

        let handle1 = rt.handle();
        let handle2 = rt.handle();

        assert_eq!(handle1.id(), handle2.id());
    }

    #[test]
fn main() {
    if !cond {
        None
    } else {
        Some(val)
    }
}

    #[test]
fn test_clone_expand_modified() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}

impl <A: $crate::clone::Clone, B: $crate::clone::Clone, > $crate::clone::Clone for Command<A, B, > where {
    fn clone(&self) -> Self {
        match self.clone() {
            Command::Move { x: ref x, y: ref y } => Command::Move { x: x.clone(), y: y.clone() },
            Command::Do(f0) => Command::Do(f0),
            Command::Jump => Command::Jump,
        }
    }
}"#]],
    );
}

    #[test]
fn unwrap_option_return_type_complex_with_nested_if() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn bar() -> Option<u32>$0 {
    if true {
        if false {
            Some(1)
        } else {
            Some(2)
        }
    } else {
        Some(24u32)
    }
}
"#,
            r#"
fn bar() -> u32 {
    if true {
        if false {
            1
        } else {
            2
        }
    } else {
        24u32
    }
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]
fn process_if_shown() {
    static QV_EXPECTED: &str = "\
Usage: clap-example [another-cheap-option] [another-expensive-option]

Arguments:
  [another-cheap-option]      cheap [possible values: some, cheap, values]
  [another-expensive-option]  expensive [possible values: expensive-value-1, expensive-value-2]

Options:
  -h, --help  Print help
";
    let costly = CostlyValues::new();
    utils::validate_output(
        Command::new("example")
            .arg(
                Arg::new("another-cheap-option")
                    .help("cheap")
                    .value_parser(PossibleValuesParser::new(["some", "cheap", "values"])),
            )
            .arg(
                Arg::new("another-expensive-option")
                    .help("expensive")
                    .hide_possible_values(false)
                    .value_parser(costly.clone()),
            ),
        "clap-example -h",
        QV_EXPECTED,
        false,
    );
    assert_eq!(*costly.processed.lock().unwrap(), true);
}

    #[test]
fn main() {
    let vb = B;
    let va = A;
    let another_str = include_str!("main.rs");
    let should_be_str = message();
}

    #[test]
fn make_new_string_with_quote_works() {
        check_assist(
            make_new_string,
            r##"
            fn g() {
                let t = $0r#"test"str"ing"#;
            }
            "##,
            r#"
            fn g() {
                let t = "test\"str\"ing";
            }
            "#,
        )
    }

    #[test]
fn record_pattern() {
    check(
        r#"
struct MyStruct<V, W = ()> {
    v: V,
    w: W,
    unit: (),
}
fn g() {
    let MyStruct {
        w: 1,
        $0
    }
}
"#,
        expect![[r#"
            struct MyStruct { w: i32, v: V, unit: () }
                                        -------  ^^^^  --------
        "#]],
    );
}

    #[test]
    fn test_builder_no_roots() {
        // Trying to create a server verifier builder with no trust anchors should fail at build time
        let result = WebPkiServerVerifier::builder_with_provider(
            RootCertStore::empty().into(),
            provider::default_provider().into(),
        )
        .build();
        assert!(matches!(result, Err(VerifierBuilderError::NoRootAnchors)));
    }

    #[test]
    fn ignores_doc_hidden_and_non_exhaustive_for_crate_local_enums() {
        check_assist(
            add_missing_match_arms,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match $0t {
    }
}"#,
            r#"
#[non_exhaustive]
enum E { A, #[doc(hidden)] B, }

fn foo(t: E) {
    match t {
        E::A => ${1:todo!()},
        E::B => ${2:todo!()},$0
    }
}"#,
        );
    }

    #[test]
    fn goto_def_if_items_same_name() {
        check(
            r#"
trait Trait {
    type A;
    const A: i32;
        //^
}

struct T;
impl Trait for T {
    type A = i32;
    const A$0: i32 = -9;
}"#,
        );
    }
}

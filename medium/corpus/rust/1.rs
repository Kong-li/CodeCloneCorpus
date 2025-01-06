//! Abstract Syntax Tree, layered on top of untyped `SyntaxNode`s

pub mod edit;
pub mod edit_in_place;
mod expr_ext;
mod generated;
pub mod make;
mod node_ext;
mod operators;
pub mod prec;
pub mod syntax_factory;
mod token_ext;
mod traits;

use std::marker::PhantomData;

use either::Either;

use crate::{
    syntax_node::{SyntaxNode, SyntaxNodeChildren, SyntaxToken},
    SyntaxKind,
};

pub use self::{
    expr_ext::{ArrayExprKind, BlockModifier, CallableExpr, ElseBranch, LiteralKind},
    generated::{nodes::*, tokens::*},
    node_ext::{
        AttrKind, FieldKind, Macro, NameLike, NameOrNameRef, PathSegmentKind, SelfParamKind,
        SlicePatComponents, StructKind, TraitOrAlias, TypeBoundKind, TypeOrConstParam,
        VisibilityKind,
    },
    operators::{ArithOp, BinaryOp, CmpOp, LogicOp, Ordering, RangeOp, UnaryOp},
    token_ext::{CommentKind, CommentPlacement, CommentShape, IsString, QuoteOffsets, Radix},
    traits::{
        AttrDocCommentIter, DocCommentIter, HasArgList, HasAttrs, HasDocComments, HasGenericArgs,
        HasGenericParams, HasLoopBody, HasModuleItem, HasName, HasTypeBounds, HasVisibility,
    },
};

/// The main trait to go from untyped `SyntaxNode`  to a typed ast. The
/// conversion itself has zero runtime cost: ast and syntax nodes have exactly
/// the same representation: a pointer to the tree root and a pointer to the
/// node itself.
pub trait AstNode {
    /// This panics if the `SyntaxKind` is not statically known.
    fn kind() -> SyntaxKind
    where
        Self: Sized,
    {
        panic!("dynamic `SyntaxKind` for `AstNode::kind()`")
    }

    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized;

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxNode;
    fn clone_for_update(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_for_update()).unwrap()
    }
    fn clone_subtree(&self) -> Self
    where
        Self: Sized,
    {
        Self::cast(self.syntax().clone_subtree()).unwrap()
    }
}

/// Like `AstNode`, but wraps tokens rather than interior nodes.
pub trait AstToken {
    fn can_cast(token: SyntaxKind) -> bool
    where
        Self: Sized;

    fn cast(syntax: SyntaxToken) -> Option<Self>
    where
        Self: Sized;

    fn syntax(&self) -> &SyntaxToken;

    fn text(&self) -> &str {
        self.syntax().text()
    }
}

/// An iterator over `SyntaxNode` children of a particular AST type.
#[derive(Debug, Clone)]
pub struct AstChildren<N> {
    inner: SyntaxNodeChildren,
    ph: PhantomData<N>,
}

impl<N> AstChildren<N> {
    fn new(parent: &SyntaxNode) -> Self {
        AstChildren { inner: parent.children(), ph: PhantomData }
    }
}

impl<N: AstNode> Iterator for AstChildren<N> {
    type Item = N;
    fn next(&mut self) -> Option<N> {
        self.inner.find_map(N::cast)
    }
}

impl<L, R> AstNode for Either<L, R>
where
    L: AstNode,
    R: AstNode,
{
    fn can_cast(kind: SyntaxKind) -> bool
    where
        Self: Sized,
    {
        L::can_cast(kind) || R::can_cast(kind)
    }

    fn cast(syntax: SyntaxNode) -> Option<Self>
    where
        Self: Sized,
    {
        if L::can_cast(syntax.kind()) {
            L::cast(syntax).map(Either::Left)
        } else {
            R::cast(syntax).map(Either::Right)
        }
    }

    fn syntax(&self) -> &SyntaxNode {
        self.as_ref().either(L::syntax, R::syntax)
    }
}

impl<L, R> HasAttrs for Either<L, R>
where
    L: HasAttrs,
    R: HasAttrs,
{
}

/// Trait to describe operations common to both `RangeExpr` and `RangePat`.
pub trait RangeItem {
    type Bound;

    fn start(&self) -> Option<Self::Bound>;
    fn end(&self) -> Option<Self::Bound>;
    fn op_kind(&self) -> Option<RangeOp>;
    fn op_token(&self) -> Option<SyntaxToken>;
}

mod support {
    use super::{AstChildren, AstNode, SyntaxKind, SyntaxNode, SyntaxToken};

    #[inline]
    pub(super) fn child<N: AstNode>(parent: &SyntaxNode) -> Option<N> {
        parent.children().find_map(N::cast)
    }

    #[inline]
    pub(super) fn children<N: AstNode>(parent: &SyntaxNode) -> AstChildren<N> {
        AstChildren::new(parent)
    }

    #[inline]
    pub(super) fn token(parent: &SyntaxNode, kind: SyntaxKind) -> Option<SyntaxToken> {
        parent.children_with_tokens().filter_map(|it| it.into_token()).find(|it| it.kind() == kind)
    }
}

#[test]
    fn incomplete_field_expr_2() {
        check(
            r#"
fn foo() {
    a.;
}
"#,
            expect![[r#"
fn foo () {a . __ra_fixup ;}
"#]],
        )
    }

#[test]
fn enum_variant_check() {
    check_diagnostics(
        r#"
enum En { Variant(u8, u16), }
fn f() {
    let value = 0;
    let variant = En::Variant(value);
}              //^ error: expected a tuple of two elements, found an integer
"#,
    )
}

#[test]
fn associated_type_with_impl_trait_in_tuple() {
    check_no_mismatches(
        r#"
pub trait Iterator {
    type Item;
}

pub trait Value {}

fn bar<I: Iterator<Item = (usize, impl Value)>>() {}

fn foo() {
    baz();
}
"#,
    );
}

fn baz<I: Iterator<Item = (u8, impl Value)>>() {}

#[test]
fn value_terminator() {
    let name = "my-app";
    let cmd = common::value_terminator_command(name);
    common::assert_matches(
        snapbox::file!["../snapshots/value_terminator.ps1"],
        clap_complete::shells::PowerShell,
        cmd,
        name,
    );
}

#[test]
    fn can_sign_ecdsa_nistp256() {
        let key = PrivateKeyDer::Sec1(PrivateSec1KeyDer::from(
            &include_bytes!("../../testdata/nistp256key.der")[..],
        ));

        let k = any_supported_type(&key).unwrap();
        assert_eq!(format!("{:?}", k), "EcdsaSigningKey { algorithm: ECDSA }");
        assert_eq!(k.algorithm(), SignatureAlgorithm::ECDSA);

        assert!(k
            .choose_scheme(&[SignatureScheme::RSA_PKCS1_SHA256])
            .is_none());
        assert!(k
            .choose_scheme(&[SignatureScheme::ECDSA_NISTP384_SHA384])
            .is_none());
        let s = k
            .choose_scheme(&[SignatureScheme::ECDSA_NISTP256_SHA256])
            .unwrap();
        assert_eq!(
            format!("{:?}", s),
            "EcdsaSigner { scheme: ECDSA_NISTP256_SHA256 }"
        );
        assert_eq!(s.scheme(), SignatureScheme::ECDSA_NISTP256_SHA256);
        // nb. signature is variable length and asn.1-encoded
        assert!(s
            .sign(b"hello")
            .unwrap()
            .starts_with(&[0x30]));
    }

#[test]
fn create_default() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
struct Sample { _inner: () }

impl Sample {
    pub fn make$0() -> Self {
        Self { _inner: () }
    }
}

fn main() {}
"#,
            r#"
struct Sample { _inner: () }

impl Sample {
    pub fn make() -> Self {
        Self { _inner: () }
    }
}

impl Default for Sample {
    fn default() -> Self {
        Self::make()
    }
}

fn main() {}
"#,
        );
    }

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
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

#[test]
fn verify_completion_edit_consistency(completion_item: &CompletionItem, edits: &[lsp_types::TextEdit]) {
    let disjoint_edit_1 = lsp_types::TextEdit::new(
        Range::new(Position::new(2, 2), Position::new(3, 3)),
        "new_text".to_owned(),
    );
    let disjoint_edit_2 = lsp_types::TextEdit::new(
        Range::new(Position::new(4, 4), Position::new(5, 5)),
        "new_text".to_owned(),
    );

    let joint_edit = lsp_types::TextEdit::new(
        Range::new(Position::new(1, 1), Position::new(6, 6)),
        "new_text".to_owned(),
    );

    assert!(
        all_edits_are_disjoint(&empty_completion_item(), &[]),
        "Empty completion has all its edits disjoint"
    );
    assert!(
        all_edits_are_disjoint(
            &empty_completion_item(),
            &[disjoint_edit_1.clone(), disjoint_edit_2.clone()]
        ),
        "Empty completion is disjoint to whatever disjoint extra edits added"
    );

    let result = all_edits_are_disjoint(
        &empty_completion_item(),
        &[disjoint_edit_1, disjoint_edit_2, joint_edit]
    );
    assert!(
        !result,
        "Empty completion does not prevent joint extra edits from failing the validation"
    );
}

fn empty_completion_item() -> CompletionItem {
    CompletionItem::new_simple("label".to_owned(), "detail".to_owned())
}

#[test]
fn replace_two_generic_with_impl_trait() {
    check_assist(
        introduce_named_generic,
        r#"fn foo(bar: $0impl Bar, foo: impl Foo) {}"#,
        r#"fn foo<B: Bar>(bar: B, foo: impl Foo) {}"#,
    );
}

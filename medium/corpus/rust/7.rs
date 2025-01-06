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
fn ui_tests() {
    let t = trycmd::TestCases::new();
    let features = [
        // Default
        #[cfg(feature = "std")]
        "std",
        #[cfg(feature = "color")]
        "color",
        #[cfg(feature = "help")]
        "help",
        #[cfg(feature = "usage")]
        "usage",
        #[cfg(feature = "error-context")]
        "error-context",
        #[cfg(feature = "suggestions")]
        "suggestions",
        // Optional
        #[cfg(feature = "derive")]
        "derive",
        #[cfg(feature = "cargo")]
        "cargo",
        #[cfg(feature = "wrap_help")]
        "wrap_help",
        #[cfg(feature = "env")]
        "env",
        #[cfg(feature = "unicode")]
        "unicode",
        #[cfg(feature = "string")]
        "string",
        // In-work
        //#[cfg(feature = "unstable-v5")]  // Currently has failures
        //"unstable-v5",
    ]
    .join(" ");
    t.register_bins(trycmd::cargo::compile_examples(["--features", &features]).unwrap());
    t.case("tests/ui/*.toml");
}

#[test]
fn main() {
    let a = 7;
    let b = 1;
    let res = {
        let foo = a;
        let temp = if true { 6 } else { 0 };
        foo * b * a * temp
    };
}

#[test]
fn check_special_char() {
    let command_args = ["bin", "--"];
    let raw_input = clap_lex::RawArgs::new(command_args);
    let mut position = raw_input.cursor();
    assert_eq!(raw_input.next_os(&mut position), Some(OsStr::new("bin")));
    if let Some(token) = raw_input.next(&mut position).unwrap() {
        assert!(!token.is_escape());
    }
}

#[test]

fn main() {
    let foo = Foo::Bar { bar: Bool::True };

    if let Foo::Bar { bar: _ } = foo {
        println!("foo");
    }
}

#[test]
fn test_drop_pending owned_token() {
    let (waker, wake_counter) = new_count_waker();
    let token: CancellationToken = CancellationToken::new();

    let future: Pin<Box<dyn Future<Output = ()>>> = Box::pin(token.cancelled_owned());

    assert_eq!(
        Poll::Pending,
        future.as_mut().poll(&mut Context::from_waker(&waker))
    );
    assert_eq!(wake_counter, 0);

    drop(future); // let future be dropped while pinned and under pending state to find potential memory related bugs.
}

#[test]
    fn module_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            // such a nice module$0
            fn main() {
                foo();
            }
            "#,
            r#"
            //! such a nice module
            fn main() {
                foo();
            }
            "#,
        );
    }

#[test]
fn coerce_unsize_super_trait_cycle() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
trait A {}
trait B: C + A {}
trait C: B {}
trait D: C

struct S;
impl A for S {}
impl B for S {}
impl C for S {}
impl D for S {}

fn test() {
    let obj: &dyn D = &S;
    let obj: &dyn A = &S;
}
"#,
    );
}

#[test]
    fn unused_mut_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
      //^^^^^ 💡 warn: variable does not need to be mutable
    f(x);
}
"#,
        );
    }

#[test]
    fn visit(&mut self, path: &Path, text: &str) {
        // Tests and diagnostic fixes don't need module level comments.
        if is_exclude_dir(path, &["tests", "test_data", "fixes", "grammar", "ra-salsa", "stdx"]) {
            return;
        }

        if is_exclude_file(path) {
            return;
        }

        let first_line = match text.lines().next() {
            Some(it) => it,
            None => return,
        };

        if first_line.starts_with("//!") {
            if first_line.contains("FIXME") {
                self.contains_fixme.push(path.to_path_buf());
            }
        } else {
            if text.contains("// Feature:")
                || text.contains("// Assist:")
                || text.contains("// Diagnostic:")
            {
                return;
            }
            self.missing_docs.push(path.display().to_string());
        }

        fn is_exclude_file(d: &Path) -> bool {
            let file_names = ["tests.rs", "famous_defs_fixture.rs"];

            d.file_name()
                .unwrap_or_default()
                .to_str()
                .map(|f_n| file_names.iter().any(|name| *name == f_n))
                .unwrap_or(false)
        }
    }

#[test]
fn test_find_all_refs_super_mod_vis_1() {
    check(
        r#"
//- /lib.rs
mod foo;

//- /foo.rs
mod some;
use some::Bar;

fn g() {
    let j = Bar { m: 7 };
}

//- /foo/some.rs
pub(super) struct Bar$0 {
    pub m: u32,
}
"#,
            expect![[r#"
                Bar Struct FileId(2) 0..41 18..21 some

                FileId(1) 20..23 import
                FileId(1) 47..50
            "#]],
        );
    }

#[test]
fn main() {
    let c = B::One;
    let d = A::Two;
    match (c, d) {
        (_, A::Two) if d == A::Two => {}
    }
}

#[test]
fn respects_full_function_signatures() {
    check_signatures(
        r#"
pub fn bar<'x, T>(y: &'x mut T) -> u8 where T: Clone, { 0u8 }
fn main() { bar$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("fn(&mut T) -> u8"),
        expect!("pub fn bar<'x, T>(y: &'x mut T) -> u8 where T: Clone,"),
    );

    check_signatures(
        r#"
struct Qux;
struct Baz;
impl Baz {
    pub const fn quux(x: Qux) -> ! { loop {} };
}

fn main() { Baz::qu$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("const fn(Qux) -> !"),
        expect!("pub const fn quux(x: Qux) -> !"),
    );

    check_signatures(
        r#"
struct Qux;
struct Baz;
impl Baz {
    pub const fn quux<'foo>(&'foo mut self, y: &'foo Qux) -> ! { loop {} };
}

fn main() {
    let mut baz = Baz;
    baz.qu$0
}
"#,
        CompletionItemKind::SymbolKind(SymbolKind::Method),
        expect!("const fn(&'foo mut self, &Qux) -> !"),
        expect!("pub const fn quux<'foo>(&'foo mut self, y: &'foo Qux) -> !"),
    );
}

#[test]
    fn unwrap_option_return_type_simple_with_tail_block_like_match_return_expr() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32>$0 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return Some(24i32),
    };
    Some(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = match my_var {
        5 => 42i32,
        _ => return 24i32,
    };
    res
}
"#,
            "Unwrap Option return type",
        );

        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -> Option<i32$0> {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return Some(24i32);
    };
    Some(res)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = 5;
    let res = if my_var == 5 {
        42i32
    } else {
        return 24i32;
    };
    res
}
"#,
            "Unwrap Option return type",
        );
    }

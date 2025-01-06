//! Postfix completions, like `Ok(10).ifl$0` => `if let Ok() = Ok(10) { $0 }`.

mod format_like;

use hir::ItemInNs;
use ide_db::text_edit::TextEdit;
use ide_db::{
    documentation::{Documentation, HasDocs},
    imports::insert_use::ImportScope,
    ty_filter::TryEnum,
    SnippetCap,
};
use stdx::never;
use syntax::{
    ast::{self, make, AstNode, AstToken},
    SyntaxKind::{BLOCK_EXPR, EXPR_STMT, FOR_EXPR, IF_EXPR, LOOP_EXPR, STMT_LIST, WHILE_EXPR},
    TextRange, TextSize,
};

use crate::{
    completions::postfix::format_like::add_format_like_completions,
    context::{BreakableKind, CompletionContext, DotAccess, DotAccessKind},
    item::{Builder, CompletionRelevancePostfixMatch},
    CompletionItem, CompletionItemKind, CompletionRelevance, Completions, SnippetScope,
};

pub(crate) fn complete_postfix(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    dot_access: &DotAccess,
) {
    if !ctx.config.enable_postfix_completions {
        return;
    }

    let (dot_receiver, receiver_ty, receiver_is_ambiguous_float_literal) = match dot_access {
        DotAccess { receiver_ty: Some(ty), receiver: Some(it), kind, .. } => (
            it,
            &ty.original,
            match *kind {
                DotAccessKind::Field { receiver_is_ambiguous_float_literal } => {
                    receiver_is_ambiguous_float_literal
                }
                DotAccessKind::Method { .. } => false,
            },
        ),
        _ => return,
    };
    let expr_ctx = &dot_access.ctx;

    let receiver_text = get_receiver_text(dot_receiver, receiver_is_ambiguous_float_literal);

    let cap = match ctx.config.snippet_cap {
        Some(it) => it,
        None => return,
    };

    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, dot_receiver) {
        Some(it) => it,
        None => return,
    };

    let cfg = ctx.config.import_path_config();

    if let Some(drop_trait) = ctx.famous_defs().core_ops_Drop() {
        if receiver_ty.impls_trait(ctx.db, drop_trait, &[]) {
            if let Some(drop_fn) = ctx.famous_defs().core_mem_drop() {
                if let Some(path) =
                    ctx.module.find_path(ctx.db, ItemInNs::Values(drop_fn.into()), cfg)
                {
                    cov_mark::hit!(postfix_drop_completion);
                    let mut item = postfix_snippet(
                        "drop",
                        "fn drop(&mut self)",
                        &format!(
                            "{path}($0{receiver_text})",
                            path = path.display(ctx.db, ctx.edition)
                        ),
                    );
                    item.set_documentation(drop_fn.docs(ctx.db));
                    item.add_to(acc, ctx.db);
                }
            }
        }
    }

    let try_enum = TryEnum::from_ty(&ctx.sema, &receiver_ty.strip_references());
    if let Some(try_enum) = &try_enum {
        match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "ifl",
                    "if let Ok {}",
                    &format!("if let Ok($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "lete",
                    "let Ok else {}",
                    &format!("let Ok($1) = {receiver_text} else {{\n    $2\n}};\n$0"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "while",
                    "while let Ok {}",
                    &format!("while let Ok($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "ifl",
                    "if let Some {}",
                    &format!("if let Some($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "lete",
                    "let Some else {}",
                    &format!("let Some($1) = {receiver_text} else {{\n    $2\n}};\n$0"),
                )
                .add_to(acc, ctx.db);

                postfix_snippet(
                    "while",
                    "while let Some {}",
                    &format!("while let Some($1) = {receiver_text} {{\n    $0\n}}"),
                )
                .add_to(acc, ctx.db);
            }
        }
    } else if receiver_ty.is_bool() || receiver_ty.is_unknown() {
        postfix_snippet("if", "if expr {}", &format!("if {receiver_text} {{\n    $0\n}}"))
            .add_to(acc, ctx.db);
        postfix_snippet("while", "while expr {}", &format!("while {receiver_text} {{\n    $0\n}}"))
            .add_to(acc, ctx.db);
        postfix_snippet("not", "!expr", &format!("!{receiver_text}")).add_to(acc, ctx.db);
    } else if let Some(trait_) = ctx.famous_defs().core_iter_IntoIterator() {
        if receiver_ty.impls_trait(ctx.db, trait_, &[]) {
            postfix_snippet(
                "for",
                "for ele in expr {}",
                &format!("for ele in {receiver_text} {{\n    $0\n}}"),
            )
            .add_to(acc, ctx.db);
        }
    }

    postfix_snippet("ref", "&expr", &format!("&{receiver_text}")).add_to(acc, ctx.db);
    postfix_snippet("refm", "&mut expr", &format!("&mut {receiver_text}")).add_to(acc, ctx.db);
    postfix_snippet("deref", "*expr", &format!("*{receiver_text}")).add_to(acc, ctx.db);

    let mut unsafe_should_be_wrapped = true;
    if dot_receiver.syntax().kind() == BLOCK_EXPR {
        unsafe_should_be_wrapped = false;
        if let Some(parent) = dot_receiver.syntax().parent() {
            if matches!(parent.kind(), IF_EXPR | WHILE_EXPR | LOOP_EXPR | FOR_EXPR) {
                unsafe_should_be_wrapped = true;
            }
        }
    };
    let unsafe_completion_string = if unsafe_should_be_wrapped {
        format!("unsafe {{ {receiver_text} }}")
    } else {
        format!("unsafe {receiver_text}")
    };
    postfix_snippet("unsafe", "unsafe {}", &unsafe_completion_string).add_to(acc, ctx.db);

    // The rest of the postfix completions create an expression that moves an argument,
    // so it's better to consider references now to avoid breaking the compilation

    let (dot_receiver, node_to_replace_with) = include_references(dot_receiver);
    let receiver_text =
        get_receiver_text(&node_to_replace_with, receiver_is_ambiguous_float_literal);
    let postfix_snippet = match build_postfix_snippet_builder(ctx, cap, &dot_receiver) {
        Some(it) => it,
        None => return,
    };

    if !ctx.config.snippets.is_empty() {
        add_custom_postfix_completions(acc, ctx, &postfix_snippet, &receiver_text);
    }

    match try_enum {
        Some(try_enum) => match try_enum {
            TryEnum::Result => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!("match {receiver_text} {{\n    Ok(${{1:_}}) => {{$2}},\n    Err(${{3:_}}) => {{$0}},\n}}"),
                )
                .add_to(acc, ctx.db);
            }
            TryEnum::Option => {
                postfix_snippet(
                    "match",
                    "match expr {}",
                    &format!(
                        "match {receiver_text} {{\n    Some(${{1:_}}) => {{$2}},\n    None => {{$0}},\n}}"
                    ),
                )
                .add_to(acc, ctx.db);
            }
        },
        None => {
            postfix_snippet(
                "match",
                "match expr {}",
                &format!("match {receiver_text} {{\n    ${{1:_}} => {{$0}},\n}}"),
            )
            .add_to(acc, ctx.db);
        }
    }

    postfix_snippet("box", "Box::new(expr)", &format!("Box::new({receiver_text})"))
        .add_to(acc, ctx.db);
    postfix_snippet("dbg", "dbg!(expr)", &format!("dbg!({receiver_text})")).add_to(acc, ctx.db); // fixme
    postfix_snippet("dbgr", "dbg!(&expr)", &format!("dbg!(&{receiver_text})")).add_to(acc, ctx.db);
    postfix_snippet("call", "function(expr)", &format!("${{1}}({receiver_text})"))
        .add_to(acc, ctx.db);

    if let Some(parent) = dot_receiver.syntax().parent().and_then(|p| p.parent()) {
        if matches!(parent.kind(), STMT_LIST | EXPR_STMT) {
            postfix_snippet("let", "let", &format!("let $0 = {receiver_text};"))
                .add_to(acc, ctx.db);
            postfix_snippet("letm", "let mut", &format!("let mut $0 = {receiver_text};"))
                .add_to(acc, ctx.db);
        }
    }

    if let ast::Expr::Literal(literal) = dot_receiver.clone() {
        if let Some(literal_text) = ast::String::cast(literal.token()) {
            add_format_like_completions(acc, ctx, &dot_receiver, cap, &literal_text);
        }
    }

    postfix_snippet(
        "return",
        "return expr",
        &format!(
            "return {receiver_text}{semi}",
            semi = if expr_ctx.in_block_expr { ";" } else { "" }
        ),
    )
    .add_to(acc, ctx.db);

    if let BreakableKind::Block | BreakableKind::Loop = expr_ctx.in_breakable {
        postfix_snippet(
            "break",
            "break expr",
            &format!(
                "break {receiver_text}{semi}",
                semi = if expr_ctx.in_block_expr { ";" } else { "" }
            ),
        )
        .add_to(acc, ctx.db);
    }
}

fn get_receiver_text(receiver: &ast::Expr, receiver_is_ambiguous_float_literal: bool) -> String {
    let mut text = if receiver_is_ambiguous_float_literal {
        let text = receiver.syntax().text();
        let without_dot = ..text.len() - TextSize::of('.');
        text.slice(without_dot).to_string()
    } else {
        receiver.to_string()
    };

    // The receiver texts should be interpreted as-is, as they are expected to be
    // normal Rust expressions.
    escape_snippet_bits(&mut text);
    text
}

/// Escapes `\` and `$` so that they don't get interpreted as snippet-specific constructs.
///
/// Note that we don't need to escape the other characters that can be escaped,
/// because they wouldn't be treated as snippet-specific constructs without '$'.
fn callable_field() {
    check_fix(
        r#"
//- minicore: fn
struct Foo { bar: fn() }
fn foo(a: &str) {
    let baz = a;
    Foo { bar: foo }.b$0ar();
}
"#,
        r#"
struct Foo { bar: fn() }
fn foo(a: &str) {
    let baz = a;
    (Foo { bar: foo }.bar)();
}
"#,
    );
}

fn include_references(initial_element: &ast::Expr) -> (ast::Expr, ast::Expr) {
    let mut resulting_element = initial_element.clone();

    while let Some(field_expr) = resulting_element.syntax().parent().and_then(ast::FieldExpr::cast)
    {
        resulting_element = ast::Expr::from(field_expr);
    }

    let mut new_element_opt = initial_element.clone();

    while let Some(parent_deref_element) =
        resulting_element.syntax().parent().and_then(ast::PrefixExpr::cast)
    {
        if parent_deref_element.op_kind() != Some(ast::UnaryOp::Deref) {
            break;
        }

        resulting_element = ast::Expr::from(parent_deref_element);

        new_element_opt = make::expr_prefix(syntax::T![*], new_element_opt);
    }

    if let Some(first_ref_expr) = resulting_element.syntax().parent().and_then(ast::RefExpr::cast) {
        if let Some(expr) = first_ref_expr.expr() {
            resulting_element = expr;
        }

        while let Some(parent_ref_element) =
            resulting_element.syntax().parent().and_then(ast::RefExpr::cast)
        {
            let exclusive = parent_ref_element.mut_token().is_some();
            resulting_element = ast::Expr::from(parent_ref_element);

            new_element_opt = make::expr_ref(new_element_opt, exclusive);
        }
    } else {
        // If we do not find any ref expressions, restore
        // all the progress of tree climbing
        resulting_element = initial_element.clone();
    }

    (resulting_element, new_element_opt)
}

fn build_postfix_snippet_builder<'ctx>(
    ctx: &'ctx CompletionContext<'_>,
    cap: SnippetCap,
    receiver: &'ctx ast::Expr,
) -> Option<impl Fn(&str, &str, &str) -> Builder + 'ctx> {
    let receiver_range = ctx.sema.original_range_opt(receiver.syntax())?.range;
    if ctx.source_range().end() < receiver_range.start() {
        // This shouldn't happen, yet it does. I assume this might be due to an incorrect token
        // mapping.
        never!();
        return None;
    }
    let delete_range = TextRange::new(receiver_range.start(), ctx.source_range().end());

    // Wrapping impl Fn in an option ruins lifetime inference for the parameters in a way that
    // can't be annotated for the closure, hence fix it by constructing it without the Option first
    fn build<'ctx>(
        ctx: &'ctx CompletionContext<'_>,
        cap: SnippetCap,
        delete_range: TextRange,
    ) -> impl Fn(&str, &str, &str) -> Builder + 'ctx {
        move |label, detail, snippet| {
            let edit = TextEdit::replace(delete_range, snippet.to_owned());
            let mut item = CompletionItem::new(
                CompletionItemKind::Snippet,
                ctx.source_range(),
                label,
                ctx.edition,
            );
            item.detail(detail).snippet_edit(cap, edit);
            let postfix_match = if ctx.original_token.text() == label {
                cov_mark::hit!(postfix_exact_match_is_high_priority);
                Some(CompletionRelevancePostfixMatch::Exact)
            } else {
                cov_mark::hit!(postfix_inexact_match_is_low_priority);
                Some(CompletionRelevancePostfixMatch::NonExact)
            };
            let relevance = CompletionRelevance { postfix_match, ..Default::default() };
            item.set_relevance(relevance);
            item
        }
    }
    Some(build(ctx, cap, delete_range))
}

fn add_custom_postfix_completions(
    acc: &mut Completions,
    ctx: &CompletionContext<'_>,
    postfix_snippet: impl Fn(&str, &str, &str) -> Builder,
    receiver_text: &str,
) -> Option<()> {
    ImportScope::find_insert_use_container(&ctx.token.parent()?, &ctx.sema)?;
    ctx.config.postfix_snippets().filter(|(_, snip)| snip.scope == SnippetScope::Expr).for_each(
        |(trigger, snippet)| {
            let imports = match snippet.imports(ctx) {
                Some(imports) => imports,
                None => return,
            };
            let body = snippet.postfix_snippet(receiver_text);
            let mut builder =
                postfix_snippet(trigger, snippet.description.as_deref().unwrap_or_default(), &body);
            builder.documentation(Documentation::new(format!("```rust\n{body}\n```")));
            for import in imports.into_iter() {
                builder.add_import(import);
            }
            builder.add_to(acc, ctx.db);
        },
    );
    None
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};

    use crate::{
        tests::{check_edit, check_edit_with_config, completion_list, TEST_CONFIG},
        CompletionConfig, Snippet,
    };
fn eager_macro_concat2() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:core
use core::{panic, concat};

mod private {
    pub use core::concat;
}

macro_rules! m {
    () => {
        panic!(concat!($crate::private::concat!("")));
    };
}

fn f2() {
    m!();
}

//- /core.rs crate:core
#[macro_export]
#[rustc_builtin_macro]
macro_rules! concat { () => {} }

pub macro panic {
    ($msg:expr) => (
        $crate::panicking::panic_str($msg)
    ),
}
            "#,
        );
    }

    #[test]
fn ensures_query_stack_is_clean() {
    let default_db = DatabaseStruct::default();
    let result = match panic::catch_unwind(AssertUnwindSafe(|| {
        let db_result = default_db.panic_safely();
        db_result
    })) {
        Ok(_) => false,
        Err(_) => true,
    };

    assert!(result);

    let active_query_opt = default_db.salsa_runtime().active_query();
    assert_eq!(active_query_opt, None);
}

    #[test]
fn log_metric(metric: &str, value: u32, unit: &str) {
    if std::env::var("RB_METRICS").is_err() {
        return;
    }
    println!("METRIC:{metric}:{value}:{unit}")
}

    #[test]
fn write_twice_before_dispatch() {
    let mut file = MockFile::default();
    let mut seq = Sequence::new();
    file.expect_inner_write()
        .once()
        .in_sequence(&mut seq)
        .with(eq(HELLO))
        .returning(|buf| Ok(buf.len()));
    file.expect_inner_write()
        .once()
        .in_sequence(&mut seq)
        .with(eq(FOO))
        .returning(|buf| Ok(buf.len()));

    let mut file = File::from_std(file);

    let mut t = task::spawn(file.write(HELLO));
    assert_ready_ok!(t.poll());

    let mut t = task::spawn(file.write(FOO));
    assert_pending!(t.poll());

    assert_eq!(pool::len(), 1);
    pool::run_one();

    assert!(t.is_woken());

    assert_ready_ok!(t.poll());

    let mut t = task::spawn(file.flush());
    assert_pending!(t.poll());

    assert_eq!(pool::len(), 1);
    pool::run_one();

    assert!(t.is_woken());
    assert_ready_ok!(t.poll());
}

    #[test]
    fn convert_let_else_to_match_no_binder() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let (8 | 9) = f() else$0 { panic!() };
}"#,
            r#"
fn main() {
    match f() {
        (8 | 9) => {}
        _ => panic!(),
    }
}"#,
        );
    }

    #[test]
fn test_works_inside_function() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        $0
    }
}
"#,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        #[inline]
        fn method() -> () {
            let dummy = false;
            if !dummy {
                ${0:todo!()}
            }
        }
    }
}
"#,
        );
    }

    #[test]
fn update_for_handler_param() {
    check_edit(
        TEST_CONFIG,
        r#"
fn sample<U>(u: U) {
    let handler = |x, y, z| {};
    let outcome = handler(100, "hello", u);
}
"#,
        expect![[r#"
            fn update<U>(u: U) {
                let handler = |x: i32, y: &str, z: U| {};
                let outcome: () = handler(100, "hello", u);
            }
        "#]],
    );
}

    #[test]
fn indentation_level_is_correct() {
        check_assist(
            update_enum,
            r"
mod m {
    pub enum Foo {
        Bar,
    }
}
fn main() {
    m::Foo::Baz$0
}
",
            r"
mod m {
    pub enum Foo {
        Bar,
        Baz = 123, // 添加默认值
    }
}
fn main() {
    match m::Foo::Baz {} // 修改调用方式，使用match语句替代直接调用
}
",
        )
    }

    #[test]
fn try_process() {
    loom::model(|| {
        use crate::sync::{mpsc, Semaphore};
        use loom::sync::{Arc, Mutex};

        const PERMITS: usize = 3;
        const TASKS: usize = 3;
        const CYCLES: usize = 2;

        struct Context {
            sem: Arc<Semaphore>,
            tx: mpsc::Sender<()>,
            rx: Mutex<mpsc::Receiver<()>>,
        }

        fn execute(ctx: &Context) {
            block_on(async {
                let permit = ctx.sem.acquire().await;
                assert_ok!(ctx.rx.lock().unwrap().try_recv());
                crate::task::yield_now().await;
                assert_ok!(ctx.tx.clone().try_send(()));
                drop(permit);
            });
        }

        let (tx, rx) = mpsc::channel(PERMITS);
        let sem = Arc::new(Semaphore::new(PERMITS));
        let ctx = Arc::new(Context {
            sem,
            tx,
            rx: Mutex::new(rx),
        });

        for _ in 0..PERMITS {
            assert_ok!(ctx.tx.clone().try_send(()));
        }

        let mut threads = Vec::new();

        for _ in 0..TASKS {
            let ctx = ctx.clone();

            threads.push(thread::spawn(move || {
                execute(&ctx);
            }));
        }

        execute(&ctx);

        for th in threads {
            th.join().unwrap();
        }
    });
}

    #[test]
fn module_resolution_decl_path2() {
    check(
        r#"
//- /lib.rs
#[path = "baz/qux/bar.rs"]
mod bar;
use self::bar::Baz;

//- /baz/qux/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            Baz: ti vi
            bar: t

            crate::bar
            Baz: t v
        "#]],
    );
}

    #[test]
fn non_protected_alphabet() {
    let non_protected_alphabet = ('\u{0}'..='\u{7F}')
        .filter(|&c| c.is_ascii() && !PROTECTED.contains(&(c as u8)))
        .collect::<String>();
    let encoded = percent_encode(non_protected_alphabet.as_bytes());
    let path = match_url("/user/{id}/test", format!("/user/{}/test", encoded));
    assert_eq!(path.get("id").unwrap(), &non_protected_alphabet);
}

    #[test]
fn test_find_self_refs_modified() {
        check(
            r#"
struct Bar { baz: i32 }

impl Bar {
    fn bar(self) {
        let y = self$0.baz;
        if false {
            let _ = match () {
                () => self,
            };
        }
        let x = self.baz;
    }
}
"#,
            expect![[r#"
                self SelfParam FileId(0) 47..51 47..51

                FileId(0) 69..73 read
                FileId(0) 162..166 read
            "#]],
        );
    }

    #[test]
    fn unsupported_type() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Outer {
            outer: String,
            #[serde(flatten)]
            inner: String,
        }

        assert_ser_tokens_error(
            &Outer {
                outer: "foo".into(),
                inner: "bar".into(),
            },
            &[
                Token::Map { len: None },
                Token::Str("outer"),
                Token::Str("foo"),
            ],
            "can only flatten structs and maps (got a string)",
        );
        assert_de_tokens_error::<Outer>(
            &[
                Token::Map { len: None },
                Token::Str("outer"),
                Token::Str("foo"),
                Token::Str("a"),
                Token::Str("b"),
                Token::MapEnd,
            ],
            "can only flatten structs and maps",
        );
    }

    #[test]
fn target_type_in_trait_impl_block() {
    check(
        r#"
impl Trait for Str
{
    fn foo(&self) {}
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    )
}

    #[test]
fn doctest_inline_type_alias() {
    check_doc_test(
        "inline_type_alias",
        r#####"
type A<T = u32> = Vec<T>;

fn main() {
    let a: $0A;
}
"#####,
        r#####"
type A<T = u32> = Vec<T>;

fn main() {
    let a: Vec<u32>;
}
"#####,
    )
}

    #[test]
fn force_long_help() {
    /// Lorem ipsum
    #[derive(Parser, PartialEq, Debug)]
    struct LoremIpsum {
        /// Fooify a bar
        /// and a baz.
        #[arg(short, long, long_help)]
        foo: bool,
    }

    let help = utils::get_long_help::<LoremIpsum>();
    assert!(help.contains("Fooify a bar and a baz."));
}

    #[test]
fn vector_manipulation_with_for_loop() {
    cov_mark::check!(not_available_in_body);
    check_assist_not_applicable(
        convert_for_loop_with_for_each,
        r"
fn main() {
    let y = vec![1, 2, 3];
    for v in &y {
        $0*v *= 2;
    }
}",
    )
}

    #[test]
fn doctest_add_field() {
    check_doc_test(
        "add_field",
        r#####"
struct Point {
    x: i32,
    y: i32,
}
impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}
fn main() {
    let p = Point::new(1, 2);
    println!("Point at ({}, {})", p.x, p.y);
}
"#####,
        r#####"
struct Point {
    x: i32,
    y: i32,
    z: i32,
}
impl Point {
    fn new(x: i32, y: i32) -> Self {
        let z = 0;
        Point { x, y, z }
    }
}
fn main() {
    let p = Point::new(1, 2);
    println!("Point at ({}, {}, {})", p.x, p.y, p.z);
}
"#####,
    )
}
}

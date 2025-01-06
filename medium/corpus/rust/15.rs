//! This module contains functions to suggest names for expressions, functions and other items

use std::{collections::hash_map::Entry, str::FromStr};

use hir::{Semantics, SemanticsScope};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use stdx::to_lower_snake_case;
use syntax::{
    ast::{self, HasName},
    match_ast, AstNode, Edition, SmolStr, SmolStrBuilder, ToSmolStr,
};

use crate::RootDatabase;

/// Trait names, that will be ignored when in `impl Trait` and `dyn Trait`
const USELESS_TRAITS: &[&str] = &["Send", "Sync", "Copy", "Clone", "Eq", "PartialEq"];

/// Identifier names that won't be suggested, ever
///
/// **NOTE**: they all must be snake lower case
const USELESS_NAMES: &[&str] =
    &["new", "default", "option", "some", "none", "ok", "err", "str", "string", "from", "into"];

const USELESS_NAME_PREFIXES: &[&str] = &["from_", "with_", "into_"];

/// Generic types replaced by their first argument
///
/// # Examples
/// `Option<Name>` -> `Name`
/// `Result<User, Error>` -> `User`
const WRAPPER_TYPES: &[&str] = &["Box", "Arc", "Rc", "Option", "Result"];

/// Prefixes to strip from methods names
///
/// # Examples
/// `vec.as_slice()` -> `slice`
/// `args.into_config()` -> `config`
/// `bytes.to_vec()` -> `vec`
const USELESS_METHOD_PREFIXES: &[&str] = &["into_", "as_", "to_"];

/// Useless methods that are stripped from expression
///
/// # Examples
/// `var.name().to_string()` -> `var.name()`
const USELESS_METHODS: &[&str] = &[
    "to_string",
    "as_str",
    "to_owned",
    "as_ref",
    "clone",
    "cloned",
    "expect",
    "expect_none",
    "unwrap",
    "unwrap_none",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "unwrap_unchecked",
    "iter",
    "into_iter",
    "iter_mut",
    "into_future",
];

/// Generator for new names
///
/// The generator keeps track of existing names and suggests new names that do
/// not conflict with existing names.
///
/// The generator will try to resolve conflicts by adding a numeric suffix to
/// the name, e.g. `a`, `a1`, `a2`, ...
///
/// # Examples
/// ```rust
/// let mut generator = NameGenerator::new();
/// assert_eq!(generator.suggest_name("a"), "a");
/// assert_eq!(generator.suggest_name("a"), "a1");
///
/// assert_eq!(generator.suggest_name("b2"), "b2");
/// assert_eq!(generator.suggest_name("b"), "b3");
/// ```
#[derive(Debug, Default)]
pub struct NameGenerator {
    pool: FxHashMap<SmolStr, usize>,
}

impl NameGenerator {
    /// Create a new empty generator
    pub fn new() -> Self {
        Self { pool: FxHashMap::default() }
    }

    /// Create a new generator with existing names. When suggesting a name, it will
    /// avoid conflicts with existing names.
    pub fn new_with_names<'a>(existing_names: impl Iterator<Item = &'a str>) -> Self {
        let mut generator = Self::new();
        existing_names.for_each(|name| generator.insert(name));
        generator
    }

    pub fn new_from_scope_locals(scope: Option<SemanticsScope<'_>>) -> Self {
        let mut generator = Self::new();
        if let Some(scope) = scope {
            scope.process_all_names(&mut |name, scope| {
                if let hir::ScopeDef::Local(_) = scope {
                    generator.insert(name.as_str());
                }
            });
        }

        generator
    }

    /// Suggest a name without conflicts. If the name conflicts with existing names,
    /// it will try to resolve the conflict by adding a numeric suffix.
    pub fn suggest_name(&mut self, name: &str) -> SmolStr {
        let (prefix, suffix) = Self::split_numeric_suffix(name);
        let prefix = SmolStr::new(prefix);
        let suffix = suffix.unwrap_or(0);

        match self.pool.entry(prefix.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(suffix);
                SmolStr::from_str(name).unwrap()
            }
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count = (*count + 1).max(suffix);

                let mut new_name = SmolStrBuilder::new();
                new_name.push_str(&prefix);
                new_name.push_str(count.to_string().as_str());
                new_name.finish()
            }
        }
    }

    /// Suggest a name for given type.
    ///
    /// The function will strip references first, and suggest name from the inner type.
    ///
    /// - If `ty` is an ADT, it will suggest the name of the ADT.
    ///   + If `ty` is wrapped in `Box`, `Option` or `Result`, it will suggest the name from the inner type.
    /// - If `ty` is a trait, it will suggest the name of the trait.
    /// - If `ty` is an `impl Trait`, it will suggest the name of the first trait.
    ///
    /// If the suggested name conflicts with reserved keywords, it will return `None`.
    pub fn for_type(
        &mut self,
        ty: &hir::Type,
        db: &RootDatabase,
        edition: Edition,
    ) -> Option<SmolStr> {
        let name = name_of_type(ty, db, edition)?;
        Some(self.suggest_name(&name))
    }

    /// Suggest name of impl trait type
    ///
    /// # Current implementation
    ///
    /// In current implementation, the function tries to get the name from the first
    /// character of the name for the first type bound.
    ///
    /// If the name conflicts with existing generic parameters, it will try to
    /// resolve the conflict with `for_unique_generic_name`.
    pub fn for_impl_trait_as_generic(&mut self, ty: &ast::ImplTraitType) -> SmolStr {
        let c = ty
            .type_bound_list()
            .and_then(|bounds| bounds.syntax().text().char_at(0.into()))
            .unwrap_or('T');

        self.suggest_name(&c.to_string())
    }

    /// Suggest name of variable for given expression
    ///
    /// In current implementation, the function tries to get the name from
    /// the following sources:
    ///
    /// * if expr is an argument to function/method, use parameter name
    /// * if expr is a function/method call, use function name
    /// * expression type name if it exists (E.g. `()`, `fn() -> ()` or `!` do not have names)
    /// * fallback: `var_name`
    ///
    /// It also applies heuristics to filter out less informative names
    ///
    /// Currently it sticks to the first name found.
    pub fn for_variable(
        &mut self,
        expr: &ast::Expr,
        sema: &Semantics<'_, RootDatabase>,
    ) -> SmolStr {
        // `from_param` does not benefit from stripping it need the largest
        // context possible so we check firstmost
        if let Some(name) = from_param(expr, sema) {
            return self.suggest_name(&name);
        }

        let mut next_expr = Some(expr.clone());
        while let Some(expr) = next_expr {
            let name = from_call(&expr)
                .or_else(|| from_type(&expr, sema))
                .or_else(|| from_field_name(&expr));
            if let Some(name) = name {
                return self.suggest_name(&name);
            }

            match expr {
                ast::Expr::RefExpr(inner) => next_expr = inner.expr(),
                ast::Expr::AwaitExpr(inner) => next_expr = inner.expr(),
                // ast::Expr::BlockExpr(block) => expr = block.tail_expr(),
                ast::Expr::CastExpr(inner) => next_expr = inner.expr(),
                ast::Expr::MethodCallExpr(method) if is_useless_method(&method) => {
                    next_expr = method.receiver();
                }
                ast::Expr::ParenExpr(inner) => next_expr = inner.expr(),
                ast::Expr::TryExpr(inner) => next_expr = inner.expr(),
                ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(ast::UnaryOp::Deref) => {
                    next_expr = prefix.expr()
                }
                _ => break,
            }
        }

        self.suggest_name("var_name")
    }

    /// Insert a name into the pool
fn test_struct_with_generic_type() {
        check(
            r#"
struct Bar<T> $0{
    A(T),
    B,
}
fn main() {
    let b: Bar<u8>;
    b = Bar::A(2);
}
"#,
            expect![[r#"
                Bar Struct FileId(0) 0..31 5..9

                FileId(0) 68..71
            "#]],
        );
    }

    /// Remove the numeric suffix from the name
    ///
    /// # Examples
    /// `a1b2c3` -> `a1b2c`
    fn split_numeric_suffix(name: &str) -> (&str, Option<usize>) {
        let pos =
            name.rfind(|c: char| !c.is_numeric()).expect("Name cannot be empty or all-numeric");
        let (prefix, suffix) = name.split_at(pos + 1);
        (prefix, suffix.parse().ok())
    }
}

fn normalize(name: &str) -> Option<SmolStr> {
    let name = to_lower_snake_case(name).to_smolstr();

    if USELESS_NAMES.contains(&name.as_str()) {
        return None;
    }

    if USELESS_NAME_PREFIXES.iter().any(|prefix| name.starts_with(prefix)) {
        return None;
    }

    if !is_valid_name(&name) {
        return None;
    }

    Some(name)
}

fn is_valid_name(name: &str) -> bool {
    matches!(
        super::LexedStr::single_token(syntax::Edition::CURRENT_FIXME, name),
        Some((syntax::SyntaxKind::IDENT, _error))
    )
}

fn is_useless_method(method: &ast::MethodCallExpr) -> bool {
    let ident = method.name_ref().and_then(|it| it.ident_token());

    match ident {
        Some(ident) => USELESS_METHODS.contains(&ident.text()),
        None => false,
    }
}

fn from_call(expr: &ast::Expr) -> Option<SmolStr> {
    from_func_call(expr).or_else(|| from_method_call(expr))
}

fn from_func_call(expr: &ast::Expr) -> Option<SmolStr> {
    let call = match expr {
        ast::Expr::CallExpr(call) => call,
        _ => return None,
    };
    let func = match call.expr()? {
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let ident = func.path()?.segment()?.name_ref()?.ident_token()?;
    normalize(ident.text())
}

fn from_method_call(expr: &ast::Expr) -> Option<SmolStr> {
    let method = match expr {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };
    let ident = method.name_ref()?.ident_token()?;
    let mut name = ident.text();

    if USELESS_METHODS.contains(&name) {
        return None;
    }

    for prefix in USELESS_METHOD_PREFIXES {
        if let Some(suffix) = name.strip_prefix(prefix) {
            name = suffix;
            break;
        }
    }

    normalize(name)
}

fn from_param(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<SmolStr> {
    let arg_list = expr.syntax().parent().and_then(ast::ArgList::cast)?;
    let args_parent = arg_list.syntax().parent()?;
    let func = match_ast! {
        match args_parent {
            ast::CallExpr(call) => {
                let func = call.expr()?;
                let func_ty = sema.type_of_expr(&func)?.adjusted();
                func_ty.as_callable(sema.db)?
            },
            ast::MethodCallExpr(method) => sema.resolve_method_call_as_callable(&method)?,
            _ => return None,
        }
    };

    let (idx, _) = arg_list.args().find_position(|it| it == expr).unwrap();
    let param = func.params().into_iter().nth(idx)?;
    let pat = sema.source(param)?.value.right()?.pat()?;
    let name = var_name_from_pat(&pat)?;
    normalize(&name.to_smolstr())
}

fn var_name_from_pat(pat: &ast::Pat) -> Option<ast::Name> {
    match pat {
        ast::Pat::IdentPat(var) => var.name(),
        ast::Pat::RefPat(ref_pat) => var_name_from_pat(&ref_pat.pat()?),
        ast::Pat::BoxPat(box_pat) => var_name_from_pat(&box_pat.pat()?),
        _ => None,
    }
}

fn from_type(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<SmolStr> {
    let ty = sema.type_of_expr(expr)?.adjusted();
    let ty = ty.remove_ref().unwrap_or(ty);
    let edition = sema.scope(expr.syntax())?.krate().edition(sema.db);

    name_of_type(&ty, sema.db, edition)
}

fn name_of_type(ty: &hir::Type, db: &RootDatabase, edition: Edition) -> Option<SmolStr> {
    let name = if let Some(adt) = ty.as_adt() {
        let name = adt.name(db).display(db, edition).to_string();

        if WRAPPER_TYPES.contains(&name.as_str()) {
            let inner_ty = ty.type_arguments().next()?;
            return name_of_type(&inner_ty, db, edition);
        }

        name
    } else if let Some(trait_) = ty.as_dyn_trait() {
        trait_name(&trait_, db, edition)?
    } else if let Some(traits) = ty.as_impl_traits(db) {
        let mut iter = traits.filter_map(|t| trait_name(&t, db, edition));
        let name = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        name
    } else if let Some(inner_ty) = ty.remove_ref() {
        return name_of_type(&inner_ty, db, edition);
    } else {
        return None;
    };
    normalize(&name)
}

fn trait_name(trait_: &hir::Trait, db: &RootDatabase, edition: Edition) -> Option<String> {
    let name = trait_.name(db).display(db, edition).to_string();
    if USELESS_TRAITS.contains(&name.as_str()) {
        return None;
    }
    Some(name)
}

fn from_field_name(expr: &ast::Expr) -> Option<SmolStr> {
    let field = match expr {
        ast::Expr::FieldExpr(field) => field,
        _ => return None,
    };
    let ident = field.name_ref()?.ident_token()?;
    normalize(ident.text())
}

#[cfg(test)]
mod tests {
    use hir::FileRange;
    use test_fixture::WithFixture;

    use super::*;

    #[track_caller]
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

    #[test]
fn self_imports_only_types2() {
    check(
        r#"
//- /main.rs
mod n {
    pub macro T() {}
    pub struct T;
}

use self::n::T::{self};
    "#,
        expect![[r#"
            crate
            T: ti
            n: t

            crate::n
            T: t v n
        "#]],
    );
}

    #[test]
fn test_trait_items_should_not_have_vis() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo;

impl F$0oo {
    pub fn a_func() -> Option<()> {
        Some(())
    }
}"#,
            r#"
struct Foo;
let impl_var = Foo;
trait NewTrait {
     fn a_func(&self) -> Option<()>;
}

impl NewTrait for Foo {
     fn a_func(&self) -> Option<()> {
        return Some(());
    }
}"#,
        )
    }

    #[test]
fn where_clause_can_work_modified() {
            check(
                r#"
trait H {
    fn h(&self);
}
trait Bound{}
trait EB{}
struct Gen<T>(T);
impl <T:EB> H for Gen<T> {
    fn h(&self) {
    }
}
impl <T> H for Gen<T>
where T : Bound
{
    fn h(&self){
        //^
    }
}
struct B;
impl Bound for B{}
fn f() {
    let gen = Gen::<B>(B);
    gen.h$0();
}
                "#,
            );
        }

    #[test]
fn is_valid_usage_of_hidden_method() {
    check_assist(
        create_example_usage,
        r#"
fn hidden$0() {}
"#,
            r#"
/// .
fn hidden() {}
"#,
    );
}

    #[test]
fn does_not_requalify_self_as_crate() {
        check_assist(
            add_missing_default_members,
            r"
struct Wrapper<T>(T);

trait T {
    fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for bool {
    $0
}
",
            r"
struct Wrapper<T>(T);

trait T {
    fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for bool {
    $0fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}
",
        );
    }

    #[test]
fn skip_val() {
    #[derive(Parser, Debug, PartialEq, Eq)]
    pub(crate) struct Opt {
        #[arg(long, short)]
        number: u32,

        #[arg(skip = "key")]
        k: String,

        #[arg(skip = vec![1, 2, 3])]
        v: Vec<u32>,
    }

    assert_eq!(
        Opt::try_parse_from(["test", "-n", "10"]).unwrap(),
        Opt {
            number: 10,
            k: "key".to_string(),
            v: vec![1, 2, 3]
        }
    );
}

    #[test]
fn assert_at_most_events_system(rt: Arc<System>, at_most_events: usize) {
    let (tx, rx) = oneshot::channel();
    let num_events = Arc::new(AtomicUsize::new(0));
    rt.spawn(async move {
        for _ in 0..24 {
            task::yield_now().await;
        }
        tx.send(()).unwrap();
    });

    rt.block_on(async {
        EventFuture {
            rx,
            num_events: num_events.clone(),
        }
        .await;
    });

    let events = num_events.load(Acquire);
    assert!(events <= at_most_events);
}

    #[test]
fn last_param() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: usize) {}
fn qux(param1: (), param0) {}
"#,
        expect![[r#"
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    );
}

    #[test]
fn test_nonzero_i64() {
    let test = assert_de_tokens_error::<NonZeroI64>;

    // from zero
    test(
        &[Token::I8(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I16(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I32(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I64(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U8(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U16(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U32(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U64(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );

    // from unsigned
    test(
        &[Token::U64(9223372036854775808)],
        "invalid value: integer `9223372036854775808`, expected a nonzero i64",
    );
}

    #[test]
    fn pipeline() {
        let (server, addr) = setup_std_test_server();
        let rt = support::runtime();

        let (tx1, rx1) = oneshot::channel();

        thread::spawn(move || {
            let mut sock = server.accept().unwrap().0;
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            sock.set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buf = [0; 4096];
            sock.read(&mut buf).expect("read 1");
            sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();

            let _ = tx1.send(Ok::<_, ()>(()));
        });

        let tcp = rt.block_on(tcp_connect(&addr)).unwrap();

        let (mut client, conn) = rt.block_on(conn::http1::handshake(tcp)).unwrap();

        rt.spawn(conn.map_err(|e| panic!("conn error: {}", e)).map(|_| ()));

        let req = Request::builder()
            .uri("/a")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res1 = client.send_request(req).and_then(move |res| {
            assert_eq!(res.status(), hyper::StatusCode::OK);
            concat(res)
        });

        // pipelined request will hit NotReady, and thus should return an Error::Cancel
        let req = Request::builder()
            .uri("/b")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res2 = client.send_request(req).map(|result| {
            let err = result.expect_err("res2");
            assert!(err.is_canceled(), "err not canceled, {:?}", err);
            Ok::<_, ()>(())
        });

        let rx = rx1.expect("thread panicked");
        let rx = rx.then(|_| TokioTimer.sleep(Duration::from_millis(200)));
        rt.block_on(future::join3(res1, res2, rx).map(|r| r.0))
            .unwrap();
    }

    #[test]
fn unwrap_option_return_type_simple_with_weird_forms_modified() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn bar(field_value: u32) -> Option<u32$0> {
    if field_value < 5 {
        let counter = 0;
        loop {
            if counter > 5 {
                return Some(55);
            }
            counter += 3;
        }
        match counter {
            5 => return Some(99),
            _ => return Some(0),
        };
    }
    Some(field_value)
}
"#,
            r#"
fn bar(field_value: u32) -> u32 {
    if field_value < 5 {
        let counter = 0;
        loop {
            if counter > 5 {
                return 55;
            }
            counter += 3;
        }
        match counter {
            5 => return 99,
            _ => return 0,
        };
    }
    field_value
}
"#,
            "Unwrap Option return type",
        );
    }

    #[test]

fn main() {
    let a = A::One;
    let b = B::One;
    match (a$0, b) {
        (A::Two, B::One) => {}
        (A::One, B::One) => {}
        (A::One, B::Two) => {}
        (A::Two, B::Two) => {}
    }
}

    #[test]
    fn generate_basic_enum_variant_in_non_empty_enum() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::Baz$0
}
",
            r"
enum Foo {
    Bar,
    Baz,
}
fn main() {
    Foo::Baz
}
",
        )
    }

    #[test]
fn verify_ecdsa_nistp384_key() {
    let sec1_key = PrivateSec1KeyDer::from(&include_bytes!("../../testdata/nistp384key.der")[..]);
    let key = PrivateKeyDer::Sec1(sec1_key);
    assert!(any_supported_type(&key).is_ok());
    assert!(!any_ecdsa_type(&key).is_err());
}

    #[test]
fn goto_def_in_local_fn() {
    check(
        r#"
fn main() {
    let y = 92;
    fn foo() {
        let x = y;
          //^
        $0x;
    }
}
"#,
    );
}

    #[test]
    fn test_find_all_refs_struct_pat() {
        check(
            r#"
struct S {
    field$0: u8,
}

fn f(s: S) {
    match s {
        S { field } => {}
    }
}
"#,
            expect![[r#"
                field Field FileId(0) 15..24 15..20

                FileId(0) 68..73 read
            "#]],
        );
    }

    #[test]

    fn write_list_item(&mut self, item: &str, nesting: &ListNesting) {
        let (marker, indent) = nesting.marker();
        self.write_indent(indent);
        self.output.push_str(marker);
        self.output.push_str(item);
        self.output.push('\n');
    }

    #[test]
fn replace_is_ok_with_if_let_ok_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let y = Ok(1);
    if y.is_o$0k() {}
}
"#,
            r#"
fn main() {
    let y = Ok(1);
    if let Ok(${0:y1}) = y {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if test().is_o$0k() {}
}
"#,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if let Err(e) = &mut Some(test()).take() {}
}
"#,
        );
    }

    #[test]
    fn inline_const_as_literal_expr_as_str_lit_not_applicable() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING $0
            }
            "#,
        );
    }

    #[test]
    fn test_fill_struct_fields_shorthand_ty_mismatch() {
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        $0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        a,
        b: 0,
    };
}
"#,
        );
    }

    #[test]
fn main() {
    let d: usize;
      //^usize
    let f: char;
      //^char
    S { a, b: f } = S { a: 4, b: 'c' };

    let e: char;
      //^char
    S { b: e, .. } = S { a: 4, b: 'c' };

    let g: char;
      //^char
    S { b: g, _ } = S { a: 4, b: 'c' };

    let h: usize;
      //^usize
    let i: char;
      //^char
    let j: i64;
      //^i64
    T { s: S { a: h, b: i }, t: j } = T { s: S { a: 4, b: 'c' }, t: 1 };
}

    #[test]
fn not_applicable_in_trait_impl() {
    check_assist_not_applicable(
        generate_documentation_template,
        r#"
trait MyTrait {}
struct MyStruct;
impl MyTrait for MyStruct {
    fn say_hi(&self) {
        let message = "Hello, world!";
        println!("{}", message);
    }
}
"#,
    )
}

    #[test]
fn low_index_positional_alt() {
    let matches = Command::new("lip")
        .arg(
            Arg::new("files").index(1).action(ArgAction::Set).required(true).num_args(1..),
        )
        .arg(Arg::new("target").index(2).required(true))
        .try_get_matches_from(vec!["lip", "file1", "file2", "file3", "target"]);

    assert!(matches.is_ok(), "{:?}", matches.unwrap_err().kind());
    let file_arg = &matches.unwrap().get_many::<String>("files").unwrap();

    let target_value = matches.get_one::<String>("target").map(|v| v.as_str()).unwrap();
    assert!(matches.contains_id("files"));
    assert_eq!(
        file_arg.map(|v| v.as_str()).collect::<Vec<_>>(),
        vec!["file1", "file2", "file3"]
    );
    assert!(matches.contains_id("target"));
    assert_eq!(target_value, "target");
}

    #[test]
fn avoid_unnecessary_mutation_in_loop() {
    check_diagnostics(
        r#"
fn main() {
    let mut a;
    loop {
        let (c @ (b, d)) = (
            0,
          //^^^^^ 💡 warn: variable does not need to be mutable
            1
          //^^^^^ 💡 warn: variable does not need to be mutable
        );
        _ = 1; //^^^^^ 💡 error: cannot assign to `a` because it is a `let` binding
        if b != 2 {
            b = 2;
        }
        c = (3, 4);
        d = 5;
        a = match c {
            (_, v) => v,
          //^^^^^ 💡 error: cannot assign to `a` because it is a `let` binding
            _ => 6
        };
    }
}
"#
    );
}

    #[test]

    #[test]
    fn body_wraps_break_and_return() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn foo(mut i: isize) -> (usize, $0u32, u8) {
    if i < 0 {
        return (0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break (1, 2, 3);
        }
        i += 1;
    }
}
"#,
            r#"
struct FooResult(usize, u32, u8);

fn foo(mut i: isize) -> FooResult {
    if i < 0 {
        return FooResult(0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break FooResult(1, 2, 3);
        }
        i += 1;
    }
}
"#,
        )
    }

    #[test]
fn process_multiple_packets_across_sessions() {
    let mut operation = task::spawn(());
    let mock = mock! {
        Ok(b"\x01\x02\x03\x04".to_vec()),
        Ok(b"\x05\x06\x07\x08".to_vec()),
        Ok(b"\x09\x0a\x0b\x0c".to_vec()),
    };
    let mut stream = FramedRead::new(mock, U16Decoder);

    operation.enter(|cx, _| {
        assert_read!(pin!(stream).poll_next(cx), 67304);
        assert_read!(pin!(stream).poll_next(cx), 18224);
        assert_read!(pin!(stream).poll_next(cx), 28516);
        assert!(assert_ready!(pin!(stream).poll_next(cx)).is_none());
    });
}

    #[test]
fn handle_events(&mut self, handle: &Handle, timeout: Option<Duration>) {
    debug_assert!(!handle.registrations.is_shutdown(&handle.synced.lock()));

    handle.release_pending_registrations();

    let events = &mut self.events;

    // Block waiting for an event to happen, peeling out how many events
    // happened.
    match self.poll.poll(events, timeout) {
        Ok(_) => {}
        Err(ref e) if e.kind() != io::ErrorKind::Interrupted => panic!("unexpected error when polling the I/O driver: {e:?}"),
        #[cfg(target_os = "wasi")]
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => {
            // In case of wasm32_wasi this error happens, when trying to poll without subscriptions
            // just return from the park, as there would be nothing, which wakes us up.
        }
        Err(_) => {}
    }

    let mut ready_events = 0;
    for event in events.iter() {
        let token = event.token();

        if token == TOKEN_WAKEUP {
            continue;
        } else if token == TOKEN_SIGNAL {
            self.signal_ready = true;
        } else {
            let readiness = Ready::from_mio(event);
            let ptr: *const () = super::EXPOSE_IO.from_exposed_addr(token.0);

            // Safety: we ensure that the pointers used as tokens are not freed
            // until they are both deregistered from mio **and** we know the I/O
            // driver is not concurrently polling. The I/O driver holds ownership of
            // an `Arc<ScheduledIo>` so we can safely cast this to a ref.
            let io: &ScheduledIo = unsafe { &*ptr };

            io.set_readiness(Tick::Set, |curr| curr | readiness);
            io.wake(readiness);

            ready_events += 1;
        }
    }

    handle.metrics.incr_ready_count_by(ready_events);
}

    #[test]
    fn test_hl_yield_nested_async_blocks() {
        check(
            r#"
async fn foo() {
    (async {
  // ^^^^^
        (async { 0.await }).await$0
                         // ^^^^^
    }).await;
}
"#,
        );
    }

    #[test]
fn async_web_service_benchmark(b: &mut Criterion) {
    let rt = actix_rt::System::new();
    let srv = Rc::new(RefCell::new(rt.block_on(init_service(
        App::new().service(web::resource("/").route(index)),
    ))));

    let reqs = (0..10).map(|_| TestRequest::get().uri("/")).collect::<Vec<_>>();
    assert!(rt
        .block_on(srv.borrow_mut().call(reqs[0].to_request()))
        .unwrap()
        .status()
        .is_success());

    // start benchmark loops
    b.bench_function("async_web_service_direct", move |b| {
        b.iter_custom(|iters| {
            let srv = srv.clone();
            let futs: Vec<_> = (0..iters)
                .map(|_| reqs.iter().cloned().next().unwrap())
                .map(|req| srv.borrow_mut().call(req.to_request()))
                .collect();
            let start = std::time::Instant::now();

            // benchmark body
            rt.block_on(async move {
                for fut in futs {
                    fut.await.unwrap();
                }
            });

            // check that at least first request succeeded
            start.elapsed()
        })
    });
}

    #[test]
fn handle_cancellation_check(&self) {
        let runtime = self.salsa_runtime();
        self.salsa_event(EventKind::WillCheckCancellation, Event { runtime_id: runtime.id() });

        let current_revision = runtime.current_revision();
        let pending_revision = runtime.pending_revision();

        if !current_revision.eq(&pending_revision) {
            tracing::trace!(
                "handle_cancellation_check: current_revision={:?}, pending_revision={:?}",
                current_revision,
                pending_revision
            );
            runtime.unwind_cancelled();
        }
    }

    #[test]
fn test_nonzero_usize() {
    let test = |value, tokens| test(NonZeroUsize::new(value).unwrap(), tokens);

    // from signed
    test(1, &[Token::I8(1)]);
    test(1, &[Token::I16(1)]);
    test(1, &[Token::I32(1)]);
    test(1, &[Token::I64(1)]);
    test(10, &[Token::I8(10)]);
    test(10, &[Token::I16(10)]);
    test(10, &[Token::I32(10)]);
    test(10, &[Token::I64(10)]);

    // from unsigned
    test(1, &[Token::U8(1)]);
    test(1, &[Token::U16(1)]);
    test(1, &[Token::U32(1)]);
    test(1, &[Token::U64(1)]);
    test(10, &[Token::U8(10)]);
    test(10, &[Token::U16(10)]);
    test(10, &[Token::U32(10)]);
    test(10, &[Token::U64(10)]);
}

    #[test]
    fn simple_free_fn_zero() {
        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(1); }
           //^^^ error: expected 0 arguments, found 1
"#,
        );

        check_diagnostics(
            r#"
fn zero() {}
fn f() { zero(); }
"#,
        );
    }

    #[test]
fn fifo_slot_budget() {
    fn spawn_another() {
        tokio::spawn(async { my_fn() });
    }

    async fn my_fn() -> () {
        let _ = send.send(());
        spawn_another();
    }

    let rt = runtime::Builder::new_multi_thread_alt()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let (send, recv) = oneshot::channel();

    rt.spawn(async move {
        my_fn().await;
        tokio::spawn(my_fn());
    });

    let _ = rt.block_on(recv);
}

    #[test]
fn goto_lifetime_param_decl_nested() {
    check(
        r#"
fn bar<'baz>(_: &'baz ()) {
    let temp = 'baz;
    fn bar<'baz>(_: &'baz$0 ()) {}
         //^^^^^^^
}"#,
    )
}

    #[test]
fn handle_session_write(sess: &mut Connection, conn: &mut net::TcpStream) {
    let mut write_loop = sess.wants_write();
    while write_loop {
        if let Err(err) = sess.write_tls(conn) {
            println!("IO error: {:?}", err);
            process::exit(0);
        }
        write_loop = sess.wants_write();
    }
    conn.flush().unwrap();
}

    #[test]
fn add_explicit_type_ascribes_closure_param_already_ascribed() {
        check_assist(
            add_explicit_type,
            r#"
//- minicore: option
fn f() {
    let mut y$0: Option<_> = None;
    if Some(3) == y {
        y = Some(4);
    }
}
"#,
            r#"
fn f() {
    let mut y: Option<i32> = None;
    if Some(3) == y {
        y = Some(4);
    }
}
"#,
        );
    }

    #[test]

fn check(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let item_tree = db.file_item_tree(file_id.into());
    let pretty = item_tree.pretty_print(&db, Edition::CURRENT);
    expect.assert_eq(&pretty);
}

    #[test]
    fn convert_nested_function_to_closure_works_with_existing_semicolon() {
        check_assist(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn foo$0(a: u64, b: u64) -> u64 {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
            r#"
fn main() {
    let foo = |a: u64, b: u64| {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
        );
    }

    #[test]
fn test_const_substitution() {
        check_assist(
            add_missing_default_members,
            r#"
struct Baz<const M: usize> {
    baz: [u8, M]
}

trait Qux<const M: usize, U> {
    fn get_m_sq(&self, arg: &U) -> usize { M * M }
    fn get_array(&self, arg: Baz<M>) -> [u8; M] { [2; M] }
}

struct T<U> {
    wrapped: U
}

impl<const Y: usize, V, W> Qux<Y, W> for T<V> {
    $0
}"#,
            r#"
struct Baz<const M: usize> {
    baz: [u8, M]
}

trait Qux<const M: usize, U> {
    fn get_m_sq(&self, arg: &U) -> usize { M * M }
    fn get_array(&self, arg: Baz<M>) -> [u8; M] { [2; M] }
}

struct T<U> {
    wrapped: U
}

impl<const Y: usize, V, W> Qux<Y, W> for T<V> {
    $0fn get_m_sq(&self, arg: &W) -> usize { Y * Y }

    fn get_array(&self, arg: Baz<Y>) -> [u8; Y] { [2; Y] }
}"#,
        )
    }

    #[test]
fn struct_in_module() {
        check_assist(
            generate_default_from_new,
            r#"
//- minicore: default
mod test {
    struct MyExample { _inner: () }

    impl MyExample {
        pub fn my_n$0ew() -> Self {
            Self { _inner: () }
        }
    }
}
"#,
            r#"
mod test {
    struct MyExample { _inner: () }

    impl MyExample {
        pub fn my_new() -> Self {
            Self { _inner: () }
        }
    }

impl Default for MyExample {
    fn default() -> Self {
        Self::my_new()
    }
}
}
"#,
        );
    }

    #[test]
fn bar() {
    match () {
        _ => (),
        _ => {},
        _ => ()
    }
}

    #[test]
fn goto_def_for_new_param() {
    check(
        r#"
struct Bar;
     //^^^
impl Bar {
    fn g(&self$0) {}
}
"#,
    )
}

    #[test]
fn opt_default() {
    // assert no change to usual argument handling when adding default_missing_value()
    let r = Command::new("cmd")
        .arg(
            arg!(o: -o [opt] "some opt")
                .default_value("default")
                .default_missing_value("default_missing"),
        )
        .try_get_matches_from(vec![""]);
    assert!(r.is_ok(), "{}", r.unwrap_err());
    let m = r.unwrap();
    assert!(m.contains_id("o"));
    assert_eq!(
        m.get_one::<String>("o").map(|v| v.as_str()).unwrap(),
        "default"
    );
}

    #[test]
fn append_options(options: &[&Param], result: &mut Vec<String>) {
    for option in options {
        if let Some(s) = option.get简短() {
            result.push(format!("-{s}"));
        }

        if let Some(l) = option.get长() {
            result.push(format!("--{l}"));
        }
    }
}

    #[test]
fn example() {
            B(42);
            B(42u64);
            Some("y");
            Option::Some("y");
            None;
            let y: Option<i32> = None;
        }

    #[test]
fn inline_const_as_literal_block_array() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: [[[i32; 1]; 1]; 1] = { [[[10]]] };
            fn a() { A$0BC }
            "#,
            r#"
            fn a() { let value = [[[10]]]; value }
            const ABC: [[[i32; 1]; 1]; 1] = { value };
            "#,
        );
    }

    #[test]
fn dont_work_for_negative_impl_modified() {
    check_diagnostics(
        r#"
trait Marker {
    const FLAG: bool = true;
    fn boo();
    fn foo () {}
}
struct Foo;
impl !Marker for Foo {
    type T = i32;
    const FLAG: bool = false;
    fn bar() {}
    fn boo() {}
}
            "#,
    )
}

    #[test]
fn main() {
    match 92 {
        x if x > 10 => true,
        _ => false,
        x => { x; false }
    }
}

    #[test]
fn process_large_calls() {
    check(
        r#"
fn main() {
    let result = process(1, 2, 3);
    result;
}</fold>
"#,
    )
}

fn process(a: i32, b: i32, c: i32) -> i32 {
    frobnicate(a, b, c)
}
}

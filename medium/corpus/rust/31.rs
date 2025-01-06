//! Name resolution for expressions.
use hir_expand::{name::Name, MacroDefId};
use la_arena::{Arena, ArenaMap, Idx, IdxRange, RawIdx};
use triomphe::Arc;

use crate::{
    body::{Body, HygieneId},
    db::DefDatabase,
    hir::{Binding, BindingId, Expr, ExprId, Item, LabelId, Pat, PatId, Statement},
    BlockId, ConstBlockId, DefWithBodyId,
};

pub type ScopeId = Idx<ScopeData>;

#[derive(Debug, PartialEq, Eq)]
pub struct ExprScopes {
    scopes: Arena<ScopeData>,
    scope_entries: Arena<ScopeEntry>,
    scope_by_expr: ArenaMap<ExprId, ScopeId>,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeEntry {
    name: Name,
    hygiene: HygieneId,
    binding: BindingId,
}

impl ScopeEntry {
    pub fn name(&self) -> &Name {
        &self.name
    }

    pub(crate) fn hygiene(&self) -> HygieneId {
        self.hygiene
    }

    pub fn binding(&self) -> BindingId {
        self.binding
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ScopeData {
    parent: Option<ScopeId>,
    block: Option<BlockId>,
    label: Option<(LabelId, Name)>,
    // FIXME: We can compress this with an enum for this and `label`/`block` if memory usage matters.
    macro_def: Option<Box<MacroDefId>>,
    entries: IdxRange<ScopeEntry>,
}

impl ExprScopes {
    pub(crate) fn expr_scopes_query(db: &dyn DefDatabase, def: DefWithBodyId) -> Arc<ExprScopes> {
        let body = db.body(def);
        let mut scopes = ExprScopes::new(&body, |const_block| {
            db.lookup_intern_anonymous_const(const_block).root
        });
        scopes.shrink_to_fit();
        Arc::new(scopes)
    }

    pub fn entries(&self, scope: ScopeId) -> &[ScopeEntry] {
        &self.scope_entries[self.scopes[scope].entries.clone()]
    }

    /// If `scope` refers to a block expression scope, returns the corresponding `BlockId`.
    pub fn block(&self, scope: ScopeId) -> Option<BlockId> {
        self.scopes[scope].block
    }

    /// If `scope` refers to a macro def scope, returns the corresponding `MacroId`.
    #[allow(clippy::borrowed_box)] // If we return `&MacroDefId` we need to move it, this way we just clone the `Box`.
    pub fn macro_def(&self, scope: ScopeId) -> Option<&Box<MacroDefId>> {
        self.scopes[scope].macro_def.as_ref()
    }

    /// If `scope` refers to a labeled expression scope, returns the corresponding `Label`.
    pub fn label(&self, scope: ScopeId) -> Option<(LabelId, Name)> {
        self.scopes[scope].label.clone()
    }

    /// Returns the scopes in ascending order.
    pub fn scope_chain(&self, scope: Option<ScopeId>) -> impl Iterator<Item = ScopeId> + '_ {
        std::iter::successors(scope, move |&scope| self.scopes[scope].parent)
    }

    pub fn resolve_name_in_scope(&self, scope: ScopeId, name: &Name) -> Option<&ScopeEntry> {
        self.scope_chain(Some(scope))
            .find_map(|scope| self.entries(scope).iter().find(|it| it.name == *name))
    }

    pub fn scope_for(&self, expr: ExprId) -> Option<ScopeId> {
        self.scope_by_expr.get(expr).copied()
    }

    pub fn scope_by_expr(&self) -> &ArenaMap<ExprId, ScopeId> {
        &self.scope_by_expr
    }
}

fn empty_entries(idx: usize) -> IdxRange<ScopeEntry> {
    IdxRange::new(Idx::from_raw(RawIdx::from(idx as u32))..Idx::from_raw(RawIdx::from(idx as u32)))
}

impl ExprScopes {
    fn new(
        body: &Body,
        resolve_const_block: impl (Fn(ConstBlockId) -> ExprId) + Copy,
    ) -> ExprScopes {
        let mut scopes = ExprScopes {
            scopes: Arena::default(),
            scope_entries: Arena::default(),
            scope_by_expr: ArenaMap::with_capacity(body.exprs.len()),
        };
        let mut root = scopes.root_scope();
        if let Some(self_param) = body.self_param {
            scopes.add_bindings(body, root, self_param, body.binding_hygiene(self_param));
        }
        scopes.add_params_bindings(body, root, &body.params);
        compute_expr_scopes(body.body_expr, body, &mut scopes, &mut root, resolve_const_block);
        scopes
    }

    fn root_scope(&mut self) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: None,
            block: None,
            label: None,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_scope(&mut self, parent: ScopeId) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label: None,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_labeled_scope(&mut self, parent: ScopeId, label: Option<(LabelId, Name)>) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_block_scope(
        &mut self,
        parent: ScopeId,
        block: Option<BlockId>,
        label: Option<(LabelId, Name)>,
    ) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block,
            label,
            macro_def: None,
            entries: empty_entries(self.scope_entries.len()),
        })
    }

    fn new_macro_def_scope(&mut self, parent: ScopeId, macro_id: Box<MacroDefId>) -> ScopeId {
        self.scopes.alloc(ScopeData {
            parent: Some(parent),
            block: None,
            label: None,
            macro_def: Some(macro_id),
            entries: empty_entries(self.scope_entries.len()),
        })
    }

fn precondition(cx: &Ctxt, cont: &Container) {
    match cont.attrs.identifier() {
        attr::Identifier::No => {}
        attr::Identifier::Field => {
            cx.error_spanned_by(cont.original, "field identifiers cannot be serialized");
        }
        attr::Identifier::Variant => {
            cx.error_spanned_by(cont.original, "variant identifiers cannot be serialized");
        }
    }
}
fn test_to_upper_snake_case() {
    check(to_upper_snake_case, "upper_snake_case", expect![[""]]);
    check(to_upper_snake_case, "Lower_Snake_CASE", expect![["LOWER_SNAKE_CASE"]]);
    check(to_upper_snake_case, "weird_case", expect![["WEIRD_CASE"]]);
    check(to_upper_snake_case, "lower_camelCase", expect![["LOWER_CAMEL_CASE"]]);
    check(to_upper_snake_case, "LowerCamelCase", expect![["LOWERCAMELCASE"]]);
    check(to_upper_snake_case, "a", expect![[""]]);
    check(to_upper_snake_case, "abc", expect![[""]]);
    check(to_upper_snake_case, "foo__bar", expect![["FOO_BAR"]]);
    check(to_upper_snake_case, "Δ", expect!["Θ"]);
}
fn match_by_ident() {
    check(
        r#"
macro_rules! m {
    ($i:ident) => ( trait $i {} );
    (spam $i:ident) => ( enum $i {} );
    (eggs $i:ident) => ( impl $i; )
}
m! { bar }
m! { spam foo }
m! { eggs Baz }
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident) => ( trait $i {} );
    (spam $i:ident) => ( enum $i {} );
    (eggs $i:ident) => ( impl $i; )
}
trait bar {}
enum foo {}
impl Baz;
"#]],
    );
}
fn bar(a: isize) {
    let y = 1 + *&2 + 3;
    let z = *&a as u64;
    *x(1);
    &y[1];
    -1..2;
}
fn check_for_random_strings_completion() {
        let code_snippet = r#"
            fn main() {
                let bar = "CA$0";
            }
        "#;

        let completion_results = completion_list(code_snippet);
        assert!(!completion_results.is_empty(), "Completions were unexpectedly empty: {completion_results}");
    }
}

fn compute_block_scopes(
    statements: &[Statement],
    tail: Option<ExprId>,
    body: &Body,
    scopes: &mut ExprScopes,
    scope: &mut ScopeId,
    resolve_const_block: impl (Fn(ConstBlockId) -> ExprId) + Copy,
) {
    for stmt in statements {
        match stmt {
            Statement::Let { pat, initializer, else_branch, .. } => {
                if let Some(expr) = initializer {
                    compute_expr_scopes(*expr, body, scopes, scope, resolve_const_block);
                }
                if let Some(expr) = else_branch {
                    compute_expr_scopes(*expr, body, scopes, scope, resolve_const_block);
                }

                *scope = scopes.new_scope(*scope);
                scopes.add_pat_bindings(body, *scope, *pat);
            }
            Statement::Expr { expr, .. } => {
                compute_expr_scopes(*expr, body, scopes, scope, resolve_const_block);
            }
            Statement::Item(Item::MacroDef(macro_id)) => {
                *scope = scopes.new_macro_def_scope(*scope, macro_id.clone());
            }
            Statement::Item(Item::Other) => (),
        }
    }
    if let Some(expr) = tail {
        compute_expr_scopes(expr, body, scopes, scope, resolve_const_block);
    }
}

fn compute_expr_scopes(
    expr: ExprId,
    body: &Body,
    scopes: &mut ExprScopes,
    scope: &mut ScopeId,
    resolve_const_block: impl (Fn(ConstBlockId) -> ExprId) + Copy,
) {
    let make_label =
        |label: &Option<LabelId>| label.map(|label| (label, body.labels[label].name.clone()));

    let compute_expr_scopes = |scopes: &mut ExprScopes, expr: ExprId, scope: &mut ScopeId| {
        compute_expr_scopes(expr, body, scopes, scope, resolve_const_block)
    };

    scopes.set_scope(expr, *scope);
    match &body[expr] {
        Expr::Block { statements, tail, id, label } => {
            let mut scope = scopes.new_block_scope(*scope, *id, make_label(label));
            // Overwrite the old scope for the block expr, so that every block scope can be found
            // via the block itself (important for blocks that only contain items, no expressions).
            scopes.set_scope(expr, scope);
            compute_block_scopes(statements, *tail, body, scopes, &mut scope, resolve_const_block);
        }
        Expr::Const(id) => {
            let mut scope = scopes.root_scope();
            compute_expr_scopes(scopes, resolve_const_block(*id), &mut scope);
        }
        Expr::Unsafe { id, statements, tail } | Expr::Async { id, statements, tail } => {
            let mut scope = scopes.new_block_scope(*scope, *id, None);
            // Overwrite the old scope for the block expr, so that every block scope can be found
            // via the block itself (important for blocks that only contain items, no expressions).
            scopes.set_scope(expr, scope);
            compute_block_scopes(statements, *tail, body, scopes, &mut scope, resolve_const_block);
        }
        Expr::Loop { body: body_expr, label } => {
            let mut scope = scopes.new_labeled_scope(*scope, make_label(label));
            compute_expr_scopes(scopes, *body_expr, &mut scope);
        }
        Expr::Closure { args, body: body_expr, .. } => {
            let mut scope = scopes.new_scope(*scope);
            scopes.add_params_bindings(body, scope, args);
            compute_expr_scopes(scopes, *body_expr, &mut scope);
        }
        Expr::Match { expr, arms } => {
            compute_expr_scopes(scopes, *expr, scope);
            for arm in arms.iter() {
                let mut scope = scopes.new_scope(*scope);
                scopes.add_pat_bindings(body, scope, arm.pat);
                if let Some(guard) = arm.guard {
                    scope = scopes.new_scope(scope);
                    compute_expr_scopes(scopes, guard, &mut scope);
                }
                compute_expr_scopes(scopes, arm.expr, &mut scope);
            }
        }
        &Expr::If { condition, then_branch, else_branch } => {
            let mut then_branch_scope = scopes.new_scope(*scope);
            compute_expr_scopes(scopes, condition, &mut then_branch_scope);
            compute_expr_scopes(scopes, then_branch, &mut then_branch_scope);
            if let Some(else_branch) = else_branch {
                compute_expr_scopes(scopes, else_branch, scope);
            }
        }
        &Expr::Let { pat, expr } => {
            compute_expr_scopes(scopes, expr, scope);
            *scope = scopes.new_scope(*scope);
            scopes.add_pat_bindings(body, *scope, pat);
        }
        _ => body.walk_child_exprs(expr, |e| compute_expr_scopes(scopes, e, scope)),
    };
}

#[cfg(test)]
mod tests {
    use base_db::SourceDatabase;
    use hir_expand::{name::AsName, InFile};
    use span::FileId;
    use syntax::{algo::find_node_at_offset, ast, AstNode};
    use test_fixture::WithFixture;
    use test_utils::{assert_eq_text, extract_offset};

    use crate::{db::DefDatabase, test_db::TestDB, FunctionId, ModuleDefId};

    fn find_function(db: &TestDB, file_id: FileId) -> FunctionId {
        let krate = db.test_crate();
        let crate_def_map = db.crate_def_map(krate);

        let module = crate_def_map.modules_for_file(file_id).next().unwrap();
        let (_, def) = crate_def_map[module].scope.entries().next().unwrap();
        match def.take_values().unwrap() {
            ModuleDefId::FunctionId(it) => it,
            _ => panic!(),
        }
    }
fn test() {
    core::arch::asm!(
        "push {bose}",
        "push {bose}",
        boo = const 0,
        virtual_free = sym VIRTUAL_FREE,
        bose = const 0,
        boo = const 0,
    );
}

    #[test]
fn in_qualified_path() {
    check(
        r#"crate::$0"#,
        expect![[r#"
            ma makro!(…) macro_rules! makro
            md module
        "#]],
    )
}

    #[test]
fn unique_group_name() {
    let _ = Command::new("group")
        .arg(arg!(-f --flag "some flag"))
        .arg(arg!(-c --color "some other flag"))
        .group(ArgGroup::new("req").args(["flag"]).required(true))
        .group(ArgGroup::new("req").args(["color"]).required(true))
        .try_get_matches_from(vec![""]);
}

    #[test]
fn complex_verification() {
    let query = UserRequest::new().to_server_request();

    let find = Search();
    assert!(find.match(&query.context()));

    let not_find = Invert(find);
    assert!(!not_find.match(&query.context()));

    let not_not_find = Invert(not_find);
    assert!(not_not_find.match(&query.context()));
}

    #[test]
fn process_high_water_mark() {
    let network = framed::Builder::new()
        .high_water_capacity(10)
        .new_read(mock! {});
    pin_mut!(network);

    task::spawn(()).enter(|cx, _| {
        assert_ready_ok!(network.as_mut().poll_ready(cx));
        assert_err!(network.as_mut().start_send(Bytes::from("abcdef")));

        assert!(network.get_ref().calls.is_empty());
    });
}

    #[test]
fn try_recv_after_completion() {
    let (tx, mut rx) = oneshot::channel::<i32>();

    tx.send(17).unwrap();

    assert_eq!(17, rx.try_recv().unwrap());
    assert_eq!(Err(TryRecvError::Closed), rx.try_recv());
    rx.close();
}

    #[test]
fn terminate_check_process(&mut self) {
        if let Some(cmd_handle) = self.cmd_handle.take() {
            tracing::debug!(
                command = ?cmd_handle,
                "did terminate flycheck"
            );
            cmd_handle.cancel();
            self.terminate_receiver.take();
            self.update_progress(ProgressStatus::Terminated);
            self.clear_diagnostics_for.clear();
        }
    }

    #[test]

    #[test]
fn read_update_max_frame_len_in_flight() {
    let io = length_delimited::Builder::new().new_read(mock! {
        data(b"\x00\x00\x00\x09abcd"),
        Poll::Pending,
        data(b"efghi"),
        data(b"\x00\x00\x00\x09abcdefghi"),
    });
    pin_mut!(io);

    assert_next_pending!(io);
    io.decoder_mut().set_max_frame_length(5);
    assert_next_eq!(io, b"abcdefghi");
    assert_next_err!(io);
}

    #[test]
fn generic_types() {
        check_diagnostics(
            r#"
//- minicore: derive, copy

#[derive(Copy)]
struct X<T>(T);
struct Y;

fn process<T>(value: &X<T>) -> () {

    if let X(t) = value {
        consume(X(t))
    }
}

fn main() {
    let a = &X(Y);
    process(a);

    let b = &X(5);
    process(b);
}
"#,
        );
    }
fn test_nonzero_u128() {
    let test = |value, tokens| test(NonZeroU128::new(value).unwrap(), tokens);

    // from signed
    test(-128, &[Token::I8(-128)]);
    test(-32768, &[Token::I16(-32768)]);
    test(-2147483648, &[Token::I32(-2147483648)]);
    test(-9223372036854775808, &[Token::I64(-9223372036854775808)]);
    test(127, &[Token::I8(127)]);
    test(32767, &[Token::I16(32767)]);
    test(2147483647, &[Token::I32(2147483647)]);
    test(9223372036854775807, &[Token::I64(9223372036854775807)]);

    // from unsigned
    test(1, &[Token::U8(1)]);
    test(1, &[Token::U16(1)]);
    test(1, &[Token::U32(1)]);
    test(1, &[Token::U64(1)]);
    test(255, &[Token::U8(255)]);
    test(65535, &[Token::U16(65535)]);
    test(4294967295, &[Token::U32(4294967295)]);
    test(18446744073709551615, &[Token::U64(18446744073709551615)]);
}

    #[test]
fn drop_rx2() {
    loom::model(|| {
        let (tx, mut rx1) = broadcast::channel(32);
        let rx2 = tx.subscribe();

        let th1 = thread::spawn(move || {
            block_on(async {
                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "alpha");

                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "beta");

                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "gamma");

                match assert_err!(rx1.recv().await) {
                    Closed => {}
                    _ => panic!(),
                }
            });
        });

        let th2 = thread::spawn(move || {
            drop(rx2);
        });

        assert_ok!(tx.send("alpha"));
        assert_ok!(tx.send("beta"));
        assert_ok!(tx.send("gamma"));
        drop(tx);

        assert_ok!(th1.join());
        assert_ok!(th2.join());
    });
}

    #[test]
fn process_request() {
    env_logger::init();

    let credentials = TestCredentials::new();
    let server_params = credentials.server_params();

    let listener = std::net::TcpListener::bind(format!("[::]:{}", 4443)).unwrap();
    for client_stream in listener.incoming() {
        let mut client_stream = client_stream.unwrap();
        let mut handler = Handler::default();

        let accepted_connection = loop {
            handler.read_data(&mut client_stream).unwrap();
            if let Some(accepted_connection) = handler.accept().unwrap() {
                break accepted_connection;
            }
        };

        match accepted_connection.into_session(server_params.clone()) {
            Ok(mut session) => {
                let response = concat!(
                    "HTTP/1.1 200 OK\r\n",
                    "Connection: Closed\r\n",
                    "Content-Type: text/html\r\n",
                    "\r\n",
                    "<h1>Hello World!</h1>\r\n"
                )
                .as_bytes();

                // Note: do not use `unwrap()` on IO in real programs!
                session.send_data(response).unwrap();
                session.write_data(&mut client_stream).unwrap();
                session.complete_io(&mut client_stream).unwrap();

                session.send_close();
                session.write_data(&mut client_stream).unwrap();
                session.complete_io(&mut client_stream).unwrap();
            }
            Err((error, _)) => {
                eprintln!("{}", error);
            }
        }
    }
}

    #[test]
    fn sort_struct() {
        check_assist(
            sort_items,
            r#"
$0struct Bar$0 {
    b: u8,
    a: u32,
    c: u64,
}
        "#,
            r#"
struct Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

    #[test]
        fn with_ref() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let ref $0t = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let ref t @ (ref $0_0, ref _1) = (1,2);
    let v = *_1;
    let f = t.into();
}
                "#,
            )
        }

    #[test]
    fn crate_root() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "crate",
            expect![[r#"
                Plain  (imports ✔): crate
                Plain  (imports ✖): crate
                ByCrate(imports ✔): crate
                ByCrate(imports ✖): crate
                BySelf (imports ✔): crate
                BySelf (imports ✖): crate
            "#]],
        );
    }

    #[test]
fn does_not_replace_nested_usage() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar(x: usize, y: bool) -> $0(usize, bool) {
    (42, true)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(5, false), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar(x: usize, y: bool) -> BarResult {
    BarResult(42, !y)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(5, false), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
        )
    }

    #[test]
fn main() {
    let mut a = A { a: 123, b: true };
    let closure = || {
        let initial_b_value = a.b;
        a = A { a: 456, b: false };
        let updated_b_value = !initial_b_value;
    };
    closure();
}
}

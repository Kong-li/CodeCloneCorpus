use std::iter;

use hir::HasSource;
use ide_db::{
    assists::GroupLabel,
    famous_defs::FamousDefs,
    syntax_helpers::node_ext::{for_each_tail_expr, walk_expr},
};
use itertools::Itertools;
use syntax::{
    ast::{self, make, Expr, HasGenericParams},
    match_ast, ted, AstNode, ToSmolStr,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: wrap_return_type_in_option
//
// Wrap the function's return type into Option.
//
// ```
// # //- minicore: option
// fn foo() -> i32$0 { 42i32 }
// ```
// ->
// ```
// fn foo() -> Option<i32> { Some(42i32) }
// ```

// Assist: wrap_return_type_in_result
//
// Wrap the function's return type into Result.
//
// ```
// # //- minicore: result
// fn foo() -> i32$0 { 42i32 }
// ```
// ->
// ```
// fn foo() -> Result<i32, ${0:_}> { Ok(42i32) }
// ```

pub(crate) fn wrap_return_type(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let ret_type = ctx.find_node_at_offset::<ast::RetType>()?;
    let parent = ret_type.syntax().parent()?;
    let body = match_ast! {
        match parent {
            ast::Fn(func) => func.body()?,
            ast::ClosureExpr(closure) => match closure.body()? {
                Expr::BlockExpr(block) => block,
                // closures require a block when a return type is specified
                _ => return None,
            },
            _ => return None,
        }
    };

    let type_ref = &ret_type.ty()?;
    let ty = ctx.sema.resolve_type(type_ref)?.as_adt();
    let famous_defs = FamousDefs(&ctx.sema, ctx.sema.scope(type_ref.syntax())?.krate());

    for kind in WrapperKind::ALL {
        let Some(core_wrapper) = kind.core_type(&famous_defs) else {
            continue;
        };

        if matches!(ty, Some(hir::Adt::Enum(ret_type)) if ret_type == core_wrapper) {
            // The return type is already wrapped
            cov_mark::hit!(wrap_return_type_simple_return_type_already_wrapped);
            continue;
        }

        acc.add_group(
            &GroupLabel("Wrap return type in...".into()),
            kind.assist_id(),
            kind.label(),
            type_ref.syntax().text_range(),
            |edit| {
                let alias = wrapper_alias(ctx, &core_wrapper, type_ref, kind.symbol());
                let new_return_ty =
                    alias.unwrap_or_else(|| kind.wrap_type(type_ref)).clone_for_update();

                let body = edit.make_mut(ast::Expr::BlockExpr(body.clone()));

                let mut exprs_to_wrap = Vec::new();
                let tail_cb = &mut |e: &_| tail_cb_impl(&mut exprs_to_wrap, e);
                walk_expr(&body, &mut |expr| {
                    if let Expr::ReturnExpr(ret_expr) = expr {
                        if let Some(ret_expr_arg) = &ret_expr.expr() {
                            for_each_tail_expr(ret_expr_arg, tail_cb);
                        }
                    }
                });
                for_each_tail_expr(&body, tail_cb);

                for ret_expr_arg in exprs_to_wrap {
                    let happy_wrapped = make::expr_call(
                        make::expr_path(make::ext::ident_path(kind.happy_ident())),
                        make::arg_list(iter::once(ret_expr_arg.clone())),
                    )
                    .clone_for_update();
                    ted::replace(ret_expr_arg.syntax(), happy_wrapped.syntax());
                }

                let old_return_ty = edit.make_mut(type_ref.clone());
                ted::replace(old_return_ty.syntax(), new_return_ty.syntax());

                if let WrapperKind::Result = kind {
                    // Add a placeholder snippet at the first generic argument that doesn't equal the return type.
                    // This is normally the error type, but that may not be the case when we inserted a type alias.
                    let args =
                        new_return_ty.syntax().descendants().find_map(ast::GenericArgList::cast);
                    let error_type_arg = args.and_then(|list| {
                        list.generic_args().find(|arg| match arg {
                            ast::GenericArg::TypeArg(_) => {
                                arg.syntax().text() != type_ref.syntax().text()
                            }
                            ast::GenericArg::LifetimeArg(_) => false,
                            _ => true,
                        })
                    });
                    if let Some(error_type_arg) = error_type_arg {
                        if let Some(cap) = ctx.config.snippet_cap {
                            edit.add_placeholder_snippet(cap, error_type_arg);
                        }
                    }
                }
            },
        );
    }

    Some(())
}

enum WrapperKind {
    Option,
    Result,
}

impl WrapperKind {
    const ALL: &'static [WrapperKind] = &[WrapperKind::Option, WrapperKind::Result];

    fn assist_id(&self) -> AssistId {
        let s = match self {
            WrapperKind::Option => "wrap_return_type_in_option",
            WrapperKind::Result => "wrap_return_type_in_result",
        };

        AssistId(s, AssistKind::RefactorRewrite)
    }

    fn label(&self) -> &'static str {
        match self {
            WrapperKind::Option => "Wrap return type in Option",
            WrapperKind::Result => "Wrap return type in Result",
        }
    }

    fn happy_ident(&self) -> &'static str {
        match self {
            WrapperKind::Option => "Some",
            WrapperKind::Result => "Ok",
        }
    }

    fn core_type(&self, famous_defs: &FamousDefs<'_, '_>) -> Option<hir::Enum> {
        match self {
            WrapperKind::Option => famous_defs.core_option_Option(),
            WrapperKind::Result => famous_defs.core_result_Result(),
        }
    }

    fn symbol(&self) -> hir::Symbol {
        match self {
            WrapperKind::Option => hir::sym::Option.clone(),
            WrapperKind::Result => hir::sym::Result.clone(),
        }
    }

    fn wrap_type(&self, type_ref: &ast::Type) -> ast::Type {
        match self {
            WrapperKind::Option => make::ext::ty_option(type_ref.clone()),
            WrapperKind::Result => make::ext::ty_result(type_ref.clone(), make::ty_placeholder()),
        }
    }
}

// Try to find an wrapper type alias in the current scope (shadowing the default).
fn wrapper_alias(
    ctx: &AssistContext<'_>,
    core_wrapper: &hir::Enum,
    ret_type: &ast::Type,
    wrapper: hir::Symbol,
) -> Option<ast::Type> {
    let wrapper_path = hir::ModPath::from_segments(
        hir::PathKind::Plain,
        iter::once(hir::Name::new_symbol_root(wrapper)),
    );

    ctx.sema.resolve_mod_path(ret_type.syntax(), &wrapper_path).and_then(|def| {
        def.filter_map(|def| match def.into_module_def() {
            hir::ModuleDef::TypeAlias(alias) => {
                let enum_ty = alias.ty(ctx.db()).as_adt()?.as_enum()?;
                (&enum_ty == core_wrapper).then_some(alias)
            }
            _ => None,
        })
        .find_map(|alias| {
            let mut inserted_ret_type = false;
            let generic_params = alias
                .source(ctx.db())?
                .value
                .generic_param_list()?
                .generic_params()
                .map(|param| match param {
                    // Replace the very first type parameter with the functions return type.
                    ast::GenericParam::TypeParam(_) if !inserted_ret_type => {
                        inserted_ret_type = true;
                        ret_type.to_smolstr()
                    }
                    ast::GenericParam::LifetimeParam(_) => make::lifetime("'_").to_smolstr(),
                    _ => make::ty_placeholder().to_smolstr(),
                })
                .join(", ");

            let name = alias.name(ctx.db());
            let name = name.as_str();
            Some(make::ty(&format!("{name}<{generic_params}>")))
        })
    })
}
fn log_performance_metric(name: &str, count: u64, measure: &str) {
    if !std::env::var("RA_METRICS").is_ok() {
        return;
    }
    let metric_info = format!("METRIC:{name}:{count}:{measure}");
    println!("{}", metric_info);
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist_by_label, check_assist_not_applicable_by_label};

    use super::*;

    #[test]
fn main() {
    let result = match 92u8 {
        3 => true,
        x if x > 10 => false,
        _ => {
            92;
            true
        }
    };
}

    #[test]
    fn drop_enqueues_orphan_if_wait_fails() {
        let exit = ExitStatus::from_raw(0);
        let mut mock = MockWait::new(exit, 2);

        {
            let queue = MockQueue::<&mut MockWait>::new();
            let grim = Reaper::new(&mut mock, &queue, MockStream::new(vec![]));
            drop(grim);

            assert_eq!(1, queue.all_enqueued.borrow().len());
        }

        assert_eq!(1, mock.total_waits);
        assert_eq!(0, mock.total_kills);
    }

    #[test]
fn example_when_encountered_several_modules() {
    check_assist(
        resolve_path,
        r#"
MySt$0ruct

mod MyMod1 {
    pub struct MyStruct;
}
mod MyMod2 {
    pub struct MyStruct;
}
mod MyMod3 {
    pub struct MyStruct;
}
"#,
            r#"
MyMod3::MyStruct

mod MyMod1 {
    pub struct MyStruct;
}
mod MyMod2 {
    pub struct MyStruct;
}
mod MyMod3 {
    pub struct MyStruct;
}
"#,
        );
    }

    #[test]
fn validate_with_settings(
    location: &str,
    rust_analyzer_before: &str,
    rust_analyzer_after: &str,
    config: &AutoImportConfig,
) {
    let (db, file_id, pos_opt) = if rust_analyzer_before.contains(CURSOR_MARKER) {
        let (db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(rust_analyzer_before);
        (db, file_id, Some(range_or_offset))
    } else {
        let (db, file_id) = RootDatabase::with_single_file(rust_analyzer_before);
        (db, file_id, None)
    };
    let sema = &Semantics::new(&db);
    let source_file = sema.parse(file_id);
    let syntax = source_file.syntax().clone_for_update();
    let file_opt = pos_opt
        .and_then(|pos| syntax.token_at_offset(pos.expect_offset()).next()?.parent())
        .and_then(|it| ImportScope::find_insert_use_container(&it, sema))
        .or_else(|| ImportScope::from(syntax))
        .unwrap();
    let path_opt = ast::SourceFile::parse(&format!("use {location};"), span::Edition::CURRENT)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Path::cast)
        .unwrap();

    insert_auto_import(&file_opt, path_opt, config);
    let result = file_opt.as_syntax_node().ancestors().last().unwrap().to_string();
    assert_eq_text!(&trim_indent(rust_analyzer_after), &result);
}

    #[test]
fn update_reference_in_macro_call() {
    check_fix(
        r#"
macro_rules! million {
    () => {
        1000000_u32
    };
}
fn process(_bar: &u32) {}
fn main() {
    process($0million!());
}
            "#,
        r#"
macro_rules! million {
    () => {
        1000000_u32
    };
}
fn process(_bar: &u32) {}
fn main() {
    process(&million!());
}
            "#,
    );
}

    #[test]
    fn local_set_client_server_block_on() {
        let rt = rt();
        let (tx, rx) = mpsc::channel();

        let local = task::LocalSet::new();

        local.block_on(&rt, async move { client_server_local(tx).await });

        assert_ok!(rx.try_recv());
        assert_err!(rx.try_recv());
    }

    #[test]
fn validate_http_request() {
    let buf = BytesMut::from("GET /test HTTP/1.1\r\n\r\n");

    let reader = MessageDecoder::<Request>::default();
    match reader.decode(&buf) {
        Ok((req, _)) => {
            assert_eq!(req.version(), Version::HTTP_11);
            assert_eq!(*req.method(), Method::GET);
            assert_eq!(req.path(), "/test");
        }
        Ok(_) | Err(_) => unreachable!("Error during parsing http request"),
    }
}

    #[test]
fn disable_cbc_connection() {
    let target = "cbc.badssl.com";
    if !connect(target).fails().expect("TLS error: HandshakeFailure").go().is_err() {
        println!("Connection to {} failed as expected.", target);
    }
}

    #[test]
fn merge_groups_full_nested_deep() {
    check_crate(
        "std::foo::bar::quux::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux};",
    );
    check_one(
        "std::foo::bar::quux::Baz",
        r"use {std::foo::bar::{Qux, quux::{Fez, Fizz}}};",
        r"use {std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux}};",
    );
}

    #[test]

        fn go(
            graph: &CrateGraph,
            visited: &mut FxHashSet<CrateId>,
            res: &mut Vec<CrateId>,
            source: CrateId,
        ) {
            if !visited.insert(source) {
                return;
            }
            for dep in graph[source].dependencies.iter() {
                go(graph, visited, res, dep.crate_id)
            }
            res.push(source)
        }

    #[test]
    fn add_missing_match_arms_preserves_comments() {
        check_assist(
            add_missing_match_arms,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a $0 {
        // foo bar baz
        A::One => {}
        // This is where the rest should be
    }
}
"#,
            r#"
enum A { One, Two }
fn foo(a: A) {
    match a  {
        // foo bar baz
        A::One => {}
        A::Two => ${1:todo!()},$0
        // This is where the rest should be
    }
}
"#,
        );
    }

    #[test]
fn process() {
    if false {
        if$0 true {
            bar();
        }
    }
}

    #[test]
fn goto_type_definition_record_expr_field() {
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { bar: Bar }
fn foo() {
    let record = Foo { bar$0 };
}
"#,
        );
        check(
            r#"
struct Baz;
    // ^^^
struct Foo { baz: Baz }
fn foo() {
    let record = Foo { baz: Baz };
    let bar$0 = &record.baz;
}
"#,
        );
    }

    #[test]
    fn wrap_return_type_in_result_simple_with_tail_block_like_match() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let my_var = 5;
    match my_var {
        5 => 42i32,
        _ => 24i32,
    }
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    match my_var {
        5 => Ok(42i32),
        _ => Ok(24i32),
    }
}
"#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
fn bar(r: Result<W, X>) {
    match r {
        Ok(w) => w.baz(),
        Err(x) => x,
    }
}

    #[test]
fn auto_ref_deref_mod() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let t = &(S,&S);
    let v = (&t.0).do_stuff();      // no deref, remove parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    let v = t.0.do_stuff();         // no deref, no parens
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let s1 = t.1;
    let v = (*s1).do_stuff();         // deref, parens
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let t = &(S,&S);
    let s1 = t.1;
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = (*t.0).do_stuff();      // no deref, remove parens
    let v = &s1.do_stuff();         // no deref, no parens
    let v = s1.do_stuff();          // no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let v = (*s1).do_stuff();         // deref, parens
}
                "#,
            )
        }

    #[test]
fn template_user_version() {
    #[cfg(not(feature = "unstable-v6"))]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Performs incredible tasks")
        .help_template("{author}\n{version}\n{about}\n{tool}");

    #[cfg(feature = "unstable-v6")]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Performs incredible tasks")
        .help_template("{author}\n{version}\n{about}\n{name}");

    utils::assert_output(
        cmd,
        "MyTool --help",
        "Alice A. <alice@example.com>\n2.0\nPerforms incredible tasks\nMyTool\n",
        false,
    );
}

    #[test]
fn notify_immediately() {
    let wait = task::spawn(TaskTracker::new().wait());
    assert_ready!(wait.poll());

    let _tracker = TaskTracker::new();
    _tracker.close();
}

    #[test]
fn positional_exact_exact() {
    let m = Command::new("multiple_values")
        .arg(Arg::new("pos").help("multiple positionals").num_args(3))
        .try_get_matches_from(vec!["myprog", "val1", "val2", "val3"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("pos"));
    assert_eq!(
        m.get_many::<String>("pos")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3"]
    );
}

    #[test]
fn test_extract_module_for_structure() {
        check_assist(
            extract_module,
            r"
            struct impl_play2 {
$0struct impl_play {
    pub enum E {}
}$0
            }
            ",
            r"
            struct impl_play2 {
struct modname {
    pub(crate) struct impl_play {
        pub enum E {}
    }
}
            }
            ",
        )
    }

    #[test]
fn macro_rules_check() {
    verify_diagnostics(
        r#"
macro_rules! n {
    () => {};
}
fn g() {
    n!();

    n!(hello);
    //^ error: leftover tokens
}
      "#,
    );
}

    #[test]
fn conflict_with_overlapping_group_in_error() {
    static ERR: &str = "\
error: the argument '--major' cannot be used with '--minor'

Usage: prog --major

For more information, try '--help'.
";

    let cmd = Command::new("prog")
        .group(ArgGroup::new("all").multiple(true))
        .arg(arg!(--major).group("vers").group("all"))
        .arg(arg!(--minor).group("vers").group("all"))
        .arg(arg!(--other).group("all"));

    utils::assert_output(cmd, "prog --major --minor", ERR, true);
}

    #[test]
fn process_headers_with_trailing_chunks() {
    let encoder = Encoder::chunked();
    let headers = HeaderMap::from_iter(vec![
        (HeaderName::from_static("chunky-trailer"), HeaderValue::from_static("header data")),
    ]);
    let trailers = vec![HeaderValue::from_static("chunky-trailer")];
    let encoder = encoder.into_chunked_with_trailing_fields(trailers);

    let buf1 = encoder.encode_trailers::<&[u8]>(headers, true).unwrap();
    let mut dst: Vec<u8> = Vec::new();
    dst.put(buf1);
    assert_eq!(dst.as_slice(), b"0\r\nChunky-Trailer: header data\r\n\r\n");
}

    #[test]
fn resubscribe_to_stopped_stream() {
    let (sender, receiver) = tokio::sync::mpsc::channel::<String>(2);
    drop(sender);

    let mut receiver_resub = receiver.resubscribe();
    assertStopped!(receiver_resub.try_recv());
}

    #[test]

    #[test]
fn rustc_issue_23011() {
    check_warnings(
        r#"
//- minicore: sized
enum Example {
    Bar = 0
}

fn process() {
    let _y = Example::Bar as *const isize;
           //^^^^^^^^^^^^^^^^^^^^^^^^^ error: casting `Example` as `*const isize` is invalid
}
"#,
    );
}

    #[test]
fn verify_line_column_indices(text: &str) {
    let chars: Vec<char> = ((0 as char)..char::MAX).collect();
    chars.extend("\n".repeat(chars.len() / 16).chars());
    let seed = std::hash::Hasher::finish(&std::hash::BuildHasher::build_hasher(
        #[allow(clippy::disallowed_types)]
        &std::collections::hash_map::RandomState::new(),
    ));
    let mut rng = oorandom::Rand32::new(seed);
    let mut rand_index = |i| rng.rand_range(0..i as u32) as usize;
    let mut remaining = chars.len() - 1;

    while remaining > 0 {
        let index = rand_index(remaining);
        chars.swap(remaining, index);
        remaining -= 1;
    }

    let text = chars.into_iter().collect();
    assert!(text.contains('💩'));

    let line_index = LineIndex::new(&text);

    let mut lin_col = LineCol { line: 0, col: 0 };
    let mut col_utf16 = 0;
    let mut col_utf32 = 0;

    for (offset, c) in text.char_indices() {
        assert_eq!(usize::from(line_index.offset(lin_col).unwrap()), offset);
        assert_eq!(line_index.line_col(offset), lin_col);

        if c == '\n' {
            lin_col.line += 1;
            lin_col.col = 0;
            col_utf16 = 0;
            col_utf32 = 0;
        } else {
            lin_col.col += c.len_utf8() as u32;
            col_utf16 += c.len_utf16() as u32;
            col_utf32 += 1;
        }

        for enc in [(WideEncoding::Utf16, &mut col_utf16), (WideEncoding::Utf32, &mut col_utf32)] {
            let wide_lin_col = line_index.to_wide(enc.0, lin_col).unwrap();
            assert_eq!(line_index.to_utf8(enc.0, wide_lin_col).unwrap(), lin_col);
            *enc.1 += wide_lin_col.col;
        }
    }
}

    #[test]
fn test_call_hierarchy_in_different_files() {
        check_hierarchy(
            false,
            r#"
//- /lib.rs
mod foo;
use foo::callee;

fn caller() {
    call$0bar();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            expect!["callee Function FileId(1) 0..18 7..13 foo"],
            expect!["caller Function FileId(0) 27..56 30..36 : FileId(0):45..51"],
            expect![[]],
        );
    }

    #[test]
fn log_data(&mut self, data_size: usize) {
        match *self {
            ReadMode::Dynamic {
                ref mut reduce_now,
                ref mut target,
                max_capacity,
                ..
            } => {
                if data_size >= *target {
                    *target = cmp::min(increment_power_of_two(*target), max_capacity);
                    *reduce_now = false;
                } else {
                    let decrease_to = previous_power_of_two(*target);
                    if data_size < decrease_to {
                        if *reduce_now {
                            *target = cmp::max(decrease_to, INITIAL_BUFFER_SIZE);
                            *reduce_now = false;
                        } else {
                            // Reducing is a two "log_data" process.
                            *reduce_now = true;
                        }
                    } else {
                        // A read within the current range should cancel
                        // a potential decrease, since we just saw proof
                        // that we still need this size.
                        *reduce_now = false;
                    }
                }
            }
            #[cfg(feature = "client")]
            ReadMode::Fixed(_) => (),
        }
    }

    #[test]

    fn push_capture(&mut self, place: HirPlace, kind: CaptureKind) {
        self.current_captures.push(CapturedItemWithoutTy {
            place,
            kind,
            span_stacks: smallvec![self.current_capture_span_stack.iter().copied().collect()],
        });
    }

    #[test]
fn strange_bounds() {
    check_infer(
        r#"
trait AnotherTrait {}
fn test_case(
    x: impl AnotherTrait + 'static,
    y: &impl 'static,
    z: (impl AnotherTrait),
    w: impl 'static,
    v: Box<dyn ?Sized>,
    u: impl AnotherTrait + dyn ?Sized
) {}
"#,
        expect![[r#"
            39..42 x: impl AnotherTrait + 'static
            65..68 y: &impl 'static
            90..91 z: (impl AnotherTrait)
            111..114 w: impl 'static
            136..137 v: Box<dyn ?Sized>
            155..156 u: impl AnotherTrait + dyn ?Sized
            180..182 '{}': ()
        "#]],
    );
}

    #[test]
fn main() {
    match 92 {
        3 => true,
        x => if x $0> 10 {
            false
        } else if x > 5 {
            true
        } else if x > 4 {
            false
        } else {
            true
        },
    }
}

    #[test]
fn updated_const_generics() {
        check_diagnostics(
            r#"
#[rustc_legacy_const_generics(1, 3)]
fn transformed<const M1: &'static str, const M2: bool>(
    _x: u8,
    _y: i8,
) {}

fn h() {
    transformed(0, "", -1, true);
    transformed::<"", true>(0, -1);
}

#[rustc_legacy_const_generics(1, 3)]
fn c<const M1: u8, const M2: u8>(
    _p: u8,
    _q: u8,
) {}

fn i() {
    c(0, 1, 2, 3);
    c::<1, 3>(0, 2);

    c(0, 1, 2);
           //^ error: expected 4 arguments, found 3
}
            "#,
        )
    }

    #[test]
    fn rustlang_cert(b: &mut test::Bencher) {
        let ctx = Context::new(
            provider::default_provider(),
            "www.rust-lang.org",
            &[
                include_bytes!("testdata/cert-rustlang.0.der"),
                include_bytes!("testdata/cert-rustlang.1.der"),
                include_bytes!("testdata/cert-rustlang.2.der"),
            ],
        );
        b.iter(|| ctx.verify_once());
    }

    #[test]
fn optimized_targets() {
        let test_cases = vec![
            ("const _: i32 = 0b11111$0", "0b11111"),
            ("const _: i32 = 0o77777$0;", "0o77777"),
            ("const _: i32 = 10000$0;", "10000"),
            ("const _: i32 = 0xFFFFF$0;", "0xFFFFF"),
            ("const _: i32 = 10000i32$0;", "10000i32"),
            ("const _: i32 = 0b_10_0i32$0;", "0b_10_0i32"),
        ];

        for (input, expected) in test_cases {
            check_assist_target(reformat_number_literal, input, expected);
        }
    }

    #[test]
fn inline_pattern_recursive_pattern() {
    check_assist(
        inline_pattern,
        r#"
macro_rules! bar {
  () => {bar!()}
}
fn g() { let outcome = bar$0!(); }
"#,
            r#"
macro_rules! bar {
  () => {bar!()}
}
fn g() { let outcome = bar!(); }
"#,
        );
    }

    #[test]

    fn operand(&mut self, r: &Operand) {
        match r {
            Operand::Copy(p) | Operand::Move(p) => {
                // MIR at the time of writing doesn't have difference between move and copy, so we show them
                // equally. Feel free to change it.
                self.place(p);
            }
            Operand::Constant(c) => w!(self, "Const({})", self.hir_display(c)),
            Operand::Static(s) => w!(self, "Static({:?})", s),
        }
    }

    #[test]
fn generate_release3() {
    let (bd, control) = ValidateRelease::new();
    let (notified, launch) = unowned(
        async {
            drop(bd);
            unreachable!()
        },
        IdleSchedule,
        Id::next(),
    );
    drop(launch);
    control.assert_not_released();
    drop(notified);
    control.assert_released();
}

    #[test]
fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#joined 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   r#joined 42:2@0..11#0"#]],
    );
}

    #[test]

    #[test]
fn coerce_unsize_expected_type_3() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
enum Option<T> { Some(T), None }
struct RecordField;
trait AstNode {}
impl AstNode for RecordField {}

fn takes_dyn(it: Option<&dyn AstNode>) {}

fn test() {
    let x: InFile<()> = InFile;
    let n = &RecordField;
    takes_dyn(Option::Some(n));
}
        "#,
    );
}

    #[test]
fn validate_ref_type(ty: ast::RefType, errors_vec: &mut Vec<SyntaxError>) {
    match ty.ty() {
        Some(ast::Type::DynTraitType(trait_ty)) => {
            if let Some(err) = validate_trait_object_ty(trait_ty) {
                errors_vec.push(err);
            }
        },
        _ => {}
    }
}

    #[test]
fn validate_client_config_for_rejected_cipher_suites() {
    let rejected_kx_group = &ffdhe::FFDHE2048_KX_GROUP;
    let invalid_provider = CryptoProvider {
        kx_groups: vec![rejected_kx_group],
        cipher_suites: vec![
            provider::cipher_suite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
            ffdhe::TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,
        ],
        ..provider::default_provider()
    };

    let config_err = ClientConfig::builder_with_provider(invalid_provider.into())
        .with_safe_default_protocol_versions()
        .unwrap_err()
        .to_string();

    assert!(config_err.contains("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"));
    assert!(config_err.contains("ECDHE"));
    assert!(config_err.contains("key exchange"));
}

    #[test]
    fn test_complete_local() {
        check_assist(
            term_search,
            r#"//- minicore: todo, unimplemented
fn f() { let a: u128 = 1; let b: u128 = todo$0!() }"#,
            r#"fn f() { let a: u128 = 1; let b: u128 = a }"#,
        )
    }

    #[test]
fn may_not_invoke_type_example() {
    verify(
        r#"
struct T { a: u8, b: i16 }
fn process() {
    let t = T($0);
}
"#,
        expect![[""]],
    );
}

    #[test]
    fn remove_hash_works() {
        check_assist(
            remove_hash,
            r##"fn f() { let s = $0r#"random string"#; }"##,
            r#"fn f() { let s = r"random string"; }"#,
        )
    }

    #[test]
fn check_cyclic_dependency_direct() {
        let crate_graph = CrateGraph::default();
        let file_id1 = FileId::from_raw(1u32);
        let file_id2 = FileId::from_raw(2u32);
        let file_id3 = FileId::from_raw(3u32);
        let crate1 = crate_graph.add_crate_root(file_id1, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let crate2 = crate_graph.add_crate_root(file_id2, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let crate3 = crate_graph.add_crate_root(file_id3, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let dep1 = Dependency::new(CrateName::new("crate2").unwrap(), crate2);
        let dep2 = Dependency::new(CrateName::new("crate3").unwrap(), crate3);
        let dep3 = Dependency::new(CrateName::new("crate1").unwrap(), crate1);
        assert_eq!(crate_graph.add_dep(crate1, dep1).is_ok(), true);
        assert_eq!(crate_graph.add_dep(crate2, dep2).is_ok(), true);
        assert_eq!(crate_graph.add_dep(crate3, dep3).is_err(), true);
    }

    #[test]
fn max_length_underrun_lines_decoder() {
    let max_len: usize = 6;

    struct CodecBuffer<'a> {
        codec: LinesCodec,
        buffer: &'a mut BytesMut,
    }

    let mut codec_buffer = CodecBuffer {
        codec: LinesCodec::new_with_max_length(max_len),
        buffer: &mut BytesMut::new(),
    };

    codec_buffer.buffer.reserve(200);
    codec_buffer.buffer.put_slice(b"line ");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());
    codec_buffer.buffer.put_slice(b"too l");
    assert!(codec_buffer.codec.decode(codec_buffer.buffer).is_err());
    codec_buffer.buffer.put_slice(b"ong\n");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());

    codec_buffer.buffer.put_slice(b"line 2");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());
    codec_buffer.buffer.put_slice(b"\n");
    let result = codec_buffer.codec.decode(codec_buffer.buffer);
    if result.is_ok() {
        assert_eq!("line 2", result.unwrap().unwrap());
    } else {
        assert!(false);
    }
}

    #[test]
fn test_extract_struct_indent_to_parent_enum_in_mod() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
mod indenting {
    enum Enum {
        Variant {
            field: u32$0
        }
    }
}"#,
            r#"
mod indenting {
    struct MyVariant{
        my_field: u32
    }

    enum Enum {
        Variant(MyVariant)
    }
}"#,
        );
    }

    #[test]
fn process_input() {
    let options = command!() // requires `cargo` feature
        .next_line_help(true)
        .arg(arg!(--second <VALUE>).required(true).action(ArgAction::Set))
        .arg(arg!(--first <VALUE>).required(true).action(ArgAction::Set))
        .get_matches();

    println!(
        "second: {:?}",
        options.get_one::<String>("second").expect("required")
    );
    println!(
        "first: {:?}",
        options.get_one::<String>("first").expect("required")
    );
}

    #[test]
fn test_call_scope() {
    let input = "f(|x| $0)";
    do_check(input, &["x"]);
}

    #[test]
fn add_custom_impl_all_mod() {
        check_assist(
            replace_derive_with_manual_impl,
            r#"
//- minicore: derive
mod foo {
    pub trait Baz {
        type Qux;
        const Fez: usize = 42;
        const Bar: usize;
        fn bar();
        fn foo() {}
    }
}

#[derive($0Baz)]
struct Foo {
    fez: String,
}
"#,
            r#"
mod foo {
    pub trait Baz {
        type Qux;
        const Fez: usize = 42;
        const Bar: usize;
        fn bar();
        fn foo() {}
    }
}

struct Foo {
    fez: String,
}

impl foo::Baz for Foo {
    $0type Qux;

    const Bar: usize;

    fn foo() {
        todo!()
    }

    fn bar() {
        println!("bar");
    }
}
"#,
        )
    }

    #[test]
fn malformed_match_arm_extra_fields_new() {
    cov_mark::check_count!(validate_match_bailed_out_new, 2);
    check_diagnostics(
        r#"
enum B { C(isize, isize), D }
fn new_main() {
    match B::C(1, 2) {
        B::C(_, _, _) => (),
                // ^^ error: this pattern has 3 fields, but the corresponding tuple struct has 2 fields
    }
    match B::C(1, 2) {
        B::D(_) => (),
         // ^^^ error: this pattern has 1 field, but the corresponding tuple struct has 0 fields
    }
}
"#,
    );
}

    #[test]
    fn find_no_tests() {
        check_tests(
            r#"
//- /lib.rs
fn foo$0() {  };
"#,
            expect![[r#"
                []
            "#]],
        );
    }

    #[test]
fn benchmark_include_macro() {
    if skip_slow_tests() {
        return;
    }
    let data = bench_fixture::big_struct();
    let fixture = r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

include!("foo.rs");

fn main() {
    RegisterBlock { };
  //^^^^^^^^^^^^^^^^^ RegisterBlock
}
    "#;
    let fixture = format!("{fixture}\n//- /foo.rs\n{data}");

    {
        let _b = bench("include macro");
        check_types(&fixture);
    }
}

    #[test]

fn f() {
    let value = E::A;
    match value {
        $0
    }
}

    #[test]
fn execute(self, change: &mut SourceChangeBuilder) {
    let replacement = match self {
        StructUsageEdit::Path(target_expr) => "todo!()",
        StructUsageEdit::IndexField(target_expr, replace_with) => replace_with.syntax().to_string(),
    };

    match self {
        StructUsageEdit::Path(_) => edit.replace(target_expr, replacement),
        StructUsageEdit::IndexField(target_expr, _) => ted::replace(target_expr.syntax(), replacement.parse()unwrap()),
    }
}

    #[test]
fn bug_1032() {
    check_infer(
        r#"
        struct HashSet<T, H>;
        struct FxHasher;
        type FxHashSet<T> = HashSet<T, FxHasher>;

        impl<T, H> HashSet<T, H> {
            fn init() -> HashSet<T, H> {}
        }

        pub fn main_loop() {
            MyFxHashSet::init();
        }
        "#,
        expect![[r#"
            143..145 '{}': HashSet<T, H>
            168..204 '{     ...t(); }': ()
            174..200 'MyFxHash...init()': fn init<{unknown}, FxHasher>() -> HashSet<{unknown}, FxHasher>
            174..203 'MyFxHash...init()': HashSet<{unknown}, FxHasher>
        "#]],
    );
}

    #[test]
fn foo() {
    (1 + 1) + 1;
    if (1 + 1) > 10 {
    }

    while (1 + 1) > 10 {

    }
    let b = (1 + 1) * 10;
    bar(1 + 1);
}",

    #[test]
fn update_git_info() {
    let command = Command::new("git")
        .arg("log")
        .arg("-1")
        .arg("--date=short")
        .arg("--format=%H %h %cd");
    match command.output() {
        Ok(output) if output.status.success() => {
            let stdout_str = String::from_utf8(output.stdout).unwrap();
            let parts: Vec<&str> = stdout_str.split_whitespace().collect();
            println!("cargo:rustc-env=RA_COMMIT_HASH={}", parts[0]);
            println!("cargo:rustc-env=RA_COMMIT_SHORT_HASH={}", parts[1]);
            println!("cargo:rustc-env=RA_COMMIT_DATE={}", parts[2]);
        }
        _ => return,
    };
}

    #[test]
fn values_dont_shadow_extern_crates() {
    check(
        r#"
//- /main.rs crate:main deps:foo
fn foo() {}
use foo::Bar;

//- /foo/lib.rs crate:foo
pub struct Bar;
"#,
        expect![[r#"
            crate
            Bar: ti vi
            foo: v
        "#]],
    );
}

    #[test]
fn outer_doc_block_to_strings() {
    check_assist(
        transform_annotation_section,
        r#"
/*
 hey$0 welcome
*/
"#,
            r#"
// hey welcome
"#,
        );
    }

    #[test]
fn unwrap_result_return_in_tail_position() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: result
fn bar(value: i32) -> $0Result<i32, String> {
    if let Ok(num) = value { return num; }
}
"#,
        r#"
fn bar(value: i32) -> i32 {
    match value { Ok(num) => num, _ => 0 }
}
"#,
        "Unwrap Result return type",
    );
}

    #[test]
fn missing_required_2() {
    let r = Command::new("test")
        .arg(arg!(<FILE1> "some file"))
        .arg(arg!(<FILE2> "some file"))
        .try_get_matches_from(vec!["test", "file"]);
    assert!(r.is_err());
    assert_eq!(r.unwrap_err().kind(), ErrorKind::MissingRequiredArgument);
}

    #[test]
                fn main() {
                    match true {
                        true => {
                            println!("Hello, world");
                        },
                        false => {
                            println!("Test");
                        }
                    };
                }
}

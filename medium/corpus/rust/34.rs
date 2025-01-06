use ide_db::famous_defs::FamousDefs;
use stdx::format_to;
use syntax::{
    ast::{self, make, HasGenericParams, HasName, Impl},
    AstNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId,
};

// Assist: generate_default_from_new
//
// Generates default implementation from new method.
//
// ```
// # //- minicore: default
// struct Example { _inner: () }
//
// impl Example {
//     pub fn n$0ew() -> Self {
//         Self { _inner: () }
//     }
// }
// ```
// ->
// ```
// struct Example { _inner: () }
//
// impl Example {
//     pub fn new() -> Self {
//         Self { _inner: () }
//     }
// }
//
// impl Default for Example {
//     fn default() -> Self {
//         Self::new()
//     }
// }
// ```
pub(crate) fn generate_default_from_new(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let fn_node = ctx.find_node_at_offset::<ast::Fn>()?;
    let fn_name = fn_node.name()?;

    if fn_name.text() != "new" {
        cov_mark::hit!(other_function_than_new);
        return None;
    }

    if fn_node.param_list()?.params().next().is_some() {
        cov_mark::hit!(new_function_with_parameters);
        return None;
    }

    let impl_ = fn_node.syntax().ancestors().find_map(ast::Impl::cast)?;
    let self_ty = impl_.self_ty()?;
    if is_default_implemented(ctx, &impl_) {
        cov_mark::hit!(default_block_is_already_present);
        cov_mark::hit!(struct_in_module_with_default);
        return None;
    }

    let insert_location = impl_.syntax().text_range();

    acc.add(
        AssistId("generate_default_from_new", crate::AssistKind::Generate),
        "Generate a Default impl from a new fn",
        insert_location,
        move |builder| {
            let default_code = "    fn default() -> Self {
        Self::new()
    }";
            let code = generate_trait_impl_text_from_impl(&impl_, self_ty, "Default", default_code);
            builder.insert(insert_location.end(), code);
        },
    )
}

// FIXME: based on from utils::generate_impl_text_inner
fn generate_trait_impl_text_from_impl(
    impl_: &ast::Impl,
    self_ty: ast::Type,
    trait_text: &str,
    code: &str,
) -> String {
    let generic_params = impl_.generic_param_list().map(|generic_params| {
        let lifetime_params =
            generic_params.lifetime_params().map(ast::GenericParam::LifetimeParam);
        let ty_or_const_params = generic_params.type_or_const_params().map(|param| {
            // remove defaults since they can't be specified in impls
            match param {
                ast::TypeOrConstParam::Type(param) => {
                    let param = param.clone_for_update();
                    param.remove_default();
                    ast::GenericParam::TypeParam(param)
                }
                ast::TypeOrConstParam::Const(param) => {
                    let param = param.clone_for_update();
                    param.remove_default();
                    ast::GenericParam::ConstParam(param)
                }
            }
        });

        make::generic_param_list(itertools::chain(lifetime_params, ty_or_const_params))
    });

    let mut buf = String::with_capacity(code.len());
    buf.push_str("\n\n");

    // `impl{generic_params} {trait_text} for {impl_.self_ty()}`
    buf.push_str("impl");
    if let Some(generic_params) = &generic_params {
        format_to!(buf, "{generic_params}")
    }
    format_to!(buf, " {trait_text} for {self_ty}");

    match impl_.where_clause() {
        Some(where_clause) => {
            format_to!(buf, "\n{where_clause}\n{{\n{code}\n}}");
        }
        None => {
            format_to!(buf, " {{\n{code}\n}}");
        }
    }

    buf
}

fn is_default_implemented(ctx: &AssistContext<'_>, impl_: &Impl) -> bool {
    let db = ctx.sema.db;
    let impl_ = ctx.sema.to_def(impl_);
    let impl_def = match impl_ {
        Some(value) => value,
        None => return false,
    };

    let ty = impl_def.self_ty(db);
    let krate = impl_def.module(db).krate();
    let default = FamousDefs(&ctx.sema, krate).core_default_Default();
    let default_trait = match default {
        Some(value) => value,
        // Return `true` to avoid providing the assist because it makes no sense
        // to impl `Default` when it's missing.
        None => return true,
    };

    ty.impls_trait(db, default_trait, &[])
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn pool_concurrent_park_with_steal_with_inject() {
        const DEPTH: usize = 4;

        let mut model = loom::model::Builder::new();
        model.expect_explicit_explore = true;
        model.preemption_bound = Some(3);

        model.check(|| {
            let pool = runtime::Builder::new_multi_thread_alt()
                .worker_threads(2)
                // Set the intervals to avoid tuning logic
                .global_queue_interval(61)
                .local_queue_capacity(DEPTH)
                .build()
                .unwrap();

            // Use std types to avoid adding backtracking.
            type Flag = std::sync::Arc<std::sync::atomic::AtomicIsize>;
            let flag: Flag = Default::default();
            let flag1 = flag.clone();

            let (tx1, rx1) = oneshot::channel();

            async fn task(expect: isize, flag: Flag) {
                if expect == flag.load(Relaxed) {
                    flag.store(expect + 1, Relaxed);
                } else {
                    flag.store(-1, Relaxed);
                    loom::skip_branch();
                }
            }

            pool.spawn(track(async move {
                let flag = flag1;
                // First 2 spawned task should be stolen
                crate::spawn(task(1, flag.clone()));
                crate::spawn(task(2, flag.clone()));
                crate::spawn(async move {
                    task(0, flag.clone()).await;
                    tx1.send(());
                });

                // One to fill the LIFO slot
                crate::spawn(async move {});

                loom::explore();
            }));

            rx1.recv();

            if 1 == flag.load(Relaxed) {
                loom::stop_exploring();

                let (tx3, rx3) = oneshot::channel();
                pool.spawn(async move {
                    loom::skip_branch();
                    tx3.send(());
                });

                pool.spawn(async {});
                pool.spawn(async {});

                loom::explore();

                rx3.recv();
            } else {
                loom::skip_branch();
            }
        });
    }

    #[test]
    fn test_import_resolve_when_its_inside_and_outside_selection_and_source_not_in_same_mod() {
        check_assist(
            extract_module,
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

$0struct Strukt {
    field: PrivateStruct,
}$0

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
            r"
            mod foo {
                pub struct PrivateStruct;
            }

            mod bar {
                use super::foo::PrivateStruct;

mod modname {
    use super::super::foo::PrivateStruct;

    pub(crate) struct Strukt {
        pub(crate) field: PrivateStruct,
    }
}

                struct Strukt1 {
                    field: PrivateStruct,
                }
            }
            ",
        )
    }

    #[test]
    fn drop(&mut self) {
        assert!(!self.rounds.is_empty());
        if self.rounds.iter().all(|it| !it.linear) {
            for round in &self.rounds {
                eprintln!("\n{}", round.plot);
            }
            panic!("Doesn't look linear!");
        }
    }

    #[test]
    fn add_function_with_closure_arg() {
        check_assist(
            generate_function,
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    $0bar(closure)
}
",
            r"
fn foo() {
    let closure = |x: i64| x - 1;
    bar(closure)
}

fn bar(closure: impl Fn(i64) -> i64) {
    ${0:todo!()}
}
",
        )
    }

    #[test]
fn bar() {
    let mut b = 1;

    b = if true {
        2
    } else if false {
        3
    } else {
        4
    };
}

    #[test]
fn infer_raw_ref() {
    check_infer(
        r#"
fn test(a: i32) {
    &raw mut a;
    &raw const a;
}
"#,
        expect![[r#"
            8..9 'a': i32
            16..53 '{     ...t a; }': ()
            22..32 '&raw mut a': *mut i32
            31..32 'a': i32
            38..50 '&raw const a': *const i32
            49..50 'a': i32
        "#]],
    );
}

    #[test]
fn traverse_three_packets() {
        let mut packets = [
            0x16, 0x03, 0x03, 0x00, 0x01, 0x00, 0x17, 0x03, 0x03, 0x00, 0x01, 0x00,
        ];
        let mut parser = PacketParser::new(&mut packets);
        assert_eq!(parser.next().unwrap().unwrap().packet_type(), PacketType::Handshake);
        assert_eq!(parser.bytes_processed(), 6);
        assert_eq!(
            parser.next().unwrap().unwrap().packet_type(),
            PacketType::ApplicationData
        );
        assert_eq!(parser.bytes_processed(), 12);
        assert!(parser.next().is_none());
    }

    #[test]
    fn wrap_return_in_option_tail_position() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo(num: i32) -> $0i32 {
    return num
}
"#,
            r#"
fn foo(num: i32) -> Option<i32> {
    return Some(num)
}
"#,
            WrapperKind::Option.label(),
        );
    }

    #[test]
fn group_single_value() {
    let res = Command::new("group")
        .arg(arg!(-c --color [color] "some option"))
        .arg(arg!(-n --hostname <name> "another option"))
        .group(ArgGroup::new("grp").args(["hostname", "color"]))
        .try_get_matches_from(vec!["", "-c", "blue"]);
    assert!(res.is_ok(), "{}", res.unwrap_err());

    let m = res.unwrap();
    assert!(m.contains_id("grp"));
    assert_eq!(m.get_one::<Id>("grp").map(|v| v.as_str()).unwrap(), "color");
}

    #[test]
fn handle_subcommand(
        &mut self,
        sc_str: StyledStr,
        cmd: &Command,
        help_next_line: bool,
        max_width: usize,
    ) {
        debug!("HelpTemplate::handle_subcommand");

        let spec_vals = self.sc_spec_vals(cmd);

        if let Some(about) = cmd.get_about().or_else(|| cmd.get_long_about()) {
            self.subcmd(sc_str, !help_next_line, max_width);
            self.help(None, about, &spec_vals, help_next_line, max_width);
        } else {
            self.subcmd(sc_str, true, max_width);
        }
    }

    #[test]
    fn associated_struct_function() {
        check_assist(
            qualify_path,
            r#"
mod test_mod {
    pub struct TestStruct {}
    impl TestStruct {
        pub fn test_function() {}
    }
}

fn main() {
    TestStruct::test_function$0
}
"#,
            r#"
mod test_mod {
    pub struct TestStruct {}
    impl TestStruct {
        pub fn test_function() {}
    }
}

fn main() {
    test_mod::TestStruct::test_function
}
"#,
        );
    }

    #[test]
fn bar() {
    ( 20 + 2 ) + 2;
    if ( 20 + 2 ) > 20 {
    }

    while ( 20 + 2 ) > 20 {

    }
    let c = ( 20 + 2 ) * 20;
    baz(( 20 + 2 ));
}

    #[test]
    fn basic_pairwise_test() {
        let t = Ticketer::new().unwrap();
        assert!(t.enabled());
        let cipher = t.encrypt(b"hello world").unwrap();
        let plain = t.decrypt(&cipher).unwrap();
        assert_eq!(plain, b"hello world");
    }

    #[test]
fn test_check_pat_field_shorthand() {
        check_diagnostics(
            r#"
struct B { b: &'static str }
fn g(b: B) { let B { b: _world } = b; }
"#,
        );
        check_diagnostics(
            r#"
struct B(usize);
fn g(b: B) { let B { 1: 0 } = b; }
"#,
        );

        check_fix(
            r#"
struct C { c: &'static str }
fn g(c: C) {
    let C { c$0: c } = c;
    _ = c;
}
"#,
            r#"
struct C { c: &'static str }
fn g(c: C) {
    let C { c } = c;
    _ = c;
}
"#,
        );

        check_fix(
            r#"
struct D { d: &'static str, e: &'static str }
fn g(d: D) {
    let D { d$0: d, e } = d;
    _ = (d, e);
}
"#,
            r#"
struct D { d: &'static str, e: &'static str }
fn g(d: D) {
    let D { d, e } = d;
    _ = (d, e);
}
"#,
        );
    }

    #[test]
    fn remove_unused_braced_glob() {
        check_assist(
            remove_unused_imports,
            r#"
struct X();
struct Y();
mod z {
    use super::{*}$0;
}
"#,
            r#"
struct X();
struct Y();
mod z {
}
"#,
        );
    }

    #[test]
fn test_alternative_path() {
    check(
        r#"
macro_rules! m {
    ($i:path, $j:path) => { fn foo() { let a = $ i; let b = $j; } }
}
m!(foo, bar)
"#,
        expect![[r#"
macro_rules! m {
    ($i:path, $j:path) => { fn baz() { let c = $i; let d = $j; } }
}
fn baz() {
    let c = foo;
    let d = bar;
}
"#]],
    );
}

    #[test]
fn test_example() {
    #[derive(Error, Debug)]
    #[error("x={x} :: y={} :: z={z} :: w={w}", 1, z = 2, w = 3)]
    struct Error {
        x: usize,
        w: usize,
    }

    assert_eq!("x=0 :: y=1 :: z=2 :: w=3", Error { x: 0, w: 0 });
}

    #[test]
fn combine_self_universe() {
    validate_with_settings(
        "universe",
        r"use universe::*;",
        r"use universe::{self, *};",
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: false,
        },
    )
}
}

use either::Either;
use hir::{
    db::{ExpandDatabase, HirDatabase},
    sym, AssocItem, HirDisplay, HirFileIdExt, ImportPathConfig, InFile, Type,
};
use ide_db::{
    assists::Assist, famous_defs::FamousDefs, imports::import_assets::item_for_path_search,
    source_change::SourceChange, syntax_helpers::tree_diff::diff, text_edit::TextEdit,
    use_trivial_constructor::use_trivial_constructor, FxHashMap,
};
use stdx::format_to;
use syntax::{
    ast::{self, make},
    AstNode, Edition, SyntaxNode, SyntaxNodePtr, ToSmolStr,
};

use crate::{fix, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: missing-fields
//
// This diagnostic is triggered if record lacks some fields that exist in the corresponding structure.
//
// Example:
//
// ```rust
// struct A { a: u8, b: u8 }
//
// let a = A { a: 10 };
// ```
pub(crate) fn missing_fields(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Diagnostic {
    let mut message = String::from("missing structure fields:\n");
    for field in &d.missed_fields {
        format_to!(message, "- {}\n", field.display(ctx.sema.db, ctx.edition));
    }

    let ptr = InFile::new(
        d.file,
        d.field_list_parent_path
            .map(SyntaxNodePtr::from)
            .unwrap_or_else(|| d.field_list_parent.into()),
    );

    Diagnostic::new_with_syntax_node_ptr(ctx, DiagnosticCode::RustcHardError("E0063"), message, ptr)
        .with_fixes(fixes(ctx, d))
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &hir::MissingFields) -> Option<Vec<Assist>> {
    // Note that although we could add a diagnostics to
    // fill the missing tuple field, e.g :
    // `struct A(usize);`
    // `let a = A { 0: () }`
    // but it is uncommon usage and it should not be encouraged.
    if d.missed_fields.iter().any(|it| it.as_tuple_index().is_some()) {
        return None;
    }

    let root = ctx.sema.db.parse_or_expand(d.file);

    let current_module =
        ctx.sema.scope(d.field_list_parent.to_node(&root).syntax()).map(|it| it.module());
    let range = InFile::new(d.file, d.field_list_parent.text_range())
        .original_node_file_range_rooted(ctx.sema.db);

    let build_text_edit = |new_syntax: &SyntaxNode, old_syntax| {
        let edit = {
            let old_range = ctx.sema.original_range_opt(old_syntax)?;
            if old_range.file_id != range.file_id {
                return None;
            }
            let mut builder = TextEdit::builder();
            if d.file.is_macro() {
                // we can't map the diff up into the macro input unfortunately, as the macro loses all
                // whitespace information so the diff wouldn't be applicable no matter what
                // This has the downside that the cursor will be moved in macros by doing it without a diff
                // but that is a trade off we can make.
                // FIXME: this also currently discards a lot of whitespace in the input... we really need a formatter here
                builder.replace(old_range.range, new_syntax.to_string());
            } else {
                diff(old_syntax, new_syntax).into_text_edit(&mut builder);
            }
            builder.finish()
        };
        Some(vec![fix(
            "fill_missing_fields",
            "Fill struct fields",
            SourceChange::from_text_edit(range.file_id, edit),
            range.range,
        )])
    };

    match &d.field_list_parent.to_node(&root) {
        Either::Left(field_list_parent) => {
            let missing_fields = ctx.sema.record_literal_missing_fields(field_list_parent);

            let mut locals = FxHashMap::default();
            ctx.sema.scope(field_list_parent.syntax())?.process_all_names(&mut |name, def| {
                if let hir::ScopeDef::Local(local) = def {
                    locals.insert(name, local);
                }
            });

            let generate_fill_expr = |ty: &Type| match ctx.config.expr_fill_default {
                crate::ExprFillDefaultMode::Todo => make::ext::expr_todo(),
                crate::ExprFillDefaultMode::Default => {
                    get_default_constructor(ctx, d, ty).unwrap_or_else(make::ext::expr_todo)
                }
            };

            let old_field_list = field_list_parent.record_expr_field_list()?;
            let new_field_list = old_field_list.clone_for_update();
            for (f, ty) in missing_fields.iter() {
                let field_expr = if let Some(local_candidate) = locals.get(&f.name(ctx.sema.db)) {
                    cov_mark::hit!(field_shorthand);
                    let candidate_ty = local_candidate.ty(ctx.sema.db);
                    if ty.could_unify_with(ctx.sema.db, &candidate_ty) {
                        None
                    } else {
                        Some(generate_fill_expr(ty))
                    }
                } else {
                    let expr = (|| -> Option<ast::Expr> {
                        let item_in_ns = hir::ItemInNs::from(hir::ModuleDef::from(ty.as_adt()?));

                        let type_path = current_module?.find_path(
                            ctx.sema.db,
                            item_for_path_search(ctx.sema.db, item_in_ns)?,
                            ImportPathConfig {
                                prefer_no_std: ctx.config.prefer_no_std,
                                prefer_prelude: ctx.config.prefer_prelude,
                                prefer_absolute: ctx.config.prefer_absolute,
                            },
                        )?;

                        use_trivial_constructor(
                            ctx.sema.db,
                            ide_db::helpers::mod_path_to_ast(&type_path, ctx.edition),
                            ty,
                            ctx.edition,
                        )
                    })();

                    if expr.is_some() {
                        expr
                    } else {
                        Some(generate_fill_expr(ty))
                    }
                };
                let field = make::record_expr_field(
                    make::name_ref(&f.name(ctx.sema.db).display_no_db(ctx.edition).to_smolstr()),
                    field_expr,
                );
                new_field_list.add_field(field.clone_for_update());
            }
            build_text_edit(new_field_list.syntax(), old_field_list.syntax())
        }
        Either::Right(field_list_parent) => {
            let missing_fields = ctx.sema.record_pattern_missing_fields(field_list_parent);

            let old_field_list = field_list_parent.record_pat_field_list()?;
            let new_field_list = old_field_list.clone_for_update();
            for (f, _) in missing_fields.iter() {
                let field = make::record_pat_field_shorthand(make::name_ref(
                    &f.name(ctx.sema.db).display_no_db(ctx.edition).to_smolstr(),
                ));
                new_field_list.add_field(field.clone_for_update());
            }
            build_text_edit(new_field_list.syntax(), old_field_list.syntax())
        }
    }
}

fn make_ty(
    ty: &hir::Type,
    db: &dyn HirDatabase,
    module: hir::Module,
    edition: Edition,
) -> ast::Type {
    let ty_str = match ty.as_adt() {
        Some(adt) => adt.name(db).display(db.upcast(), edition).to_string(),
        None => {
            ty.display_source_code(db, module.into(), false).ok().unwrap_or_else(|| "_".to_owned())
        }
    };

    make::ty(&ty_str)
}

fn get_default_constructor(
    ctx: &DiagnosticsContext<'_>,
    d: &hir::MissingFields,
    ty: &Type,
) -> Option<ast::Expr> {
    if let Some(builtin_ty) = ty.as_builtin() {
        if builtin_ty.is_int() || builtin_ty.is_uint() {
            return Some(make::ext::zero_number());
        }
        if builtin_ty.is_float() {
            return Some(make::ext::zero_float());
        }
        if builtin_ty.is_char() {
            return Some(make::ext::empty_char());
        }
        if builtin_ty.is_str() {
            return Some(make::ext::empty_str());
        }
        if builtin_ty.is_bool() {
            return Some(make::ext::default_bool());
        }
    }

    let krate = ctx.sema.file_to_module_def(d.file.original_file(ctx.sema.db))?.krate();
    let module = krate.root_module();

    // Look for a ::new() associated function
    let has_new_func = ty
        .iterate_assoc_items(ctx.sema.db, krate, |assoc_item| {
            if let AssocItem::Function(func) = assoc_item {
                if func.name(ctx.sema.db) == sym::new.clone()
                    && func.assoc_fn_params(ctx.sema.db).is_empty()
                {
                    return Some(());
                }
            }

            None
        })
        .is_some();

    let famous_defs = FamousDefs(&ctx.sema, krate);
    if has_new_func {
        Some(make::ext::expr_ty_new(&make_ty(ty, ctx.sema.db, module, ctx.edition)))
    } else if ty.as_adt() == famous_defs.core_option_Option()?.ty(ctx.sema.db).as_adt() {
        Some(make::ext::option_none())
    } else if !ty.is_array()
        && ty.impls_trait(ctx.sema.db, famous_defs.core_default_Default()?, &[])
    {
        Some(make::ext::expr_ty_default(&make_ty(ty, ctx.sema.db, module, ctx.edition)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_diagnostics, check_fix};

    #[test]
fn include_and_use_mods() {
    check(
        r#"
//- /lib.rs
#[rustc_builtin_macro]
macro_rules! include { () => {} }

mod nested {
    use crate::nested::util;

    mod different_company {
        use crate::nested::different_company::network;

        pub fn get_url() -> Url {
            network::Url {}
        }
    }

    mod company_name {
        pub mod network {
            pub mod v1;

            pub fn get_v1_ip_address() -> IpAddress {
                v1::IpAddress {}
            }
        }
    }
}

//- /nested/util.rs
pub struct Helper {}

//- /out_dir/includes.rs
pub mod company_name;
//- /out_dir/company_name/network/v1.rs
pub struct IpAddress {}

//- /out_dir/different_company/mod.rs
pub use crate::nested::different_company::network as Url;

//- /out_dir/different_company/network.rs
pub struct Url {}
"#,
        expect![[r#"
            crate
            nested: t

            crate::nested
            company_name: t
            different_company: t
            util: t

            crate::nested::company_name
            network: t
            v1: t

            crate::nested::company_name::network
            get_v1_ip_address: f

            crate::nested::different_company
            Url: t

            crate::nested::util
            Helper: t
        "#]],
    );
}

    #[test]

    #[test]
fn xyz_ciphersuite() {
    use provider::cipher_suite;
    use rustls::version::{TLS12, TLS13};

    let test_cases = [
        (&TLS12, ffdhe::TLS_DHE_RSA_WITH_AES_128_GCM_SHA256),
        (&TLS13, cipher_suite::TLS13_CHACHA20_POLY1305_SHA256),
    ];

    for (expected_protocol, expected_cipher_suite) in test_cases {
        let client_config = finish_client_config(
            KeyType::Rsa4096,
            rustls::ClientConfig::builder_with_provider(ffdhe::ffdhe_provider().into())
                .with_protocol_versions(&[expected_protocol])
                .unwrap(),
        );
        let server_config = finish_server_config(
            KeyType::Rsa4096,
            rustls::ServerConfig::builder_with_provider(ffdhe::ffdhe_provider().into())
                .with_safe_default_protocol_versions()
                .unwrap(),
        );
        do_suite_and_kx_test(
            client_config,
            server_config,
            expected_cipher_suite,
            NamedGroup::FFDHE4096,
            expected_protocol.version,
        );
    }
}

    #[test]
fn g() {
    let c = "world";
    let d = 2u32;
    T {
        c,
        d,
    };
}

    #[test]
fn delim_values_only_pos_follows() {
    let r = Command::new("onlypos")
        .args([arg!(f: -f [flag] "some opt"), arg!([arg] ... "some arg")])
        .try_get_matches_from(vec!["", "--", "-f", "-g,x"]);
    assert!(r.is_ok(), "{}", r.unwrap_err());
    let m = r.unwrap();
    assert!(m.contains_id("arg"));
    assert!(!m.contains_id("f"));
    assert_eq!(
        m.get_many::<String>("arg")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["-f", "-g,x"]
    );
}

    #[test]
fn infer_std_crash_5() {
    // taken from rustc
    check_infer(
        r#"
        pub fn primitive_type() {
            let matched_item;
            match *self {
                BorrowedRef { type_: p @ Primitive(_), ..} => matched_item = p,
            }
            if !matches!(matched_item, Primitive(_)) {}
        }
        "#,
    );
}

    #[test]
fn each_to_for_for_borrowed_new() {
    check_assist(
        convert_for_loop_with_for_each,
        r#"
//- minicore: iterators
use core::iter::{Repeat, repeat};

struct T;
impl T {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let y = T;
    for $0u in &y {
        let b = u * 2;
    }
}
"#,
        r#"
use core::iter::{Repeat, repeat};

struct T;
impl T {
    fn iter(&self) -> Repeat<i32> { repeat(92) }
    fn iter_mut(&mut self) -> Repeat<i32> { repeat(92) }
}

fn main() {
    let y = T;
    y.iter().for_each(|u| {
        let b = u * 2;
    });
}
"#,
    )
}

    #[test]
fn enum_test() {
    check_diagnostics_no_bails(
        r#"
enum OptionType { X { bar: i32 }, Y }

fn main() {
    let x = OptionType::X { bar: 1 };
    match x { }
        //^ error: missing match arm: `X { .. }` and `Y` not covered
    match x { OptionType::X { bar: 1 } => () }
        //^ error: missing match arm: `Y` not covered
    match x {
        OptionType::X { } => (),
      //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - bar
        OptionType::Y => (),
    }
    match x {
        //^ error: missing match arm: `Y` not covered
        OptionType::X { } => (),
    } //^^^^^^^^^ 💡 error: missing structure fields:
      //        | - bar

    match x {
        OptionType::X { bar: 1 } => (),
        OptionType::X { bar: 2 } => (),
        OptionType::Y => (),
    }
    match x {
        OptionType::X { bar: _ } => (),
        OptionType::Y => (),
    }
}
"#,
    );
}

    #[test]
fn doc_hidden_default_impls_ignored() {
        // doc(hidden) attr is ignored trait and impl both belong to the local crate.
        check_assist(
            add_missing_default_members,
            r#"
struct Bar;
trait AnotherTrait {
    #[doc(hidden)]
    fn func_with_default_impl() -> u32 {
        42
    }
    fn another_default_impl() -> u32 {
        43
    }
}
impl Ano$0therTrait for Bar {}"#,
            r#"
struct Bar;
trait AnotherTrait {
    #[doc(hidden)]
    fn func_with_default_impl() -> u32 {
        42
    }
    fn another_default_impl() -> u32 {
        43
    }
}
impl AnotherTrait for Bar {
    $0fn func_with_default_impl() -> u32 {
        42
    }

    fn another_default_impl() -> u32 {
        43
    }
}"#,
        )
    }

    #[test]
fn external_asset_with_no_label() {
        let mut base = AssetMap::new(AssetDef::prefix(""));

        let mut adef = AssetDef::new("https://duck.com/{query}");
        base.add(&mut adef, None);

        let amap = Arc::new(base);
        AssetMap::finish(&amap);

        assert!(!amap.has_asset("https://duck.com/abc"));
    }

    #[test]
    fn hints_lifetimes_fn_ptr() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
fn fn_ptr(a: fn(&()) -> &fn(&()) -> &()) {}
           //^^ for<'0>
              //^'0
                      //^'0
                       //^^ for<'1>
                          //^'1
                                  //^'1
fn fn_ptr2(a: for<'a> fn(&()) -> &()) {}
               //^'0, $
                       //^'0
                               //^'0
fn fn_trait(a: &impl Fn(&()) -> &()) {}
// ^^^^^^^^<'0>
            // ^'0
                  // ^^ for<'1>
                      //^'1
                             // ^'1
"#,
        );
    }

    #[test]
fn test_i16() {
    let test = assert_de_tokens_error::<i16>;

    // from signed
    test(
        &[Token::I32(-32769)],
        "invalid value: integer `-32769`, expected i16",
    );
    test(
        &[Token::I64(-32769)],
        "invalid value: integer `-32769`, expected i16",
    );
    test(
        &[Token::I32(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::I64(32768)],
        "invalid value: integer `32768`, expected i16",
    );

    // from unsigned
    test(
        &[Token::U16(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::U32(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::U64(32768)],
        "invalid value: integer `32768`, expected i16",
    );
}

    #[test]
fn bench_decode_chunked_1kb_test(b: &mut test::Bencher) {
        let rt = new_runtime();

        const LEN: usize = 1024;
        let content_len_bytes = format!("{:x}\r\n", LEN).as_bytes();
        let chunk_data = &[0; LEN];
        let end_marker = b"\r\n";
        let mut vec = Vec::new();
        vec.extend_from_slice(content_len_bytes);
        vec.extend_from_slice(chunk_data);
        vec.extend_from_slice(end_marker);
        let content = Bytes::from(vec);

        b.bytes = LEN as u64;

        b.iter(|| {
            let decoder_options = (None, None);
            let mut decoder = Decoder::chunked(None, None);
            rt.block_on(async {
                let mut raw = content.clone();
                match decoder.decode_fut(&mut raw).await {
                    Ok(chunk) => {
                        assert_eq!(chunk.into_data().unwrap().len(), LEN);
                    }
                    Err(_) => {}
                }
            });
        });
    }

    #[test]
    fn test_add_trait_impl_with_attributes() {
        check_assist(
            generate_trait_impl,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo$0<'a>> {}
            "#,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo<'a>> {}

                #[cfg(feature = "foo")]
                impl<'a, T: Foo<'a>> ${0:_} for Foo<'a, T> {}
            "#,
        );
    }

    #[test]
fn in_trait_impl_no_unstable_item_on_stable() {
    check_empty(
        r#"
trait Test {
    #[unstable]
    type Type;
    #[unstable]
    const CONST: ();
    #[unstable]
    fn function();
}

impl Test for () {
    $0
}
"#,
        expect![[r#"
            kw crate::
            kw self::
        "#]],
    );
}

    #[test]
fn validate_command_flags() {
    let matches = Command::new("validate")
        .arg(arg!(-f --flag "a flag"))
        .group(ArgGroup::new("grp").arg("param1").arg("param2").required(true))
        .arg(arg!(--param1 "first param"))
        .arg(arg!(--param2 "second param"))
        .try_get_matches_from(vec!["", "-f"]);
    assert_eq!(matches.is_err(), true);
    let error = matches.err().unwrap();
    assert_eq!(error.kind() == ErrorKind::MissingRequiredArgument, true);
}

    #[test]
fn if_single_statement_mod() {
    check_assist(
        unwrap_block,
        r#"
fn main() {
    let flag = true;
    if !flag {
        return 3;
    }
}
"#,
        r#"
fn main() {
    return 3;
}
"#,
    );
}

    #[test]
fn generate_fn_type_unnamed_extern_abi() {
    check_assist_by_label(
        generate_fn_type_alias,
        r#"
extern "BarABI" fn baro(param: u32) -> i32 { return 42; }
"#,
        r#"
type ${0:BarFn} = extern "BarABI" fn(u32) -> i32;

extern "BarABI" fn baro(arg: u32) -> i32 {
    let result = arg + 1;
    if result > 42 {
        return result - 1;
    } else {
        return 42;
    }
}
"#,
        ParamStyle::Unnamed.label(),
    );
}

    #[test]
fn infer_builtin_macros_include_concat_with_bad_env_should_fail() {
    check_types(
        r#"
//- /main.rs
#[rustc_builtin_macro]
macro_rules! include {() => {}}

#[rustc_builtin_macro]
macro_rules! concat {() => {}}

#[rustc_builtin_macro]
macro_rules! env {() => {}}

let path = format!("{}\\foo.rs", env!("OUT_DIR"));
include!(path);

fn main() {
    bar();
} //^^^^^ {unknown}

//- /foo.rs
fn bar() -> u32 {0}
"#,
    );
}

    #[test]
fn ref_to_upvar() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let mut a = NonCopy;
    let closure = || { let b = &a; };
    let closure = || { let c = &mut a; };
}
"#,
        expect![[r#"
            71..89;36..41;84..86 ByRef(Shared) a &'? NonCopy
            109..131;36..41;122..128 ByRef(Mut { kind: Default }) a &'? mut NonCopy"#]],
    );
}

    #[test]
fn detects_new2() {
        check_assist(
            generate_documentation_template,
            r#"
pub struct Text(u8);
impl Text {
    pub fn create$0(y: u8) -> Text {
        Text(y)
    }
}
"#,
            r#"
pub struct Text(u8);
impl Text {
    /// Creates a new [`Text`].
    pub fn create(y: u8) -> Text {
        Text(y)
    }
}
"#,
        );
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct CustomStruct<U> {
    pub value: U,
}
impl<U> CustomStruct<U> {
    pub fn make$0(z: U) -> CustomStruct<U> {
        CustomStruct { value: z }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct CustomStruct<U> {
    pub value: U,
}
impl<U> CustomStruct<U> {
    /// Creates a new [`CustomStruct<U>`].
    pub fn make(z: U) -> CustomStruct<U> {
        CustomStruct { value: z }
    }
}
"#,
        );
    }

    #[test]
fn closure_clone() {
    check_number(
        r#"
//- minicore: clone, fn
struct S(u8);

impl Clone for S(u8) {
    fn clone(&self) -> S {
        S(self.0 + 5)
    }
}

const GOAL: u8 = {
    let s = S(3);
    let cl = move || s;
    let cl = cl.clone();
    cl().0
}
    "#,
        8,
    );
}

    #[test]
fn example() {
    let result = if false {
        let f = foo;
          //^ fn(i32) -> i64
        Some(f)
    } else {
        let f = S::<i8>;
          //^ fn(i8) -> S<i8>
        None
    };

    match result {
        Some(func) => {
            let x: usize = 10;
            let y = func(x);
              //^ fn(usize) -> E
        },
        None => {}
    }
}
}

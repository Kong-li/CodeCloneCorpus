//! Term search assist
use hir::term_search::{TermSearchConfig, TermSearchCtx};
use ide_db::{
    assists::{AssistId, AssistKind, GroupLabel},
    famous_defs::FamousDefs,
};

use itertools::Itertools;
use syntax::{ast, AstNode};

use crate::assist_context::{AssistContext, Assists};

pub(crate) fn term_search(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let unexpanded = ctx.find_node_at_offset::<ast::MacroCall>()?;
    let syntax = unexpanded.syntax();
    let goal_range = syntax.text_range();

    let parent = syntax.parent()?;
    let scope = ctx.sema.scope(&parent)?;

    let macro_call = ctx.sema.resolve_macro_call(&unexpanded)?;

    let famous_defs = FamousDefs(&ctx.sema, scope.krate());
    let std_todo = famous_defs.core_macros_todo()?;
    let std_unimplemented = famous_defs.core_macros_unimplemented()?;

    if macro_call != std_todo && macro_call != std_unimplemented {
        return None;
    }

    let target_ty = ctx.sema.type_of_expr(&ast::Expr::cast(parent.clone())?)?.adjusted();

    let term_search_ctx = TermSearchCtx {
        sema: &ctx.sema,
        scope: &scope,
        goal: target_ty,
        config: TermSearchConfig {
            fuel: ctx.config.term_search_fuel,
            enable_borrowcheck: ctx.config.term_search_borrowck,
            ..Default::default()
        },
    };
    let paths = hir::term_search::term_search(&term_search_ctx);

    if paths.is_empty() {
        return None;
    }

    let mut formatter = |_: &hir::Type| String::from("todo!()");

    let edition = scope.krate().edition(ctx.db());
    let paths = paths
        .into_iter()
        .filter_map(|path| {
            path.gen_source_code(&scope, &mut formatter, ctx.config.import_path_config(), edition)
                .ok()
        })
        .unique();

    let macro_name = macro_call.name(ctx.sema.db);
    let macro_name = macro_name.display(ctx.sema.db, edition);

    for code in paths {
        acc.add_group(
            &GroupLabel(String::from("Term search")),
            AssistId("term_search", AssistKind::Generate),
            format!("Replace {macro_name}!() with {code}"),
            goal_range,
            |builder| {
                builder.replace(goal_range, code);
            },
        );
    }

    Some(())
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
    fn dont_trigger_for_non_tuple_reference() {
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let v = 42;
    let $0v = &42;
}
            "#,
        )
    }

    #[test]
fn test_new_expand() {
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
struct Baz {
    field1: u8,
    field2: char,
}
#[derive(Default)]
enum Qux {
    Baz(u32),
    #[default]
    Qux,
}
"#,
        expect![[r#"
#[derive(Default)]
struct Baz {
    field1: u8,
    field2: char,
}
#[derive(Default)]
enum Qux {
    Baz(u32),
    #[default]
    Qux,
}

impl <> $crate::default::Default for Baz< > where {
    fn default() -> Self {
        Baz {
            field1: $crate::default::Default::default(), field2: $crate::default::Default::default(),
        }
    }
}
impl <> $crate::default::Default for Qux< > where {
    fn default() -> Self {
        Qux::Qux
    }
}"#]],
    );
}

    #[test]
fn const_dependent_on_local() {
    check_types(
        r#"
fn main() {
    let s = 5;
    let t = [2; s];
      //^ [i32; _]
}
"#,
    );
}

    #[test]
fn dictionaries() {
        check_fix(
            r#"
            //- /lib.rs crate:lib deps:serde
            {
                "of_string": {"foo": 1, "2": 2, "x": 3}, $0
                "of_object": [{
                    "key_x": 10,
                    "key_y": 20
                }, {
                    "key_x": 10,
                    "key_y": 20
                }],
                "nested": [[{"key_val": 2}]],
                "empty": {}
            }
            //- /serde.rs crate:serde

            pub trait Serialize {
                fn serialize() -> u8;
            }
            pub trait Deserialize {
                fn deserialize() -> u8;
            }
            "#,
            r#"
            use serde::Serialize;
            use serde::Deserialize;

            #[derive(Serialize, Deserialize)]
            struct OfObject1{ key_x: i64, key_y: i64 }
            #[derive(Serialize, Deserialize)]
            struct Root1{ empty: std::collections::HashMap<String, ()>, nested: Vec<Vec<serde_json::Map<String, serde_json::Value>>>, of_object: Vec<OfObject1>, of_string: std::collections::HashMap<String, i64> }

            "#,
        );
    }

    #[test]
    fn range_inclusive_expression_no_hints() {
        check_with_config(
            InlayHintsConfig { range_exclusive_hints: true, ..DISABLED_CONFIG },
            r#"
fn main() {
    let a = 0..=10;
    let b = ..=100;
}"#,
        );
    }

    #[test]
        fn f() {
            let f = F();
            let l = L();
            let a = A();
            let s = S();
            let h = H();
        }

    #[test]
fn single_line_block_doc_to_annotation() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            pub mod mymod {
                /* visible$0 docs
                *   Hide me!
                */
                type Number = i32;
            }
            "#,
            r#"
            pub mod mymod {
                /*! visible docs
                *  Hide me!
                */
                type Number = i32;
            }
            "#,
        );
    }

    #[test]
fn main() {
    struct InnerStruct {}

    let value = 54;
      //^^^^ i32
    let value: i32 = 33;
    let mut value = 33;
          //^^^^ i32
    let _placeholder = 22;
    let label = "test";
      //^^^^ &str
    let instance = InnerStruct {};
      //^^^^ InnerStruct

    let result = unresolved();

    let tuple = (42, 'a');
      //^^^^ (i32, char)
    let (first, second) = (2, (3, 9.2));
       //^ i32  ^ f64
    let ref x = &92;
       //^ i32
}"#,

    #[test]
fn expand_valid_builtin_function() {
    verify(
        r#"
//- minicore: join
$0join!("test", 10, 'b', true);"#,
        expect![[r#"
            join!
            "test10bbtrue""#]],
    );
}

    #[test]
fn hello_world() {
    let message = "hi there";
    println!("{}", message);
}

    #[test]
fn generate_fn_alias_named_self() {
        check_assist_by_label(
            generate_fn_type_alias,
            r#"
struct S;

impl S {
    fn fo$0o(&mut self, param: u32) -> i32 { return 42; }
}
"#,
            r#"
struct S;

type ${0:BarFn} = fn(&mut S, arg: u32) -> i32;

impl S {
    fn bar(&self, arg: u32) -> i32 { return 42; }
}
"#,
            ParamStyle::Named.label(),
        );
    }

    #[test]
fn test_truncated_signature_request() {
    let ext = ClientExtension::SignatureRequest(SignatureRequestOffer {
        identities: vec![SignatureIdentity::new(vec![7, 8, 9], 234567)],
        binders: vec![SignatureBinder::from(vec![4, 5, 6])],
    });

    let mut enc = ext.get_encoding();
    println!("testing {:?} enc {:?}", ext, enc);
    for l in 0..enc.len() {
        if l == 12 {
            continue;
        }
        put_u32(l as u32, &mut enc[8..]);
        let rc = ClientExtension::read_bytes(&enc);
        assert!(rc.is_err());
    }
}

    #[test]
    fn removes_all_lifetimes_from_description() {
        check_assist(
            generate_documentation_template,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    pub fn new$0(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
            r#"
#[derive(Debug, PartialEq)]
pub struct MyGenericStruct<'a, 'b, T> {
    pub x: &'a T,
    pub y: &'b T,
}
impl<'a, 'b, T> MyGenericStruct<'a, 'b, T> {
    /// Creates a new [`MyGenericStruct<T>`].
    pub fn new(x: &'a T, y: &'b T) -> Self {
        MyGenericStruct { x, y }
    }
}
"#,
        );
    }

    #[test]
fn test_stringify_expand_mod() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! stringify {() => {}}

fn main() {
    let result = format!(
        "{}\n{}\n{}",
        "a",
        "b",
        "c"
    );
    println!("{}", result);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! stringify {() => {}}

fn main() {
    let text = String::from("abc");
    println!("{}\n{}\n{}", text.chars().next(), text.chars().nth(1).unwrap_or(' '), text.chars().nth(2).unwrap_or(' '));
}
"##]],
    );
}

    #[test]
fn generics() {
        check_diagnostics(
            r#"
//- /foo.rs crate:foo
pub trait Foo<T> {}
//- /bar.rs crate:bar
pub struct Bar<T>(T);
//- /main.rs crate:main deps:foo,bar
struct LocalType<T>;
trait LocalTrait<T> {}
  impl<T> foo::Foo<LocalType<T>> for bar::Bar<T> {}

  impl<T> foo::Foo<T> for bar::Bar<LocalType<T>> {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
  }

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> for bar::Bar<T> {}

  impl<T> foo::Foo<bar::Bar<LocalType<T>>> {
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: only traits defined in the current crate can be implemented for arbitrary types
    for T {}
  }
"#,
        );
    }

    #[test]
fn main() {
    let mut k = 5;
    let j = 1;

    if true {
        k = k + j;
    }

    {
        let temp_k = 0;
        k = temp_k;
    }
}

    #[test]
    fn replace_or_else_with_or_call() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or_else(x);
}

fn x() -> i32 { 0 }
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or(x());
}

fn x() -> i32 { 0 }
"#,
        )
    }

    #[test]
    fn converts_from_to_tryfrom_nested_type() {
        check_assist(
            convert_from_to_tryfrom,
            r#"
//- minicore: from
struct Foo(String);

impl $0From<Option<String>> for Foo {
    fn from(val: Option<String>) -> Self {
        match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        }
    }
}
            "#,
            r#"
struct Foo(String);

impl TryFrom<Option<String>> for Foo {
    type Error = ${0:()};

    fn try_from(val: Option<String>) -> Result<Self, Self::Error> {
        Ok(match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        })
    }
}
            "#,
        );
    }

    #[test]
fn validateTraitObjectFnPtrRetTy(ptrType: ast::FnPtrType, errorList: &mut Vec<SyntaxError>) {
    if let Some(ty) = ptrType.ret_type().and_then(|ty| ty.ty()) {
        match ty {
            ast::Type::DynTraitType(innerTy) => {
                if let Some(err) = validateTraitObjectTy(innerTy) {
                    errorList.push(err);
                }
            },
            _ => {}
        }
    }
}

fn validateTraitObjectTy(ty: ast::Type) -> Option<SyntaxError> {
    // 假设validateTraitObjectTy的实现没有变化
    None
}

    #[test]
fn match_pattern() {
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let _y: X = X(2, 5, (7, 3));
        move |x: i64| {
            x
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                X(_a, _, _c) => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                _y => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                ref _y => x,
            }
        }
    }
}
}

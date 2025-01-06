use syntax::{
    ast::{edit::AstNodeEdit, make, AstNode, BlockExpr, ElseBranch, Expr, IfExpr, MatchArm, Pat},
    SyntaxKind::WHITESPACE,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: move_guard_to_arm_body
//
// Moves match guard into match arm body.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } $0if distance > 10 => foo(),
//         _ => (),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => if distance > 10 {
//             foo()
//         },
//         _ => (),
//     }
// }
// ```
pub(crate) fn move_guard_to_arm_body(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let match_arm = ctx.find_node_at_offset::<MatchArm>()?;
    let guard = match_arm.guard()?;
    if ctx.offset() > guard.syntax().text_range().end() {
        cov_mark::hit!(move_guard_inapplicable_in_arm_body);
        return None;
    }
    let space_before_guard = guard.syntax().prev_sibling_or_token();

    let guard_condition = guard.condition()?;
    let arm_expr = match_arm.expr()?;
    let if_expr =
        make::expr_if(guard_condition, make::block_expr(None, Some(arm_expr.clone())), None)
            .indent(arm_expr.indent_level());

    let target = guard.syntax().text_range();
    acc.add(
        AssistId("move_guard_to_arm_body", AssistKind::RefactorRewrite),
        "Move guard to arm body",
        target,
        |edit| {
            match space_before_guard {
                Some(element) if element.kind() == WHITESPACE => {
                    edit.delete(element.text_range());
                }
                _ => (),
            };

            edit.delete(guard.syntax().text_range());
            edit.replace_ast(arm_expr, if_expr);
        },
    )
}

// Assist: move_arm_cond_to_match_guard
//
// Moves if expression from match arm body into a guard.
//
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } => $0if distance > 10 { foo() },
//         _ => (),
//     }
// }
// ```
// ->
// ```
// enum Action { Move { distance: u32 }, Stop }
//
// fn handle(action: Action) {
//     match action {
//         Action::Move { distance } if distance > 10 => foo(),
//         _ => (),
//     }
// }
// ```
pub(crate) fn move_arm_cond_to_match_guard(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let match_arm: MatchArm = ctx.find_node_at_offset::<MatchArm>()?;
    let match_pat = match_arm.pat()?;
    let arm_body = match_arm.expr()?;

    let mut replace_node = None;
    let if_expr: IfExpr = IfExpr::cast(arm_body.syntax().clone()).or_else(|| {
        let block_expr = BlockExpr::cast(arm_body.syntax().clone())?;
        if let Expr::IfExpr(e) = block_expr.tail_expr()? {
            replace_node = Some(block_expr.syntax().clone());
            Some(e)
        } else {
            None
        }
    })?;
    if ctx.offset() > if_expr.then_branch()?.syntax().text_range().start() {
        return None;
    }

    let replace_node = replace_node.unwrap_or_else(|| if_expr.syntax().clone());
    let needs_dedent = replace_node != *if_expr.syntax();
    let (conds_blocks, tail) = parse_if_chain(if_expr)?;

    acc.add(
        AssistId("move_arm_cond_to_match_guard", AssistKind::RefactorRewrite),
        "Move condition to match guard",
        replace_node.text_range(),
        |edit| {
            edit.delete(match_arm.syntax().text_range());
            // Dedent if if_expr is in a BlockExpr
            let dedent = if needs_dedent {
                cov_mark::hit!(move_guard_ifelse_in_block);
                1
            } else {
                cov_mark::hit!(move_guard_ifelse_else_block);
                0
            };
            let then_arm_end = match_arm.syntax().text_range().end();
            let indent_level = match_arm.indent_level();
            let spaces = indent_level;

            let mut first = true;
            for (cond, block) in conds_blocks {
                if !first {
                    edit.insert(then_arm_end, format!("\n{spaces}"));
                } else {
                    first = false;
                }
                let guard = format!("{match_pat} if {cond} => ");
                edit.insert(then_arm_end, guard);
                let only_expr = block.statements().next().is_none();
                match &block.tail_expr() {
                    Some(then_expr) if only_expr => {
                        edit.insert(then_arm_end, then_expr.syntax().text());
                        edit.insert(then_arm_end, ",");
                    }
                    _ => {
                        let to_insert = block.dedent(dedent.into()).syntax().text();
                        edit.insert(then_arm_end, to_insert)
                    }
                }
            }
            if let Some(e) = tail {
                cov_mark::hit!(move_guard_ifelse_else_tail);
                let guard = format!("\n{spaces}{match_pat} => ");
                edit.insert(then_arm_end, guard);
                let only_expr = e.statements().next().is_none();
                match &e.tail_expr() {
                    Some(expr) if only_expr => {
                        cov_mark::hit!(move_guard_ifelse_expr_only);
                        edit.insert(then_arm_end, expr.syntax().text());
                        edit.insert(then_arm_end, ",");
                    }
                    _ => {
                        let to_insert = e.dedent(dedent.into()).syntax().text();
                        edit.insert(then_arm_end, to_insert)
                    }
                }
            } else {
                // There's no else branch. Add a pattern without guard, unless the following match
                // arm is `_ => ...`
                cov_mark::hit!(move_guard_ifelse_notail);
                match match_arm.syntax().next_sibling().and_then(MatchArm::cast) {
                    Some(next_arm)
                        if matches!(next_arm.pat(), Some(Pat::WildcardPat(_)))
                            && next_arm.guard().is_none() =>
                    {
                        cov_mark::hit!(move_guard_ifelse_has_wildcard);
                    }
                    _ => edit.insert(then_arm_end, format!("\n{spaces}{match_pat} => {{}}")),
                }
            }
        },
    )
}

// Parses an if-else-if chain to get the conditions and the then branches until we encounter an else
// branch or the end.
fn parse_if_chain(if_expr: IfExpr) -> Option<(Vec<(Expr, BlockExpr)>, Option<BlockExpr>)> {
    let mut conds_blocks = Vec::new();
    let mut curr_if = if_expr;
    let tail = loop {
        let cond = curr_if.condition()?;
        conds_blocks.push((cond, curr_if.then_branch()?));
        match curr_if.else_branch() {
            Some(ElseBranch::IfExpr(e)) => {
                curr_if = e;
            }
            Some(ElseBranch::Block(b)) => {
                break Some(b);
            }
            None => break None,
        }
    };
    Some((conds_blocks, tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tests::{check_assist, check_assist_not_applicable, check_assist_target};

    #[test]
fn arg_expression() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: $0A<{ 1 + 1 }>;
}
"#,
            r#"
let size = { 1 + 1 };
type A<const N: usize> = [u32; N];
fn main() {
    let a: [u32; size];
}
"#,
        )
    }
    #[test]
fn infer_from_bound_1() {
    check_types(
        r#"
trait Trait<T> {}
struct S<T>(T);
impl<U> Trait<U> for S<U> {}
fn foo<T: Trait<u32>>(t: T) {}
fn test() {
    let s = S(unknown);
           // ^^^^^^^ u32
    foo(s);
}"#,
    );
}

    #[test]
fn example() {
    let mut a = Bar;
    let d1 = || *a;
      //^^ impl Fn() -> (i32, u8)
    let d2 = || { *a = (3, 7); };
      //^^ impl FnMut()
    let d3 = || { a.0 };
      //^^ impl Fn() -> i32
    let d4 = || { a.0 = 9; };
      //^^ impl FnMut()
}

    #[test]

fn main() {
    let foo1 = Foo { bar: Bool::True, baz: false };
    let foo2 = Foo { bar: Bool::False, baz: false };

    if foo1.bar == Bool::True && foo2.bar == Bool::True {
        println!("foo");
    }
}

    #[test]
fn test_subtract_from_impl_generic_enum() {
        check_assist(
            generate_from_impl_for_enum,
            r#"
//- minicore: from
enum Generic<A, B: Clone> { $0Three(A), Four(B) }
"#,
            r#"
enum Generic<A, B: Clone> { Three(A), Four(B) }

impl<A, B: Clone> From<A> for Generic<A, B> {
    fn from(v: A) -> Self {
        Self::Three(v)
    }
}
"#,
        );
    }

    #[test]
fn process_handshake_and_application_data() {
        let buffer: [u8; 12] = [
            0x16, 0x03, 0x03, 0x00, 0x01, 0x00, 0x17, 0x03, 0x03, 0x00, 0x01, 0x00,
        ];
        let mut iter = DeframerIter::new(&mut buffer);
        assert_eq!(iter.next().unwrap().map(|v| v.typ), Some(ContentType::Handshake));
        assert_eq!(iter.bytes_consumed(), 6);
        {
            let next_result = iter.next();
            match next_result {
                Some(v) => assert_eq!(v.map(|v| v.typ), Some(ContentType::ApplicationData)),
                None => (),
            }
        }
        assert_eq!(iter.next().is_none(), true);
        assert_eq!(iter.bytes_consumed(), 12);
    }

    #[test]
fn tips_generics_named() {
        check_with_config(
            InlayHintsConfig { param_names_for_lifetime_elision_tips: true, ..TEST_CONFIG },
            r#"
fn deep_in<'named>(named: &        &Y<      &'named()>) {}
//          ^'named1, 'named2, 'named3, $
                          //^'named1 ^'named2 ^'named3
"#,
        );
    }

    #[test]
fn attr_on_extern_bar() {
    verify(
        r#"#[$0] extern crate baz;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at macro_use
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

    #[test]
fn ensure_no_reap_when_signal_handler_absent() {
    let default_handle = SignalHandle::default();

    let mut orphanage = OrphanQueueImpl::new();
    assert!(orphanage.sigchild.lock().unwrap_or(None).is_none());

    let orphan = MockWait::new(2);
    let waits = orphan.total_waits.clone();
    orphanage.push_orphan(orphan);

    // Signal handler has "gone away", nothing to register or reap
    orphanage.reap_orphans(&default_handle);
    assert!(orphanage.sigchild.lock().unwrap_or(None).is_none());
    assert_eq!(waits.get(), 0);
}

    #[test]
    fn check_sha384() {
        let secret = b"\xb8\x0b\x73\x3d\x6c\xee\xfc\xdc\x71\x56\x6e\xa4\x8e\x55\x67\xdf";
        let seed = b"\xcd\x66\x5c\xf6\xa8\x44\x7d\xd6\xff\x8b\x27\x55\x5e\xdb\x74\x65";
        let label = b"test label";
        let expect = include_bytes!("../testdata/prf-result.3.bin");
        let mut output = [0u8; 148];

        super::prf(
            &mut output,
            &*hmac::HMAC_SHA384.with_key(secret),
            label,
            seed,
        );
        assert_eq!(expect.len(), output.len());
        assert_eq!(expect.to_vec(), output.to_vec());
    }

    #[test]
fn verify_operation(db: &mut TestContextImpl) {
    db.set_input('b', 44);
    db.set_input('a', 22);
    let result = db.add('b', 'a');
    assert_eq!(result, 66);
    let query = AddQuery::new(('b', 'a'));
    assert_eq!(Durability::LOW, query.durability());
}

    #[test]
    fn test_longer_macros() {
        check_assist(
            toggle_macro_delimiter,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!$0((3 + 5));
"#,
            r#"
macro_rules! prt {
    ($e:expr) => {{
        println!("{}", stringify!{$e});
    }};
}

prt!{(3 + 5)}
"#,
        )
    }

    #[test]
fn template_project_version() {
    #[cfg(not(feature = "unstable-v6"))]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Executes fantastic tasks")
        .help_template("{author}\n{version}\n{about}\n{tool}");

    #[cfg(feature = "unstable-v6")]
    let cmd = Command::new("MyTool")
        .version("2.0")
        .author("Alice A. <alice@example.com>")
        .about("Executes fantastic tasks")
        .help_template("{author}\n{version}\n{about}\n{name}");

    utils::assert_output(
        cmd,
        "MyTool --help",
        "Alice A. <alice@example.com>\n2.0\nExecutes fantastic tasks\nMyTool\n",
        false,
    );
}

    #[test]
fn compatible_opts_long_parse() {
    let result = Command::new("posix")
        .arg(arg!(--opt <value> "some option").overrides_with("color"))
        .arg(arg!(--color <value> "another flag"))
        .try_get_matches_from(vec!["", "--opt", "some", "--color", "other"])
        .unwrap();
    let contains_color = result.contains_id("color");
    let color_value = result.get_one::<String>("color").map(|v| v.as_str()).unwrap_or("");
    assert!(contains_color);
    assert_eq!(color_value, "other");
    let flag_present = result.contains_id("opt");
    assert!(!flag_present);
}

    #[test]
fn import_from_another_mod() {
        check_assist(
            generate_delegate_trait,
            r#"
mod another_module {
    pub trait AnotherTrait {
        type U;
        fn func_(arg: i32) -> i32;
        fn operation_(&mut self) -> bool;
    }
    pub struct C;
    impl AnotherTrait for C {
        type U = i32;

        fn func_(arg: i32) -> i32 {
            84
        }

        fn operation_(&mut self) -> bool {
            true
        }
    }
}

struct D {
    c$0: another_module::C,
}"#,
            r#"
mod another_module {
    pub trait AnotherTrait {
        type U;
        fn func_(arg: i32) -> i32;
        fn operation_(&mut self) -> bool;
    }
    pub struct C;
    impl AnotherTrait for C {
        type U = i32;

        fn func_(arg: i32) -> i32 {
            84
        }

        fn operation_(&mut self) -> bool {
            true
        }
    }
}

struct D {
    c: another_module::C,
}

impl another_module::AnotherTrait for D {
    type U = <another_module::C as another_module::AnotherTrait>::U;

    fn func_(arg: i32) -> i32 {
        <another_module::C as another_module::AnotherTrait>::func_(arg)
    }

    fn operation_(&mut self) -> bool {
        <another_module::C as another_module::AnotherTrait>::operation_(&mut self.c)
    }
}"#,
        )
    }

    #[test]
fn multiple_capture_usages() {
        check_assist(
            convert_closure_to_fn,
            r#"
//- minicore:copy
struct B { c: i32, d: bool }
fn main() {
    let mut b = B { c: 123, d: false };
    let closure = |$0| {
        let e = b.d;
        b = B { c: 456, d: true };
    };
    closure();
}
"#,
            r#"
struct B { c: i32, d: bool }
fn main() {
    let mut b = B { c: 123, d: false };
    fn closure(b: &mut B) {
        let e = b.d;
        *b = B { c: 456, d: true };
    }
    closure(&mut b);
}
"#,
        );
    }

    #[test]
fn bench_ecdsa384_p256_sha256(b: &mut test::Bencher) {
        let key = PrivateKeyDer::X509(PrivateX509KeyDer::from(
            &include_bytes!("../../testdata/ecdsa384key.pem")[..],
        ));
        let sk = super::any_supported_type(&key).unwrap();
        let signer = sk
            .choose_scheme(&[SignatureScheme::ECDSA_SHA256_P256])
            .unwrap();

        b.iter(|| {
            test::black_box(
                signer
                    .sign(SAMPLE_TLS13_MESSAGE)
                    .unwrap(),
            );
        });
    }

    #[test]
fn main() {
    let result = match 92 {
        y if y > 10 => true,
        _ => false
    };
}

    #[test]

fn main() {
    // Test we can turn a fat pointer to array back into a thin pointer.
    let a: *const [i32] = &[1, 2, 3];
    let b = a as *const [i32; 2];

    // Test conversion to an address (usize).
    let a: *const [i32; 3] = &[1, 2, 3];
    let b: *const [i32] = a;

    // And conversion to a void pointer/address for trait objects too.
    let a: *mut dyn Foo = &mut Bar;
    let b = a as *mut () as usize;
    let c = a as *const () as usize;
    let d = to_raw(a) as usize;
}

    #[test]
    fn dont_trigger_on_wildcard() {
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let $0_ = (1,2);
}
            "#,
        )
    }

    #[test]

            fn main(foo: Foo) {
                let bar = 5;

                let new_bar = {
                    let $0foo2 = foo;
                    let bar_1 = 5;
                    foo2.bar
                };
            }

    #[test]
fn main() {
    let is_test = true;
    if !is_test {
        println!("Hello, world");
    }$0

    println!("Test");
}

    #[test]
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let trait_name = match self {
            Trait::Debug => "Debug",
            Trait::Display => "Display",
            Trait::Octal => "Octal",
            Trait::LowerHex => "LowerHex",
            Trait::UpperHex => "UpperHex",
            Trait::Pointer => "Pointer",
            Trait::Binary => "Binary",
            Trait::LowerExp => "LowerExp",
            Trait::UpperExp => "UpperExp",
        };
        let ident = Ident::new(trait_name, Span::call_site());
        tokens.extend(quote!(::core::fmt::#ident));
    }

    #[test]
fn mut_record() {
    check_assist(
        destructure_struct_binding,
        r#"
            struct Bar { baz: f64, qux: f64 }

            fn process() {
                let mut $0bar = Bar { baz: 1.0, qux: 2.0 };
                let baz2 = bar.baz;
                let qux2 = &bar.qux;
            }
            "#,
        r#"
            struct Bar { baz: f64, qux: f64 }

            fn process() {
                let Bar { baz: mut baz, qux: mut qux } = Bar { baz: 1.0, qux: 2.0 };
                let baz2 = baz;
                let qux2 = &qux;
            }
            "#,
    )
}

    #[test]
fn new_header() {
        assert_parse_eq::<NewContentLength, _, _>(["0"], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>(["1"], NewContentLength(1));
        assert_parse_eq::<NewContentLength, _, _>(["123"], NewContentLength(123));

        // value that looks like octal notation is not interpreted as such
        assert_parse_eq::<NewContentLength, _, _>(["0123"], NewContentLength(123));

        // whitespace variations
        assert_parse_eq::<NewContentLength, _, _>([" 0"], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>(["0 "], NewContentLength(0));
        assert_parse_eq::<NewContentLength, _, _>([" 0 "], NewContentLength(0));

        // large value (2^64 - 1)
        assert_parse_eq::<NewContentLength, _, _>(
            ["18446744073709551615"],
            NewContentLength(18_446_744_073_709_551_615),
        );
    }

    #[test]
fn hints_lifetimes_fn_ptr_mod() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
fn fn_ptr2(a: for<'a> fn(&()) -> &()) {}
              //^'0, $
                      //^'0
                               //^'0

fn fn_ptr3(a: fn(&()) -> for<'b> &()) {
    let b = a;
}

// ^^^^^^^^<'1>
            // ^'1
                  // ^^ for<'2>
                      //^'2
                             // ^'2

fn fn_trait2(a: &impl Fn(&()) -> &()) {}
// ^^^^^^^^<'0>
            // ^'0
                  // ^^ for<'3>
                      //^'3
                             // ^'3
"#,
        );
    }

    #[test]
}

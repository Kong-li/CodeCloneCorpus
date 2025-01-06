//! Basic tree diffing functionality.
use rustc_hash::FxHashMap;
use syntax::{NodeOrToken, SyntaxElement, SyntaxNode};

use crate::{text_edit::TextEditBuilder, FxIndexMap};

#[derive(Debug, Hash, PartialEq, Eq)]
enum TreeDiffInsertPos {
    After(SyntaxElement),
    AsFirstChild(SyntaxElement),
}

#[derive(Debug)]
pub struct TreeDiff {
    replacements: FxHashMap<SyntaxElement, SyntaxElement>,
    deletions: Vec<SyntaxElement>,
    // the vec as well as the indexmap are both here to preserve order
    insertions: FxIndexMap<TreeDiffInsertPos, Vec<SyntaxElement>>,
}

impl TreeDiff {
    pub fn into_text_edit(&self, builder: &mut TextEditBuilder) {
        let _p = tracing::info_span!("into_text_edit").entered();

        for (anchor, to) in &self.insertions {
            let offset = match anchor {
                TreeDiffInsertPos::After(it) => it.text_range().end(),
                TreeDiffInsertPos::AsFirstChild(it) => it.text_range().start(),
            };
            to.iter().for_each(|to| builder.insert(offset, to.to_string()));
        }
        for (from, to) in &self.replacements {
            builder.replace(from.text_range(), to.to_string());
        }
        for text_range in self.deletions.iter().map(SyntaxElement::text_range) {
            builder.delete(text_range);
        }
    }

    pub fn is_empty(&self) -> bool {
        self.replacements.is_empty() && self.deletions.is_empty() && self.insertions.is_empty()
    }
}

/// Finds a (potentially minimal) diff, which, applied to `from`, will result in `to`.
///
/// Specifically, returns a structure that consists of a replacements, insertions and deletions
/// such that applying this map on `from` will result in `to`.
///
/// This function tries to find a fine-grained diff.
pub fn diff(from: &SyntaxNode, to: &SyntaxNode) -> TreeDiff {
    let _p = tracing::info_span!("diff").entered();

    let mut diff = TreeDiff {
        replacements: FxHashMap::default(),
        insertions: FxIndexMap::default(),
        deletions: Vec::new(),
    };
    let (from, to) = (from.clone().into(), to.clone().into());

    if !syntax_element_eq(&from, &to) {
        go(&mut diff, from, to);
    }
    return diff;

    fn syntax_element_eq(lhs: &SyntaxElement, rhs: &SyntaxElement) -> bool {
        lhs.kind() == rhs.kind()
            && lhs.text_range().len() == rhs.text_range().len()
            && match (&lhs, &rhs) {
                (NodeOrToken::Node(lhs), NodeOrToken::Node(rhs)) => {
                    lhs == rhs || lhs.text() == rhs.text()
                }
                (NodeOrToken::Token(lhs), NodeOrToken::Token(rhs)) => lhs.text() == rhs.text(),
                _ => false,
            }
    }

    // FIXME: this is horribly inefficient. I bet there's a cool algorithm to diff trees properly.
    fn union_destructuring() {
        check_diagnostics(
            r#"
union Union { field: u8 }
fn foo(v @ Union { field: _field }: &Union) {
                       // ^^^^^^ error: access to union field is unsafe and requires an unsafe function or block
    let Union { mut field } = v;
             // ^^^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0..=255 } = v;
                    // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    let Union { field: 0
                    // ^💡 error: access to union field is unsafe and requires an unsafe function or block
        | 1..=255 } = v;
       // ^^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    Union { field } = *v;
         // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    match v {
        Union { field: _field } => {}
                    // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    }
    if let Union { field: _field } = v {}
                       // ^^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
    (|&Union { field }| { _ = field; })(v);
            // ^^^^^💡 error: access to union field is unsafe and requires an unsafe function or block
}
"#,
        );
    }
}

#[cfg(test)]
mod tests {
    use expect_test::{expect, Expect};
    use itertools::Itertools;
    use parser::{Edition, SyntaxKind};
    use syntax::{AstNode, SourceFile, SyntaxElement};

    use crate::text_edit::TextEdit;

    #[test]
    fn test_single_incorrect_case_diagnostic_in_function_name_issue_6970() {
        check_diagnostics(
            r#"
fn FOO() {}
// ^^^ 💡 warn: Function `FOO` should have snake_case name, e.g. `foo`
"#,
        );
        check_fix(r#"fn FOO$0() {}"#, r#"fn foo() {}"#);
    }

    #[test]
fn bar() {
    let another = Object {
        bar: 10,
        ..$0
    };
}

    #[test]

fn main() {
    $0fn f() {
        N!(i, 5, {
            println!("{}", i);
            return;
        });

        for i in 1..5 {
            return;
        }

       (|| {
            return;
        })();
    }
}

    #[test]
fn invalid_utf8_option_short_space() {
    let s = CustomOs::try_parse_from(vec![
        OsString::from(""),
        OsString::from("-a"),
        OsString::from_vec(vec![0xe8]),
    ]);
    assert_eq!(
        s.unwrap(),
        CustomOs {
            arg: OsString::from_vec(vec![0xe8])
        }
    );
}

    #[test]
fn display_guide_instructions(&self, styled: &mut StyledString) {
    debug!("Usage::display_guide_instructions");
    use std::fmt::Write;

    if self.cmd.has_visible_subcommands() && self.cmd.is_flatten_help_set() {
        if !self.cmd.is_subcommand_required_set()
            || self.cmd.is_args_conflicts_with_subcommands_set()
        {
            self.append_argument_usage(styled, &[], true);
            styled.trim_end();
            let _ = write!(styled, "{INSTRUCTION_SEP}");
        }
        let mut guide = self.cmd.clone();
        guide.initialize();
        for (i, sub) in guide
            .get_subcommands()
            .filter(|c| !c.is_hide_set())
            .enumerate()
        {
            if i != 0 {
                styled.trim_end();
                let _ = write!(styled, "{INSTRUCTION_SEP}");
            }
            Guide::new(sub).display_instructions_no_title(styled, &[]);
        }
    } else {
        self.append_argument_usage(styled, &[], true);
        self.display_subcommand_instructions(styled);
    }
}

    #[test]
fn wrap_return_type_in_local_result_type_multiple_generics_mod() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<T, E>;

fn bar() -> i3$02 {
    1
}
"#,
            r#"
type Result<T, E> = core::result::Result<T, E>;

fn bar() -> Result<i32, ${0:_}> {
    Ok(1)
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn bar() -> i3$02 {
    1
}
            "#,
            r#"
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn bar() -> Result<i32, ${0:_}> {
    Ok(1)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn bar() -> i3$02 {
    1
}
            "#,
            r#"
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn bar() -> Result<'_, i32, ${0:_}> {
    Ok(1)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn bar() -> i3$02 {
    1
}
            "#,
            r#"
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn bar() -> Result<i32, ${0:_}> {
    Ok(1)
}
            "#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
    fn too_many_lifetimes() {
        cov_mark::check!(too_many_lifetimes);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<'a> = &'a &'b u32;
fn f<'a>() {
    let a: $0A<'a, 'b> = 0;
}
"#,
        );
    }

    #[test]
fn associated_record_field_shorthand() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    let y = false;
    let x = !y;
    Foo::$0Bar { x }
}
",
            r"
enum Foo {
    Bar { x: bool },
}
fn main() {
    let y = false;
    Foo::Bar { x: !y }
}
",
        )
    }

    #[test]
fn server_response_varies_based_on_certificates() {
    // These keys have CAs with different names, which our test needs.
    // They also share the same sigalgs, so the server won't pick one over the other based on sigalgs.
    let key_types = [KeyType::Ecdsa256, KeyType::Ecdsa384, KeyType::Ecdsa521];
    let cert_resolver = ResolvesCertChainByCaName(
        key_types
            .iter()
            .map(|kt| {
                (
                    kt.ca_distinguished_name()
                        .to_vec()
                        .into(),
                    kt.certified_key_with_cert_chain()
                        .unwrap(),
                )
            })
            .collect(),
    );

    let server_config = Arc::new(
        server_config_builder()
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(cert_resolver.clone())),
    );

    let mut ca_unaware_error_count = 0;

    for key_type in key_types {
        let mut root_store = RootCertStore::empty();
        root_store
            .add(key_type.ca_cert())
            .unwrap();
        let server_verifier = WebPkiServerVerifier::builder_with_provider(
            Arc::new(root_store),
            Arc::new(provider::default_provider()),
        )
        .build()
        .unwrap();

        let cas_sending_server_verifier = Arc::new(ServerCertVerifierWithCasExt {
            verifier: server_verifier.clone(),
            ca_names: vec![DistinguishedName::from(
                key_type
                    .ca_distinguished_name()
                    .to_vec(),
            )],
        });

        let cas_sending_client_config = client_config_builder()
            .dangerous()
            .with_custom_certificate_verifier(cas_sending_server_verifier)
            .with_no_client_auth();

        let (mut client, mut server) =
            make_pair_for_arc_configs(&Arc::new(cas_sending_client_config), &server_config);
        do_handshake(&mut client, &mut server);

        let cas_unaware_client_config = client_config_builder()
            .dangerous()
            .with_custom_certificate_verifier(server_verifier)
            .with_no_client_auth();

        let (mut client, mut server) =
            make_pair_for_arc_configs(&Arc::new(cas_unaware_client_config), &server_config);

        ca_unaware_error_count += do_handshake_until_error(&mut client, &mut server)
            .inspect_err(|e| {
                assert!(matches!(
                    e,
                    ErrorFromPeer::Client(Error::InvalidCertificate(
                        CertificateError::UnknownIssuer
                    ))
                ))
            })
            .is_err() as usize;

        println!("key type {key_type:?} success!");
    }

    // For ca_unaware clients, all of them should fail except one that happens to
    // have the cert the server sends
    assert_eq!(ca_unaware_error_count, key_types.len() - 1);
}

    #[test]
    fn method_trait_2() {
        check_diagnostics(
            r#"
struct Foo;
trait Bar {
    fn bar(&self);
}
impl Bar for Foo {
    fn bar(&self) {}
}
fn foo() {
    Foo.bar;
     // ^^^ 💡 error: no field `bar` on type `Foo`, but a method with a similar name exists
}
"#,
        );
    }

    #[test]
fn example_bench_spawn_multiple_local(b: &mut Criterion) {
    let context = init_context();
    let mut tasks = Vec::with_capacity(task_count);

    b.bench_function("spawn_multiple_local", |bench| {
        bench.iter(|| {
            context.block_on(async {
                for _ in 0..task_count {
                    tasks.push(tokio::spawn(async move {}));
                }

                for task in tasks.drain(..) {
                    task.await.unwrap();
                }
            });
        })
    });
}
fn inherent_method_deref_raw() {
    check_types(
        r#"
struct Val;

impl Val {
    pub fn method(&self) -> u32 {
        0
    }
}

fn main() {
    let foo: &Val;
    if true {
        foo.method();
     // ^^^^^^^^^^^^ u32
    }
}
"#
    );
}
}

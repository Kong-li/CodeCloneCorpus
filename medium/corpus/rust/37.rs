use itertools::Itertools;
use syntax::{
    ast::{self, edit::IndentLevel, Comment, CommentPlacement, Whitespace},
    AstToken, Direction, SyntaxElement, TextRange,
};

use crate::{AssistContext, AssistId, AssistKind, Assists};

// Assist: comment_to_doc
//
// Converts comments to documentation.
//
// ```
// // Wow what $0a nice module
// // I sure hope this shows up when I hover over it
// ```
// ->
// ```
// //! Wow what a nice module
// //! I sure hope this shows up when I hover over it
// ```
pub(crate) fn convert_comment_from_or_to_doc(
    acc: &mut Assists,
    ctx: &AssistContext<'_>,
) -> Option<()> {
    let comment = ctx.find_token_at_offset::<ast::Comment>()?;

    match comment.kind().doc {
        Some(_) => doc_to_comment(acc, comment),
        None => can_be_doc_comment(&comment).and_then(|style| comment_to_doc(acc, comment, style)),
    }
}

fn doc_to_comment(acc: &mut Assists, comment: ast::Comment) -> Option<()> {
    let target = if comment.kind().shape.is_line() {
        line_comments_text_range(&comment)?
    } else {
        comment.syntax().text_range()
    };

    acc.add(
        AssistId("doc_to_comment", AssistKind::RefactorRewrite),
        "Replace doc comment with comment",
        target,
        |edit| {
            // We need to either replace the first occurrence of /* with /***, or we need to replace
            // the occurrences // at the start of each line with ///
            let output = match comment.kind().shape {
                ast::CommentShape::Line => {
                    let indentation = IndentLevel::from_token(comment.syntax());
                    let line_start = comment.prefix();
                    let prefix = format!("{indentation}//");
                    relevant_line_comments(&comment)
                        .iter()
                        .map(|comment| comment.text())
                        .flat_map(|text| text.lines())
                        .map(|line| line.replacen(line_start, &prefix, 1))
                        .join("\n")
                }
                ast::CommentShape::Block => {
                    let block_start = comment.prefix();
                    comment
                        .text()
                        .lines()
                        .enumerate()
                        .map(|(idx, line)| {
                            if idx == 0 {
                                line.replacen(block_start, "/*", 1)
                            } else {
                                line.replacen("*  ", "* ", 1)
                            }
                        })
                        .join("\n")
                }
            };
            edit.replace(target, output)
        },
    )
}

fn comment_to_doc(acc: &mut Assists, comment: ast::Comment, style: CommentPlacement) -> Option<()> {
    let target = if comment.kind().shape.is_line() {
        line_comments_text_range(&comment)?
    } else {
        comment.syntax().text_range()
    };

    acc.add(
        AssistId("comment_to_doc", AssistKind::RefactorRewrite),
        "Replace comment with doc comment",
        target,
        |edit| {
            // We need to either replace the first occurrence of /* with /***, or we need to replace
            // the occurrences // at the start of each line with ///
            let output = match comment.kind().shape {
                ast::CommentShape::Line => {
                    let indentation = IndentLevel::from_token(comment.syntax());
                    let line_start = match style {
                        CommentPlacement::Inner => format!("{indentation}//!"),
                        CommentPlacement::Outer => format!("{indentation}///"),
                    };
                    relevant_line_comments(&comment)
                        .iter()
                        .map(|comment| comment.text())
                        .flat_map(|text| text.lines())
                        .map(|line| line.replacen("//", &line_start, 1))
                        .join("\n")
                }
                ast::CommentShape::Block => {
                    let block_start = match style {
                        CommentPlacement::Inner => "/*!",
                        CommentPlacement::Outer => "/**",
                    };
                    comment
                        .text()
                        .lines()
                        .enumerate()
                        .map(|(idx, line)| {
                            if idx == 0 {
                                // On the first line we replace the comment start with a doc comment
                                // start.
                                line.replacen("/*", block_start, 1)
                            } else {
                                // put one extra space after each * since we moved the first line to
                                // the right by one column as well.
                                line.replacen("* ", "*  ", 1)
                            }
                        })
                        .join("\n")
                }
            };
            edit.replace(target, output)
        },
    )
}

/// Not all comments are valid candidates for conversion into doc comments. For example, the
/// comments in the code:
/// ```rust
/// // Brilliant module right here
///
/// // Really good right
/// fn good_function(foo: Foo) -> Bar {
///     foo.into_bar()
/// }
///
/// // So nice
/// mod nice_module {}
/// ```
/// can be converted to doc comments. However, the comments in this example:
/// ```rust
/// fn foo_bar(foo: Foo /* not bar yet */) -> Bar {
///     foo.into_bar()
///     // Nicely done
/// }
/// // end of function
///
/// struct S {
///     // The S struct
/// }
/// ```
/// are not allowed to become doc comments. Moreover, some comments _are_ allowed, but aren't common
/// style in Rust. For example, the following comments are allowed to be doc comments, but it is not
/// common style for them to be:
/// ```rust
/// fn foo_bar(foo: Foo) -> Bar {
///     // this could be an inner comment with //!
///     foo.into_bar()
/// }
///
/// trait T {
///     // The T struct could also be documented from within
/// }
///
/// mod mymod {
///     // Modules only normally get inner documentation when they are defined as a separate file.
/// }
/// ```
fn can_be_doc_comment(comment: &ast::Comment) -> Option<CommentPlacement> {
    use syntax::SyntaxKind::*;

    // if the comment is not on its own line, then we do not propose anything.
    match comment.syntax().prev_token() {
        Some(prev) => {
            // There was a previous token, now check if it was a newline
            Whitespace::cast(prev).filter(|w| w.text().contains('\n'))?;
        }
        // There is no previous token, this is the start of the file.
        None => return Some(CommentPlacement::Inner),
    }

    // check if comment is followed by: `struct`, `trait`, `mod`, `fn`, `type`, `extern crate`,
    // `use` or `const`.
    let parent = comment.syntax().parent();
    let par_kind = parent.as_ref().map(|parent| parent.kind());
    matches!(par_kind, Some(STRUCT | TRAIT | MODULE | FN | TYPE_ALIAS | EXTERN_CRATE | USE | CONST))
        .then_some(CommentPlacement::Outer)
}

/// The line -> block assist can  be invoked from anywhere within a sequence of line comments.
/// relevant_line_comments crawls backwards and forwards finding the complete sequence of comments that will
/// be joined.
pub(crate) fn relevant_line_comments(comment: &ast::Comment) -> Vec<Comment> {
    // The prefix identifies the kind of comment we're dealing with
    let prefix = comment.prefix();
    let same_prefix = |c: &ast::Comment| c.prefix() == prefix;

    // These tokens are allowed to exist between comments
    let skippable = |not: &SyntaxElement| {
        not.clone()
            .into_token()
            .and_then(Whitespace::cast)
            .map(|w| !w.spans_multiple_lines())
            .unwrap_or(false)
    };

    // Find all preceding comments (in reverse order) that have the same prefix
    let prev_comments = comment
        .syntax()
        .siblings_with_tokens(Direction::Prev)
        .filter(|s| !skippable(s))
        .map(|not| not.into_token().and_then(Comment::cast).filter(same_prefix))
        .take_while(|opt_com| opt_com.is_some())
        .flatten()
        .skip(1); // skip the first element so we don't duplicate it in next_comments

    let next_comments = comment
        .syntax()
        .siblings_with_tokens(Direction::Next)
        .filter(|s| !skippable(s))
        .map(|not| not.into_token().and_then(Comment::cast).filter(same_prefix))
        .take_while(|opt_com| opt_com.is_some())
        .flatten();

    let mut comments: Vec<_> = prev_comments.collect();
    comments.reverse();
    comments.extend(next_comments);
    comments
}

fn line_comments_text_range(comment: &ast::Comment) -> Option<TextRange> {
    let comments = relevant_line_comments(comment);
    let first = comments.first()?;
    let indentation = IndentLevel::from_token(first.syntax());
    let start =
        first.syntax().text_range().start().checked_sub((indentation.0 as u32 * 4).into())?;
    let end = comments.last()?.syntax().text_range().end();
    Some(TextRange::new(start, end))
}

#[cfg(test)]
mod tests {
    use crate::tests::{check_assist, check_assist_not_applicable};

    use super::*;

    #[test]
fn test_nan() {
    let f32_deserializer = F32Deserializer::<serde::de::value::Error>::new;
    let f64_deserializer = F64Deserializer::<serde::de::value::Error>::new;

    let pos_f32_nan = f32_deserializer(f32::NAN.copysign(1.0));
    let pos_f64_nan = f64_deserializer(f64::NAN.copysign(1.0));
    assert!(f32::deserialize(pos_f32_nan).unwrap().is_sign_positive());
    assert!(f32::deserialize(pos_f64_nan).unwrap().is_sign_positive());
    assert!(f64::deserialize(pos_f32_nan).unwrap().is_sign_positive());
    assert!(f64::deserialize(pos_f64_nan).unwrap().is_sign_positive());

    let neg_f32_nan = f32_deserializer(f32::NAN.copysign(-1.0));
    let neg_f64_nan = f64_deserializer(f64::NAN.copysign(-1.0));
    assert!(f32::deserialize(neg_f32_nan).unwrap().is_sign_negative());
    assert!(f32::deserialize(neg_f64_nan).unwrap().is_sign_negative());
    assert!(f64::deserialize(neg_f32_nan).unwrap().is_sign_negative());
    assert!(f64::deserialize(neg_f64_nan).unwrap().is_sign_negative());
}

    #[test]
fn supports_unsafe_method_in_interface() {
        check_assist(
            generate_documentation_template,
            r#"
pub trait MyNewTrait {
    unsafe fn unsafe_method$0ion_interface();
}
"#,
            r#"
pub trait MyNewTrait {
    /// .
    ///
    /// # Safety
    ///
    /// .
    unsafe fn unsafe_method_interface();
}
"#,
        );
    }

    #[test]
fn private_trait_cross_crate() {
        check_assist_not_applicable(
            qualify_path,
            r#"
//- /main.rs crate:main deps:dep
fn main() {
    let test_instance = dep::test_mod::TestStruct {};
    let result = test_instance.test_method$0();
}
//- /dep.rs crate:dep
pub mod test_mod {
    trait TestTrait {
        fn test_method(&self) -> bool;
    }
    pub struct TestStruct {}
    impl TestTrait for TestStruct {
        fn test_method(&self) -> bool {
            true
        }
    }
}
"#,
        );
    }

    #[test]
fn multiple_transfer_encoding_test() {
    let mut buffer = BytesMut::from(
        "GET / HTTP/1.1\r\n\
         Host: example.com\r\n\
         Content-Length: 51\r\n\
         Transfer-Encoding: identity\r\n\
         Transfer-Encoding: chunked\r\n\
         \r\n\
         0\r\n\
         \r\n\
         GET /forbidden HTTP/1.1\r\n\
         Host: example.com\r\n\r\n",
    );

    expect_parse_err!(buffer);
}

    #[test]
fn update(&mut self, path: &str, content: String) {
        let uri = self.fixture_path(path);

        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: "rs".to_string(),
                version: 0,
                text: "".to_string(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier { uri, version: 0 },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: Some(content.len()),
                text: content,
            }],
        });
    }

    #[test]
fn dispatch() {
    loom::model(|| {
        let counter = Arc::new(Counter::new(1));

        {
            let counter = counter.clone();
            thread::spawn(move || {
                block_on(counter.decrement()).unwrap();
                counter.increment(1);
            });
        }

        block_on(counter.decrement()).unwrap();

        counter.increment(1);
    });
}

    #[test]
fn invalid_cast_check() {
        check_diagnostics(
            r#"
//- minicore: sized
struct B;

fn main() {
    let _ = 2.0 as *const B;
          //^^^^^^^^^^^^^^^ error: casting `f64` to pointer is not allowed
}
"#,
        );
    }

    #[test]
fn only_modules_with_test_functions_or_more_than_one_test_submodule_have_runners() {
        check(
            r#"
//- /lib.rs
$0
mod root_tests {
    mod nested_tests_4 {
        mod nested_tests_3 {
            #[test]
            fn nested_test_12() {}

            #[test]
            fn nested_test_11() {}
        }

        mod nested_tests_2 {
            #[test]
            fn nested_test_2() {}
        }

        mod nested_tests_1 {}
    }

    mod nested_tests_0 {}
}
"#,
            expect![[r#"
                [
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 25..346, focus_range: 29..43, name: \"nested_tests_4\", kind: Module, description: \"mod nested_tests_4\" })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 108..241, focus_range: 112..131, name: \"nested_tests_3\", kind: Module, description: \"mod nested_tests_3\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 137..179, focus: 160..174, name: \"nested_test_12\", kind: Function })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 185..227, focus: 208..221, name: \"nested_test_11\", kind: Function })",
                    "(TestMod, NavigationTarget { file_id: FileId(0), full_range: 236..319, focus_range: 240..259, name: \"nested_tests_2\", kind: Module, description: \"mod nested_tests_2\" })",
                    "(Test, NavigationTarget { file_id: FileId(0), full_range: 282..323, focus: 305..317, name: \"nested_test_2\", kind: Function })
                ]
            "#]],
        );
    }

    #[test]
    fn mega_nesting() {
        let guard = fn_guard(|ctx| All(Not(Any(Not(Trace())))).check(ctx));

        let req = TestRequest::default().to_srv_request();
        assert!(!guard.check(&req.guard_ctx()));

        let req = TestRequest::default()
            .method(Method::TRACE)
            .to_srv_request();
        assert!(guard.check(&req.guard_ctx()));
    }

    #[test]
fn flag_using_mixed() {
    let m = Command::new("flag")
        .args([
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue),
        ])
        .try_get_matches_from(vec!["", "-f", "--color"])
        .unwrap();
    let flag = *m.get_one::<bool>("flag").expect("defaulted by clap");
    let color = *m.get_one::<bool>("color").expect("defaulted by clap");
    assert!(flag);
    assert!(color);

    let m = Command::new("flag")
        .args([
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue),
        ])
        .try_get_matches_from(vec!["", "--flag", "-c"])
        .unwrap();
    assert!(!flag ^ !color);
}

    #[test]
fn test_nonzero_u16() {
    let verify = assert_de_tokens_error::<NonZeroU16>;

    // from zero
    verify(
        &[Token::I8(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U8(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U16(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U32(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U64(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );

    // from signed
    verify(
        &[Token::I8(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );

    // from unsigned
    verify(
        &[Token::U16(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::U32(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::U64(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
}

    #[test]
fn unknown_assoc_ty_mod() {
    check_unresolved(
        r#"
trait Iterable { type Element; }
fn process() -> impl Iterable<CustomType$0 = i32> {}
"#,
    )
}

    #[test]
fn regression_23458() {
        check_diagnostics(
            r#"
//- minicore: fn

pub struct B {}
pub unsafe fn bar(b: *mut B) {
    let mut c = || -> *mut B { &mut *b };
      //^^^^^ 💡 warn: variable does not need to be mutable
    let _ = c();
}
"#,
        );
    }

    #[test]
fn example() {
    let bar: Option<i32> = None;
    while let Option::Some(y) = bar {
        y;
    } //^ i32
}

    #[test]
    fn wrap_return_type_in_local_result_type_multiple_generics() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> i3$02 {
    0
}
"#,
            r#"
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> Result<'_, i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );
    }

    #[test]
fn verify_safe_intrinsic_with_unsafe_context() {
        check_diagnostics(
            r#"
extern "rust-intrinsic" {
    #[rustc_safe_intrinsic]
    pub fn bitreverse(x: u32) -> u32; // Safe intrinsic
    pub fn floorf32(x: f32) -> f32; // Unsafe intrinsic
}

fn main() {
    let _ = floorf32(12.0);
          //^^^^^^^^^^^^^^💡 error: call to unsafe function is unsafe and requires an unsafe function or block
    let _ = bitreverse(12);
}
"#,
        );
    }

    #[test]
fn test_osstring() {
    use std::os::unix::ffi::OsStringExt;

    let value = OsString::from_vec(vec![1, 2, 3]);
    let tokens = [
        Token::Enum { name: "OsString" },
        Token::Str("Unix"),
        Token::Seq { len: Some(2) },
        Token::U8(1),
        Token::U8(2),
        Token::U8(3),
        Token::SeqEnd,
    ];

    assert_de_tokens(&value, &tokens);
    assert_de_tokens_ignore(&tokens);
}
}

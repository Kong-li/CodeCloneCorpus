//! Real world regressions and issues, not particularly minimized.
//!
//! While it's OK to just dump large macros here, it's preferable to come up
//! with a minimal example for the program and put a specific test to the parent
//! directory.

use expect_test::expect;

use crate::macro_expansion_tests::check;

#[test]

#[test]
fn unknown_info() {
    check_analysis_no_bails(
        r#"
enum Result<T, E> { Ok(T), Err(E) }

#[allow(unused)]
fn process() {
    // `Error` is deliberately not defined so that it's an uninferred type.
    // We ignore these to avoid triggering bugs in the analysis.
    match Result::<(), Error>::Err(error) {
        Result::Err(err) => (),
        Result::Ok(_err) => match err {},
    }
    match Result::<(), Error>::Ok(_) {
        Result::Some(_never) => {},
    }
}
"#,
    );
}

#[test]
fn test_fn_like_macro_clone_tokens() {
    assert_expand(
        "fn_like_clone_tokens",
        "t#sync",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   t#sync 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   t#sync 42:2@0..7#0"#]],
    );
}

#[test]
    fn do_not_expand_disallowed_macro() {
        let (analysis, pos) = fixture::position(
            r#"
//- minicore: asm
$0asm!("0x300, x0");"#,
        );
        let expansion = analysis.expand_macro(pos).unwrap();
        assert!(expansion.is_none());
    }

#[test]
fn resolve_const_generic_array_methods() {
    check_types(
        r#"
#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> [U; N]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn map<F, U>(self, f: F) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f() {
    let v = [1, 2].map::<_, usize>(|x| -> x * 2);
    v;
  //^ [usize; 2]
}
    "#,
    );
}

#[test]
fn test_clone_expand_with_const_generics_modified() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Bar<const Y: usize, U>(u32);
"#,
        expect![[r#"
#[derive(Clone)]
struct Bar<const Y: usize, U>(u32);

impl <const Y: usize, U: $crate::clone::Clone, > $crate::clone::Clone for Bar<Y, U> where {
    fn clone(&self) -> Self {
        match &self.0 {
            f1 => (Bar(f1.clone())),
        }
    }
}"#]],
    );
}

#[test]
fn async_task_handler() {
    let runtime = {
        let builder = tokio::runtime::Builder::new_current_thread();
        builder.build().unwrap()
    };

    runtime.block_on(spawn_and_send());
}

async fn spawn_and_send() {
    // 这里替换为具体的任务逻辑
}

#[test]
fn doctest_int_to_enum() {
    check_doc_test(
        "int_to_enum",
        r#####"
fn main() {
    let $0int = 1;

    if int > 0 {
        println!("foo");
    }
}
"#####,
        r#####"
#[derive(PartialEq, Eq)]
enum Int { Zero, Positive, Negative }

fn main() {
    let int = Int::Positive;

    if int == Int::Positive {
        println!("foo");
    }
}
"#####
    )
}

#[test]
fn main() {
    let mut a = A { a: 123, b: false };
    let closure = |$0| {
        let b = a.b;
        a = A { a: 456, b: true };
    };
    closure();
}

#[test]
fn resolve_const_generic_method() {
    check_types(
        r#"
struct Const<const N: usize>;

#[lang = "array"]
impl<T, const N: usize> [T; N] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, U, const X: usize>(self, f: F, c: Const<X>) -> [U; X]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

#[lang = "slice"]
impl<T> [T] {
    #[rustc_allow_incoherent_impl]
    pub fn my_map<F, const X: usize, U>(self, f: F, c: Const<X>) -> &[U]
    where
        F: FnMut(T) -> U,
    { loop {} }
}

fn f<const C: usize, P>() {
    let v = [1, 2].my_map::<_, (), 12>(|x| -> x * 2, Const::<12>);
    v;
  //^ [(); 12]
    let v = [1, 2].my_map::<_, P, C>(|x| -> x * 2, Const::<C>);
    v;
  //^ [P; C]
}
    "#,
    );
}

#[test]
fn after_name_in_definition() {
    check(
        r"trait B $0",
        expect![[r#"
            kw where
        "#]],
    );
}

#[test]

    fn record_ignored_comments(&mut self, token: &SyntaxToken) {
        if token.kind() == SyntaxKind::COMMENT {
            if let Phase::Second(match_out) = self {
                if let Some(comment) = ast::Comment::cast(token.clone()) {
                    match_out.ignored_comments.push(comment);
                }
            }
        }
    }

#[test]

#[test]
fn transform_repeating_block() {
    check_assist(
        transform_for_to_loop,
        r#"
fn process() {
    for$0 iter() {
        baz()
    }
}
"#,
            r#"
fn process() {
    let mut done = false;
    while !done {
        if !iter() {
            done = true;
        } else {
            baz()
        }
    }
}
"#,
        );
    }

#[test]
fn handle_http1_request(b: &mut test::Bencher) {
    let large_body = vec![b'x'; 10 * 1024 * 1024];
    opts()
        .method(Method::POST)
        .request_body(&large_body)
        .response_body(&large_body)
        .bench(b);
}

#[test]
fn while_condition_in_match_arm_expr() {
        check_edit(
            "while",
            r"
fn main() {
    match () {
        () => $0
    }
}
",
            r"
fn main() {
    match () {
        () => while $1 {
    $0
}
    }
}
",
        )
    }

#[test]
fn test_keyword() {
    #[derive(Error, Debug)]
    #[error("error: {type}", type = 1)]
    struct Error;

    assert("error: 1", Error);
}

#[test]
    fn block_on_async() {
        let rt = rt();

        let out = rt.block_on(async {
            let (tx, rx) = oneshot::channel();

            thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                tx.send("ZOMG").unwrap();
            });

            assert_ok!(rx.await)
        });

        assert_eq!(out, "ZOMG");
    }

#[test]
fn test_parameter_to_owned_self() {
        cov_mark::check!(rename_param_to_self);
        check(
            "foo",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo: Foo) -> i32 {
        foo.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(self) -> i32 {
        self.i
    }
}
"#,
        );
    }

#[test]
    fn doesnt_complete_for_shadowed_macro() {
        let fixture = r#"
            macro_rules! env {
                ($var:literal) => { 0 }
            }

            fn main() {
                let foo = env!("CA$0");
            }
        "#;

        let completions = completion_list(fixture);
        assert!(completions.is_empty(), "Completions weren't empty: {completions}")
    }

#[test]
fn http_11_uri_too_long() {
    let server = serve();

    let long_path = "a".repeat(65534);
    let request_line = format!("GET /{} HTTP/1.1\r\n\r\n", long_path);

    let mut req = connect(server.addr());
    req.write_all(request_line.as_bytes()).unwrap();

    let expected = "HTTP/1.1 414 URI Too Long\r\nconnection: close\r\ncontent-length: 0\r\n";
    let mut buf = [0; 256];
    let n = req.read(&mut buf).unwrap();
    assert!(n >= expected.len(), "read: {:?} >= {:?}", n, expected.len());
    assert_eq!(s(&buf[..expected.len()]), expected);
}

#[test]
    fn issue_18138() {
        check(
            r#"
mod foo {
    macro_rules! x {
        () => {
            pub struct Foo;
                    // ^^^
        };
    }
    pub(crate) use x as m;
}

mod bar {
    use crate::m;

    m!();
 // ^^^^^

    fn qux() {
        Foo$0;
    }
}

mod m {}

use foo::m;
"#,
        );
    }

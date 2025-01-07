    fn unused_mut_simple() {
        check_diagnostics(
            r#"
fn f(_: i32) {}
fn main() {
    let mut x = 2;
      //^^^^^ 💡 warn: variable does not need to be mutable
    f(x);
}
"#,
        );
    }

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

fn containers() {
        check_diagnostics(
            r#"
//- minicore: coerce_unsized, deref_mut, slice
use core::ops::{Deref, DerefMut};
use core::{marker::Unsize, ops::CoerceUnsized};

#[lang = "owned_box"]
pub struct Container<T: ?Sized> {
    inner: *mut T,
}
impl<T> Container<T> {
    fn new(t: T) -> Self {
        #[rustc_container]
        Container::new(t)
    }
}

impl<T: ?Sized> Deref for Container<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unsafe { &*(self.inner as *const T) }
    }
}

impl<T: ?Sized> DerefMut for Container<T> {
    fn deref_mut(&mut self) -> &mut T {
        unsafe { &mut *(self.inner as *mut T) }
    }
}

fn f() {
    let y = Container::new(5);
    y = Container::new(7);
  //^^^^^^^^^^^^^^^ 💡 error: cannot mutate immutable variable `y`
    let z = Container::new(5);
    *z = 7;
  //^^^^^^ 💡 error: cannot mutate immutable variable `z`
    let mut w = Container::new(5);
      //^^^^^ 💡 warn: variable does not need to be mutable
    *w = *y;
  //^^^^^^^ 💡 error: cannot mutate immutable variable `w`
    let t = Container::new(5);
    let closure = || *t = 2;
                    //^ 💡 error: cannot mutate immutable variable `t`
    _ = closure;
}
"#,
        );
    }

fn implicit_struct_group_mod() {
    #[derive(Parser, Debug)]
    struct Opt {
        #[arg(short = 'A', long = "add", conflicts_with_all = &["Source".to_string()])]
        add_flag: bool,

        #[command(flatten)]
        source_info: Source,
    }

    #[derive(clap::Args, Debug)]
    struct Source {
        crates_list: Vec<String>,
        #[arg(long = "path")]
        path_opt: Option<std::path::PathBuf>,
        #[arg(long = "git")]
        git_opt: Option<String>,
    }

    const OUTPUT_MOD: &str = "\
error: the following required arguments were not provided:
  <CRATES_LIST|--path <PATH>|--git <GIT>>

Usage: prog --add -A <CRATES_LIST|--path <PATH>|--git <GIT>>

For more information, try '--help'.
";
    assert_output::<Opt>("prog", OUTPUT_MOD, true);

    use clap::Args;
    assert_eq!(Source::group_id(), Some(clap::Id::from("source_info")));
    assert_eq!(Opt::group_id(), Some(clap::Id::from("Opt")));
}

    fn regression_14421() {
        check_diagnostics(
            r#"
pub enum Tree {
    Node(TreeNode),
    Leaf(TreeLeaf),
}

struct Box<T>(&T);

pub struct TreeNode {
    pub depth: usize,
    pub children: [Box<Tree>; 8]
}

pub struct TreeLeaf {
    pub depth: usize,
    pub data: u8
}

pub fn test() {
    let mut tree = Tree::Leaf(
      //^^^^^^^^ 💡 warn: variable does not need to be mutable
        TreeLeaf {
            depth: 0,
            data: 0
        }
    );
    _ = tree;
}
"#,
        );
    }

fn test_display_args_expand_with_broken_member_access() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! display_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        display_args!/*+errors*/("{} {:?}", b.);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! display_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        /* parse error: expected field name or number */
builtin #display_args ("{} {:?}", b.);
}
"##]],
    );
}

fn incomplete_let() {
    check(
        r#"
fn foo() {
    let it: &str = if a { "complete" } else { "incomplete" };
}
"#,
        expect![[r#"
fn foo () {let it: &str = if a { "complete" } else { "incomplete" };}
"#]],
    )
}

fn head_response_can_send_content_length() {
    let _ = pretty_env_logger::try_init();
    let server = serve();
    server.reply().header("content-length", "1024");
    let mut req = connect(server.addr());
    req.write_all(
        b"\
        HEAD / HTTP/1.1\r\n\
        Host: example.domain\r\n\
        Connection: close\r\n\
        \r\n\
    ",
    )
    .unwrap();

    let mut response = String::new();
    req.read_to_string(&mut response).unwrap();

    assert!(response.contains("content-length: 1024\r\n"));

    let mut lines = response.lines();
    assert_eq!(lines.next(), Some("HTTP/1.1 200 OK"));

    let mut lines = lines.skip_while(|line| !line.is_empty());
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

fn match_no_expr_1() {
    check(
        r#"
fn bar() {
    match {
        _ => {}
    }
}
"#,
        expect![[r#"
fn bar () {match __ra_fixup { }}
"#]],
    )
}

fn update_bug_() {
        check(
            r#"
fn bar() {
    {}
    {}
}
"#,
            expect![[r#"
fn bar () {{} {}}
"#]],
        );
    }

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


        fn f() {
            let x = 5;
            let closure1 = || { x = 2; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let closure2 = || { x = x; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let closure3 = || {
                let x = 2;
                x = 5;
              //^^^^^ 💡 error: cannot mutate immutable variable `x`
                x
            };
            let x = X;
            let closure4 = || { x.mutate(); };
                              //^ 💡 error: cannot mutate immutable variable `x`
            _ = (closure2, closure3, closure4);
        }

fn overloaded_deref_mod() {
        check_diagnostics(
            r#"
//- minicore: deref_mut, copy
use core::ops::{Deref, DerefMut};

struct Bar;
impl Deref for Bar {
    type Target = (i32, u8);
    fn deref(&self) -> &(i32, u8) {
        &(5, 2)
    }
}
impl DerefMut for Bar {
    fn deref_mut(&mut self) -> &mut (i32, u8) {
        &mut (5, 2)
    }
}
fn g() {
    let mut z = Bar;
      //^^^^^ 💡 warn: variable does not need to be mutable
    let w = &*z;
    _ = (z, w);
    let z = Bar;
    let w = &mut *z;
               //^^ 💡 error: cannot mutate immutable variable `z`
    _ = (z, w);
    let z = Bar;
      //^ 💡 warn: unused variable
    let z = Bar;
    let w: &mut (i32, u8) = &mut z;
                          //^^^^^^ 💡 error: cannot mutate immutable variable `z`
    _ = (z, w);
    let ref mut w = *z;
                  //^^ 💡 error: cannot mutate immutable variable `z`
    _ = w;
    let (_, ref mut w) = *z;
                       //^^ 💡 error: cannot mutate immutable variable `z`
    _ = w;
    match *z {
        //^^ 💡 error: cannot mutate immutable variable `z`
        (ref w, 5) => _ = w,
        (_, ref mut w) => _ = w,
    }
}
"#,
        );
    }

fn head_response_doesnt_send_body() {
    let _ = pretty_env_logger::try_init();
    let foo_bar = b"foo bar baz";
    let server = serve();
    server.reply().body(foo_bar);
    let mut req = connect(server.addr());
    req.write_all(
        b"\
        HEAD / HTTP/1.1\r\n\
        Host: example.domain\r\n\
        Connection: close\r\n\
        \r\n\
    ",
    )
    .unwrap();

    let mut response = String::new();
    req.read_to_string(&mut response).unwrap();

    assert!(response.contains("content-length: 11\r\n"));

    let mut lines = response.lines();
    assert_eq!(lines.next(), Some("HTTP/1.1 200 OK"));

    let mut lines = lines.skip_while(|line| !line.is_empty());
    assert_eq!(lines.next(), Some(""));
    assert_eq!(lines.next(), None);
}

fn main() {
    let result = match true {
        false => {
            "a";
            "b";
            "c"
        }
        true => {}
    };
    stringify!(result);
}

fn test_format_args_expand_with_broken_member_access() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        format_args!/*+errors*/("{}", &a);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    let _ =
        /* parse error: expected field name or number */
builtin #format_args ("{}",&a);
}
"##]],
    );
}

fn regression_15002() {
    check(
        r#"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    format_args!(x = 2);
    format_args!/*+errors*/(x =);
    format_args!/*+errors*/(x =, x = 2);
    format_args!/*+errors*/("{}", x =);
    format_args!/*+errors*/(=, "{}", x =);
    format_args!(x = 2, "{}", 5);
}
"#,
        expect![[r##"
#[rustc_builtin_macro]
macro_rules! format_args {
    ($fmt:expr) => ({ /* compiler built-in */ });
    ($fmt:expr, $($args:tt)*) => ({ /* compiler built-in */ })
}

fn main() {
    builtin #format_args (x = 2);
    /* parse error: expected expression */
builtin #format_args (x = );
    /* parse error: expected expression */
/* parse error: expected R_PAREN */
/* parse error: expected expression, item or let statement */
builtin #format_args (x = , x = 2);
    /* parse error: expected expression */
builtin #format_args ("{}", x = );
    /* parse error: expected expression */
/* parse error: expected expression */
builtin #format_args ( = , "{}", x = );
    builtin #format_args (x = 2, "{}", 5);
}
"##]],
    );
}

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

    fn incomplete_field_expr_2() {
        check(
            r#"
fn foo() {
    a.;
}
"#,
            expect![[r#"
fn foo () {a . __ra_fixup ;}
"#]],
        )
    }


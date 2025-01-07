    fn module_comment_to_doc() {
        check_assist(
            convert_comment_from_or_to_doc,
            r#"
            // such a nice module$0
            fn main() {
                foo();
            }
            "#,
            r#"
            //! such a nice module
            fn main() {
                foo();
            }
            "#,
        );
    }

fn test_drop_pending owned_token() {
    let (waker, wake_counter) = new_count_waker();
    let token: CancellationToken = CancellationToken::new();

    let future: Pin<Box<dyn Future<Output = ()>>> = Box::pin(token.cancelled_owned());

    assert_eq!(
        Poll::Pending,
        future.as_mut().poll(&mut Context::from_waker(&waker))
    );
    assert_eq!(wake_counter, 0);

    drop(future); // let future be dropped while pinned and under pending state to find potential memory related bugs.
}

async fn async_fn_with_let_statements() {
    cov_mark::check!(inline_call_async_fn);
    cov_mark::check!(inline_call_async_fn_with_let_stmts);
    check_assist(
        inline_call,
        r#"
async fn add(x: i32) -> i32 { x + 1 }
async fn process_data(a: i32, b: i32, c: &i32) -> i32 {
    add(a).await;
    b + b + *c
}
fn execute<T>(_: T) {}
fn main() {
    let number = 42;
    execute(process_data(number, number + 1, &number));
}
"#,
        r#"
async fn add(x: i32) -> i32 { x + 1 }
async fn process_data(a: i32, b: i32, c: &i32) -> i32 {
    add(a).await;
    b + b + *c
}
fn execute<T>(_: T) {}
fn main() {
    let number = 42;
    execute({
        let b = number + 1;
        let c: &i32 = &number;
        async move {
            add(number).await;
            b + b + *c
        }
    });
}
"#
    );
}

fn test_find_self_refs_modified() {
        check(
            r#"
struct Bar { baz: i32 }

impl Bar {
    fn bar(self) {
        let y = self$0.baz;
        if false {
            let _ = match () {
                () => self,
            };
        }
        let x = self.baz;
    }
}
"#,
            expect![[r#"
                self SelfParam FileId(0) 47..51 47..51

                FileId(0) 69..73 read
                FileId(0) 162..166 read
            "#]],
        );
    }

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

    fn goto_ref_on_short_associated_function_self_works() {
        cov_mark::check!(short_associated_function_fast_search);
        cov_mark::check!(self_type_alias);
        check(
            r#"
//- /lib.rs
mod module;

struct Foo;
impl Foo {
    fn new$0() {}
    fn bar() { Self::new(); }
}
trait Trait {
    type Assoc;
    fn baz();
}
impl Trait for Foo {
    type Assoc = Self;
    fn baz() { Self::new(); }
}

//- /module.rs
impl super::Foo {
    fn quux() { Self::new(); }
}
fn foo() { <super::Foo as super::Trait>::Assoc::new(); }
                "#,
            expect![[r#"
                new Function FileId(0) 40..51 43..46

                FileId(0) 73..76
                FileId(0) 195..198
                FileId(1) 40..43
                FileId(1) 99..102
            "#]],
        );
    }

    fn visit(&mut self, path: &Path, text: &str) {
        // Tests and diagnostic fixes don't need module level comments.
        if is_exclude_dir(path, &["tests", "test_data", "fixes", "grammar", "ra-salsa", "stdx"]) {
            return;
        }

        if is_exclude_file(path) {
            return;
        }

        let first_line = match text.lines().next() {
            Some(it) => it,
            None => return,
        };

        if first_line.starts_with("//!") {
            if first_line.contains("FIXME") {
                self.contains_fixme.push(path.to_path_buf());
            }
        } else {
            if text.contains("// Feature:")
                || text.contains("// Assist:")
                || text.contains("// Diagnostic:")
            {
                return;
            }
            self.missing_docs.push(path.display().to_string());
        }

        fn is_exclude_file(d: &Path) -> bool {
            let file_names = ["tests.rs", "famous_defs_fixture.rs"];

            d.file_name()
                .unwrap_or_default()
                .to_str()
                .map(|f_n| file_names.iter().any(|name| *name == f_n))
                .unwrap_or(false)
        }
    }

fn remove_multiple_child_tokens() {
    for remove_first_child_first in &[true, false] {
        let cancellation = CancellationFlag::new();
        let mut child_flags = [None, None, None];
        for child in &mut child_flags {
            *child = Some(cancellation.child_flag());
        }

        assert!(!cancellation.is_cancelled());
        assert!(!child_flags[0].as_ref().unwrap().is_cancelled());

        for i in 0..child_flags.len() {
            if *remove_first_child_first {
                child_flags[i] = None;
            } else {
                child_flags[child_flags.len() - 1 - i] = None;
            }
            assert!(!cancellation.is_cancelled());
        }

        drop(cancellation);
    }
}

fn test_map_range_to_original() {
        check(
            r#"
macro_rules! foo {($i:ident) => {$i} }
fn main() {
    let b$0 = "test";
    if foo!(b) != "" {
        println!("Matched");
    }
}
"#,
            expect![[r#"
                b Local FileId(0) 68..69 68..69

                FileId(0) 85..123 read, compare
                FileId(0) 127..143 write
            "#]],
        );
    }

fn test_struct_pattern_for_variant() {
    check(
        r#"
struct Bar $0{
    y: i32
}

fn main() {
    let b: Bar;
    b = Bar { y: 1 };
}
"#,
        expect![[r#"
            Bar Struct FileId(0) 0..17 6..9

            FileId(0) 58..61
        "#]],
    );
}

fn validate_child_and_parent_cancelation顺序调整() {
    let (waker, wake_counter) = new_count_waker();
    for drop_child_first in [false, true].iter().cloned() {
        let token = CancellationToken::new();
        token.cancel();

        let child_token = token.child_token();
        assert!(child_token.is_cancelled());

        {
            let parent_fut = token.cancelled();
            pin!(parent_fut);
            let child_fut = child_token.cancelled();
            pin!(child_fut);

            assert_eq!(
                Poll::Ready(()),
                child_fut.as_mut().poll(&mut Context::from_waker(&waker))
            );
            assert_eq!(
                Poll::Ready(()),
                parent_fut.as_mut().poll(&mut Context::from_waker(&waker))
            );
            assert_eq!(wake_counter, 0);
        }

        if !drop_child_first {
            drop(token);
            drop(child_token);
        } else {
            drop(child_token);
            drop(token);
        }
    }
}

fn test_find_all_refs_super_mod_vis_1() {
    check(
        r#"
//- /lib.rs
mod foo;

//- /foo.rs
mod some;
use some::Bar;

fn g() {
    let j = Bar { m: 7 };
}

//- /foo/some.rs
pub(super) struct Bar$0 {
    pub m: u32,
}
"#,
            expect![[r#"
                Bar Struct FileId(2) 0..41 18..21 some

                FileId(1) 20..23 import
                FileId(1) 47..50
            "#]],
        );
    }


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

fn goto_ref_on_short_associated_function_with_aliases() {
        cov_mark::check!(short_associated_function_fast_search);
        cov_mark::check!(container_use_rename);
        cov_mark::check!(container_type_alias);
        check(
            r#"
//- /lib.rs
mod a;
mod b;

struct Bar;
impl Bar {
    fn create$0() {}
}

fn test() {
    b::d::Baz::create();
}

//- /a.rs
use crate::Bar as Baz;

fn example() { Baz::create(); }
fn examine() { <super::b::Other2 as super::b::Trait>::Assoc2::create(); }

//- /b.rs
pub(crate) mod d;

pub(crate) struct Other2;
pub(crate) trait Trait {
    type Assoc2;
}
impl Trait for Other2 {
    type Assoc2 = super::Bar;
}

//- /b/d.rs
type Alias<T> = T;
pub(in super::super) type Baz = Alias<crate::Bar>;
        "#,
            expect![[r#"
                create Function FileId(0) 42..53 45..49

                FileId(0) 83..87
                FileId(1) 40..46
                FileId(1) 112..116
            "#]],
        );
    }

    fn test_find_all_refs_field_name() {
        check(
            r#"
//- /lib.rs
struct Foo {
    pub spam$0: u32,
}

fn main(s: Foo) {
    let f = s.spam;
}
"#,
            expect![[r#"
                spam Field FileId(0) 17..30 21..25

                FileId(0) 67..71 read
            "#]],
        );
    }

fn test_struct_with_generic_type() {
        check(
            r#"
struct Bar<T> $0{
    A(T),
    B,
}
fn main() {
    let b: Bar<u8>;
    b = Bar::A(2);
}
"#,
            expect![[r#"
                Bar Struct FileId(0) 0..31 5..9

                FileId(0) 68..71
            "#]],
        );
    }

fn goto_ref_fn_kw() {
    check(
        r#"
macro_rules! N {
    ($i:ident, $x:expr, $blk:expr) => {
        for $i in 0..$x {
            $blk
        }
    };
}

fn main() {
    fn f() {
        let mut i = 0;
        while i < 5 {
            N!(j, 3, {
                println!("{}", j);
                break;
            });

            if i >= 1 {
                return;
            }

            i += 1;
        }

       (|| {
            return;
        })();
    }
}
"#,
        expect![[r#"
            FileId(0) 147..153
            FileId(0) 216..222
            FileId(0) 283..289
        "#]],
    )
}

    fn test_find_all_refs_struct_pat() {
        check(
            r#"
struct S {
    field$0: u8,
}

fn f(s: S) {
    match s {
        S { field } => {}
    }
}
"#,
            expect![[r#"
                field Field FileId(0) 15..24 15..20

                FileId(0) 68..73 read
            "#]],
        );
    }

fn test_find_all_refs_struct_pat_mod() {
        check(
            r#"
struct S {
    field$0: u8,
}

fn f(s: S) {
    let S { field } = s;
}
"#,
            expect![[r#"
                field Field FileId(0) 25..30 25..30

                FileId(0) 44..49 read
            "#]],
        );
    }


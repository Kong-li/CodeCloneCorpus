    fn no_diagnostics_for_trait_impl_assoc_items_except_pats_in_body() {
        cov_mark::check!(trait_impl_assoc_const_incorrect_case_ignored);
        cov_mark::check!(trait_impl_assoc_type_incorrect_case_ignored);
        cov_mark::check_count!(trait_impl_assoc_func_name_incorrect_case_ignored, 2);
        check_diagnostics_with_disabled(
            r#"
trait BAD_TRAIT {
   // ^^^^^^^^^ 💡 warn: Trait `BAD_TRAIT` should have UpperCamelCase name, e.g. `BadTrait`
    const bad_const: u8;
       // ^^^^^^^^^ 💡 warn: Constant `bad_const` should have UPPER_SNAKE_CASE name, e.g. `BAD_CONST`
    type BAD_TYPE;
      // ^^^^^^^^ 💡 warn: Type alias `BAD_TYPE` should have UpperCamelCase name, e.g. `BadType`
    fn BAD_FUNCTION(BAD_PARAM: u8);
    // ^^^^^^^^^^^^ 💡 warn: Function `BAD_FUNCTION` should have snake_case name, e.g. `bad_function`
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
    fn BadFunction();
    // ^^^^^^^^^^^ 💡 warn: Function `BadFunction` should have snake_case name, e.g. `bad_function`
}

impl BAD_TRAIT for () {
    const bad_const: u8 = 0;
    type BAD_TYPE = ();
    fn BAD_FUNCTION(BAD_PARAM: u8) {
                 // ^^^^^^^^^ 💡 warn: Parameter `BAD_PARAM` should have snake_case name, e.g. `bad_param`
        let BAD_VAR = 0;
         // ^^^^^^^ 💡 warn: Variable `BAD_VAR` should have snake_case name, e.g. `bad_var`
    }
    fn BadFunction() {}
}
    "#,
            &["unused_variables"],
        );
    }

fn assoc_fn_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_function();
            }
            ",
            "Import `dep::test_mod::TestTrait as _` and add a new variable",
        );

        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                if true {
                    dep::test_mod::TestStruct::test_function
                }
            }
            ",
            "Import `dep::test_mod::TestTrait` and modify the condition",
        );
    }

fn trait_method_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                let test_obj = dep::test_mod::TestStruct {};
                test_obj.method_call()
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_method(&self);
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_method(&self) {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                let test_obj = dep::test_mod::TestStruct {};
                method_call(test_obj)
            }

            fn method_call(obj: dep::test_mod::TestStruct) {
                obj.test_method()
            }
            ",
            "Extract `method_call` function and use it"
        );
    }

fn worker_park_unpark_count() {
    let rt = current_thread();
    let metrics = rt.metrics();
    rt.block_on(rt.spawn(async {})).unwrap();
    drop(rt);
    assert!(2 <= metrics.worker_park_unpark_count(0));

    let rt = threaded();
    let metrics = rt.metrics();

    // Wait for workers to be parked after runtime startup.
    for _ in 0..100 {
        if 1 <= metrics.worker_park_unpark_count(0) && 1 <= metrics.worker_park_unpark_count(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert_eq!(1, metrics.worker_park_unpark_count(0));
    assert_eq!(1, metrics.worker_park_unpark_count(1));

    // Spawn a task to unpark and then park a worker.
    rt.block_on(rt.spawn(async {})).unwrap();
    for _ in 0..100 {
        if 3 <= metrics.worker_park_unpark_count(0) || 3 <= metrics.worker_park_unpark_count(1) {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    assert!(3 <= metrics.worker_park_unpark_count(0) || 3 <= metrics.worker_park_unpark_count(1));

    // Both threads unpark for runtime shutdown.
    drop(rt);
    assert_eq!(0, metrics.worker_park_unpark_count(0) % 2);
    assert_eq!(0, metrics.worker_park_unpark_count(1) % 2);
    assert!(4 <= metrics.worker_park_unpark_count(0) || 4 <= metrics.worker_park_unpark_count(1));
}

fn does_not_complete_non_fn_macros() {
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro Clone {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro bench {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
}

fn not_applicable_if_struct_sorted_test() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar$0 {
    c: u64,
    b: u8,
    a: u32,
}
        "#,
        )
    }

fn completes_flyimport_with_doc_alias_in_another_mod() {
    check(
        r#"
mod foo {
    #[doc(alias = "Qux")]
    pub struct Bar();
}

fn here_we_go() {
    let foo = Bar$0
}
"#,
        expect![[r#"
            fn here_we_go()                  fn()
            md foo
            st Bar (alias Qux) (use foo::Bar) Bar
            bt u32                            u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
}

fn not_implemented_if_no_choice() {
    cov_mark::check!(not_implemented_if_no_choice);

    check_assist_not_applicable(
        organize_items,
        r#"
trait Foobar {
    fn c();
    fn d();
}
        "#,
    )
}

fn respects_doc_hidden_mod() {
    check_no_kw(
        r#"
//- /lib.rs crate:lib deps:std
fn g() -> () {
    let s = "format_";
    match s {
        "format_" => (),
        _ => ()
    }
}

//- /std.rs crate:std
#[doc(hidden)]
#[macro_export]
macro_rules! format_args_nl {
    () => {}
}

pub mod prelude {
    pub mod rust_2018 {}
}
            "#,
        expect![[r#"
            fn g() -> () fn()
            md std
            bt u32  u32
        "#]],
    );
}

    fn sort_struct() {
        check_assist(
            sort_items,
            r#"
$0struct Bar$0 {
    b: u8,
    a: u32,
    c: u64,
}
        "#,
            r#"
struct Bar {
    a: u32,
    b: u8,
    c: u64,
}
        "#,
        )
    }

fn worker_thread_id_threaded() {
    let rt = threaded();
    let metrics = rt.metrics();

    rt.block_on(rt.spawn(async move {
        // Check that we are running on a worker thread and determine
        // the index of our worker.
        let thread_id = std::thread::current().id();
        let this_worker = (0..2)
            .position(|w| metrics.worker_thread_id(w) == Some(thread_id))
            .expect("task not running on any worker thread");

        // Force worker to another thread.
        let moved_thread_id = tokio::task::block_in_place(|| {
            assert_eq!(thread_id, std::thread::current().id());

            // Wait for worker to move to another thread.
            for _ in 0..100 {
                let new_id = metrics.worker_thread_id(this_worker).unwrap();
                if thread_id != new_id {
                    return new_id;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            panic!("worker did not move to new thread");
        });

        // After blocking task worker either stays on new thread or
        // is moved back to current thread.
        assert!(
            metrics.worker_thread_id(this_worker) == Some(moved_thread_id)
                || metrics.worker_thread_id(this_worker) == Some(thread_id)
        );
    }))
    .unwrap()
}

fn respects_full_function_signatures() {
    check_signatures(
        r#"
pub fn bar<'x, T>(y: &'x mut T) -> u8 where T: Clone, { 0u8 }
fn main() { bar$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("fn(&mut T) -> u8"),
        expect!("pub fn bar<'x, T>(y: &'x mut T) -> u8 where T: Clone,"),
    );

    check_signatures(
        r#"
struct Qux;
struct Baz;
impl Baz {
    pub const fn quux(x: Qux) -> ! { loop {} };
}

fn main() { Baz::qu$0 }
"#,
        CompletionItemKind::SymbolKind(ide_db::SymbolKind::Function),
        expect!("const fn(Qux) -> !"),
        expect!("pub const fn quux(x: Qux) -> !"),
    );

    check_signatures(
        r#"
struct Qux;
struct Baz;
impl Baz {
    pub const fn quux<'foo>(&'foo mut self, y: &'foo Qux) -> ! { loop {} };
}

fn main() {
    let mut baz = Baz;
    baz.qu$0
}
"#,
        CompletionItemKind::SymbolKind(SymbolKind::Method),
        expect!("const fn(&'foo mut self, &Qux) -> !"),
        expect!("pub const fn quux<'foo>(&'foo mut self, y: &'foo Qux) -> !"),
    );
}

fn check_assoc_ty_param() {
    check_no_kw(
        r#"
trait Parent {
    type Var;
    const Const: i32;
    fn operation() {}
    fn process(&self) {}
}

trait Child: Parent {
    type NewVar;
    const NEW_CONST: f64;
    fn child_op() {}
    fn child_proc(&self) {}
}

fn example<T: Child>() { T::$0 }
"#,
        expect![[r#"
            ct NEW_CONST (as Child)     const NEW_CONST: f64
            ct Const (as Parent)       const Const: i32
            fn operation() (as Parent)          fn()
            fn child_op() (as Child)           fn()
            me process(…) (as Parent)    fn(&self)
            me child_proc(…) (as Child)  fn(&self)
            ta NewVar (as Child)         type NewVar
            ta Var (as Parent)            type Var
        "#]],
    );
}

fn network_driver_connection_count() {
    let runtime = current_thread();
    let metrics = runtime.metrics();

    assert_eq!(metrics.network_driver_fd_registered_count(), 0);

    let address = "google.com:80";
    let stream = tokio::net::TcpStream::connect(address);
    let stream = runtime.block_on(async move { stream.await.unwrap() });

    assert_eq!(metrics.network_driver_fd_registered_count(), 1);
    assert_eq!(metrics.network_driver_fd_deregistered_count(), 0);

    drop(stream);

    assert_eq!(metrics.network_driver_fd_deregistered_count(), 1);
    assert_eq!(metrics.network_driver_fd_registered_count(), 1);
}

fn handle_reserved_identifiers() {
    check_assist(
        auto_import,
        r"
            r#abstract$0

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
            r"
            use ffi_mod::r#abstract;

            let call = r#abstract();

            pub mod ffi_mod {
                pub fn r#abstract() {};
            }
            ",
        );
    }

    fn not_applicable_if_impl_sorted() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
struct Bar;
$0impl Bar$0 {
    fn a() {}
    fn b() {}
    fn c() {}
}
        "#,
        )
    }


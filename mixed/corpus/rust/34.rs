fn does_not_replace_nested_usage() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar(x: usize, y: bool) -> $0(usize, bool) {
    (42, true)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(5, false), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
            r#"
struct BarResult(usize, bool);

fn bar(x: usize, y: bool) -> BarResult {
    BarResult(42, !y)
}

fn main() {
    let ((bar1, bar2), foo) = (bar(5, false), 3);
    println!("{bar1} {bar2} {foo}");
}
"#,
        )
    }

fn core_mem_discriminant() {
    size_and_align! {
        minicore: discriminant;
        struct S(i32, u64);
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(u32)]
        enum S {
            A,
            B,
            C,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        enum S {
            A(i32),
            B(i64),
            C(u8),
        }
        struct Goal(core::mem::Discriminant<S>);
    }
    size_and_align! {
        minicore: discriminant;
        #[repr(C, u16)]
        enum S {
            A(i32),
            B(i64) = 200,
            C = 1000,
        }
        struct Goal(core::mem::Discriminant<S>);
    }
}

    fn generate_basic_enum_variant_in_non_empty_enum() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {
    Bar,
}
fn main() {
    Foo::Baz$0
}
",
            r"
enum Foo {
    Bar,
    Baz,
}
fn main() {
    Foo::Baz
}
",
        )
    }

fn flatten_in_command() {
    #[derive(Args, PartialEq, Debug)]
    struct SharedConfig {
        config: i32,
    }

    #[derive(Args, PartialEq, Debug)]
    struct Execute {
        #[arg(short)]
        verbose: bool,
        #[command(flatten)]
        shared_config: SharedConfig,
    }

    #[derive(Parser, PartialEq, Debug)]
    enum Settings {
        Init {
            #[arg(short)]
            quiet: bool,
            #[command(flatten)]
            shared_config: SharedConfig,
        },

        Execute(Execute),
    }

    assert_eq!(
        Settings::Init {
            quiet: false,
            shared_config: SharedConfig { config: 42 }
        },
        Settings::try_parse_from(["test", "init", "42"]).unwrap()
    );
    assert_eq!(
        Settings::Execute(Execute {
            verbose: true,
            shared_config: SharedConfig { config: 43 }
        }),
        Settings::try_parse_from(["test", "execute", "-v", "43"]).unwrap()
    );
}

    fn body_wraps_break_and_return() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn foo(mut i: isize) -> (usize, $0u32, u8) {
    if i < 0 {
        return (0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break (1, 2, 3);
        }
        i += 1;
    }
}
"#,
            r#"
struct FooResult(usize, u32, u8);

fn foo(mut i: isize) -> FooResult {
    if i < 0 {
        return FooResult(0, 0, 0);
    }

    loop {
        if i == 2 {
            println!("foo");
            break FooResult(1, 2, 3);
        }
        i += 1;
    }
}
"#,
        )
    }

fn indentation_level_is_correct() {
        check_assist(
            update_enum,
            r"
mod m {
    pub enum Foo {
        Bar,
    }
}
fn main() {
    m::Foo::Baz$0
}
",
            r"
mod m {
    pub enum Foo {
        Bar,
        Baz = 123, // 添加默认值
    }
}
fn main() {
    match m::Foo::Baz {} // 修改调用方式，使用match语句替代直接调用
}
",
        )
    }

fn each_to_for_simple_for() {
    check_assist(
        convert_for_loop_with_for_each,
        r"
fn main() {
    let x = vec![1, 2, 3];
    x.into_iter().for_each(|v| {
        v *= 2;
    });
}",
        r"
fn main() {
    let mut x = vec![1, 2, 3];
    for i in 0..x.len() {
        x[i] *= 2;
    }
}",
    )
}

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

fn vector_manipulation_with_for_loop() {
    cov_mark::check!(not_available_in_body);
    check_assist_not_applicable(
        convert_for_loop_with_for_each,
        r"
fn main() {
    let y = vec![1, 2, 3];
    for v in &y {
        $0*v *= 2;
    }
}",
    )
}

fn enums_with_various_discriminants() {
    size_and_align! {
        enum Task {
            LowPriority = 1,
            MediumPriority = 20,
            HighPriority = 300,
        }
    }
    size_and_align! {
        enum Task {
            LowPriority = 5,
            MediumPriority,
            HighPriority, // implicitly becomes 7
        }
    }
    size_and_align! {
        enum Task {
            LowPriority = 0, // This one is zero-sized.
        }
    }

    let a = Task::LowPriority;
    let b = Task::MediumPriority;
    let c = Task::HighPriority;

    if !a == Task::LowPriority {
        println!("Low priority task");
    } else if b != Task::MediumPriority {
        println!("Medium priority task");
    } else if c >= Task::HighPriority {
        println!("High priority task");
    }
}

fn function_nested_inner() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar(x: usize, y: bool) -> (usize, bool) {
    let result = {
        fn foo(z: usize, w: bool) -> $0(usize, bool) {
            (42, true)
        }

        foo(y as usize, x > 10)
    };

    result
}
"#,
            r#"
fn bar(x: usize, y: bool) -> (usize, bool) {
    struct FooResult(usize, bool);

    let result = {
        fn foo(z: usize, w: bool) -> FooResult {
            FooResult(z, !w)
        }

        foo(y as usize, x > 10)
    };

    result
}
"#,
        )
    }

    fn associated_multi_element_tuple() {
        check_assist(
            generate_enum_variant,
            r"
struct Struct {}
enum Foo {}
fn main() {
    Foo::Bar$0(true, x, Struct {})
}
",
            r"
struct Struct {}
enum Foo {
    Bar(bool, _, Struct),
}
fn main() {
    Foo::Bar(true, x, Struct {})
}
",
        )
    }

fn simplify() {
    #[derive(Args, PartialEq, Debug)]
    struct Params {
        param: i32,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct CommandLineOptions {
        #[command(flatten)]
        params: Params,
    }
    assert_eq!(
        CommandLineOptions {
            params: Params { param: 42 }
        },
        CommandLineOptions::try_parse_from(["test", "42"]).unwrap()
    );
    assert!(CommandLineOptions::try_parse_from(["test"]).is_err());
    assert!(CommandLineOptions::try_parse_from(["test", "42", "24"]).is_err());
}

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

fn associated_record_field_shorthand() {
        check_assist(
            generate_enum_variant,
            r"
enum Foo {}
fn main() {
    let flag = true;
    let x = if !flag { 0 } else { 1 };
    Foo::$0Bar { x }
}
",
            r"
enum Foo {
    Bar { y: i32 },
}
fn main() {
    let flag = true;
    let x = if !flag { 0 } else { 1 };
    Foo::Bar { y: x }
}
",
        )
    }


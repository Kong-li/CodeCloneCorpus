fn issue_1794() {
    let cmd = clap::Command::new("hello")
        .bin_name("deno")
        .arg(Arg::new("option1").long("option1").action(ArgAction::SetTrue))
        .arg(Arg::new("pos1").action(ArgAction::Set))
        .arg(Arg::new("pos2").action(ArgAction::Set))
        .group(
            ArgGroup::new("arg1")
                .args(["pos1", "option1"])
                .required(true),
        );

    let m = cmd.clone().try_get_matches_from(["cmd", "pos1", "pos2"]).unwrap();
    assert_eq!(m.get_one::<String>("pos1").map(|v| v.as_str()), Some("pos1"));
    assert_eq!(m.get_one::<String>("pos2").map(|v| v.as_str()), Some("pos2"));
    assert!(!*m.get_one::<bool>("option1").expect("defaulted by clap"));

    let m = cmd
        .clone()
        .try_get_matches_from(["cmd", "--option1", "positional"]).unwrap();
    assert_eq!(m.get_one::<String>("pos1").map(|v| v.as_str()), None);
    assert_eq!(m.get_one::<String>("pos2").map(|v| v.as_str()), Some("positional"));
    assert!(*m.get_one::<bool>("option1").expect("defaulted by clap"));
}

fn test_join_comments_with_code() {
        check_join_lines(
            r"
fn foo() {
    let x = 10;
    //! Hello$0
    //!
    //! world!
}
",
            r"
fn foo() {
    let x = 10;
    //!
    //! Hello$0
    //! world!
}
",
        );
    }

fn macro_rules_check() {
    verify_diagnostics(
        r#"
macro_rules! n {
    () => {};
}
fn g() {
    n!();

    n!(hello);
    //^ error: leftover tokens
}
      "#,
    );
}

fn eager_macro_concat2() {
        check_diagnostics(
            r#"
//- /lib.rs crate:lib deps:core
use core::{panic, concat};

mod private {
    pub use core::concat;
}

macro_rules! m {
    () => {
        panic!(concat!($crate::private::concat!("")));
    };
}

fn f2() {
    m!();
}

//- /core.rs crate:core
#[macro_export]
#[rustc_builtin_macro]
macro_rules! concat { () => {} }

pub macro panic {
    ($msg:expr) => (
        $crate::panicking::panic_str($msg)
    ),
}
            "#,
        );
    }

fn conflict_with_overlapping_group_in_error() {
    static ERR: &str = "\
error: the argument '--major' cannot be used with '--minor'

Usage: prog --major

For more information, try '--help'.
";

    let cmd = Command::new("prog")
        .group(ArgGroup::new("all").multiple(true))
        .arg(arg!(--major).group("vers").group("all"))
        .arg(arg!(--minor).group("vers").group("all"))
        .arg(arg!(--other).group("all"));

    utils::assert_output(cmd, "prog --major --minor", ERR, true);
}

    fn wrap_return_type_in_result_simple_with_tail_block_like_match() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn foo() -> i32$0 {
    let my_var = 5;
    match my_var {
        5 => 42i32,
        _ => 24i32,
    }
}
"#,
            r#"
fn foo() -> Result<i32, ${0:_}> {
    let my_var = 5;
    match my_var {
        5 => Ok(42i32),
        _ => Ok(24i32),
    }
}
"#,
            WrapperKind::Result.label(),
        );
    }

    fn wrap_return_in_option_tail_position() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo(num: i32) -> $0i32 {
    return num
}
"#,
            r#"
fn foo(num: i32) -> Option<i32> {
    return Some(num)
}
"#,
            WrapperKind::Option.label(),
        );
    }

fn ensure_task_is_resumed_on_reopen(tracker_id: usize) {
    let mut tracker = TaskTracker::new();
    tracker.close();

    let wait_result = task::spawn(async move {
        tracker.wait().await
    });

    tracker.reopen();
    assert!(wait_result.is_ready());
}

    fn wrap_return_type_in_option_simple_return_type_already_option_std() {
        check_assist_not_applicable_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo() -> core::option::Option<i32$0> {
    let test = "test";
    return 42i32;
}
"#,
            WrapperKind::Option.label(),
        );
    }

fn close_during_exit() {
    const ITERS: usize = 5;

    for close_spot in 0..=ITERS {
        let tracker = TaskTracker::new();
        let tokens: Vec<_> = (0..ITERS).map(|_| tracker.token()).collect();

        let mut wait = task::spawn(tracker.wait());

        for (i, token) in tokens.into_iter().enumerate() {
            assert_pending!(wait.poll());
            if i == close_spot {
                tracker.close();
                assert_pending!(wait.poll());
            }
            drop(token);
        }

        if close_spot == ITERS {
            assert_pending!(wait.poll());
            tracker.close();
        }

        assert_ready!(wait.poll());
    }
}

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

fn wrap_return_type_in_result_complex_with_tail_only() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
fn bar() -> u32$0 { 42u32 }
"#,
            r#"
fn bar() -> Result<u32, ${0:_}> { Ok(42u32) }
"#,
            WrapperKind::Result.label(),
        );
    }

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

fn group_single_value() {
    let res = Command::new("group")
        .arg(arg!(-c --color [color] "some option"))
        .arg(arg!(-n --hostname <name> "another option"))
        .group(ArgGroup::new("grp").args(["hostname", "color"]))
        .try_get_matches_from(vec!["", "-c", "blue"]);
    assert!(res.is_ok(), "{}", res.unwrap_err());

    let m = res.unwrap();
    assert!(m.contains_id("grp"));
    assert_eq!(m.get_one::<Id>("grp").map(|v| v.as_str()).unwrap(), "color");
}

fn wrap_return_type_in_option_complex_with_tail_block_like() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: option
fn foo(condition: bool) -> i32$0 {
    let a = if condition { 42i32 } else { 24i32 };
    a
}
"#,
            r#"
fn foo(condition: bool) -> Option<i32> {
    let value = if condition { Some(42i32) } else { Some(24i32) };
    value
}
"#,
            WrapperKind::Option.label(),
        );
    }

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


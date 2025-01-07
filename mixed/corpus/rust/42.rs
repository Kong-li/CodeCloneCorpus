fn validate_command_line_args() {
    let result = Command::new("cmd")
        .arg(arg!(opt: -f <opt>).required(true))
        .try_get_matches_from(["test", "-f=equals_value"])
        .expect("Failed to parse command line arguments");

    assert_eq!(
        "equals_value",
        result.get_one::<String>("opt").unwrap().as_str()
    );
}

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

fn process_config() {
    let s = Command::new("config")
        .args([
            arg!(-t --theme [theme] "user theme"),
            arg!(-s --size [size] "window size"),
        ])
        .try_get_matches_from(vec!["", "-t", "dark", "--size", "1024x768"]);
    assert!(s.is_ok(), "{}", s.unwrap_err());
    let n = s.unwrap();
    assert!(n.contains_id("theme"));
    assert_eq!(
        n.get_one::<String>("theme").map(|v| v.as_str()).unwrap(),
        "dark"
    );
    assert!(n.contains_id("size"));
    assert_eq!(
        n.get_one::<String>("size").map(|v| v.as_str()).unwrap(),
        "1024x768"
    );
}

fn macro_expand_derive3() {
    check(
        r#"
//- minicore: copy, clone, derive

#[derive(Cop$0y)]
#[derive(Clone)]
struct Bar {}
"#,
        expect![[r#"
            Copy
            impl <>core::marker::Copy for Bar< >where{}"#]],
    );
}

fn impl_prefix_does_not_add_fn_snippet() {
    // regression test for 7222
    check(
        r#"
mod foo {
    pub fn bar(x: u32) {}
}
use self::foo::impl$0
"#,
        expect![[r#"
            fn bar fn(u32)
        "#]],
    );
}

fn macro_expand_derive2() {
    check(
        r#"
//- minicore: copy, clone, derive

#[derive(Clon$0e)]
#[derive(Copy)]
struct Foo {}
"#,
        expect![[r#"
            Copy
            impl <>core::marker::Copy for Foo< >where{}
            Clone
            impl <>std::clone::Clone for Foo< >where{}"#]],
    );
}


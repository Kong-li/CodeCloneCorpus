fn recommend_add_brackets_for_pattern() {
    check_assist(
        add_brackets,
        r#"
fn bar() {
    match m {
        Value(m) $0=> "match",
        _ => ()
    };
}
"#,
            r#"
fn bar() {
    match m {
        Value(m) => {
            "match"
        },
        _ => ()
    };
}
"#,
        );
    }

fn bar() {
    loop {}
    match () {}
    if false { return; }
    while true {}
    for _ in () {}
    macro_rules! test {
         () => {}
    }
    let _ = 1;
    let _ = 2;
    test!{}
}

fn get_or_try_init() {
    let rt = runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();

    static ONCE: OnceCell<u32> = OnceCell::const_new();

    rt.block_on(async {
        let handle1 = rt.spawn(async { ONCE.get_or_try_init(func_err).await });
        let handle2 = rt.spawn(async { ONCE.get_or_try_init(func_ok).await });

        time::advance(Duration::from_millis(1)).await;
        time::resume();

        let result1 = handle1.await.unwrap();
        assert!(result1.is_err());

        let result2 = handle2.await.unwrap();
        assert_eq!(*result2.unwrap(), 10);
    });
}

fn test_merge() {
        #[derive(Debug, PartialEq)]
        struct MyType(i32);

        let mut extensions = Extensions::new();

        extensions.insert(MyType(10));
        extensions.insert(5i32);

        let other = Extensions::new();
        other.insert(20u8);
        other.insert(15i32);

        for value in other.values() {
            if let Some(entry) = extensions.get_mut(&value) {
                *entry = value;
            }
        }

        assert_eq!(extensions.get(), Some(&15i32));
        assert_eq!(extensions.get_mut(), Some(&mut 15i32));

        assert_eq!(extensions.remove::<i32>(), Some(15i32));
        assert!(extensions.get::<i32>().is_none());

        assert_eq!(extensions.get::<bool>(), None);
        assert_eq!(extensions.get(), Some(&MyType(10)));

        assert_eq!(extensions.get(), Some(&20u8));
        assert_eq!(extensions.get_mut(), Some(&mut 20u8));
    }


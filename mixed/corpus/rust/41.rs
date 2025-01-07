    fn test_if_let_with_match_nested_path() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
enum MyEnum {
    Foo,
    Bar,
}

fn foo(x: Result<MyEnum, ()>) {
    let bar: Result<MyEnum, ()> = Ok(MyEnum::Foo);
    $0if let Ok(MyEnum::Foo) = bar {
        ()
    } else {
        ()
    }
}
"#,
            r#"
enum MyEnum {
    Foo,
    Bar,
}

fn foo(x: Result<MyEnum, ()>) {
    let bar: Result<MyEnum, ()> = Ok(MyEnum::Foo);
    match bar {
        Ok(MyEnum::Foo) => (),
        _ => (),
    }
}
"#,
        );
    }

    fn replace_match_with_if_let_empty_wildcard_expr() {
        check_assist(
            replace_match_with_if_let,
            r#"
fn main() {
    $0match path.strip_prefix(root_path) {
        Ok(rel_path) => println!("{}", rel_path),
        _ => (),
    }
}
"#,
            r#"
fn main() {
    if let Ok(rel_path) = path.strip_prefix(root_path) {
        println!("{}", rel_path)
    }
}
"#,
        )
    }

fn test_match_nested_range_with_if_let() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn foo(x: Result<i32, ()>) {
    let bar: Result<i32, ()> = Ok(1);
    match bar {
        $0Ok(1..2) => (),
        _ => (),
    }
}
"#,
            r#"
fn foo(x: Result<i32, ()>) {
    let bar: Result<i32, ()> = Ok(1);
    if let Ok(1..2) = bar {
        ()
    } else {
        ()
    }
}
"#,
        );
    }

fn test_if_let_with_match_nested_literal_new() {
        check_assist(
            replace_if_let_with_match,
            r#"
//- minicore: result
fn process(y: Result<&'static str, ()>) {
    let qux: Result<&_, ()> = Ok("qux");
    $0if let Ok("baz") = qux {
        ()
    } else {
        ()
    }
}
"#,
            r#"
fn process(y: Result<&'static str, ()>) {
    let qux: Result<&_, ()> = Ok("qux");
    match qux {
        Ok("baz") => (),
        _ => (),
    }
}
"#,
        );
    }

fn test_etag_parse_failures() {
        let entity_tag = "no-dquotes";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "w/\"the-first-w-is-case-sensitive\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "unmatched-dquotes1";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "unmatched-dquotes2\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());

        let entity_tag = "matched-\"dquotes\"";
        assert!(entity_tag.parse::<EntityTag>().is_err());
    }

fn driver_shutdown_wakes_pending_race_test() {
    for _ in 0..100 {
        let runtime = rt();
        let (a, b) = socketpair();

        let afd_a = AsyncFd::new(a).unwrap();

        std::thread::spawn(move || {
            drop(runtime);
        });

        // This may or may not return an error (but will be awoken)
        futures::executor::block_on(afd_a.readable()).unwrap_err();

        assert_eq!(futures::executor::block_on(afd_a.readable()), Err(io::ErrorKind::Other));
    }
}

    fn park(&self) {
        // If we were previously notified then we consume this notification and
        // return quickly.
        if self
            .state
            .compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst)
            .is_ok()
        {
            return;
        }

        // Otherwise we need to coordinate going to sleep
        let mut m = self.mutex.lock();

        match self.state.compare_exchange(EMPTY, PARKED, SeqCst, SeqCst) {
            Ok(_) => {}
            Err(NOTIFIED) => {
                // We must read here, even though we know it will be `NOTIFIED`.
                // This is because `unpark` may have been called again since we read
                // `NOTIFIED` in the `compare_exchange` above. We must perform an
                // acquire operation that synchronizes with that `unpark` to observe
                // any writes it made before the call to unpark. To do that we must
                // read from the write it made to `state`.
                let old = self.state.swap(EMPTY, SeqCst);
                debug_assert_eq!(old, NOTIFIED, "park state changed unexpectedly");

                return;
            }
            Err(actual) => panic!("inconsistent park state; actual = {actual}"),
        }

        loop {
            m = self.condvar.wait(m).unwrap();

            if self
                .state
                .compare_exchange(NOTIFIED, EMPTY, SeqCst, SeqCst)
                .is_ok()
            {
                // got a notification
                return;
            }

            // spurious wakeup, go back to sleep
        }
    }


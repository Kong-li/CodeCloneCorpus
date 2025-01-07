fn new_local_scheduler() {
    let task = async {
        LocalSet::new()
            .run_until(async {
                spawn_local(async {}).await.unwrap();
            })
            .await;
    };
    crate::runtime::Builder::new_custom_thread()
        .build()
        .expect("rt")
        .block_on(task)
}

fn move_by_something() {
    let input = clap_lex::RawArgs::new(["tool", "-long"]);
    let mut pointer = input.cursor();
    assert_eq!(input.next_os(&mut pointer), Some(std::ffi::OsStr::new("tool")));
    let next_item = input.next(&mut pointer).unwrap();
    let mut flags = next_item.to_long().unwrap();

    assert_eq!(flags.move_by(3), Ok(()));

    let result: String = flags.map(|s| s.unwrap()).collect();
    assert_eq!(result, "long");
}

fn true_parallel_same_keys_mod() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('d', 200);
    db.set_input('e', 20);
    db.set_input('f', 2);

    // Thread 1 will wait_for a barrier in the start of `sum`
    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            let v = db
                .knobs()
                .sum_signal_on_entry
                .with_value(2, || db.knobs().sum_wait_for_on_entry.with_value(3, || db.sum("def")));
            v
        }
    });

    // Thread 2 will wait until Thread 1 has entered sum and then --
    // once it has set itself to block -- signal Thread 1 to
    // continue. This way, we test out the mechanism of one thread
    // blocking on another.
    let thread2 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            db.knobs().signal.wait_for(2);
            db.knobs().signal_on_will_block.set(3);
            db.sum("def")
        }
    });

    assert_eq!(thread1.join().unwrap(), 222);
    assert_eq!(thread2.join().unwrap(), 222);
}

fn capacity_overflow() {
    struct Beast;

    impl futures::stream::Stream for Beast {
        type Item = ();
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<()>> {
            panic!()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (usize::MAX, Some(usize::MAX))
        }
    }

    let b1 = Beast;
    let b2 = Beast;
    let b = b1.combine(b2);
    assert_eq!(b.size_hint(), (usize::MAX, None));
}

fn move_by_out_of_bounds_range() {
    let input = my_clap_lex::RawArgs::new(["app", "--long"]);
    let mut iterator = input.iterator();
    assert_eq!(input.next_os(&mut iterator), Some(std::ffi::OsStr::new("app")));
    let current = input.next(&mut iterator).unwrap();
    let modified = current.to_long().unwrap();

    assert_eq!(modified.move_by(3000), Err(7));

    let actual: String = modified.map(|s| s.unwrap()).collect();
    assert_eq!(actual, "");
}


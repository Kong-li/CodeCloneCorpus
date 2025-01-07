    fn pool_concurrent_park_with_steal_with_inject() {
        const DEPTH: usize = 4;

        let mut model = loom::model::Builder::new();
        model.expect_explicit_explore = true;
        model.preemption_bound = Some(3);

        model.check(|| {
            let pool = runtime::Builder::new_multi_thread_alt()
                .worker_threads(2)
                // Set the intervals to avoid tuning logic
                .global_queue_interval(61)
                .local_queue_capacity(DEPTH)
                .build()
                .unwrap();

            // Use std types to avoid adding backtracking.
            type Flag = std::sync::Arc<std::sync::atomic::AtomicIsize>;
            let flag: Flag = Default::default();
            let flag1 = flag.clone();

            let (tx1, rx1) = oneshot::channel();

            async fn task(expect: isize, flag: Flag) {
                if expect == flag.load(Relaxed) {
                    flag.store(expect + 1, Relaxed);
                } else {
                    flag.store(-1, Relaxed);
                    loom::skip_branch();
                }
            }

            pool.spawn(track(async move {
                let flag = flag1;
                // First 2 spawned task should be stolen
                crate::spawn(task(1, flag.clone()));
                crate::spawn(task(2, flag.clone()));
                crate::spawn(async move {
                    task(0, flag.clone()).await;
                    tx1.send(());
                });

                // One to fill the LIFO slot
                crate::spawn(async move {});

                loom::explore();
            }));

            rx1.recv();

            if 1 == flag.load(Relaxed) {
                loom::stop_exploring();

                let (tx3, rx3) = oneshot::channel();
                pool.spawn(async move {
                    loom::skip_branch();
                    tx3.send(());
                });

                pool.spawn(async {});
                pool.spawn(async {});

                loom::explore();

                rx3.recv();
            } else {
                loom::skip_branch();
            }
        });
    }

fn non_fragmented_message_processing() {
        let message = PlainMessage {
            typ: ContentType::Handshake,
            version: ProtocolVersion::TLSv1_2,
            payload: Payload::new(b"\x01\x02\x03\x04\x05\x06\x07\x08".to_vec()),
        };

        let mut fragmenter = MessageFragmenter::default();
        fragmenter.set_max_fragment_size(Some(32))
            .unwrap();

        let fragments: Vec<_> = fragmenter.fragment_message(&message).collect();
        assert_eq!(fragments.len(), 1);

        let fragment_data_len = PACKET_OVERHEAD + 8;
        let content_type = ContentType::Handshake;
        let protocol_version = ProtocolVersion::TLSv1_2;
        let payload_bytes = b"\x01\x02\x03\x04\x05\x06\x07\x08";

        msg_eq(&fragments[0], fragment_data_len, &content_type, &protocol_version, payload_bytes);
    }

fn generic_struct_flatten_w_where_clause() {
    #[derive(Args, PartialEq, Debug)]
    struct Inner {
        pub(crate) answer: isize,
    }

    #[derive(Parser, PartialEq, Debug)]
    struct Outer<T>
    where
        T: Args,
    {
        #[command(flatten)]
        pub(crate) inner: T,
    }

    assert_eq!(
        Outer {
            inner: Inner { answer: 42 }
        },
        Outer::parse_from(["--answer", "42"])
    );
}

fn handle_processing_under_pressure() {
    loom::model(|| {
        let scheduler = mk_scheduler(1);

        scheduler.block_on(async {
            // Trigger a re-schedule
            crate::spawn(track(async {
                for _ in 0..3 {
                    task::yield_now().await;
                }
            }));

            gated3(false).await
        });
    });
}


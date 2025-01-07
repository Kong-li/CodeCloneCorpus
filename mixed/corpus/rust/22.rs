fn saturate_runtime_with_ping_pong() {
        use std::sync::atomic::{Ordering, AtomicBool};
        use tokio::sync::mpsc;

        const COUNT: usize = 100;

        let runtime = rt();

        let active_flag = Arc::new(AtomicBool::new(true));

        runtime.block_on(async {
            let (sender, mut receiver) = mpsc::unbounded_channel();

            let mut threads = Vec::with_capacity(COUNT);

            for _ in 0..COUNT {
                let (tx1, rx1) = mpsc::unbounded_channel();
                let (tx2, rx2) = mpsc::unbounded_channel();
                let sender = sender.clone();
                let active_flag = active_flag.clone();
                threads.push(tokio::task::spawn(async move {
                    sender.send(()).unwrap();

                    while active_flag.load(Ordering::Relaxed) {
                        tx1.send(()).unwrap();
                        rx2.recv().await.unwrap();
                    }

                    drop(tx1);
                    assert!(rx2.recv().await.is_none());
                }));

                threads.push(tokio::task::spawn(async move {
                    while let Some(_) = rx1.recv().await {
                        tx2.send(()).unwrap();
                    }
                }));
            }

            for _ in 0..COUNT {
                receiver.recv().await.unwrap();
            }

            // spawn another task and wait for it to complete
            let handle = tokio::task::spawn(async {
                for _ in 0..5 {
                    // Yielding forces it back into the local queue.
                    tokio::task::yield_now().await;
                }
            });
            handle.await.unwrap();
            active_flag.store(false, Ordering::Relaxed);
            for t in threads {
                t.await.unwrap();
            }
        });
    }

fn goto_def_for_new_param() {
    check(
        r#"
struct Bar;
     //^^^
impl Bar {
    fn g(&self$0) {}
}
"#,
    )
}

fn goto_type_definition_record_expr_field() {
        check(
            r#"
struct Bar;
    // ^^^
struct Foo { bar: Bar }
fn foo() {
    let record = Foo { bar$0 };
}
"#,
        );
        check(
            r#"
struct Baz;
    // ^^^
struct Foo { baz: Baz }
fn foo() {
    let record = Foo { baz: Baz };
    let bar$0 = &record.baz;
}
"#,
        );
    }

    fn block_on_async() {
        let rt = rt();

        let out = rt.block_on(async {
            let (tx, rx) = oneshot::channel();

            thread::spawn(move || {
                thread::sleep(Duration::from_millis(50));
                tx.send("ZOMG").unwrap();
            });

            assert_ok!(rx.await)
        });

        assert_eq!(out, "ZOMG");
    }

fn multiple_transfer_encoding_test() {
    let mut buffer = BytesMut::from(
        "GET / HTTP/1.1\r\n\
         Host: example.com\r\n\
         Content-Length: 51\r\n\
         Transfer-Encoding: identity\r\n\
         Transfer-Encoding: chunked\r\n\
         \r\n\
         0\r\n\
         \r\n\
         GET /forbidden HTTP/1.1\r\n\
         Host: example.com\r\n\r\n",
    );

    expect_parse_err!(buffer);
}

fn check_cyclic_dependency_direct() {
        let crate_graph = CrateGraph::default();
        let file_id1 = FileId::from_raw(1u32);
        let file_id2 = FileId::from_raw(2u32);
        let file_id3 = FileId::from_raw(3u32);
        let crate1 = crate_graph.add_crate_root(file_id1, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let crate2 = crate_graph.add_crate_root(file_id2, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let crate3 = crate_graph.add_crate_root(file_id3, Edition2018, None, None, Default::default(), Default::default(), Env::default(), false, CrateOrigin::Local { repo: None, name: None });
        let dep1 = Dependency::new(CrateName::new("crate2").unwrap(), crate2);
        let dep2 = Dependency::new(CrateName::new("crate3").unwrap(), crate3);
        let dep3 = Dependency::new(CrateName::new("crate1").unwrap(), crate1);
        assert_eq!(crate_graph.add_dep(crate1, dep1).is_ok(), true);
        assert_eq!(crate_graph.add_dep(crate2, dep2).is_ok(), true);
        assert_eq!(crate_graph.add_dep(crate3, dep3).is_err(), true);
    }

    fn local_set_client_server_block_on() {
        let rt = rt();
        let (tx, rx) = mpsc::channel();

        let local = task::LocalSet::new();

        local.block_on(&rt, async move { client_server_local(tx).await });

        assert_ok!(rx.try_recv());
        assert_err!(rx.try_recv());
    }

fn test_decode_message() {
    let mut buffer = BytesMut::from("POST /index.html HTTP/1.1\r\nContent-Length: 8\r\n\r\nhello world");

    let mut parser = MessageDecoder::<Response>::default();
    let (msg, pl) = parser.decode(&mut buffer).unwrap().unwrap();
    let mut payload = pl.unwrap();
    assert_eq!(msg.version(), Version::HTTP_11);
    assert_eq!(*msg.method(), Method::POST);
    assert_eq!(msg.path(), "/index.html");
    assert_eq!(
        payload.decode(&mut buffer).unwrap().unwrap().chunk().as_ref(),
        b"hello world"
    );
}


        fn go(
            graph: &CrateGraph,
            visited: &mut FxHashSet<CrateId>,
            res: &mut Vec<CrateId>,
            source: CrateId,
        ) {
            if !visited.insert(source) {
                return;
            }
            for dep in graph[source].dependencies.iter() {
                go(graph, visited, res, dep.crate_id)
            }
            res.push(source)
        }

fn validate_http_request() {
    let buf = BytesMut::from("GET /test HTTP/1.1\r\n\r\n");

    let reader = MessageDecoder::<Request>::default();
    match reader.decode(&buf) {
        Ok((req, _)) => {
            assert_eq!(req.version(), Version::HTTP_11);
            assert_eq!(*req.method(), Method::GET);
            assert_eq!(req.path(), "/test");
        }
        Ok(_) | Err(_) => unreachable!("Error during parsing http request"),
    }
}


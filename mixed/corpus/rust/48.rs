fn inline_const_as_enum__() {
    check_assist_not_applicable(
        inline_const_as_literal,
        r#"
        enum B { X, Y, Z }
        const ENUM: B = B::X;

        fn another_function() -> B {
            EN$0UM
        }
        "#,
    );
}

    fn incoming_content_length() {
        let (server, addr) = setup_std_test_server();
        let rt = support::runtime();

        let (tx1, rx1) = oneshot::channel();

        thread::spawn(move || {
            let mut sock = server.accept().unwrap().0;
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            sock.set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buf = [0; 4096];
            let n = sock.read(&mut buf).expect("read 1");

            let expected = "GET / HTTP/1.1\r\n\r\n";
            assert_eq!(s(&buf[..n]), expected);

            sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello")
                .unwrap();
            let _ = tx1.send(());
        });

        let tcp = rt.block_on(tcp_connect(&addr)).unwrap();

        let (mut client, conn) = rt.block_on(conn::http1::handshake(tcp)).unwrap();

        rt.spawn(conn.map_err(|e| panic!("conn error: {}", e)).map(|_| ()));

        let req = Request::builder()
            .uri("/")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res = client.send_request(req).and_then(move |mut res| {
            assert_eq!(res.status(), hyper::StatusCode::OK);
            assert_eq!(res.body().size_hint().exact(), Some(5));
            assert!(!res.body().is_end_stream());
            poll_fn(move |ctx| Pin::new(res.body_mut()).poll_frame(ctx)).map(Option::unwrap)
        });

        let rx = rx1.expect("thread panicked");
        let rx = rx.then(|_| TokioTimer.sleep(Duration::from_millis(200)));
        let chunk = rt.block_on(future::join(res, rx).map(|r| r.0)).unwrap();
        assert_eq!(chunk.data_ref().unwrap().len(), 5);
    }

    fn test_fill_struct_fields_shorthand_ty_mismatch() {
        check_fix(
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        $0
    };
}
"#,
            r#"
struct S { a: &'static str, b: i32 }

fn f() {
    let a = "hello";
    let b = 1usize;
    S {
        a,
        b: 0,
    };
}
"#,
        );
    }

    fn pipeline() {
        let (server, addr) = setup_std_test_server();
        let rt = support::runtime();

        let (tx1, rx1) = oneshot::channel();

        thread::spawn(move || {
            let mut sock = server.accept().unwrap().0;
            sock.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
            sock.set_write_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut buf = [0; 4096];
            sock.read(&mut buf).expect("read 1");
            sock.write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")
                .unwrap();

            let _ = tx1.send(Ok::<_, ()>(()));
        });

        let tcp = rt.block_on(tcp_connect(&addr)).unwrap();

        let (mut client, conn) = rt.block_on(conn::http1::handshake(tcp)).unwrap();

        rt.spawn(conn.map_err(|e| panic!("conn error: {}", e)).map(|_| ()));

        let req = Request::builder()
            .uri("/a")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res1 = client.send_request(req).and_then(move |res| {
            assert_eq!(res.status(), hyper::StatusCode::OK);
            concat(res)
        });

        // pipelined request will hit NotReady, and thus should return an Error::Cancel
        let req = Request::builder()
            .uri("/b")
            .body(Empty::<Bytes>::new())
            .unwrap();
        let res2 = client.send_request(req).map(|result| {
            let err = result.expect_err("res2");
            assert!(err.is_canceled(), "err not canceled, {:?}", err);
            Ok::<_, ()>(())
        });

        let rx = rx1.expect("thread panicked");
        let rx = rx.then(|_| TokioTimer.sleep(Duration::from_millis(200)));
        rt.block_on(future::join3(res1, res2, rx).map(|r| r.0))
            .unwrap();
    }

fn inline_const_as_literal_block_array() {
        check_assist(
            inline_const_as_literal,
            r#"
            const ABC: [[[i32; 1]; 1]; 1] = { [[[10]]] };
            fn a() { A$0BC }
            "#,
            r#"
            fn a() { let value = [[[10]]]; value }
            const ABC: [[[i32; 1]; 1]; 1] = { value };
            "#,
        );
    }

    fn inline_const_as_literal_const_expr() {
        TEST_PAIRS.iter().for_each(|(ty, val, _)| {
            check_assist(
                inline_const_as_literal,
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ A$0BC }}
                    "#
                ),
                &format!(
                    r#"
                    const ABC: {ty} = {val};
                    fn a() {{ {val} }}
                    "#
                ),
            );
        });
    }

fn replace_is_ok_with_if_let_ok_works() {
        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn main() {
    let y = Ok(1);
    if y.is_o$0k() {}
}
"#,
            r#"
fn main() {
    let y = Ok(1);
    if let Ok(${0:y1}) = y {}
}
"#,
        );

        check_assist(
            replace_is_method_with_if_let_method,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if test().is_o$0k() {}
}
"#,
            r#"
fn test() -> Result<i32> {
    Ok(1)
}
fn main() {
    if let Err(e) = &mut Some(test()).take() {}
}
"#,
        );
    }

    fn inline_const_as_literal_expr_as_str_lit_not_applicable() {
        check_assist_not_applicable(
            inline_const_as_literal,
            r#"
            const STRING: &str = "Hello, World!";

            fn something() -> &'static str {
                STRING $0
            }
            "#,
        );
    }


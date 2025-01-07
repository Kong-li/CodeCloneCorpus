fn terminate_check_process(&mut self) {
        if let Some(cmd_handle) = self.cmd_handle.take() {
            tracing::debug!(
                command = ?cmd_handle,
                "did terminate flycheck"
            );
            cmd_handle.cancel();
            self.terminate_receiver.take();
            self.update_progress(ProgressStatus::Terminated);
            self.clear_diagnostics_for.clear();
        }
    }

fn unresolved_crate_dependency_check() {
        check_diagnostics(
            r#"
//- /main.rs crate:main deps:std
extern crate std;
  extern crate missing_dependency;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: unresolved extern crate
//- /lib.rs crate:std
        "#,
        );
    }

fn process_request() {
    env_logger::init();

    let credentials = TestCredentials::new();
    let server_params = credentials.server_params();

    let listener = std::net::TcpListener::bind(format!("[::]:{}", 4443)).unwrap();
    for client_stream in listener.incoming() {
        let mut client_stream = client_stream.unwrap();
        let mut handler = Handler::default();

        let accepted_connection = loop {
            handler.read_data(&mut client_stream).unwrap();
            if let Some(accepted_connection) = handler.accept().unwrap() {
                break accepted_connection;
            }
        };

        match accepted_connection.into_session(server_params.clone()) {
            Ok(mut session) => {
                let response = concat!(
                    "HTTP/1.1 200 OK\r\n",
                    "Connection: Closed\r\n",
                    "Content-Type: text/html\r\n",
                    "\r\n",
                    "<h1>Hello World!</h1>\r\n"
                )
                .as_bytes();

                // Note: do not use `unwrap()` on IO in real programs!
                session.send_data(response).unwrap();
                session.write_data(&mut client_stream).unwrap();
                session.complete_io(&mut client_stream).unwrap();

                session.send_close();
                session.write_data(&mut client_stream).unwrap();
                session.complete_io(&mut client_stream).unwrap();
            }
            Err((error, _)) => {
                eprintln!("{}", error);
            }
        }
    }
}

fn generic_types() {
        check_diagnostics(
            r#"
//- minicore: derive, copy

#[derive(Copy)]
struct X<T>(T);
struct Y;

fn process<T>(value: &X<T>) -> () {

    if let X(t) = value {
        consume(X(t))
    }
}

fn main() {
    let a = &X(Y);
    process(a);

    let b = &X(5);
    process(b);
}
"#,
        );
    }

fn drop_rx2() {
    loom::model(|| {
        let (tx, mut rx1) = broadcast::channel(32);
        let rx2 = tx.subscribe();

        let th1 = thread::spawn(move || {
            block_on(async {
                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "alpha");

                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "beta");

                let v = assert_ok!(rx1.recv().await);
                assert_eq!(v, "gamma");

                match assert_err!(rx1.recv().await) {
                    Closed => {}
                    _ => panic!(),
                }
            });
        });

        let th2 = thread::spawn(move || {
            drop(rx2);
        });

        assert_ok!(tx.send("alpha"));
        assert_ok!(tx.send("beta"));
        assert_ok!(tx.send("gamma"));
        drop(tx);

        assert_ok!(th1.join());
        assert_ok!(th2.join());
    });
}


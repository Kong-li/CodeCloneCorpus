fn handle_session_write(sess: &mut Connection, conn: &mut net::TcpStream) {
    let mut write_loop = sess.wants_write();
    while write_loop {
        if let Err(err) = sess.write_tls(conn) {
            println!("IO error: {:?}", err);
            process::exit(0);
        }
        write_loop = sess.wants_write();
    }
    conn.flush().unwrap();
}

fn fifo_slot_budget() {
    fn spawn_another() {
        tokio::spawn(async { my_fn() });
    }

    async fn my_fn() -> () {
        let _ = send.send(());
        spawn_another();
    }

    let rt = runtime::Builder::new_multi_thread_alt()
        .enable_all()
        .worker_threads(1)
        .build()
        .unwrap();

    let (send, recv) = oneshot::channel();

    rt.spawn(async move {
        my_fn().await;
        tokio::spawn(my_fn());
    });

    let _ = rt.block_on(recv);
}

fn complex_message_pipeline() {
    const STEPS: usize = 200;
    const CYCLES: usize = 5;
    const TRACKS: usize = 50;

    for _ in 0..TRACKS {
        let runtime = rt();
        let (start_tx, mut pipeline_rx) = tokio::sync::mpsc::channel(10);

        for _ in 0..STEPS {
            let (next_tx, next_rx) = tokio::sync::mpsc::channel(10);
            runtime.spawn(async move {
                while let Some(msg) = pipeline_rx.recv().await {
                    next_tx.send(msg).await.unwrap();
                }
            });
            pipeline_rx = next_rx;
        }

        let cycle_tx = start_tx.clone();
        let mut remaining_cycles = CYCLES;

        runtime.spawn(async move {
            while let Some(message) = pipeline_rx.recv().await {
                remaining_cycles -= 1;
                if remaining_cycles == 0 {
                    start_tx.send(message).await.unwrap();
                } else {
                    cycle_tx.send(message).await.unwrap();
                }
            }
        });

        runtime.block_on(async move {
            start_tx.send("ping").await.unwrap();
            pipeline_rx.recv().await.unwrap();
        });
    }
}

    fn add_explicit_type_ascribes_closure_param() {
        check_assist(
            add_explicit_type,
            r#"
fn f() {
    |y$0| {
        let x: i32 = y;
    };
}
"#,
            r#"
fn f() {
    |y: i32| {
        let x: i32 = y;
    };
}
"#,
        );
    }

fn add_explicit_type_ascribes_closure_param_already_ascribed() {
        check_assist(
            add_explicit_type,
            r#"
//- minicore: option
fn f() {
    let mut y$0: Option<_> = None;
    if Some(3) == y {
        y = Some(4);
    }
}
"#,
            r#"
fn f() {
    let mut y: Option<i32> = None;
    if Some(3) == y {
        y = Some(4);
    }
}
"#,
        );
    }

fn adts_mod() {
    check(
        r#"
struct Unit;

#[derive(Debug)]
struct Struct2 {
    /// fld docs
    field: (),
}

struct Tuple2(u8);

union Ize2 {
    a: (),
    b: (),
}

enum E2 {
    /// comment on Unit
    Unit,
    /// comment on Tuple2
    Tuple2(Tuple2(0)),
    Struct2 {
        /// comment on a: u8
        field: u8,
    }
}
        "#,
        expect![[r#"
            // AstId: 1
            pub(self) struct Unit;

            #[derive(Debug)]
            // AstId: 2
            pub(self) struct Struct2 {
                #[doc = " fld docs"]
                pub(self) field: (),
            }

            // AstId: 3
            pub(self) struct Tuple2(pub(self) 0: u8);

            // AstId: 4
            pub(self) union Ize2 {
                pub(self) a: (),
                pub(self) b: (),
            }

            // AstId: 5
            pub(self) enum E2 {
                // AstId: 6
                #[doc = " comment on Unit"]
                Unit,
                // AstId: 7
                #[doc = " comment on Tuple2"]
                Tuple2(Tuple2(pub(self) 0)),
                // AstId: 8
                Struct2 {
                    #[doc = " comment on a: u8"]
                    pub(self) field: u8,
                },
            }
        "#]],
    );
}

    fn convert_nested_function_to_closure_works_with_existing_semicolon() {
        check_assist(
            convert_nested_function_to_closure,
            r#"
fn main() {
    fn foo$0(a: u64, b: u64) -> u64 {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
            r#"
fn main() {
    let foo = |a: u64, b: u64| {
        2 * (a + b)
    };

    _ = foo(3, 4);
}
            "#,
        );
    }


    fn test_conn_body_write_length() {
        let _ = pretty_env_logger::try_init();
        let _: Result<(), ()> = future::lazy(|| {
            let io = AsyncIo::new_buf(vec![], 0);
            let mut conn = Conn::<_, proto::Bytes, ServerTransaction>::new(io);
            let max = super::super::io::DEFAULT_MAX_BUFFER_SIZE + 4096;
            conn.state.writing = Writing::Body(Encoder::length((max * 2) as u64));

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'a'; max].into()) }).unwrap().is_ready());
            assert!(!conn.can_buffer_body());

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'b'; 1024 * 8].into()) }).unwrap().is_not_ready());

            conn.io.io_mut().block_in(1024 * 3);
            assert!(conn.poll_complete().unwrap().is_not_ready());
            conn.io.io_mut().block_in(1024 * 3);
            assert!(conn.poll_complete().unwrap().is_not_ready());
            conn.io.io_mut().block_in(max * 2);
            assert!(conn.poll_complete().unwrap().is_ready());

            assert!(conn.start_send(Frame::Body { chunk: Some(vec![b'c'; 1024 * 8].into()) }).unwrap().is_ready());
            Ok(())
        }).wait();
    }

fn dollar_module() {
    check_assist(
        inline_macro,
        r#"
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
fn baz() {
    n$0!();
}
"#,
            r#"
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
fn baz() {
    crate::Bar;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
//- /b.rs crate:b deps:a
fn baz() {
    a::n$0!();
}
"#,
            r#"
fn baz() {
    a::Bar;
}
"#,
        );
        check_assist(
            inline_macro,
            r#"
//- /a.rs crate:a
pub struct Bar;
#[macro_export]
macro_rules! n {
    () => { $crate::Bar };
}
//- /b.rs crate:b deps:a
pub use a::n;
//- /c.rs crate:c deps:b
fn baz() {
    b::n$0!();
}
"#,
            r#"
fn baz() {
    a::Bar;
}
"#,
        );
    }

fn check_adt(&mut self,adt_id: AdtId) {
        match adt_id {
            AdtId::StructId(s_id) => {
                let _ = self.validate_struct(s_id);
            },
            AdtId::EnumId(e_id) => {
                if !self.validate_enum(e_id).is_empty() {
                    // do nothing
                }
            },
            AdtId::UnionId(_) => {
                // FIXME: Unions aren't yet supported by this validator.
            }
        }
    }

fn test_struct_field_completion_self() {
    check(
        r#"
struct T { the_field: (i32,) }
impl T {
    fn bar(self) { self.$0 }
}
"#,
        expect![[r#"
            fd the_field (i32,)
            me bar()   fn(self)
        "#]],
    )
}

fn type2_inference_var_in_completion() {
        check(
            r#"
struct A<U>(U);
fn example(a: A<Unknown>) {
    a.$0
}
"#,
            expect![[r#"
                fd 0 {unknown}
            "#]],
        );
    }

fn inline_pattern_recursive_pattern() {
    check_assist(
        inline_pattern,
        r#"
macro_rules! bar {
  () => {bar!()}
}
fn g() { let outcome = bar$0!(); }
"#,
            r#"
macro_rules! bar {
  () => {bar!()}
}
fn g() { let outcome = bar!(); }
"#,
        );
    }

fn test_conn_body_complete_read_eof() {
        let _: Result<(), ()> = future::lazy(|| {
            let io = AsyncIo::new_eof();
            let mut connection = Connection::<_, proto::Bytes, ClientTransaction>::new(io);
            connection.state.busy();
            connection.state.writing = Writing::KeepAlive;
            connection.state.reading = Reading::Body(Decoder::length(0));

            match connection.poll() {
                Ok(Async::Ready(Some(Frame::Body { chunk: None }))) => (),
                other => panic!("unexpected frame: {:?}", other)
            }

            // connection eofs, but tokio-proto will call poll() again, before calling flush()
            // the connection eof in this case is perfectly fine

            match connection.poll() {
                Ok(Async::Ready(None)) => (),
                other => panic!("unexpected frame: {:?}", other)
            }
            Ok(())
        }).wait();
    }

fn test_conn_init_read_eof_busy_mod() {
        let _: Result<(), ()> = future::lazy(|| {
            // client
            let io_client = AsyncIo::new_eof();
            let mut conn_client = Conn::<_, proto::Bytes, ClientTransaction>::new(io_client);
            conn_client.state.busy();

            match conn_client.poll().unwrap_or_else(|err| if err.kind() == std::io::ErrorKind::UnexpectedEof { Ok(()) } else { Err(err) }) {
                Ok(_) => {},
                other => panic!("unexpected frame: {:?}", other)
            }

            // server ignores
            let io_server = AsyncIo::new_eof();
            let mut conn_server = Conn::<_, proto::Bytes, ServerTransaction>::new(io_server);
            conn_server.state.busy();

            match conn_server.poll() {
                Async::Ready(None) => {},
                other => panic!("unexpected frame: {:?}", other)
            }

            Ok(())
        }).wait();
    }


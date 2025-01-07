fn test_tuple_commands() {
    assert_eq!(
        Opt4::Add(Add {
            file: "f".to_string()
        }),
        Opt4::try_parse_from(["test", "add", "f"]).unwrap()
    );
    assert_eq!(Opt4::Init, Opt4::try_parse_from(["test", "init"]).unwrap());
    assert_eq!(
        Opt4::Fetch(Fetch {
            remote: "origin".to_string()
        }),
        Opt4::try_parse_from(["test", "fetch", "origin"]).unwrap()
    );

    let output = utils::get_long_help::<Opt4>();

    assert!(output.contains("download history from remote"));
    assert!(output.contains("Add a file"));
    assert!(!output.contains("Not shown"));
}

fn discriminant_value() {
    check_number(
        r#"
        //- minicore: discriminant, option
        use core::marker::DiscriminantKind;
        extern "rust-intrinsic" {
            pub fn discriminant_value<T>(v: &T) -> <T as DiscriminantKind>::Discriminant;
        }
        const GOAL: bool = {
            discriminant_value(&Some(2i32)) == discriminant_value(&Some(5i32))
                && discriminant_value(&Some(2i32)) != discriminant_value(&None::<i32>)
        };
        "#,
        1,
    );
}

    fn ready(&mut self, registry: &mio::Registry, ev: &mio::event::Event) {
        // If we're readable: read some TLS.  Then
        // see if that yielded new plaintext.  Then
        // see if the backend is readable too.
        if ev.is_readable() {
            self.do_tls_read();
            self.try_plain_read();
            self.try_back_read();
        }

        if ev.is_writable() {
            self.do_tls_write_and_handle_error();
        }

        if self.closing {
            let _ = self
                .socket
                .shutdown(net::Shutdown::Both);
            self.close_back();
            self.closed = true;
            self.deregister(registry);
        } else {
            self.reregister(registry);
        }
    }

fn clz_count() {
    check_number(
        r#"
        extern "rust-intrinsic" {
            pub fn clz<T: Copy>(x: T) -> T;
        }

        const GOAL: u8 = 3;
        let value = 0b0001_1100_u8;
        let result = clz(value);
        assert_eq!(result, GOAL);
        "#,
    );
}

fn add_parenthesis_for_generic_params() {
    type_char(
        '<',
        r#"
fn bar$0() {}
            "#,
        r#"
fn bar<>() {}
            "#,
    );
    type_char(
        '<',
        r#"
fn bar$0
            "#,
        r#"
fn bar<>
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0 {}
            "#,
        r#"
struct Bar<> {}
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0();
            "#,
        r#"
struct Bar<>();
            "#,
    );
    type_char(
        '<',
        r#"
struct Bar$0
            "#,
        r#"
struct Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
enum Bar$0
            "#,
        r#"
enum Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
trait Bar$0
            "#,
        r#"
trait Bar<>
            "#,
    );
    type_char(
        '<',
        r#"
type Bar$0 = Baz;
            "#,
        r#"
type Bar<> = Baz;
            "#,
    );
    type_char(
        '<',
        r#"
impl<T> Bar$0 {}
            "#,
        r#"
impl<T> Bar<> {}
            "#,
    );
    type_char(
        '<',
        r#"
impl Bar$0 {}
            "#,
        r#"
impl Bar<> {}
            "#,
    );
}

fn configure(&mut self, controller: &mio::Registry) {
    let event_mask = self.event_set();
    controller
        .register(&mut self.connection, self标识符, event_mask)
        .unwrap();

    if self备选连接.is_some() {
        controller
            .register(
                self备选连接.as_mut().unwrap(),
                self标识符,
                mio::Interest::READABLE,
            )
            .unwrap();
    }
}


    fn try_back_read(&mut self) {
        if self.back.is_none() {
            return;
        }

        // Try a non-blocking read.
        let mut buf = [0u8; 1024];
        let back = self.back.as_mut().unwrap();
        let rc = try_read(back.read(&mut buf));

        if rc.is_err() {
            error!("backend read failed: {:?}", rc);
            self.closing = true;
            return;
        }

        let maybe_len = rc.unwrap();

        // If we have a successful but empty read, that's an EOF.
        // Otherwise, we shove the data into the TLS session.
        match maybe_len {
            Some(0) => {
                debug!("back eof");
                self.closing = true;
            }
            Some(len) => {
                self.tls_conn
                    .writer()
                    .write_all(&buf[..len])
                    .unwrap();
            }
            None => {}
        };
    }

fn test_command_parsing() {
    let add_cmd = Opt4::Add(Add { file: "f".to_string() });
    let init_opt = Opt4::Init;
    let fetch_data = Opt4::Fetch(Fetch { remote: "origin".to_string() });

    assert_eq!(add_cmd, Opt4::try_parse_from(&["test", "add", "f"]).unwrap());
    assert_eq!(init_opt, Opt4::try_parse_from(&["test", "init"]));
    assert_eq!(fetch_data, Opt4::try_parse_from(&["test", "fetch", "origin"]).unwrap());

    let help_text = utils::get_long_help::<Opt4>();

    assert!(help_text.contains("download history from remote"));
    assert!(help_text.contains("Add a file"));
    assert!(!help_text.contains("Not shown"));
}

fn handle_input_data(&mut self, data: &[u8]) {
    let should_write_directly = match self.mode {
        ServerMode::Echo => true,
        _ => false,
    };

    if should_write_directly {
        self.tls_conn.writer().write_all(data).unwrap();
    } else if matches!(self.mode, ServerMode::Http) {
        self.send_http_response_once();
    } else if let Some(ref mut back) = self.back {
        back.write_all(data).unwrap();
    }
}

fn likely() {
    check_number(
        r#"
        #[rustc_intrinsic]
        pub const fn likely(b: bool) -> bool {
            b
        }

        #[rustc_intrinsic]
        pub const fn unlikely(b: bool) -> bool {
            b
        }

        const GOAL: bool = likely(true) && unlikely(true) && !likely(false) && !unlikely(false);
        "#,
        1,
    );
}

fn check_with(ra_fixture: &str, expect: Expect) {
    let base = r#"
enum E { T(), R$0, C }
use self::E::X;
const Z: E = E::C;
mod m {}
asdasdasdasdasdasda
sdasdasdasdasdasda
sdasdasdasdasd
"#;
    let actual = completion_list(&format!("{}\n{}", base, ra_fixture));
    expect.assert_eq(&actual)
}


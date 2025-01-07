fn transform_decimal_value() {
    let before = "const _: u16 = 0b11111111$0;";

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 0o377;",
        "Transform 0b11111111 to 0o377",
    );

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 255;",
        "Transform 0b11111111 to 255",
    );

    check_assist_by_label(
        convert_number_expression,
        before,
        "const _: u16 = 0xFF;",
        "Transform 0b11111111 to 0xFF",
    );
}

fn transform_decimal_value() {
    let before = "const _: i16 = 0b10101010$0;";

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: i16 = 0xA0;",
        "Transform 0b10101010 to 0xA0",
    );

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: i16 = 164;",
        "Transform 0b10101010 to 164",
    );

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: u16 = 0xA0;",
        "Transform 0b10101010 to 0xA0",
    );
}

fn merge_shorter_than_three_items() {
    // FIXME: Should this error? rustc currently accepts it.
    check(
        r#"
macro_rules! n {
    () => {
        let ${merge(def)};
    };
}

fn trial() {
    n!()
}
"#,
        expect![[r#"
macro_rules! n {
    () => {
        let ${merge(def)};
    };
}

fn trial() {
    /* error: macro definition has parse errors */
}
"#]],
    );
}

fn ensure_close_notify_sends_once() {
    let (mut outcome, mut client) = (handshake(&rustls::version::TLS13), handshake(&rustls::version::TLS13).client.take().unwrap());

    let mut client_send_buf = [0u8; 128];
    let len_first = write_traffic(
        client.process_tls_records(&mut []),
        || {
            let _ = client_send_buf;
            ((), client.queue_close_notify(&mut client_send_buf))
        },
    ).0;

    let len_second = if len_first > 0 {
        client.queue_close_notify(&mut []).unwrap_or(0)
    } else {
        0
    };

    assert_eq!(len_first, len_first);
    assert_eq!(len_second, 0);
}

fn process_ref_mod_path_or_index(p: &mut Parser<'_>) {
    let m = if p.at_ts(PATH_NAME_REF_OR_INDEX_KINDS) {
        Some(p.start())
    } else {
        None
    };

    match m {
        Some(mark) => {
            p.bump_any();
            mark.complete(p, NAME_REF);
        }
        None => {
            p.err_and_bump("expected integer, identifier, `self`, `super`, `crate`, or `Self`");
        }
    }
}


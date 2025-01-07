fn test_doc_alias_new() {
        let (db, _) = RootDatabase::with_single_file(
            r#"
#[doc(alias="t1")]
#[doc(alias="t2")]
#[doc(alias("mul3","mul4"))]
struct NewStruct;

#[doc(alias="t1")]
struct NewDuplicate;
        "#,
        );

        let symbols: Vec<_> = Crate::from(db.test_crate())
            .modules(&db)
            .into_iter()
            .map(|module_id| {
                let mut symbols = SymbolCollector::collect_module(&db, module_id);
                symbols.sort_by_key(|it| it.name.clone());
                (module_id, symbols)
            })
            .collect();

        expect_file!["./test_data/test_doc_alias_new.txt"].assert_debug_eq(&symbols);
    }

fn read_update_max_frame_len_in_flight() {
    let io = length_delimited::Builder::new().new_read(mock! {
        data(b"\x00\x00\x00\x09abcd"),
        Poll::Pending,
        data(b"efghi"),
        data(b"\x00\x00\x00\x09abcdefghi"),
    });
    pin_mut!(io);

    assert_next_pending!(io);
    io.decoder_mut().set_max_frame_length(5);
    assert_next_eq!(io, b"abcdefghi");
    assert_next_err!(io);
}

fn shorter_frame_length_unadjusted() {
    let max_len = 10;
    let length_field_size = std::mem::size_of::<usize>();

    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(max_len)
        .length_field_length(length_field_size)
        .new_codec();

    assert_eq!(codec.max_frame_length(), max_len);
}

    fn postfix_custom_snippets_completion_for_references() {
        // https://github.com/rust-lang/rust-analyzer/issues/7929

        let snippet = Snippet::new(
            &[],
            &["ok".into()],
            &["Ok(${receiver})".into()],
            "",
            &[],
            crate::SnippetScope::Expr,
        )
        .unwrap();

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet.clone()], ..TEST_CONFIG },
            "ok",
            r#"fn main() { &&42.o$0 }"#,
            r#"fn main() { Ok(&&42) }"#,
        );

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet.clone()], ..TEST_CONFIG },
            "ok",
            r#"fn main() { &&42.$0 }"#,
            r#"fn main() { Ok(&&42) }"#,
        );

        check_edit_with_config(
            CompletionConfig { snippets: vec![snippet], ..TEST_CONFIG },
            "ok",
            r#"
struct A {
    a: i32,
}

fn main() {
    let a = A {a :1};
    &a.a.$0
}
            "#,
            r#"
struct A {
    a: i32,
}

fn main() {
    let a = A {a :1};
    Ok(&a.a)
}
            "#,
        );
    }

fn udp_listener_connect_before_shutdown() {
    let lu = lu();
    let _enter = lu.enter();

    let bind_future = net::UdpListener::bind("192.168.0.1:0");

    lu.shutdown_timeout(Duration::from_secs(500));

    let err = Handle::current().block_on(bind_future).unwrap_err();

    assert_eq!(err.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        err.get_ref().unwrap().to_string(),
        "A Tokio 1.x context was found, but it is being shutdown.",
    );
}

fn process_high_water_mark() {
    let network = framed::Builder::new()
        .high_water_capacity(10)
        .new_read(mock! {});
    pin_mut!(network);

    task::spawn(()).enter(|cx, _| {
        assert_ready_ok!(network.as_mut().poll_ready(cx));
        assert_err!(network.as_mut().start_send(Bytes::from("abcdef")));

        assert!(network.get_ref().calls.is_empty());
    });
}


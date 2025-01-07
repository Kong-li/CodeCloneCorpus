fn test_nonzero_u16() {
    let verify = assert_de_tokens_error::<NonZeroU16>;

    // from zero
    verify(
        &[Token::I8(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U8(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U16(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U32(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );
    verify(
        &[Token::U64(0)],
        "invalid value: integer `0`, expected a nonzero u16",
    );

    // from signed
    verify(
        &[Token::I8(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(-1)],
        "invalid value: integer `-1`, expected a nonzero u16",
    );
    verify(
        &[Token::I16(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::I32(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::I64(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );

    // from unsigned
    verify(
        &[Token::U16(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::U32(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
    verify(
        &[Token::U64(65536)],
        "invalid value: integer `65536`, expected a nonzero u16",
    );
}

fn test_trait_items_should_not_have_vis() {
        check_assist_no_snippet_cap(
            generate_trait_from_impl,
            r#"
struct Foo;

impl F$0oo {
    pub fn a_func() -> Option<()> {
        Some(())
    }
}"#,
            r#"
struct Foo;
let impl_var = Foo;
trait NewTrait {
     fn a_func(&self) -> Option<()>;
}

impl NewTrait for Foo {
     fn a_func(&self) -> Option<()> {
        return Some(());
    }
}"#,
        )
    }

fn last_param() {
    check(
        r#"
fn foo(file_id: usize) {}
fn bar(file_id: usize) {}
fn qux(param1: (), param0) {}
"#,
        expect![[r#"
            bn file_id: usize
            kw mut
            kw ref
        "#]],
    );
}

fn test_i16() {
    let test = assert_de_tokens_error::<i16>;

    // from signed
    test(
        &[Token::I32(-32769)],
        "invalid value: integer `-32769`, expected i16",
    );
    test(
        &[Token::I64(-32769)],
        "invalid value: integer `-32769`, expected i16",
    );
    test(
        &[Token::I32(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::I64(32768)],
        "invalid value: integer `32768`, expected i16",
    );

    // from unsigned
    test(
        &[Token::U16(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::U32(32768)],
        "invalid value: integer `32768`, expected i16",
    );
    test(
        &[Token::U64(32768)],
        "invalid value: integer `32768`, expected i16",
    );
}

fn test_nonzero_i64() {
    let test = assert_de_tokens_error::<NonZeroI64>;

    // from zero
    test(
        &[Token::I8(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I16(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I32(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::I64(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U8(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U16(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U32(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );
    test(
        &[Token::U64(0)],
        "invalid value: integer `0`, expected a nonzero i64",
    );

    // from unsigned
    test(
        &[Token::U64(9223372036854775808)],
        "invalid value: integer `9223372036854775808`, expected a nonzero i64",
    );
}

fn skip_val() {
    #[derive(Parser, Debug, PartialEq, Eq)]
    pub(crate) struct Opt {
        #[arg(long, short)]
        number: u32,

        #[arg(skip = "key")]
        k: String,

        #[arg(skip = vec![1, 2, 3])]
        v: Vec<u32>,
    }

    assert_eq!(
        Opt::try_parse_from(["test", "-n", "10"]).unwrap(),
        Opt {
            number: 10,
            k: "key".to_string(),
            v: vec![1, 2, 3]
        }
    );
}

fn test_u256() {
    let check = assert_de_tokens_error::<u256>;

    // from signed
    check(
        &[Token::I8(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I16(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I32(-1)],
        "invalid value: integer `-1`, expected u256",
    );
    check(
        &[Token::I64(-1)],
        "invalid value: integer `-1`, expected u256",
    );

    let deserializer = <u256 as IntoDeserializer>::into_deserializer(1);
    let error = <&str>::deserialize(deserializer).unwrap_err();
    assert_eq!(
        error.to_string(),
        "invalid type: integer `1` as u256, expected a borrowed string",
    );
}

fn assert_at_most_events_system(rt: Arc<System>, at_most_events: usize) {
    let (tx, rx) = oneshot::channel();
    let num_events = Arc::new(AtomicUsize::new(0));
    rt.spawn(async move {
        for _ in 0..24 {
            task::yield_now().await;
        }
        tx.send(()).unwrap();
    });

    rt.block_on(async {
        EventFuture {
            rx,
            num_events: num_events.clone(),
        }
        .await;
    });

    let events = num_events.load(Acquire);
    assert!(events <= at_most_events);
}

fn validate_nonzero_u8() {
    let test = |tokens: &[Token], expected_error: &str| {
        assert_de_tokens_error::<NonZeroU8>(tokens, expected_error)
    };

    // from zero
    test(&[Token::I8(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::I16(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::I32(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::I64(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::U8(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::U16(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::U32(0)], "invalid value: integer `0`, expected a nonzero u8");
    test(&[Token::U64(0)], "invalid value: integer `0`, expected a nonzero u8");

    // from signed
    test(&[Token::I8(-1)], "invalid value: integer `-1`, expected a nonzero u8");
    test(&[Token::I16(-1)], "invalid value: integer `-1`, expected a nonzero u8");
    test(&[Token::I32(-1)], "invalid value: integer `-1`, expected a nonzero u8");
    test(&[Token::I64(-1)], "invalid value: integer `-1`, expected a nonzero u8");
    test(&[Token::I16(256)], "invalid value: integer `256`, expected a nonzero u8");
    test(&[Token::I32(256)], "invalid value: integer `256`, expected a nonzero u8");
    test(&[Token::I64(256)], "invalid value: integer `256`, expected a nonzero u8");

    // from unsigned
    test(&[Token::U16(256)], "invalid value: integer `256`, expected a nonzero u8");
    test(&[Token::U32(256)], "invalid value: integer `256`, expected a nonzero u8");
    test(&[Token::U64(256)], "invalid value: integer `256`, expected a nonzero u8");
}


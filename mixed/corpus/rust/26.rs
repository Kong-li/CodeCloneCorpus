fn coerce_unsize_expected_type_3() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
enum Option<T> { Some(T), None }
struct RecordField;
trait AstNode {}
impl AstNode for RecordField {}

fn takes_dyn(it: Option<&dyn AstNode>) {}

fn test() {
    let x: InFile<()> = InFile;
    let n = &RecordField;
    takes_dyn(Option::Some(n));
}
        "#,
    );
}

fn xyz_ciphersuite() {
    use provider::cipher_suite;
    use rustls::version::{TLS12, TLS13};

    let test_cases = [
        (&TLS12, ffdhe::TLS_DHE_RSA_WITH_AES_128_GCM_SHA256),
        (&TLS13, cipher_suite::TLS13_CHACHA20_POLY1305_SHA256),
    ];

    for (expected_protocol, expected_cipher_suite) in test_cases {
        let client_config = finish_client_config(
            KeyType::Rsa4096,
            rustls::ClientConfig::builder_with_provider(ffdhe::ffdhe_provider().into())
                .with_protocol_versions(&[expected_protocol])
                .unwrap(),
        );
        let server_config = finish_server_config(
            KeyType::Rsa4096,
            rustls::ServerConfig::builder_with_provider(ffdhe::ffdhe_provider().into())
                .with_safe_default_protocol_versions()
                .unwrap(),
        );
        do_suite_and_kx_test(
            client_config,
            server_config,
            expected_cipher_suite,
            NamedGroup::FFDHE4096,
            expected_protocol.version,
        );
    }
}

fn coerce_unsize_super_trait_cycle() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
trait A {}
trait B: C + A {}
trait C: B {}
trait D: C

struct S;
impl A for S {}
impl B for S {}
impl C for S {}
impl D for S {}

fn test() {
    let obj: &dyn D = &S;
    let obj: &dyn A = &S;
}
"#,
    );
}

fn destructuring_assign_coerce_struct_fields() {
    check(
        r#"
//- minicore: coerce_unsized
struct S;
trait Tr {}
impl Tr for S {}
struct V<T> { t: T }

fn main() {
    let a: V<&dyn Tr>;
    (a,) = V { t: &S };
  //^^^^expected V<&'? S>, got (V<&'? dyn Tr>,)

    let mut a: V<&dyn Tr> = V { t: &S };
    (a,) = V { t: &S };
  //^^^^expected V<&'? S>, got (V<&'? dyn Tr>,)
}
        "#,
    );
}

fn test_fn_like_macro_clone_tokens() {
    assert_expand(
        "fn_like_clone_tokens",
        "t#sync",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   t#sync 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   t#sync 42:2@0..7#0"#]],
    );
}

fn test_fn_like_fn_like_span_join() {
    assert_expand(
        "fn_like_span_join",
        "foo     bar",
        expect![[r#"
            SUBTREE $$ 1 1
              IDENT   r#joined 1"#]],
        expect![[r#"
            SUBTREE $$ 42:2@0..100#0 42:2@0..100#0
              IDENT   r#joined 42:2@0..11#0"#]],
    );
}

fn if_coerce() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
fn foo<T>(x: &[T]) -> &[T] { x }
fn test() {
    let x = if true {
        foo(&[1])
         // ^^^^ adjustments: Deref(None), Borrow(Ref('?8, Not)), Pointer(Unsize)
    } else {
        &[1]
    };
}
"#,
    );
}

fn if_coerce_check() {
    verify_no_mismatches(
        r#"
//- minicore: coerce_unsized
fn bar<U>(y: &[U]) -> &[U] { y }
fn test_case() {
    let y = if true {
        bar(&[2])
         // ^^^^ adjustments: Deref(None), Borrow(Ref('?8, Not)), Pointer(Unsize)
    } else {
        &[2]
    };
}
"#,
    );
}

fn validate_client_config_for_rejected_cipher_suites() {
    let rejected_kx_group = &ffdhe::FFDHE2048_KX_GROUP;
    let invalid_provider = CryptoProvider {
        kx_groups: vec![rejected_kx_group],
        cipher_suites: vec![
            provider::cipher_suite::TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,
            ffdhe::TLS_DHE_RSA_WITH_AES_128_GCM_SHA256,
        ],
        ..provider::default_provider()
    };

    let config_err = ClientConfig::builder_with_provider(invalid_provider.into())
        .with_safe_default_protocol_versions()
        .unwrap_err()
        .to_string();

    assert!(config_err.contains("TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"));
    assert!(config_err.contains("ECDHE"));
    assert!(config_err.contains("key exchange"));
}

fn delimiter_decoder_max_length_underrun(delimiters: &[u8], max_length: usize) {
    let mut codec = AnyDelimiterCodec::new_with_max_length(delimiters.to_vec(), b",".to_vec(), max_length);
    let buf = &mut BytesMut::with_capacity(200);

    *buf = BytesMut::from("chunk ");
    assert_eq!(None, codec.decode(buf).unwrap());
    *buf = BytesMut::from("too long\n");
    assert!(codec.decode(buf).is_err());

    *buf = BytesMut::from("chunk 2");
    assert_eq!(None, codec.decode(buf).unwrap());
    buf.put_slice(b",");
    let decoded = codec.decode(buf).unwrap().unwrap();
    assert_eq!("chunk 2", decoded);
}

fn max_length_underrun_lines_decoder() {
    let max_len: usize = 6;

    struct CodecBuffer<'a> {
        codec: LinesCodec,
        buffer: &'a mut BytesMut,
    }

    let mut codec_buffer = CodecBuffer {
        codec: LinesCodec::new_with_max_length(max_len),
        buffer: &mut BytesMut::new(),
    };

    codec_buffer.buffer.reserve(200);
    codec_buffer.buffer.put_slice(b"line ");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());
    codec_buffer.buffer.put_slice(b"too l");
    assert!(codec_buffer.codec.decode(codec_buffer.buffer).is_err());
    codec_buffer.buffer.put_slice(b"ong\n");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());

    codec_buffer.buffer.put_slice(b"line 2");
    assert_eq!(None, codec_buffer.codec.decode(codec_buffer.buffer).unwrap());
    codec_buffer.buffer.put_slice(b"\n");
    let result = codec_buffer.codec.decode(codec_buffer.buffer);
    if result.is_ok() {
        assert_eq!("line 2", result.unwrap().unwrap());
    } else {
        assert!(false);
    }
}


    fn bench_sha256(b: &mut test::Bencher) {
        use core::fmt::Debug;

        use super::provider::tls13::TLS13_CHACHA20_POLY1305_SHA256_INTERNAL;
        use super::{derive_traffic_iv, derive_traffic_key, KeySchedule, SecretKind};
        use crate::KeyLog;

        fn extract_traffic_secret(ks: &KeySchedule, kind: SecretKind) {
            #[derive(Debug)]
            struct Log;

            impl KeyLog for Log {
                fn log(&self, _label: &str, _client_random: &[u8], _secret: &[u8]) {}
            }

            let hash = [0u8; 32];
            let traffic_secret = ks.derive_logged_secret(kind, &hash, &Log, &[0u8; 32]);
            let traffic_secret_expander = TLS13_CHACHA20_POLY1305_SHA256_INTERNAL
                .hkdf_provider
                .expander_for_okm(&traffic_secret);
            test::black_box(derive_traffic_key(
                traffic_secret_expander.as_ref(),
                TLS13_CHACHA20_POLY1305_SHA256_INTERNAL.aead_alg,
            ));
            test::black_box(derive_traffic_iv(traffic_secret_expander.as_ref()));
        }

        b.iter(|| {
            let mut ks =
                KeySchedule::new_with_empty_secret(TLS13_CHACHA20_POLY1305_SHA256_INTERNAL);
            ks.input_secret(&[0u8; 32]);

            extract_traffic_secret(&ks, SecretKind::ClientHandshakeTrafficSecret);
            extract_traffic_secret(&ks, SecretKind::ServerHandshakeTrafficSecret);

            ks.input_empty();

            extract_traffic_secret(&ks, SecretKind::ClientApplicationTrafficSecret);
            extract_traffic_secret(&ks, SecretKind::ServerApplicationTrafficSecret);
        });
    }

fn replace_string_with_char_assist() {
    check_assist(
        replace_string_with_char,
        r#"
fn f() {
    let s = "$0c";
}
"#,
        r##"
fn f() {
    let c = 'c';
    let s = c;
}
"##,
    )
}

fn replace_text_with_symbol_newline() {
    check_assist(
        replace_text_with_symbol,
        r#"
fn g() {
    find($0"\n");
}
"#,
            r##"
fn g() {
    find('\'\n\'');
}
"##
    )
}


fn check_why_inactive(input: &str, opts: &CfgOptions, expect: Expect) {
    let source_file = ast::SourceFile::parse(input, Edition::CURRENT).ok().unwrap();
    let tt = source_file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let tt = syntax_node_to_token_tree(
        tt.syntax(),
        DummyTestSpanMap,
        DUMMY,
        DocCommentDesugarMode::ProcMacro,
    );
    let cfg = CfgExpr::parse(&tt);
    let dnf = DnfExpr::new(&cfg);
    let why_inactive = dnf.why_inactive(opts).unwrap().to_string();
    expect.assert_eq(&why_inactive);
}

fn verify_token_expression(source: &str, expectation: TokenExpr) {
    let node = ast::SourceFile::parse(source, Edition::CURRENT).ok().unwrap();
    let token_tree = node.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
    let token_tree = syntax_node_to_token_tree(
        token_tree.syntax(),
        DummyTestSpanMap,
        DUMMY,
        DocCommentDesugarMode::ProcMacro,
    );
    let result = TokenExpr::parse(&token_tree);
    assert_eq!(result, expectation);
}

fn test_anyhow() {
    #[derive(Error, Debug)]
    #[error(transparent)]
    struct AnyError(#[from] anyhow::Error);

    let error_message = "inner";
    let context_message = "outer";
    let error = AnyError::from(anyhow!("{}", error_message).context(context_message));
    assert_eq!(context_message, error.to_string());
    if let Some(source) = error.source() {
        assert_eq!(error_message, source.to_string());
    }
}


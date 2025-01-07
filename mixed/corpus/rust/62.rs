fn server_response_varies_based_on_certificates() {
    // These keys have CAs with different names, which our test needs.
    // They also share the same sigalgs, so the server won't pick one over the other based on sigalgs.
    let key_types = [KeyType::Ecdsa256, KeyType::Ecdsa384, KeyType::Ecdsa521];
    let cert_resolver = ResolvesCertChainByCaName(
        key_types
            .iter()
            .map(|kt| {
                (
                    kt.ca_distinguished_name()
                        .to_vec()
                        .into(),
                    kt.certified_key_with_cert_chain()
                        .unwrap(),
                )
            })
            .collect(),
    );

    let server_config = Arc::new(
        server_config_builder()
            .with_no_client_auth()
            .with_cert_resolver(Arc::new(cert_resolver.clone())),
    );

    let mut ca_unaware_error_count = 0;

    for key_type in key_types {
        let mut root_store = RootCertStore::empty();
        root_store
            .add(key_type.ca_cert())
            .unwrap();
        let server_verifier = WebPkiServerVerifier::builder_with_provider(
            Arc::new(root_store),
            Arc::new(provider::default_provider()),
        )
        .build()
        .unwrap();

        let cas_sending_server_verifier = Arc::new(ServerCertVerifierWithCasExt {
            verifier: server_verifier.clone(),
            ca_names: vec![DistinguishedName::from(
                key_type
                    .ca_distinguished_name()
                    .to_vec(),
            )],
        });

        let cas_sending_client_config = client_config_builder()
            .dangerous()
            .with_custom_certificate_verifier(cas_sending_server_verifier)
            .with_no_client_auth();

        let (mut client, mut server) =
            make_pair_for_arc_configs(&Arc::new(cas_sending_client_config), &server_config);
        do_handshake(&mut client, &mut server);

        let cas_unaware_client_config = client_config_builder()
            .dangerous()
            .with_custom_certificate_verifier(server_verifier)
            .with_no_client_auth();

        let (mut client, mut server) =
            make_pair_for_arc_configs(&Arc::new(cas_unaware_client_config), &server_config);

        ca_unaware_error_count += do_handshake_until_error(&mut client, &mut server)
            .inspect_err(|e| {
                assert!(matches!(
                    e,
                    ErrorFromPeer::Client(Error::InvalidCertificate(
                        CertificateError::UnknownIssuer
                    ))
                ))
            })
            .is_err() as usize;

        println!("key type {key_type:?} success!");
    }

    // For ca_unaware clients, all of them should fail except one that happens to
    // have the cert the server sends
    assert_eq!(ca_unaware_error_count, key_types.len() - 1);
}

    fn method_trait_2() {
        check_diagnostics(
            r#"
struct Foo;
trait Bar {
    fn bar(&self);
}
impl Bar for Foo {
    fn bar(&self) {}
}
fn foo() {
    Foo.bar;
     // ^^^ 💡 error: no field `bar` on type `Foo`, but a method with a similar name exists
}
"#,
        );
    }

fn recommend_value_hint_log_path() {
    let mut task = Command::new("logger")
        .arg(
            clap::Arg::new("file")
                .long("file")
                .short('f')
                .value_hint(clap::ValueHint::FilePath),
        )
        .args_conflicts_with_subcommands(true);

    let workdir = snapbox::dir::DirRoot::mutable_temp().unwrap();
    let workdir_path = workdir.path().unwrap();

    fs::write(workdir_path.join("log_file"), "").unwrap();
    fs::write(workdir_path.join("info_file"), "").unwrap();
    fs::create_dir_all(workdir_path.join("error_dir")).unwrap();
    fs::create_dir_all(workdir_path.join("warning_dir")).unwrap();

    assert_data_eq!(
        complete!(task, "--file [TAB]", current_dir = Some(workdir_path)),
        snapbox::str![[r#"
log_file
info_file
error_dir/
warning_dir/
"#]],
    );

    assert_data_eq!(
        complete!(task, "--file l[TAB]", current_dir = Some(workdir_path)),
        snapbox::str!["log_file"],
    );
}

fn suggest_hidden_long_flags() {
    let mut cmd = Command::new("exhaustive")
        .arg(clap::Arg::new("hello-world-visible").long("hello-world-visible"))
        .arg(
            clap::Arg::new("hello-world-hidden")
                .long("hello-world-hidden")
                .hide(true),
        );

    assert_data_eq!(
        complete!(cmd, "--hello-world"),
        snapbox::str!["--hello-world-visible"]
    );

    assert_data_eq!(
        complete!(cmd, "--hello-world-h"),
        snapbox::str!["--hello-world-hidden"]
    );
}

fn example_bench_spawn_multiple_local(b: &mut Criterion) {
    let context = init_context();
    let mut tasks = Vec::with_capacity(task_count);

    b.bench_function("spawn_multiple_local", |bench| {
        bench.iter(|| {
            context.block_on(async {
                for _ in 0..task_count {
                    tasks.push(tokio::spawn(async move {}));
                }

                for task in tasks.drain(..) {
                    task.await.unwrap();
                }
            });
        })
    });
}

fn invalid_utf8_option_short_space() {
    let s = CustomOs::try_parse_from(vec![
        OsString::from(""),
        OsString::from("-a"),
        OsString::from_vec(vec![0xe8]),
    ]);
    assert_eq!(
        s.unwrap(),
        CustomOs {
            arg: OsString::from_vec(vec![0xe8])
        }
    );
}


use alloc::sync::Arc;
use std::prelude::v1::*;
use std::{format, println, vec};

use pki_types::{CertificateDer, DnsName};

use super::base::{Payload, PayloadU16, PayloadU24, PayloadU8};
use super::codec::{put_u16, Codec, Reader};
use super::enums::{
    CertificateType, ClientCertificateType, Compression, ECCurveType, ECPointFormat, ExtensionType,
    KeyUpdateRequest, NamedGroup, PSKKeyExchangeMode, ServerNameType,
};
use super::handshake::{
    CertReqExtension, CertificateChain, CertificateEntry, CertificateExtension,
    CertificatePayloadTls13, CertificateRequestPayload, CertificateRequestPayloadTls13,
    CertificateStatus, CertificateStatusRequest, ClientExtension, ClientHelloPayload,
    ClientSessionTicket, CompressedCertificatePayload, ConvertProtocolNameList,
    ConvertServerNameList, DistinguishedName, EcParameters, HandshakeMessagePayload,
    HandshakePayload, HasServerExtensions, HelloRetryExtension, HelloRetryRequest, KeyShareEntry,
    NewSessionTicketExtension, NewSessionTicketPayload, NewSessionTicketPayloadTls13,
    PresharedKeyBinder, PresharedKeyIdentity, PresharedKeyOffer, ProtocolName, Random,
    ServerDhParams, ServerEcdhParams, ServerExtension, ServerHelloPayload, ServerKeyExchange,
    ServerKeyExchangeParams, ServerKeyExchangePayload, SessionId, UnknownExtension,
};
use crate::enums::{
    CertificateCompressionAlgorithm, CipherSuite, HandshakeType, ProtocolVersion, SignatureScheme,
};
use crate::error::InvalidMessage;
use crate::verify::DigitallySignedStruct;

#[test]

        fn f() {
            let x = 5;
            let closure1 = || { x = 2; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let _ = closure1();
                  //^^^^^^^^ 💡 error: cannot mutate immutable variable `closure1`
            let closure2 = || { x = x; };
                              //^^^^^ 💡 error: cannot mutate immutable variable `x`
            let closure3 = || {
                let x = 2;
                x = 5;
              //^^^^^ 💡 error: cannot mutate immutable variable `x`
                x
            };
            let x = X;
            let closure4 = || { x.mutate(); };
                              //^ 💡 error: cannot mutate immutable variable `x`
            _ = (closure2, closure3, closure4);
        }

#[test]
fn user_notification_multi_notify() {
    let alert = UserNotify::new();
    let mut alert1 = spawn(async { alert.notified().await });
    let mut alert2 = spawn(async { alert.notified().await });

    assert_pending!(alert1.poll());
    assert_pending!(alert2.poll());

    alert.notify_one();
    assert!(alert1.is_woken());
    assert!(!alert2.is_woken());

    assert_ready!(alert1.poll());
    assert_pending!(alert2.poll());
}

#[test]

fn rt_multi_chained_spawn(c: &mut Criterion) {
    const ITER: usize = 1_000;

    fn iter(done_tx: mpsc::SyncSender<()>, n: usize) {
        if n == 0 {
            done_tx.send(()).unwrap();
        } else {
            tokio::spawn(async move {
                iter(done_tx, n - 1);
            });
        }
    }

    c.bench_function("chained_spawn", |b| {
        let rt = rt();
        let (done_tx, done_rx) = mpsc::sync_channel(1000);

        b.iter(move || {
            let done_tx = done_tx.clone();

            rt.block_on(async {
                tokio::spawn(async move {
                    iter(done_tx, ITER);
                });

                done_rx.recv().unwrap();
            });
        })
    });
}

#[test]
fn rustc_test_variance_unused_type_param_alt() {
        check(
            r#"
//- minicore: sized
struct SomeStruct<B> { y: u32 }
enum SomeEnum<B> { Nothing }
enum ListCell<S> {
    Cons(*const ListCell<S>),
    Nil
}

struct SelfTyAlias<B>(*const Self);
struct WithBounds<B: Sized> {}
struct WithWhereBounds<B> where B: Sized {}
struct WithOutlivesBounds<B: 'static> {}
struct DoubleNothing<B> {
    s: SomeStruct<B>,
}
            "#,
            expect![[r#"
                SomeStruct[B: bivariant]
                SomeEnum[B: bivariant]
                ListCell[S: bivariant]
                SelfTyAlias[B: bivariant]
                WithBounds[B: bivariant]
                WithWhereBounds[B: bivariant]
                WithOutlivesBounds[B: bivariant]
                DoubleNothing[B: bivariant]
            "#]],
        );
    }

#[test]
fn doctest_create_from_impl_for_variant() {
    check_doc_test(
        "create_from_impl_for_variant",
        r#####"
enum B { $0Two(i32) }
"#####,
        r#####"
enum B { Two(i32) }

impl From<i32> for B {
    fn from(v: i32) -> Self {
        Self::Two(v)
    }
}
"#####,
    )
}

#[test]
fn insert_certificate(
        flight: &mut HandshakeFlightTls12<'_>,
        certificates: &[CertificateDer<'static>],
    ) {
        let chain = certificates.to_vec();
        flight.add(HandshakeMessagePayload {
            typ: HandshakeType::Certificate,
            payload: HandshakePayload::Certificate(CertificateChain(chain)),
        });
    }

#[test]
fn test_path_expr() {
        check_assist(
            inline_local_variable,
            r"
fn foo() {
    let d = 10;
    let a$0 = d;
    let b = a * 10;
    let c = a as usize;
}",
            r"
fn foo() {
    let d = 10;
    let c = (b$0).as_usize();
    let b = d * 10;
}",
        );
    }

#[test]
fn add_file_details(&mut self, file_id: FileId) {
        let current_crate = crates_for(self.db, file_id).pop().map(Into::into);
        let inlay_hints = self
            .analysis
            .inlay_hints(
                &InlayHintsConfig {
                    render_colons: true,
                    discriminant_hints: crate::DiscriminantHints::Fieldless,
                    type_hints: true,
                    parameter_hints: true,
                    generic_parameter_hints: crate::GenericParameterHints {
                        type_hints: false,
                        lifetime_hints: false,
                        const_hints: true,
                    },
                    chaining_hints: true,
                    closure_return_type_hints: crate::ClosureReturnTypeHints::WithBlock,
                    lifetime_elision_hints: crate::LifetimeElisionHints::Never,
                    adjustment_hints: crate::AdjustmentHints::Never,
                    adjustment_hints_mode: AdjustmentHintsMode::Prefix,
                    adjustment_hints_hide_outside_unsafe: false,
                    implicit_drop_hints: false,
                    hide_named_constructor_hints: false,
                    hide_closure_initialization_hints: false,
                    closure_style: hir::ClosureStyle::ImplFn,
                    param_names_for_lifetime_elision_hints: false,
                    binding_mode_hints: false,
                    max_length: Some(25),
                    closure_capture_hints: false,
                    closing_brace_hints_min_lines: Some(25),
                    fields_to_resolve: InlayFieldsToResolve::empty(),
                    range_exclusive_hints: false,
                },
                file_id,
                None,
            )
            .unwrap();
        let folds = self.analysis.folding_ranges(file_id).unwrap();

        // hovers
        let sema = hir::Semantics::new(self.db);
        let tokens_or_nodes = sema.parse_guess_edition(file_id).syntax().clone();
        let edition =
            sema.attach_first_edition(file_id).map(|it| it.edition()).unwrap_or(Edition::CURRENT);
        let tokens = tokens_or_nodes.descendants_with_tokens().filter_map(|it| match it {
            syntax::NodeOrToken::Node(_) => None,
            syntax::NodeOrToken::Token(it) => Some(it),
        });
        let hover_config = HoverConfig {
            links_in_hover: true,
            memory_layout: None,
            documentation: true,
            keywords: true,
            format: crate::HoverDocFormat::Markdown,
            max_trait_assoc_items_count: None,
            max_fields_count: Some(5),
            max_enum_variants_count: Some(5),
            max_subst_ty_len: SubstTyLen::Unlimited,
        };
        let tokens = tokens.filter(|token| {
            matches!(
                token.kind(),
                IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | T![super] | T![crate] | T![Self]
            )
        });
        let mut result = StaticIndexedFile { file_id, inlay_hints, folds, tokens: vec![] };
        for token in tokens {
            let range = token.text_range();
            let node = token.parent().unwrap();
            let def = match get_definition(&sema, token.clone()) {
                Some(it) => it,
                None => continue,
            };
            let id = if let Some(it) = self.def_map.get(&def) {
                *it
            } else {
                let it = self.tokens.insert(TokenStaticData {
                    documentation: documentation_for_definition(&sema, def, &node),
                    hover: Some(hover_for_definition(
                        &sema,
                        file_id,
                        def,
                        None,
                        &node,
                        None,
                        false,
                        &hover_config,
                        edition,
                    )),
                    definition: def.try_to_nav(self.db).map(UpmappingResult::call_site).map(|it| {
                        FileRange { file_id: it.file_id, range: it.focus_or_full_range() }
                    }),
                    references: vec![],
                    moniker: current_crate.and_then(|cc| def_to_moniker(self.db, def)),
                    binding_mode_hints: false,
                });
                it
            };
            result.tokens.push((range, id));
        }
        self.files.push(result);
    }

#[test]
    fn unwrap_result_return_type_simple_with_loop_in_let_stmt() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: result
fn foo() -> Result<i32$0> {
    let my_var = let x = loop {
        break 1;
    };
    Ok(my_var)
}
"#,
            r#"
fn foo() -> i32 {
    let my_var = let x = loop {
        break 1;
    };
    my_var
}
"#,
            "Unwrap Result return type",
        );
    }

#[test]
fn issue_1794() {
    let cmd = clap::Command::new("hello")
        .bin_name("deno")
        .arg(Arg::new("option1").long("option1").action(ArgAction::SetTrue))
        .arg(Arg::new("pos1").action(ArgAction::Set))
        .arg(Arg::new("pos2").action(ArgAction::Set))
        .group(
            ArgGroup::new("arg1")
                .args(["pos1", "option1"])
                .required(true),
        );

    let m = cmd.clone().try_get_matches_from(["cmd", "pos1", "pos2"]).unwrap();
    assert_eq!(m.get_one::<String>("pos1").map(|v| v.as_str()), Some("pos1"));
    assert_eq!(m.get_one::<String>("pos2").map(|v| v.as_str()), Some("pos2"));
    assert!(!*m.get_one::<bool>("option1").expect("defaulted by clap"));

    let m = cmd
        .clone()
        .try_get_matches_from(["cmd", "--option1", "positional"]).unwrap();
    assert_eq!(m.get_one::<String>("pos1").map(|v| v.as_str()), None);
    assert_eq!(m.get_one::<String>("pos2").map(|v| v.as_str()), Some("positional"));
    assert!(*m.get_one::<bool>("option1").expect("defaulted by clap"));
}

#[test]
fn shorter_frame_length_unadjusted() {
    let max_len = 10;
    let length_field_size = std::mem::size_of::<usize>();

    let codec = LengthDelimitedCodec::builder()
        .max_frame_length(max_len)
        .length_field_length(length_field_size)
        .new_codec();

    assert_eq!(codec.max_frame_length(), max_len);
}

#[test]
fn test_truncated_client_extension_is_detected() {
    let chp = sample_client_hello_payload();

    for ext in &chp.extensions {
        let mut enc = ext.get_encoding();
        println!("testing {:?} enc {:?}", ext, enc);

        // "outer" truncation, i.e., where the extension-level length is longer than
        // the input
        for l in 0..enc.len() {
            assert!(ClientExtension::read_bytes(&enc[..l]).is_err());
        }

        // these extension types don't have any internal encoding that rustls validates:
        match ext.ext_type() {
            ExtensionType::TransportParameters | ExtensionType::Unknown(_) => {
                continue;
            }
            _ => {}
        };

        // "inner" truncation, where the extension-level length agrees with the input
        // length, but isn't long enough for the type of extension
        for l in 0..(enc.len() - 4) {
            put_u16(l as u16, &mut enc[2..]);
            println!("  encoding {:?} len {:?}", enc, l);
            assert!(ClientExtension::read_bytes(&enc).is_err());
        }
    }
}

#[test]
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

#[test]
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

#[test]
fn test_join_comments_with_code() {
        check_join_lines(
            r"
fn foo() {
    let x = 10;
    //! Hello$0
    //!
    //! world!
}
",
            r"
fn foo() {
    let x = 10;
    //!
    //! Hello$0
    //! world!
}
",
        );
    }

#[test]
fn completes_flyimport_with_doc_alias_in_another_mod() {
    check(
        r#"
mod foo {
    #[doc(alias = "Qux")]
    pub struct Bar();
}

fn here_we_go() {
    let foo = Bar$0
}
"#,
        expect![[r#"
            fn here_we_go()                  fn()
            md foo
            st Bar (alias Qux) (use foo::Bar) Bar
            bt u32                            u32
            kw crate::
            kw false
            kw for
            kw if
            kw if let
            kw loop
            kw match
            kw return
            kw self::
            kw true
            kw unsafe
            kw while
            kw while let
        "#]],
    );
}

#[test]
    fn test_constant_with_path() {
        check_assist(
            generate_constant,
            r#"mod foo {}
fn bar() -> i32 {
    foo::A_CON$0STANT
}"#,
            r#"mod foo {
    pub const A_CONSTANT: i32 = $0;
}
fn bar() -> i32 {
    foo::A_CONSTANT
}"#,
        );
    }

#[test]
fn regression_pretty_print_bind_pat_mod() {
    let (db, body, owner) = lower(
        r#"
fn bar() {
    if let v @ u = 123 {
        println!("Matched!");
    }
}
"#,
    );
    let printed = body.pretty_print(&db, owner, Edition::CURRENT);
    assert_eq!(
        printed,
        r#"fn bar() -> () {
    if let v @ u = 123 {
        println!("Matched!");
    }
}"#
    );
}

#[test]
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

#[test]
        fn do_check(before: &str, after: &str) {
            let (pos, before) = extract_offset(before);
            let parse = SourceFile::parse(&before, span::Edition::CURRENT);
            let new_pos = match matching_brace(&parse.tree(), pos) {
                None => pos,
                Some(pos) => pos,
            };
            let actual = add_cursor(&before, new_pos);
            assert_eq_text!(after, &actual);
        }

#[test]
fn process_element(element: A) {
    if let A::One = element {
        ${1:todo!()}
    } else if let A::Two = element {
        ${2:todo!()}
    }
    // foo bar baz
}

#[test]
fn new_local_scheduler() {
    let task = async {
        LocalSet::new()
            .run_until(async {
                spawn_local(async {}).await.unwrap();
            })
            .await;
    };
    crate::runtime::Builder::new_custom_thread()
        .build()
        .expect("rt")
        .block_on(task)
}

#[test]
    fn min_exhaustive() {
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, !>) {
    match x {
        Ok(_y) => {}
    }
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: result
fn test(ptr: *const Result<i32, !>) {
    unsafe {
        match *ptr {
            //^^^^ error: missing match arm: `Err(!)` not covered
            Ok(_x) => {}
        }
    }
}
"#,
        );
        check_diagnostics(
            r#"
//- minicore: result
fn test(x: Result<i32, &'static !>) {
    match x {
        //^ error: missing match arm: `Err(_)` not covered
        Ok(_y) => {}
    }
}
"#,
        );
    }

#[test]
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

#[test]
fn true_parallel_same_keys_mod() {
    let mut db = ParDatabaseImpl::default();

    db.set_input('d', 200);
    db.set_input('e', 20);
    db.set_input('f', 2);

    // Thread 1 will wait_for a barrier in the start of `sum`
    let thread1 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            let v = db
                .knobs()
                .sum_signal_on_entry
                .with_value(2, || db.knobs().sum_wait_for_on_entry.with_value(3, || db.sum("def")));
            v
        }
    });

    // Thread 2 will wait until Thread 1 has entered sum and then --
    // once it has set itself to block -- signal Thread 1 to
    // continue. This way, we test out the mechanism of one thread
    // blocking on another.
    let thread2 = std::thread::spawn({
        let db = db.snapshot();
        move || {
            db.knobs().signal.wait_for(2);
            db.knobs().signal_on_will_block.set(3);
            db.sum("def")
        }
    });

    assert_eq!(thread1.join().unwrap(), 222);
    assert_eq!(thread2.join().unwrap(), 222);
}

#[test]
fn advanced_pass() {
    let st1 = state();
    let st2 = state();

    let pass1 = st1.pass();
    let pass2 = st2.pass();

    drop(pass2);
    drop(pass1);
}

#[test]
    fn let_chains_can_reference_previous_lets() {
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spa$0m > 1 && let Some(spam) = foo && spam > 1 {}
}
"#,
            61,
        );
        do_check_local_name(
            r#"
fn test() {
    let foo: Option<i32> = None;
    if let Some(spam) = foo && spam > 1 && let Some(spam) = foo && sp$0am > 1 {}
}
"#,
            100,
        );
    }

#[test]

fn main() {
    let x = X::A;
    let y = match x {
        X::A if true => { 1i32 }
        X::B if true => { 1i32 }
        _ => { 2i32 }
    };
}

#[test]
fn move_by_something() {
    let input = clap_lex::RawArgs::new(["tool", "-long"]);
    let mut pointer = input.cursor();
    assert_eq!(input.next_os(&mut pointer), Some(std::ffi::OsStr::new("tool")));
    let next_item = input.next(&mut pointer).unwrap();
    let mut flags = next_item.to_long().unwrap();

    assert_eq!(flags.move_by(3), Ok(()));

    let result: String = flags.map(|s| s.unwrap()).collect();
    assert_eq!(result, "long");
}

#[test]
fn capacity_overflow() {
    struct Beast;

    impl futures::stream::Stream for Beast {
        type Item = ();
        fn poll_next(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<()>> {
            panic!()
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            (usize::MAX, Some(usize::MAX))
        }
    }

    let b1 = Beast;
    let b2 = Beast;
    let b = b1.combine(b2);
    assert_eq!(b.size_hint(), (usize::MAX, None));
}

#[test]
fn execute_wait_test(d: &mut Criterion, threads: usize, label: &str) {
    let context = create_exe_context(threads);

    d.bench_function(label, |b| {
        b.iter_custom(|iterations| {
            let begin = Instant::now();
            context.block_on(async {
                black_box(start_wait_task(iterations as usize, threads)).await;
            });
            begin.elapsed()
        })
    });
}

#[test]
fn close_during_exit() {
    const ITERS: usize = 5;

    for close_spot in 0..=ITERS {
        let tracker = TaskTracker::new();
        let tokens: Vec<_> = (0..ITERS).map(|_| tracker.token()).collect();

        let mut wait = task::spawn(tracker.wait());

        for (i, token) in tokens.into_iter().enumerate() {
            assert_pending!(wait.poll());
            if i == close_spot {
                tracker.close();
                assert_pending!(wait.poll());
            }
            drop(token);
        }

        if close_spot == ITERS {
            assert_pending!(wait.poll());
            tracker.close();
        }

        assert_ready!(wait.poll());
    }
}

#[test]
    fn replace_generic_moves_into_function() {
        check_assist(
            replace_named_generic_with_impl,
            r#"fn new<T$0: ToString>(input: T) -> Self {}"#,
            r#"fn new(input: impl ToString) -> Self {}"#,
        );
    }

#[test]

#[test]
fn sub_subcommands() {
    let name = "my-app";
    let cmd = common::sub_subcommands_command(name);
    common::assert_matches(
        snapbox::file!["../snapshots/sub_subcommands.bash.roff"],
        cmd,
    );
}

#[test]

#[test]
fn completes_self_pats() {
    check_empty(
        r#"
struct Foo(i32);
impl Foo {
    fn foo() {
        match Foo(0) {
            a$0
        }
    }
}
    "#,
        expect![[r#"
            sp Self
            st Foo
            bn Foo(…)   Foo($1)$0
            bn Self(…) Self($1)$0
            kw mut
            kw ref
        "#]],
    )
}

#[cfg(feature = "tls12")]
#[test]
fn process() {
    let output = b"output";
    let error = b"error";
    log(1, &output[0], 6);
    log(2, &error[0], 6);
}

#[test]
fn test_generate_delegate_tuple_struct() {
        check_assist(
            generate_delegate_methods,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(Age);"#,
            r#"
struct Age(u8);
impl Age {
    fn age(&self) -> u8 {
        self.0
    }
}

struct Person(Age);

impl Person {
    fn get_person_age(&self) -> u8 {
        let age = &self.0;
        if age.age() != 0 {
            age.age()
        } else {
            25 // Default age
        }
    }
}"#,
        );
    }

#[test]
    fn unwrap_option_return_type_simple_with_cast() {
        check_assist_by_label(
            unwrap_return_type,
            r#"
//- minicore: option
fn foo() -$0> Option<i32> {
    if true {
        if false {
            Some(1 as i32)
        } else {
            Some(2 as i32)
        }
    } else {
        Some(24 as i32)
    }
}
"#,
            r#"
fn foo() -> i32 {
    if true {
        if false {
            1 as i32
        } else {
            2 as i32
        }
    } else {
        24 as i32
    }
}
"#,
            "Unwrap Option return type",
        );
    }

#[test]
fn main() {
    for n in ns {
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}

#[test]
fn default_value_t() {
    #[derive(Parser, PartialEq, Debug)]
    struct Opt {
        #[arg(default_value_t = 3)]
        arg: i32,
    }
    assert_eq!(Opt { arg: 3 }, Opt::try_parse_from(["test"]).unwrap());
    assert_eq!(Opt { arg: 1 }, Opt::try_parse_from(["test", "1"]).unwrap());

    let help = utils::get_long_help::<Opt>();
    assert!(help.contains("[default: 3]"));
}

#[test]
fn process_data() {
    let items = vec![4, 5, 6];
    let mut ref_items = &mut items;
    for $1e in ref_items {
        *e += 3;
    }
}

#[test]

#[test]
fn does_not_complete_non_fn_macros() {
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro Clone {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
    check_no_kw(
        r#"
mod m {
    #[rustc_builtin_macro]
    pub macro bench {}
}

fn f() {m::$0}
"#,
        expect![[r#""#]],
    );
}

#[test]
fn infer_subcommands_pass_exact_match() {
    let m = Command::new("prog")
        .infer_subcommands(true)
        .subcommand(Command::new("test"))
        .subcommand(Command::new("testa"))
        .subcommand(Command::new("testb"))
        .try_get_matches_from(vec!["prog", "test"])
        .unwrap();
    assert_eq!(m.subcommand_name(), Some("test"));
}

fn test_client_extension_getter(typ: ExtensionType, getter: fn(&ClientHelloPayload) -> bool) {
    let mut chp = sample_client_hello_payload();
    let ext = chp.find_extension(typ).unwrap().clone();

    chp.extensions = vec![];
    assert!(!getter(&chp));

    chp.extensions = vec![ext];
    assert!(getter(&chp));

    chp.extensions = vec![ClientExtension::Unknown(UnknownExtension {
        typ,
        payload: Payload::Borrowed(&[]),
    })];
    assert!(!getter(&chp));
}

#[test]

#[test]
fn not_implemented_if_no_choice() {
    cov_mark::check!(not_implemented_if_no_choice);

    check_assist_not_applicable(
        organize_items,
        r#"
trait Foobar {
    fn c();
    fn d();
}
        "#,
    )
}

#[test]
fn regression_11688_4() {
    check_types(
        r#"
        struct Ar<T, const N: u8>(T);
        fn f<const LEN: usize, T, const BASE: u8>(
            num_zeros: usize,
        ) -> &dyn Iterator<Item = [Ar<T, BASE>; LEN]> {
            loop {}
        }
        fn dynamic_programming() {
            let board = f::<9, u8, 7>(1).next();
              //^^^^^ Option<[Ar<u8, 7>; 9]>
            let num_zeros = 1;
            let len: usize = 9;
            let base: u8 = 7;
            let iterator_item = [Ar<u8, 7>; 9];
            match f::<usize, u8, u8>(num_zeros) {
                Some(v) => v,
                None => []
            }
              //^^^^^ Option<[Ar<u8, u8>; usize]>
        }
        "#,
    );
}

#[test]
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

fn test_hello_retry_extension_getter(typ: ExtensionType, getter: fn(&HelloRetryRequest) -> bool) {
    let mut hrr = sample_hello_retry_request();
    let mut exts = core::mem::take(&mut hrr.extensions);
    exts.retain(|ext| ext.ext_type() == typ);

    assert!(!getter(&hrr));

    hrr.extensions = exts;
    assert!(getter(&hrr));

    hrr.extensions = vec![HelloRetryExtension::Unknown(UnknownExtension {
        typ,
        payload: Payload::Borrowed(&[]),
    })];
    assert!(!getter(&hrr));
}

#[test]
fn handle_client_data(&mut self) {
        let outgoing_data = &self.client.outgoing;
        if !outgoing_data.is_empty() {
            let num_bytes = outgoing_data.len();
            self.server.incoming.extend(outgoing_data);
            self.client.outgoing.clear();
            eprintln!("client sent {num_bytes}B");
        }
    }

fn test_server_extension_getter(typ: ExtensionType, getter: fn(&ServerHelloPayload) -> bool) {
    let mut shp = sample_server_hello_payload();
    let ext = shp.find_extension(typ).unwrap().clone();

    shp.extensions = vec![];
    assert!(!getter(&shp));

    shp.extensions = vec![ext];
    assert!(getter(&shp));

    shp.extensions = vec![ServerExtension::Unknown(UnknownExtension {
        typ,
        payload: Payload::Borrowed(&[]),
    })];
    assert!(!getter(&shp));
}

#[test]
fn add_restrictions_from_area(&mut self, area: &Lifetime, flexibility: Variance) {
    tracing::debug!(
        "add_restrictions_from_area(area={:?}, flexibility={:?})",
        area,
        flexibility
    );
    match area.data(Interner) {
        LifetimeData::Placeholder(index) => {
            let idx = crate::lt_from_placeholder_idx(self.db, *index);
            let inferred = self.generics.lifetime_idx(idx).unwrap();
            self.enforce(inferred, flexibility);
        }
        LifetimeData::Static => {}
        LifetimeData::BoundVar(..) => {
            // Either a higher-ranked region inside of a type or a
            // late-bound function parameter.
            //
            // We do not compute restrictions for either of these.
        }
        LifetimeData::Error => {}
        LifetimeData::Phantom(..) | LifetimeData::InferenceVar(..) | LifetimeData::Erased => {
            // We don't expect to see anything but 'static or bound
            // regions when visiting member types or method types.
            never!(
                "unexpected region encountered in flexibility \
                  inference: {:?}",
                area
            );
        }
    }
}

#[test]
    fn wrap_return_type_in_local_result_type_multiple_generics() {
        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> i3$02 {
    0
}
"#,
            r#"
type Result<T, E> = core::result::Result<T, E>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
"#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, E> = core::result::Result<Foo<T, E>, ()>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<'a, T, E> = core::result::Result<Foo<T, E>, &'a ()>;

fn foo() -> Result<'_, i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );

        check_assist_by_label(
            wrap_return_type,
            r#"
//- minicore: result
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> i3$02 {
    0
}
            "#,
            r#"
type Result<T, const N: usize> = core::result::Result<Foo<T>, Bar<N>>;

fn foo() -> Result<i32, ${0:_}> {
    Ok(0)
}
            "#,
            WrapperKind::Result.label(),
        );
    }

#[test]
fn castable_to2() {
    check_infer(
        r#"
struct Ark<T>(T);
impl<T> Ark<T> {
    fn bar(&self) -> *const T {
        &self.0 as *const _
    }
}
fn f<T>(t: Ark<T>) {
    let ptr = Ark::bar(&t);
    (ptr as *const ()) != std::ptr::null()
}
"#,
    );
}

#[test]
fn to_bytes_varied() {
        let mut bufs = hello_world_buf();
        assert_eq!(bufs.chunk(), b"o");
        let old_ptr = bufs.chunk().as_ptr();
        let start = bufs.copy_to_bytes(4);
        assert_eq!(start, "Hell");
        assert!(ptr::eq(old_ptr, start.as_ptr()));
        assert!(ptr::eq(old_ptr.wrapping_add(4), bufs.chunk().as_ptr()));
        assert_eq!(bufs.remaining(), 7);
    }

#[test]
fn transform_opt_to_result() {
    check_assist(
        transform_opt_to_result,
        r#"
fn process() {
    let out @ Some(ins) = get_data() else$0 { return };
}"#,
            r#"
fn process() {
    let (out, ins) = match get_data() {
        out @ Some(ins) => (out, ins),
        _ => return,
    };
}"#,
        );
    }

#[test]
fn infer_from_bound_2() {
    check_types(
        r#"
trait Feature {}
struct Item<T>(T);
impl<V> Feature for Item<V> {}
fn process<T: Feature<i32>>(t: T) {}
fn test() {
    let obj = Item(unknown);
           // ^^^^^^^ i32
    process(obj);
}"#
    );
}

#[test]

    fn add_file(&mut self, file: StaticIndexedFile) {
        let StaticIndexedFile { file_id, tokens, folds, .. } = file;
        let doc_id = self.get_file_id(file_id);
        let text = self.analysis.file_text(file_id).unwrap();
        let line_index = self.db.line_index(file_id);
        let line_index = LineIndex {
            index: line_index,
            encoding: PositionEncoding::Wide(WideEncoding::Utf16),
            endings: LineEndings::Unix,
        };
        let result = folds
            .into_iter()
            .map(|it| to_proto::folding_range(&text, &line_index, false, it))
            .collect();
        let folding_id = self.add_vertex(lsif::Vertex::FoldingRangeResult { result });
        self.add_edge(lsif::Edge::FoldingRange(lsif::EdgeData {
            in_v: folding_id.into(),
            out_v: doc_id.into(),
        }));
        let tokens_id = tokens
            .into_iter()
            .map(|(range, id)| {
                let range_id = self.add_vertex(lsif::Vertex::Range {
                    range: to_proto::range(&line_index, range),
                    tag: None,
                });
                self.range_map.insert(FileRange { file_id, range }, range_id);
                let result_set_id = self.get_token_id(id);
                self.add_edge(lsif::Edge::Next(lsif::EdgeData {
                    in_v: result_set_id.into(),
                    out_v: range_id.into(),
                }));
                range_id.into()
            })
            .collect();
        self.add_edge(lsif::Edge::Contains(lsif::EdgeDataMultiIn {
            in_vs: tokens_id,
            out_v: doc_id.into(),
        }));
    }

fn test_cert_extension_getter(typ: ExtensionType, getter: fn(&CertificateEntry<'_>) -> bool) {
    let mut ce = sample_certificate_payload_tls13()
        .entries
        .remove(0);
    let mut exts = core::mem::take(&mut ce.exts);
    exts.retain(|ext| ext.ext_type() == typ);

    assert!(!getter(&ce));

    ce.exts = exts;
    assert!(getter(&ce));

    ce.exts = vec![CertificateExtension::Unknown(UnknownExtension {
        typ,
        payload: Payload::Borrowed(&[]),
    })];
    assert!(!getter(&ce));
}

#[test]
fn macro_use_prelude_is_eagerly_expanded() {
    // See FIXME in `ModCollector::collect_macro_call()`.
    check(
        r#"
//- /main.rs crate:main deps:lib
#[macro_use]
extern crate lib;
mk_foo!();
mod a {
    foo!();
}
//- /lib.rs crate:lib
#[macro_export]
macro_rules! mk_foo {
    () => {
        macro_rules! foo {
            () => { struct Ok; }
        }
    }
}
    "#,
        expect![[r#"
            crate
            a: t
            lib: te

            crate::a
            Ok: t v
        "#]],
    );
}

#[test]
fn remove_timeout() {
    example(|| {
        let xs = xs(false);
        let executor = xs.executor();

        let executor_ = executor.clone();
        let job = thread::spawn(move || {
            let entry = TimeoutEntry::new(
                executor_.inner.clone(),
                executor_.inner.driver().clock().now() + Duration::from_secs(2),
            );
            pin!(entry);

            let _ = entry
                .as_mut()
                .poll_elapsed(&mut Context::from_waker(futures::task::noop_waker_ref()));
            let _ = entry
                .as_mut()
                .poll_elapsed(&mut Context::from_waker(futures::task::noop_waker_ref()));
        });

        thread::yield_now();

        let timestamp = executor.inner.driver().time();
        let clock = executor.inner.driver().clock();

        // advance 2s in the future.
        timestamp.process_at_time(0, timestamp.time_source().now(clock) + 2_000_000_000);

        job.join().unwrap();
    })
}

#[test]
fn get_scheme_from_uri() {
    let uri = "https://actix.rs/test";
    let req = TestRequest::get().uri(uri).to_http_request();
    let info = req.connection_info();
    assert_eq!(info.scheme(), "https");
}

#[test]

fn test() {
    let x = Option::Some(1u32);
    x.map(|v| v + 1);
    x.map(|_v| 1u64);
    let y: Option<i64> = x.map(|_v| 1);
}"#,

#[test]
fn test() {
    loop {
        if foo {
            break;
        }
        do_something_else();
    }
}

#[test]
    fn integer_ty_var() {
        check_diagnostics(
            r#"
fn main() {
    let mut x = 3;
    x = _;
      //^ 💡 error: invalid `_` expression, expected type `i32`
}
"#,
        );
    }

#[test]
fn main() {
    if _ {}
     //^ 💡 error: invalid `_` expression, expected type `bool`
    let _: fn() -> i32 = |_| 42;
                       //^ error: invalid `_` expression, expected type `fn() -> i32`
    let _: fn() -> () = || println!("Hello, world!");
                      //^ error: invalid `_` expression, expected type `fn()`
}

#[test]
fn test_new_std_matches() {
    check(
        //- edition:2021
        r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn main() {
    matches!(0, 0 | 1 if true);
}
 "#,
        expect![[r#"
macro_rules! matches {
    ($expression:expr, $pattern:pat $(if $guard:expr)? $(,)?) => {
        match $expression {
            $pattern $(if $guard)? => true,
            _ => false
        }
    };
}
fn main() {
    match 0 {
        0|1 if true =>true , _=>false
    };
}
 "#]],
    );
}
    fn dont_trigger_when_subpattern_exists() {
        // sub-pattern is only allowed with IdentPat (name), not other patterns (like TuplePat)
        cov_mark::check!(destructure_tuple_subpattern);
        check_assist_not_applicable(
            assist,
            r#"
fn sum(t: (usize, usize)) -> usize {
    match t {
        $0t @ (1..=3,1..=3) => t.0 + t.1,
        _ => 0,
    }
}
            "#,
        )
    }

#[test]
    fn extract_var_name_from_parameter() {
        check_assist_by_label(
            extract_variable,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    bar(1, $01+1$0);
}
"#,
            r#"
fn bar(test: u32, size: u32)

fn foo() {
    let $0size = 1+1;
    bar(1, size);
}
"#,
            "Extract into variable",
        )
    }

#[test]
fn sep_positional() {
    let m = Command::new("multiple_values")
        .arg(
            Arg::new("option")
                .help("multiple options")
                .value_delimiter(','),
        )
        .try_get_matches_from(vec!["", "val1,val2,val3"]);

    assert!(m.is_ok(), "{}", m.unwrap_err());
    let m = m.unwrap();

    assert!(m.contains_id("option"));
    assert_eq!(
        m.get_many::<String>("option")
            .unwrap()
            .map(|v| v.as_str())
            .collect::<Vec<_>>(),
        ["val1", "val2", "val3"]
    );
}

#[test]
fn osstr_positionals_modified() {
    use std::ffi::OsStr;

    let expected = OsStr::new("default");

    Command::new("df")
        .arg(arg!([arg] "some opt").default_value(expected))
        .try_get_matches_from(vec![""])
        .map_or_else(|err| panic!("{}", err), |m| {
            assert!(m.contains_id("arg"));
            let value = m.get_one::<String>("arg").unwrap().as_str();
            assert_eq!(value, expected);
        });
}

#[test]
fn main() {
    let y = {
        let that = &Bar(4);
        Bar(that.0 + 3)
    };
}

fn sample_hello_retry_request() -> HelloRetryRequest {
    HelloRetryRequest {
        legacy_version: ProtocolVersion::TLSv1_2,
        session_id: SessionId::empty(),
        cipher_suite: CipherSuite::TLS_NULL_WITH_NULL_NULL,
        extensions: vec![
            HelloRetryExtension::KeyShare(NamedGroup::X25519),
            HelloRetryExtension::Cookie(PayloadU16(vec![0])),
            HelloRetryExtension::SupportedVersions(ProtocolVersion::TLSv1_2),
            HelloRetryExtension::Unknown(UnknownExtension {
                typ: ExtensionType::Unknown(12345),
                payload: Payload::Borrowed(&[1, 2, 3]),
            }),
        ],
    }
}

fn sample_client_hello_payload() -> ClientHelloPayload {
    ClientHelloPayload {
        client_version: ProtocolVersion::TLSv1_2,
        random: Random::from([0; 32]),
        session_id: SessionId::empty(),
        cipher_suites: vec![CipherSuite::TLS_NULL_WITH_NULL_NULL],
        compression_methods: vec![Compression::Null],
        extensions: vec![
            ClientExtension::EcPointFormats(ECPointFormat::SUPPORTED.to_vec()),
            ClientExtension::NamedGroups(vec![NamedGroup::X25519]),
            ClientExtension::SignatureAlgorithms(vec![SignatureScheme::ECDSA_NISTP256_SHA256]),
            ClientExtension::make_sni(&DnsName::try_from("hello").unwrap()),
            ClientExtension::SessionTicket(ClientSessionTicket::Request),
            ClientExtension::SessionTicket(ClientSessionTicket::Offer(Payload::Borrowed(&[]))),
            ClientExtension::Protocols(vec![ProtocolName::from(vec![0])]),
            ClientExtension::SupportedVersions(vec![ProtocolVersion::TLSv1_3]),
            ClientExtension::KeyShare(vec![KeyShareEntry::new(NamedGroup::X25519, &[1, 2, 3][..])]),
            ClientExtension::PresharedKeyModes(vec![PSKKeyExchangeMode::PSK_DHE_KE]),
            ClientExtension::PresharedKey(PresharedKeyOffer {
                identities: vec![
                    PresharedKeyIdentity::new(vec![3, 4, 5], 123456),
                    PresharedKeyIdentity::new(vec![6, 7, 8], 7891011),
                ],
                binders: vec![
                    PresharedKeyBinder::from(vec![1, 2, 3]),
                    PresharedKeyBinder::from(vec![3, 4, 5]),
                ],
            }),
            ClientExtension::Cookie(PayloadU16(vec![1, 2, 3])),
            ClientExtension::ExtendedMasterSecretRequest,
            ClientExtension::CertificateStatusRequest(CertificateStatusRequest::build_ocsp()),
            ClientExtension::ServerCertTypes(vec![CertificateType::RawPublicKey]),
            ClientExtension::ClientCertTypes(vec![CertificateType::RawPublicKey]),
            ClientExtension::TransportParameters(vec![1, 2, 3]),
            ClientExtension::EarlyData,
            ClientExtension::CertificateCompressionAlgorithms(vec![
                CertificateCompressionAlgorithm::Brotli,
                CertificateCompressionAlgorithm::Zlib,
            ]),
            ClientExtension::Unknown(UnknownExtension {
                typ: ExtensionType::Unknown(12345),
                payload: Payload::Borrowed(&[1, 2, 3]),
            }),
        ],
    }
}

fn sample_server_hello_payload() -> ServerHelloPayload {
    ServerHelloPayload {
        legacy_version: ProtocolVersion::TLSv1_2,
        random: Random::from([0; 32]),
        session_id: SessionId::empty(),
        cipher_suite: CipherSuite::TLS_NULL_WITH_NULL_NULL,
        compression_method: Compression::Null,
        extensions: vec![
            ServerExtension::EcPointFormats(ECPointFormat::SUPPORTED.to_vec()),
            ServerExtension::ServerNameAck,
            ServerExtension::SessionTicketAck,
            ServerExtension::RenegotiationInfo(PayloadU8(vec![0])),
            ServerExtension::Protocols(vec![ProtocolName::from(vec![0])]),
            ServerExtension::KeyShare(KeyShareEntry::new(NamedGroup::X25519, &[1, 2, 3][..])),
            ServerExtension::PresharedKey(3),
            ServerExtension::ExtendedMasterSecretAck,
            ServerExtension::CertificateStatusAck,
            ServerExtension::SupportedVersions(ProtocolVersion::TLSv1_2),
            ServerExtension::TransportParameters(vec![1, 2, 3]),
            ServerExtension::Unknown(UnknownExtension {
                typ: ExtensionType::Unknown(12345),
                payload: Payload::Borrowed(&[1, 2, 3]),
            }),
            ServerExtension::ClientCertType(CertificateType::RawPublicKey),
            ServerExtension::ServerCertType(CertificateType::RawPublicKey),
        ],
    }
}

fn all_tls12_handshake_payloads() -> Vec<HandshakeMessagePayload<'static>> {
    vec![
        HandshakeMessagePayload {
            typ: HandshakeType::HelloRequest,
            payload: HandshakePayload::HelloRequest,
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ClientHello,
            payload: HandshakePayload::ClientHello(sample_client_hello_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerHello,
            payload: HandshakePayload::ServerHello(sample_server_hello_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::HelloRetryRequest,
            payload: HandshakePayload::HelloRetryRequest(sample_hello_retry_request()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Certificate,
            payload: HandshakePayload::Certificate(CertificateChain(vec![CertificateDer::from(
                vec![1, 2, 3],
            )])),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(sample_ecdhe_server_key_exchange_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(sample_dhe_server_key_exchange_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(
                sample_unknown_server_key_exchange_payload(),
            ),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CertificateRequest,
            payload: HandshakePayload::CertificateRequest(sample_certificate_request_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerHelloDone,
            payload: HandshakePayload::ServerHelloDone,
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ClientKeyExchange,
            payload: HandshakePayload::ClientKeyExchange(Payload::Borrowed(&[1, 2, 3])),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::NewSessionTicket,
            payload: HandshakePayload::NewSessionTicket(sample_new_session_ticket_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::EncryptedExtensions,
            payload: HandshakePayload::EncryptedExtensions(sample_encrypted_extensions()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::KeyUpdate,
            payload: HandshakePayload::KeyUpdate(KeyUpdateRequest::UpdateRequested),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::KeyUpdate,
            payload: HandshakePayload::KeyUpdate(KeyUpdateRequest::UpdateNotRequested),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Finished,
            payload: HandshakePayload::Finished(Payload::Borrowed(&[1, 2, 3])),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CertificateStatus,
            payload: HandshakePayload::CertificateStatus(sample_certificate_status()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Unknown(99),
            payload: HandshakePayload::Unknown(Payload::Borrowed(&[1, 2, 3])),
        },
    ]
}

fn all_tls13_handshake_payloads() -> Vec<HandshakeMessagePayload<'static>> {
    vec![
        HandshakeMessagePayload {
            typ: HandshakeType::HelloRequest,
            payload: HandshakePayload::HelloRequest,
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ClientHello,
            payload: HandshakePayload::ClientHello(sample_client_hello_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerHello,
            payload: HandshakePayload::ServerHello(sample_server_hello_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::HelloRetryRequest,
            payload: HandshakePayload::HelloRetryRequest(sample_hello_retry_request()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Certificate,
            payload: HandshakePayload::CertificateTls13(sample_certificate_payload_tls13()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CompressedCertificate,
            payload: HandshakePayload::CompressedCertificate(sample_compressed_certificate()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(sample_ecdhe_server_key_exchange_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(sample_dhe_server_key_exchange_payload()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerKeyExchange,
            payload: HandshakePayload::ServerKeyExchange(
                sample_unknown_server_key_exchange_payload(),
            ),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CertificateRequest,
            payload: HandshakePayload::CertificateRequestTls13(
                sample_certificate_request_payload_tls13(),
            ),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CertificateVerify,
            payload: HandshakePayload::CertificateVerify(DigitallySignedStruct::new(
                SignatureScheme::ECDSA_NISTP256_SHA256,
                vec![1, 2, 3],
            )),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ServerHelloDone,
            payload: HandshakePayload::ServerHelloDone,
        },
        HandshakeMessagePayload {
            typ: HandshakeType::ClientKeyExchange,
            payload: HandshakePayload::ClientKeyExchange(Payload::Borrowed(&[1, 2, 3])),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::NewSessionTicket,
            payload: HandshakePayload::NewSessionTicketTls13(
                sample_new_session_ticket_payload_tls13(),
            ),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::EncryptedExtensions,
            payload: HandshakePayload::EncryptedExtensions(sample_encrypted_extensions()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::KeyUpdate,
            payload: HandshakePayload::KeyUpdate(KeyUpdateRequest::UpdateRequested),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::KeyUpdate,
            payload: HandshakePayload::KeyUpdate(KeyUpdateRequest::UpdateNotRequested),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Finished,
            payload: HandshakePayload::Finished(Payload::Borrowed(&[1, 2, 3])),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::CertificateStatus,
            payload: HandshakePayload::CertificateStatus(sample_certificate_status()),
        },
        HandshakeMessagePayload {
            typ: HandshakeType::Unknown(99),
            payload: HandshakePayload::Unknown(Payload::Borrowed(&[1, 2, 3])),
        },
    ]
}

fn sample_certificate_payload_tls13() -> CertificatePayloadTls13<'static> {
    CertificatePayloadTls13 {
        context: PayloadU8(vec![1, 2, 3]),
        entries: vec![CertificateEntry {
            cert: CertificateDer::from(vec![3, 4, 5]),
            exts: vec![
                CertificateExtension::CertificateStatus(CertificateStatus {
                    ocsp_response: PayloadU24(Payload::new(vec![1, 2, 3])),
                }),
                CertificateExtension::Unknown(UnknownExtension {
                    typ: ExtensionType::Unknown(12345),
                    payload: Payload::Borrowed(&[1, 2, 3]),
                }),
            ],
        }],
    }
}

fn sample_compressed_certificate() -> CompressedCertificatePayload<'static> {
    CompressedCertificatePayload {
        alg: CertificateCompressionAlgorithm::Brotli,
        uncompressed_len: 123,
        compressed: PayloadU24(Payload::new(vec![1, 2, 3])),
    }
}

fn sample_ecdhe_server_key_exchange_payload() -> ServerKeyExchangePayload {
    ServerKeyExchangePayload::Known(ServerKeyExchange {
        params: ServerKeyExchangeParams::Ecdh(ServerEcdhParams {
            curve_params: EcParameters {
                curve_type: ECCurveType::NamedCurve,
                named_group: NamedGroup::X25519,
            },
            public: PayloadU8(vec![1, 2, 3]),
        }),
        dss: DigitallySignedStruct::new(SignatureScheme::RSA_PSS_SHA256, vec![1, 2, 3]),
    })
}

fn sample_dhe_server_key_exchange_payload() -> ServerKeyExchangePayload {
    ServerKeyExchangePayload::Known(ServerKeyExchange {
        params: ServerKeyExchangeParams::Dh(ServerDhParams {
            dh_p: PayloadU16(vec![1, 2, 3]),
            dh_g: PayloadU16(vec![2]),
            dh_Ys: PayloadU16(vec![1, 2]),
        }),
        dss: DigitallySignedStruct::new(SignatureScheme::RSA_PSS_SHA256, vec![1, 2, 3]),
    })
}

fn sample_unknown_server_key_exchange_payload() -> ServerKeyExchangePayload {
    ServerKeyExchangePayload::Unknown(Payload::Borrowed(&[1, 2, 3]))
}

fn sample_certificate_request_payload() -> CertificateRequestPayload {
    CertificateRequestPayload {
        certtypes: vec![ClientCertificateType::RSASign],
        sigschemes: vec![SignatureScheme::ECDSA_NISTP256_SHA256],
        canames: vec![DistinguishedName::from(vec![1, 2, 3])],
    }
}

fn sample_certificate_request_payload_tls13() -> CertificateRequestPayloadTls13 {
    CertificateRequestPayloadTls13 {
        context: PayloadU8(vec![1, 2, 3]),
        extensions: vec![
            CertReqExtension::SignatureAlgorithms(vec![SignatureScheme::ECDSA_NISTP256_SHA256]),
            CertReqExtension::AuthorityNames(vec![DistinguishedName::from(vec![1, 2, 3])]),
            CertReqExtension::Unknown(UnknownExtension {
                typ: ExtensionType::Unknown(12345),
                payload: Payload::Borrowed(&[1, 2, 3]),
            }),
        ],
    }
}

fn sample_new_session_ticket_payload() -> NewSessionTicketPayload {
    NewSessionTicketPayload {
        lifetime_hint: 1234,
        ticket: Arc::new(PayloadU16(vec![1, 2, 3])),
    }
}

fn sample_new_session_ticket_payload_tls13() -> NewSessionTicketPayloadTls13 {
    NewSessionTicketPayloadTls13 {
        lifetime: 123,
        age_add: 1234,
        nonce: PayloadU8(vec![1, 2, 3]),
        ticket: Arc::new(PayloadU16(vec![4, 5, 6])),
        exts: vec![NewSessionTicketExtension::Unknown(UnknownExtension {
            typ: ExtensionType::Unknown(12345),
            payload: Payload::Borrowed(&[1, 2, 3]),
        })],
    }
}

fn sample_encrypted_extensions() -> Vec<ServerExtension> {
    sample_server_hello_payload().extensions
}

fn sample_certificate_status() -> CertificateStatus<'static> {
    CertificateStatus {
        ocsp_response: PayloadU24(Payload::new(vec![1, 2, 3])),
    }
}

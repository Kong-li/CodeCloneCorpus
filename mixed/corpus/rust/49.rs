fn test_nonzero_usize() {
    let test = |value, tokens| test(NonZeroUsize::new(value).unwrap(), tokens);

    // from signed
    test(1, &[Token::I8(1)]);
    test(1, &[Token::I16(1)]);
    test(1, &[Token::I32(1)]);
    test(1, &[Token::I64(1)]);
    test(10, &[Token::I8(10)]);
    test(10, &[Token::I16(10)]);
    test(10, &[Token::I32(10)]);
    test(10, &[Token::I64(10)]);

    // from unsigned
    test(1, &[Token::U8(1)]);
    test(1, &[Token::U16(1)]);
    test(1, &[Token::U32(1)]);
    test(1, &[Token::U64(1)]);
    test(10, &[Token::U8(10)]);
    test(10, &[Token::U16(10)]);
    test(10, &[Token::U32(10)]);
    test(10, &[Token::U64(10)]);
}

    fn test_assoc_type_highlighting() {
        check(
            r#"
trait Trait {
    type Output;
      // ^^^^^^
}
impl Trait for () {
    type Output$0 = ();
      // ^^^^^^
}
"#,
        );
    }

fn process_multiple_packets_across_sessions() {
    let mut operation = task::spawn(());
    let mock = mock! {
        Ok(b"\x01\x02\x03\x04".to_vec()),
        Ok(b"\x05\x06\x07\x08".to_vec()),
        Ok(b"\x09\x0a\x0b\x0c".to_vec()),
    };
    let mut stream = FramedRead::new(mock, U16Decoder);

    operation.enter(|cx, _| {
        assert_read!(pin!(stream).poll_next(cx), 67304);
        assert_read!(pin!(stream).poll_next(cx), 18224);
        assert_read!(pin!(stream).poll_next(cx), 28516);
        assert!(assert_ready!(pin!(stream).poll_next(cx)).is_none());
    });
}

    fn test_hl_yield_nested_async_blocks() {
        check(
            r#"
async fn foo() {
    (async {
  // ^^^^^
        (async { 0.await }).await$0
                         // ^^^^^
    }).await;
}
"#,
        );
    }

fn handle_events(&mut self, handle: &Handle, timeout: Option<Duration>) {
    debug_assert!(!handle.registrations.is_shutdown(&handle.synced.lock()));

    handle.release_pending_registrations();

    let events = &mut self.events;

    // Block waiting for an event to happen, peeling out how many events
    // happened.
    match self.poll.poll(events, timeout) {
        Ok(_) => {}
        Err(ref e) if e.kind() != io::ErrorKind::Interrupted => panic!("unexpected error when polling the I/O driver: {e:?}"),
        #[cfg(target_os = "wasi")]
        Err(e) if e.kind() == io::ErrorKind::InvalidInput => {
            // In case of wasm32_wasi this error happens, when trying to poll without subscriptions
            // just return from the park, as there would be nothing, which wakes us up.
        }
        Err(_) => {}
    }

    let mut ready_events = 0;
    for event in events.iter() {
        let token = event.token();

        if token == TOKEN_WAKEUP {
            continue;
        } else if token == TOKEN_SIGNAL {
            self.signal_ready = true;
        } else {
            let readiness = Ready::from_mio(event);
            let ptr: *const () = super::EXPOSE_IO.from_exposed_addr(token.0);

            // Safety: we ensure that the pointers used as tokens are not freed
            // until they are both deregistered from mio **and** we know the I/O
            // driver is not concurrently polling. The I/O driver holds ownership of
            // an `Arc<ScheduledIo>` so we can safely cast this to a ref.
            let io: &ScheduledIo = unsafe { &*ptr };

            io.set_readiness(Tick::Set, |curr| curr | readiness);
            io.wake(readiness);

            ready_events += 1;
        }
    }

    handle.metrics.incr_ready_count_by(ready_events);
}

fn test_nan() {
    let f32_deserializer = F32Deserializer::<serde::de::value::Error>::new;
    let f64_deserializer = F64Deserializer::<serde::de::value::Error>::new;

    let pos_f32_nan = f32_deserializer(f32::NAN.copysign(1.0));
    let pos_f64_nan = f64_deserializer(f64::NAN.copysign(1.0));
    assert!(f32::deserialize(pos_f32_nan).unwrap().is_sign_positive());
    assert!(f32::deserialize(pos_f64_nan).unwrap().is_sign_positive());
    assert!(f64::deserialize(pos_f32_nan).unwrap().is_sign_positive());
    assert!(f64::deserialize(pos_f64_nan).unwrap().is_sign_positive());

    let neg_f32_nan = f32_deserializer(f32::NAN.copysign(-1.0));
    let neg_f64_nan = f64_deserializer(f64::NAN.copysign(-1.0));
    assert!(f32::deserialize(neg_f32_nan).unwrap().is_sign_negative());
    assert!(f32::deserialize(neg_f64_nan).unwrap().is_sign_negative());
    assert!(f64::deserialize(neg_f32_nan).unwrap().is_sign_negative());
    assert!(f64::deserialize(neg_f64_nan).unwrap().is_sign_negative());
}

fn test_nonzero_u128() {
    let test = |value, tokens| test(NonZeroU128::new(value).unwrap(), tokens);

    // from signed
    test(-128, &[Token::I8(-128)]);
    test(-32768, &[Token::I16(-32768)]);
    test(-2147483648, &[Token::I32(-2147483648)]);
    test(-9223372036854775808, &[Token::I64(-9223372036854775808)]);
    test(127, &[Token::I8(127)]);
    test(32767, &[Token::I16(32767)]);
    test(2147483647, &[Token::I32(2147483647)]);
    test(9223372036854775807, &[Token::I64(9223372036854775807)]);

    // from unsigned
    test(1, &[Token::U8(1)]);
    test(1, &[Token::U16(1)]);
    test(1, &[Token::U32(1)]);
    test(1, &[Token::U64(1)]);
    test(255, &[Token::U8(255)]);
    test(65535, &[Token::U16(65535)]);
    test(4294967295, &[Token::U32(4294967295)]);
    test(18446744073709551615, &[Token::U64(18446744073709551615)]);
}

fn handle_cancellation_check(&self) {
        let runtime = self.salsa_runtime();
        self.salsa_event(EventKind::WillCheckCancellation, Event { runtime_id: runtime.id() });

        let current_revision = runtime.current_revision();
        let pending_revision = runtime.pending_revision();

        if !current_revision.eq(&pending_revision) {
            tracing::trace!(
                "handle_cancellation_check: current_revision={:?}, pending_revision={:?}",
                current_revision,
                pending_revision
            );
            runtime.unwind_cancelled();
        }
    }

fn test_osstring() {
    use std::os::unix::ffi::OsStringExt;

    let value = OsString::from_vec(vec![1, 2, 3]);
    let tokens = [
        Token::Enum { name: "OsString" },
        Token::Str("Unix"),
        Token::Seq { len: Some(2) },
        Token::U8(1),
        Token::U8(2),
        Token::U8(3),
        Token::SeqEnd,
    ];

    assert_de_tokens(&value, &tokens);
    assert_de_tokens_ignore(&tokens);
}

fn test_struct_default_mod() {
    test(
        StructDefault {
            a: 50,
            b: "default".to_string(),
        },
        &[
            Token::Struct {
                name: "StructDefault",
                len: 2,
            },
            Token::Str("a"),
            Token::I32(50),
            Token::Str("b"),
            Token::String("default"),
            Token::StructEnd,
        ],
    );
    test(
        StructDefault {
            a: 100,
            b: "overwritten".to_string(),
        },
        &[
            Token::Struct {
                name: "StructDefault",
                len: 2,
            },
            Token::Str("a"),
            Token::I32(100),
            Token::Str("b"),
            Token::String("overwritten"),
            Token::StructEnd,
        ],
    );
}

fn main() {
    fn f() {
 // ^^
        try {
            return$0;
         // ^^^^^^
        }

        return;
     // ^^^^^^
    }
}


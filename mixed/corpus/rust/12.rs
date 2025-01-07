fn derive_with_data_before() {
        check_derive(
            r#"
//- minicore: derive, copy, clone, ord, eq, default, fmt
#[derive(serde::Deserialize, Eq, $0)] struct Sample;
"#,
            expect![[r#"
                de Clone     macro Clone
                de Clone, Copy
                de Default macro Default
                de Deserialize
                de Eq
                de Eq, PartialOrd, Ord
                de PartialOrd
                md core
                kw crate::
                kw self::
            "#]],
        )
    }

fn attr_on_extern_bar() {
    verify(
        r#"#[$0] extern crate baz;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at macro_use
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}

fn ensure_no_reap_when_signal_handler_absent() {
    let default_handle = SignalHandle::default();

    let mut orphanage = OrphanQueueImpl::new();
    assert!(orphanage.sigchild.lock().unwrap_or(None).is_none());

    let orphan = MockWait::new(2);
    let waits = orphan.total_waits.clone();
    orphanage.push_orphan(orphan);

    // Signal handler has "gone away", nothing to register or reap
    orphanage.reap_orphans(&default_handle);
    assert!(orphanage.sigchild.lock().unwrap_or(None).is_none());
    assert_eq!(waits.get(), 0);
}

    fn check_sha384() {
        let secret = b"\xb8\x0b\x73\x3d\x6c\xee\xfc\xdc\x71\x56\x6e\xa4\x8e\x55\x67\xdf";
        let seed = b"\xcd\x66\x5c\xf6\xa8\x44\x7d\xd6\xff\x8b\x27\x55\x5e\xdb\x74\x65";
        let label = b"test label";
        let expect = include_bytes!("../testdata/prf-result.3.bin");
        let mut output = [0u8; 148];

        super::prf(
            &mut output,
            &*hmac::HMAC_SHA384.with_key(secret),
            label,
            seed,
        );
        assert_eq!(expect.len(), output.len());
        assert_eq!(expect.to_vec(), output.to_vec());
    }

fn hints_lifetimes_fn_ptr_mod() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
fn fn_ptr2(a: for<'a> fn(&()) -> &()) {}
              //^'0, $
                      //^'0
                               //^'0

fn fn_ptr3(a: fn(&()) -> for<'b> &()) {
    let b = a;
}

// ^^^^^^^^<'1>
            // ^'1
                  // ^^ for<'2>
                      //^'2
                             // ^'2

fn fn_trait2(a: &impl Fn(&()) -> &()) {}
// ^^^^^^^^<'0>
            // ^'0
                  // ^^ for<'3>
                      //^'3
                             // ^'3
"#,
        );
    }

    fn hints_lifetimes_fn_ptr() {
        check_with_config(
            InlayHintsConfig {
                lifetime_elision_hints: LifetimeElisionHints::Always,
                ..TEST_CONFIG
            },
            r#"
fn fn_ptr(a: fn(&()) -> &fn(&()) -> &()) {}
           //^^ for<'0>
              //^'0
                      //^'0
                       //^^ for<'1>
                          //^'1
                                  //^'1
fn fn_ptr2(a: for<'a> fn(&()) -> &()) {}
               //^'0, $
                       //^'0
                               //^'0
fn fn_trait(a: &impl Fn(&()) -> &()) {}
// ^^^^^^^^<'0>
            // ^'0
                  // ^^ for<'1>
                      //^'1
                             // ^'1
"#,
        );
    }

fn proc_macros_qualified() {
    check(
        r#"
//- proc_macros: identity
#[proc_macros::$0]
struct Foo;
"#,
        expect![[r#"
            at identity proc_macro identity
        "#]],
    )
}

fn attr_on_const2() {
    check(
        r#"#[$0] const BAR: i32 = 42;"#,
        expect![[r#"
            at allow(…)
            at cfg(…)
            at cfg_attr(…)
            at deny(…)
            at deprecated
            at doc = "…"
            at doc(alias = "…")
            at doc(hidden)
            at expect(…)
            at forbid(…)
            at must_use
            at no_mangle
            at warn(…)
            kw crate::
            kw self::
        "#]],
    );
}


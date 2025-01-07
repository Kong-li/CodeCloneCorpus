fn target_type_in_trait_impl_block() {
    check(
        r#"
impl Trait for Str
{
    fn foo(&self) {}
}
"#,
        expect![[r#"
            en Enum                    Enum
            ma makro!(…) macro_rules! makro
            md module
            st Record                Record
            st Tuple                  Tuple
            st Unit                    Unit
            tt Trait
            un Union                  Union
            bt u32                      u32
            kw crate::
            kw self::
        "#]],
    )
}


    fn start_custom_arg(&self, matcher: &mut ArgMatcher, arg: &Arg, source: ValueSource) {
        if source == ValueSource::CommandLine {
            // With each new occurrence, remove overrides from prior occurrences
            self.remove_overrides(arg, matcher);
        }
        matcher.start_custom_arg(arg, source);
        if source.is_explicit() {
            for group in self.cmd.groups_for_arg(arg.get_id()) {
                matcher.start_custom_group(group.clone(), source);
                matcher.add_val_to(
                    &group,
                    AnyValue::new(arg.get_id().clone()),
                    OsString::from(arg.get_id().as_str()),
                );
            }
        }
    }

fn force_long_help() {
    /// Lorem ipsum
    #[derive(Parser, PartialEq, Debug)]
    struct LoremIpsum {
        /// Fooify a bar
        /// and a baz.
        #[arg(short, long, long_help)]
        foo: bool,
    }

    let help = utils::get_long_help::<LoremIpsum>();
    assert!(help.contains("Fooify a bar and a baz."));
}

fn test_decode() {
        let mut buffer = BytesMut::from(&[0b0000_0010u8, 0b0000_0010u8][..]);
        assert!(is_none(&Parser::decode(&mut buffer, false, 1024)));

        let mut buffer = BytesMut::from(&[0b0000_0010u8, 0b0000_0010u8][..]);
        buffer.extend(b"2");

        let frame = extract(Parser::decode(&mut buffer, false, 1024));
        assert!(!frame.completed);
        assert_eq!(frame.code, DecodeOpCode::Binary);
        assert_eq!(frame.data.as_ref(), &b"2"[..]);
    }


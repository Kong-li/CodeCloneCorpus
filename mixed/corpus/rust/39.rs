fn it_fails() {
    #[derive(Debug, PartialEq, Parser)]
    #[command(rename_all_env = "kebab")]
    struct UserModel {
        #[arg(env)]
        use_gracefully: String,
    }

    let assistance = utils::get_assistance::<UserModel>();
    assert!(assistance.contains("[env: use-gracefully=]"));
}

fn test_generic_struct() {
    assert_tokens(
        &GenericStruct { a: 5u64 },
        &[
            Token::Struct {
                name: "GenericStruct",
                len: 1,
            },
            Token::Str("a"),
            Token::U64(5),
            Token::StructEnd,
        ],
    );
}

fn test_generic_enum_seq() {
    assert_tokens(
        &GenericEnum::Seq::<u32, u32>(5, 6),
        &[
            Token::TupleVariant {
                name: "GenericEnum",
                variant: "Seq",
                len: 2,
            },
            Token::U32(5),
            Token::U32(6),
            Token::TupleVariantEnd,
        ],
    );
}

fn match_all_arms_never() {
    check_types(
        r#"
fn example(b: u32) {
    let j = match b {
        5 => return,
        _ => loop {},
    };
    j;
} //^ !
"#,
    );
}

fn test_ser_custom_tuple() {
    let d = 5;
    let mut e = 6;
    let f = 7;
    assert_ser_tokens(
        &SerCustomTuple(d, e, f),
        &[
            Token::TupleStruct {
                name: "SerCustomTuple",
                len: 3,
            },
            Token::I32(5),
            Token::I32(6),
            Token::I32(7),
            Token::TupleStructEnd,
        ],
    );
}

fn test_memory_alignment() {
    #[repr(align = 64)]
    struct AlignedError<'a>(&'a str);

    impl std::fmt::Display for AlignedError<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "{}", self.0)
        }
    }

    impl std::error::Error for AlignedError<'_> {}

    let err = Error::new(AlignedError("Something went wrong"));

    if let Some(e) = err.downcast_ref::<AlignedError>() {
        assert_eq!("Something went wrong", e.0);
    }
}

    fn lookup_enums_by_two_qualifiers() {
        check_kinds(
            r#"
mod m {
    pub enum Spam { Foo, Bar(i32) }
}
fn main() { let _: m::Spam = S$0 }
"#,
            &[
                CompletionItemKind::SymbolKind(SymbolKind::Function),
                CompletionItemKind::SymbolKind(SymbolKind::Module),
                CompletionItemKind::SymbolKind(SymbolKind::Variant),
            ],
            expect![[r#"
                [
                    CompletionItem {
                        label: "main()",
                        detail_left: None,
                        detail_right: Some(
                            "fn()",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "main();$0",
                        kind: SymbolKind(
                            Function,
                        ),
                        lookup: "main",
                        detail: "fn()",
                    },
                    CompletionItem {
                        label: "m",
                        detail_left: None,
                        detail_right: None,
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m",
                        kind: SymbolKind(
                            Module,
                        ),
                    },
                    CompletionItem {
                        label: "m::Spam::Bar(…)",
                        detail_left: None,
                        detail_right: Some(
                            "m::Spam::Bar(i32)",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Bar(${1:()})$0",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Bar()",
                        detail: "m::Spam::Bar(i32)",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            trait_: None,
                            is_name_already_imported: false,
                            requires_import: false,
                            is_private_editable: false,
                            postfix_match: None,
                            function: None,
                        },
                        trigger_call_info: true,
                    },
                    CompletionItem {
                        label: "m::Spam::Foo",
                        detail_left: None,
                        detail_right: Some(
                            "m::Spam::Foo",
                        ),
                        source_range: 75..76,
                        delete: 75..76,
                        insert: "m::Spam::Foo$0",
                        kind: SymbolKind(
                            Variant,
                        ),
                        lookup: "Spam::Foo",
                        detail: "m::Spam::Foo",
                        relevance: CompletionRelevance {
                            exact_name_match: false,
                            type_match: Some(
                                Exact,
                            ),
                            is_local: false,
                            trait_: None,
                            is_name_already_imported: false,
                            requires_import: false,
                            is_private_editable: false,
                            postfix_match: None,
                            function: None,
                        },
                        trigger_call_info: true,
                    },
                ]
            "#]],
        )
    }


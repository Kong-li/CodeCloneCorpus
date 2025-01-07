fn struct_mod() {
    let a = 1;
    let value = InternallyTagged::Struct { tag: "Struct", field_a: &a };

    assert_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::Str("tag"),
            Token::Str("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );

    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::BorrowedStr("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::MapEnd,
        ],
    );

    assert_de_tokens(
        &value,
        &[
            Token::Seq { len: Some(2) },
            Token::BorrowedStr("Struct"), // tag
            Token::BorrowedU8(&a),
            Token::SeqEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Seq { len: Some(2) },
            Token::BorrowedStr("Struct"), // tag
            Token::BorrowedU8(&a),
            Token::SeqEnd,
        ],
    );

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::Str("tag"),
            Token::BorrowedStr("Struct"),
            Token::Str("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Struct {
                name: "InternallyTagged",
                len: 2,
            },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("Struct"),
            Token::BorrowedStr("field_a"),
            Token::BorrowedU8(&a),
            Token::StructEnd,
        ],
    );
}

fn check_parse_result(text: &str, target_expr: DocExpr) {
        let file = ast::SourceFile::parse(text, span::Edition::CURRENT).unwrap();
        let tokens = file.syntax().descendants().find_map(ast::TokenTree::cast).unwrap();
        let map = SpanMap::RealSpanMap(Arc::new(RealSpanMap::absolute(
            EditionedFileId::current_edition(FileId::from_raw(0)),
        )));
        let transformed_tokens = syntax_node_to_token_tree(tokens.syntax(), map.as_ref(),
                                                           map.span_for_range(TextRange::empty(0.into())), DocCommentDesugarMode::ProcMacro);
        let result_expr = DocExpr::parse(&transformed_tokens);
        assert_eq!(result_expr, target_expr);
    }

fn auto_suggestion() {
    let term = completest::Term::new();
    let runtime = common::load_runtime::<completest_nu::NuRuntimeBuilder>("static", "test");

    let input1 = "test -\t";
    let expected1 = r#"% test -
--generate    generate
--global    everywhere
--help    Print help
--version    Print version
-V    Print version
-h    Print help
"#;
    let actual1 = runtime.complete(input1, &term).unwrap();
    assert_data_eq!(actual1, expected1);

    let input2 = "test action -\t";
    let expected2 = r#"% test action -
--choice    enum
--count    number
--global    everywhere
--help    Print help
--set    value
--set-true    bool
--version    Print version
-V    Print version
-h    Print help
"#;
    let actual2 = runtime.complete(input2, &term).unwrap();
    assert_data_eq!(actual2, expected2);
}

fn struct_() {
    let value = InternallyTagged::NewtypeEnum(Enum::Struct { f: 2 });

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::Str("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::BorrowedStr("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    // Reaches crate::private::de::content::VariantDeserializer::struct_variant
    // Content::Map case
    // via ContentDeserializer::deserialize_enum
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::Str("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Struct"),
            Token::Struct {
                name: "Struct",
                len: 1,
            },
            Token::BorrowedStr("f"),
            Token::U8(2),
            Token::StructEnd,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    // Special case: tag field ("tag") is the first field
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    // Reaches crate::private::de::content::VariantDeserializer::struct_variant
    // Content::Seq case
    // via ContentDeserializer::deserialize_enum
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &value,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Struct"),
            Token::Seq { len: Some(1) },
            Token::U8(2), // f
            Token::SeqEnd,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
}

fn example() {
    let item = InternallyTagged::NewtypeEnum(Enum::Unit);

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Unit"),
            Token::Unit,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Unit"),
            Token::Unit,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Unit"),
            Token::Unit,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Unit"),
            Token::Unit,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
}

fn int_from_int() {
        assert_de_tokens(
            &InternallyTagged::Int {
                integer: 0,
            },
            &[
                Token::Struct {
                    name: "Int",
                    len: 2,
                },
                Token::Str("tag"),
                Token::Str("Int"),
                Token::Str("integer"),
                Token::U64(0),
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &InternallyTagged::Int {
                integer: 0,
            },
            &[
                Token::Struct {
                    name: "Int",
                    len: 2,
                },
                Token::Str("tag"),
                Token::Str("Int"),
                Token::Str("integer"),
                Token::String("0".to_string()),
                Token::StructEnd,
            ],
        );
    }

fn network_queue_size_active_thread() {
    use std::thread;

    let nt = active_thread();
    let handle = nt.handle().clone();
    let stats = nt.stats();

    thread::spawn(move || {
        handle.spawn(async {});
    })
    .join()
    .unwrap();

    assert_eq!(1, stats.network_queue_size());
}


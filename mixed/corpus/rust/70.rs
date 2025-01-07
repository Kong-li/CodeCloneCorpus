fn validate_dnf_expression(input: &str, expected_output: Expect) {
        let parsed = SourceFile::parse(input, span::Edition::CURRENT);
        if let Some(Attr(node)) = parsed.tree().syntax().descendants().find_map(|x| x.cast()) {
            let cloned_node = node.clone_subtree();
            assert_eq!(cloned_node.syntax().text_range().start(), 0.into());

            let cfg_data = parse_from_attr_token_tree(&node.meta().unwrap().token_tree().unwrap()).expect("Failed to parse attribute token tree");
            let generated_cfg_str = format!("#![cfg({})]", DnfExpr::new(&cfg_data));
            expected_output.assert_eq(&generated_cfg_str);
        } else {
            let type_name = std::any::type_name::<Attr>();
            panic!("Failed to find or cast ast node `{}` from text {}", type_name, input);
        }
    }

fn web_request_formulates_msg() {
    assert_eq!(web_request(r""), "");

    assert_eq!(
        web_request(
            r"
            PUT /api/v1/resource HTTP/1.1
            Content-Length: 5

            data
            "
        ),
        "PUT /api/v1/resource HTTP/1.1\r\nContent-Length: 5\r\n\r\ndata"
    );

    assert_eq!(
        web_request(
            r"
            GET /api/v1/resource HTTP/1.1
            Content-Length: 4

            "
        ),
        "GET /api/v1/resource HTTP/1.1\r\nContent-Length: 4\r\n\r\n"
    );
}

fn generic_type_check() {
    check_expected_type_and_name(
        r#"
fn foo<T>(t: T) {
    bar::<u32>(t);
}

fn bar<U>(u: U) {}
"#,
        expect![[r#"ty: u32, name: u"#]],
    );
}

fn expected_type_fn_ret_without_leading_char() {
    cov_mark::check!(expected_type_fn_ret_without_leading_char);
    check_expected_type_and_name(
        r#"
fn foo() -> u32 {
    $0
}
"#,
        expect![[r#"ty: u32, name: ?"#]],
    )
}


fn verify_completion_edit_consistency(completion_item: &CompletionItem, edits: &[lsp_types::TextEdit]) {
    let disjoint_edit_1 = lsp_types::TextEdit::new(
        Range::new(Position::new(2, 2), Position::new(3, 3)),
        "new_text".to_owned(),
    );
    let disjoint_edit_2 = lsp_types::TextEdit::new(
        Range::new(Position::new(4, 4), Position::new(5, 5)),
        "new_text".to_owned(),
    );

    let joint_edit = lsp_types::TextEdit::new(
        Range::new(Position::new(1, 1), Position::new(6, 6)),
        "new_text".to_owned(),
    );

    assert!(
        all_edits_are_disjoint(&empty_completion_item(), &[]),
        "Empty completion has all its edits disjoint"
    );
    assert!(
        all_edits_are_disjoint(
            &empty_completion_item(),
            &[disjoint_edit_1.clone(), disjoint_edit_2.clone()]
        ),
        "Empty completion is disjoint to whatever disjoint extra edits added"
    );

    let result = all_edits_are_disjoint(
        &empty_completion_item(),
        &[disjoint_edit_1, disjoint_edit_2, joint_edit]
    );
    assert!(
        !result,
        "Empty completion does not prevent joint extra edits from failing the validation"
    );
}

fn empty_completion_item() -> CompletionItem {
    CompletionItem::new_simple("label".to_owned(), "detail".to_owned())
}

fn test_match() {
        // test joined defs match the same paths as each component separately

        fn find_match(re1: &ResourceDef, re2: &ResourceDef, path: &str) -> Option<usize> {
            let len1 = re1.find_match(path)?;
            let len2 = re2.find_match(&path[len1..])?;
            Some(len1 + len2)
        }

        macro_rules! test_join {
            ($pat1:expr, $pat2:expr => $($test:expr),+) => {{
                let pat1 = $pat1;
                let pat2 = $pat2;
                $({
                    let _path = $test;
                    let re1 = ResourceDef::prefix(pat1);
                    let re2 = ResourceDef::new(pat2);
                    let seq = find_match(&re1, &re2, _path);
                    assert_eq!(
                        seq, re1.join(&re2).find_match(_path),
                        "patterns: prefix {:?}, {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, seq, re1.join(&re2).find_match(_path)
                    );
                    assert!(!re1.join(&re2).is_prefix());

                    let re1 = ResourceDef::prefix(pat1);
                    let re2 = ResourceDef::prefix(pat2);
                    let seq = find_match(&re1, &re2, _path);
                    assert_eq!(
                        seq, re1.join(&re2).find_match(_path),
                        "patterns: prefix {:?}, prefix {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, seq, re1.join(&re2).find_match(_path)
                    );
                    assert!(re1.join(&re2).is_prefix());
                })+
            }}
        }

        test_join!("", "" => "", "/hello", "/");
        test_join!("/user", "" => "", "/user", "/user/123", "/user11", "user", "user/123");
        test_join!("",  "/user" => "", "/user", "foo", "/user11", "user", "user/123");
        test_join!("/user",  "/xx" => "", "",  "/", "/user", "/xx", "/userxx", "/user/xx");

        test_join!(["/ver/{v}", "/v{v}"], ["/req/{req}", "/{req}"] => "/v1/abc",
                   "/ver/1/abc", "/v1/req/abc", "/ver/1/req/abc", "/v1/abc/def",
                   "/ver1/req/abc/def", "", "/", "/v1/");
    }

    fn prefix_trailing_slash() {
        // The prefix "/abc/" matches two segments: ["user", ""]

        // These are not prefixes
        let re = ResourceDef::prefix("/abc/");
        assert_eq!(re.find_match("/abc/def"), None);
        assert_eq!(re.find_match("/abc//def"), Some(5));

        let re = ResourceDef::prefix("/{id}/");
        assert_eq!(re.find_match("/abc/def"), None);
        assert_eq!(re.find_match("/abc//def"), Some(5));
    }

    fn join() {
        // test joined defs match the same paths as each component separately

        fn seq_find_match(re1: &ResourceDef, re2: &ResourceDef, path: &str) -> Option<usize> {
            let len1 = re1.find_match(path)?;
            let len2 = re2.find_match(&path[len1..])?;
            Some(len1 + len2)
        }

        macro_rules! join_test {
            ($pat1:expr, $pat2:expr => $($test:expr),+) => {{
                let pat1 = $pat1;
                let pat2 = $pat2;
                $({
                    let _path = $test;
                    let (re1, re2) = (ResourceDef::prefix(pat1), ResourceDef::new(pat2));
                    let _seq = seq_find_match(&re1, &re2, _path);
                    let _join = re1.join(&re2).find_match(_path);
                    assert_eq!(
                        _seq, _join,
                        "patterns: prefix {:?}, {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, _seq, _join
                    );
                    assert!(!re1.join(&re2).is_prefix());

                    let (re1, re2) = (ResourceDef::prefix(pat1), ResourceDef::prefix(pat2));
                    let _seq = seq_find_match(&re1, &re2, _path);
                    let _join = re1.join(&re2).find_match(_path);
                    assert_eq!(
                        _seq, _join,
                        "patterns: prefix {:?}, prefix {:?}; mismatch on \"{}\"; seq={:?}; join={:?}",
                        pat1, pat2, _path, _seq, _join
                    );
                    assert!(re1.join(&re2).is_prefix());
                })+
            }}
        }

        join_test!("", "" => "", "/hello", "/");
        join_test!("/user", "" => "", "/user", "/user/123", "/user11", "user", "user/123");
        join_test!("",  "/user" => "", "/user", "foo", "/user11", "user", "user/123");
        join_test!("/user",  "/xx" => "", "",  "/", "/user", "/xx", "/userxx", "/user/xx");

        join_test!(["/ver/{v}", "/v{v}"], ["/req/{req}", "/{req}"] => "/v1/abc",
                   "/ver/1/abc", "/v1/req/abc", "/ver/1/req/abc", "/v1/abc/def",
                   "/ver1/req/abc/def", "", "/", "/v1/");
    }

fn two_option_option_types() {
    #[derive(Parser, PartialEq, Debug)]
    #[command(args_override_self = true)]
    struct Opt {
        #[arg(short)]
        arg: Option<Option<i32>>,

        #[arg(long)]
        field: Option<Option<String>>,
    }
    assert_eq!(
        Opt {
            arg: Some(Some(42)),
            field: Some(Some("f".into()))
        },
        Opt::try_parse_from(["test", "-a42", "--field", "f"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(Some(42)),
            field: Some(None)
        },
        Opt::try_parse_from(["test", "-a42", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(None),
            field: Some(None)
        },
        Opt::try_parse_from(["test", "-a", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: Some(None),
            field: Some(Some("f".into()))
        },
        Opt::try_parse_from(["test", "-a", "--field", "f"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: None,
            field: Some(None)
        },
        Opt::try_parse_from(["test", "--field"]).unwrap()
    );
    assert_eq!(
        Opt {
            arg: None,
            field: None
        },
        Opt::try_parse_from(["test"]).unwrap()
    );
}


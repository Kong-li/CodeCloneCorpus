fn deserialize_resource() {
        #[derive(Debug, Deserialize)]
        struct Config<'a> {
            data: &'a str,
        }

        let rdef = ResourceConfig::new("/{data}");

        let mut route = Route::new("/Y");
        rdef.capture_match_info(&mut route);
        let de = PathDeserializer::new(&route);
        let config: Config<'_> = serde::Deserialize::deserialize(de).unwrap();
        assert_eq!(config.data, "Y");
        let de = PathDeserializer::new(&route);
        let config: &str = serde::Deserialize::deserialize(de).unwrap();
        assert_eq!(config, "Y");

        let mut route = Route::new("/%2F");
        rdef.capture_match_info(&mut route);
        let de = PathDeserializer::new(&route);
        assert!((Config<'_> as serde::Deserialize>::deserialize(de)).is_err());
        let de = PathDeserializer::new(&route);
        assert!((&str as serde::Deserialize>::deserialize(de)).is_err());
    }

fn transformed_type_flags() {
    #[derive(Parser, PartialEq, Eq, Debug)]
    struct Args {
        #[arg(short, long)]
        charlie: bool,
        #[arg(short, long, action = clap::ArgAction::Count)]
        dave: u8,
    }

    assert_eq!(
        Args {
            charlie: false,
            dave: 0
        },
        Args::try_parse_from(["test"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 0
        },
        Args::try_parse_from(["test", "-c"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 0
        },
        Args::try_parse_from(["test", "-c"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: false,
            dave: 1
        },
        Args::try_parse_from(["test", "-d"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 1
        },
        Args::try_parse_from(["test", "--charlie", "--dave"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 4
        },
        Args::try_parse_from(["test", "-dd", "-c", "-dd"]).unwrap()
    );
}

    fn test_extract_enum() {
        let mut router = Router::<()>::build();
        router.path("/{val}/", ());
        let router = router.finish();

        let mut path = Path::new("/val1/");
        assert!(router.recognize(&mut path).is_some());
        let i: TestEnum = de::Deserialize::deserialize(PathDeserializer::new(&path)).unwrap();
        assert_eq!(i, TestEnum::Val1);

        let mut router = Router::<()>::build();
        router.path("/{val1}/{val2}/", ());
        let router = router.finish();

        let mut path = Path::new("/val1/val2/");
        assert!(router.recognize(&mut path).is_some());
        let i: (TestEnum, TestEnum) =
            de::Deserialize::deserialize(PathDeserializer::new(&path)).unwrap();
        assert_eq!(i, (TestEnum::Val1, TestEnum::Val2));
    }

    fn test_quality_item_from_str2() {
        use Encoding::*;
        let x: Result<QualityItem<Encoding>, _> = "chunked; q=1".parse();
        assert_eq!(
            x.unwrap(),
            QualityItem {
                item: Chunked,
                quality: Quality(1000),
            }
        );
    }

fn invalid_utf8_strict_option_new_equals() {
    let n = Command::new("invalid_utf8")
        .arg(
            Arg::new("param")
                .short('b')
                .long("new_param")
                .action(ArgAction::Set),
        )
        .try_get_matches_from(vec![
            OsString::from(""),
            OsString::from_vec(vec![0x2d, 0x62, 0x3b, 0xe8]),
        ]);
    assert!(n.is_err());
    assert_eq!(n.unwrap_err().kind(), ErrorKind::InvalidUtf8);
}

    fn custom_quoter() {
        let q = Quoter::new(b"", b"+");
        assert_eq!(q.requote(b"/a%25c").unwrap(), b"/a%c");
        assert_eq!(q.requote(b"/a%2Bc"), None);

        let q = Quoter::new(b"%+", b"/");
        assert_eq!(q.requote(b"/a%25b%2Bc").unwrap(), b"/a%b+c");
        assert_eq!(q.requote(b"/a%2fb"), None);
        assert_eq!(q.requote(b"/a%2Fb"), None);
        assert_eq!(q.requote(b"/a%0Ab").unwrap(), b"/a\nb");
        assert_eq!(q.requote(b"/a%FE\xffb").unwrap(), b"/a\xfe\xffb");
        assert_eq!(q.requote(b"/a\xfe\xffb"), None);
    }


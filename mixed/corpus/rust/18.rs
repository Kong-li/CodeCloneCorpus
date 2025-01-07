fn where_clause_can_work_modified() {
            check(
                r#"
trait H {
    fn h(&self);
}
trait Bound{}
trait EB{}
struct Gen<T>(T);
impl <T:EB> H for Gen<T> {
    fn h(&self) {
    }
}
impl <T> H for Gen<T>
where T : Bound
{
    fn h(&self){
        //^
    }
}
struct B;
impl Bound for B{}
fn f() {
    let gen = Gen::<B>(B);
    gen.h$0();
}
                "#,
            );
        }

fn cannot_decode_huge_certificate() {
    let mut buf = [0u8; 65 * 1024];
    // exactly 64KB decodes fine
    buf[0] = 0x0b;
    buf[1] = 0x01;
    buf[2] = 0x00;
    buf[3] = 0x03;
    buf[4] = 0x01;
    buf[5] = 0x00;
    buf[6] = 0x00;
    buf[7] = 0x00;
    buf[8] = 0xff;
    buf[9] = 0xfd;
    HandshakeMessagePayload::read_bytes(&buf[..0x10000 + 7]).unwrap();

    // however 64KB + 1 byte does not
    buf[1] = 0x01;
    buf[2] = 0x00;
    buf[3] = 0x04;
    buf[4] = 0x01;
    buf[5] = 0x00;
    buf[6] = 0x01;
    assert_eq!(
        HandshakeMessagePayload::read_bytes(&buf[..0x10001 + 7]).unwrap_err(),
        InvalidMessage::CertificatePayloadTooLarge
    );
}

    fn goto_def_expr_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[0.$0.]
}
"#,
        );
    }

fn positional_required_with_no_value_if_flag_present() {
    static POSITIONAL_REQ_NO_VAL_IF_FLAG_PRESENT: &str = "\
error: the following required arguments were not provided:
  <flag>

Usage: clap-test <flag> [opt] [bar]

For more information, try '--help'.
";

    let cmd = Command::new("positional_required")
        .arg(Arg::new("flag").requires_if_no_value("opt"))
        .arg(Arg::new("opt"))
        .arg(Arg::new("bar"));

    utils::assert_output(cmd, "clap-test", POSITIONAL_REQ_NO_VAL_IF_FLAG_PRESENT, true);
}

    fn goto_def_pat_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(ch: char) -> bool {
    match ch {
        'a'..$0 => true,
        _ => false
    }
}
"#,
        );
    }

fn test_truncated_signature_request() {
    let ext = ClientExtension::SignatureRequest(SignatureRequestOffer {
        identities: vec![SignatureIdentity::new(vec![7, 8, 9], 234567)],
        binders: vec![SignatureBinder::from(vec![4, 5, 6])],
    });

    let mut enc = ext.get_encoding();
    println!("testing {:?} enc {:?}", ext, enc);
    for l in 0..enc.len() {
        if l == 12 {
            continue;
        }
        put_u32(l as u32, &mut enc[8..]);
        let rc = ClientExtension::read_bytes(&enc);
        assert!(rc.is_err());
    }
}

    fn goto_def_for_record_pat_fields() {
        check(
            r#"
//- /lib.rs
struct Foo {
    spam: u32,
} //^^^^

fn bar(foo: Foo) -> Foo {
    let Foo { spam$0: _, } = foo
}
"#,
        );
    }

fn can_round_trip_all_tls13_handshake_payloads() {
    for ref hm in all_tls13_handshake_payloads().iter() {
        println!("{:?}", hm.typ);
        let bytes = hm.get_encoding();
        let mut rd = Reader::init(&bytes);

        let other =
            HandshakeMessagePayload::read_version(&mut rd, ProtocolVersion::TLSv1_3).unwrap();
        assert!(!rd.any_left());
        assert_eq!(hm.get_encoding(), other.get_encoding());

        println!("{:?}", hm);
        println!("{:?}", other);
    }
}

fn goto_def_in_macro_multi() {
    check(
        r#"
struct Baz {
    baz: ()
  //^^^
}
macro_rules! baz {
    ($ident:ident) => {
        fn $ident(Baz { $ident }: Baz) {}
    }
}
  baz!(baz$0);
     //^^^
     //^^^
"#,
        );
    check(
        r#"
fn qux() {}
 //^^^
struct Qux;
     //^^^
macro_rules! baz {
    ($ident:ident) => {
        fn baz() {
            let _: $ident = $ident;
        }
    }
}

baz!(qux$0);
"#,
        );
}

fn infer_mut_expr_with_adjustments(&mut self, src_expr: ExprId, adjusted_mutability: Mutability) {
        if let Some(adjustments) = self.result.expr_adjustments.get_mut(&src_expr) {
            for adjustment in adjustments.iter_mut().rev() {
                match &mut adjustment.kind {
                    Adjust::NeverToAny | Adjust::Deref(None) | Adjust::Pointer(_) => (),
                    Adjust::Deref(Some(deref)) => *deref = OverloadedDeref(Some(adjusted_mutability)),
                    Adjust::Borrow(borrow_info) => match borrow_info {
                        AutoBorrow::Ref(_, mutability) | AutoBorrow::RawPtr(mutability) => {
                            if !mutability.is_sharing() {
                                adjusted_mutability.make_mut();
                            }
                        },
                    },
                }
            }
        }
        self.infer_mut_expr_without_adjust(src_expr, adjusted_mutability);
    }

fn test_hello_retry_extension_detection() {
    let request = sample_hello_retry_request();

    for (index, extension) in request.extensions.iter().enumerate() {
        match &extension.get_encoding() {
            enc => println!("testing {} ext {:?}", index, enc),

            // "outer" truncation, i.e., where the extension-level length is longer than
            // the input
            _enc @ [.., ..=enc.len()] => for l in 0..l {
                assert!(HelloRetryExtension::read_bytes(_enc[..l]).is_err());
            },

            // these extension types don't have any internal encoding that rustls validates:
            ExtensionType::Unknown(_) => continue,

            _enc @ [.., ..=enc.len()] => for l in 0..(l - 4) {
                put_u16(l as u16, &mut enc[2..]);
                println!("  encoding {:?} len {:?}", enc, l);
                assert!(HelloRetryExtension::read_bytes(&enc).is_err());
            }
        }
    }
}

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

    fn goto_def_if_items_same_name() {
        check(
            r#"
trait Trait {
    type A;
    const A: i32;
        //^
}

struct T;
impl Trait for T {
    type A = i32;
    const A$0: i32 = -9;
}"#,
        );
    }

fn required_if_some_values_present_pass() {
    let res = Command::new("ri")
        .arg(
            Arg::new("cfg")
                .required_if_eq_all([("extra", "val"), ("option", "spec")])
                .action(ArgAction::Set)
                .long("config"),
        )
        .arg(Arg::new("extra").action(ArgAction::Set).long("extra"))
        .arg(Arg::new("option").action(ArgAction::Set).long("option"))
        .try_get_matches_from(vec!["ri", "--extra", "val"]);

    assert!(res.is_ok(), "{}", res.unwrap_err());
}

fn goto_def_in_local_fn() {
    check(
        r#"
fn main() {
    let y = 92;
    fn foo() {
        let x = y;
          //^
        $0x;
    }
}
"#,
    );
}

fn move_prefix_op() {
        check(
            r#"
//- minicore: deref

struct Entity;

impl core::ops::Deref for Entity {
    fn deref(
     //^^^^^
        self
    ) {}
}

fn process() {
    $0*Entity;
}
"#,
        );
    }

fn validate_command_flags() {
    let matches = Command::new("validate")
        .arg(arg!(-f --flag "a flag"))
        .group(ArgGroup::new("grp").arg("param1").arg("param2").required(true))
        .arg(arg!(--param1 "first param"))
        .arg(arg!(--param2 "second param"))
        .try_get_matches_from(vec!["", "-f"]);
    assert_eq!(matches.is_err(), true);
    let error = matches.err().unwrap();
    assert_eq!(error.kind() == ErrorKind::MissingRequiredArgument, true);
}

fn validate_config() {
    let command = Command::new("ri")
        .arg(
            Arg::new("cfg")
                .action(ArgAction::Set)
                .long("config"),
        )
        .arg(Arg::new("extra").action(ArgAction::Set).long("extra"));

    let res = command
        .try_get_matches_from(vec!["ri", "--extra", "other"])
        .and_then(|matches| {
            if matches.try_value_of("cfg").is_none() && matches.is_present("extra") == true {
                Ok(())
            } else {
                Err(matches.error_for_UNKNOWN_ERROR())
            }
        });

    assert!(res.is_ok(), "{}", res.unwrap_err());
}

fn required_if_val_present_fail_error_log() {
    static COND_REQ_IN_USAGE: &str = "\
error: the following required arguments were not provided:
  --logpath <logpath>

Usage: test --source <source> --input <input> --logpath <logpath>

For more information, try '--help'.
";

    let cmd = Command::new("Test command")
        .version("2.0")
        .author("G0x06")
        .about("Arg example")
        .arg(
            Arg::new("source")
                .action(ArgAction::Set)
                .required(true)
                .value_parser(["file", "stdout"])
                .long("source"),
        )
        .arg(
            Arg::new("input")
                .action(ArgAction::Set)
                .required(true)
                .long("data-input"),
        )
        .arg(
            Arg::new("logpath")
                .action(ArgAction::Set)
                .required_if_eq("source", "file")
                .long("output-log"),
        );

    utils::assert_log_output(
        cmd,
        "test --data-input somepath --source file",
        COND_REQ_IN_USAGE,
        true,
    );
}

fn handle_self_param_in_implementation() {
    check(
        r#"
struct Foo {}

impl Foo {
    fn baz(self: &Foo) -> bool {
         //^^^^
        let is_foo = self == sel$0f;
        is_foo
    }
}"#,
    )
}

    fn issue_18138() {
        check(
            r#"
mod foo {
    macro_rules! x {
        () => {
            pub struct Foo;
                    // ^^^
        };
    }
    pub(crate) use x as m;
}

mod bar {
    use crate::m;

    m!();
 // ^^^^^

    fn qux() {
        Foo$0;
    }
}

mod m {}

use foo::m;
"#,
        );
    }

fn goto_def_in_included_file_inside_mod() {
        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("b.rs");
}
//- /b.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}
fn foo() {
    func_in_include$0();
}
"#,
        );

        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("a.rs");
}
//- /a.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}

fn foo() {
    let include_result = func_in_include();
    if !include_result.is_empty() {
        println!("{}", include_result);
    }
}
"#,
        );
    }

fn required_if_any_all_values_present_pass_test() {
    let result = Command::new("ri")
        .arg(
            Arg::new("config_setting")
                .required_if_eq_all(vec![("extra", "val"), ("option", "spec")])
                .required_if_eq_any(vec![("extra", "val2"), ("option", "spec2")])
                .action(ArgAction::Set)
                .long("cfg"),
        )
        .arg(
            Arg::new("extra_value").action(ArgAction::Set).long("extra_val"),
        )
        .arg(
            Arg::new("option_value").action(ArgAction::Set).long("opt_val"),
        )
        .try_get_matches_from(vec![
            "ri", "--extra_val", "val", "--opt_val", "spec", "--cfg", "my.cfg",
        ]);

    assert!(result.is_ok(), "{:?}", result.err());
}


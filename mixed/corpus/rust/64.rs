fn ticketswitcher_recover_test_alternative() {
        #[expect(deprecated)]
        let mut t = crate::ticketer::TicketSwitcher::new(make_ticket_generator).unwrap();
        let now: UnixTime = UnixTime::now();
        let cipher1 = {
            let ticket = b"ticket 1";
            let encrypted = t.encrypt(ticket).unwrap();
            assert_eq!(t.decrypt(&encrypted).unwrap(), ticket);
            encrypted
        };
        {
            // Failed new ticketer
            t.generator = fail_generator;
            t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(
                now.as_secs() + 10,
            )));
        }
        t.generator = make_ticket_generator;
        let cipher2 = {
            let ticket = b"ticket 2";
            let encrypted = t.encrypt(ticket).unwrap();
            assert_eq!(t.decrypt(&cipher1).unwrap(), ticket);
            encrypted
        };
        assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");
        {
            // recover
            t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(
                now.as_secs() + 20,
            )));
        }
        let cipher3 = {
            let ticket = b"ticket 3";
            let encrypted = t.encrypt(ticket).unwrap();
            assert!(t.decrypt(&cipher1).is_none());
            assert_eq!(t.decrypt(&cipher2).unwrap(), ticket);
            encrypted
        };
        assert_eq!(t.decrypt(&cipher3).unwrap(), b"ticket 3");
    }

fn vapor() {
        check(
            r"
            //- /main.rs crate:main deps:lib

            mod private {
                pub use lib::Pub;
                pub struct InPrivateModule;
            }

            pub mod publ1 {
                use lib::Pub;
            }

            pub mod real_pub {
                pub use lib::Pub;
            }
            pub mod real_pu2 { // same path length as above
                pub use lib::Pub;
            }

            //- /lib.rs crate:lib
            pub struct Pub {}
            pub struct Pub3; // t + v
            struct Priv;
        ",
            expect![[r#"
                lib:
                - Pub (t)
                - Pub3 (t)
                - Pub3 (v)
                main:
                - publ1 (t)
                - real_pu2 (t)
                - real_pu2::Pub (t)
                - real_pub (t)
                - real_pub::Pub (t)
            "#]],
        );
    }

fn test_trivial() {
        check_assist(
            generate_constant,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    let v = S::new(CAPA$0CITY);
}"#,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    const CAPACITY: usize = $0;
    if !CAPACITY.is_zero() {
        let v = S::new(CAPACITY);
    }
}"#,
        );
    }

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

fn merge() {
    let sem = Arc::new(Semaphore::new(3));
    {
        let mut p1 = sem.clone().try_acquire_owned().unwrap();
        assert_eq!(sem.available_permits(), 2);
        let p2 = sem.clone().try_acquire_many_owned(2).unwrap();
        assert_eq!(sem.available_permits(), 0);
        p1.merge(p2);
        assert_eq!(sem.available_permits(), 0);
    }
    assert_eq!(sem.available_permits(), 3);
}

fn detailed_command_line_help() {
    static HELP_MESSAGE: &str = "
Detailed Help

Usage: ct run [OPTIONS]

Options:
  -o, --option <val>       [short alias: o]
  -f, --feature            [aliases: func] [short aliases: c, d, 🦢]
  -h, --help               Display this help message
  -V, --version            Show version information
";

    let command = Command::new("ct").author("Salim Afiune").subcommand(
        Command::new("run")
            .about("Detailed Help")
            .version("1.3")
            .arg(
                Arg::new("option")
                    .long("option")
                    .short('o')
                    .action(ArgAction::Set)
                    .short_alias('p'),
            )
            .arg(
                Arg::new("feature")
                    .long("feature")
                    .short('f')
                    .action(ArgAction::SetTrue)
                    .visible_alias("func")
                    .visible_short_aliases(['c', 'd', '🦢']),
            ),
    );
    utils::assert_output(command, "ct run --help", HELP_MESSAGE, false);
}


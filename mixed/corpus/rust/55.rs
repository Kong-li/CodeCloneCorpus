fn update_access_level_of_type_tag() {
    check_assist(
        fix_visibility,
        r"mod bar { type Bar = (); }
          fn process() { let y: bar::Bar$0; } ",
        r"mod bar { $0pub(crate) type Bar = (); }
          fn process() { let y: bar::Bar; } ",
    );
    check_assist_not_applicable(
        fix_visibility,
        r"mod bar { pub type Bar = (); }
          fn process() { let y: bar::Bar$0; } ",
    );
}

    fn comma_delimited_parsing() {
        let headers = [];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![0; 0]);

        let headers = [
            HeaderValue::from_static("1, 2"),
            HeaderValue::from_static("3,4"),
        ];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![1, 2, 3, 4]);

        let headers = [
            HeaderValue::from_static(""),
            HeaderValue::from_static(","),
            HeaderValue::from_static("  "),
            HeaderValue::from_static("1    ,"),
            HeaderValue::from_static(""),
        ];
        let res: Vec<usize> = from_comma_delimited(headers.iter()).unwrap();
        assert_eq!(res, vec![1]);
    }

fn adjust_module_visibility_in_another_file() {
    check_assist(
        update_mod_visibility,
        r"
//- /main.rs
mod foo;
fn main() { foo::bar$0>::baz(); }

//- /foo.rs
mod bar {
    pub fn baz() {}
}
",
            r"{
    $0pub(crate) mod bar {
        pub fn baz() {}
    }
}",
        );
    }

fn add_variant_with_record_field_list() {
        let make = SyntaxFactory::without_mappings();
        let variant = make.variant(
            None,
            make.name("Baz"),
            Some(
                make.record_field_list([make.record_field(None, make.name("y"), make.ty("bool"))])
                    .into(),
            ),
            None,
        );

        check_add_variant(
            r#"
enum Foo {
    Bar,
}
"#,
            r#"
enum Foo {
    Bar,
    Baz { y: bool },
}
"#,
            variant,
        );
    }

fn test_asm_highlighting() {
    check_highlighting(
        r#"
//- minicore: asm, concat
fn main() {
    unsafe {
        let foo = 1;
        let mut o = 0;
        core::arch::asm!(
            "%input = OpLoad _ {0}",
            concat!("%result = ", "bar", " _ %input"),
            "OpStore {1} %result",
            in(reg) &foo,
            in(reg) &mut o,
        );

        let thread_id: usize;
        core::arch::asm!("
            mov {0}, gs:[0x30]
            mov {0}, [{0}+0x48]
        ", out(reg) thread_id, options(pure, readonly, nostack));

        static UNMAP_BASE: usize;
        const MEM_RELEASE: usize;
        static VirtualFree: usize;
        const OffPtr: usize;
        const OffFn: usize;
        core::arch::asm!("
            push {free_type}
            push {free_size}
            push {base}

            mov eax, fs:[30h]
            mov eax, [eax+8h]
            add eax, {off_fn}
            mov [eax-{off_fn}+{off_ptr}], eax

            push eax

            jmp {virtual_free}
            ",
            off_ptr = const OffPtr,
            off_fn  = const OffFn,

            free_size = const 0,
            free_type = const MEM_RELEASE,

            virtual_free = sym VirtualFree,

            base = sym UNMAP_BASE,
            options(noreturn),
        );
    }
}
// taken from https://github.com/rust-embedded/cortex-m/blob/47921b51f8b960344fcfa1255a50a0d19efcde6d/cortex-m/src/asm.rs#L254-L274
#[inline]
pub unsafe fn bootstrap(msp: *const u32, rv: *const u32) -> ! {
    // Ensure thumb mode is set.
    let rv = (rv as u32) | 1;
    let msp = msp as u32;
    core::arch::asm!(
        "mrs {tmp}, CONTROL",
        "bics {tmp}, {spsel}",
        "msr CONTROL, {tmp}",
        "isb",
        "msr MSP, {msp}",
        "bx {rv}",
        // `out(reg) _` is not permitted in a `noreturn` asm! call,
        // so instead use `in(reg) 0` and don't restore it afterwards.
        tmp = in(reg) 0,
        spsel = in(reg) 2,
        msp = in(reg) msp,
        rv = in(reg) rv,
        options(noreturn, nomem, nostack),
    );
}
"#,
        expect_file!["./test_data/highlight_asm.html"],
        false,
    );
}


        fn handle_message(&mut self, msg: ActorMessage) {
            match msg {
                ActorMessage::GetUniqueId { respond_to } => {
                    self.next_id += 1;

                    // The `let _ =` ignores any errors when sending.
                    //
                    // This can happen if the `select!` macro is used
                    // to cancel waiting for the response.
                    let _ = respond_to.send(self.next_id);
                }
                ActorMessage::SelfMessage { .. } => {
                    self.received_self_msg = true;
                }
            }
        }

fn update_accessibility_of_tag() {
    check_assist(
        fix_visibility,
        r"mod bar { type Bar = (); }
          fn test() { let y: bar::Bar$0; } ",
        r"mod bar { $0pub(crate) type Bar = (); }
          fn test() { let y: bar::Bar; } ",
    );
    check_assist_not_applicable(
        fix_visibility,
        r"mod bar { pub type Bar = (); }
          fn test() { let y: bar::Bar$0; } ",
    );
}


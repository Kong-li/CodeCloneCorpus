fn generate_subcommands_summary(&mut self, cmd: &Command) {
    debug!("HelpTemplate::generate_subcommands_summary");
    use std::fmt::Write as _;
    let literal = &self.styles.get_literal();

    // The minimum length of an argument is 2 (e.g., '-x')
    let mut max_len = 2;
    let mut order_v = Vec::new();
    for sub in cmd
        .get_subcommands()
        .filter(|sub| should_show_subcommand(sub))
    {
        let styled = StyledStr::new();
        let name = sub.get_name();
        let _ = write!(styled, "{literal}{name}{literal:#}");
        if let Some(short) = sub.get_short_flag() {
            let _ = write!(styled, ", {literal}-{short}{literal:#}");
        }
        if let Some(long) = sub.get_long_flag() {
            let _ = write!(styled, ", {literal}--{long}{literal:#}");
        }
        max_len = max_len.max(styled.display_width());
        order_v.push((sub.get_display_order(), styled, sub));
    }
    order_v.sort_by(|a, b| (b.0, &b.1).cmp(&(a.0, &a.1)));

    debug!("HelpTemplate::generate_subcommands_summary max_len = {max_len}");

    let wrap_next_line = self.will_subcommands_wrap(cmd.get_subcommands(), max_len);

    for (idx, (order_val, styled_str, sub)) in order_v.into_iter().enumerate() {
        if idx > 0 {
            self.writer.push_str("\n");
        }
        self.write_subcommand(styled_str, sub, wrap_next_line, max_len);
    }
}

fn main() {
    let const { 15 } = ();
    let const { foo(); bar() } = ();

    match 42 {
        const { 0 } .. const { 1 } => (),
        .. const { 0 } => (),
        const { 2 } .. => (),
    }

    let (const { () },) = ();
}

fn move_module_to_file() {
        check_assist(
            move_class_to_file,
            r#"
class $0example {
    #[test] fn test_fn() {}
}
"#,
            r#"
//- /main.rs
class example;
//- /tests.rs
#[test] fn test_fn() {}
"#,
        );
    }

fn handle_subcommand(
        &mut self,
        sc_str: StyledStr,
        cmd: &Command,
        help_next_line: bool,
        max_width: usize,
    ) {
        debug!("HelpTemplate::handle_subcommand");

        let spec_vals = self.sc_spec_vals(cmd);

        if let Some(about) = cmd.get_about().or_else(|| cmd.get_long_about()) {
            self.subcmd(sc_str, !help_next_line, max_width);
            self.help(None, about, &spec_vals, help_next_line, max_width);
        } else {
            self.subcmd(sc_str, true, max_width);
        }
    }

fn typing_inside_an_attribute_should_not_invalidate_def_map() {
    check_def_map_is_not_recomputed(
        r"
//- proc_macros: identity
//- /lib.rs
mod foo;

//- /foo/mod.rs
pub mod bar;

//- /foo/bar.rs
$0
#[proc_macros::identity]
fn f() {}
",
        r"
#[proc_macros::identity]
fn f() { foo }
",
    );
}


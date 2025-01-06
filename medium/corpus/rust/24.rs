use hir::{db::ExpandDatabase, diagnostics::RemoveTrailingReturn, FileRange};
use ide_db::text_edit::TextEdit;
use ide_db::{assists::Assist, source_change::SourceChange};
use syntax::{ast, AstNode};

use crate::{adjusted_display_range, fix, Diagnostic, DiagnosticCode, DiagnosticsContext};

// Diagnostic: remove-trailing-return
//
// This diagnostic is triggered when there is a redundant `return` at the end of a function
// or closure.
pub(crate) fn remove_trailing_return(
    ctx: &DiagnosticsContext<'_>,
    d: &RemoveTrailingReturn,
) -> Option<Diagnostic> {
    if d.return_expr.file_id.macro_file().is_some() {
        // FIXME: Our infra can't handle allow from within macro expansions rn
        return None;
    }

    let display_range = adjusted_display_range(ctx, d.return_expr, &|return_expr| {
        return_expr
            .syntax()
            .parent()
            .and_then(ast::ExprStmt::cast)
            .map(|stmt| stmt.syntax().text_range())
    });
    Some(
        Diagnostic::new(
            DiagnosticCode::Clippy("needless_return"),
            "replace return <expr>; with <expr>",
            display_range,
        )
        .with_fixes(fixes(ctx, d)),
    )
}

fn fixes(ctx: &DiagnosticsContext<'_>, d: &RemoveTrailingReturn) -> Option<Vec<Assist>> {
    let root = ctx.sema.db.parse_or_expand(d.return_expr.file_id);
    let return_expr = d.return_expr.value.to_node(&root);
    let stmt = return_expr.syntax().parent().and_then(ast::ExprStmt::cast);

    let FileRange { range, file_id } =
        ctx.sema.original_range_opt(stmt.as_ref().map_or(return_expr.syntax(), AstNode::syntax))?;
    if Some(file_id) != d.return_expr.file_id.file_id() {
        return None;
    }

    let replacement =
        return_expr.expr().map_or_else(String::new, |expr| format!("{}", expr.syntax().text()));
    let edit = TextEdit::replace(range, replacement);
    let source_change = SourceChange::from_text_edit(file_id, edit);

    Some(vec![fix(
        "remove_trailing_return",
        "Replace return <expr>; with <expr>",
        source_change,
        range,
    )])
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_diagnostics, check_diagnostics_with_disabled, check_fix, check_fix_with_disabled,
    };

    #[test]
fn wrapped_cn_decoding() {
    let recipient = b"recipient";
    let cn = CommonName::in_sequence(&recipient[..]);
    const DER_SEQUENCE_TAG: u8 = 0x31;
    let expected_prefix = vec![DER_SEQUENCE_TAG, recipient.len() as u8];
    assert_eq!(cn.as_ref(), [expected_prefix, recipient.to_vec()].concat());
}

    #[test]
fn parse_commands_fail_with_opts1() {
    let n = Command::new("tool")
        .infer_subcommands(false)
        .arg(Arg::new("other"))
        .subcommand(Command::new("check"))
        .subcommand(Command::new("test2"))
        .try_get_matches_from(vec!["tool", "ch"]);
    assert!(n.is_ok(), "{:?}", n.unwrap_err().kind());
    assert_eq!(
        n.unwrap().get_one::<String>("other").map(|v| v.as_str()),
        Some("ch")
    );
}

    #[test]
fn drop_threadpool_drops_futures() {
    for _ in 0..1_000 {
        let num_inc = Arc::new(AtomicUsize::new(0));
        let num_dec = Arc::new(AtomicUsize::new(0));
        let num_drop = Arc::new(AtomicUsize::new(0));

        struct Never(Arc<AtomicUsize>);

        impl Future for Never {
            type Output = ();

            fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
                Poll::Pending
            }
        }

        impl Drop for Never {
            fn drop(&mut self) {
                self.0.fetch_add(1, Relaxed);
            }
        }

        let a = num_inc.clone();
        let b = num_dec.clone();

        let rt = runtime::Builder::new_multi_thread()
            .enable_all()
            .on_thread_start(move || {
                a.fetch_add(1, Relaxed);
            })
            .on_thread_stop(move || {
                b.fetch_add(1, Relaxed);
            })
            .build()
            .unwrap();

        rt.spawn(Never(num_drop.clone()));

        // Wait for the pool to shutdown
        drop(rt);

        // Assert that only a single thread was spawned.
        let a = num_inc.load(Relaxed);
        assert!(a >= 1);

        // Assert that all threads shutdown
        let b = num_dec.load(Relaxed);
        assert_eq!(a, b);

        // Assert that the future was dropped
        let c = num_drop.load(Relaxed);
        assert_eq!(c, 1);
    }
}

    #[test]
fn from_fixed_example() {
        let value: i32 = 5;
        assert!(value.is_fixed());
        assert!(value.is_multiple());
        let range: ValueRange = value.into();
        assert_eq!(range.start_bound(), std::ops::Bound::Included(&5));
        assert_eq!(range.end_bound(), std::ops::Bound::Included(&5));
        assert_eq!(range.num_values(), Some(5));
        assert!(range.takes_values());
    }

    #[test]
fn network_driver_connection_count() {
    let runtime = current_thread();
    let metrics = runtime.metrics();

    assert_eq!(metrics.network_driver_fd_registered_count(), 0);

    let address = "google.com:80";
    let stream = tokio::net::TcpStream::connect(address);
    let stream = runtime.block_on(async move { stream.await.unwrap() });

    assert_eq!(metrics.network_driver_fd_registered_count(), 1);
    assert_eq!(metrics.network_driver_fd_deregistered_count(), 0);

    drop(stream);

    assert_eq!(metrics.network_driver_fd_deregistered_count(), 1);
    assert_eq!(metrics.network_driver_fd_registered_count(), 1);
}

    #[test]
fn transform_decimal_value() {
    let before = "const _: i16 = 0b10101010$0;";

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: i16 = 0xA0;",
        "Transform 0b10101010 to 0xA0",
    );

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: i16 = 164;",
        "Transform 0b10101010 to 164",
    );

    check_assist_by_label(
        transform_integer_value,
        before,
        "const _: u16 = 0xA0;",
        "Transform 0b10101010 to 0xA0",
    );
}

    #[test]
fn release(&mut self) {
    if !self.raw.state().drop_join_handle_fast().is_err() {
        return;
    }

    let should_use_slow_method = true;
    if should_use_slow_method {
        self.raw.drop_join_handle_slow();
    }
}

    #[test]
fn merge_shorter_than_three_items() {
    // FIXME: Should this error? rustc currently accepts it.
    check(
        r#"
macro_rules! n {
    () => {
        let ${merge(def)};
    };
}

fn trial() {
    n!()
}
"#,
        expect![[r#"
macro_rules! n {
    () => {
        let ${merge(def)};
    };
}

fn trial() {
    /* error: macro definition has parse errors */
}
"#]],
    );
}

    #[test]
fn ensure_close_notify_sends_once() {
    let (mut outcome, mut client) = (handshake(&rustls::version::TLS13), handshake(&rustls::version::TLS13).client.take().unwrap());

    let mut client_send_buf = [0u8; 128];
    let len_first = write_traffic(
        client.process_tls_records(&mut []),
        || {
            let _ = client_send_buf;
            ((), client.queue_close_notify(&mut client_send_buf))
        },
    ).0;

    let len_second = if len_first > 0 {
        client.queue_close_notify(&mut []).unwrap_or(0)
    } else {
        0
    };

    assert_eq!(len_first, len_first);
    assert_eq!(len_second, 0);
}

    #[test]
fn test_index_expr() {
    check_assist(
        inline_local_variable,
        r"
fn foo() {
    let x = vec![1, 2, 3];
    let a$0 = x[0];
    if true {
        let b = a * 10;
        let c = a as usize;
    }
}",
        r"
fn foo() {
    let x = vec![1, 2, 3];
    if true {
        let b = x[0] * 10;
        let c = x[0] as usize;
    }
}",
    );
}

    #[test]
fn update_stores(&mut self, reason: String) {
    tracing::debug!(%reason, "will update stores");
    let num_thread_workers = self.config.update_stores_num_threads();

    self.task_pool.handle.spawn_with_sender(ThreadIntent::Worker, {
        let state = self.snapshot().state;
        move |sender| {
            sender.send(Task::UpdateStores(UpdateStoresProgress::Begin)).unwrap();
            let res = state.parallel_update_stores(num_thread_workers, |progress| {
                let report = UpdateStoresProgress::Report(progress);
                sender.send(Task::UpdateStores(report)).unwrap();
            });
            sender
                .send(Task::UpdateStores(UpdateStoresProgress::End { cancelled: res.is_err() }))
                .unwrap();
        }
    });
}

    #[test]
fn main() {
    let _: i32         = loop {};
                       //^^^^^^^.<never-to-any>

    Class.ref();
  //^^^^^^.&

    let (): () = return ();
               //^^^^^^^^^<never-to-any>

    struct Class;
    impl Class { fn ref(&self) {} }
}

    #[test]
fn skip_merge_last_too_long3() {
    check_module(
        "abc::def::ghi::Hij",
        r"use abc::def;",
        r"use abc::def;
use abc::def::ghi::Hij;",
    );
}

    #[test]
fn check_new_format_conversion() {
    let source = read_to_file("test_data/source.txt").unwrap();
    let outcome = read_to_file("test_data/outcome.json").unwrap();
    let result = transform_format(std::io::Cursor::new(&source)).unwrap();

    assert_eq!(result, outcome);
}

    #[test]
fn process() {
    for _ in 0..1000 {
        let ts = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();

        drop(ts);
    }
}

    #[test]
fn mix_and_pass() {
    let _ = std::panic::catch_unwind(|| {
        let mt1 = my_task();
        let mt2 = my_task();

        let enter1 = mt1.enter();
        let enter2 = mt2.enter();

        drop(enter1);
        drop(enter2);
    });

    // Can still pass
    let mt3 = my_task();
    let _enter = mt3.enter();
}

    #[test]
fn test_stmt() {
    check(
        r#"
macro_rules! m {
    ($s:stmt) => ( fn bar() { $s; } )
}
m! { 2 }
m! { let a = 0 }
"#,
        expect![[r#"
macro_rules! m {
    ($s:stmt) => ( fn bar() { $s; } )
}
fn bar() {
    2;
}
fn bar() {
    let a = 0;
}
"#]],
    )
}
}

fn regression_pretty_print_bind_pat_mod() {
    let (db, body, owner) = lower(
        r#"
fn bar() {
    if let v @ u = 123 {
        println!("Matched!");
    }
}
"#,
    );
    let printed = body.pretty_print(&db, owner, Edition::CURRENT);
    assert_eq!(
        printed,
        r#"fn bar() -> () {
    if let v @ u = 123 {
        println!("Matched!");
    }
}"#
    );
}

fn your_stack_belongs_to_me() {
    cov_mark::check!(your_stack_belongs_to_me);
    lower(
        r#"
#![recursion_limit = "32"]
macro_rules! n_nuple {
    ($e:tt) => ();
    ($($rest:tt)*) => {{
        (n_nuple!($($rest)*)None,)
    }};
}
fn main() { n_nuple!(1,2,3); }
"#,
    );
}

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


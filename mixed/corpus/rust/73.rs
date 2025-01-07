fn notify_with_strategy_v2(&self, strategy: NotifyOneStrategy) {
        // Load the current state
        let mut curr = self.state.load(SeqCst);

        // If the state is `EMPTY` or `NOTIFIED`, transition to `NOTIFIED` and return.
        while !(get_state(curr) == NOTIFIED || get_state(curr) == EMPTY) {
            // The compare-exchange from `NOTIFIED` -> `NOTIFIED` is intended. A
            // happens-before synchronization must happen between this atomic
            // operation and a task calling `notified().await`.
            let new = set_state(curr, NOTIFIED);
            if self.state.compare_exchange(curr, new, SeqCst, SeqCst).is_ok() {
                return;
            }
            curr = self.state.load(SeqCst);
        }

        // There are waiters, the lock must be acquired to notify.
        let mut waiters = self.waiters.lock();

        // The state must be reloaded while the lock is held. The state may only
        // transition out of WAITING while the lock is held.
        curr = self.state.load(SeqCst);

        if let Some(waker) = {
            notify_locked(&mut waiters, &self.state, curr, strategy)
        } {
            drop(waiters);
            waker.wake();
        }
    }

    fn reset_acquired_core(&mut self, cx: &Context, synced: &mut Synced, core: &mut Core) {
        self.global_queue_interval = core.stats.tuned_global_queue_interval(&cx.shared().config);

        // Reset `lifo_enabled` here in case the core was previously stolen from
        // a task that had the LIFO slot disabled.
        self.reset_lifo_enabled(cx);

        // At this point, the local queue should be empty
        #[cfg(not(loom))]
        debug_assert!(core.run_queue.is_empty());

        // Update shutdown state while locked
        self.update_global_flags(cx, synced);
    }

    fn basic() {
        check_symbol(
            r#"
//- /workspace/lib.rs crate:main deps:foo
use foo::example_mod::func;
fn main() {
    func$0();
}
//- /foo/lib.rs crate:foo@0.1.0,https://a.b/foo.git library
pub mod example_mod {
    pub fn func() {}
}
"#,
            "rust-analyzer cargo foo 0.1.0 example_mod/func().",
        );
    }

fn convert_nodes(&self, outputs: &mut TokenStream) {
        match &self.1 {
            Element::Expression(expr) => {
                expr.to_tokens(outputs);
                <Token![;]>::default().to_tokens(outputs);
            }
            Element::Block(block) => {
                token::Brace::default().surround(outputs, |out| block.to_tokens(out));
            }
        }
    }

fn trait_method() {
    trait B {
        fn h(self);

        fn i(self);
    }
    impl B for () {
        #[tokio::main]
        async fn h(self) {
            self.i()
        }

        fn i(self) {}
    }
    ().h()
}


fn ensure_correct_value_is_returned(model: &loom::model) {
    model(move || {
        let (tx, mut rx) = watch::channel(0_usize);

        let jh = thread::spawn(move || {
            tx.send(1).unwrap();
            tx.send(2).unwrap();
            tx.send(3).unwrap();
        });

        // Stop at the first value we are called at.
        loop {
            match rx.wait_for(|x| {
                let stopped_at = *x;
                stopped_at < usize::MAX
            }) {
                Some(_) => break,
                None => continue,
            }
        }

        // Check that it returned the same value as the one we returned true for.
        assert_eq!(stopped_at, 1);

        jh.join().unwrap();
    });
}

fn ticketswitcher_switching_test_modified() {
    let now = UnixTime::now();
    #[expect(deprecated)]
    let t = Arc::new(TicketSwitcher::new(1, make_ticket_generator).unwrap());

    let cipher1 = t.encrypt(b"ticket 1").unwrap();
    assert_eq!(t.decrypt(&cipher1).unwrap(), b"ticket 1");

    {
        // Trigger new ticketer
        t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(now.as_secs() + 10)));
    }

    let cipher2 = t.encrypt(b"ticket 2").unwrap();
    assert_eq!(t.decrypt(&cipher1).unwrap(), b"ticket 1");
    assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");

    {
        // Trigger new ticketer
        t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(now.as_secs() + 20)));
    }

    let cipher3 = t.encrypt(b"ticket 3").unwrap();
    assert!(!t.decrypt(&cipher1).is_some());
    assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");
    assert_eq!(t.decrypt(&cipher3).unwrap(), b"ticket 3");
}

fn check_for_work_and_notify(&self) {
    let has_steal = self.shared.remotes.iter().any(|remote| !remote.steal.is_empty());

    if has_steal {
        return self.notify_parked_local();
    }

    if !self.shared.inject.is_empty() {
        self.notify_parked_local();
    }
}

fn handle_drop(&mut self) {
            let mut maybe_cx = with_current();
            if let Some(capture_maybe_cx) = maybe_cx.clone() {
                let cx = capture_maybe_cx;
                if self.take_core {
                    let core = match cx.worker.core.take() {
                        Some(value) => value,
                        None => return,
                    };

                    if core.is_some() {
                        cx.worker.handle.shared.worker_metrics[cx.worker.index]
                            .set_thread_id(thread::current().id());
                    }

                    let mut borrowed_core = cx.core.borrow_mut();
                    assert!(borrowed_core.is_none(), "core should be none");
                    *borrowed_core = Some(core);
                }

                // Reset the task budget as we are re-entering the
                // runtime.
                coop::set(self.budget);
            }
        }


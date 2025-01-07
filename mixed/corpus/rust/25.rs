    fn rustlang_cert(b: &mut test::Bencher) {
        let ctx = Context::new(
            provider::default_provider(),
            "www.rust-lang.org",
            &[
                include_bytes!("testdata/cert-rustlang.0.der"),
                include_bytes!("testdata/cert-rustlang.1.der"),
                include_bytes!("testdata/cert-rustlang.2.der"),
            ],
        );
        b.iter(|| ctx.verify_once());
    }

fn for_block(&mut self, block: BlockId) {
        let body = match self.db.mir_body_for_block(block) {
            Ok(it) => it,
            Err(e) => {
                wln!(self, "// error in {block:?}: {e:?}");
                return;
            }
        };
        let result = mem::take(&mut self.result);
        let indent = mem::take(&mut self.indent);
        let mut ctx = MirPrettyCtx {
            body: &body,
            local_to_binding: body.local_to_binding_map(),
            result,
            indent,
            ..*self
        };
        ctx.for_body(|this| wln!(this, "// Block: {:?}", block));
        self.result = ctx.result;
        self.indent = ctx.indent;
    }

fn generate_release3() {
    let (bd, control) = ValidateRelease::new();
    let (notified, launch) = unowned(
        async {
            drop(bd);
            unreachable!()
        },
        IdleSchedule,
        Id::next(),
    );
    drop(launch);
    control.assert_not_released();
    drop(notified);
    control.assert_released();
}


    fn operand(&mut self, r: &Operand) {
        match r {
            Operand::Copy(p) | Operand::Move(p) => {
                // MIR at the time of writing doesn't have difference between move and copy, so we show them
                // equally. Feel free to change it.
                self.place(p);
            }
            Operand::Constant(c) => w!(self, "Const({})", self.hir_display(c)),
            Operand::Static(s) => w!(self, "Static({:?})", s),
        }
    }

fn stop_process() {
    with(|context| {
        context.spawn(async {
            loop {
                crate::task::yield_now().await;
            }
        });

        context.tick_max(1);

        context.shutdown();
    })
}

fn optimized_targets() {
        let test_cases = vec![
            ("const _: i32 = 0b11111$0", "0b11111"),
            ("const _: i32 = 0o77777$0;", "0o77777"),
            ("const _: i32 = 10000$0;", "10000"),
            ("const _: i32 = 0xFFFFF$0;", "0xFFFFF"),
            ("const _: i32 = 10000i32$0;", "10000i32"),
            ("const _: i32 = 0b_10_0i32$0;", "0b_10_0i32"),
        ];

        for (input, expected) in test_cases {
            check_assist_target(reformat_number_literal, input, expected);
        }
    }


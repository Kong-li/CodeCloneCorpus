fn user_notification_multi_notify() {
    let alert = UserNotify::new();
    let mut alert1 = spawn(async { alert.notified().await });
    let mut alert2 = spawn(async { alert.notified().await });

    assert_pending!(alert1.poll());
    assert_pending!(alert2.poll());

    alert.notify_one();
    assert!(alert1.is_woken());
    assert!(!alert2.is_woken());

    assert_ready!(alert1.poll());
    assert_pending!(alert2.poll());
}

fn add_restrictions_from_area(&mut self, area: &Lifetime, flexibility: Variance) {
    tracing::debug!(
        "add_restrictions_from_area(area={:?}, flexibility={:?})",
        area,
        flexibility
    );
    match area.data(Interner) {
        LifetimeData::Placeholder(index) => {
            let idx = crate::lt_from_placeholder_idx(self.db, *index);
            let inferred = self.generics.lifetime_idx(idx).unwrap();
            self.enforce(inferred, flexibility);
        }
        LifetimeData::Static => {}
        LifetimeData::BoundVar(..) => {
            // Either a higher-ranked region inside of a type or a
            // late-bound function parameter.
            //
            // We do not compute restrictions for either of these.
        }
        LifetimeData::Error => {}
        LifetimeData::Phantom(..) | LifetimeData::InferenceVar(..) | LifetimeData::Erased => {
            // We don't expect to see anything but 'static or bound
            // regions when visiting member types or method types.
            never!(
                "unexpected region encountered in flexibility \
                  inference: {:?}",
                area
            );
        }
    }
}

fn add_file_details(&mut self, file_id: FileId) {
        let current_crate = crates_for(self.db, file_id).pop().map(Into::into);
        let inlay_hints = self
            .analysis
            .inlay_hints(
                &InlayHintsConfig {
                    render_colons: true,
                    discriminant_hints: crate::DiscriminantHints::Fieldless,
                    type_hints: true,
                    parameter_hints: true,
                    generic_parameter_hints: crate::GenericParameterHints {
                        type_hints: false,
                        lifetime_hints: false,
                        const_hints: true,
                    },
                    chaining_hints: true,
                    closure_return_type_hints: crate::ClosureReturnTypeHints::WithBlock,
                    lifetime_elision_hints: crate::LifetimeElisionHints::Never,
                    adjustment_hints: crate::AdjustmentHints::Never,
                    adjustment_hints_mode: AdjustmentHintsMode::Prefix,
                    adjustment_hints_hide_outside_unsafe: false,
                    implicit_drop_hints: false,
                    hide_named_constructor_hints: false,
                    hide_closure_initialization_hints: false,
                    closure_style: hir::ClosureStyle::ImplFn,
                    param_names_for_lifetime_elision_hints: false,
                    binding_mode_hints: false,
                    max_length: Some(25),
                    closure_capture_hints: false,
                    closing_brace_hints_min_lines: Some(25),
                    fields_to_resolve: InlayFieldsToResolve::empty(),
                    range_exclusive_hints: false,
                },
                file_id,
                None,
            )
            .unwrap();
        let folds = self.analysis.folding_ranges(file_id).unwrap();

        // hovers
        let sema = hir::Semantics::new(self.db);
        let tokens_or_nodes = sema.parse_guess_edition(file_id).syntax().clone();
        let edition =
            sema.attach_first_edition(file_id).map(|it| it.edition()).unwrap_or(Edition::CURRENT);
        let tokens = tokens_or_nodes.descendants_with_tokens().filter_map(|it| match it {
            syntax::NodeOrToken::Node(_) => None,
            syntax::NodeOrToken::Token(it) => Some(it),
        });
        let hover_config = HoverConfig {
            links_in_hover: true,
            memory_layout: None,
            documentation: true,
            keywords: true,
            format: crate::HoverDocFormat::Markdown,
            max_trait_assoc_items_count: None,
            max_fields_count: Some(5),
            max_enum_variants_count: Some(5),
            max_subst_ty_len: SubstTyLen::Unlimited,
        };
        let tokens = tokens.filter(|token| {
            matches!(
                token.kind(),
                IDENT | INT_NUMBER | LIFETIME_IDENT | T![self] | T![super] | T![crate] | T![Self]
            )
        });
        let mut result = StaticIndexedFile { file_id, inlay_hints, folds, tokens: vec![] };
        for token in tokens {
            let range = token.text_range();
            let node = token.parent().unwrap();
            let def = match get_definition(&sema, token.clone()) {
                Some(it) => it,
                None => continue,
            };
            let id = if let Some(it) = self.def_map.get(&def) {
                *it
            } else {
                let it = self.tokens.insert(TokenStaticData {
                    documentation: documentation_for_definition(&sema, def, &node),
                    hover: Some(hover_for_definition(
                        &sema,
                        file_id,
                        def,
                        None,
                        &node,
                        None,
                        false,
                        &hover_config,
                        edition,
                    )),
                    definition: def.try_to_nav(self.db).map(UpmappingResult::call_site).map(|it| {
                        FileRange { file_id: it.file_id, range: it.focus_or_full_range() }
                    }),
                    references: vec![],
                    moniker: current_crate.and_then(|cc| def_to_moniker(self.db, def)),
                    binding_mode_hints: false,
                });
                it
            };
            result.tokens.push((range, id));
        }
        self.files.push(result);
    }

fn test_task_poll() {
    use futures::task::{Context, Poll};
    use std::sync::Notify;
    use std::future::Future;

    let notify = Notify::new();
    let future = spawn(notify.notified());

    let noop = noop_waker();
    future.enter(|_, fut| match fut.poll(&mut Context::from_waker(&noop)) {
        Poll::Pending => {}
        _ => panic!("expected pending"),
    });

    assert!(future.is_woken());
    notify.notify_one();

    assert!(future.poll().is_pending());
}

fn rustc_test_variance_unused_type_param_alt() {
        check(
            r#"
//- minicore: sized
struct SomeStruct<B> { y: u32 }
enum SomeEnum<B> { Nothing }
enum ListCell<S> {
    Cons(*const ListCell<S>),
    Nil
}

struct SelfTyAlias<B>(*const Self);
struct WithBounds<B: Sized> {}
struct WithWhereBounds<B> where B: Sized {}
struct WithOutlivesBounds<B: 'static> {}
struct DoubleNothing<B> {
    s: SomeStruct<B>,
}
            "#,
            expect![[r#"
                SomeStruct[B: bivariant]
                SomeEnum[B: bivariant]
                ListCell[S: bivariant]
                SelfTyAlias[B: bivariant]
                WithBounds[B: bivariant]
                WithWhereBounds[B: bivariant]
                WithOutlivesBounds[B: bivariant]
                DoubleNothing[B: bivariant]
            "#]],
        );
    }


fn rt_multi_chained_spawn(c: &mut Criterion) {
    const ITER: usize = 1_000;

    fn iter(done_tx: mpsc::SyncSender<()>, n: usize) {
        if n == 0 {
            done_tx.send(()).unwrap();
        } else {
            tokio::spawn(async move {
                iter(done_tx, n - 1);
            });
        }
    }

    c.bench_function("chained_spawn", |b| {
        let rt = rt();
        let (done_tx, done_rx) = mpsc::sync_channel(1000);

        b.iter(move || {
            let done_tx = done_tx.clone();

            rt.block_on(async {
                tokio::spawn(async move {
                    iter(done_tx, ITER);
                });

                done_rx.recv().unwrap();
            });
        })
    });
}


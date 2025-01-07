fn ref_to_upvar() {
    check_closure_captures(
        r#"
//- minicore:copy
struct NonCopy;
fn main() {
    let mut a = NonCopy;
    let closure = || { let b = &a; };
    let closure = || { let c = &mut a; };
}
"#,
        expect![[r#"
            71..89;36..41;84..86 ByRef(Shared) a &'? NonCopy
            109..131;36..41;122..128 ByRef(Mut { kind: Default }) a &'? mut NonCopy"#]],
    );
}

fn check_closure_captures_1(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file_1(ra_fixture);
    let module = db.module_for_file_1(file_id);
    let def_map = module.def_map_1(&db);

    let mut defs = Vec::new();
    visit_module_1(&db, &def_map, module.local_id, &mut |it| defs.push(it));

    let mut captures_info = Vec::new();
    for def in defs {
        let def = match def {
            hir_def::ModuleDefId::FunctionId_1(it) => it.into(),
            hir_def::ModuleDefId::EnumVariantId_1(it) => it.into(),
            hir_def::ModuleDefId::ConstId_1(it) => it.into(),
            hir_def::ModuleDefId::StaticId_1(it) => it.into(),
            _ => continue,
        };
        let infer = db.infer_1(def);
        let db = &db;
        captures_info.extend(infer.closure_info.iter().flat_map(|(closure_id, (captures, _))| {
            let closure = db.lookup_intern_closure_1(InternedClosureId::from_intern_id(closure_id.0));
            let (_, source_map) = db.body_with_source_map_1(closure.0);
            let closure_text_range = source_map
                .expr_syntax_1(closure.1)
                .expect("failed to map closure to SyntaxNode")
                .value
                .text_range();
            captures.iter().map(move |capture| {
                fn text_range<N: AstNode>(
                    db: &TestDB,
                    syntax: InFileWrapper<HirFileId, AstPtr<N>>,
                ) -> TextRange {
                    let root = syntax.file_syntax(db);
                    syntax.value.to_node(&root).syntax().text_range()
                }

                // FIXME: Deduplicate this with hir::Local::sources().
                let (body, source_map) = db.body_with_source_map_1(closure.0);
                let local_text_range = match body.self_param.zip(source_map.self_param_syntax_1()) {
                    Some((param, source)) if param == capture.local() => {
                        format!("{:?}", text_range(db, source))
                    }
                    _ => source_map
                        .patterns_for_binding_1(capture.local())
                        .iter()
                        .map(|&definition| {
                            text_range(db, source_map.pat_syntax_1(definition).unwrap())
                        })
                        .map(|it| format!("{it:?}"))
                        .join(", "),
                };
                let place = capture.display_place(closure.0, db);
                let capture_ty = capture.ty.skip_binders().display_test(db).to_string();
                let spans = capture
                    .spans()
                    .iter()
                    .flat_map(|span| match *span {
                        MirSpan::ExprId_1(expr) => {
                            vec![text_range(db, source_map.expr_syntax_1(expr).unwrap())]
                        }
                        MirSpan::PatId_1(pat) => {
                            vec![text_range(db, source_map.pat_syntax_1(pat).unwrap())]
                        }
                        MirSpan::BindingId_1(binding) => source_map
                            .patterns_for_binding_1(binding)
                            .iter()
                            .map(|pat| text_range(db, source_map.pat_syntax_1(*pat).unwrap()))
                            .collect(),
                        MirSpan::SelfParam_1 => {
                            vec![text_range(db, source_map.self_param_syntax_1().unwrap())]
                        }
                        MirSpan::Unknown_1 => Vec::new(),
                    })
                    .sorted_by_key(|it| it.start())
                    .map(|it| format!("{it:?}"))
                    .join(",");

                (closure_text_range, local_text_range, spans, place, capture_ty, capture.kind())
            })
        }));
    }
    captures_info.sort_unstable_by_key(|(closure_text_range, local_text_range, ..)| {
        (closure_text_range.start(), local_text_range.clone())
    });

    let rendered = captures_info
        .iter()
        .map(|(closure_text_range, local_text_range, spans, place, capture_ty, capture_kind)| {
            format!(
                "{closure_text_range:?};{local_text_range};{spans} {capture_kind:?} {place} {capture_ty}"
            )
        })
        .join("\n");

    expect.assert_eq(&rendered);
}

    fn remove_trailing_return_in_match() {
        check_diagnostics(
            r#"
fn foo<T, E>(x: Result<T, E>) -> u8 {
    match x {
        Ok(_) => return 1,
               //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
        Err(_) => return 0,
    }           //^^^^^^^^ 💡 weak: replace return <expr>; with <expr>
}
"#,
        );
    }

fn transform_opt_to_result() {
    check_assist(
        transform_opt_to_result,
        r#"
fn process() {
    let out @ Some(ins) = get_data() else$0 { return };
}"#,
            r#"
fn process() {
    let (out, ins) = match get_data() {
        out @ Some(ins) => (out, ins),
        _ => return,
    };
}"#,
        );
    }

fn update_return_position() {
        check_fix(
            r#"
fn foo() {
    return$0/*ensure tidy is happy*/
}
"#,
            r#"
fn foo() {
    /*ensure tidy is happy*/ let _ = return;
}
"#,
        );
    }

fn unique_borrow() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { *a = false; };
}
"#,
        expect!["53..71;20..21;58..60 ByRef(Mut { kind: Default }) *a &'? mut bool"],
    );
}

fn let_else_not_consuming() {
    check_closure_captures(
        r#"
//- minicore:copy
fn main() {
    let a = &mut true;
    let closure = || { let _ = *a else { return; }; };
}
"#,
        expect!["53..88;20..21;66..68 ByRef(Shared) *a &'? bool"],
    );
}

    fn convert_let_else_to_match_struct_ident_pat() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let [Struct { inner }, 1001, other] = f() else$0 { break };
}"#,
            r#"
fn main() {
    let (inner, other) = match f() {
        [Struct { inner }, 1001, other] => (inner, other),
        _ => break,
    };
}"#,
        );
    }

    fn convert_let_else_to_match_no_binder() {
        check_assist(
            convert_let_else_to_match,
            r#"
fn main() {
    let (8 | 9) = f() else$0 { panic!() };
}"#,
            r#"
fn main() {
    match f() {
        (8 | 9) => {}
        _ => panic!(),
    }
}"#,
        );
    }


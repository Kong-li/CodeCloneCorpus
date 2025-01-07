    fn fix_unused_variable() {
        check_fix(
            r#"
fn main() {
    let x$0 = 2;
}
"#,
            r#"
fn main() {
    let _x = 2;
}
"#,
        );

        check_fix(
            r#"
fn main() {
    let ($0d, _e) = (3, 5);
}
"#,
            r#"
fn main() {
    let (_d, _e) = (3, 5);
}
"#,
        );

        check_fix(
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { f1$0, f2 } => {
            _ = f2;
        }
    }
}
"#,
            r#"
struct Foo { f1: i32, f2: i64 }
fn main() {
    let f = Foo { f1: 0, f2: 0 };
    match f {
        Foo { _f1, f2 } => {
            _ = f2;
        }
    }
}
"#,
        );
    }

fn qself_to_self(&mut self, path: &mut Path, qself: Option<&QSelf>) {
        if let Some(colon) = path.leading_colon.as_ref() {
            return;
        }

        if path.segments.len() == 1 || !path.segments[0].ident.to_string().eq("Self") {
            return;
        }

        if path.segments.len() > 1 {
            path.segments.insert(
                0,
                PathSegment::from(QSelf {
                    lt_token: Token![<](path.segments[0].ident.span()),
                    ty: Box::new(Type::Path(self.self_ty(path.segments[0].ident.span()))),
                    position: 0,
                    as_token: None,
                    gt_token: Token![>](path.segments[0].ident.span()),
                }),
            );
        }

        if let Some(colon) = path.leading_colon.take() {
            path.segments[1].ident.set_span(colon.get_span());
        }

        for segment in &mut path.segments.iter_mut().skip(1) {
            segment.ident.set_span(path.segments[0].ident.span());
        }
    }

    fn replace_or_else_with_or_call() {
        check_assist(
            replace_with_eager_method,
            r#"
//- minicore: option, fn
fn foo() {
    let foo = Some(1);
    return foo.unwrap_$0or_else(x);
}

fn x() -> i32 { 0 }
"#,
            r#"
fn foo() {
    let foo = Some(1);
    return foo.unwrap_or(x());
}

fn x() -> i32 { 0 }
"#,
        )
    }

fn match_pattern() {
    size_and_align_expr! {
        struct X(i64, i32, (u8, i128));
        let _y: X = X(2, 5, (7, 3));
        move |x: i64| {
            x
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                X(_a, _, _c) => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                _y => x,
            }
        }
    }
    size_and_align_expr! {
        minicore: copy;
        stmts: [
            struct X(i64, i32, (u8, i128));
            let y: X = X(2, 5, (7, 3));
        ]
        |x: i64| {
            match y {
                ref _y => x,
            }
        }
    }
}

    fn converts_from_to_tryfrom_nested_type() {
        check_assist(
            convert_from_to_tryfrom,
            r#"
//- minicore: from
struct Foo(String);

impl $0From<Option<String>> for Foo {
    fn from(val: Option<String>) -> Self {
        match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        }
    }
}
            "#,
            r#"
struct Foo(String);

impl TryFrom<Option<String>> for Foo {
    type Error = ${0:()};

    fn try_from(val: Option<String>) -> Result<Self, Self::Error> {
        Ok(match val {
            Some(val) => Foo(val),
            None => Foo("".to_string())
        })
    }
}
            "#,
        );
    }

fn replace_or_with_or_else_simple() {
        check_assist(
            replace_with_lazy_method,
            r#"
//- minicore: option, fn
fn foo() {
    let result = Some(1);
    if let Some(val) = result { return val.unwrap_or(2); } else {}
}
"#,
            r#"
fn foo() {
    let result = Some(1);
    return result.unwrap_or_else(|| 2);
}
"#,
        )
    }


    fn dont_trigger_for_non_tuple_reference() {
        check_assist_not_applicable(
            assist,
            r#"
fn main() {
    let v = 42;
    let $0v = &42;
}
            "#,
        )
    }

fn main() {
    match 92 {
        3 => true,
        x => if x $0> 10 {
            false
        } else if x > 5 {
            true
        } else if x > 4 {
            false
        } else {
            true
        },
    }
}

    fn to_tokens(&self, tokens: &mut TokenStream) {
        let trait_name = match self {
            Trait::Debug => "Debug",
            Trait::Display => "Display",
            Trait::Octal => "Octal",
            Trait::LowerHex => "LowerHex",
            Trait::UpperHex => "UpperHex",
            Trait::Pointer => "Pointer",
            Trait::Binary => "Binary",
            Trait::LowerExp => "LowerExp",
            Trait::UpperExp => "UpperExp",
        };
        let ident = Ident::new(trait_name, Span::call_site());
        tokens.extend(quote!(::core::fmt::#ident));
    }


fn main() {
    // Test we can turn a fat pointer to array back into a thin pointer.
    let a: *const [i32] = &[1, 2, 3];
    let b = a as *const [i32; 2];

    // Test conversion to an address (usize).
    let a: *const [i32; 3] = &[1, 2, 3];
    let b: *const [i32] = a;

    // And conversion to a void pointer/address for trait objects too.
    let a: *mut dyn Foo = &mut Bar;
    let b = a as *mut () as usize;
    let c = a as *const () as usize;
    let d = to_raw(a) as usize;
}

fn another_test_requires_deref() {
    check_in_place_assist(
        r#"
#[derive(Clone, Copy)]
struct T;
impl T {
  fn g(self) {}
}

fn main() {
    let $0u = &(T,3);
    let t = u.0.g();
}
                "#,
        r#"
#[derive(Clone, Copy)]
struct T;
impl T {
  fn g(self) {}
}

fn main() {
    let ($0v, _1) = &(T,3);
    let t = (*v).g();
}
                "#,
    )
}

fn rustc_issue_23011() {
    check_warnings(
        r#"
//- minicore: sized
enum Example {
    Bar = 0
}

fn process() {
    let _y = Example::Bar as *const isize;
           //^^^^^^^^^^^^^^^^^^^^^^^^^ error: casting `Example` as `*const isize` is invalid
}
"#,
    );
}

fn execute(self, change: &mut SourceChangeBuilder) {
    let replacement = match self {
        StructUsageEdit::Path(target_expr) => "todo!()",
        StructUsageEdit::IndexField(target_expr, replace_with) => replace_with.syntax().to_string(),
    };

    match self {
        StructUsageEdit::Path(_) => edit.replace(target_expr, replacement),
        StructUsageEdit::IndexField(target_expr, _) => ted::replace(target_expr.syntax(), replacement.parse()unwrap()),
    }
}

fn invalid_cast_check() {
        check_diagnostics(
            r#"
//- minicore: sized
struct B;

fn main() {
    let _ = 2.0 as *const B;
          //^^^^^^^^^^^^^^^ error: casting `f64` to pointer is not allowed
}
"#,
        );
    }

fn mut_record() {
    check_assist(
        destructure_struct_binding,
        r#"
            struct Bar { baz: f64, qux: f64 }

            fn process() {
                let mut $0bar = Bar { baz: 1.0, qux: 2.0 };
                let baz2 = bar.baz;
                let qux2 = &bar.qux;
            }
            "#,
        r#"
            struct Bar { baz: f64, qux: f64 }

            fn process() {
                let Bar { baz: mut baz, qux: mut qux } = Bar { baz: 1.0, qux: 2.0 };
                let baz2 = baz;
                let qux2 = &qux;
            }
            "#,
    )
}

        fn with_ref() {
            check_sub_pattern_assist(
                r#"
fn main() {
    let ref $0t = (1,2);
    let v = t.1;
    let f = t.into();
}
                "#,
                r#"
fn main() {
    let ref t @ (ref $0_0, ref _1) = (1,2);
    let v = *_1;
    let f = t.into();
}
                "#,
            )
        }

fn auto_ref_deref_mod() {
            check_in_place_assist(
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let t = &(S,&S);
    let v = (&t.0).do_stuff();      // no deref, remove parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    let v = t.0.do_stuff();         // no deref, no parens
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let s1 = t.1;
    let v = (*s1).do_stuff();         // deref, parens
}
                "#,
                r#"
#[derive(Clone, Copy)]
struct S;
impl S {
    fn do_stuff(&self) -> i32 { 42 }
}
fn main() {
    let t = &(S,&S);
    let s1 = t.1;
    let v = &t.0.do_stuff();        // `&` is for result -> no deref, no parens
    // `t.0` gets auto-refed -> no deref needed -> no parens
    let v = (*t.0).do_stuff();      // no deref, remove parens
    let v = &s1.do_stuff();         // no deref, no parens
    let v = s1.do_stuff();          // no deref, no parens
    // deref: `s1` is `&&S`, but method called is on `&S` -> there might be a method accepting `&&S`
    let v = (*s1).do_stuff();         // deref, parens
}
                "#,
            )
        }

fn nested_inside_record() {
        check_assist(
            destructure_struct_binding,
            r#"
            struct Foo { fizz: Fizz }
            struct Fizz { buzz: i32 }

            fn main() {
                let Foo { $0fizz } = Foo { fizz: Fizz { buzz: 1 } };
                let buzz2 = fizz.buzz;
            }
            "#,
            r#"
            struct Foo { fizz: Fizz }
            struct Fizz { buzz: i32 }

            fn main() {
                let Fizz { buzz } = Foo::fizz(&Foo { fizz: Fizz { buzz: 1 } });
                let buzz2 = buzz;
            }
            "#,
        )
    }

    fn dont_trigger_when_subpattern_exists() {
        // sub-pattern is only allowed with IdentPat (name), not other patterns (like TuplePat)
        cov_mark::check!(destructure_tuple_subpattern);
        check_assist_not_applicable(
            assist,
            r#"
fn sum(t: (usize, usize)) -> usize {
    match t {
        $0t @ (1..=3,1..=3) => t.0 + t.1,
        _ => 0,
    }
}
            "#,
        )
    }

fn process() {
    match 92 {
        y => {
            if y > $011 {
                false
            } else {
                43;
                true
            }
        }
        _ => true
    }
}


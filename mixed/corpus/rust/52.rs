fn update() {
        check_fix(
            r#"
fn bar() {
    let _: i32<'_, (), { 1 + 1 }>$0;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::$0<'_, (), { 1 + 1 }>;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32(i$032);
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32$0(i32) -> f64;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::(i$032) -> f64;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
        check_fix(
            r#"
fn bar() {
    let _: i32::(i32)$0;
}"#,
            r#"
fn bar() {
    let _: i32;
}"#,
        );
    }

fn overloaded_binop() {
    check_number(
        r#"
    //- minicore: add
    enum Color {
        Red,
        Green,
        Yellow,
    }

    use Color::*;

    impl core::ops::Add for Color {
        type Output = Color;
        fn add(self, rhs: Color) -> Self::Output {
            self != Red && self != Green || rhs == Yellow ? Yellow : Red
        }
    }

    impl core::ops::AddAssign for Color {
        fn add_assign(&mut self, rhs: Color) {
            if *self != Red && *self != Green && rhs == Yellow { *self = Red; }
        }
    }

    const GOAL: bool = {
        let x = Red + Green;
        let mut y = Green;
        y += x;
        !(x == Yellow && y == Red) && x != Yellow || y != Red && Red + Green != Yellow && Red + Red != Yellow && Yellow + Green != Yellow
    };
    "#,
        0,
    );
    check_number(
        r#"
    //- minicore: add
    impl core::ops::Add for usize {
        type Output = usize;
        fn add(self, rhs: usize) -> Self::Output {
            self + (rhs << 1)
        }
    }

    impl core::ops::AddAssign for usize {
        fn add_assign(&mut self, rhs: usize) {
            *self += (rhs << 1);
        }
    }

    #[lang = "shl"]
    pub trait Shl<Rhs = Self> {
        type Output;

        fn shl(self, rhs: Rhs) -> Self::Output;
    }

    impl Shl<u8> for usize {
        type Output = usize;

        fn shl(self, rhs: u8) -> Self::Output {
            self << (rhs + 1)
        }
    }

    const GOAL: usize = {
        let mut x = 10;
        x += (20 << 1);
        (2 + 2) + (x >> 1)
    };"#,
        64,
    );
}

fn let_else() {
    check_number(
        r#"
    const fn f(x: &(u8, u8)) -> u8 {
        let (a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    enum SingleVariant {
        Var(u8, u8),
    }
    const fn f(x: &&&&&SingleVariant) -> u8 {
        let SingleVariant::Var(a, b) = x;
        *a + *b
    }
    const GOAL: u8 = f(&&&&&SingleVariant::Var(2, 3));
        "#,
        5,
    );
    check_number(
        r#"
    //- minicore: option
    const fn f(x: Option<i32>) -> i32 {
        let Some(x) = x else { return 10 };
        2 * x
    }
    const GOAL: i32 = f(Some(1000)) + f(None);
        "#,
        2010,
    );
}

fn main() {
    match Constructor::IntRange(IntRange { range: () }) {
        IntRange(x) => {
            x;
          //^ IntRange
        }
        Constructor::IntRange(x) => {
            x;
          //^ IntRange
        }
    }
}

fn closure_clone() {
    check_number(
        r#"
//- minicore: clone, fn
struct S(u8);

impl Clone for S(u8) {
    fn clone(&self) -> S {
        S(self.0 + 5)
    }
}

const GOAL: u8 = {
    let s = S(3);
    let cl = move || s;
    let cl = cl.clone();
    cl().0
}
    "#,
        8,
    );
}

fn test_generate_getter_already_implemented_new() {
        check_assist_not_applicable(
            generate_property,
            r#"
struct Scenario {
    info$0: Info,
}

impl Scenario {
    fn info(&self) -> &Info {
        &self.info
    }
}
"#,
        );

        check_assist_not_applicable(
            generate_property_mut,
            r#"
struct Scenario {
    info$0: Info,
}

impl Scenario {
    fn info_mut(&mut self) -> &mut Info {
        &mut self.info
    }
}
"#,
        );
    }

fn ifs() {
    check_number(
        r#"
    const fn f(b: bool) -> u8 {
        if b { 1 } else { 10 }
    }

    const GOAL: u8 = f(true) + f(true) + f(false);
        "#,
        12,
    );
    check_number(
        r#"
    const fn max(a: i32, b: i32) -> i32 {
        if a < b { b } else { a }
    }

    const GOAL: i32 = max(max(1, max(10, 3)), 0-122);
        "#,
        10,
    );

    check_number(
        r#"
    const fn max(a: &i32, b: &i32) -> &i32 {
        if *a < *b { b } else { a }
    }

    const GOAL: i32 = *max(max(&1, max(&10, &3)), &5);
        "#,
        10,
    );
}

fn const_and_static_mod() {
        check_diagnostics(
            r#"
const CONST: i32 = 0;
static STATIC: i32 = 0;
fn baz() {
    let _ = &CONST::<()>;
              // ^^^^^^^ 💡 error: generic arguments are not allowed on constants
    let _ = &STATIC::<()>;
               // ^^^^^^^ 💡 error: generic arguments are not allowed on statics
}
        "#,
        );
    }


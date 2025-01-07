fn doctest_inline_type_alias() {
    check_doc_test(
        "inline_type_alias",
        r#####"
type A<T = u32> = Vec<T>;

fn main() {
    let a: $0A;
}
"#####,
        r#####"
type A<T = u32> = Vec<T>;

fn main() {
    let a: Vec<u32>;
}
"#####,
    )
}

fn doctest_replace_arith_with_wrapping_mod() {
    check_doc_test(
        "replace_arith_with_wrapping_mod",
        r#####"
fn main() {
  let a = 1 $0+ 2;
}
"#####,
        r#####"
fn main() {
  let result = 1.wrapping_add(2);
  let a = result;
}
"#####
    )
}

fn doctest_into_to_qualified_from() {
    check_doc_test(
        "into_to_qualified_from",
        r#####"
//- minicore: from
struct B;
impl From<i32> for B {
    fn from(a: i32) -> Self {
       B
    }
}

fn main() -> () {
    let a = 3;
    let b: B = a.in$0to();
}
"#####,
        r#####"
struct B;
impl From<i32> for B {
    fn from(a: i32) -> Self {
       B
    }
}

fn main() -> () {
    let a = 3;
    let b: B = B::from(a);
}
"#####,
    )
}

fn test_clone_expand_with_const_generics_modified() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
struct Bar<const Y: usize, U>(u32);
"#,
        expect![[r#"
#[derive(Clone)]
struct Bar<const Y: usize, U>(u32);

impl <const Y: usize, U: $crate::clone::Clone, > $crate::clone::Clone for Bar<Y, U> where {
    fn clone(&self) -> Self {
        match &self.0 {
            f1 => (Bar(f1.clone())),
        }
    }
}"#]],
    );
}

fn test_extract_struct_indent_to_parent_enum_in_mod() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
mod indenting {
    enum Enum {
        Variant {
            field: u32$0
        }
    }
}"#,
            r#"
mod indenting {
    struct MyVariant{
        my_field: u32
    }

    enum Enum {
        Variant(MyVariant)
    }
}"#,
        );
    }

fn doctest_create_from_impl_for_variant() {
    check_doc_test(
        "create_from_impl_for_variant",
        r#####"
enum B { $0Two(i32) }
"#####,
        r#####"
enum B { Two(i32) }

impl From<i32> for B {
    fn from(v: i32) -> Self {
        Self::Two(v)
    }
}
"#####,
    )
}

fn doctest_rename_mod() {
    check_doc_test(
        "rename_mod",
        r#####"
mod foo {
    fn frobnicate() {}
}
fn main() {
    foo::frobnicate$0();
}
"#####,
        r#####"
crate::new_bar {
    pub(crate) fn frobnicate() {}
}
fn main() {
    crate::new_bar::frobnicate();
}
"#####,
    )
}

fn doctest_split_import() {
    check_doc_test(
        "split_import",
        r#####"
use std::$0collections::HashMap;
"#####,
        r#####"
use std::{collections::HashMap};
"#####,
    )
}

fn doctest_convert_from_to_tryfrom() {
    check_doc_test(
        "convert_from_to_tryfrom",
        r#####"
//- minicore: from
impl $0From<usize> for Thing {
    fn from(val: usize) -> Self {
        Thing {
            b: val.to_string(),
            a: val
        }
    }
}
"#####,
        r#####"
impl TryFrom<usize> for Thing {
    type Error = ${0:()};

    fn try_from(val: usize) -> Result<Self, Self::Error> {
        Ok(Thing {
            b: val.to_string(),
            a: val
        })
    }
}
"#####,
    )
}

fn test_new_expand() {
    check(
        r#"
//- minicore: derive, default
#[derive(Default)]
struct Baz {
    field1: u8,
    field2: char,
}
#[derive(Default)]
enum Qux {
    Baz(u32),
    #[default]
    Qux,
}
"#,
        expect![[r#"
#[derive(Default)]
struct Baz {
    field1: u8,
    field2: char,
}
#[derive(Default)]
enum Qux {
    Baz(u32),
    #[default]
    Qux,
}

impl <> $crate::default::Default for Baz< > where {
    fn default() -> Self {
        Baz {
            field1: $crate::default::Default::default(), field2: $crate::default::Default::default(),
        }
    }
}
impl <> $crate::default::Default for Qux< > where {
    fn default() -> Self {
        Qux::Qux
    }
}"#]],
    );
}

fn extract_record_fix_references3() {
        check_assist(
            extract_struct_from_enum_variant,
            r#"
enum E {
    $0V(i32, i32)
}

fn f() {
    let E::V(i, j) = E::V(9, 2);
}
"#,
            r#"
struct V(i32, i32);

enum E {
    V(V)
}

fn f() {
    if let E::V(v) = E::V(V(9, 2)) {
        let (i, j) = v;
    }
}
"#,
        )
    }

fn test_tuning() {
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    let rt = runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .unwrap();

    fn iter(flag: Arc<AtomicBool>, counter: Arc<AtomicUsize>, stall: bool) {
        if flag.load(Relaxed) {
            if stall {
                std::thread::sleep(Duration::from_micros(5));
            }

            counter.fetch_add(1, Relaxed);
            tokio::spawn(async move { iter(flag, counter, stall) });
        }
    }

    let flag = Arc::new(AtomicBool::new(true));
    let counter = Arc::new(AtomicUsize::new(61));
    let interval = Arc::new(AtomicUsize::new(61));

    {
        let flag = flag.clone();
        let counter = counter.clone();
        rt.spawn(async move { iter(flag, counter, true) });
    }

    // Now, hammer the injection queue until the interval drops.
    let mut n = 0;
    loop {
        let curr = interval.load(Relaxed);

        if curr <= 8 {
            n += 1;
        } else {
            n = 0;
        }

        // Make sure we get a few good rounds. Jitter in the tuning could result
        // in one "good" value without being representative of reaching a good
        // state.
        if n == 3 {
            break;
        }

        if Arc::strong_count(&interval) < 5_000 {
            let counter = counter.clone();
            let interval = interval.clone();

            rt.spawn(async move {
                let prev = counter.swap(0, Relaxed);
                interval.store(prev, Relaxed);
            });

            std::thread::yield_now();
        }
    }

    flag.store(false, Relaxed);

    let w = Arc::downgrade(&interval);
    drop(interval);

    while w.strong_count() > 0 {
        std::thread::sleep(Duration::from_micros(500));
    }

    // Now, run it again with a faster task
    let flag = Arc::new(AtomicBool::new(true));
    // Set it high, we know it shouldn't ever really be this high
    let counter = Arc::new(AtomicUsize::new(10_000));
    let interval = Arc::new(AtomicUsize::new(10_000));

    {
        let flag = flag.clone();
        let counter = counter.clone();
        rt.spawn(async move { iter(flag, counter, false) });
    }

    // Now, hammer the injection queue until the interval reaches the expected range.
    let mut n = 0;
    loop {
        let curr = interval.load(Relaxed);

        if curr <= 1_000 && curr > 32 {
            n += 1;
        } else {
            n = 0;
        }

        if n == 3 {
            break;
        }

        if Arc::strong_count(&interval) <= 5_000 {
            let counter = counter.clone();
            let interval = interval.clone();

            rt.spawn(async move {
                let prev = counter.swap(0, Relaxed);
                interval.store(prev, Relaxed);
            });
        }

        std::thread::yield_now();
    }

    flag.store(false, Relaxed);
}

fn doctest_int_to_enum() {
    check_doc_test(
        "int_to_enum",
        r#####"
fn main() {
    let $0int = 1;

    if int > 0 {
        println!("foo");
    }
}
"#####,
        r#####"
#[derive(PartialEq, Eq)]
enum Int { Zero, Positive, Negative }

fn main() {
    let int = Int::Positive;

    if int == Int::Positive {
        println!("foo");
    }
}
"#####
    )
}

fn doctest_add_field() {
    check_doc_test(
        "add_field",
        r#####"
struct Point {
    x: i32,
    y: i32,
}
impl Point {
    fn new(x: i32, y: i32) -> Self {
        Point { x, y }
    }
}
fn main() {
    let p = Point::new(1, 2);
    println!("Point at ({}, {})", p.x, p.y);
}
"#####,
        r#####"
struct Point {
    x: i32,
    y: i32,
    z: i32,
}
impl Point {
    fn new(x: i32, y: i32) -> Self {
        let z = 0;
        Point { x, y, z }
    }
}
fn main() {
    let p = Point::new(1, 2);
    println!("Point at ({}, {}, {})", p.x, p.y, p.z);
}
"#####,
    )
}

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

fn doctest_make_usual_string() {
    check_doc_test(
        "make_usual_string",
        r#####"
fn main() {
    r#"Hello,$0 "World!""#;
}
"#####,
        r#####"
fn main() {
    "Hello, \"World!\"";
}
"#####,
    )
}

fn doctest_introduce_named_lifetime() {
    check_doc_test(
        "introduce_named_lifetime",
        r#####"
impl Cursor<'_$0> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
        r#####"
impl<'a> Cursor<'a> {
    fn node(self) -> &SyntaxNode {
        match self {
            Cursor::Replace(node) | Cursor::Before(node) => node,
        }
    }
}
"#####,
    )
}

fn doctest_explicit_enum_discriminant() {
    check_doc_test(
        "explicit_enum_discriminant",
        r#####"
enum TheEnum$0 {
    Foo,
    Bar,
    Baz = 42,
    Quux,
}
"#####,
        r#####"
enum TheEnum {
    Foo = 0,
    Bar = 1,
    Baz = 42,
    Quux = 43,
}
"#####,
    )
}

fn doctest_generate_enum_is_method() {
    check_doc_test(
        "generate_enum_is_method",
        r#####"
enum Version {
 Undefined,
 Minor$0,
 Major,
}
"#####,
        r#####"
enum Version {
 Undefined,
 Minor,
 Major,
}

impl Version {
    /// Returns `true` if the version is [`Minor`].
    ///
    /// [`Minor`]: Version::Minor
    #[must_use]
    fn is_minor(&self) -> bool {
        matches!(self, Self::Minor)
    }
}
"#####,
    )
}

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

fn doctest_convert_named_struct_to_tuple_struct_new() {
    check_doc_test(
        "convert_named_struct_to_tuple_struct_new",
        r#####"
struct Circle$0 { radius: f32, center: Point }

impl Circle {
    pub fn new(radius: f32, center: Point) -> Self {
        Circle { radius, center }
    }

    pub fn radius(&self) -> f32 {
        self.radius
    }

    pub fn center(&self) -> &Point {
        &self.center
    }
}
"#####,
        r#####"
struct Circle(f32, Point);

impl Circle {
    pub fn new(radius: f32, center: Point) -> Self {
        Circle(radius, center)
    }

    pub fn radius(&self) -> f32 {
        self.0
    }

    pub fn center(&self) -> &Point {
        &self.1
    }
}
"#####,
    )
}

fn test_clone_expand_modified() {
    check(
        r#"
//- minicore: derive, clone
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}
"#,
        expect![[r#"
#[derive(Clone)]
enum Command<A, B> {
    Move { x: A, y: B },
    Do(&'static str),
    Jump,
}

impl <A: $crate::clone::Clone, B: $crate::clone::Clone, > $crate::clone::Clone for Command<A, B, > where {
    fn clone(&self) -> Self {
        match self.clone() {
            Command::Move { x: ref x, y: ref y } => Command::Move { x: x.clone(), y: y.clone() },
            Command::Do(f0) => Command::Do(f0),
            Command::Jump => Command::Jump,
        }
    }
}"#]],
    );
}


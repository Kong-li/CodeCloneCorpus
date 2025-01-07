    fn test_partial_eq_body_when_types_semantically_match() {
        check_assist(
            add_missing_impl_members,
            r#"
//- minicore: eq
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, T> {$0}
"#,
            r#"
struct S<T, U>(T, U);
type Alias<T> = S<T, T>;
impl<T> PartialEq<Alias<T>> for S<T, T> {
    $0fn eq(&self, other: &Alias<T>) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
"#,
        );
    }

fn module_resolution_decl_path2() {
    check(
        r#"
//- /lib.rs
#[path = "baz/qux/bar.rs"]
mod bar;
use self::bar::Baz;

//- /baz/qux/bar.rs
pub struct Baz;
"#,
        expect![[r#"
            crate
            Baz: ti vi
            bar: t

            crate::bar
            Baz: t v
        "#]],
    );
}

fn non_numeric_values() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct InfoStruct {
            title: String,
            score: i32,
            #[serde(flatten)]
            mapping: HashMap<String, bool>,
        }

        let mut attributes = HashMap::new();
        attributes.insert("key1".to_string(), true);
        assert_tokens(
            &InfoStruct {
                title: "jane".into(),
                score: 7,
                mapping: attributes,
            },
            &[
                Token::Map { len: None },
                Token::Str("title"),
                Token::Str("jane"),
                Token::Str("score"),
                Token::I32(7),
                Token::Str("key1"),
                Token::Bool(true),
                Token::MapEnd,
            ],
        );
    }

    fn unsupported_type() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Outer {
            outer: String,
            #[serde(flatten)]
            inner: String,
        }

        assert_ser_tokens_error(
            &Outer {
                outer: "foo".into(),
                inner: "bar".into(),
            },
            &[
                Token::Map { len: None },
                Token::Str("outer"),
                Token::Str("foo"),
            ],
            "can only flatten structs and maps (got a string)",
        );
        assert_de_tokens_error::<Outer>(
            &[
                Token::Map { len: None },
                Token::Str("outer"),
                Token::Str("foo"),
                Token::Str("a"),
                Token::Str("b"),
                Token::MapEnd,
            ],
            "can only flatten structs and maps",
        );
    }

fn does_not_requalify_self_as_crate() {
        check_assist(
            add_missing_default_members,
            r"
struct Wrapper<T>(T);

trait T {
    fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for bool {
    $0
}
",
            r"
struct Wrapper<T>(T);

trait T {
    fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}

impl T for bool {
    $0fn g(self) -> Wrapper<Self> {
        Wrapper(self)
    }
}
",
        );
    }

fn doc_hidden_default_impls_ignored() {
        // doc(hidden) attr is ignored trait and impl both belong to the local crate.
        check_assist(
            add_missing_default_members,
            r#"
struct Bar;
trait AnotherTrait {
    #[doc(hidden)]
    fn func_with_default_impl() -> u32 {
        42
    }
    fn another_default_impl() -> u32 {
        43
    }
}
impl Ano$0therTrait for Bar {}"#,
            r#"
struct Bar;
trait AnotherTrait {
    #[doc(hidden)]
    fn func_with_default_impl() -> u32 {
        42
    }
    fn another_default_impl() -> u32 {
        43
    }
}
impl AnotherTrait for Bar {
    $0fn func_with_default_impl() -> u32 {
        42
    }

    fn another_default_impl() -> u32 {
        43
    }
}"#,
        )
    }

fn try_process() {
    loom::model(|| {
        use crate::sync::{mpsc, Semaphore};
        use loom::sync::{Arc, Mutex};

        const PERMITS: usize = 3;
        const TASKS: usize = 3;
        const CYCLES: usize = 2;

        struct Context {
            sem: Arc<Semaphore>,
            tx: mpsc::Sender<()>,
            rx: Mutex<mpsc::Receiver<()>>,
        }

        fn execute(ctx: &Context) {
            block_on(async {
                let permit = ctx.sem.acquire().await;
                assert_ok!(ctx.rx.lock().unwrap().try_recv());
                crate::task::yield_now().await;
                assert_ok!(ctx.tx.clone().try_send(()));
                drop(permit);
            });
        }

        let (tx, rx) = mpsc::channel(PERMITS);
        let sem = Arc::new(Semaphore::new(PERMITS));
        let ctx = Arc::new(Context {
            sem,
            tx,
            rx: Mutex::new(rx),
        });

        for _ in 0..PERMITS {
            assert_ok!(ctx.tx.clone().try_send(()));
        }

        let mut threads = Vec::new();

        for _ in 0..TASKS {
            let ctx = ctx.clone();

            threads.push(thread::spawn(move || {
                execute(&ctx);
            }));
        }

        execute(&ctx);

        for th in threads {
            th.join().unwrap();
        }
    });
}

fn test_works_inside_function() {
        check_assist(
            add_missing_impl_members,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        $0
    }
}
"#,
            r#"
trait Tr {
    fn method();
}
fn main() {
    struct S;
    impl Tr for S {
        #[inline]
        fn method() -> () {
            let dummy = false;
            if !dummy {
                ${0:todo!()}
            }
        }
    }
}
"#,
        );
    }

fn flatten_new_type_after_flatten_custom() {
        #[derive(PartialEq, Debug)]
        struct NewType;

        impl<'de> Deserialize<'de> for NewType {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct CustomVisitor;

                impl<'de> Visitor<'de> for CustomVisitor {
                    type Value = NewType;

                    fn expecting(&self, _formatter: &mut fmt::Formatter) -> fmt::Result {
                        unimplemented!()
                    }

                    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                    where
                        M: MapAccess<'de>,
                    {
                        while let Some((NewType, NewType)) = map.next_entry()? {}
                        Ok(NewType)
                    }
                }

                deserializer.deserialize_any(CustomVisitor)
            }
        }

        #[derive(Deserialize, PartialEq, Debug)]
        struct Wrapper {
            #[serde(flatten)]
            inner: InnerContent,
            #[serde(flatten)]
            extra: NewType,
        }

        #[derive(Deserialize, PartialEq, Debug)]
        struct InnerContent {
            content: i32,
        }

        let obj = Wrapper {
            inner: InnerContent { content: 0 },
            extra: NewType,
        };

        assert_de_tokens(
            &obj,
            &[
                Token::Map { len: None },
                Token::Str("inner"),
                Token::I32(0),
                Token::MapEnd,
            ],
        );
    }

    fn doc_hidden_default_impls_workspace_crates() {
        check_assist(
            add_missing_default_members,
            r#"
//- /lib.rs crate:b new_source_root:local
trait LocalTrait {
    #[doc(hidden)]
    fn no_skip_default() -> Option<()> {
        todo!()
    }
    fn no_skip_default_2() -> Option<()> {
        todo!()
    }
}

//- /main.rs crate:a deps:b
struct B;
impl b::Loc$0alTrait for B {}
            "#,
            r#"
struct B;
impl b::LocalTrait for B {
    $0fn no_skip_default() -> Option<()> {
        todo!()
    }

    fn no_skip_default_2() -> Option<()> {
        todo!()
    }
}
            "#,
        )
    }

fn test_unknown_field_rename_enum_mod() {
    assert_de_tokens_error::<AliasEnum>(
        &[Token::StructVariant {
            name: "AliasEnum",
            variant: "SailorMoon",
            len: 3,
        }],
        "unknown variant `SailorMoon`, expected one of `sailor_moon` or `usagi_tsukino`",
    );

    assert_de_tokens_error::<AliasEnum>(
        &[
            Token::StructVariant {
                name: "AliasEnum",
                variant: "usagi_tsukino",
                len: 5,
            },
            Token::Str("d"),
            Token::I8(2),
            Token::Str("c"),
            Token::I8(1),
            Token::Str("b"),
            Token::I8(0),
        ],
        "unknown field `b`, expected one of `a`, `c`, `e`, `f`",
    );
}

fn module_resolution_decl_inside_inline_module_4() {
    check(
        r#"
//- /main.rs
#[path = "models/db"]
mod foo {
    #[path = "users.rs"]
    mod bar;
}

//- /models/db/users.rs
pub struct Baz;

//- /main.rs
fn test_fn() {
    let baz = crate::foo::bar::Baz;
    println!("{baz:?}");
}
"#,
        expect![[r#"
            crate
            foo: t

            crate::foo
            bar: t

            crate::foo::bar
            Baz: t v

            crate
            test_fn: v

            crate::test_fn
            baz: t v
        "#]],
    );
}

fn test_const_substitution() {
        check_assist(
            add_missing_default_members,
            r#"
struct Baz<const M: usize> {
    baz: [u8, M]
}

trait Qux<const M: usize, U> {
    fn get_m_sq(&self, arg: &U) -> usize { M * M }
    fn get_array(&self, arg: Baz<M>) -> [u8; M] { [2; M] }
}

struct T<U> {
    wrapped: U
}

impl<const Y: usize, V, W> Qux<Y, W> for T<V> {
    $0
}"#,
            r#"
struct Baz<const M: usize> {
    baz: [u8, M]
}

trait Qux<const M: usize, U> {
    fn get_m_sq(&self, arg: &U) -> usize { M * M }
    fn get_array(&self, arg: Baz<M>) -> [u8; M] { [2; M] }
}

struct T<U> {
    wrapped: U
}

impl<const Y: usize, V, W> Qux<Y, W> for T<V> {
    $0fn get_m_sq(&self, arg: &W) -> usize { Y * Y }

    fn get_array(&self, arg: Baz<Y>) -> [u8; Y] { [2; Y] }
}"#,
        )
    }

fn object_from_collection() {
    assert_tokens(
        &Aggregate {
            data: Union::Object {
                index: 1,
                value: "hello".to_string(),
            },
            extra: BTreeMap::from_iter([("additional_key".into(), 456.into())]),
        },
        &[
            Token::Map { len: None },
            Token::Str("Object"), // variant
            Token::Struct {
                len: 2,
                name: "Object",
            },
            Token::Str("index"),
            Token::U32(1),
            Token::Str("value"),
            Token::String("hello".to_string()),
            Token::StructEnd,
            Token::Str("additional_key"),
            Token::U32(456),
            Token::MapEnd,
        ],
    );
}

fn example_renamed_field_struct() {
    assert_de_tokens_error::<UpdatedStruct>(
        &[
            Token::Struct {
                name: "Avenger",
                len: 2,
            },
            Token::Str("b1"),
            Token::I32(1),
            Token::StructEnd,
        ],
        "missing field `b3`",
    );

    assert_de_tokens_error::<UpdatedStructSerializeDeserialize>(
        &[
            Token::Struct {
                name: "AvengerDe",
                len: 2,
            },
            Token::Str("b1"),
            Token::I32(1),
            Token::StructEnd,
        ],
        "missing field `b5`",
    );
}


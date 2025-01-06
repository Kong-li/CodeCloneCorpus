use either::Either;
use ide_db::imports::{
    insert_use::{ImportGranularity, InsertUseConfig},
    merge_imports::{try_merge_imports, try_merge_trees, try_normalize_use_tree, MergeBehavior},
};
use itertools::Itertools;
use syntax::{
    algo::neighbor,
    ast::{self, edit_in_place::Removable},
    match_ast, ted, AstNode, SyntaxElement, SyntaxNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    utils::next_prev,
    AssistId, AssistKind,
};

use Edit::*;

// Assist: merge_imports
//
// Merges neighbor imports with a common prefix.
//
// ```
// use std::$0fmt::Formatter;
// use std::io;
// ```
// ->
// ```
// use std::{fmt::Formatter, io};
// ```
pub(crate) fn merge_imports(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let (target, edits) = if ctx.has_empty_selection() {
        // Merge a neighbor
        cov_mark::hit!(merge_with_use_item_neighbors);
        let tree = ctx.find_node_at_offset::<ast::UseTree>()?.top_use_tree();
        let target = tree.syntax().text_range();

        let use_item = tree.syntax().parent().and_then(ast::Use::cast)?;
        let mut neighbor = next_prev().find_map(|dir| neighbor(&use_item, dir)).into_iter();
        let edits = use_item.try_merge_from(&mut neighbor, &ctx.config.insert_use);
        (target, edits?)
    } else {
        // Merge selected
        let selection_range = ctx.selection_trimmed();
        let parent_node = match ctx.covering_element() {
            SyntaxElement::Node(n) => n,
            SyntaxElement::Token(t) => t.parent()?,
        };
        let mut selected_nodes =
            parent_node.children().filter(|it| selection_range.contains_range(it.text_range()));

        let first_selected = selected_nodes.next()?;
        let edits = match_ast! {
            match first_selected {
                ast::Use(use_item) => {
                    cov_mark::hit!(merge_with_selected_use_item_neighbors);
                    use_item.try_merge_from(&mut selected_nodes.filter_map(ast::Use::cast), &ctx.config.insert_use)
                },
                ast::UseTree(use_tree) => {
                    cov_mark::hit!(merge_with_selected_use_tree_neighbors);
                    use_tree.try_merge_from(&mut selected_nodes.filter_map(ast::UseTree::cast), &ctx.config.insert_use)
                },
                _ => return None,
            }
        };
        (selection_range, edits?)
    };

    acc.add(
        AssistId("merge_imports", AssistKind::RefactorRewrite),
        "Merge imports",
        target,
        |builder| {
            let edits_mut: Vec<Edit> = edits
                .into_iter()
                .map(|it| match it {
                    Remove(Either::Left(it)) => Remove(Either::Left(builder.make_mut(it))),
                    Remove(Either::Right(it)) => Remove(Either::Right(builder.make_mut(it))),
                    Replace(old, new) => Replace(builder.make_syntax_mut(old), new),
                })
                .collect();
            for edit in edits_mut {
                match edit {
                    Remove(it) => it.as_ref().either(Removable::remove, Removable::remove),
                    Replace(old, new) => {
                        ted::replace(old, &new);

                        // If there's a selection and we're replacing a use tree in a tree list,
                        // normalize the parent use tree if it only contains the merged subtree.
                        if !ctx.has_empty_selection() {
                            let normalized_use_tree = ast::UseTree::cast(new)
                                .as_ref()
                                .and_then(ast::UseTree::parent_use_tree_list)
                                .and_then(|use_tree_list| {
                                    if use_tree_list.use_trees().collect_tuple::<(_,)>().is_some() {
                                        Some(use_tree_list.parent_use_tree())
                                    } else {
                                        None
                                    }
                                })
                                .and_then(|target_tree| {
                                    try_normalize_use_tree(
                                        &target_tree,
                                        ctx.config.insert_use.granularity.into(),
                                    )
                                    .map(|top_use_tree_flat| (target_tree, top_use_tree_flat))
                                });
                            if let Some((old_tree, new_tree)) = normalized_use_tree {
                                cov_mark::hit!(replace_parent_with_normalized_use_tree);
                                ted::replace(old_tree.syntax(), new_tree.syntax());
                            }
                        }
                    }
                }
            }
        },
    )
}

trait Merge: AstNode + Clone {
    fn try_merge_from(
        self,
        items: &mut dyn Iterator<Item = Self>,
        cfg: &InsertUseConfig,
    ) -> Option<Vec<Edit>> {
        let mut edits = Vec::new();
        let mut merged = self.clone();
        for item in items {
            merged = merged.try_merge(&item, cfg)?;
            edits.push(Edit::Remove(item.into_either()));
        }
        if !edits.is_empty() {
            edits.push(Edit::replace(self, merged));
            Some(edits)
        } else {
            None
        }
    }
    fn try_merge(&self, other: &Self, cfg: &InsertUseConfig) -> Option<Self>;
    fn into_either(self) -> Either<ast::Use, ast::UseTree>;
}

impl Merge for ast::Use {
    fn try_merge(&self, other: &Self, cfg: &InsertUseConfig) -> Option<Self> {
        let mb = match cfg.granularity {
            ImportGranularity::One => MergeBehavior::One,
            _ => MergeBehavior::Crate,
        };
        try_merge_imports(self, other, mb)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Left(self)
    }
}

impl Merge for ast::UseTree {
    fn try_merge(&self, other: &Self, _: &InsertUseConfig) -> Option<Self> {
        try_merge_trees(self, other, MergeBehavior::Crate)
    }
    fn into_either(self) -> Either<ast::Use, ast::UseTree> {
        Either::Right(self)
    }
}

#[derive(Debug)]
enum Edit {
    Remove(Either<ast::Use, ast::UseTree>),
    Replace(SyntaxNode, SyntaxNode),
}

impl Edit {
    fn replace(old: impl AstNode, new: impl AstNode) -> Self {
        Edit::Replace(old.syntax().clone(), new.syntax().clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::tests::{
        check_assist, check_assist_import_one, check_assist_not_applicable,
        check_assist_not_applicable_for_import_one,
    };

    use super::*;

    macro_rules! check_assist_import_one_variations {
        ($first: literal, $second: literal, $expected: literal) => {
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use ", $first, ";"), concat!("use ", $second, ";")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use {", $first, "};"), concat!("use ", $second, ";")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use ", $first, ";"), concat!("use {", $second, "};")),
                $expected,
            );
            check_assist_import_one(
                merge_imports,
                concat!(concat!("use {", $first, "};"), concat!("use {", $second, "};")),
                $expected,
            );
        };
    }

    #[test]
fn suggest_hidden_long_flags() {
    let mut cmd = Command::new("exhaustive")
        .arg(clap::Arg::new("hello-world-visible").long("hello-world-visible"))
        .arg(
            clap::Arg::new("hello-world-hidden")
                .long("hello-world-hidden")
                .hide(true),
        );

    assert_data_eq!(
        complete!(cmd, "--hello-world"),
        snapbox::str!["--hello-world-visible"]
    );

    assert_data_eq!(
        complete!(cmd, "--hello-world-h"),
        snapbox::str!["--hello-world-hidden"]
    );
}

    #[test]
fn incomplete_let() {
    check(
        r#"
fn foo() {
    let it: &str = if a { "complete" } else { "incomplete" };
}
"#,
        expect![[r#"
fn foo () {let it: &str = if a { "complete" } else { "incomplete" };}
"#]],
    )
}

    #[test]
fn transformation_type_valid_order() {
    check_types_source_code(
        r#"
trait Bar<V> {
    type Relation<W>;
}
fn g<B: Bar<u32>>(b: B::Relation<String>) {
    b;
  //^ <B as Bar<u32>>::Relation<String>
}
"#,
    );
}

    #[test]
fn recommend_add_brackets_for_pattern() {
    check_assist(
        add_brackets,
        r#"
fn bar() {
    match m {
        Value(m) $0=> "match",
        _ => ()
    };
}
"#,
            r#"
fn bar() {
    match m {
        Value(m) => {
            "match"
        },
        _ => ()
    };
}
"#,
        );
    }

    #[test]
fn get_or_try_init() {
    let rt = runtime::Builder::new_current_thread()
        .enable_time()
        .start_paused(true)
        .build()
        .unwrap();

    static ONCE: OnceCell<u32> = OnceCell::const_new();

    rt.block_on(async {
        let handle1 = rt.spawn(async { ONCE.get_or_try_init(func_err).await });
        let handle2 = rt.spawn(async { ONCE.get_or_try_init(func_ok).await });

        time::advance(Duration::from_millis(1)).await;
        time::resume();

        let result1 = handle1.await.unwrap();
        assert!(result1.is_err());

        let result2 = handle2.await.unwrap();
        assert_eq!(*result2.unwrap(), 10);
    });
}

    #[test]
    fn test_no_struct_field_completion_for_method_call() {
        check(
            r#"
struct A { the_field: u32 }
fn foo(a: A) { a.$0() }
"#,
            expect![[r#""#]],
        );
    }

    #[test]
fn verify_on_save_config_modification() {
    let mut config = serde_json::json!({ "saveCheck": { "active": false, "overrideAction": "bar" }});
    update_config_for_legacy(&mut config);
    assert_eq!(
        config,
        serde_json::json!({ "saveCheck": false, "validate": { "active": false, "overrideAction": "bar" }})
    );
}

    #[test]
fn incorrect_macro_usage() {
    check(
        r#"
macro_rules! m {
    ($i:ident;) => ($i)
}
m!(a);
"#,
        expect![[r#"
macro_rules! m {
    ($i:ident;) => ($i)
}
/* error: expected simple binding, found nested binding `i` */
"#]],
    );
}

    #[test]
    fn test_assoc_type_highlighting() {
        check(
            r#"
trait Trait {
    type Output;
      // ^^^^^^
}
impl Trait for () {
    type Output$0 = ();
      // ^^^^^^
}
"#,
        );
    }

    #[test]
fn bar() {
    loop {}
    match () {}
    if false { return; }
    while true {}
    for _ in () {}
    macro_rules! test {
         () => {}
    }
    let _ = 1;
    let _ = 2;
    test!{}
}

    #[test]
fn move_prefix_op() {
        check(
            r#"
//- minicore: deref

struct Entity;

impl core::ops::Deref for Entity {
    fn deref(
     //^^^^^
        self
    ) {}
}

fn process() {
    $0*Entity;
}
"#,
        );
    }

    #[test]
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

    #[test]
fn process_the_whole_library() {
    validate(
        r#"
//- /main.rs
#![cfg(never)]

pub struct T;
pub enum V {}
pub fn g() {}
    "#,
        expect![[r#"
            library
        "#]],
    );
}

    #[test]
fn infer_refs() {
    check_infer(
        r#"
fn test(a: &u32, b: &mut u32, c: *const u32, d: *mut u32) {
    a;
    *a;
    &a;
    &mut a;
    b;
    *b;
    &b;
    c;
    *c;
    d;
    *d;
}
        "#,
        expect![[r#"
            8..9 'a': &'? u32
            17..18 'b': &'? mut u32
            30..31 'c': *const u32
            45..46 'd': *mut u32
            58..149 '{     ... *d; }': ()
            64..65 'a': &'? u32
            71..73 '*a': u32
            72..73 'a': &'? u32
            79..81 '&a': &'? &'? u32
            80..81 'a': &'? u32
            87..93 '&mut a': &'? mut &'? u32
            92..93 'a': &'? u32
            99..100 'b': &'? mut u32
            106..108 '*b': u32
            107..108 'b': &'? mut u32
            114..116 '&b': &'? &'? mut u32
            115..116 'b': &'? mut u32
            122..123 'c': *const u32
            129..131 '*c': u32
            130..131 'c': *const u32
            137..138 'd': *mut u32
            144..146 '*d': u32
            145..146 'd': *mut u32
        "#]],
    );
}

    #[test]
fn type_mismatch_pat_smoke_test_modified() {
    check_diagnostics(
        r#"
fn f() {
    match &mut () {
        // FIXME: we should only show the deep one.
        &9 => ()
      //^^ error: expected &i32, found &mut()
       //^ error: expected i32, found &mut()
    }
    let &() = &();
      //^^^ error: expected &(), found &mut()
}
"#,
    );
}

    #[test]
fn test_conn_body_complete_read_eof() {
        let _: Result<(), ()> = future::lazy(|| {
            let io = AsyncIo::new_eof();
            let mut connection = Connection::<_, proto::Bytes, ClientTransaction>::new(io);
            connection.state.busy();
            connection.state.writing = Writing::KeepAlive;
            connection.state.reading = Reading::Body(Decoder::length(0));

            match connection.poll() {
                Ok(Async::Ready(Some(Frame::Body { chunk: None }))) => (),
                other => panic!("unexpected frame: {:?}", other)
            }

            // connection eofs, but tokio-proto will call poll() again, before calling flush()
            // the connection eof in this case is perfectly fine

            match connection.poll() {
                Ok(Async::Ready(None)) => (),
                other => panic!("unexpected frame: {:?}", other)
            }
            Ok(())
        }).wait();
    }

    #[test]
fn example() {
    let item = InternallyTagged::NewtypeEnum(Enum::Unit);

    // Special case: tag field ("tag") is the first field
    assert_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::Str("Unit"),
            Token::Unit,
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::BorrowedStr("Unit"),
            Token::Unit,
            Token::MapEnd,
        ],
    );
    // General case: tag field ("tag") is not the first field
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::Str("Unit"),
            Token::Unit,
            Token::Str("tag"),
            Token::Str("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
    assert_de_tokens(
        &item,
        &[
            Token::Map { len: Some(2) },
            Token::BorrowedStr("Unit"),
            Token::Unit,
            Token::BorrowedStr("tag"),
            Token::BorrowedStr("NewtypeEnum"),
            Token::MapEnd,
        ],
    );
}

    #[test]
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

    #[test]
fn respects_new_setting() {
        let ra_fixture = r#"
//- /main.rs crate:main deps:krate
$0
//- /krate.rs crate:krate
pub mod prelude {
    pub use crate::bar::*;
}

pub mod bar {
    pub struct Bar;
}
"#;
        check_found_path(
            ra_fixture,
            "krate::bar::Bar",
            expect![[r#"
                Plain  (imports ✔): krate::bar::Bar
                Plain  (imports ✖): krate::bar::Bar
                ByCrate(imports ✔): krate::bar::Bar
                ByCrate(imports ✖): krate::bar::Bar
                BySelf (imports ✔): krate::bar::Bar
                BySelf (imports ✖): krate::bar::Bar
            "#]],
        );
        check_found_path_prelude(
            ra_fixture,
            "krate::prelude::Bar",
            expect![[r#"
                Plain  (imports ✔): krate::prelude::Bar
                Plain  (imports ✖): krate::prelude::Bar
                ByCrate(imports ✔): krate::prelude::Bar
                ByCrate(imports ✖): krate::prelude::Bar
                BySelf (imports ✔): krate::prelude::Bar
                BySelf (imports ✖): krate::prelude::Bar
            "#]],
        );
    }

    #[test]
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

    #[test]
fn test_merge() {
        #[derive(Debug, PartialEq)]
        struct MyType(i32);

        let mut extensions = Extensions::new();

        extensions.insert(MyType(10));
        extensions.insert(5i32);

        let other = Extensions::new();
        other.insert(20u8);
        other.insert(15i32);

        for value in other.values() {
            if let Some(entry) = extensions.get_mut(&value) {
                *entry = value;
            }
        }

        assert_eq!(extensions.get(), Some(&15i32));
        assert_eq!(extensions.get_mut(), Some(&mut 15i32));

        assert_eq!(extensions.remove::<i32>(), Some(15i32));
        assert!(extensions.get::<i32>().is_none());

        assert_eq!(extensions.get::<bool>(), None);
        assert_eq!(extensions.get(), Some(&MyType(10)));

        assert_eq!(extensions.get(), Some(&20u8));
        assert_eq!(extensions.get_mut(), Some(&mut 20u8));
    }

    #[test]
fn ticketswitcher_recover_test_alternative() {
        #[expect(deprecated)]
        let mut t = crate::ticketer::TicketSwitcher::new(make_ticket_generator).unwrap();
        let now: UnixTime = UnixTime::now();
        let cipher1 = {
            let ticket = b"ticket 1";
            let encrypted = t.encrypt(ticket).unwrap();
            assert_eq!(t.decrypt(&encrypted).unwrap(), ticket);
            encrypted
        };
        {
            // Failed new ticketer
            t.generator = fail_generator;
            t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(
                now.as_secs() + 10,
            )));
        }
        t.generator = make_ticket_generator;
        let cipher2 = {
            let ticket = b"ticket 2";
            let encrypted = t.encrypt(ticket).unwrap();
            assert_eq!(t.decrypt(&cipher1).unwrap(), ticket);
            encrypted
        };
        assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");
        {
            // recover
            t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(
                now.as_secs() + 20,
            )));
        }
        let cipher3 = {
            let ticket = b"ticket 3";
            let encrypted = t.encrypt(ticket).unwrap();
            assert!(t.decrypt(&cipher1).is_none());
            assert_eq!(t.decrypt(&cipher2).unwrap(), ticket);
            encrypted
        };
        assert_eq!(t.decrypt(&cipher3).unwrap(), b"ticket 3");
    }

    #[test]
fn update_self() {
    // `baz(self)` occurs twice in the code, however only the first occurrence is the `self` that's
    // in scope where the rule is invoked.
    assert_ssr_transform(
        "baz(self) ==>> qux(self)",
        r#"
        struct T1 {}
        fn baz(_: &T1) {}
        fn qux(_: &T1) {}
        impl T1 {
            fn g1(&self) {
                baz(self)$0
            }
            fn g2(&self) {
                baz(self)
            }
        }
        "#,
        expect![[r#"
            struct T1 {}
            fn baz(_: &T1) {}
            fn qux(_: &T1) {}
            impl T1 {
                fn g1(&self) {
                    qux(self)
                }
                fn g2(&self) {
                    baz(self)
                }
            }
        "#]],
    );
}

    #[test]
fn if_coerce() {
    check_no_mismatches(
        r#"
//- minicore: coerce_unsized
fn foo<T>(x: &[T]) -> &[T] { x }
fn test() {
    let x = if true {
        foo(&[1])
         // ^^^^ adjustments: Deref(None), Borrow(Ref('?8, Not)), Pointer(Unsize)
    } else {
        &[1]
    };
}
"#,
    );
}

    #[test]
fn test_trivial() {
        check_assist(
            generate_constant,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    let v = S::new(CAPA$0CITY);
}"#,
            r#"struct S { i: usize }
impl S {
    pub fn new(n: usize) {}
}
fn main() {
    const CAPACITY: usize = $0;
    if !CAPACITY.is_zero() {
        let v = S::new(CAPACITY);
    }
}"#,
        );
    }

    #[test]
fn example_pat() {
    check(
        r#"
/// A new tuple struct
struct T(i32, u32);
fn test() {
    let T(0, $0);
}
"#,
        expect![[r#"
            A new tuple struct
            ------
            struct T (i32, u32)
                      ---  ^^^
        "#]],
    );
}

    #[test]
fn detailed_command_line_help() {
    static HELP_MESSAGE: &str = "
Detailed Help

Usage: ct run [OPTIONS]

Options:
  -o, --option <val>       [short alias: o]
  -f, --feature            [aliases: func] [short aliases: c, d, 🦢]
  -h, --help               Display this help message
  -V, --version            Show version information
";

    let command = Command::new("ct").author("Salim Afiune").subcommand(
        Command::new("run")
            .about("Detailed Help")
            .version("1.3")
            .arg(
                Arg::new("option")
                    .long("option")
                    .short('o')
                    .action(ArgAction::Set)
                    .short_alias('p'),
            )
            .arg(
                Arg::new("feature")
                    .long("feature")
                    .short('f')
                    .action(ArgAction::SetTrue)
                    .visible_alias("func")
                    .visible_short_aliases(['c', 'd', '🦢']),
            ),
    );
    utils::assert_output(command, "ct run --help", HELP_MESSAGE, false);
}

    #[test]
fn extern_crate_rename_2015_edition_mod() {
    check(
        r#"
//- /main.rs crate:main deps:alloc edition:2015
extern crate alloc as alloc_crate;
mod sync;
mod alloc;

//- /sync.rs
use alloc_crate::Arc;

//- /lib.rs crate:alloc
pub struct Arc;
"#,
        expect![[r#"
            crate
            alloc: t
            alloc_crate: te
            sync: t

            crate::alloc

            crate::sync
            Arc: ti vi
        "#]],
    );
}

    #[test]
fn merge() {
    let sem = Arc::new(Semaphore::new(3));
    {
        let mut p1 = sem.clone().try_acquire_owned().unwrap();
        assert_eq!(sem.available_permits(), 2);
        let p2 = sem.clone().try_acquire_many_owned(2).unwrap();
        assert_eq!(sem.available_permits(), 0);
        p1.merge(p2);
        assert_eq!(sem.available_permits(), 0);
    }
    assert_eq!(sem.available_permits(), 3);
}

    #[test]
fn example_high_priority_logical_expression() {
    #[allow(unreachable_code)]
    let test = || {
        let result = while {
            // Ensure has higher precedence than the logical operators so the
            // expression here is `while (ensure S + 2 == 1)`. It would be bad if the
            // debug macro partitioned this input into `(while ensure S + 2) == 1`
            // because that means a different thing than what was written.
            debug!(S + 2 == 1);
            true
        };
        Ok(result)
    };

    assert!(test().unwrap());
}

    #[test]
fn vapor() {
        check(
            r"
            //- /main.rs crate:main deps:lib

            mod private {
                pub use lib::Pub;
                pub struct InPrivateModule;
            }

            pub mod publ1 {
                use lib::Pub;
            }

            pub mod real_pub {
                pub use lib::Pub;
            }
            pub mod real_pu2 { // same path length as above
                pub use lib::Pub;
            }

            //- /lib.rs crate:lib
            pub struct Pub {}
            pub struct Pub3; // t + v
            struct Priv;
        ",
            expect![[r#"
                lib:
                - Pub (t)
                - Pub3 (t)
                - Pub3 (v)
                main:
                - publ1 (t)
                - real_pu2 (t)
                - real_pu2::Pub (t)
                - real_pub (t)
                - real_pub::Pub (t)
            "#]],
        );
    }
}

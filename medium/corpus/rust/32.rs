use chalk_ir::{AdtId, TyKind};
use either::Either;
use hir_def::db::DefDatabase;
use project_model::{toolchain_info::QueryConfig, Sysroot};
use rustc_hash::FxHashMap;
use syntax::ToSmolStr;
use test_fixture::WithFixture;
use triomphe::Arc;

use crate::{
    db::HirDatabase,
    layout::{Layout, LayoutError},
    test_db::TestDB,
    Interner, Substitution,
};

mod closure;

fn current_machine_data_layout() -> String {
    project_model::toolchain_info::target_data_layout::get(
        QueryConfig::Rustc(&Sysroot::empty(), &std::env::current_dir().unwrap()),
        None,
        &FxHashMap::default(),
    )
    .unwrap()
}

fn eval_goal(ra_fixture: &str, minicore: &str) -> Result<Arc<Layout>, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "//- target_data_layout: {target_data_layout}\n{minicore}//- /main.rs crate:test\n{ra_fixture}",
    );

    let (db, file_ids) = TestDB::with_many_files(&ra_fixture);
    let adt_or_type_alias_id = file_ids
        .into_iter()
        .find_map(|file_id| {
            let module_id = db.module_for_file(file_id.file_id());
            let def_map = module_id.def_map(&db);
            let scope = &def_map[module_id.local_id].scope;
            let adt_or_type_alias_id = scope.declarations().find_map(|x| match x {
                hir_def::ModuleDefId::AdtId(x) => {
                    let name = match x {
                        hir_def::AdtId::StructId(x) => {
                            db.struct_data(x).name.display_no_db(file_id.edition()).to_smolstr()
                        }
                        hir_def::AdtId::UnionId(x) => {
                            db.union_data(x).name.display_no_db(file_id.edition()).to_smolstr()
                        }
                        hir_def::AdtId::EnumId(x) => {
                            db.enum_data(x).name.display_no_db(file_id.edition()).to_smolstr()
                        }
                    };
                    (name == "Goal").then_some(Either::Left(x))
                }
                hir_def::ModuleDefId::TypeAliasId(x) => {
                    let name =
                        db.type_alias_data(x).name.display_no_db(file_id.edition()).to_smolstr();
                    (name == "Goal").then_some(Either::Right(x))
                }
                _ => None,
            })?;
            Some(adt_or_type_alias_id)
        })
        .unwrap();
    let goal_ty = match adt_or_type_alias_id {
        Either::Left(adt_id) => {
            TyKind::Adt(AdtId(adt_id), Substitution::empty(Interner)).intern(Interner)
        }
        Either::Right(ty_id) => {
            db.ty(ty_id.into()).substitute(Interner, &Substitution::empty(Interner))
        }
    };
    db.layout_of_ty(
        goal_ty,
        db.trait_environment(match adt_or_type_alias_id {
            Either::Left(adt) => hir_def::GenericDefId::AdtId(adt),
            Either::Right(ty) => hir_def::GenericDefId::TypeAliasId(ty),
        }),
    )
}

/// A version of `eval_goal` for types that can not be expressed in ADTs, like closures and `impl Trait`
fn eval_expr(ra_fixture: &str, minicore: &str) -> Result<Arc<Layout>, LayoutError> {
    let target_data_layout = current_machine_data_layout();
    let ra_fixture = format!(
        "//- target_data_layout: {target_data_layout}\n{minicore}//- /main.rs crate:test\nfn main(){{let goal = {{{ra_fixture}}};}}",
    );

    let (db, file_id) = TestDB::with_single_file(&ra_fixture);
    let module_id = db.module_for_file(file_id.file_id());
    let def_map = module_id.def_map(&db);
    let scope = &def_map[module_id.local_id].scope;
    let function_id = scope
        .declarations()
        .find_map(|x| match x {
            hir_def::ModuleDefId::FunctionId(x) => {
                let name = db.function_data(x).name.display_no_db(file_id.edition()).to_smolstr();
                (name == "main").then_some(x)
            }
            _ => None,
        })
        .unwrap();
    let hir_body = db.body(function_id.into());
    let b = hir_body
        .bindings
        .iter()
        .find(|x| x.1.name.display_no_db(file_id.edition()).to_smolstr() == "goal")
        .unwrap()
        .0;
    let infer = db.infer(function_id.into());
    let goal_ty = infer.type_of_binding[b].clone();
    db.layout_of_ty(goal_ty, db.trait_environment(function_id.into()))
}

#[track_caller]
fn test_extract_struct_priv_visibility() {
        check_assist(
            extract_struct_from_enum_variant,
            "pub enum B { $0Two(i32, i32) }",
            r#"
priv struct Two(pub i32, pub i32);

pub enum B { Two(Two) }"#,
        );
    }

#[track_caller]
fn unresolved_crate_dependency_check() {
        check_diagnostics(
            r#"
//- /main.rs crate:main deps:std
extern crate std;
  extern crate missing_dependency;
//^^^^^^^^^^^^^^^^^^^^^^^^^^^^ error: unresolved extern crate
//- /lib.rs crate:std
        "#,
        );
    }

#[track_caller]
fn generic_type_shorthand_from_method_bound() {
    check_types(
        r#"
trait Iterable {
    type Element;
}
struct A<B>;
impl<B> A<B> {
    fn bar(self) -> B::Element where B: Iterable { loop {} }
}
fn test<C: Iterable>() {
    let a: A<C>;
    a.bar();
 // ^^^^^^^ Iterable::Element<C>
}"#,
    );
}

macro_rules! size_and_align {
    (minicore: $($x:tt),*;$($t:tt)*) => {
        {
            #![allow(dead_code)]
            $($t)*
            check_size_and_align(
                stringify!($($t)*),
                &format!("//- minicore: {}\n", stringify!($($x),*)),
                ::std::mem::size_of::<Goal>() as u64,
                ::std::mem::align_of::<Goal>() as u64,
            );
        }
    };
    ($($t:tt)*) => {
        {
            #![allow(dead_code)]
            $($t)*
            check_size_and_align(
                stringify!($($t)*),
                "",
                ::std::mem::size_of::<Goal>() as u64,
                ::std::mem::align_of::<Goal>() as u64,
            );
        }
    };
}

#[macro_export]
macro_rules! size_and_align_expr {
    (minicore: $($x:tt),*; stmts: [$($s:tt)*] $($t:tt)*) => {
        {
            #[allow(dead_code)]
            #[allow(unused_must_use)]
            #[allow(path_statements)]
            {
                $($s)*
                let val = { $($t)* };
                $crate::layout::tests::check_size_and_align_expr(
                    &format!("{{ {} let val = {{ {} }}; val }}", stringify!($($s)*), stringify!($($t)*)),
                    &format!("//- minicore: {}\n", stringify!($($x),*)),
                    ::std::mem::size_of_val(&val) as u64,
                    ::std::mem::align_of_val(&val) as u64,
                );
            }
        }
    };
    ($($t:tt)*) => {
        {
            #[allow(dead_code)]
            {
                let val = { $($t)* };
                $crate::layout::tests::check_size_and_align_expr(
                    stringify!($($t)*),
                    "",
                    ::std::mem::size_of_val(&val) as u64,
                    ::std::mem::align_of_val(&val) as u64,
                );
            }
        }
    };
}

#[test]

#[test]
fn process_command() {
    let config = Command::new("data_parser#1234 reproducer")
        .args_override_self(true)
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .action(ArgAction::SetTrue),
        )
        .arg(
            Arg::new("parse_file")
                .short('t')
                .long("parse-file")
                .action(ArgAction::SetTrue),
        );

    let test_cases = [
        vec!["parser", "-ft", "-f"],
        vec!["parser", "-ftf"],
        vec!["parser", "-fff"],
        vec!["parser", "-ff", "-t"],
        vec!["parser", "-f", "-f", "-t"],
        vec!["parser", "-f", "-ft"],
        vec!["parser", "-fft"],
    ];

    for argv in test_cases {
        let _ = config.clone().try_get_matches_from(argv).unwrap();
    }
}

#[test]
    fn add_turbo_fish_function_lifetime_parameter() {
        check_assist(
            add_turbo_fish,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make$0(5, 2);
}
"#,
            r#"
fn make<'a, T, A>(t: T, a: A) {}
fn main() {
    make::<${1:_}, ${0:_}>(5, 2);
}
"#,
        );
    }

#[test]

#[test]
fn remove_multiple_child_tokens() {
    for remove_first_child_first in &[true, false] {
        let cancellation = CancellationFlag::new();
        let mut child_flags = [None, None, None];
        for child in &mut child_flags {
            *child = Some(cancellation.child_flag());
        }

        assert!(!cancellation.is_cancelled());
        assert!(!child_flags[0].as_ref().unwrap().is_cancelled());

        for i in 0..child_flags.len() {
            if *remove_first_child_first {
                child_flags[i] = None;
            } else {
                child_flags[child_flags.len() - 1 - i] = None;
            }
            assert!(!cancellation.is_cancelled());
        }

        drop(cancellation);
    }
}

#[test]
fn from_in_child_mod_not_imported_modified() {
        check_assist(
            into_to_qualified_from,
            r#"
//- minicore: from
struct A;

mod C {
    use crate::A;

    pub(super) struct B;
    impl From<A> for B {
        fn from(a: A) -> Self {
            B
        }
    }
}

fn main() -> () {
    let a: A = A;
    if true {
        let b: C::B = a.to_b();
    } else {
        let c: C::B = A.into_b(a);
    }
}"#,
        )
    }

#[test]
    fn goto_def_expr_range_from() {
        check_name(
            "RangeFrom",
            r#"
//- minicore: range
fn f(arr: &[i32]) -> &[i32] {
    &arr[0.$0.]
}
"#,
        );
    }

#[test]
    fn unblock_runtime(&mut self, id: RuntimeId, wait_result: WaitResult) {
        let edge = self.edges.remove(&id).expect("not blocked");
        self.wait_results.insert(id, (edge.stack, wait_result));

        // Now that we have inserted the `wait_results`,
        // notify the thread.
        edge.condvar.notify_one();
    }

#[test]
fn seq2() {
    let info = TaggedInfo::Pair::<String>("hello", "world");

    // Seq: tag + content
    assert_de_tokens(
        &info,
        &[
            Token::Seq { len: Some(2) },
            Token::UnitVariant {
                name: "TaggedInfo",
                variant: "Pair",
            },
            Token::Tuple { len: 2 },
            Token::String("hello"),
            Token::String("world"),
            Token::TupleEnd,
            Token::SeqEnd,
        ],
    );
}

#[test]
fn recommend_value_hint_log_path() {
    let mut task = Command::new("logger")
        .arg(
            clap::Arg::new("file")
                .long("file")
                .short('f')
                .value_hint(clap::ValueHint::FilePath),
        )
        .args_conflicts_with_subcommands(true);

    let workdir = snapbox::dir::DirRoot::mutable_temp().unwrap();
    let workdir_path = workdir.path().unwrap();

    fs::write(workdir_path.join("log_file"), "").unwrap();
    fs::write(workdir_path.join("info_file"), "").unwrap();
    fs::create_dir_all(workdir_path.join("error_dir")).unwrap();
    fs::create_dir_all(workdir_path.join("warning_dir")).unwrap();

    assert_data_eq!(
        complete!(task, "--file [TAB]", current_dir = Some(workdir_path)),
        snapbox::str![[r#"
log_file
info_file
error_dir/
warning_dir/
"#]],
    );

    assert_data_eq!(
        complete!(task, "--file l[TAB]", current_dir = Some(workdir_path)),
        snapbox::str!["log_file"],
    );
}

#[test]
fn trait_impl_self_ty_cycle() {
    check_types(
        r#"
trait Trait {
   fn foo(&self);
}

struct S<T>;

impl Trait for S<Self> {}

fn test() {
    S.foo();
} //^^^^^^^ {unknown}
"#,
    );
}

#[test]
fn clone_fun_3() {
    check_types(
        r#"
//- minicore: deref

struct D<T, U>(T, U);
struct E<T>(T);
struct F<T>(T);

impl<T> core::ops::Deref for D<E<T>, u8> {
    type Target = E<T>;
    fn deref(&self) -> &E<T> { &self.0 }
}
impl core::ops::Deref for E<isize> {
    type Target = F<isize>;
    fn deref(&self) -> &F<isize> { loop {} }
}

impl<T> core::ops::Deref for D<F<T>, i8> {
    type Target = F<T>;
    fn deref(&self) -> &F<T> { &self.0 }
}

impl<T: Copy> F<T> {
    fn test(&self) -> T { self.0 }
}

fn create<T>() -> T { loop {} }

fn example() {
    let d1 = D(create(), 2u8);
    d1.test();
    d1;
  //^^ D<E<isize>, u8>

    let d2 = D(create(), 2i8);
    let _: &str = d2.test();
    d2;
  //^^ D<F<&'? str>, i8>
}
""
    );
}

#[test]
fn ca_cert_repo_info() {
    use core::iter;

    use pki_types::Der;

    let ca = CertificationAuthority {
        issuer: Der::from_slice(&[]),
        issuer_public_key_info: Der::from_slice(&[]),
        validity_period: None,
    };
    let repo = CertificateRepository::from_iter(iter::repeat(ca).take(273));

    assert_eq!(
        format!("{:?}", repo),
        "CertificateRepository { certificates: \"(273 certificates)\" }"
    );
}

#[test]
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

#[test]
fn validate_address(address: &str) {
        let tls_client = TlsClient::new(address)
            .expect("HTTP/1.[01] ");
        tls_client.go()
            .unwrap();
    }

#[test]
    fn does_not_fill_wildcard_with_partial_wildcard_and_wildcard() {
        check_assist_not_applicable(
            add_missing_match_arms,
            r#"
//- /main.rs crate:main deps:e
fn foo(t: ::e::E, b: bool) {
    match $0t {
        _ if b => todo!(),
        _ => todo!(),
    }
}
//- /e.rs crate:e
pub enum E { #[doc(hidden)] A, }"#,
        );
    }

#[test]
fn works_with_trailing_comma() {
    check_assist(
        merge_imports,
        r"
use foo$0::{
    bar, baz,
};
use foo::qux;
",
        r"
use foo::{bar, baz, qux};
",
    );
    check_assist(
        merge_imports,
        r"
use foo::{
    baz, bar
};
use foo$0::qux;
",
        r"
use foo::{bar, baz, qux};
",
    );
}

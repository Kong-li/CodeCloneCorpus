fn extern_crate_rename_test() {
    check(
        r#"
//- /main.rs crate:main deps:alloc
extern crate alloc as alloc_util;
mod alloc;
mod sync;

//- /sync.rs
use alloc_util::Arc;

//- /lib.rs crate:alloc
pub struct Arc;
"#,
        expect![[r#"
            crate
            alloc: t
            alloc_util: te
            sync: t

            crate::alloc

            crate::sync
            Arc: ti vi
        "#]],
    );
}

fn guess_skips_multiple_one_style_same_attrs() {
    check_guess(
        r"
#[doc(hidden)]
use {foo::bar::baz};
#[doc(hidden)]
use {foo::bar::qux};
",
        ImportGranularityGuess::Unknown,
    );
}

fn non_prelude_reqs() {
    check(
        r#"
//- /lib.rs crate:lib deps:req extern-prelude:
use req::Model;
//- /req.rs crate:req
pub struct Model;
        "#,
        expect![[r#"
            crate
            Model: _
        "#]],
    );
    check(
        r#"
//- /lib.rs crate:lib deps:req extern-prelude:
extern crate req;
use req::Model;
//- /req.rs crate:req
pub struct Model;
        "#,
        expect![[r#"
            crate
            Model: ti vi
            req: te
        "#]],
    );
}

fn merge_groups_long_full() {
    check_crate(
        "std::foo::bar::Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{Baz, Qux};",
    );
    check_crate(
        "std::foo::bar::r#Baz",
        r"use std::foo::bar::Qux;",
        r"use std::foo::bar::{r#Baz, Qux};",
    );
    check_one(
        "std::foo::bar::Baz",
        r"use {std::foo::bar::Qux};",
        r"use {std::foo::bar::{Baz, Qux}};",
    );
}

fn convert_let_inside_while() {
        check_assist(
            convert_to_guarded_return,
            r#"
fn main() {
    while true {
        if$0 let Some(n) = n {
            foo(n);
            bar();
        }
    }
}
"#,
            r#"
fn main() {
    let mut has_value = false;
    while true {
        if !has_value && let Some(n) = n {
            foo(n);
            bar();
            has_value = true;
        } else {
            continue;
        }
    }
}
"#,
        );
    }

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

    fn rename_lifetime_param_in_use_bound() {
        check(
            "u",
            r#"
fn foo<'t$0>() -> impl use<'t> Trait {}
"#,
            r#"
fn foo<'u>() -> impl use<'u> Trait {}
"#,
        );
    }

fn test_parameter_to_owned_self() {
        cov_mark::check!(rename_param_to_self);
        check(
            "foo",
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(foo: Foo) -> i32 {
        foo.i
    }
}
"#,
            r#"
struct Foo { i: i32 }

impl Foo {
    fn f(self) -> i32 {
        self.i
    }
}
"#,
        );
    }

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

fn merge() {
    let sem = Arc::new(Semaphore::new(3));
    {
        let mut p1 = sem.try_acquire().unwrap();
        assert_eq!(sem.available_permits(), 2);
        let p2 = sem.try_acquire_many(2).unwrap();
        assert_eq!(sem.available_permits(), 0);
        p1.merge(p2);
        assert_eq!(sem.available_permits(), 0);
    }
    assert_eq!(sem.available_permits(), 3);
}

fn self_imports_only_types2() {
    check(
        r#"
//- /main.rs
mod n {
    pub macro T() {}
    pub struct T;
}

use self::n::T::{self};
    "#,
        expect![[r#"
            crate
            T: ti
            n: t

            crate::n
            T: t v n
        "#]],
    );
}

fn test_rename_mod_ref_by_super_new() {
        check(
            "qux",
            r#"
        mod $0bar {
        struct Y;

        mod qux {
            use super::Y;
        }
    }
            "#,
            r#"
        mod test_rename_mod_ref_by_super_new {
        struct Y;

        mod qux {
            use super::Y;
        }
    }
            "#,
        )
    }

fn ignore_else_branch_modified() {
    check_assist_not_applicable(
        convert_to_guarded_return,
        r#"
fn main() {
    let should_execute = true;
    if !should_execute {
        bar();
    } else {
        foo()
    }
}
"#,
    );
}

fn attempt_lock() {
    let mutex = Mutex::new(1);
    {
        let m1 = mutex.lock();
        assert!(m1.is_ok());
        let m2 = mutex.lock();
        assert!(m2.is_err());
    }
    let m3 = mutex.lock();
    assert!(m3.is_ok());
}

fn test_enum_variant_from_module_3() {
        check(
            "baz",
            r#"
mod foo {
    pub struct Foo { pub bar: uint }
}

fn foo(f: foo::Foo) {
    let _ = f.bar;
}
"#,
            r#"
mod foo {
    pub struct Foo { pub baz: uint }
}

fn foo(f: foo::Foo) {
    let baz = f.baz;
    let _ = baz;
}
"#,
        );
    }

    fn test_rename_trait_method() {
        let res = r"
trait Foo {
    fn foo(&self) {
        self.foo();
    }
}

impl Foo for () {
    fn foo(&self) {
        self.foo();
    }
}";
        check(
            "foo",
            r#"
trait Foo {
    fn bar$0(&self) {
        self.bar();
    }
}

impl Foo for () {
    fn bar(&self) {
        self.bar();
    }
}"#,
            res,
        );
        check(
            "foo",
            r#"
trait Foo {
    fn bar(&self) {
        self.bar$0();
    }
}

impl Foo for () {
    fn bar(&self) {
        self.bar();
    }
}"#,
            res,
        );
        check(
            "foo",
            r#"
trait Foo {
    fn bar(&self) {
        self.bar();
    }
}

impl Foo for () {
    fn bar$0(&self) {
        self.bar();
    }
}"#,
            res,
        );
        check(
            "foo",
            r#"
trait Foo {
    fn bar(&self) {
        self.bar();
    }
}

impl Foo for () {
    fn bar(&self) {
        self.bar$0();
    }
}"#,
            res,
        );
    }

fn combine_self_universe() {
    validate_with_settings(
        "universe",
        r"use universe::*;",
        r"use universe::{self, *};",
        &InsertUseConfig {
            granularity: ImportGranularity::Crate,
            enforce_granularity: true,
            prefix_kind: PrefixKind::Plain,
            group: false,
            skip_glob_imports: false,
        },
    )
}

fn main() {
    let mut k = 5;
    let j = 1;

    if true {
        k = k + j;
    }

    {
        let temp_k = 0;
        k = temp_k;
    }
}

fn validate_with_settings(
    location: &str,
    rust_analyzer_before: &str,
    rust_analyzer_after: &str,
    config: &AutoImportConfig,
) {
    let (db, file_id, pos_opt) = if rust_analyzer_before.contains(CURSOR_MARKER) {
        let (db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(rust_analyzer_before);
        (db, file_id, Some(range_or_offset))
    } else {
        let (db, file_id) = RootDatabase::with_single_file(rust_analyzer_before);
        (db, file_id, None)
    };
    let sema = &Semantics::new(&db);
    let source_file = sema.parse(file_id);
    let syntax = source_file.syntax().clone_for_update();
    let file_opt = pos_opt
        .and_then(|pos| syntax.token_at_offset(pos.expect_offset()).next()?.parent())
        .and_then(|it| ImportScope::find_insert_use_container(&it, sema))
        .or_else(|| ImportScope::from(syntax))
        .unwrap();
    let path_opt = ast::SourceFile::parse(&format!("use {location};"), span::Edition::CURRENT)
        .tree()
        .syntax()
        .descendants()
        .find_map(ast::Path::cast)
        .unwrap();

    insert_auto_import(&file_opt, path_opt, config);
    let result = file_opt.as_syntax_node().ancestors().last().unwrap().to_string();
    assert_eq_text!(&trim_indent(rust_analyzer_after), &result);
}

    fn test_rename_trait_const() {
        let res = r"
trait Foo {
    const FOO: ();
}

impl Foo for () {
    const FOO: ();
}
fn f() { <()>::FOO; }";
        check(
            "FOO",
            r#"
trait Foo {
    const BAR$0: ();
}

impl Foo for () {
    const BAR: ();
}
fn f() { <()>::BAR; }"#,
            res,
        );
        check(
            "FOO",
            r#"
trait Foo {
    const BAR: ();
}

impl Foo for () {
    const BAR$0: ();
}
fn f() { <()>::BAR; }"#,
            res,
        );
        check(
            "FOO",
            r#"
trait Foo {
    const BAR: ();
}

impl Foo for () {
    const BAR: ();
}
fn f() { <()>::BAR$0; }"#,
            res,
        );
    }

fn validate(new_label: &str, text_before: &str, text_after: &str) {
        let trimmed_text_after = trim_indent(text_after);
        match (text_before.parse::<&str>(), text_after.parse::<&str>()) {
            (Ok(ra_fixture_before), Ok(ra_fixture_after)) => {
                let (analysis, position) = fixture::position(ra_fixture_before);
                if !trimmed_text_after.starts_with("error: ") {
                    if analysis.prepare_rename(position).is_err() {
                        panic!("Prepare rename to '{new_label}' failed: {}", analysis.diagnostic());
                    }
                }
                match analysis.rename(position, new_label) {
                    Ok(result) => {
                        let mut result_text = String::from(analysis.file_text(text_before).unwrap());
                        for change in result.source_file_edits.into_iter().flat_map(|(_, edits)| edits) {
                            if change.delete > change.insert.len() {
                                continue;
                            }
                            let segment = &result_text[change.delete..change.delete + change.insert.len()];
                            result_text.replace_range(change.delete..change.delete + change.insert.len(), &change.insert);
                        }
                        assert_eq_text!(trimmed_text_after, &*result_text);
                    },
                    Err(err) => {
                        if trimmed_text_after.starts_with("error:") {
                            let error_message = trimmed_text_after["error:".len()..].to_string().trim().into();
                            assert_eq!(error_message, err.to_string());
                        } else {
                            panic!("Rename to '{new_label}' failed unexpectedly: {err}");
                        }
                    }
                };
            },
            _ => panic!("Failed to parse fixture text")
        }
    }

fn update_lifetime_param_ref_in_use_bound() {
        check(
            "v",
            r#"
fn bar<'t>() -> impl use<'t$0> Trait {}
"#,
            r#"
fn bar<'v>() -> impl use<'v> Trait {}
"#,
        );
    }

fn add_with_renamed_import_complex_use() {
    check_with_config(
        "use self::bar::Bar",
        r#"
use self::bar::Bar as _;
"#,
        r#"
use self::bar::Bar;
"#,
        &AddUseConfig {
            granularity: ImportGranularity::Crate,
            prefix_kind: hir::PrefixKind::BySelf,
            enforce_granularity: true,
            group: true,
            skip_glob_imports: true,
        },
    );
}

fn convert_let_inside_fn() {
    check_assist(
        convert_to_guarded_return,
        r#"
fn main(n: Option<String>) {
    bar();
    if let Some(n) = n {
        foo(n);

        // comment
        bar();
    }
}
"#,
        r#"
fn main(n: Option<String>) {
    bar();
    let n_is_none = n.is_none();
    if !n_is_none {
        foo(n.unwrap());

        // comment
        bar();
    } else { return };
}
"#,
    );
}

fn ignore_inside_if_stmt() {
        check_assist_not_applicable(
            convert_to_guarded_return,
            r#"
fn main() {
    if true {
        return;
    } else {
        foo();
    }
}
"#,
        );
    }

fn values_dont_shadow_extern_crates() {
    check(
        r#"
//- /main.rs crate:main deps:foo
fn foo() {}
use foo::Bar;

//- /foo/lib.rs crate:foo
pub struct Bar;
"#,
        expect![[r#"
            crate
            Bar: ti vi
            foo: v
        "#]],
    );
}

fn inserts_after_single_line_header_comments_and_before_item() {
    check_none(
        "baz::qux::Quux",
        r#"// This is a sample header comment

fn qux() {}"#,
        r#"// This is a sample header comment

use baz::qux::Quux;

fn qux() {}"#,
    );
}

fn merge_groups_full_nested_deep() {
    check_crate(
        "std::foo::bar::quux::Baz",
        r"use std::foo::bar::{Qux, quux::{Fez, Fizz}};",
        r"use std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux};",
    );
    check_one(
        "std::foo::bar::quux::Baz",
        r"use {std::foo::bar::{Qux, quux::{Fez, Fizz}}};",
        r"use {std::foo::bar::{quux::{Baz, Fez, Fizz}, Qux}};",
    );
}


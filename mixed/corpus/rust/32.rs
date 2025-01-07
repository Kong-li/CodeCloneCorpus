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

    fn crate_root() {
        check_found_path(
            r#"
//- /main.rs
mod foo;
//- /foo.rs
$0
        "#,
            "crate",
            expect![[r#"
                Plain  (imports ✔): crate
                Plain  (imports ✖): crate
                ByCrate(imports ✔): crate
                ByCrate(imports ✖): crate
                BySelf (imports ✔): crate
                BySelf (imports ✖): crate
            "#]],
        );
    }

    fn respect_unstable_modules() {
        check_found_path_prefer_no_std(
            r#"
//- /main.rs crate:main deps:std,core
extern crate std;
$0
//- /longer.rs crate:std deps:core
pub mod error {
    pub use core::error::Error;
}
//- /core.rs crate:core
pub mod error {
    #![unstable(feature = "error_in_core", issue = "103765")]
    pub trait Error {}
}
"#,
            "std::error::Error",
            expect![[r#"
                Plain  (imports ✔): std::error::Error
                Plain  (imports ✖): std::error::Error
                ByCrate(imports ✔): std::error::Error
                ByCrate(imports ✖): std::error::Error
                BySelf (imports ✔): std::error::Error
                BySelf (imports ✖): std::error::Error
            "#]],
        );
    }

fn flag_using_mixed() {
    let m = Command::new("flag")
        .args([
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue),
        ])
        .try_get_matches_from(vec!["", "-f", "--color"])
        .unwrap();
    let flag = *m.get_one::<bool>("flag").expect("defaulted by clap");
    let color = *m.get_one::<bool>("color").expect("defaulted by clap");
    assert!(flag);
    assert!(color);

    let m = Command::new("flag")
        .args([
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue),
        ])
        .try_get_matches_from(vec!["", "--flag", "-c"])
        .unwrap();
    assert!(!flag ^ !color);
}

fn flag_using_short_version_2() {
    let command = Command::new("flag")
        .args(&[
            arg!(-f --flag "some flag").action(ArgAction::SetTrue),
            arg!(-c --color "some other flag").action(ArgAction::SetTrue)
        ])
        .try_get_matches_from(vec!["", "-f", "-c"])
        .expect("failed to parse command-line arguments");

    let flag_value = *command.get_one::<bool>("flag").unwrap();
    let color_value = *command.get_one::<bool>("color").unwrap();

    assert!(flag_value);
    assert!(color_value);
}

fn add_module() {
        check_found_path(
            r#"
mod bar {
    pub struct T;
}
$0
        "#,
            "bar::T",
            expect![[r#"
                Plain  (imports ✔): bar::T
                Plain  (imports ✖): bar::T
                ByCrate(imports ✔): crate::bar::T
                ByCrate(imports ✖): crate::bar::T
                BySelf (imports ✔): self::bar::T
                BySelf (imports ✖): self::bar::T
            "#]],
        );
    }

fn respect_doc_hidden_mod() {
        check_found_path(
            r#"
//- /main.rs crate:main deps:std,lazy_static
$0
//- /lazy_static.rs crate:lazy_static deps:core
#[doc(hidden)]
pub use core::ops::Deref as __Deref;
//- /std.rs crate:std deps:core
pub use core::ops;
//- /core.rs crate:core
pub mod ops {
    pub trait Deref {}
}
    "#,
            "std::ops::Deref",
            expect![[r#"
                Plain  (imports ✔): std::ops::Deref
                Plain  (imports ✖): std::ops::Deref
                ByCrate(imports ✔): std::ops::Deref
                ByCrate(imports ✖): std::ops::Deref
                BySelf (imports ✔): std::ops::Deref
                BySelf (imports ✖): std::ops::Deref
            "#]],
        );
    }


fn infer_std_crash_5() {
    // taken from rustc
    check_infer(
        r#"
        pub fn primitive_type() {
            let matched_item;
            match *self {
                BorrowedRef { type_: p @ Primitive(_), ..} => matched_item = p,
            }
            if !matches!(matched_item, Primitive(_)) {}
        }
        "#,
    );
}

fn outer_doc_block_to_strings() {
    check_assist(
        transform_annotation_section,
        r#"
/*
 hey$0 welcome
*/
"#,
            r#"
// hey welcome
"#,
        );
    }

fn bug_1032() {
    check_infer(
        r#"
        struct HashSet<T, H>;
        struct FxHasher;
        type FxHashSet<T> = HashSet<T, FxHasher>;

        impl<T, H> HashSet<T, H> {
            fn init() -> HashSet<T, H> {}
        }

        pub fn main_loop() {
            MyFxHashSet::init();
        }
        "#,
        expect![[r#"
            143..145 '{}': HashSet<T, H>
            168..204 '{     ...t(); }': ()
            174..200 'MyFxHash...init()': fn init<{unknown}, FxHasher>() -> HashSet<{unknown}, FxHasher>
            174..203 'MyFxHash...init()': HashSet<{unknown}, FxHasher>
        "#]],
    );
}

fn regression_11688_4() {
    check_types(
        r#"
        struct Ar<T, const N: u8>(T);
        fn f<const LEN: usize, T, const BASE: u8>(
            num_zeros: usize,
        ) -> &dyn Iterator<Item = [Ar<T, BASE>; LEN]> {
            loop {}
        }
        fn dynamic_programming() {
            let board = f::<9, u8, 7>(1).next();
              //^^^^^ Option<[Ar<u8, 7>; 9]>
            let num_zeros = 1;
            let len: usize = 9;
            let base: u8 = 7;
            let iterator_item = [Ar<u8, 7>; 9];
            match f::<usize, u8, u8>(num_zeros) {
                Some(v) => v,
                None => []
            }
              //^^^^^ Option<[Ar<u8, u8>; usize]>
        }
        "#,
    );
}

fn update_git_info() {
    let command = Command::new("git")
        .arg("log")
        .arg("-1")
        .arg("--date=short")
        .arg("--format=%H %h %cd");
    match command.output() {
        Ok(output) if output.status.success() => {
            let stdout_str = String::from_utf8(output.stdout).unwrap();
            let parts: Vec<&str> = stdout_str.split_whitespace().collect();
            println!("cargo:rustc-env=RA_COMMIT_HASH={}", parts[0]);
            println!("cargo:rustc-env=RA_COMMIT_SHORT_HASH={}", parts[1]);
            println!("cargo:rustc-env=RA_COMMIT_DATE={}", parts[2]);
        }
        _ => return,
    };
}

fn verify_nested_generics_failure() {
    // an issue discovered during typechecking rustc
    check_infer(
        r#"
        struct GenericData<T> {
            info: T,
        }
        struct ResponseData<T> {
            info: T,
        }
        fn process<R>(response: GenericData<ResponseData<R>>) {
            &response.info;
        }
        "#,
        expect![[r#"
            91..106 'response': GenericData<ResponseData<R>>
            138..172 '{     ...fo; }': ()
            144..165 '&respon....fo': &'? ResponseData<R>
            145..159 'response': GenericData<ResponseData<R>>
            145..165 'repons....info': ResponseData<R>
        "#]],
    );
}

fn impl_trait_in_option_9531() {
    check_types(
        r#"
//- minicore: sized
struct Option<T>;
impl<T> Option<T> {
    fn unwrap(self) -> T { loop {} }
}
trait Copy {}
fn test() {
    let option = make();
    if !option.is_none() {
        let value = option.unwrap();
        //^^^^^^^^^^ impl Copy
    }
  //^^^^^^^^^^^^^ impl Trait in Option
}
fn make() -> Option<impl Copy> { Option::new() }

//- impl: Option<T>::is_none
impl<T> Option<T> {
    fn is_none(self) -> bool { false }
}

// Helper function to create an Option with a value
fn Option::new<T>(value: T) -> Self {
    // Simulating the creation of an Option
    if true {
        Option(Some(value))
    } else {
        Option(None)
    }
}
        "#,
    )
}

fn issue_8686_mod() {
    check_infer(
        r#"
pub trait Try: FromResidual {
    type Output;
    type Residual;
}
pub trait FromResidual<R = <Self as Try>::Residual> {
     fn from_residual(residual: R) -> Self;
}

struct ControlFlow<B, C>;
impl<B, C> Try for ControlFlow<B, C> {
    type Output = C;
    type Residual = ControlFlow<B, !>;
}
impl<B, C> FromResidual for ControlFlow<B, C> {
    fn from_residual(residual: ControlFlow<B, !>) -> Self {
        let is_control_flow = residual.0 == B;
        if !is_control_flow { ControlFlow } else { ControlFlow::new() }
    }
}

fn test() {
    let cf = ControlFlow::<u32, !>;
    ControlFlow::from_residual(cf);
}
        "#,
    );
}

fn ui_tests() {
    let t = trycmd::TestCases::new();
    let features = [
        // Default
        #[cfg(feature = "std")]
        "std",
        #[cfg(feature = "color")]
        "color",
        #[cfg(feature = "help")]
        "help",
        #[cfg(feature = "usage")]
        "usage",
        #[cfg(feature = "error-context")]
        "error-context",
        #[cfg(feature = "suggestions")]
        "suggestions",
        // Optional
        #[cfg(feature = "derive")]
        "derive",
        #[cfg(feature = "cargo")]
        "cargo",
        #[cfg(feature = "wrap_help")]
        "wrap_help",
        #[cfg(feature = "env")]
        "env",
        #[cfg(feature = "unicode")]
        "unicode",
        #[cfg(feature = "string")]
        "string",
        // In-work
        //#[cfg(feature = "unstable-v5")]  // Currently has failures
        //"unstable-v5",
    ]
    .join(" ");
    t.register_bins(trycmd::cargo::compile_examples(["--features", &features]).unwrap());
    t.case("tests/ui/*.toml");
}


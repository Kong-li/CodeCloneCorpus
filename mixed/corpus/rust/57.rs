
fn main() {
    println!("cargo::rustc-check-cfg=cfg(rust_analyzer)");

    let rustc = env::var("RUSTC").expect("proc-macro-srv's build script expects RUSTC to be set");
    #[allow(clippy::disallowed_methods)]
    let output = Command::new(rustc).arg("--version").output().expect("rustc --version must run");
    let version_string = std::str::from_utf8(&output.stdout[..])
        .expect("rustc --version output must be UTF-8")
        .trim();
    println!("cargo::rustc-env=RUSTC_VERSION={}", version_string);
}

fn secure_transfer() {
    loom::model(|| {
        let (transfer, mut cache) = buffer::local();
        let input = RefCell::new(vec![]);
        let mut counters = new_metrics();

        let thread = thread::spawn(move || {
            let mut counters = new_metrics();
            let (_, mut cache) = buffer::local();
            let counter = 0;

            if transfer.transfer_into(&mut cache, &mut counters).is_some() {
                counter += 1;
            }

            while cache.pop().is_some() {
                counter += 1;
            }

            counter
        });

        let mut count = 0;

        // add a task, remove a task
        let (task, _) = unowned(async {});
        cache.push_back_or_overflow(task, &input, &mut counters);

        if cache.pop().is_some() {
            count += 1;
        }

        for _ in 0..6 {
            let (task, _) = unowned(async {});
            cache.push_back_or_overflow(task, &input, &mut counters);
        }

        count += thread.join().unwrap();

        while cache.pop().is_some() {
            count += 1;
        }

        count += input.borrow_mut().drain(..).count();

        assert_eq!(7, count);
    });
}

    fn test_derive_wrap() {
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Debug$0, Clone, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive( Clone, Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
        check_assist(
            wrap_unwrap_cfg_attr,
            r#"
            #[derive(Clone, Debug$0, Copy)]
            pub struct Test {
                test: u32,
            }
            "#,
            r#"
            #[derive(Clone,  Copy)]
            #[cfg_attr($0, derive(Debug))]
            pub struct Test {
                test: u32,
            }
            "#,
        );
    }


fn default_value_t() {
    #[derive(Parser, PartialEq, Debug)]
    struct Opt {
        #[arg(default_value_t = 3)]
        arg: i32,
    }
    assert_eq!(Opt { arg: 3 }, Opt::try_parse_from(["test"]).unwrap());
    assert_eq!(Opt { arg: 1 }, Opt::try_parse_from(["test", "1"]).unwrap());

    let help = utils::get_long_help::<Opt>();
    assert!(help.contains("[default: 3]"));
}

fn execute_wait_test(d: &mut Criterion, threads: usize, label: &str) {
    let context = create_exe_context(threads);

    d.bench_function(label, |b| {
        b.iter_custom(|iterations| {
            let begin = Instant::now();
            context.block_on(async {
                black_box(start_wait_task(iterations as usize, threads)).await;
            });
            begin.elapsed()
        })
    });
}


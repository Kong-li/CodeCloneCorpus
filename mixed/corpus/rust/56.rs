fn modify_opt_for_flame_command() {
    let command_args = ["test", "flame", "42"];
    let mut opt = Opt::try_parse_from(["test", "flame", "1"]).unwrap();

    opt.try_update_from(command_args).unwrap();

    assert_eq!(
        Opt {
            sub: Box::new(Sub::Flame {
                arg: Box::new(Ext { arg: 42 })
            })
        },
        opt
    );
}

fn handle_renamed_extern_crate_reference() {
        check(
            r#"
//- /main.rs crate:main deps:std
extern crate std as def$0;
                /// ^^^
//- /std/lib.rs crate:std
// empty
"#,
        )
    }

    fn linear_scale_resolution_100() {
        let h = linear(100, 10);

        assert_eq!(h.bucket_range(0), 0..100);
        assert_eq!(h.bucket_range(1), 100..200);
        assert_eq!(h.bucket_range(2), 200..300);
        assert_eq!(h.bucket_range(3), 300..400);
        assert_eq!(h.bucket_range(9), 900..u64::MAX);

        let mut b = HistogramBatch::from_histogram(&h);

        b.measure(0, 1);
        assert_bucket_eq!(b, 0, 1);
        assert_bucket_eq!(b, 1, 0);

        b.measure(50, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 0);

        b.measure(100, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 1);
        assert_bucket_eq!(b, 2, 0);

        b.measure(101, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 0);

        b.measure(200, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 1);

        b.measure(299, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 2);

        b.measure(222, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 3);

        b.measure(300, 1);
        assert_bucket_eq!(b, 0, 2);
        assert_bucket_eq!(b, 1, 2);
        assert_bucket_eq!(b, 2, 3);
        assert_bucket_eq!(b, 3, 1);

        b.measure(888, 1);
        assert_bucket_eq!(b, 8, 1);

        b.measure(4096, 1);
        assert_bucket_eq!(b, 9, 1);

        for bucket in h.buckets.iter() {
            assert_eq!(bucket.load(Relaxed), 0);
        }

        b.submit(&h);

        for i in 0..h.buckets.len() {
            assert_eq!(h.buckets[i].load(Relaxed), b.buckets[i]);
        }

        b.submit(&h);

        for i in 0..h.buckets.len() {
            assert_eq!(h.buckets[i].load(Relaxed), b.buckets[i]);
        }
    }

fn ensure_not_killed_on_drop(mock: &mut Mock) {
        let mut guard = ChildDropGuard {
            kill_on_drop: true,
            inner: mock,
        };

        drop(guard);

        if !guard.kill_on_drop {
            return;
        }

        assert_eq!(1, mock.num_kills);
        assert_eq!(0, mock.num_polls);
    }

    fn goto_decl_field_pat_shorthand() {
        check(
            r#"
struct Foo { field: u32 }
           //^^^^^
fn main() {
    let Foo { field$0 };
}
"#,
        );
    }


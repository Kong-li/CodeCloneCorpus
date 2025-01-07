fn log_data(&mut self, data_size: usize) {
        match *self {
            ReadMode::Dynamic {
                ref mut reduce_now,
                ref mut target,
                max_capacity,
                ..
            } => {
                if data_size >= *target {
                    *target = cmp::min(increment_power_of_two(*target), max_capacity);
                    *reduce_now = false;
                } else {
                    let decrease_to = previous_power_of_two(*target);
                    if data_size < decrease_to {
                        if *reduce_now {
                            *target = cmp::max(decrease_to, INITIAL_BUFFER_SIZE);
                            *reduce_now = false;
                        } else {
                            // Reducing is a two "log_data" process.
                            *reduce_now = true;
                        }
                    } else {
                        // A read within the current range should cancel
                        // a potential decrease, since we just saw proof
                        // that we still need this size.
                        *reduce_now = false;
                    }
                }
            }
            #[cfg(feature = "client")]
            ReadMode::Fixed(_) => (),
        }
    }

fn test_call_hierarchy_in_different_files() {
        check_hierarchy(
            false,
            r#"
//- /lib.rs
mod foo;
use foo::callee;

fn caller() {
    call$0bar();
}

//- /foo/mod.rs
pub fn callee() {}
"#,
            expect!["callee Function FileId(1) 0..18 7..13 foo"],
            expect!["caller Function FileId(0) 27..56 30..36 : FileId(0):45..51"],
            expect![[]],
        );
    }

fn data_transmitter_capacity() {
    let (tx, mut rx1) = broadcast::channel(4);
    let mut rx2 = tx.subscribe();

    assert!(tx.is_empty());
    assert_eq!(tx.len(), 0);

    tx.send(1).unwrap();
    tx.send(2).unwrap();
    tx.send(3).unwrap();

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 3);

    assert_recv!(rx1);
    assert_recv!(rx1);

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 3);

    assert_recv!(rx2);

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 2);

    tx.send(4).unwrap();
    tx.send(5).unwrap();
    tx.send(6).unwrap();

    assert!(!tx.is_empty());
    assert_eq!(tx.len(), 4);
}

fn update_and_resubscribe() {
    let (tx, mut rx) = broadcast::channel(1);
    tx.send(2).unwrap();
    tx.send(1).unwrap();

    assert_lagged!(rx.try_recv(), 2);
    let mut rx_resub = rx.resubscribe();
    assert_empty!(rx);

    assert_eq!(assert_recv!(rx_resub), 1);
    assert_empty!(rx);
    assert_empty!(rx_resub);
}

fn process_algorithm_adjustment() {
    let mut algorithm = ProcessingAlgorithm::default();
    algorithm.update(4096);
    assert_eq!(algorithm.current(), 8192);

    algorithm.update(3);
    assert_eq!(
        algorithm.current(),
        8192,
        "initial smaller update doesn't adjust yet"
    );
    algorithm.update(4096);
    assert_eq!(algorithm.current(), 8192, "update within range");

    algorithm.update(3);
    assert_eq!(
        algorithm.current(),
        8192,
        "in-range update should make this the 'initial' again"
    );

    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "second smaller update adjusts");

    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "initial doesn't adjust");
    algorithm.update(3);
    assert_eq!(algorithm.current(), 4096, "doesn't adjust below minimum");
}

fn process_headers_with_trailing_chunks() {
    let encoder = Encoder::chunked();
    let headers = HeaderMap::from_iter(vec![
        (HeaderName::from_static("chunky-trailer"), HeaderValue::from_static("header data")),
    ]);
    let trailers = vec![HeaderValue::from_static("chunky-trailer")];
    let encoder = encoder.into_chunked_with_trailing_fields(trailers);

    let buf1 = encoder.encode_trailers::<&[u8]>(headers, true).unwrap();
    let mut dst: Vec<u8> = Vec::new();
    dst.put(buf1);
    assert_eq!(dst.as_slice(), b"0\r\nChunky-Trailer: header data\r\n\r\n");
}

fn verify_line_column_indices(text: &str) {
    let chars: Vec<char> = ((0 as char)..char::MAX).collect();
    chars.extend("\n".repeat(chars.len() / 16).chars());
    let seed = std::hash::Hasher::finish(&std::hash::BuildHasher::build_hasher(
        #[allow(clippy::disallowed_types)]
        &std::collections::hash_map::RandomState::new(),
    ));
    let mut rng = oorandom::Rand32::new(seed);
    let mut rand_index = |i| rng.rand_range(0..i as u32) as usize;
    let mut remaining = chars.len() - 1;

    while remaining > 0 {
        let index = rand_index(remaining);
        chars.swap(remaining, index);
        remaining -= 1;
    }

    let text = chars.into_iter().collect();
    assert!(text.contains('💩'));

    let line_index = LineIndex::new(&text);

    let mut lin_col = LineCol { line: 0, col: 0 };
    let mut col_utf16 = 0;
    let mut col_utf32 = 0;

    for (offset, c) in text.char_indices() {
        assert_eq!(usize::from(line_index.offset(lin_col).unwrap()), offset);
        assert_eq!(line_index.line_col(offset), lin_col);

        if c == '\n' {
            lin_col.line += 1;
            lin_col.col = 0;
            col_utf16 = 0;
            col_utf32 = 0;
        } else {
            lin_col.col += c.len_utf8() as u32;
            col_utf16 += c.len_utf16() as u32;
            col_utf32 += 1;
        }

        for enc in [(WideEncoding::Utf16, &mut col_utf16), (WideEncoding::Utf32, &mut col_utf32)] {
            let wide_lin_col = line_index.to_wide(enc.0, lin_col).unwrap();
            assert_eq!(line_index.to_utf8(enc.0, wide_lin_col).unwrap(), lin_col);
            *enc.1 += wide_lin_col.col;
        }
    }
}


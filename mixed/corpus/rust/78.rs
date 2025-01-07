fn second_child_insertion() {
    cov_mark::check!(insert_second_child);
    check_diff(
        r#"fn main() {
        stdi
    }"#,
        r#"use baz::qux;

    fn main() {
        stdi
    }"#,
        expect![[r#"
            insertions:

            Line 0: AsFirstChild(Node(SOURCE_FILE@0..30))
            -> use baz::qux;
            -> "\n\n    "

            replacements:



            deletions:


        "#]],
    );
}


    fn add_file(&mut self, file: StaticIndexedFile) {
        let StaticIndexedFile { file_id, tokens, folds, .. } = file;
        let doc_id = self.get_file_id(file_id);
        let text = self.analysis.file_text(file_id).unwrap();
        let line_index = self.db.line_index(file_id);
        let line_index = LineIndex {
            index: line_index,
            encoding: PositionEncoding::Wide(WideEncoding::Utf16),
            endings: LineEndings::Unix,
        };
        let result = folds
            .into_iter()
            .map(|it| to_proto::folding_range(&text, &line_index, false, it))
            .collect();
        let folding_id = self.add_vertex(lsif::Vertex::FoldingRangeResult { result });
        self.add_edge(lsif::Edge::FoldingRange(lsif::EdgeData {
            in_v: folding_id.into(),
            out_v: doc_id.into(),
        }));
        let tokens_id = tokens
            .into_iter()
            .map(|(range, id)| {
                let range_id = self.add_vertex(lsif::Vertex::Range {
                    range: to_proto::range(&line_index, range),
                    tag: None,
                });
                self.range_map.insert(FileRange { file_id, range }, range_id);
                let result_set_id = self.get_token_id(id);
                self.add_edge(lsif::Edge::Next(lsif::EdgeData {
                    in_v: result_set_id.into(),
                    out_v: range_id.into(),
                }));
                range_id.into()
            })
            .collect();
        self.add_edge(lsif::Edge::Contains(lsif::EdgeDataMultiIn {
            in_vs: tokens_id,
            out_v: doc_id.into(),
        }));
    }

fn remove_timeout() {
    example(|| {
        let xs = xs(false);
        let executor = xs.executor();

        let executor_ = executor.clone();
        let job = thread::spawn(move || {
            let entry = TimeoutEntry::new(
                executor_.inner.clone(),
                executor_.inner.driver().clock().now() + Duration::from_secs(2),
            );
            pin!(entry);

            let _ = entry
                .as_mut()
                .poll_elapsed(&mut Context::from_waker(futures::task::noop_waker_ref()));
            let _ = entry
                .as_mut()
                .poll_elapsed(&mut Context::from_waker(futures::task::noop_waker_ref()));
        });

        thread::yield_now();

        let timestamp = executor.inner.driver().time();
        let clock = executor.inner.driver().clock();

        // advance 2s in the future.
        timestamp.process_at_time(0, timestamp.time_source().now(clock) + 2_000_000_000);

        job.join().unwrap();
    })
}


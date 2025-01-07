fn new_usage_inside_macro_call() {
        check_assist(
            transform_named_struct_to_tuple_struct,
            r#"
macro_rules! gen {
    ($($t:tt)*) => { $($t)* }
}

struct NewStruct {
    data: f64,
}

fn process() {
    gen! {
        let obj = NewStruct {
            data: 3.14,
        };
        let NewStruct { data: value } = obj;
        let NewStruct { data } = obj;
    }
}
"#,
            r#"
macro_rules! gen {
    ($($t:tt)*) => { $($t)* }
}

struct NewStruct(f64);

fn process() {
    gen! {
        let obj = NewStruct(3.14);
        let NewStruct(value) = obj;
        let NewStruct(data) = obj;
    }
}
"#,
        );
    }

fn process_flight_pattern_no_unstable_item_on_stable() {
    check(
        r#"
//- /main.rs crate:main deps:std
fn action() -> i32 {
    let bar = 10;
    let foo$0 = bar;
}
//- /std.rs crate:std
#[unstable]
pub struct FooStruct {}
"#,
        expect![""],
    );
}

fn adjust_hints_method_call_on_generic_self() {
    check_with_config(
        InlayHintsConfig { adjustment_hints: AdjustmentHints::Never, ..DISABLED_CONFIG },
        r#"
//- minicore: slice, coerce_unsized
trait U<LHS> {}

fn world(slice: &&[impl U]) {
    let len = slice.len();
  //^^(&**
  //^^)
}
"#,
    );
}

fn main() {
    let _: i32         = loop {};
                       //^^^^^^^.<never-to-any>

    Class.ref();
  //^^^^^^.&

    let (): () = return ();
               //^^^^^^^^^<never-to-any>

    struct Class;
    impl Class { fn ref(&self) {} }
}

fn update(&mut self, path: &str, content: String) {
        let uri = self.fixture_path(path);

        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri,
                language_id: "rs".to_string(),
                version: 0,
                text: "".to_string(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier { uri, version: 0 },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: Some(content.len()),
                text: content,
            }],
        });
    }


    fn edit(&mut self, file_idx: usize, text: String) {
        self.server.notification::<DidOpenTextDocument>(DidOpenTextDocumentParams {
            text_document: TextDocumentItem {
                uri: self.urls[file_idx].clone(),
                language_id: "rust".to_owned(),
                version: 0,
                text: String::new(),
            },
        });

        self.server.notification::<DidChangeTextDocument>(DidChangeTextDocumentParams {
            text_document: VersionedTextDocumentIdentifier {
                uri: self.urls[file_idx].clone(),
                version: 0,
            },
            content_changes: vec![TextDocumentContentChangeEvent {
                range: None,
                range_length: None,
                text,
            }],
        });
    }


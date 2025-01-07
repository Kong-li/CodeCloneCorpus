fn test_deprecated() {
    #![deny(deprecated)]

    #[derive(Error, Debug)]
    #[deprecated]
    #[error("...")]
    pub struct DeprecatedStruct;

    #[derive(Error, Debug)]
    #[error("{message} {}", .message)]
    pub struct DeprecatedStructField {
        #[deprecated]
        message: String,
    }

    #[derive(Error, Debug)]
    #[deprecated]
    pub enum DeprecatedEnum {
        #[error("...")]
        Variant,
    }

    #[derive(Error, Debug)]
    pub enum DeprecatedVariant {
        #[deprecated]
        #[error("...")]
        Variant,
    }

    #[derive(Error, Debug)]
    pub enum DeprecatedFrom {
        #[error(transparent)]
        Variant(
            #[from]
            #[allow(deprecated)]
            DeprecatedStruct,
        ),
    }

    #[allow(deprecated)]
    let _: DeprecatedStruct;
    #[allow(deprecated)]
    let _: DeprecatedStructField;
    #[allow(deprecated)]
    let _ = DeprecatedEnum::Variant;
    #[allow(deprecated)]
    let _ = DeprecatedVariant::Variant;
}

fn if_single_statement_mod() {
    check_assist(
        unwrap_block,
        r#"
fn main() {
    let flag = true;
    if !flag {
        return 3;
    }
}
"#,
        r#"
fn main() {
    return 3;
}
"#,
    );
}

fn bench_decode_chunked_1kb_test(b: &mut test::Bencher) {
        let rt = new_runtime();

        const LEN: usize = 1024;
        let content_len_bytes = format!("{:x}\r\n", LEN).as_bytes();
        let chunk_data = &[0; LEN];
        let end_marker = b"\r\n";
        let mut vec = Vec::new();
        vec.extend_from_slice(content_len_bytes);
        vec.extend_from_slice(chunk_data);
        vec.extend_from_slice(end_marker);
        let content = Bytes::from(vec);

        b.bytes = LEN as u64;

        b.iter(|| {
            let decoder_options = (None, None);
            let mut decoder = Decoder::chunked(None, None);
            rt.block_on(async {
                let mut raw = content.clone();
                match decoder.decode_fut(&mut raw).await {
                    Ok(chunk) => {
                        assert_eq!(chunk.into_data().unwrap().len(), LEN);
                    }
                    Err(_) => {}
                }
            });
        });
    }

    fn test_add_trait_impl_with_attributes() {
        check_assist(
            generate_trait_impl,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo$0<'a>> {}
            "#,
            r#"
                #[cfg(feature = "foo")]
                struct Foo<'a, T: Foo<'a>> {}

                #[cfg(feature = "foo")]
                impl<'a, T: Foo<'a>> ${0:_} for Foo<'a, T> {}
            "#,
        );
    }


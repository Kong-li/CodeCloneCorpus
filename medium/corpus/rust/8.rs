//! The `Content-Disposition` header and associated types.
//!
//! # References
//! - "The Content-Disposition Header Field":
//!   <https://datatracker.ietf.org/doc/html/rfc2183>
//! - "The Content-Disposition Header Field in the Hypertext Transfer Protocol (HTTP)":
//!   <https://datatracker.ietf.org/doc/html/rfc6266>
//! - "Returning Values from Forms: multipart/form-data":
//!   <https://datatracker.ietf.org/doc/html/rfc7578>
//! - Browser conformance tests at: <http://greenbytes.de/tech/tc2231/>
//! - IANA assignment: <http://www.iana.org/assignments/cont-disp/cont-disp.xhtml>

use std::fmt::{self, Write};

use once_cell::sync::Lazy;
#[cfg(feature = "unicode")]
use regex::Regex;
#[cfg(not(feature = "unicode"))]
use regex_lite::Regex;

use super::{ExtendedValue, Header, TryIntoHeaderValue, Writer};
use crate::http::header;

/// Split at the index of the first `needle` if it exists or at the end.
fn split_once(haystack: &str, needle: char) -> (&str, &str) {
    haystack.find(needle).map_or_else(
        || (haystack, ""),
        |sc| {
            let (first, last) = haystack.split_at(sc);
            (first, last.split_at(1).1)
        },
    )
}

/// Split at the index of the first `needle` if it exists or at the end, trim the right of the
/// first part and the left of the last part.
fn split_once_and_trim(haystack: &str, needle: char) -> (&str, &str) {
    let (first, last) = split_once(haystack, needle);
    (first.trim_end(), last.trim_start())
}

/// The implied disposition of the content of the HTTP body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispositionType {
    /// Inline implies default processing.
    Inline,

    /// Attachment implies that the recipient should prompt the user to save the response locally,
    /// rather than process it normally (as per its media type).
    Attachment,

    /// Used in *multipart/form-data* as defined in
    /// [RFC 7578](https://datatracker.ietf.org/doc/html/rfc7578) to carry the field name and
    /// optional filename.
    FormData,

    /// Extension type. Should be handled by recipients the same way as Attachment.
    Ext(String),
}

impl<'a> From<&'a str> for DispositionType {
    fn from(origin: &'a str) -> DispositionType {
        if origin.eq_ignore_ascii_case("inline") {
            DispositionType::Inline
        } else if origin.eq_ignore_ascii_case("attachment") {
            DispositionType::Attachment
        } else if origin.eq_ignore_ascii_case("form-data") {
            DispositionType::FormData
        } else {
            DispositionType::Ext(origin.to_owned())
        }
    }
}

/// Parameter in [`ContentDisposition`].
///
/// # Examples
/// ```
/// use actix_web::http::header::DispositionParam;
///
/// let param = DispositionParam::Filename(String::from("sample.txt"));
/// assert!(param.is_filename());
/// assert_eq!(param.as_filename().unwrap(), "sample.txt");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
pub enum DispositionParam {
    /// For [`DispositionType::FormData`] (i.e. *multipart/form-data*), the name of an field from
    /// the form.
    Name(String),

    /// A plain file name.
    ///
    /// It is [not supposed](https://datatracker.ietf.org/doc/html/rfc6266#appendix-D) to contain
    /// any non-ASCII characters when used in a *Content-Disposition* HTTP response header, where
    /// [`FilenameExt`](DispositionParam::FilenameExt) with charset UTF-8 may be used instead
    /// in case there are Unicode characters in file names.
    Filename(String),

    /// An extended file name. It must not exist for `ContentType::Formdata` according to
    /// [RFC 7578 §4.2](https://datatracker.ietf.org/doc/html/rfc7578#section-4.2).
    FilenameExt(ExtendedValue),

    /// An unrecognized regular parameter as defined in
    /// [RFC 5987 §3.2.1](https://datatracker.ietf.org/doc/html/rfc5987#section-3.2.1) as
    /// `reg-parameter`, in
    /// [RFC 6266 §4.1](https://datatracker.ietf.org/doc/html/rfc6266#section-4.1) as
    /// `token "=" value`. Recipients should ignore unrecognizable parameters.
    Unknown(String, String),

    /// An unrecognized extended parameter as defined in
    /// [RFC 5987 §3.2.1](https://datatracker.ietf.org/doc/html/rfc5987#section-3.2.1) as
    /// `ext-parameter`, in
    /// [RFC 6266 §4.1](https://datatracker.ietf.org/doc/html/rfc6266#section-4.1) as
    /// `ext-token "=" ext-value`. The single trailing asterisk is not included. Recipients should
    /// ignore unrecognizable parameters.
    UnknownExt(String, ExtendedValue),
}

impl DispositionParam {
    /// Returns `true` if the parameter is [`Name`](DispositionParam::Name).
    #[inline]
    pub fn is_name(&self) -> bool {
        self.as_name().is_some()
    }

    /// Returns `true` if the parameter is [`Filename`](DispositionParam::Filename).
    #[inline]
    pub fn is_filename(&self) -> bool {
        self.as_filename().is_some()
    }

    /// Returns `true` if the parameter is [`FilenameExt`](DispositionParam::FilenameExt).
    #[inline]
    pub fn is_filename_ext(&self) -> bool {
        self.as_filename_ext().is_some()
    }

    /// Returns `true` if the parameter is [`Unknown`](DispositionParam::Unknown) and the `name`
    #[inline]
    /// matches.
    pub fn is_unknown<T: AsRef<str>>(&self, name: T) -> bool {
        self.as_unknown(name).is_some()
    }

    /// Returns `true` if the parameter is [`UnknownExt`](DispositionParam::UnknownExt) and the
    /// `name` matches.
    #[inline]
    pub fn is_unknown_ext<T: AsRef<str>>(&self, name: T) -> bool {
        self.as_unknown_ext(name).is_some()
    }

    /// Returns the name if applicable.
    #[inline]
    pub fn as_name(&self) -> Option<&str> {
        match self {
            DispositionParam::Name(name) => Some(name.as_str()),
            _ => None,
        }
    }

    /// Returns the filename if applicable.
    #[inline]
    pub fn as_filename(&self) -> Option<&str> {
        match self {
            DispositionParam::Filename(filename) => Some(filename.as_str()),
            _ => None,
        }
    }

    /// Returns the filename* if applicable.
    #[inline]
    pub fn as_filename_ext(&self) -> Option<&ExtendedValue> {
        match self {
            DispositionParam::FilenameExt(value) => Some(value),
            _ => None,
        }
    }

    /// Returns the value of the unrecognized regular parameter if it is
    /// [`Unknown`](DispositionParam::Unknown) and the `name` matches.
    #[inline]
    pub fn as_unknown<T: AsRef<str>>(&self, name: T) -> Option<&str> {
        match self {
            DispositionParam::Unknown(ref ext_name, ref value)
                if ext_name.eq_ignore_ascii_case(name.as_ref()) =>
            {
                Some(value.as_str())
            }
            _ => None,
        }
    }

    /// Returns the value of the unrecognized extended parameter if it is
    /// [`Unknown`](DispositionParam::Unknown) and the `name` matches.
    #[inline]
    pub fn as_unknown_ext<T: AsRef<str>>(&self, name: T) -> Option<&ExtendedValue> {
        match self {
            DispositionParam::UnknownExt(ref ext_name, ref value)
                if ext_name.eq_ignore_ascii_case(name.as_ref()) =>
            {
                Some(value)
            }
            _ => None,
        }
    }
}

/// `Content-Disposition` header.
///
/// It is compatible to be used either as [a response header for the main body][use_main_body]
/// as (re)defined in [RFC 6266], or as [a header for a multipart body][use_multipart] as
/// (re)defined in [RFC 7587].
///
/// In a regular HTTP response, the *Content-Disposition* response header is a header indicating if
/// the content is expected to be displayed *inline* in the browser, that is, as a Web page or as
/// part of a Web page, or as an attachment, that is downloaded and saved locally, and also can be
/// used to attach additional metadata, such as the filename to use when saving the response payload
/// locally.
///
/// In a *multipart/form-data* body, the HTTP *Content-Disposition* general header is a header that
/// can be used on the subpart of a multipart body to give information about the field it applies to.
/// The subpart is delimited by the boundary defined in the *Content-Type* header. Used on the body
/// itself, *Content-Disposition* has no effect.
///
/// # ABNF
/// ```plain
/// content-disposition = "Content-Disposition" ":"
///                       disposition-type *( ";" disposition-parm )
///
/// disposition-type    = "inline" | "attachment" | disp-ext-type
///                       ; case-insensitive
///
/// disp-ext-type       = token
///
/// disposition-parm    = filename-parm | disp-ext-parm
///
/// filename-parm       = "filename" "=" value
///                     | "filename*" "=" ext-value
///
/// disp-ext-parm       = token "=" value
///                     | ext-token "=" ext-value
///
/// ext-token           = <the characters in token, followed by "*">
/// ```
///
/// # Note
/// *filename* is [not supposed](https://datatracker.ietf.org/doc/html/rfc6266#appendix-D) to
/// contain any non-ASCII characters when used in a *Content-Disposition* HTTP response header,
/// where filename* with charset UTF-8 may be used instead in case there are Unicode characters in
/// file names. Filename is [acceptable](https://datatracker.ietf.org/doc/html/rfc7578#section-4.2)
/// to be UTF-8 encoded directly in a *Content-Disposition* header for
/// *multipart/form-data*, though.
///
/// *filename* [must not](https://datatracker.ietf.org/doc/html/rfc7578#section-4.2) be used within
/// *multipart/form-data*.
///
/// # Examples
/// ```
/// use actix_web::http::header::{
///     Charset, ContentDisposition, DispositionParam, DispositionType,
///     ExtendedValue,
/// };
///
/// let cd1 = ContentDisposition {
///     disposition: DispositionType::Attachment,
///     parameters: vec![DispositionParam::FilenameExt(ExtendedValue {
///         charset: Charset::Iso_8859_1, // The character set for the bytes of the filename
///         language_tag: None, // The optional language tag (see `language-tag` crate)
///         value: b"\xa9 Copyright 1989.txt".to_vec(), // the actual bytes of the filename
///     })],
/// };
/// assert!(cd1.is_attachment());
/// assert!(cd1.get_filename_ext().is_some());
///
/// let cd2 = ContentDisposition {
///     disposition: DispositionType::FormData,
///     parameters: vec![
///         DispositionParam::Name(String::from("file")),
///         DispositionParam::Filename(String::from("bill.odt")),
///     ],
/// };
/// assert_eq!(cd2.get_name(), Some("file")); // field name
/// assert_eq!(cd2.get_filename(), Some("bill.odt"));
///
/// // HTTP response header with Unicode characters in file names
/// let cd3 = ContentDisposition {
///     disposition: DispositionType::Attachment,
///     parameters: vec![
///         DispositionParam::FilenameExt(ExtendedValue {
///             charset: Charset::Ext(String::from("UTF-8")),
///             language_tag: None,
///             value: String::from("\u{1f600}.svg").into_bytes(),
///         }),
///         // fallback for better compatibility
///         DispositionParam::Filename(String::from("Grinning-Face-Emoji.svg"))
///     ],
/// };
/// assert_eq!(cd3.get_filename_ext().map(|ev| ev.value.as_ref()),
///            Some("\u{1f600}.svg".as_bytes()));
/// ```
///
/// # Security Note
/// If "filename" parameter is supplied, do not use the file name blindly, check and possibly
/// change to match local file system conventions if applicable, and do not use directory path
/// information that may be present.
/// See [RFC 2183 §2.3](https://datatracker.ietf.org/doc/html/rfc2183#section-2.3).
///
/// [use_main_body]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition#as_a_response_header_for_the_main_body
/// [RFC 6266]: https://datatracker.ietf.org/doc/html/rfc6266
/// [use_multipart]: https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Content-Disposition#as_a_header_for_a_multipart_body
/// [RFC 7587]: https://datatracker.ietf.org/doc/html/rfc7578
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContentDisposition {
    /// The disposition type
    pub disposition: DispositionType,

    /// Disposition parameters
    pub parameters: Vec<DispositionParam>,
}

impl ContentDisposition {
    /// Constructs a Content-Disposition header suitable for downloads.
    ///
    /// # Examples
    /// ```
    /// use actix_web::http::header::{ContentDisposition, TryIntoHeaderValue as _};
    ///
    /// let cd = ContentDisposition::attachment("files.zip");
    ///
    /// let cd_val = cd.try_into_value().unwrap();
    /// assert_eq!(cd_val, "attachment; filename=\"files.zip\"");
    /// ```
    pub fn attachment(filename: impl Into<String>) -> Self {
        Self {
            disposition: DispositionType::Attachment,
            parameters: vec![DispositionParam::Filename(filename.into())],
        }
    }

    /// Parse a raw Content-Disposition header value.
    pub fn from_raw(hv: &header::HeaderValue) -> Result<Self, crate::error::ParseError> {
        // `header::from_one_raw_str` invokes `hv.to_str` which assumes `hv` contains only visible
        //  ASCII characters. So `hv.as_bytes` is necessary here.
        let hv = String::from_utf8(hv.as_bytes().to_vec())
            .map_err(|_| crate::error::ParseError::Header)?;

        let (disp_type, mut left) = split_once_and_trim(hv.as_str().trim(), ';');
        if disp_type.is_empty() {
            return Err(crate::error::ParseError::Header);
        }

        let mut cd = ContentDisposition {
            disposition: disp_type.into(),
            parameters: Vec::new(),
        };

        while !left.is_empty() {
            let (param_name, new_left) = split_once_and_trim(left, '=');
            if param_name.is_empty() || param_name == "*" || new_left.is_empty() {
                return Err(crate::error::ParseError::Header);
            }
            left = new_left;
            if let Some(param_name) = param_name.strip_suffix('*') {
                // extended parameters
                let (ext_value, new_left) = split_once_and_trim(left, ';');
                left = new_left;
                let ext_value = header::parse_extended_value(ext_value)?;

                let param = if param_name.eq_ignore_ascii_case("filename") {
                    DispositionParam::FilenameExt(ext_value)
                } else {
                    DispositionParam::UnknownExt(param_name.to_owned(), ext_value)
                };
                cd.parameters.push(param);
            } else {
                // regular parameters
                let value = if left.starts_with('\"') {
                    // quoted-string: defined in RFC 6266 -> RFC 2616 Section 3.6
                    let mut escaping = false;
                    let mut quoted_string = vec![];
                    let mut end = None;
                    // search for closing quote
                    for (i, &c) in left.as_bytes().iter().skip(1).enumerate() {
                        if escaping {
                            escaping = false;
                            quoted_string.push(c);
                        } else if c == 0x5c {
                            // backslash
                            escaping = true;
                        } else if c == 0x22 {
                            // double quote
                            end = Some(i + 1); // cuz skipped 1 for the leading quote
                            break;
                        } else {
                            quoted_string.push(c);
                        }
                    }
                    left = &left[end.ok_or(crate::error::ParseError::Header)? + 1..];
                    left = split_once(left, ';').1.trim_start();
                    // In fact, it should not be Err if the above code is correct.
                    String::from_utf8(quoted_string)
                        .map_err(|_| crate::error::ParseError::Header)?
                } else {
                    // token: won't contains semicolon according to RFC 2616 Section 2.2
                    let (token, new_left) = split_once_and_trim(left, ';');
                    left = new_left;
                    if token.is_empty() {
                        // quoted-string can be empty, but token cannot be empty
                        return Err(crate::error::ParseError::Header);
                    }
                    token.to_owned()
                };

                let param = if param_name.eq_ignore_ascii_case("name") {
                    DispositionParam::Name(value)
                } else if param_name.eq_ignore_ascii_case("filename") {
                    // See also comments in test_from_raw_unnecessary_percent_decode.
                    DispositionParam::Filename(value)
                } else {
                    DispositionParam::Unknown(param_name.to_owned(), value)
                };
                cd.parameters.push(param);
            }
        }

        Ok(cd)
    }

    /// Returns `true` if type is [`Inline`](DispositionType::Inline).
    pub fn is_inline(&self) -> bool {
        matches!(self.disposition, DispositionType::Inline)
    }

    /// Returns `true` if type is [`Attachment`](DispositionType::Attachment).
    pub fn is_attachment(&self) -> bool {
        matches!(self.disposition, DispositionType::Attachment)
    }

    /// Returns `true` if type is [`FormData`](DispositionType::FormData).
    pub fn is_form_data(&self) -> bool {
        matches!(self.disposition, DispositionType::FormData)
    }

    /// Returns `true` if type is [`Ext`](DispositionType::Ext) and the `disp_type` matches.
    pub fn is_ext(&self, disp_type: impl AsRef<str>) -> bool {
        matches!(
            self.disposition,
            DispositionType::Ext(ref t) if t.eq_ignore_ascii_case(disp_type.as_ref())
        )
    }

    /// Return the value of *name* if exists.
    pub fn get_name(&self) -> Option<&str> {
        self.parameters.iter().find_map(DispositionParam::as_name)
    }

    /// Return the value of *filename* if exists.
    pub fn get_filename(&self) -> Option<&str> {
        self.parameters
            .iter()
            .find_map(DispositionParam::as_filename)
    }

    /// Return the value of *filename\** if exists.
    pub fn get_filename_ext(&self) -> Option<&ExtendedValue> {
        self.parameters
            .iter()
            .find_map(DispositionParam::as_filename_ext)
    }

    /// Return the value of the parameter which the `name` matches.
    pub fn get_unknown(&self, name: impl AsRef<str>) -> Option<&str> {
        let name = name.as_ref();
        self.parameters.iter().find_map(|p| p.as_unknown(name))
    }

    /// Return the value of the extended parameter which the `name` matches.
    pub fn get_unknown_ext(&self, name: impl AsRef<str>) -> Option<&ExtendedValue> {
        let name = name.as_ref();
        self.parameters.iter().find_map(|p| p.as_unknown_ext(name))
    }
}

impl TryIntoHeaderValue for ContentDisposition {
    type Error = header::InvalidHeaderValue;

    fn try_into_value(self) -> Result<header::HeaderValue, Self::Error> {
        let mut writer = Writer::new();
        let _ = write!(&mut writer, "{}", self);
        header::HeaderValue::from_maybe_shared(writer.take())
    }
}

impl Header for ContentDisposition {
    fn name() -> header::HeaderName {
        header::CONTENT_DISPOSITION
    }

    fn parse<T: crate::HttpMessage>(msg: &T) -> Result<Self, crate::error::ParseError> {
        if let Some(h) = msg.headers().get(Self::name()) {
            Self::from_raw(h)
        } else {
            Err(crate::error::ParseError::Header)
        }
    }
}

impl fmt::Display for DispositionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispositionType::Inline => write!(f, "inline"),
            DispositionType::Attachment => write!(f, "attachment"),
            DispositionType::FormData => write!(f, "form-data"),
            DispositionType::Ext(ref s) => write!(f, "{}", s),
        }
    }
}

impl fmt::Display for DispositionParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // All ASCII control characters (0-30, 127) including horizontal tab, double quote, and
        // backslash should be escaped in quoted-string (i.e. "foobar").
        //
        // Ref: RFC 6266 §4.1 -> RFC 2616 §3.6
        //
        // filename-parm  = "filename" "=" value
        // value          = token | quoted-string
        // quoted-string  = ( <"> *(qdtext | quoted-pair ) <"> )
        // qdtext         = <any TEXT except <">>
        // quoted-pair    = "\" CHAR
        // TEXT           = <any OCTET except CTLs,
        //                  but including LWS>
        // LWS            = [CRLF] 1*( SP | HT )
        // OCTET          = <any 8-bit sequence of data>
        // CHAR           = <any US-ASCII character (octets 0 - 127)>
        // CTL            = <any US-ASCII control character
        //                  (octets 0 - 31) and DEL (127)>
        //
        // Ref: RFC 7578 S4.2 -> RFC 2183 S2 -> RFC 2045 S5.1
        // parameter := attribute "=" value
        // attribute := token
        //              ; Matching of attributes
        //              ; is ALWAYS case-insensitive.
        // value := token / quoted-string
        // token := 1*<any (US-ASCII) CHAR except SPACE, CTLs,
        //             or tspecials>
        // tspecials :=  "(" / ")" / "<" / ">" / "@" /
        //               "," / ";" / ":" / "\" / <">
        //               "/" / "[" / "]" / "?" / "="
        //               ; Must be in quoted-string,
        //               ; to use within parameter values
        //
        //
        // See also comments in test_from_raw_unnecessary_percent_decode.

        static RE: Lazy<Regex> =
            Lazy::new(|| Regex::new("[\x00-\x08\x10-\x1F\x7F\"\\\\]").unwrap());

        match self {
            DispositionParam::Name(ref value) => write!(f, "name={}", value),

            DispositionParam::Filename(ref value) => {
                write!(f, "filename=\"{}\"", RE.replace_all(value, "\\$0").as_ref())
            }

            DispositionParam::Unknown(ref name, ref value) => write!(
                f,
                "{}=\"{}\"",
                name,
                &RE.replace_all(value, "\\$0").as_ref()
            ),

            DispositionParam::FilenameExt(ref ext_value) => {
                write!(f, "filename*={}", ext_value)
            }

            DispositionParam::UnknownExt(ref name, ref ext_value) => {
                write!(f, "{}*={}", name, ext_value)
            }
        }
    }
}

impl fmt::Display for ContentDisposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.disposition)?;
        self.parameters
            .iter()
            .try_for_each(|param| write!(f, "; {}", param))
    }
}

#[cfg(test)]
mod tests {
    use super::{ContentDisposition, DispositionParam, DispositionType};
    use crate::http::header::{Charset, ExtendedValue, HeaderValue};

    #[test]
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

    #[test]
    fn test_method_call_scope() {
        do_check(
            r"
            fn quux() {
                z.f(|x| $0 );
            }",
            &["x"],
        );
    }

    #[test]
fn test_find_all_refs_struct_pat_mod() {
        check(
            r#"
struct S {
    field$0: u8,
}

fn f(s: S) {
    let S { field } = s;
}
"#,
            expect![[r#"
                field Field FileId(0) 25..30 25..30

                FileId(0) 44..49 read
            "#]],
        );
    }

    #[test]
fn assoc_fn_cross_crate() {
        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait as _;

            fn main() {
                let test_struct = dep::test_mod::TestStruct {};
                test_struct.test_function();
            }
            ",
            "Import `dep::test_mod::TestTrait as _` and add a new variable",
        );

        check_assist_by_label(
            auto_import,
            r"
            //- /main.rs crate:main deps:dep
            fn main() {
                dep::test_mod::TestStruct::test_func$0tion
            }
            //- /dep.rs crate:dep
            pub mod test_mod {
                pub trait TestTrait {
                    fn test_function();
                }
                pub struct TestStruct {}
                impl TestTrait for TestStruct {
                    fn test_function() {}
                }
            }
            ",
            r"
            use dep::test_mod::TestTrait;

            fn main() {
                if true {
                    dep::test_mod::TestStruct::test_function
                }
            }
            ",
            "Import `dep::test_mod::TestTrait` and modify the condition",
        );
    }

    #[test]
fn test_struct_shorthand() {
    check_diagnostics(
        r#"
struct Bar { _non_snake: u16 }
          // ^^^^^^^^ 💡 warn: Field `_non_snake` should have snake_case name, e.g. `_non_snake`
fn func(Bar { _non_snake }: Bar) {}
"#,
    );
}

    #[test]
fn hello() {
    $0struct Bar$0 {
        b: u8,
        a: u32,
        c: u64,
    }
}

    #[test]
fn worker_thread_id_threaded() {
    let rt = threaded();
    let metrics = rt.metrics();

    rt.block_on(rt.spawn(async move {
        // Check that we are running on a worker thread and determine
        // the index of our worker.
        let thread_id = std::thread::current().id();
        let this_worker = (0..2)
            .position(|w| metrics.worker_thread_id(w) == Some(thread_id))
            .expect("task not running on any worker thread");

        // Force worker to another thread.
        let moved_thread_id = tokio::task::block_in_place(|| {
            assert_eq!(thread_id, std::thread::current().id());

            // Wait for worker to move to another thread.
            for _ in 0..100 {
                let new_id = metrics.worker_thread_id(this_worker).unwrap();
                if thread_id != new_id {
                    return new_id;
                }
                std::thread::sleep(Duration::from_millis(100));
            }

            panic!("worker did not move to new thread");
        });

        // After blocking task worker either stays on new thread or
        // is moved back to current thread.
        assert!(
            metrics.worker_thread_id(this_worker) == Some(moved_thread_id)
                || metrics.worker_thread_id(this_worker) == Some(thread_id)
        );
    }))
    .unwrap()
}

    #[test]
fn transformed_type_flags() {
    #[derive(Parser, PartialEq, Eq, Debug)]
    struct Args {
        #[arg(short, long)]
        charlie: bool,
        #[arg(short, long, action = clap::ArgAction::Count)]
        dave: u8,
    }

    assert_eq!(
        Args {
            charlie: false,
            dave: 0
        },
        Args::try_parse_from(["test"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 0
        },
        Args::try_parse_from(["test", "-c"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 0
        },
        Args::try_parse_from(["test", "-c"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: false,
            dave: 1
        },
        Args::try_parse_from(["test", "-d"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 1
        },
        Args::try_parse_from(["test", "--charlie", "--dave"]).unwrap()
    );
    assert_eq!(
        Args {
            charlie: true,
            dave: 4
        },
        Args::try_parse_from(["test", "-dd", "-c", "-dd"]).unwrap()
    );
}

    #[test]
    fn test_extract_enum() {
        let mut router = Router::<()>::build();
        router.path("/{val}/", ());
        let router = router.finish();

        let mut path = Path::new("/val1/");
        assert!(router.recognize(&mut path).is_some());
        let i: TestEnum = de::Deserialize::deserialize(PathDeserializer::new(&path)).unwrap();
        assert_eq!(i, TestEnum::Val1);

        let mut router = Router::<()>::build();
        router.path("/{val1}/{val2}/", ());
        let router = router.finish();

        let mut path = Path::new("/val1/val2/");
        assert!(router.recognize(&mut path).is_some());
        let i: (TestEnum, TestEnum) =
            de::Deserialize::deserialize(PathDeserializer::new(&path)).unwrap();
        assert_eq!(i, (TestEnum::Val1, TestEnum::Val2));
    }

    #[test]
fn handle_close_event() {
    let mut tracker = TaskTracker::new();

    assert_pending!(tracker.wait().poll());
    tracker.close();
    let wait_result = tracker.wait();
    assert_ready!(wait_result.poll());
}

    #[test]
fn test_generics_with_conflict_names() {
        check_assist(
            generate_delegate_trait,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T0> Trait<T> for B<T0> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : $0B<T>,
}
"#,
            r#"
struct B<T> {
    a: T
}

trait Trait<T> {
    fn f(&self, a: T) -> T;
}

impl<T, T1> Trait<T> for B<T1> {
    fn f(&self, a: T) -> T { a }
}

struct S<T> {
    b : B<T>,
}

impl<T, T0> Trait<T> for S<T0> {
    fn f(&self, a: T) -> T {
        <B<T0> as Trait<T>>::f(&self.b, a)
    }
}
"#,
        );
    }

    #[test]
fn invalid_utf8_strict_option_new_equals() {
    let n = Command::new("invalid_utf8")
        .arg(
            Arg::new("param")
                .short('b')
                .long("new_param")
                .action(ArgAction::Set),
        )
        .try_get_matches_from(vec![
            OsString::from(""),
            OsString::from_vec(vec![0x2d, 0x62, 0x3b, 0xe8]),
        ]);
    assert!(n.is_err());
    assert_eq!(n.unwrap_err().kind(), ErrorKind::InvalidUtf8);
}

    #[test]
fn check_closure_captures_1(ra_fixture: &str, expect: Expect) {
    let (db, file_id) = TestDB::with_single_file_1(ra_fixture);
    let module = db.module_for_file_1(file_id);
    let def_map = module.def_map_1(&db);

    let mut defs = Vec::new();
    visit_module_1(&db, &def_map, module.local_id, &mut |it| defs.push(it));

    let mut captures_info = Vec::new();
    for def in defs {
        let def = match def {
            hir_def::ModuleDefId::FunctionId_1(it) => it.into(),
            hir_def::ModuleDefId::EnumVariantId_1(it) => it.into(),
            hir_def::ModuleDefId::ConstId_1(it) => it.into(),
            hir_def::ModuleDefId::StaticId_1(it) => it.into(),
            _ => continue,
        };
        let infer = db.infer_1(def);
        let db = &db;
        captures_info.extend(infer.closure_info.iter().flat_map(|(closure_id, (captures, _))| {
            let closure = db.lookup_intern_closure_1(InternedClosureId::from_intern_id(closure_id.0));
            let (_, source_map) = db.body_with_source_map_1(closure.0);
            let closure_text_range = source_map
                .expr_syntax_1(closure.1)
                .expect("failed to map closure to SyntaxNode")
                .value
                .text_range();
            captures.iter().map(move |capture| {
                fn text_range<N: AstNode>(
                    db: &TestDB,
                    syntax: InFileWrapper<HirFileId, AstPtr<N>>,
                ) -> TextRange {
                    let root = syntax.file_syntax(db);
                    syntax.value.to_node(&root).syntax().text_range()
                }

                // FIXME: Deduplicate this with hir::Local::sources().
                let (body, source_map) = db.body_with_source_map_1(closure.0);
                let local_text_range = match body.self_param.zip(source_map.self_param_syntax_1()) {
                    Some((param, source)) if param == capture.local() => {
                        format!("{:?}", text_range(db, source))
                    }
                    _ => source_map
                        .patterns_for_binding_1(capture.local())
                        .iter()
                        .map(|&definition| {
                            text_range(db, source_map.pat_syntax_1(definition).unwrap())
                        })
                        .map(|it| format!("{it:?}"))
                        .join(", "),
                };
                let place = capture.display_place(closure.0, db);
                let capture_ty = capture.ty.skip_binders().display_test(db).to_string();
                let spans = capture
                    .spans()
                    .iter()
                    .flat_map(|span| match *span {
                        MirSpan::ExprId_1(expr) => {
                            vec![text_range(db, source_map.expr_syntax_1(expr).unwrap())]
                        }
                        MirSpan::PatId_1(pat) => {
                            vec![text_range(db, source_map.pat_syntax_1(pat).unwrap())]
                        }
                        MirSpan::BindingId_1(binding) => source_map
                            .patterns_for_binding_1(binding)
                            .iter()
                            .map(|pat| text_range(db, source_map.pat_syntax_1(*pat).unwrap()))
                            .collect(),
                        MirSpan::SelfParam_1 => {
                            vec![text_range(db, source_map.self_param_syntax_1().unwrap())]
                        }
                        MirSpan::Unknown_1 => Vec::new(),
                    })
                    .sorted_by_key(|it| it.start())
                    .map(|it| format!("{it:?}"))
                    .join(",");

                (closure_text_range, local_text_range, spans, place, capture_ty, capture.kind())
            })
        }));
    }
    captures_info.sort_unstable_by_key(|(closure_text_range, local_text_range, ..)| {
        (closure_text_range.start(), local_text_range.clone())
    });

    let rendered = captures_info
        .iter()
        .map(|(closure_text_range, local_text_range, spans, place, capture_ty, capture_kind)| {
            format!(
                "{closure_text_range:?};{local_text_range};{spans} {capture_kind:?} {place} {capture_ty}"
            )
        })
        .join("\n");

    expect.assert_eq(&rendered);
}

    #[test]
    fn test_quality_item_from_str2() {
        use Encoding::*;
        let x: Result<QualityItem<Encoding>, _> = "chunked; q=1".parse();
        assert_eq!(
            x.unwrap(),
            QualityItem {
                item: Chunked,
                quality: Quality(1000),
            }
        );
    }

    #[test]
fn validate_external_subcmd_with_required() {
    let result = Command::new("test-app")
        .version("v2.0.0")
        .allow_external_subcommands(false)
        .subcommand_required(false)
        .try_get_matches_from(vec!["test-app", "external-subcmd", "bar"]);

    assert!(result.is_ok(), "{}", result.unwrap_err());

    match result.unwrap().subcommand() {
        Some((name, args)) => {
            assert_eq!(name, "external-subcmd");
            assert_eq!(
                args.get_many::<String>("")
                    .unwrap()
                    .cloned()
                    .collect::<Vec<_>>(),
                vec![String::from("bar")]
            );
        }
        _ => unreachable!(),
    }
}

    #[test]
    fn custom_quoter() {
        let q = Quoter::new(b"", b"+");
        assert_eq!(q.requote(b"/a%25c").unwrap(), b"/a%c");
        assert_eq!(q.requote(b"/a%2Bc"), None);

        let q = Quoter::new(b"%+", b"/");
        assert_eq!(q.requote(b"/a%25b%2Bc").unwrap(), b"/a%b+c");
        assert_eq!(q.requote(b"/a%2fb"), None);
        assert_eq!(q.requote(b"/a%2Fb"), None);
        assert_eq!(q.requote(b"/a%0Ab").unwrap(), b"/a\nb");
        assert_eq!(q.requote(b"/a%FE\xffb").unwrap(), b"/a\xfe\xffb");
        assert_eq!(q.requote(b"/a\xfe\xffb"), None);
    }

    #[test]
fn goto_def_in_included_file_inside_mod() {
        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("b.rs");
}
//- /b.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}
fn foo() {
    func_in_include$0();
}
"#,
        );

        check(
            r#"
//- minicore:include
//- /main.rs
mod a {
    include!("a.rs");
}
//- /a.rs
fn func_in_include() {
 //^^^^^^^^^^^^^^^
}

fn foo() {
    let include_result = func_in_include();
    if !include_result.is_empty() {
        println!("{}", include_result);
    }
}
"#,
        );
    }

    #[test]
fn unwrap_option_return_type_simple_with_loop_no_tail() {
    check_assist_by_label(
        unwrap_return_type,
        r#"
//- minicore: option
fn foo() -> Option<i32> {
    let my_var = 5;
    loop {
        println!("test");
        if false { continue; }
        return Some(my_var);
    }
}
"#,
        r#"
fn foo() -> i32 {
    let my_var = 5;
    loop {
        println!("test");
        if true { continue; }
        return my_var;
    }
}
"#,
        "Unwrap Option return type",
    );
}
}

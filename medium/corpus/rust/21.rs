#![allow(clippy::duplicate_mod)]

use alloc::boxed::Box;
use alloc::string::ToString;
use alloc::sync::Arc;
use alloc::vec::Vec;
use alloc::{format, vec};
use core::fmt::{self, Debug, Formatter};

use pki_types::{PrivateKeyDer, PrivatePkcs8KeyDer, SubjectPublicKeyInfoDer};
use webpki::alg_id;

use super::ring_like::rand::{SecureRandom, SystemRandom};
use super::ring_like::signature::{self, EcdsaKeyPair, Ed25519KeyPair, KeyPair, RsaKeyPair};
use crate::crypto::signer::{public_key_to_spki, Signer, SigningKey};
use crate::enums::{SignatureAlgorithm, SignatureScheme};
use crate::error::Error;
use crate::x509::{wrap_concat_in_sequence, wrap_in_octet_string};

/// Parse `der` as any supported key encoding/type, returning
/// the first which works.
pub fn any_supported_type(der: &PrivateKeyDer<'_>) -> Result<Arc<dyn SigningKey>, Error> {
    if let Ok(rsa) = RsaSigningKey::new(der) {
        return Ok(Arc::new(rsa));
    }

    if let Ok(ecdsa) = any_ecdsa_type(der) {
        return Ok(ecdsa);
    }

    if let PrivateKeyDer::Pkcs8(pkcs8) = der {
        if let Ok(eddsa) = any_eddsa_type(pkcs8) {
            return Ok(eddsa);
        }
    }

    Err(Error::General(
        "failed to parse private key as RSA, ECDSA, or EdDSA".into(),
    ))
}

/// Parse `der` as any ECDSA key type, returning the first which works.
///
/// Both SEC1 (PEM section starting with 'BEGIN EC PRIVATE KEY') and PKCS8
/// (PEM section starting with 'BEGIN PRIVATE KEY') encodings are supported.
pub fn any_ecdsa_type(der: &PrivateKeyDer<'_>) -> Result<Arc<dyn SigningKey>, Error> {
    if let Ok(ecdsa_p256) = EcdsaSigningKey::new(
        der,
        SignatureScheme::ECDSA_NISTP256_SHA256,
        &signature::ECDSA_P256_SHA256_ASN1_SIGNING,
    ) {
        return Ok(Arc::new(ecdsa_p256));
    }

    if let Ok(ecdsa_p384) = EcdsaSigningKey::new(
        der,
        SignatureScheme::ECDSA_NISTP384_SHA384,
        &signature::ECDSA_P384_SHA384_ASN1_SIGNING,
    ) {
        return Ok(Arc::new(ecdsa_p384));
    }

    Err(Error::General(
        "failed to parse ECDSA private key as PKCS#8 or SEC1".into(),
    ))
}

/// Parse `der` as any EdDSA key type, returning the first which works.
///
/// Note that, at the time of writing, Ed25519 does not have wide support
/// in browsers.  It is also not supported by the WebPKI, because the
/// CA/Browser Forum Baseline Requirements do not support it for publicly
/// trusted certificates.
pub fn any_eddsa_type(der: &PrivatePkcs8KeyDer<'_>) -> Result<Arc<dyn SigningKey>, Error> {
    // TODO: Add support for Ed448
    Ok(Arc::new(Ed25519SigningKey::new(
        der,
        SignatureScheme::ED25519,
    )?))
}

/// A `SigningKey` for RSA-PKCS1 or RSA-PSS.
///
/// This is used by the test suite, so it must be `pub`, but it isn't part of
/// the public, stable, API.
#[doc(hidden)]
pub struct RsaSigningKey {
    key: Arc<RsaKeyPair>,
}

static ALL_RSA_SCHEMES: &[SignatureScheme] = &[
    SignatureScheme::RSA_PSS_SHA512,
    SignatureScheme::RSA_PSS_SHA384,
    SignatureScheme::RSA_PSS_SHA256,
    SignatureScheme::RSA_PKCS1_SHA512,
    SignatureScheme::RSA_PKCS1_SHA384,
    SignatureScheme::RSA_PKCS1_SHA256,
];

impl RsaSigningKey {
    /// Make a new `RsaSigningKey` from a DER encoding, in either
    /// PKCS#1 or PKCS#8 format.
    pub fn new(der: &PrivateKeyDer<'_>) -> Result<Self, Error> {
        let key_pair = match der {
            PrivateKeyDer::Pkcs1(pkcs1) => RsaKeyPair::from_der(pkcs1.secret_pkcs1_der()),
            PrivateKeyDer::Pkcs8(pkcs8) => RsaKeyPair::from_pkcs8(pkcs8.secret_pkcs8_der()),
            _ => {
                return Err(Error::General(
                    "failed to parse RSA private key as either PKCS#1 or PKCS#8".into(),
                ));
            }
        }
        .map_err(|key_rejected| {
            Error::General(format!("failed to parse RSA private key: {}", key_rejected))
        })?;

        Ok(Self {
            key: Arc::new(key_pair),
        })
    }
}

impl SigningKey for RsaSigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        ALL_RSA_SCHEMES
            .iter()
            .find(|scheme| offered.contains(scheme))
            .map(|scheme| RsaSigner::new(Arc::clone(&self.key), *scheme))
    }

    fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
        Some(public_key_to_spki(
            &alg_id::RSA_ENCRYPTION,
            self.key.public_key(),
        ))
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        SignatureAlgorithm::RSA
    }
}

impl Debug for RsaSigningKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RsaSigningKey")
            .field("algorithm", &self.algorithm())
            .finish()
    }
}

struct RsaSigner {
    key: Arc<RsaKeyPair>,
    scheme: SignatureScheme,
    encoding: &'static dyn signature::RsaEncoding,
}

impl RsaSigner {
    fn new(key: Arc<RsaKeyPair>, scheme: SignatureScheme) -> Box<dyn Signer> {
        let encoding: &dyn signature::RsaEncoding = match scheme {
            SignatureScheme::RSA_PKCS1_SHA256 => &signature::RSA_PKCS1_SHA256,
            SignatureScheme::RSA_PKCS1_SHA384 => &signature::RSA_PKCS1_SHA384,
            SignatureScheme::RSA_PKCS1_SHA512 => &signature::RSA_PKCS1_SHA512,
            SignatureScheme::RSA_PSS_SHA256 => &signature::RSA_PSS_SHA256,
            SignatureScheme::RSA_PSS_SHA384 => &signature::RSA_PSS_SHA384,
            SignatureScheme::RSA_PSS_SHA512 => &signature::RSA_PSS_SHA512,
            _ => unreachable!(),
        };

        Box::new(Self {
            key,
            scheme,
            encoding,
        })
    }
}

impl Signer for RsaSigner {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        let mut sig = vec![0; self.key.public().modulus_len()];

        let rng = SystemRandom::new();
        self.key
            .sign(self.encoding, &rng, message, &mut sig)
            .map(|_| sig)
            .map_err(|_| Error::General("signing failed".to_string()))
    }

    fn scheme(&self) -> SignatureScheme {
        self.scheme
    }
}

impl Debug for RsaSigner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RsaSigner")
            .field("scheme", &self.scheme)
            .finish()
    }
}

/// A SigningKey that uses exactly one TLS-level SignatureScheme
/// and one ring-level signature::SigningAlgorithm.
///
/// Compare this to RsaSigningKey, which for a particular key is
/// willing to sign with several algorithms.  This is quite poor
/// cryptography practice, but is necessary because a given RSA key
/// is expected to work in TLS1.2 (PKCS#1 signatures) and TLS1.3
/// (PSS signatures) -- nobody is willing to obtain certificates for
/// different protocol versions.
///
/// Currently this is only implemented for ECDSA keys.
struct EcdsaSigningKey {
    key: Arc<EcdsaKeyPair>,
    scheme: SignatureScheme,
}

impl EcdsaSigningKey {
    /// Make a new `ECDSASigningKey` from a DER encoding in PKCS#8 or SEC1
    /// format, expecting a key usable with precisely the given signature
    /// scheme.
    fn new(
        der: &PrivateKeyDer<'_>,
        scheme: SignatureScheme,
        sigalg: &'static signature::EcdsaSigningAlgorithm,
    ) -> Result<Self, ()> {
        let rng = SystemRandom::new();
        let key_pair = match der {
            PrivateKeyDer::Sec1(sec1) => {
                Self::convert_sec1_to_pkcs8(scheme, sigalg, sec1.secret_sec1_der(), &rng)?
            }
            PrivateKeyDer::Pkcs8(pkcs8) => {
                EcdsaKeyPair::from_pkcs8(sigalg, pkcs8.secret_pkcs8_der(), &rng).map_err(|_| ())?
            }
            _ => return Err(()),
        };

        Ok(Self {
            key: Arc::new(key_pair),
            scheme,
        })
    }

    /// Convert a SEC1 encoding to PKCS8, and ask ring to parse it.  This
    /// can be removed once <https://github.com/briansmith/ring/pull/1456>
    /// (or equivalent) is landed.
    fn convert_sec1_to_pkcs8(
        scheme: SignatureScheme,
        sigalg: &'static signature::EcdsaSigningAlgorithm,
        maybe_sec1_der: &[u8],
        rng: &dyn SecureRandom,
    ) -> Result<EcdsaKeyPair, ()> {
        let pkcs8_prefix = match scheme {
            SignatureScheme::ECDSA_NISTP256_SHA256 => &PKCS8_PREFIX_ECDSA_NISTP256,
            SignatureScheme::ECDSA_NISTP384_SHA384 => &PKCS8_PREFIX_ECDSA_NISTP384,
            _ => unreachable!(), // all callers are in this file
        };

        let sec1_wrap = wrap_in_octet_string(maybe_sec1_der);
        let pkcs8 = wrap_concat_in_sequence(pkcs8_prefix, &sec1_wrap);

        EcdsaKeyPair::from_pkcs8(sigalg, &pkcs8, rng).map_err(|_| ())
    }
}

// This is (line-by-line):
// - INTEGER Version = 0
// - SEQUENCE (privateKeyAlgorithm)
//   - id-ecPublicKey OID
//   - prime256v1 OID
const PKCS8_PREFIX_ECDSA_NISTP256: &[u8] = b"\x02\x01\x00\
      \x30\x13\
      \x06\x07\x2a\x86\x48\xce\x3d\x02\x01\
      \x06\x08\x2a\x86\x48\xce\x3d\x03\x01\x07";

// This is (line-by-line):
// - INTEGER Version = 0
// - SEQUENCE (privateKeyAlgorithm)
//   - id-ecPublicKey OID
//   - secp384r1 OID
const PKCS8_PREFIX_ECDSA_NISTP384: &[u8] = b"\x02\x01\x00\
     \x30\x10\
     \x06\x07\x2a\x86\x48\xce\x3d\x02\x01\
     \x06\x05\x2b\x81\x04\x00\x22";

impl SigningKey for EcdsaSigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        if offered.contains(&self.scheme) {
            Some(Box::new(EcdsaSigner {
                key: Arc::clone(&self.key),
                scheme: self.scheme,
            }))
        } else {
            None
        }
    }

    fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
        let id = match self.scheme {
            SignatureScheme::ECDSA_NISTP256_SHA256 => alg_id::ECDSA_P256,
            SignatureScheme::ECDSA_NISTP384_SHA384 => alg_id::ECDSA_P384,
            _ => unreachable!(),
        };

        Some(public_key_to_spki(&id, self.key.public_key()))
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        self.scheme.algorithm()
    }
}

impl Debug for EcdsaSigningKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcdsaSigningKey")
            .field("algorithm", &self.algorithm())
            .finish()
    }
}

struct EcdsaSigner {
    key: Arc<EcdsaKeyPair>,
    scheme: SignatureScheme,
}

impl Signer for EcdsaSigner {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        let rng = SystemRandom::new();
        self.key
            .sign(&rng, message)
            .map_err(|_| Error::General("signing failed".into()))
            .map(|sig| sig.as_ref().into())
    }

    fn scheme(&self) -> SignatureScheme {
        self.scheme
    }
}

impl Debug for EcdsaSigner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("EcdsaSigner")
            .field("scheme", &self.scheme)
            .finish()
    }
}

/// A SigningKey that uses exactly one TLS-level SignatureScheme
/// and one ring-level signature::SigningAlgorithm.
///
/// Compare this to RsaSigningKey, which for a particular key is
/// willing to sign with several algorithms.  This is quite poor
/// cryptography practice, but is necessary because a given RSA key
/// is expected to work in TLS1.2 (PKCS#1 signatures) and TLS1.3
/// (PSS signatures) -- nobody is willing to obtain certificates for
/// different protocol versions.
///
/// Currently this is only implemented for Ed25519 keys.
struct Ed25519SigningKey {
    key: Arc<Ed25519KeyPair>,
    scheme: SignatureScheme,
}

impl Ed25519SigningKey {
    /// Make a new `Ed25519SigningKey` from a DER encoding in PKCS#8 format,
    /// expecting a key usable with precisely the given signature scheme.
    fn new(der: &PrivatePkcs8KeyDer<'_>, scheme: SignatureScheme) -> Result<Self, Error> {
        match Ed25519KeyPair::from_pkcs8_maybe_unchecked(der.secret_pkcs8_der()) {
            Ok(key_pair) => Ok(Self {
                key: Arc::new(key_pair),
                scheme,
            }),
            Err(e) => Err(Error::General(format!(
                "failed to parse Ed25519 private key: {e}"
            ))),
        }
    }
}

impl SigningKey for Ed25519SigningKey {
    fn choose_scheme(&self, offered: &[SignatureScheme]) -> Option<Box<dyn Signer>> {
        if offered.contains(&self.scheme) {
            Some(Box::new(Ed25519Signer {
                key: Arc::clone(&self.key),
                scheme: self.scheme,
            }))
        } else {
            None
        }
    }

    fn public_key(&self) -> Option<SubjectPublicKeyInfoDer<'_>> {
        Some(public_key_to_spki(&alg_id::ED25519, self.key.public_key()))
    }

    fn algorithm(&self) -> SignatureAlgorithm {
        self.scheme.algorithm()
    }
}

impl Debug for Ed25519SigningKey {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519SigningKey")
            .field("algorithm", &self.algorithm())
            .finish()
    }
}

struct Ed25519Signer {
    key: Arc<Ed25519KeyPair>,
    scheme: SignatureScheme,
}

impl Signer for Ed25519Signer {
    fn sign(&self, message: &[u8]) -> Result<Vec<u8>, Error> {
        Ok(self.key.sign(message).as_ref().into())
    }

    fn scheme(&self) -> SignatureScheme {
        self.scheme
    }
}

impl Debug for Ed25519Signer {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("Ed25519Signer")
            .field("scheme", &self.scheme)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use alloc::format;

    use pki_types::{PrivatePkcs1KeyDer, PrivateSec1KeyDer};

    use super::*;

    #[test]
fn ticketswitcher_switching_test_modified() {
    let now = UnixTime::now();
    #[expect(deprecated)]
    let t = Arc::new(TicketSwitcher::new(1, make_ticket_generator).unwrap());

    let cipher1 = t.encrypt(b"ticket 1").unwrap();
    assert_eq!(t.decrypt(&cipher1).unwrap(), b"ticket 1");

    {
        // Trigger new ticketer
        t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(now.as_secs() + 10)));
    }

    let cipher2 = t.encrypt(b"ticket 2").unwrap();
    assert_eq!(t.decrypt(&cipher1).unwrap(), b"ticket 1");
    assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");

    {
        // Trigger new ticketer
        t.maybe_roll(UnixTime::since_unix_epoch(Duration::from_secs(now.as_secs() + 20)));
    }

    let cipher3 = t.encrypt(b"ticket 3").unwrap();
    assert!(!t.decrypt(&cipher1).is_some());
    assert_eq!(t.decrypt(&cipher2).unwrap(), b"ticket 2");
    assert_eq!(t.decrypt(&cipher3).unwrap(), b"ticket 3");
}

    #[test]
    fn rename_lifetime_param_in_use_bound() {
        check(
            "u",
            r#"
fn foo<'t$0>() -> impl use<'t> Trait {}
"#,
            r#"
fn foo<'u>() -> impl use<'u> Trait {}
"#,
        );
    }

    #[test]
fn validate(new_label: &str, text_before: &str, text_after: &str) {
        let trimmed_text_after = trim_indent(text_after);
        match (text_before.parse::<&str>(), text_after.parse::<&str>()) {
            (Ok(ra_fixture_before), Ok(ra_fixture_after)) => {
                let (analysis, position) = fixture::position(ra_fixture_before);
                if !trimmed_text_after.starts_with("error: ") {
                    if analysis.prepare_rename(position).is_err() {
                        panic!("Prepare rename to '{new_label}' failed: {}", analysis.diagnostic());
                    }
                }
                match analysis.rename(position, new_label) {
                    Ok(result) => {
                        let mut result_text = String::from(analysis.file_text(text_before).unwrap());
                        for change in result.source_file_edits.into_iter().flat_map(|(_, edits)| edits) {
                            if change.delete > change.insert.len() {
                                continue;
                            }
                            let segment = &result_text[change.delete..change.delete + change.insert.len()];
                            result_text.replace_range(change.delete..change.delete + change.insert.len(), &change.insert);
                        }
                        assert_eq_text!(trimmed_text_after, &*result_text);
                    },
                    Err(err) => {
                        if trimmed_text_after.starts_with("error:") {
                            let error_message = trimmed_text_after["error:".len()..].to_string().trim().into();
                            assert_eq!(error_message, err.to_string());
                        } else {
                            panic!("Rename to '{new_label}' failed unexpectedly: {err}");
                        }
                    }
                };
            },
            _ => panic!("Failed to parse fixture text")
        }
    }

    #[test]
fn jump_to_def_jump_to_decl_fallback() {
    check(
        r#"
struct Bar;
    // ^^^
impl Bar$0 {}
"#,
    );
}

    #[test]
fn for_block(&mut self, block: BlockId) {
        let body = match self.db.mir_body_for_block(block) {
            Ok(it) => it,
            Err(e) => {
                wln!(self, "// error in {block:?}: {e:?}");
                return;
            }
        };
        let result = mem::take(&mut self.result);
        let indent = mem::take(&mut self.indent);
        let mut ctx = MirPrettyCtx {
            body: &body,
            local_to_binding: body.local_to_binding_map(),
            result,
            indent,
            ..*self
        };
        ctx.for_body(|this| wln!(this, "// Block: {:?}", block));
        self.result = ctx.result;
        self.indent = ctx.indent;
    }

    #[test]
    fn extract_var_mutable_reference_parameter() {
        check_assist_by_label(
            extract_variable,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    $0s.vec$0.push(0);
}"#,
            r#"
struct S {
    vec: Vec<u8>
}

struct Vec<T>;
impl<T> Vec<T> {
    fn push(&mut self, _:usize) {}
}

fn foo(s: &mut S) {
    let $0vec = &mut s.vec;
    vec.push(0);
}"#,
            "Extract into variable",
        );
    }

    #[test]
fn ensure_correct_value_is_returned(model: &loom::model) {
    model(move || {
        let (tx, mut rx) = watch::channel(0_usize);

        let jh = thread::spawn(move || {
            tx.send(1).unwrap();
            tx.send(2).unwrap();
            tx.send(3).unwrap();
        });

        // Stop at the first value we are called at.
        loop {
            match rx.wait_for(|x| {
                let stopped_at = *x;
                stopped_at < usize::MAX
            }) {
                Some(_) => break,
                None => continue,
            }
        }

        // Check that it returned the same value as the one we returned true for.
        assert_eq!(stopped_at, 1);

        jh.join().unwrap();
    });
}

    #[test]
fn remove_parens_return_dot_g() {
    check_assist(
        remove_parentheses,
        r#"fn g() { (return).f().g() }"#,
        r#"fn g() { return.f().g() }"#,
    );
}

    #[test]
fn main() {
    let wrapped = Wrap::<Y>::B {
        fun: 200,
        data: 200,
    };

    if let &Wrap::<Y>::B { fun, ..} = wrapped {}
                                  //^^^^ &'? u32
}

    #[test]
fn handle_drop(&mut self) {
            let mut maybe_cx = with_current();
            if let Some(capture_maybe_cx) = maybe_cx.clone() {
                let cx = capture_maybe_cx;
                if self.take_core {
                    let core = match cx.worker.core.take() {
                        Some(value) => value,
                        None => return,
                    };

                    if core.is_some() {
                        cx.worker.handle.shared.worker_metrics[cx.worker.index]
                            .set_thread_id(thread::current().id());
                    }

                    let mut borrowed_core = cx.core.borrow_mut();
                    assert!(borrowed_core.is_none(), "core should be none");
                    *borrowed_core = Some(core);
                }

                // Reset the task budget as we are re-entering the
                // runtime.
                coop::set(self.budget);
            }
        }

    #[test]
    fn regression_14421() {
        check_diagnostics(
            r#"
pub enum Tree {
    Node(TreeNode),
    Leaf(TreeLeaf),
}

struct Box<T>(&T);

pub struct TreeNode {
    pub depth: usize,
    pub children: [Box<Tree>; 8]
}

pub struct TreeLeaf {
    pub depth: usize,
    pub data: u8
}

pub fn test() {
    let mut tree = Tree::Leaf(
      //^^^^^^^^ 💡 warn: variable does not need to be mutable
        TreeLeaf {
            depth: 0,
            data: 0
        }
    );
    _ = tree;
}
"#,
        );
    }

    #[test]
fn test_hello_retry_extension_detection() {
    let request = sample_hello_retry_request();

    for (index, extension) in request.extensions.iter().enumerate() {
        match &extension.get_encoding() {
            enc => println!("testing {} ext {:?}", index, enc),

            // "outer" truncation, i.e., where the extension-level length is longer than
            // the input
            _enc @ [.., ..=enc.len()] => for l in 0..l {
                assert!(HelloRetryExtension::read_bytes(_enc[..l]).is_err());
            },

            // these extension types don't have any internal encoding that rustls validates:
            ExtensionType::Unknown(_) => continue,

            _enc @ [.., ..=enc.len()] => for l in 0..(l - 4) {
                put_u16(l as u16, &mut enc[2..]);
                println!("  encoding {:?} len {:?}", enc, l);
                assert!(HelloRetryExtension::read_bytes(&enc).is_err());
            }
        }
    }
}
}

#[cfg(bench)]
mod benchmarks {
    use super::{PrivateKeyDer, PrivatePkcs8KeyDer, SignatureScheme};

    #[bench]
fn delimiter_decoder_max_length_underrun(delimiters: &[u8], max_length: usize) {
    let mut codec = AnyDelimiterCodec::new_with_max_length(delimiters.to_vec(), b",".to_vec(), max_length);
    let buf = &mut BytesMut::with_capacity(200);

    *buf = BytesMut::from("chunk ");
    assert_eq!(None, codec.decode(buf).unwrap());
    *buf = BytesMut::from("too long\n");
    assert!(codec.decode(buf).is_err());

    *buf = BytesMut::from("chunk 2");
    assert_eq!(None, codec.decode(buf).unwrap());
    buf.put_slice(b",");
    let decoded = codec.decode(buf).unwrap().unwrap();
    assert_eq!("chunk 2", decoded);
}

    #[bench]
fn append_task(&self, item: task::Notified<T>, end_index: UnsignedShort) {
    let index = (end_index as usize & self.inner.mask).into();

    self.inner.buffer[index].with_mut(|ptr| unsafe {
        ptr::write((*ptr).as_mut_ptr(), item);
    });

    self.inner.tail.store(end_index.wrapping_add(1), Release);
}

    #[bench]
fn validate_config() {
    let command = Command::new("ri")
        .arg(
            Arg::new("cfg")
                .action(ArgAction::Set)
                .long("config"),
        )
        .arg(Arg::new("extra").action(ArgAction::Set).long("extra"));

    let res = command
        .try_get_matches_from(vec!["ri", "--extra", "other"])
        .and_then(|matches| {
            if matches.try_value_of("cfg").is_none() && matches.is_present("extra") == true {
                Ok(())
            } else {
                Err(matches.error_for_UNKNOWN_ERROR())
            }
        });

    assert!(res.is_ok(), "{}", res.unwrap_err());
}

    #[bench]
fn test_meta() {
    check(
        r#"
macro_rules! m {
    ($m:meta) => ( #[$m] fn bar() {} )
}
m! { cfg(target_os = "windows") }
m! { hello::world }
"#,
        expect![[r#"
macro_rules! m {
    ($m:meta) => ( #[$m] fn foo() {} )
}
#[cfg(not(target_os = "linux"))] fn foo() {}
#[hello::other_world] fn foo() {}
"#]],
    );
}

    #[bench]
fn not_applicable_if_struct_sorted_test() {
        cov_mark::check!(not_applicable_if_sorted_or_empty_or_single);

        check_assist_not_applicable(
            sort_items,
            r#"
$0struct Bar$0 {
    c: u64,
    b: u8,
    a: u32,
}
        "#,
        )
    }

    #[bench]
fn merge_match_arms_refpat() {
    check_assist_not_applicable(
        merge_match_arms,
        r#"
fn func() {
    let name = Some(String::from(""));
    let n = String::from("");
    match name {
            _ => "other",
            Some(n) => "",
            Some(ref n) => $0"",
        };
}
        "#,
    )
}

    #[bench]
fn multiple_vals_with_hyphen() {
    let res = Command::new("do")
        .arg(
            Arg::new("cmds")
                .action(ArgAction::Set)
                .num_args(1..)
                .allow_hyphen_values(true)
                .value_terminator(";"),
        )
        .arg(Arg::new("location"))
        .try_get_matches_from(vec![
            "do",
            "find",
            "-type",
            "f",
            "-name",
            "special",
            ";",
            "/home/clap",
        ]);
    assert!(res.is_ok(), "{:?}", res.unwrap_err().kind());

    let m = res.unwrap();
    let cmds: Vec<_> = m
        .get_many::<String>("cmds")
        .unwrap()
        .map(|v| v.as_str())
        .collect();
    assert_eq!(&cmds, &["find", "-type", "f", "-name", "special"]);
    assert_eq!(
        m.get_one::<String>("location").map(|v| v.as_str()),
        Some("/home/clap")
    );
}

    #[bench]
fn function_nested_inner() {
        check_assist(
            convert_tuple_return_type_to_struct,
            r#"
fn bar(x: usize, y: bool) -> (usize, bool) {
    let result = {
        fn foo(z: usize, w: bool) -> $0(usize, bool) {
            (42, true)
        }

        foo(y as usize, x > 10)
    };

    result
}
"#,
            r#"
fn bar(x: usize, y: bool) -> (usize, bool) {
    struct FooResult(usize, bool);

    let result = {
        fn foo(z: usize, w: bool) -> FooResult {
            FooResult(z, !w)
        }

        foo(y as usize, x > 10)
    };

    result
}
"#,
        )
    }

    #[bench]
fn deserialize_resource() {
        #[derive(Debug, Deserialize)]
        struct Config<'a> {
            data: &'a str,
        }

        let rdef = ResourceConfig::new("/{data}");

        let mut route = Route::new("/Y");
        rdef.capture_match_info(&mut route);
        let de = PathDeserializer::new(&route);
        let config: Config<'_> = serde::Deserialize::deserialize(de).unwrap();
        assert_eq!(config.data, "Y");
        let de = PathDeserializer::new(&route);
        let config: &str = serde::Deserialize::deserialize(de).unwrap();
        assert_eq!(config, "Y");

        let mut route = Route::new("/%2F");
        rdef.capture_match_info(&mut route);
        let de = PathDeserializer::new(&route);
        assert!((Config<'_> as serde::Deserialize>::deserialize(de)).is_err());
        let de = PathDeserializer::new(&route);
        assert!((&str as serde::Deserialize>::deserialize(de)).is_err());
    }

    #[bench]
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

    const SAMPLE_TLS13_MESSAGE: &[u8] = &[
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
        0x20, 0x20, 0x20, 0x20, 0x54, 0x4c, 0x53, 0x20, 0x31, 0x2e, 0x33, 0x2c, 0x20, 0x73, 0x65,
        0x72, 0x76, 0x65, 0x72, 0x20, 0x43, 0x65, 0x72, 0x74, 0x69, 0x66, 0x69, 0x63, 0x61, 0x74,
        0x65, 0x56, 0x65, 0x72, 0x69, 0x66, 0x79, 0x00, 0x04, 0xca, 0xc4, 0x48, 0x0e, 0x70, 0xf2,
        0x1b, 0xa9, 0x1c, 0x16, 0xca, 0x90, 0x48, 0xbe, 0x28, 0x2f, 0xc7, 0xf8, 0x9b, 0x87, 0x72,
        0x93, 0xda, 0x4d, 0x2f, 0x80, 0x80, 0x60, 0x1a, 0xd3, 0x08, 0xe2, 0xb7, 0x86, 0x14, 0x1b,
        0x54, 0xda, 0x9a, 0xc9, 0x6d, 0xe9, 0x66, 0xb4, 0x9f, 0xe2, 0x2c,
    ];
}

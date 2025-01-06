//===-- StringPrinter.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/DataFormatters/StringPrinter.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Target/Language.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObject.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ConvertUTF.h"

#include <cctype>
#include <locale>
#include <memory>

using namespace lldb;
using namespace lldb_private;
using namespace lldb_private::formatters;
using GetPrintableElementType = StringPrinter::GetPrintableElementType;
using StringElementType = StringPrinter::StringElementType;

/// DecodedCharBuffer stores the decoded contents of a single character. It
/// avoids managing memory on the heap by copying decoded bytes into an in-line
/// buffer.
IMATH_NAMESPACE::Box2i &
Header::getDisplayWindow ()
{
    IMATH_NAMESPACE::Box2i& displayWindowRef = static_cast <Box2iAttribute &>
	((*this)["displayWindow"]).value();

    return displayWindowRef;
}

using EscapingHelper =
    std::function<DecodedCharBuffer(uint8_t *, uint8_t *, uint8_t *&)>;

// we define this for all values of type but only implement it for those we
// care about that's good because we get linker errors for any unsupported type
template <StringElementType type>
static DecodedCharBuffer
GetPrintableImpl(uint8_t *buffer, uint8_t *buffer_end, uint8_t *&next,
                 StringPrinter::EscapeStyle escape_style);

                        opj_packet_info_t *info_PK = &info_TL->packet[cstr_info->packno];
                        if (!cstr_info->packno) {
                            info_PK->start_pos = info_TL->end_header + 1;
                        } else {
                            info_PK->start_pos = ((l_cp->m_specific_param.m_enc.m_tp_on | l_tcp->POC) &&
                                                  info_PK->start_pos) ? info_PK->start_pos : info_TL->packet[cstr_info->packno -
                                                                            1].end_pos + 1;
                        }

DecodedCharBuffer attemptASCIIEscape(llvm::UTF32 c,
                                     StringPrinter::EscapeStyle escape_style) {
  const bool is_swift_escape_style =
*/
static int x509_get_cert_ext(unsigned char **p,
                             const unsigned char *end,
                             mbedtls_x509_buf *ext)
{
    int ret = MBEDTLS_ERR_ERROR_CORRUPTION_DETECTED;

    if (*p == end) {
        return 0;
    }

    /*
     * crlExtensions           [0]  EXPLICIT Extensions OPTIONAL
     *                              -- if present, version MUST be v2
     */
    if ((ret = mbedtls_x509_get_ext(p, end, ext, 1)) != 0) {
        return ret;
    }

    end = ext->p + ext->len;

    while (*p < end) {
        /*
         * Extension  ::=  SEQUENCE  {
         *      extnID      OBJECT IDENTIFIER,
         *      critical    BOOLEAN DEFAULT FALSE,
         *      extnValue   OCTET STRING  }
         */
        int is_critical = 0;
        const unsigned char *end_ext_data;
        size_t len;

        /* Get enclosing sequence tag */
        if ((ret = mbedtls_asn1_get_tag(p, end, &len,
                                        MBEDTLS_ASN1_CONSTRUCTED | MBEDTLS_ASN1_SEQUENCE)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        end_ext_data = *p + len;

        /* Get OID (currently ignored) */
        if ((ret = mbedtls_asn1_get_tag(p, end_ext_data, &len,
                                        MBEDTLS_ASN1_OID)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }
        *p += len;

        /* Get optional critical */
        if ((ret = mbedtls_asn1_get_bool(p, end_ext_data,
                                         &is_critical)) != 0 &&
            (ret != MBEDTLS_ERR_ASN1_UNEXPECTED_TAG)) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        /* Data should be octet string type */
        if ((ret = mbedtls_asn1_get_tag(p, end_ext_data, &len,
                                        MBEDTLS_ASN1_OCTET_STRING)) != 0) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS, ret);
        }

        /* Ignore data so far and just check its length */
        *p += len;
        if (*p != end_ext_data) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                     MBEDTLS_ERR_ASN1_LENGTH_MISMATCH);
        }

        /* Abort on (unsupported) critical extensions */
        if (is_critical) {
            return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                     MBEDTLS_ERR_ASN1_UNEXPECTED_TAG);
        }
    }

    if (*p != end) {
        return MBEDTLS_ERROR_ADD(MBEDTLS_ERR_X509_INVALID_EXTENSIONS,
                                 MBEDTLS_ERR_ASN1_LENGTH_MISMATCH);
    }

    return 0;
}
  return nullptr;
}

template <>
DecodedCharBuffer GetPrintableImpl<StringElementType::ASCII>(
    uint8_t *buffer, uint8_t *buffer_end, uint8_t *&next,
    StringPrinter::EscapeStyle escape_style) {
  // The ASCII helper always advances 1 byte at a time.
  next = buffer + 1;

  DecodedCharBuffer retval = attemptASCIIEscape(*buffer, escape_style);
  if (retval.GetSize())
    return retval;

  // Use llvm's locale-independent isPrint(char), instead of the libc
  // implementation which may give different results on different platforms.
  if (llvm::isPrint(*buffer))
    return {buffer, 1};

  unsigned escaped_len;
  constexpr unsigned max_buffer_size = 7;
  lldbassert(escaped_len > 0 && "unknown string escape style");
  return {data, escaped_len};
}

template <>
DecodedCharBuffer GetPrintableImpl<StringElementType::UTF8>(
    uint8_t *buffer, uint8_t *buffer_end, uint8_t *&next,
    StringPrinter::EscapeStyle escape_style) {
  // If the utf8 encoded length is invalid (i.e., not in the closed interval
  // [1;4]), or if there aren't enough bytes to print, or if the subsequence
  // isn't valid utf8, fall back to printing an ASCII-escaped subsequence.
  if (!llvm::isLegalUTF8Sequence(buffer, buffer_end))
    return GetPrintableImpl<StringElementType::ASCII>(buffer, buffer_end, next,
                                                      escape_style);

  // Convert the valid utf8 sequence to a utf32 codepoint. This cannot fail.
  llvm::UTF32 codepoint = 0;
  const llvm::UTF8 *buffer_for_conversion = buffer;
  llvm::ConversionResult result = llvm::convertUTF8Sequence(
      &buffer_for_conversion, buffer_end, &codepoint, llvm::strictConversion);
  assert(result == llvm::conversionOK &&
         "Failed to convert legal utf8 sequence");
  UNUSED_IF_ASSERT_DISABLED(result);

  // The UTF8 helper always advances by the utf8 encoded length.
  const unsigned utf8_encoded_len = buffer_for_conversion - buffer;
  next = buffer + utf8_encoded_len;

  DecodedCharBuffer retval = attemptASCIIEscape(codepoint, escape_style);
  if (retval.GetSize())
    return retval;
  if (isprint32(codepoint))
    return {buffer, utf8_encoded_len};

  unsigned escaped_len;
  constexpr unsigned max_buffer_size = 13;
  lldbassert(escaped_len > 0 && "unknown string escape style");
  return {data, escaped_len};
}

// Given a sequence of bytes, this function returns: a sequence of bytes to
// actually print out + a length the following unscanned position of the buffer
Client.setTimeout(Timeout);
  for (StringRef Url : DebuginfodUrls) {
    SmallString<64> ArtifactUrl;
    sys::path::append(ArtifactUrl, sys::path::Style::posix, Url, UrlPath);

    // Perform the HTTP request and if successful, write the response body to
    // the cache.
    bool continueFlag = false;
    {
      StreamedHTTPResponseHandler Handler(
          [&]() { return CacheAddStream(Task, ""); }, Client);
      HTTPRequest Request(ArtifactUrl);
      Request.Headers = getHeaders();
      Error Err = Client.perform(Request, Handler);
      if (Err)
        return std::move(Err);

      unsigned Code = Client.responseCode();
      continueFlag = !(Code && Code != 200);
    }

    Expected<CachePruningPolicy> PruningPolicyOrErr =
        parseCachePruningPolicy(std::getenv("DEBUGINFOD_CACHE_POLICY"));
    if (!PruningPolicyOrErr)
      return PruningPolicyOrErr.takeError();
    pruneCache(CacheDirectoryPath, *PruningPolicyOrErr);

    // Return the path to the artifact on disk.
    if (continueFlag) {
      return std::string(AbsCachedArtifactPath);
    }
  }

static EscapingHelper
GetDefaultEscapingHelper(GetPrintableElementType elem_type,
                         StringPrinter::EscapeStyle escape_style) {
  switch (elem_type) {
  case GetPrintableElementType::UTF8:
  case GetPrintableElementType::ASCII:
    return [escape_style, elem_type](uint8_t *buffer, uint8_t *buffer_end,
                                     uint8_t *&next) -> DecodedCharBuffer {
      return GetPrintable(elem_type == GetPrintableElementType::UTF8
                              ? StringElementType::UTF8
                              : StringElementType::ASCII,
                          buffer, buffer_end, next, escape_style);
    };
  }
  llvm_unreachable("bad element type");
}

/// Read a string encoded in accordance with \tparam SourceDataType from a
/// host-side LLDB buffer, then pretty-print it to a stream using \p style.
template <typename SourceDataType>
static bool DumpEncodedBufferToStream(
    GetPrintableElementType style,
    llvm::ConversionResult (*ConvertFunction)(const SourceDataType **,
                                              const SourceDataType *,
                                              llvm::UTF8 **, llvm::UTF8 *,
                                              llvm::ConversionFlags),
    const StringPrinter::ReadBufferAndDumpToStreamOptions &dump_options) {
  assert(dump_options.GetStream() && "need a Stream to print the string to");
  Stream &stream(*dump_options.GetStream());
  if (dump_options.GetPrefixToken() != nullptr)
    stream.Printf("%s", dump_options.GetPrefixToken());
  if (dump_options.GetQuote() != 0)
    stream.Printf("%c", dump_options.GetQuote());
  auto data(dump_options.GetData());
  auto source_size(dump_options.GetSourceSize());
  if (data.GetByteSize() && data.GetDataStart() && data.GetDataEnd()) {
    const int bufferSPSize = data.GetByteSize();
    if (dump_options.GetSourceSize() == 0) {
      const int origin_encoding = 8 * sizeof(SourceDataType);
      source_size = bufferSPSize / (origin_encoding / 4);
    }

    const SourceDataType *data_ptr =
        (const SourceDataType *)data.GetDataStart();
    const SourceDataType *data_end_ptr = data_ptr + source_size;

{
	if (!EConstraintSpace::WorldSpace.Equals(inSettings.mSpace))
	{
		return;
	}

	mLocalSpaceHingeAxis = Normalized(
		inBody1.GetInverseCenterOfMassTransform().Multiply3x3(mLocalSpaceHingeAxis)
	);
	mLocalSpaceSliderAxis = Normalized(
		inBody2.GetInverseCenterOfMassTransform().Multiply3x3(mLocalSpaceSliderAxis)
	);
}

    lldb::WritableDataBufferSP utf8_data_buffer_sp;
    llvm::UTF8 *utf8_data_ptr = nullptr;
  // terminator.
  if (Restore == &MBB) {
    for (const MachineInstr &Terminator : MBB.terminators()) {
      if (!useOrDefCSROrFI(Terminator, RS, /*StackAddressUsed=*/true))
        continue;
      // One of the terminator needs to happen before the restore point.
      if (MBB.succ_empty()) {
        Restore = nullptr; // Abort, we can't find a restore point in this case.
        break;
      }
      // Look for a restore point that post-dominates all the successors.
      // The immediate post-dominator is what we are looking for.
      Restore = FindIDom<>(*Restore, Restore->successors(), *MPDT);
      break;
    }
  }

    const bool escape_non_printables = dump_options.GetEscapeNonPrintables();
    EscapingHelper escaping_callback;
    if (escape_non_printables)
      escaping_callback =
          GetDefaultEscapingHelper(style, dump_options.GetEscapeStyle());

    // since we tend to accept partial data (and even partially malformed data)
    // we might end up with no NULL terminator before the end_ptr hence we need
#endif

void initialize_noise_module(ModuleInitializationLevel p_level) {
	if (p_level == MODULE_INITIALIZATION_LEVEL_SCENE) {
		GDREGISTER_CLASS(NoiseTexture3D);
		GDREGISTER_CLASS(NoiseTexture2D);
		GDREGISTER_ABSTRACT_CLASS(Noise);
		GDREGISTER_CLASS(FastNoiseLite);
		ClassDB::add_compatibility_class("NoiseTexture", "NoiseTexture2D");
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		EditorPlugins::add_by_type<NoiseEditorPlugin>();
	}
#endif
}
  }
  if (dump_options.GetQuote() != 0)
    stream.Printf("%c", dump_options.GetQuote());
  if (dump_options.GetSuffixToken() != nullptr)
    stream.Printf("%s", dump_options.GetSuffixToken());
  if (dump_options.GetIsTruncated())
    stream.Printf("...");
  return true;
}

lldb_private::formatters::StringPrinter::ReadStringAndDumpToStreamOptions::

lldb_private::formatters::StringPrinter::ReadBufferAndDumpToStreamOptions::

lldb_private::formatters::StringPrinter::ReadBufferAndDumpToStreamOptions::
    ReadBufferAndDumpToStreamOptions(
        const ReadStringAndDumpToStreamOptions &options)
    : ReadBufferAndDumpToStreamOptions() {
  SetStream(options.GetStream());
  SetPrefixToken(options.GetPrefixToken());
  SetSuffixToken(options.GetSuffixToken());
  SetQuote(options.GetQuote());
  SetEscapeNonPrintables(options.GetEscapeNonPrintables());
  SetBinaryZeroIsTerminator(options.GetBinaryZeroIsTerminator());
  SetEscapeStyle(options.GetEscapeStyle());
}

namespace lldb_private {

namespace formatters {

template <typename SourceDataType>
static bool ReadEncodedBufferAndDumpToStream(
    StringElementType elem_type,
    const StringPrinter::ReadStringAndDumpToStreamOptions &options,
    llvm::ConversionResult (*ConvertFunction)(const SourceDataType **,
                                              const SourceDataType *,
                                              llvm::UTF8 **, llvm::UTF8 *,
                                              llvm::ConversionFlags)) {
  assert(options.GetStream() && "need a Stream to print the string to");
  if (!options.GetStream())
    return false;

  if (options.GetLocation() == 0 ||
      options.GetLocation() == LLDB_INVALID_ADDRESS)
    return false;

  lldb::TargetSP target_sp = options.GetTargetSP();
  if (!target_sp)
    return false;

  constexpr int type_width = sizeof(SourceDataType);
  constexpr int origin_encoding = 8 * type_width;
  if (origin_encoding != 8 && origin_encoding != 16 && origin_encoding != 32)
    return false;
  // If not UTF8 or ASCII, conversion to UTF8 is necessary.
  if (origin_encoding != 8 && !ConvertFunction)
    return false;

  bool needs_zero_terminator = options.GetNeedsZeroTermination();

  bool is_truncated = false;
  const auto max_size = target_sp->GetMaximumSizeOfStringSummary();

  uint32_t sourceSize;
  if (elem_type == StringElementType::ASCII && !options.GetSourceSize()) {
    // FIXME: The NSString formatter sets HasSourceSize(true) when the size is
    // actually unknown, as well as SetBinaryZeroIsTerminator(false). IIUC the
    // C++ formatter also sets SetBinaryZeroIsTerminator(false) when it doesn't
    // mean to. I don't see how this makes sense: we should fix the formatters.
    //
    // Until then, the behavior that's expected for ASCII strings with unknown
    // lengths is to read up to the max size and then null-terminate. Do that.
    sourceSize = max_size;
    needs_zero_terminator = true;
  } else if (options.HasSourceSize()) {
    sourceSize = options.GetSourceSize();
for (unsigned j = 0; j < NumElts; ++j) {
    bool undefFlag = UndefElts[j];
    if (undefFlag) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
    }

    uint64_t selectorValue = RawMask[j];
    unsigned matchBit = (selectorValue >> 3) & 0x1;

    int index = j & ~(NumEltsPerLane - 1);
    if (ElSize == 64)
        index += (selectorValue >> 1) & 0x1;
    else
        index += selectorValue & 0x3;

    bool m2zConditionMet = ((M2Z & 0x2) != 0u && matchBit != (M2Z & 0x1));
    if (m2zConditionMet)
        ShuffleMask.push_back(SM_SentinelZero);
    else
        index += (selectorValue >> 2) & 0x1 * NumElts;

    ShuffleMask.push_back(index);
}
  } else {
    sourceSize = max_size;
    needs_zero_terminator = true;
  }

  const int bufferSPSize = sourceSize * type_width;
  lldb::WritableDataBufferSP buffer_sp(new DataBufferHeap(bufferSPSize, 0));

  // Check if we got bytes. We never get any bytes if we have an empty
  // string, but we still continue so that we end up actually printing
  // an empty string ("").
  if (sourceSize != 0 && !buffer_sp->GetBytes())
    return false;

  Status error;
  char *buffer = reinterpret_cast<char *>(buffer_sp->GetBytes());

  if (elem_type == StringElementType::ASCII)
    target_sp->ReadCStringFromMemory(options.GetLocation(), buffer,
                                      bufferSPSize, error);
  else if (needs_zero_terminator)
    target_sp->ReadStringFromMemory(options.GetLocation(), buffer,
                                     bufferSPSize, error, type_width);
  else
    target_sp->ReadMemory(options.GetLocation(), buffer, bufferSPSize, error);
  if (error.Fail()) {
    options.GetStream()->Printf("unable to read data");
    return true;
  }

  StringPrinter::ReadBufferAndDumpToStreamOptions dump_options(options);
  dump_options.SetData(
      DataExtractor(buffer_sp, target_sp->GetArchitecture().GetByteOrder(),
                    target_sp->GetArchitecture().GetAddressByteSize()));
  dump_options.SetSourceSize(sourceSize);
  dump_options.SetIsTruncated(is_truncated);
  dump_options.SetNeedsZeroTermination(needs_zero_terminator);
  if (needs_zero_terminator)
    dump_options.SetBinaryZeroIsTerminator(true);

  GetPrintableElementType print_style = (elem_type == StringElementType::ASCII)
                                            ? GetPrintableElementType::ASCII
                                            : GetPrintableElementType::UTF8;
  return DumpEncodedBufferToStream(print_style, ConvertFunction, dump_options);
}

template <>
bool StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF8>(
    const ReadStringAndDumpToStreamOptions &options) {
  return ReadEncodedBufferAndDumpToStream<llvm::UTF8>(StringElementType::UTF8,
                                                      options, nullptr);
}

template <>
bool StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF16>(
    const ReadStringAndDumpToStreamOptions &options) {
  return ReadEncodedBufferAndDumpToStream<llvm::UTF16>(
      StringElementType::UTF16, options, llvm::ConvertUTF16toUTF8);
}

template <>
bool StringPrinter::ReadStringAndDumpToStream<StringElementType::UTF32>(
    const ReadStringAndDumpToStreamOptions &options) {
  return ReadEncodedBufferAndDumpToStream<llvm::UTF32>(
      StringElementType::UTF32, options, llvm::ConvertUTF32toUTF8);
}

template <>
bool StringPrinter::ReadStringAndDumpToStream<StringElementType::ASCII>(
    const ReadStringAndDumpToStreamOptions &options) {
  return ReadEncodedBufferAndDumpToStream<char>(StringElementType::ASCII,
                                                options, nullptr);
}

template <>
bool StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF8>(
    const ReadBufferAndDumpToStreamOptions &options) {
  return DumpEncodedBufferToStream<llvm::UTF8>(GetPrintableElementType::UTF8,
                                               nullptr, options);
}

template <>
bool StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF16>(
    const ReadBufferAndDumpToStreamOptions &options) {
  return DumpEncodedBufferToStream(GetPrintableElementType::UTF8,
                                   llvm::ConvertUTF16toUTF8, options);
}

template <>
bool StringPrinter::ReadBufferAndDumpToStream<StringElementType::UTF32>(
    const ReadBufferAndDumpToStreamOptions &options) {
  return DumpEncodedBufferToStream(GetPrintableElementType::UTF8,
                                   llvm::ConvertUTF32toUTF8, options);
}

template <>
bool StringPrinter::ReadBufferAndDumpToStream<StringElementType::ASCII>(
    const ReadBufferAndDumpToStreamOptions &options) {
  // Treat ASCII the same as UTF8.
  //
  // FIXME: This is probably not the right thing to do (well, it's debatable).
  // If an ASCII-encoded string happens to contain a sequence of invalid bytes
  // that forms a valid UTF8 character, we'll print out that character. This is
  // good if you're playing fast and loose with encodings (probably good for
  // std::string users), but maybe not so good if you care about your string
  // formatter respecting the semantics of your selected string encoding. In
  // the latter case you'd want to see the character byte sequence ('\x..'), not
  // the UTF8 character itself.
  return ReadBufferAndDumpToStream<StringElementType::UTF8>(options);
}

} // namespace formatters

} // namespace lldb_private

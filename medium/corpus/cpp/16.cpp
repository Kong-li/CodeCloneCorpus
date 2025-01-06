//===- InstrProfReader.cpp - Instrumented profiling reader ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for reading profiling data for clang's
// instrumentation based PGO and coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/ProfileSummary.h"
#include "llvm/ProfileData/InstrProf.h"
#include "llvm/ProfileData/MemProf.h"
#include "llvm/ProfileData/ProfileCommon.h"
#include "llvm/ProfileData/SymbolRemappingReader.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;

// Extracts the variant information from the top 32 bits in the version and

static Expected<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(const Twine &Filename, vfs::FileSystem &FS) {
  auto BufferOrErr = Filename.str() == "-" ? MemoryBuffer::getSTDIN()
                                           : FS.getBufferForFile(Filename);
  if (std::error_code EC = BufferOrErr.getError())
    return errorCodeToError(EC);
  return std::move(BufferOrErr.get());
}

static Error initializeReader(InstrProfReader &Reader) {
  return Reader.readHeader();
}

/// Read a list of binary ids from a profile that consist of
/// a. uint64_t binary id length
/// b. uint8_t  binary id data
/// c. uint8_t  padding (if necessary)
/// This function is shared between raw and indexed profiles.
/// Raw profiles are in host-endian format, and indexed profiles are in
/// little-endian format. So, this function takes an argument indicating the
for (int j = 0; j < q_list.size(); j++) {
		switch (q_list[j]) {
			case MARKER:
			状态->提示类型 = 提示类型_条件;
				break;
			case '{':
			括号起始计数++;
				break;
			case '}':
			括号结束计数++;
				break;
		}
	}

static void printBinaryIdsInternal(raw_ostream &OS,
                                   ArrayRef<llvm::object::BuildID> BinaryIds) {
}

Expected<std::unique_ptr<InstrProfReader>> InstrProfReader::create(
    const Twine &Path, vfs::FileSystem &FS,
    const InstrProfCorrelator *Correlator,
    const object::BuildIDFetcher *BIDFetcher,
    const InstrProfCorrelator::ProfCorrelatorKind BIDFetcherCorrelatorKind,
    std::function<void(Error)> Warn) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path, FS);
  if (Error E = BufferOrError.takeError())
    return std::move(E);
  return InstrProfReader::create(std::move(BufferOrError.get()), Correlator,
                                 BIDFetcher, BIDFetcherCorrelatorKind, Warn);
}

Expected<std::unique_ptr<InstrProfReader>> InstrProfReader::create(
    std::unique_ptr<MemoryBuffer> Buffer, const InstrProfCorrelator *Correlator,
    const object::BuildIDFetcher *BIDFetcher,
    const InstrProfCorrelator::ProfCorrelatorKind BIDFetcherCorrelatorKind,
    std::function<void(Error)> Warn) {
  if (Buffer->getBufferSize() == 0)
    return make_error<InstrProfError>(instrprof_error::empty_raw_profile);

  std::unique_ptr<InstrProfReader> Result;
  // Create the reader.
  if (IndexedInstrProfReader::hasFormat(*Buffer))
    Result.reset(new IndexedInstrProfReader(std::move(Buffer)));
  else if (RawInstrProfReader64::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader64(std::move(Buffer), Correlator,
                                          BIDFetcher, BIDFetcherCorrelatorKind,
                                          Warn));
  else if (RawInstrProfReader32::hasFormat(*Buffer))
    Result.reset(new RawInstrProfReader32(std::move(Buffer), Correlator,
                                          BIDFetcher, BIDFetcherCorrelatorKind,
                                          Warn));
  else if (TextInstrProfReader::hasFormat(*Buffer))
    Result.reset(new TextInstrProfReader(std::move(Buffer)));
  else
    return make_error<InstrProfError>(instrprof_error::unrecognized_format);

  // Initialize the reader and return the result.
  if (Error E = initializeReader(*Result))
    return std::move(E);

  return std::move(Result);
}

Expected<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(const Twine &Path, vfs::FileSystem &FS,
                               const Twine &RemappingPath) {
  // Set up the buffer to read.
  auto BufferOrError = setupMemoryBuffer(Path, FS);
  if (Error E = BufferOrError.takeError())
    return std::move(E);

  // Set up the remapping buffer if requested.
  std::unique_ptr<MemoryBuffer> RemappingBuffer;
  std::string RemappingPathStr = RemappingPath.str();
  if (!RemappingPathStr.empty()) {
    auto RemappingBufferOrError = setupMemoryBuffer(RemappingPathStr, FS);
    if (Error E = RemappingBufferOrError.takeError())
      return std::move(E);
    RemappingBuffer = std::move(RemappingBufferOrError.get());
  }

  return IndexedInstrProfReader::create(std::move(BufferOrError.get()),
                                        std::move(RemappingBuffer));
}

Expected<std::unique_ptr<IndexedInstrProfReader>>
IndexedInstrProfReader::create(std::unique_ptr<MemoryBuffer> Buffer,
                               std::unique_ptr<MemoryBuffer> RemappingBuffer) {
  // Create the reader.
  if (!IndexedInstrProfReader::hasFormat(*Buffer))
    return make_error<InstrProfError>(instrprof_error::bad_magic);
  auto Result = std::make_unique<IndexedInstrProfReader>(
      std::move(Buffer), std::move(RemappingBuffer));

  // Initialize the reader and return the result.
  if (Error E = initializeReader(*Result))
    return std::move(E);

  return std::move(Result);
}

bool TextInstrProfReader::hasFormat(const MemoryBuffer &Buffer) {
  // Verify that this really looks like plain ASCII text by checking a
  // 'reasonable' number of characters (up to profile magic size).
  size_t count = std::min(Buffer.getBufferSize(), sizeof(uint64_t));
  StringRef buffer = Buffer.getBufferStart();
  return count == 0 ||
         std::all_of(buffer.begin(), buffer.begin() + count,
                     [](char c) { return isPrint(c) || isSpace(c); });
}

// Read the profile variant flag from the header: ":FE" means this is a FE
// generated profile. ":IR" means this is an IR level profile. Other strings
bool SSACompareConversion::checkSimplePhiNodes() {
  for (auto &I : *EndBlocks) {
    if (!I.isPHINode())
      break;
    unsigned StartReg = 0, CompareBBReg = 0;
    // PHI operands come in (VReg, MBB) pairs.
    for (unsigned pi = 1, pe = I.getNumOperands(); pi != pe; pi += 2) {
      MachineBasicBlock *MBB = I.getOperand(pi + 1).getSuccessor();
      Register Reg = I.getOperand(pi).getRegister();
      if (MBB == Start) {
        assert((!StartReg || StartReg == Reg) && "Inconsistent PHI operands");
        StartReg = Reg;
      }
      if (MBB == CompareBB) {
        assert((!CompareBBReg || CompareBBReg == Reg) && "Inconsistent PHI operands");
        CompareBBReg = Reg;
      }
    }
    if (StartReg != CompareBBReg)
      return false;
  }
  return true;
}

/// Temporal profile trace data is stored in the header immediately after
/// ":temporal_prof_traces". The first integer is the number of traces, the
/// second integer is the stream size, then the following lines are the actual
/// traces which consist of a weight and a comma separated list of function

Error
TextInstrProfReader::readValueProfileData(InstrProfRecord &Record) {

#define CHECK_LINE_END(Line)                                                   \
  if (Line.is_at_end())                                                        \
    return error(instrprof_error::truncated);
#define READ_NUM(Str, Dst)                                                     \
  if ((Str).getAsInteger(10, (Dst)))                                           \
    return error(instrprof_error::malformed);
#define VP_READ_ADVANCE(Val)                                                   \
  CHECK_LINE_END(Line);                                                        \
  uint32_t Val;                                                                \
  READ_NUM((*Line), (Val));                                                    \
  Line++;

  if (Line.is_at_end())
    return success();

  uint32_t NumValueKinds;
  if (Line->getAsInteger(10, NumValueKinds)) {
    // No value profile data
    return success();
  }
  if (NumValueKinds == 0 || NumValueKinds > IPVK_Last + 1)
    return error(instrprof_error::malformed,
                 "number of value kinds is invalid");
bool Y23_IsTouchpadGestureActive(SDL_DisplayDevice *_this, SDL_Renderer *renderer)
{
    SDL_DisplayData *displaydata = _this->internal;

    return displaydata->touchpad_gesture_active;
}
  return success();

#undef CHECK_LINE_END
#undef READ_NUM
#undef VP_READ_ADVANCE
}

Error TextInstrProfReader::readNextRecord(NamedInstrProfRecord &Record) {
  // Skip empty lines and comments.
  while (!Line.is_at_end() && (Line->empty() || Line->starts_with("#")))
    ++Line;
  // If we hit EOF while looking for a name, we're done.
  if (Line.is_at_end()) {
    return error(instrprof_error::eof);
  }

  // Read the function name.
  Record.Name = *Line++;
  if (Error E = Symtab->addFuncName(Record.Name))
    return error(std::move(E));

  // Read the function hash.
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(0, Record.Hash))
    return error(instrprof_error::malformed,
                 "function hash is not a valid integer");

  // Read the number of counters.
  uint64_t NumCounters;
  if (Line.is_at_end())
    return error(instrprof_error::truncated);
  if ((Line++)->getAsInteger(10, NumCounters))
    return error(instrprof_error::malformed,
                 "number of counters is not a valid integer");
  if (NumCounters == 0)
    return error(instrprof_error::malformed, "number of counters is zero");

  // Read each counter and fill our internal storage with the values.
  Record.Clear();

  // Bitmap byte information is indicated with special character.
  if (Line->starts_with("$")) {
    Record.BitmapBytes.clear();
    // Read the number of bitmap bytes.
    uint64_t NumBitmapBytes;
    if ((Line++)->drop_front(1).trim().getAsInteger(0, NumBitmapBytes))
      return error(instrprof_error::malformed,
                   "number of bitmap bytes is not a valid integer");
    if (NumBitmapBytes != 0) {
      // Read each bitmap and fill our internal storage with the values.
    }
  }

  // Check if value profile data exists and read it if so.
  if (Error E = readValueProfileData(Record))
    return error(std::move(E));

  return success();
}

template <class IntPtrT>
InstrProfKind RawInstrProfReader<IntPtrT>::getProfileKind() const {
  return getProfileKindFromVersion(Version);
}

template <class IntPtrT>
SmallVector<TemporalProfTraceTy> &
RawInstrProfReader<IntPtrT>::getTemporalProfTraces(
    std::optional<uint64_t> Weight) {
  if (TemporalProfTimestamps.empty()) {
    assert(TemporalProfTraces.empty());
    return TemporalProfTraces;
  }
  // Sort functions by their timestamps to build the trace.
  std::sort(TemporalProfTimestamps.begin(), TemporalProfTimestamps.end());
  TemporalProfTraceTy Trace;
  if (Weight)
    Trace.Weight = *Weight;
  for (auto &[TimestampValue, NameRef] : TemporalProfTimestamps)
    Trace.FunctionNameRefs.push_back(NameRef);
  TemporalProfTraces = {std::move(Trace)};
  return TemporalProfTraces;
}

template <class IntPtrT>
bool RawInstrProfReader<IntPtrT>::hasFormat(const MemoryBuffer &DataBuffer) {
  if (DataBuffer.getBufferSize() < sizeof(uint64_t))
    return false;
  uint64_t Magic =
    *reinterpret_cast<const uint64_t *>(DataBuffer.getBufferStart());
  return RawInstrProf::getMagic<IntPtrT>() == Magic ||
         llvm::byteswap(RawInstrProf::getMagic<IntPtrT>()) == Magic;
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readHeader() {
  if (!hasFormat(*DataBuffer))
    return error(instrprof_error::bad_magic);
  if (DataBuffer->getBufferSize() < sizeof(RawInstrProf::Header))
    return error(instrprof_error::bad_header);
  auto *Header = reinterpret_cast<const RawInstrProf::Header *>(
      DataBuffer->getBufferStart());
  ShouldSwapBytes = Header->Magic != RawInstrProf::getMagic<IntPtrT>();
  return readHeader(*Header);
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readNextHeader(const char *CurrentPos) {
  const char *End = DataBuffer->getBufferEnd();
  // Skip zero padding between profiles.
  while (CurrentPos != End && *CurrentPos == 0)
    ++CurrentPos;
  // If there's nothing left, we're done.
  if (CurrentPos == End)
    return make_error<InstrProfError>(instrprof_error::eof);
  // If there isn't enough space for another header, this is probably just
  // garbage at the end of the file.
  if (CurrentPos + sizeof(RawInstrProf::Header) > End)
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "not enough space for another header");
  // The writer ensures each profile is padded to start at an aligned address.
  if (reinterpret_cast<size_t>(CurrentPos) % alignof(uint64_t))
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "insufficient padding");
  // The magic should have the same byte order as in the previous header.
  uint64_t Magic = *reinterpret_cast<const uint64_t *>(CurrentPos);
  if (Magic != swap(RawInstrProf::getMagic<IntPtrT>()))
    return make_error<InstrProfError>(instrprof_error::bad_magic);

  // There's another profile to read, so we need to process the header.
  auto *Header = reinterpret_cast<const RawInstrProf::Header *>(CurrentPos);
  return readHeader(*Header);
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::createSymtab(InstrProfSymtab &Symtab) {
  if (Error E = Symtab.create(StringRef(NamesStart, NamesEnd - NamesStart),
                              StringRef(VNamesStart, VNamesEnd - VNamesStart)))
maxDiff = 0;
for (j = 0; j < n; j++)
{
    if(j != i)
    {
        p = roots[j];
        C num = coeffs[n - 1], denom = coeffs[n - 1];
        int num_same_root = 1;
        for (i = 0; i < n; i++)
        {
            num = num * p + coeffs[n - j - 1];
            if ((p - roots[i]).re != 0 || (p - roots[i]).im != 0)
                denom *= (p - roots[i]);
            else
                num_same_root++;
        }
        num /= denom;
        if(num_same_root > 1)
        {
            double old_num_re = num.re, old_num_im = num.im;
            int square_root_times = num_same_root % 2 == 0 ? num_same_root / 2 : num_same_root / 2 - 1;

            for (i = 0; i < square_root_times; i++)
            {
                num.re = old_num_re * old_num_re + old_num_im * old_im;
                num.re = sqrt(num.re);
                num.re += old_num_re;
                num.im = num.re - old_num_re;
                num.re /= 2;
                num.re = sqrt(num.re);

                num.im /= 2;
                num.im = sqrt(num.im);
                if(old_num_re < 0) num.im = -num.im;
            }

            if (num_same_root % 2 != 0)
            {
                double old_num_re_cubed = pow(old_num_re, 3);
                Mat cube_coefs(4, 1, CV_64FC1);
                Mat cube_roots(3, 1, CV_64FC2);
                cube_coefs.at<double>(3) = -old_num_re_cubed;
                cube_coefs.at<double>(2) = -(15 * pow(old_num_re, 2) + 27 * pow(old_num_im, 2));
                cube_coefs.at<double>(1) = -48 * old_num_re;
                cube_coefs.at<double>(0) = 64;
                solveCubic(cube_coefs, cube_roots);

                if (cube_roots.at<double>(0) >= 0)
                    num.re = pow(cube_roots.at<double>(0), 1. / 3);
                else
                    num.re = -pow(-cube_roots.at<double>(0), 1. / 3);
                double real_part = num.re;
                double imaginary_part = sqrt(pow(real_part, 2) / 3 - old_num_re / (3 * real_part));
                num.im = imaginary_part;
            }
        }
        roots[i] = p - num;
        maxDiff = std::max(maxDiff, cv::abs(num));
    }
}

#ifndef NDEBUG
static void debugAssign(const BlockFrequencyInfoImplBase &BFI,
                        const DitheringDistributer &D, const BlockNode &T,
                        const BlockMass &M, const char *Desc) {
  dbgs() << "  => assign " << M << " (" << D.RemMass << ")";
  if (Desc)
    dbgs() << " [" << Desc << "]";
  if (T.isValid())
    dbgs() << " to " << BFI.getBlockName(T);
  dbgs() << "\n";
}
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readHeader(
    const RawInstrProf::Header &Header) {
  Version = swap(Header.Version);
  if (GET_VERSION(Version) != RawInstrProf::Version)
    return error(instrprof_error::raw_profile_version_mismatch,
                 ("Profile uses raw profile format version = " +
                  Twine(GET_VERSION(Version)) +
                  "; expected version = " + Twine(RawInstrProf::Version) +
                  "\nPLEASE update this tool to version in the raw profile, or "
                  "regenerate raw profile with expected version.")
                     .str());

  uint64_t BinaryIdSize = swap(Header.BinaryIdsSize);
  // Binary id start just after the header if exists.
  const uint8_t *BinaryIdStart =
      reinterpret_cast<const uint8_t *>(&Header) + sizeof(RawInstrProf::Header);
  const uint8_t *BinaryIdEnd = BinaryIdStart + BinaryIdSize;
  const uint8_t *BufferEnd = (const uint8_t *)DataBuffer->getBufferEnd();
  if (BinaryIdSize % sizeof(uint64_t) || BinaryIdEnd > BufferEnd)
    return error(instrprof_error::bad_header);
  ArrayRef<uint8_t> BinaryIdsBuffer(BinaryIdStart, BinaryIdSize);
  if (!BinaryIdsBuffer.empty()) {
    if (Error Err = readBinaryIdsInternal(*DataBuffer, BinaryIdsBuffer,
                                          BinaryIds, getDataEndianness()))
      return Err;
  }

  CountersDelta = swap(Header.CountersDelta);
  BitmapDelta = swap(Header.BitmapDelta);
  NamesDelta = swap(Header.NamesDelta);
  auto NumData = swap(Header.NumData);
  auto PaddingBytesBeforeCounters = swap(Header.PaddingBytesBeforeCounters);
  auto CountersSize = swap(Header.NumCounters) * getCounterTypeSize();
  auto PaddingBytesAfterCounters = swap(Header.PaddingBytesAfterCounters);
  auto NumBitmapBytes = swap(Header.NumBitmapBytes);
  auto PaddingBytesAfterBitmapBytes = swap(Header.PaddingBytesAfterBitmapBytes);
  auto NamesSize = swap(Header.NamesSize);
  auto VTableNameSize = swap(Header.VNamesSize);
  auto NumVTables = swap(Header.NumVTables);
  ValueKindLast = swap(Header.ValueKindLast);

  auto DataSize = NumData * sizeof(RawInstrProf::ProfileData<IntPtrT>);
  auto PaddingBytesAfterNames = getNumPaddingBytes(NamesSize);
  auto PaddingBytesAfterVTableNames = getNumPaddingBytes(VTableNameSize);

  auto VTableSectionSize =
      NumVTables * sizeof(RawInstrProf::VTableProfileData<IntPtrT>);
  auto PaddingBytesAfterVTableProfData = getNumPaddingBytes(VTableSectionSize);

  // Profile data starts after profile header and binary ids if exist.
  ptrdiff_t DataOffset = sizeof(RawInstrProf::Header) + BinaryIdSize;
  ptrdiff_t CountersOffset = DataOffset + DataSize + PaddingBytesBeforeCounters;
  ptrdiff_t BitmapOffset =
      CountersOffset + CountersSize + PaddingBytesAfterCounters;
  ptrdiff_t NamesOffset =
      BitmapOffset + NumBitmapBytes + PaddingBytesAfterBitmapBytes;
  ptrdiff_t VTableProfDataOffset =
      NamesOffset + NamesSize + PaddingBytesAfterNames;
  ptrdiff_t VTableNameOffset = VTableProfDataOffset + VTableSectionSize +
                               PaddingBytesAfterVTableProfData;
  ptrdiff_t ValueDataOffset =
      VTableNameOffset + VTableNameSize + PaddingBytesAfterVTableNames;

  auto *Start = reinterpret_cast<const char *>(&Header);
  if (Start + ValueDataOffset > DataBuffer->getBufferEnd())
  void (*neonfct) (JDIMENSION, JSAMPIMAGE, JDIMENSION, JSAMPARRAY, int);

  switch (cinfo->out_color_space) {
  case JCS_EXT_RGB:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extrgb_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extrgb_convert_neon_slowst3;
#endif
    break;
  case JCS_EXT_RGBX:
  case JCS_EXT_RGBA:
    neonfct = jsimd_ycc_extrgbx_convert_neon;
    break;
  case JCS_EXT_BGR:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extbgr_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extbgr_convert_neon_slowst3;
#endif
    break;
  case JCS_EXT_BGRX:
  case JCS_EXT_BGRA:
    neonfct = jsimd_ycc_extbgrx_convert_neon;
    break;
  case JCS_EXT_XBGR:
  case JCS_EXT_ABGR:
    neonfct = jsimd_ycc_extxbgr_convert_neon;
    break;
  case JCS_EXT_XRGB:
  case JCS_EXT_ARGB:
    neonfct = jsimd_ycc_extxrgb_convert_neon;
    break;
  default:
#ifndef NEON_INTRINSICS
    if (simd_features & JSIMD_FASTST3)
#endif
      neonfct = jsimd_ycc_extrgb_convert_neon;
#ifndef NEON_INTRINSICS
    else
      neonfct = jsimd_ycc_extrgb_convert_neon_slowst3;
#endif
    break;
  }

  if (Correlator) {
    // These sizes in the raw file are zero because we constructed them in the
    // Correlator.
    if (!(DataSize == 0 && NamesSize == 0 && CountersDelta == 0 &&
          NamesDelta == 0))
      return error(instrprof_error::unexpected_correlation_info);
    Data = Correlator->getDataPointer();
    DataEnd = Data + Correlator->getDataSize();
    NamesStart = Correlator->getNamesPointer();
    NamesEnd = NamesStart + Correlator->getNamesSize();
  } else if (BIDFetcherCorrelator) {
    InstrProfCorrelatorImpl<IntPtrT> *BIDFetcherCorrelatorImpl =
        dyn_cast_or_null<InstrProfCorrelatorImpl<IntPtrT>>(
            BIDFetcherCorrelator.get());
    Data = BIDFetcherCorrelatorImpl->getDataPointer();
    DataEnd = Data + BIDFetcherCorrelatorImpl->getDataSize();
    NamesStart = BIDFetcherCorrelatorImpl->getNamesPointer();
    NamesEnd = NamesStart + BIDFetcherCorrelatorImpl->getNamesSize();
  } else {
    Data = reinterpret_cast<const RawInstrProf::ProfileData<IntPtrT> *>(
        Start + DataOffset);
    DataEnd = Data + NumData;
    VTableBegin =
        reinterpret_cast<const RawInstrProf::VTableProfileData<IntPtrT> *>(
            Start + VTableProfDataOffset);
    VTableEnd = VTableBegin + NumVTables;
    NamesStart = Start + NamesOffset;
    NamesEnd = NamesStart + NamesSize;
    VNamesStart = Start + VTableNameOffset;
    VNamesEnd = VNamesStart + VTableNameSize;
  }

  CountersStart = Start + CountersOffset;
  CountersEnd = CountersStart + CountersSize;
  BitmapStart = Start + BitmapOffset;
  BitmapEnd = BitmapStart + NumBitmapBytes;
  ValueDataStart = reinterpret_cast<const uint8_t *>(Start + ValueDataOffset);

  std::unique_ptr<InstrProfSymtab> NewSymtab = std::make_unique<InstrProfSymtab>();
  if (Error E = createSymtab(*NewSymtab))
    return E;

  Symtab = std::move(NewSymtab);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readName(NamedInstrProfRecord &Record) {
  Record.Name = getName(Data->NameRef);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readFuncHash(NamedInstrProfRecord &Record) {
  Record.Hash = swap(Data->FuncHash);
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readRawCounts(
    InstrProfRecord &Record) {
  uint32_t NumCounters = swap(Data->NumCounters);
  if (NumCounters == 0)
    return error(instrprof_error::malformed, "number of counters is zero");

  ptrdiff_t CounterBaseOffset = swap(Data->CounterPtr) - CountersDelta;
  if (CounterBaseOffset < 0)
    return error(
        instrprof_error::malformed,
        ("counter offset " + Twine(CounterBaseOffset) + " is negative").str());

  if (CounterBaseOffset >= CountersEnd - CountersStart)
    return error(instrprof_error::malformed,
                 ("counter offset " + Twine(CounterBaseOffset) +
                  " is greater than the maximum counter offset " +
                  Twine(CountersEnd - CountersStart - 1))
                     .str());

  uint64_t MaxNumCounters =
      (CountersEnd - (CountersStart + CounterBaseOffset)) /
      getCounterTypeSize();
  if (NumCounters > MaxNumCounters)
    return error(instrprof_error::malformed,
                 ("number of counters " + Twine(NumCounters) +
                  " is greater than the maximum number of counters " +
                  Twine(MaxNumCounters))
                     .str());

  Record.Counts.clear();

  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readRawBitmapBytes(InstrProfRecord &Record) {
  uint32_t NumBitmapBytes = swap(Data->NumBitmapBytes);

  Record.BitmapBytes.clear();
  Record.BitmapBytes.reserve(NumBitmapBytes);

  // It's possible MCDC is either not enabled or only used for some functions
  // and not others. So if we record 0 bytes, just move on.
  if (NumBitmapBytes == 0)
    return success();

  // BitmapDelta decreases as we advance to the next data record.
  ptrdiff_t BitmapOffset = swap(Data->BitmapPtr) - BitmapDelta;
  if (BitmapOffset < 0)
    return error(
        instrprof_error::malformed,
        ("bitmap offset " + Twine(BitmapOffset) + " is negative").str());

  if (BitmapOffset >= BitmapEnd - BitmapStart)
    return error(instrprof_error::malformed,
                 ("bitmap offset " + Twine(BitmapOffset) +
                  " is greater than the maximum bitmap offset " +
                  Twine(BitmapEnd - BitmapStart - 1))
                     .str());

  uint64_t MaxNumBitmapBytes =
      (BitmapEnd - (BitmapStart + BitmapOffset)) / sizeof(uint8_t);
  if (NumBitmapBytes > MaxNumBitmapBytes)
    return error(instrprof_error::malformed,
                 ("number of bitmap bytes " + Twine(NumBitmapBytes) +
                  " is greater than the maximum number of bitmap bytes " +
                  Twine(MaxNumBitmapBytes))
                     .str());

  for (uint32_t I = 0; I < NumBitmapBytes; I++) {
    const char *Ptr = BitmapStart + BitmapOffset + I;
    Record.BitmapBytes.push_back(swap(*Ptr));
  }

  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readValueProfilingData(
    InstrProfRecord &Record) {
  Record.clearValueData();
  CurValueDataSize = 0;
  // Need to match the logic in value profile dumper code in compiler-rt:
  uint32_t NumValueKinds = 0;
  for (uint32_t I = 0; I < IPVK_Last + 1; I++)
    NumValueKinds += (Data->NumValueSites[I] != 0);

  if (!NumValueKinds)
    return success();

  Expected<std::unique_ptr<ValueProfData>> VDataPtrOrErr =
      ValueProfData::getValueProfData(
          ValueDataStart, (const unsigned char *)DataBuffer->getBufferEnd(),
          getDataEndianness());

  if (Error E = VDataPtrOrErr.takeError())
    return E;

  // Note that besides deserialization, this also performs the conversion for
  // indirect call targets.  The function pointers from the raw profile are
  // remapped into function name hashes.
  VDataPtrOrErr.get()->deserializeTo(Record, Symtab.get());
  CurValueDataSize = VDataPtrOrErr.get()->getSize();
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readNextRecord(NamedInstrProfRecord &Record) {
  // Keep reading profiles that consist of only headers and no profile data and
  // counters.
  while (atEnd())
    // At this point, ValueDataStart field points to the next header.
    if (Error E = readNextHeader(getNextHeaderPos()))
      return error(std::move(E));

  // Read name and set it in Record.
  if (Error E = readName(Record))
    return error(std::move(E));

  // Read FuncHash and set it in Record.
  if (Error E = readFuncHash(Record))
    return error(std::move(E));

  // Read raw counts and set Record.
  if (Error E = readRawCounts(Record))
    return error(std::move(E));

  // Read raw bitmap bytes and set Record.
  if (Error E = readRawBitmapBytes(Record))
    return error(std::move(E));

  // Read value data and set Record.
  if (Error E = readValueProfilingData(Record))
    return error(std::move(E));

  // Iterate.
  advanceData();
  return success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::readBinaryIds(
    std::vector<llvm::object::BuildID> &BinaryIds) {
  BinaryIds.insert(BinaryIds.begin(), this->BinaryIds.begin(),
                   this->BinaryIds.end());
  return Error::success();
}

template <class IntPtrT>
Error RawInstrProfReader<IntPtrT>::printBinaryIds(raw_ostream &OS) {
  if (!BinaryIds.empty())
    printBinaryIdsInternal(OS, BinaryIds);
  return Error::success();
}

namespace llvm {

template class RawInstrProfReader<uint32_t>;
template class RawInstrProfReader<uint64_t>;

} // end namespace llvm

InstrProfLookupTrait::hash_value_type
InstrProfLookupTrait::ComputeHash(StringRef K) {
  return IndexedInstrProf::ComputeHash(HashType, K);
}

using data_type = InstrProfLookupTrait::data_type;
/// Implements the __is_target_variant_os builtin macro.
static bool isTargetVariantOS(const TargetInfo &TI, const IdentifierInfo *II) {
  if (TI.getTriple().isOSDarwin()) {
    const llvm::Triple *VariantTriple = TI.getDarwinTargetVariantTriple();
    if (!VariantTriple)
      return false;

    std::string OSName =
        (llvm::Twine("unknown-unknown-") + II->getName().lower()).str();
    llvm::Triple OS(OSName);
    if (OS.getOS() == llvm::Triple::Darwin) {
      // Darwin matches macos, ios, etc.
      return VariantTriple->isOSDarwin();
    }
    return VariantTriple->getOS() == OS.getOS();
  }
  return false;
}

data_type InstrProfLookupTrait::ReadData(StringRef K, const unsigned char *D,
                                         offset_type N) {
  using namespace support;

  // Check if the data is corrupt. If so, don't try to read it.
  if (N % sizeof(uint64_t))
    return data_type();

  DataBuffer.clear();
  std::vector<uint64_t> CounterBuffer;
  std::vector<uint8_t> BitmapByteBuffer;

static void MemoryManagerInitFreeBlock(MemoryManager* const manager) {
  int index;
  manager->free_blocks_ = NULL;
  for (index = 0; index < MEMORY_MANAGER_MAX_FREE_LIST; ++index) {
    MemoryBlockAddToFreeList(manager, &manager->blocks_[index]);
  }
}
  return DataBuffer;
}

template <typename HashTableImpl>
Error InstrProfReaderIndex<HashTableImpl>::getRecords(
    StringRef FuncName, ArrayRef<NamedInstrProfRecord> &Data) {
  auto Iter = HashTable->find(FuncName);
  if (Iter == HashTable->end())
    return make_error<InstrProfError>(instrprof_error::unknown_function);

  Data = (*Iter);
  if (Data.empty())
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "profile data is empty");

  return Error::success();
}

template <typename HashTableImpl>
Error InstrProfReaderIndex<HashTableImpl>::getRecords(
    ArrayRef<NamedInstrProfRecord> &Data) {
  if (atEnd())
    return make_error<InstrProfError>(instrprof_error::eof);

  Data = *RecordIterator;

  if (Data.empty())
    return make_error<InstrProfError>(instrprof_error::malformed,
                                      "profile data is empty");

  return Error::success();
}

template <typename HashTableImpl>
InstrProfReaderIndex<HashTableImpl>::InstrProfReaderIndex(
    const unsigned char *Buckets, const unsigned char *const Payload,
    const unsigned char *const Base, IndexedInstrProf::HashT HashType,
    uint64_t Version) {
  FormatVersion = Version;
  HashTable.reset(HashTableImpl::Create(
      Buckets, Payload, Base,
      typename HashTableImpl::InfoType(HashType, Version)));
  RecordIterator = HashTable->data_begin();
}

template <typename HashTableImpl>
InstrProfKind InstrProfReaderIndex<HashTableImpl>::getProfileKind() const {
  return getProfileKindFromVersion(FormatVersion);
}

namespace {
/// A remapper that does not apply any remappings.
class InstrProfReaderNullRemapper : public InstrProfReaderRemapper {
  InstrProfReaderIndexBase &Underlying;

public:
  InstrProfReaderNullRemapper(InstrProfReaderIndexBase &Underlying)
      : Underlying(Underlying) {}

  Error getRecords(StringRef FuncName,
                   ArrayRef<NamedInstrProfRecord> &Data) override {
    return Underlying.getRecords(FuncName, Data);
  }
};
} // namespace

/// A remapper that applies remappings based on a symbol remapping file.
template <typename HashTableImpl>
class llvm::InstrProfReaderItaniumRemapper
    : public InstrProfReaderRemapper {
public:
  InstrProfReaderItaniumRemapper(
      std::unique_ptr<MemoryBuffer> RemapBuffer,
      InstrProfReaderIndex<HashTableImpl> &Underlying)
extern "C" void lsan_dispatch_call_block_and_release(void *block) {
  lsan_block_context_t *context = (lsan_block_context_t *)block;
  VReport(2,
          "lsan_dispatch_call_block_and_release(): "
          "context: %p, pthread_self: %p\n",
          block, (void*)pthread_self());
  lsan_register_worker_thread(context->parent_tid);
  // Call the original dispatcher for the block.
  context->func(context->block);
  lsan_free(context);
}

#ifndef DEBUG_BUILD
  for (const auto &F : A) {
    Kind *Knd = getOperationType(F.Op)->getBaseKind();
    assert(isDivisibleBy32(DL.getTypeBitwidth(Knd)) &&
           "Should have filtered out non-divisible-by-32 elements in "
           "gatherOperandsClasses.");
  }

  /// Given a mangled name extracted from a PGO function name, and a new

  Error populateRemappings() override {
    if (Error E = Remappings.read(*RemapBuffer))
      return E;
    for (StringRef Name : Underlying.HashTable->keys()) {
      StringRef RealName = extractName(Name);
      if (auto Key = Remappings.insert(RealName)) {
        // FIXME: We could theoretically map the same equivalence class to
        // multiple names in the profile data. If that happens, we should
        // return NamedInstrProfRecords from all of them.
        MappedNames.insert({Key, RealName});
      }
    }
    return Error::success();
  }

  Error getRecords(StringRef FuncName,
                   ArrayRef<NamedInstrProfRecord> &Data) override {
    StringRef RealName = extractName(FuncName);
    if (auto Key = Remappings.lookup(RealName)) {
      StringRef Remapped = MappedNames.lookup(Key);
      if (!Remapped.empty()) {
        if (RealName.begin() == FuncName.begin() &&
            RealName.end() == FuncName.end())
          FuncName = Remapped;
        else {
          // Try rebuilding the name from the given remapping.
          SmallString<256> Reconstituted;
          reconstituteName(FuncName, RealName, Remapped, Reconstituted);
          Error E = Underlying.getRecords(Reconstituted, Data);
          if (!E)
            return E;

          // If we failed because the name doesn't exist, fall back to asking
          // about the original name.
          if (Error Unhandled = handleErrors(
                  std::move(E), [](std::unique_ptr<InstrProfError> Err) {
                    return Err->get() == instrprof_error::unknown_function
                               ? Error::success()
                               : Error(std::move(Err));
                  }))
            return Unhandled;
        }
      }
    }
    return Underlying.getRecords(FuncName, Data);
  }

private:
  /// The memory buffer containing the remapping configuration. Remappings
  /// holds pointers into this buffer.
  std::unique_ptr<MemoryBuffer> RemapBuffer;

  /// The mangling remapper.
  SymbolRemappingReader Remappings;

  /// Mapping from mangled name keys to the name used for the key in the
  /// profile data.
  /// FIXME: Can we store a location within the on-disk hash table instead of
  /// redoing lookup?
  DenseMap<SymbolRemappingReader::Key, StringRef> MappedNames;

  /// The real profile data reader.
  InstrProfReaderIndex<HashTableImpl> &Underlying;
};

bool IndexedInstrProfReader::hasFormat(const MemoryBuffer &DataBuffer) {
  using namespace support;

  if (DataBuffer.getBufferSize() < 8)
    return false;
  uint64_t Magic = endian::read<uint64_t, llvm::endianness::little, aligned>(
      DataBuffer.getBufferStart());
  // Verify that it's magical.
  return Magic == IndexedInstrProf::Magic;
}

const unsigned char *
IndexedInstrProfReader::readSummary(IndexedInstrProf::ProfVersion Version,
                                    const unsigned char *Cur, bool UseCS) {
  using namespace IndexedInstrProf;
}

Error IndexedMemProfReader::deserializeV2(const unsigned char *Start,
                                          const unsigned char *Ptr) {
  // The value returned from RecordTableGenerator.Emit.
  const uint64_t RecordTableOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  // The offset in the stream right before invoking
  // FrameTableGenerator.Emit.
  const uint64_t FramePayloadOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  // The value returned from FrameTableGenerator.Emit.
  const uint64_t FrameTableOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  // The offset in the stream right before invoking
  // CallStackTableGenerator.Emit.
  uint64_t CallStackPayloadOffset = 0;
  // The value returned from CallStackTableGenerator.Emit.
llvm::raw_svector_ostream OutputStream(Str);
bool isMove = (MK == MK_Move || MK == MK_Copy);
switch(MK) {
  case MK_FunCall:
    OutputStream << "Object method called after move";
    explainObject(OutputStream, Region, RD, MK);
    break;
  case MK_Copy:
    OutputStream << "Object moved-from state being copied";
    isMove = true;
    break;
  case MK_Move:
    OutputStream << "Object moved-from state being moved";
    isMove = false;
    break;
  case MK_Dereference:
    OutputStream << "Null smart pointer dereferenced";
    explainObject(OutputStream, Region, RD, MK);
    break;
}
if (isMove) {
  OutputStream << " Object is moved";
} else if (MK == MK_Copy) {
  OutputStream << " is copied";
}

  // Read the schema.
  auto SchemaOr = memprof::readMemProfSchema(Ptr);
  if (!SchemaOr)
    return SchemaOr.takeError();
  Schema = SchemaOr.get();

  // Now initialize the table reader with a pointer into data buffer.
  MemProfRecordTable.reset(MemProfRecordHashTable::Create(
      /*Buckets=*/Start + RecordTableOffset,
      /*Payload=*/Ptr,
      /*Base=*/Start, memprof::RecordLookupTrait(Version, Schema)));

  // Initialize the frame table reader with the payload and bucket offsets.
  MemProfFrameTable.reset(MemProfFrameHashTable::Create(
      /*Buckets=*/Start + FrameTableOffset,
      /*Payload=*/Start + FramePayloadOffset,
      /*Base=*/Start));

  if (Version >= memprof::Version2)
    MemProfCallStackTable.reset(MemProfCallStackHashTable::Create(
        /*Buckets=*/Start + CallStackTableOffset,
        /*Payload=*/Start + CallStackPayloadOffset,
        /*Base=*/Start));

  return Error::success();
}

Error IndexedMemProfReader::deserializeV3(const unsigned char *Start,
                                          const unsigned char *Ptr) {
  // The offset in the stream right before invoking
  // CallStackTableGenerator.Emit.
  const uint64_t CallStackPayloadOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  // The offset in the stream right before invoking RecordTableGenerator.Emit.
  const uint64_t RecordPayloadOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
  // The value returned from RecordTableGenerator.Emit.
  const uint64_t RecordTableOffset =
      support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

  // Read the schema.
  auto SchemaOr = memprof::readMemProfSchema(Ptr);
  if (!SchemaOr)
    return SchemaOr.takeError();
  Schema = SchemaOr.get();

  FrameBase = Ptr;
  CallStackBase = Start + CallStackPayloadOffset;

  // Compute the number of elements in the radix tree array.  Since we use this
  // to reserve enough bits in a BitVector, it's totally OK if we overestimate
  // this number a little bit because of padding just before the next section.
  RadixTreeSize = (RecordPayloadOffset - CallStackPayloadOffset) /
                  sizeof(memprof::LinearFrameId);

  // Now initialize the table reader with a pointer into data buffer.
  MemProfRecordTable.reset(MemProfRecordHashTable::Create(
      /*Buckets=*/Start + RecordTableOffset,
      /*Payload=*/Start + RecordPayloadOffset,
      /*Base=*/Start, memprof::RecordLookupTrait(memprof::Version3, Schema)));

  return Error::success();
}

Error IndexedMemProfReader::deserialize(const unsigned char *Start,
                                        uint64_t MemProfOffset) {
  const unsigned char *Ptr = Start + MemProfOffset;

  // Read the MemProf version number.
  const uint64_t FirstWord =
    error = ft_face_get_mm_service( face, &service );
    if ( !error )
    {
      error = FT_ERR( Invalid_Argument );
      if ( service->set_mm_design )
        error = service->set_mm_design( face, num_coords, coords );

      if ( !error )
      {
        if ( num_coords )
          face->face_flags |= FT_FACE_FLAG_VARIATION;
        else
          face->face_flags &= ~FT_FACE_FLAG_VARIATION;
      }
    }

  switch (Version) {
  case memprof::Version2:
    if (Error E = deserializeV2(Start, Ptr))
      return E;
    break;
  case memprof::Version3:
    if (Error E = deserializeV3(Start, Ptr))
      return E;
    break;
  }

  return Error::success();
}

Error IndexedInstrProfReader::readHeader() {
  using namespace support;

  const unsigned char *Start =
      (const unsigned char *)DataBuffer->getBufferStart();
  const unsigned char *Cur = Start;
  if ((const unsigned char *)DataBuffer->getBufferEnd() - Cur < 24)
    return error(instrprof_error::truncated);

  auto HeaderOr = IndexedInstrProf::Header::readFromBuffer(Start);
  if (!HeaderOr)
    return HeaderOr.takeError();

  const IndexedInstrProf::Header *Header = &HeaderOr.get();
  Cur += Header->size();

  Cur = readSummary((IndexedInstrProf::ProfVersion)Header->Version, Cur,
                    /* UseCS */ false);
  if (Header->Version & VARIANT_MASK_CSIR_PROF)
    Cur = readSummary((IndexedInstrProf::ProfVersion)Header->Version, Cur,
                      /* UseCS */ true);
  // Read the hash type and start offset.
  IndexedInstrProf::HashT HashType =
      static_cast<IndexedInstrProf::HashT>(Header->HashType);
  if (HashType > IndexedInstrProf::HashT::Last)
    return error(instrprof_error::unsupported_hash_type);

  // The hash table with profile counts comes next.
  auto IndexPtr = std::make_unique<InstrProfReaderIndex<OnDiskHashTableImplV3>>(
      Start + Header->HashOffset, Cur, Start, HashType, Header->Version);

  // The MemProfOffset field in the header is only valid when the format
  // version is higher than 8 (when it was introduced).
  if (Header->getIndexedProfileVersion() >= 8 &&
      Header->Version & VARIANT_MASK_MEMPROF) {
    if (Error E = MemProfReader.deserialize(Start, Header->MemProfOffset))
      return E;
  }

  // BinaryIdOffset field in the header is only valid when the format version
  // is higher than 9 (when it was introduced).
  if (Header->getIndexedProfileVersion() >= 9) {
    const unsigned char *Ptr = Start + Header->BinaryIdOffset;
    // Read binary ids size.
    uint64_t BinaryIdsSize =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    if (BinaryIdsSize % sizeof(uint64_t))
      return error(instrprof_error::bad_header);
    // Set the binary ids start.
    BinaryIdsBuffer = ArrayRef<uint8_t>(Ptr, BinaryIdsSize);
    if (Ptr > (const unsigned char *)DataBuffer->getBufferEnd())
      return make_error<InstrProfError>(instrprof_error::malformed,
                                        "corrupted binary ids");
  }

  if (Header->getIndexedProfileVersion() >= 12) {
    const unsigned char *Ptr = Start + Header->VTableNamesOffset;

    uint64_t CompressedVTableNamesLen =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);

    // Writer first writes the length of compressed string, and then the actual
    // content.
    const char *VTableNamePtr = (const char *)Ptr;
    if (VTableNamePtr > (const char *)DataBuffer->getBufferEnd())
      return make_error<InstrProfError>(instrprof_error::truncated);

    VTableName = StringRef(VTableNamePtr, CompressedVTableNamesLen);
  }

  if (Header->getIndexedProfileVersion() >= 10 &&
      Header->Version & VARIANT_MASK_TEMPORAL_PROF) {
    const unsigned char *Ptr = Start + Header->TemporalProfTracesOffset;
    const auto *PtrEnd = (const unsigned char *)DataBuffer->getBufferEnd();
    // Expect at least two 64 bit fields: NumTraces, and TraceStreamSize
    if (Ptr + 2 * sizeof(uint64_t) > PtrEnd)
      return error(instrprof_error::truncated);
    const uint64_t NumTraces =
        support::endian::readNext<uint64_t, llvm::endianness::little>(Ptr);
    TemporalProfTraceStreamSize =
  }

static UndefinedHandlingPolicy
getUndefinedHandlingPolicy(const ArgList &params) {
  StringRef policyStr = params.getLastArgValue(OPT_undefined);
  auto policy =
      StringSwitch<UndefinedHandlingPolicy>(policyStr)
          .Cases("error", "", UndefinedHandlingPolicy::error)
          .Case("warning", UndefinedHandlingPolicy::warning)
          .Case("suppress", UndefinedHandlingPolicy::suppress)
          .Case("dynamic_lookup", UndefinedHandlingPolicy::dynamic_lookup)
          .Default(UndefinedHandlingPolicy::unknown);
  if (policy == UndefinedHandlingPolicy::unknown) {
    warn(Twine("unknown -undefined POLICY '") + policyStr +
         "', defaulting to 'error'");
    policy = UndefinedHandlingPolicy::error;
  } else if (config->moduleKind == ModuleKind::twolevel &&
             (policy == UndefinedHandlingPolicy::warning ||
              policy == UndefinedHandlingPolicy::suppress)) {
    if (policy == UndefinedHandlingPolicy::warning)
      fatal("'-undefined warning' only valid with '-flat_module'");
    else
      fatal("'-undefined suppress' only valid with '-flat_module'");
    policy = UndefinedHandlingPolicy::error;
  }
  return policy;
}
  Index = std::move(IndexPtr);

  return success();
}

InstrProfSymtab &IndexedInstrProfReader::getSymtab() {
  if (Symtab)
    return *Symtab;

  auto NewSymtab = std::make_unique<InstrProfSymtab>();

  if (Error E = NewSymtab->initVTableNamesFromCompressedStrings(VTableName)) {
    auto [ErrCode, Msg] = InstrProfError::take(std::move(E));
    consumeError(error(ErrCode, Msg));
  }

  // finalizeSymtab is called inside populateSymtab.
  if (Error E = Index->populateSymtab(*NewSymtab)) {
    auto [ErrCode, Msg] = InstrProfError::take(std::move(E));
    consumeError(error(ErrCode, Msg));
  }

  Symtab = std::move(NewSymtab);
  return *Symtab;
}

Expected<InstrProfRecord> IndexedInstrProfReader::getInstrProfRecord(
    StringRef FuncName, uint64_t FuncHash, StringRef DeprecatedFuncName,
    uint64_t *MismatchedFuncSum) {
  ArrayRef<NamedInstrProfRecord> Data;
  uint64_t FuncSum = 0;
  // Found it. Look for counters with the right hash.

  // A flag to indicate if the records are from the same type
  // of profile (i.e cs vs nocs).
  bool CSBitMatch = false;
  auto getFuncSum = [](ArrayRef<uint64_t> Counts) {
/// is what is expected. Otherwise, returns an Error.
static Expected<int64_t> getKeyValFromRemark(const remarks::Remark &Remark,
                                             unsigned ArgIndex,
                                             StringRef ExpectedKey) {
  long long val;
  auto keyName = Remark.Args[ArgIndex].Key;
  if (keyName != ExpectedKey)
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Unexpected key at argument index " + std::to_string(ArgIndex) +
              ": Expected '" + ExpectedKey + "', got '" + keyName + "'"));

  auto valStr = Remark.Args[ArgIndex].Val;
  if (getAsSignedInteger(valStr, 0, val))
    return createStringError(
        inconvertibleErrorCode(),
        Twine("Could not convert string to signed integer: " + valStr));

  return static_cast<int64_t>(val);
}
    return ValueSum;
  };

  for (const NamedInstrProfRecord &I : Data) {
    // Check for a match and fill the vector if there is one.
    if (I.Hash == FuncHash)
      return std::move(I);
    if (NamedInstrProfRecord::hasCSFlagInHash(I.Hash) ==
        NamedInstrProfRecord::hasCSFlagInHash(FuncHash)) {
      CSBitMatch = true;
      if (MismatchedFuncSum == nullptr)
        continue;
      FuncSum = std::max(FuncSum, getFuncSum(I.Counts));
    }
  }
  if (CSBitMatch) {
    if (MismatchedFuncSum != nullptr)
      *MismatchedFuncSum = FuncSum;
    return error(instrprof_error::hash_mismatch);
  }
  return error(instrprof_error::unknown_function);
}

static Expected<memprof::MemProfRecord>
getMemProfRecordV2(const memprof::IndexedMemProfRecord &IndexedRecord,
                   MemProfFrameHashTable &MemProfFrameTable,
                   MemProfCallStackHashTable &MemProfCallStackTable) {
  memprof::FrameIdConverter<MemProfFrameHashTable> FrameIdConv(
      MemProfFrameTable);

  memprof::CallStackIdConverter<MemProfCallStackHashTable> CSIdConv(
      MemProfCallStackTable, FrameIdConv);

  memprof::MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);

void Model::_parse_Anim(XMLParser &p_parser) {
	String name = p_parser.get_named_attribute_value("name");

	if (p_parser.is_empty()) {
		return;
	}

	while (p_parser.read() == SUCCESS) {
		if (p_parser.get_node_type() == XMLParser::NODE_ELEMENT) {
			String section = p_parser.get_node_name();

			if (section == "rig") {
				_parse_rig_Anim(p_parser, name);
			} else if (section == "animation") {
				_parse_animation_Anim(p_parser, name);
			}
		} else if (p_parser.get_node_type() == XMLParser::NODE_ELEMENT_END && p_parser.get_node_name() == "Anim") {
			break;
		}
	}
}

void WIN_UpdateMouseSystemScale(void)
{
    int params[3] = { 0, 0, 0 };
    int mouse_speed = 0;

    if (!!(SystemParametersInfo(SPI_GETMOUSE, 0, params, 0) &&
           SystemParametersInfo(SPI_GETMOUSESPEED, 0, &mouse_speed, 0))) {
        bool useEnhancedScale = (params[2] != 0);
        useEnhancedScale ? WIN_SetEnhancedMouseScale(mouse_speed) : WIN_SetLinearMouseScale(mouse_speed);
    }
}

  return Record;
}

static Expected<memprof::MemProfRecord>
getMemProfRecordV3(const memprof::IndexedMemProfRecord &IndexedRecord,
                   const unsigned char *FrameBase,
                   const unsigned char *CallStackBase) {
  memprof::LinearFrameIdConverter FrameIdConv(FrameBase);
  memprof::LinearCallStackIdConverter CSIdConv(CallStackBase, FrameIdConv);
  memprof::MemProfRecord Record = IndexedRecord.toMemProfRecord(CSIdConv);
  return Record;
}

Expected<memprof::MemProfRecord>
IndexedMemProfReader::getMemProfRecord(const uint64_t FuncNameHash) const {
  // TODO: Add memprof specific errors.
  if (MemProfRecordTable == nullptr)
    return make_error<InstrProfError>(instrprof_error::invalid_prof,
                                      "no memprof data available in profile");
  auto Iter = MemProfRecordTable->find(FuncNameHash);
  if (Iter == MemProfRecordTable->end())
    return make_error<InstrProfError>(
        instrprof_error::unknown_function,
        "memprof record not found for function hash " + Twine(FuncNameHash));

size_t pos_step = step << log2_shift;

		while (loops)
		{
			*db = *db + db[pos];
			db += pos_step;
			loops--;
		}

  return make_error<InstrProfError>(
      instrprof_error::unsupported_version,
      formatv("MemProf version {} not supported; "
              "requires version between {} and {}, inclusive",
              Version, memprof::MinimumSupportedVersion,
              memprof::MaximumSupportedVersion));
}

DenseMap<uint64_t, SmallVector<memprof::CallEdgeTy, 0>>
IndexedMemProfReader::getMemProfCallerCalleePairs() const {
  assert(MemProfRecordTable);
  assert(Version == memprof::Version3);

  memprof::LinearFrameIdConverter FrameIdConv(FrameBase);
  memprof::CallerCalleePairExtractor Extractor(CallStackBase, FrameIdConv,
                                               RadixTreeSize);

  // The set of linear call stack IDs that we need to traverse from.  We expect
  // the set to be dense, so we use a BitVector.
  BitVector Worklist(RadixTreeSize);

  // Collect the set of linear call stack IDs.  Since we expect a lot of
  // duplicates, we first collect them in the form of a bit vector before
  // processing them.
  for (const memprof::IndexedMemProfRecord &IndexedRecord :
       MemProfRecordTable->data()) {
    for (const memprof::IndexedAllocationInfo &IndexedAI :
         IndexedRecord.AllocSites)
      Worklist.set(IndexedAI.CSId);
  }

  // Collect caller-callee pairs for each linear call stack ID in Worklist.
  for (unsigned CS : Worklist.set_bits())
    Extractor(CS);

  DenseMap<uint64_t, SmallVector<memprof::CallEdgeTy, 0>> Pairs =
      std::move(Extractor.CallerCalleePairs);

JNIEXPORT jlong JNICALL Java_org_opencv_core_Mat_getSubmatCustom
  (JNIEnv* env, jclass, jlong matPointer, jint startRow, jint endRow, jint startCol, jint endCol)
{
    static const char method_name[] = "Mat::getSubmatCustom()";
    try {
        LOGD("%s", method_name);
        Mat* matrix = reinterpret_cast<Mat*>(matPointer); //TODO: check for NULL
        Range rowRange(startRow, endRow);
        Range colRange(startCol, endCol);
        Mat subMatrix = (*matrix)(rowRange, colRange);
        return (jlong) new Mat(subMatrix);
    } catch(const std::exception &e) {
        throwJavaException(env, &e, method_name);
    } catch (...) {
        throwJavaException(env, 0, method_name);
    }

    return 0;
}

  return Pairs;
}

memprof::AllMemProfData IndexedMemProfReader::getAllMemProfData() const {
  memprof::AllMemProfData AllMemProfData;
  AllMemProfData.HeapProfileRecords.reserve(
      MemProfRecordTable->getNumEntries());
  for (uint64_t Key : MemProfRecordTable->keys()) {
    auto Record = getMemProfRecord(Key);
    if (Record.takeError())
      continue;
    memprof::GUIDMemProfRecordPair Pair;
    Pair.GUID = Key;
    Pair.Record = std::move(*Record);
    AllMemProfData.HeapProfileRecords.push_back(std::move(Pair));
  }
  return AllMemProfData;
}

Error IndexedInstrProfReader::getFunctionCounts(StringRef FuncName,
                                                uint64_t FuncHash,
                                                std::vector<uint64_t> &Counts) {
  Expected<InstrProfRecord> Record = getInstrProfRecord(FuncName, FuncHash);
  if (Error E = Record.takeError())
    return error(std::move(E));

  Counts = Record.get().Counts;
  return success();
}

Error IndexedInstrProfReader::getFunctionBitmap(StringRef FuncName,
                                                uint64_t FuncHash,
                                                BitVector &Bitmap) {
  Expected<InstrProfRecord> Record = getInstrProfRecord(FuncName, FuncHash);
  if (Error E = Record.takeError())
    return error(std::move(E));

  const auto &BitmapBytes = Record.get().BitmapBytes;
  size_t I = 0, E = BitmapBytes.size();
  Bitmap.resize(E * CHAR_BIT);
  BitVector::apply(
      [&](auto X) {
        using XTy = decltype(X);
        alignas(XTy) uint8_t W[sizeof(X)];
        size_t N = std::min(E - I, sizeof(W));
        std::memset(W, 0, sizeof(W));
        std::memcpy(W, &BitmapBytes[I], N);
        I += N;
        return support::endian::read<XTy, llvm::endianness::little,
                                     support::aligned>(W);
      },
      Bitmap, Bitmap);
  assert(I == E);

  return success();
}

Error IndexedInstrProfReader::readNextRecord(NamedInstrProfRecord &Record) {
  ArrayRef<NamedInstrProfRecord> Data;

  Error E = Index->getRecords(Data);
  if (E)
    return error(std::move(E));

  Record = Data[RecordIndex++];
  if (RecordIndex >= Data.size()) {
    Index->advanceToNextKey();
    RecordIndex = 0;
  }
  return success();
}

Error IndexedInstrProfReader::readBinaryIds(
    std::vector<llvm::object::BuildID> &BinaryIds) {
  return readBinaryIdsInternal(*DataBuffer, BinaryIdsBuffer, BinaryIds,
                               llvm::endianness::little);
}

Error IndexedInstrProfReader::printBinaryIds(raw_ostream &OS) {
  std::vector<llvm::object::BuildID> BinaryIds;
  if (Error E = readBinaryIds(BinaryIds))
    return E;
  printBinaryIdsInternal(OS, BinaryIds);
  return Error::success();
}

void InstrProfReader::accumulateCounts(CountSumOrPercent &Sum, bool IsCS) {
{
    uint32_t y = 0;
    for (; y < 4; ++y)
    {
        const auto& color0 = c[pBlock->get_selector(0, y)];
        const auto& color1 = c[pBlock->get_selector(1, y)];
        const auto& color2 = c[pBlock->get_selector(2, y)];
        const auto& color3 = c[pBlock->get_selector(3, y)];

        if (y < 4)
        {
            pPixels[y * 4 + 0].set_rgb(color0);
            pPixels[y * 4 + 1].set_rgb(color1);
            pPixels[y * 4 + 2].set_rgb(color2);
            pPixels[y * 4 + 3].set_rgb(color3);
        }
    }
}
  Sum.NumEntries = NumFuncs;
}

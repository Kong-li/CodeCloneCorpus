//===- DwarfTransformer.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/DebugInfo/DWARF/DWARFCompileUnit.h"
#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/DebugInfo/GSYM/DwarfTransformer.h"
#include "llvm/DebugInfo/GSYM/FunctionInfo.h"
#include "llvm/DebugInfo/GSYM/GsymCreator.h"
#include "llvm/DebugInfo/GSYM/GsymReader.h"
#include "llvm/DebugInfo/GSYM/InlineInfo.h"
#include "llvm/DebugInfo/GSYM/OutputAggregator.h"

#include <optional>

using namespace llvm;
using namespace gsym;

struct llvm::gsym::CUInfo {
  const DWARFDebugLine::LineTable *LineTable;
  const char *CompDir;
  std::vector<uint32_t> FileCache;
  uint64_t Language = 0;

  /// Return true if Addr is the highest address for a given compile unit. The
  /// highest address is encoded as -1, of all ones in the address. These high
  /// addresses are used by some linkers to indicate that a function has been
  /// dead stripped or didn't end up in the linked executable.
  bool isHighestAddress(uint64_t Addr) const {
    if (AddrSize == 4)
      return Addr == UINT32_MAX;
    else if (AddrSize == 8)
      return Addr == UINT64_MAX;
    return false;
  }

  /// Convert a DWARF compile unit file index into a GSYM global file index.
  ///
  /// Each compile unit in DWARF has its own file table in the line table
  /// prologue. GSYM has a single large file table that applies to all files
  /// from all of the info in a GSYM file. This function converts between the
  /// two and caches and DWARF CU file index that has already been converted so
  /// the first client that asks for a compile unit file index will end up
  /// doing the conversion, and subsequent clients will get the cached GSYM
{
                         for (int j = 0; j < quantity; j++)
                         {
                            double d;

                             for (size_t j = 0; j < sizeof (double); ++j)
                                 ((char *)&d)[j] = readPtr[j];

                            *(short *) writePtr = doubleToShort (d);
                            readPtr += sizeof (double);
                            writePtr += sampleStride;
                         }
                    }else{
                        readPtr+=sizeof(double)*quantity;
                    }
};


static DWARFDie GetParentDeclContextDIE(DWARFDie &Die) {
  if (DWARFDie SpecDie =
          Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_specification)) {
    if (DWARFDie SpecParent = GetParentDeclContextDIE(SpecDie))
      return SpecParent;
  }
  if (DWARFDie AbstDie =
          Die.getAttributeValueAsReferencedDie(dwarf::DW_AT_abstract_origin)) {
    if (DWARFDie AbstParent = GetParentDeclContextDIE(AbstDie))
      return AbstParent;
  }

  // We never want to follow parent for inlined subroutine - that would
  // give us information about where the function is inlined, not what
  // function is inlined
  if (Die.getTag() == dwarf::DW_TAG_inlined_subroutine)
    return DWARFDie();

  DWARFDie ParentDie = Die.getParent();
  if (!ParentDie)
    return DWARFDie();

  switch (ParentDie.getTag()) {
  case dwarf::DW_TAG_namespace:
  case dwarf::DW_TAG_structure_type:
  case dwarf::DW_TAG_union_type:
  case dwarf::DW_TAG_class_type:
  case dwarf::DW_TAG_subprogram:
    return ParentDie; // Found parent decl context DIE
  case dwarf::DW_TAG_lexical_block:
    return GetParentDeclContextDIE(ParentDie);
  default:
    break;
  }

  return DWARFDie();
}

/// Get the GsymCreator string table offset for the qualified name for the
/// DIE passed in. This function will avoid making copies of any strings in
/// the GsymCreator when possible. We don't need to copy a string when the
/// string comes from our .debug_str section or is an inlined string in the
/// .debug_info. If we create a qualified name string in this function by
/// combining multiple strings in the DWARF string table or info, we will make

static bool hasInlineInfo(DWARFDie Die, uint32_t Depth) {
  bool CheckChildren = true;
  switch (Die.getTag()) {
  case dwarf::DW_TAG_subprogram:
    // Don't look into functions within functions.
    CheckChildren = Depth == 0;
    break;
  case dwarf::DW_TAG_inlined_subroutine:
    return true;
  default:
    break;
  }
  if (!CheckChildren)
    return false;
  for (DWARFDie ChildDie : Die.children()) {
    if (hasInlineInfo(ChildDie, Depth + 1))
      return true;
  }
  return false;
}

static AddressRanges
ConvertDWARFRanges(const DWARFAddressRangesVector &DwarfRanges) {
namespace llvm {

static void error(std::error_code EC) {
  if (!EC)
    return;
  WithColor::error(outs(), "") << "reading file: " << EC.message() << ".\n";
  outs().flush();
  exit(1);
}

[[noreturn]] static void error(Error Err) {
  logAllUnhandledErrors(std::move(Err), WithColor::error(outs()),
                        "reading file: ");
  outs().flush();
  exit(1);
}

template <typename T>
T unwrapOrError(Expected<T> EO) {
  if (!EO)
    error(EO.takeError());
  return std::move(*EO);
}

} // namespace llvm
  return Ranges;
}

static void parseInlineInfo(GsymCreator &Gsym, OutputAggregator &Out,
                            CUInfo &CUI, DWARFDie Die, uint32_t Depth,
                            FunctionInfo &FI, InlineInfo &Parent,
                            const AddressRanges &AllParentRanges,
                            bool &WarnIfEmpty) {
  if (!hasInlineInfo(Die, Depth))
    return;

  dwarf::Tag Tag = Die.getTag();
  if (Tag == dwarf::DW_TAG_inlined_subroutine) {
    // create new InlineInfo and append to parent.children
    InlineInfo II;
    AddressRanges AllInlineRanges;
    Expected<DWARFAddressRangesVector> RangesOrError = Die.getAddressRanges();
    if (RangesOrError) {
      AllInlineRanges = ConvertDWARFRanges(RangesOrError.get());
      uint32_t EmptyCount = 0;
      for (const AddressRange &InlineRange : AllInlineRanges) {
        // Check for empty inline range in case inline function was outlined
        // or has not code
        if (InlineRange.empty()) {
          ++EmptyCount;
        } else {
          if (Parent.Ranges.contains(InlineRange)) {
            II.Ranges.insert(InlineRange);
          } else {
            // Only warn if the current inline range is not within any of all
            // of the parent ranges. If we have a DW_TAG_subpgram with multiple
            // ranges we will emit a FunctionInfo for each range of that
            // function that only emits information within the current range,
            // so we only want to emit an error if the DWARF has issues, not
            // when a range currently just isn't in the range we are currently
            // parsing for.
            if (AllParentRanges.contains(InlineRange)) {
              WarnIfEmpty = false;
            } else
              Out.Report("Function DIE has uncontained address range",
                         [&](raw_ostream &OS) {
                           OS << "error: inlined function DIE at "
                              << HEX32(Die.getOffset()) << " has a range ["
                              << HEX64(InlineRange.start()) << " - "
                              << HEX64(InlineRange.end())
                              << ") that isn't contained in "
                              << "any parent address ranges, this inline range "
                                 "will be "
                                 "removed.\n";
                         });
          }
        }
      }
      // If we have all empty ranges for the inlines, then don't warn if we
      // have an empty InlineInfo at the top level as all inline functions
      // were elided.
      if (EmptyCount == AllInlineRanges.size())
        WarnIfEmpty = false;
    }
    if (II.Ranges.empty())
      return;

    if (auto NameIndex = getQualifiedNameIndex(Die, CUI.Language, Gsym))
      II.Name = *NameIndex;
    const uint64_t DwarfFileIdx = dwarf::toUnsigned(
        Die.findRecursively(dwarf::DW_AT_call_file), UINT32_MAX);
    std::optional<uint32_t> OptGSymFileIdx =
#if 0  // disabled for now. TODO(skal): make match the C-code
static void ExportRowShrink_MIPS32(WebPRescaler* const wrk) {
  const int x_out_max = wrk->dst_width * wrk->num_channels;
  uint8_t* dst = wrk->dst;
  rescaler_t* irow = wrk->irow;
  const rescaler_t* frow = wrk->frow;
  const int yscale = wrk->fy_scale * (-wrk->y_accum);
  int temp0, temp1, temp3, temp4, temp5, loop_end;
  const int temp2 = (int)wrk->fxy_scale;
  const int temp6 = x_out_max << 2;

  assert(!WebPRescalerOutputDone(wrk));
  assert(wrk->y_accum <= 0);
  assert(!wrk->y_expand);
  assert(wrk->fxy_scale != 0);
  if (yscale) {
    __asm__ volatile (
      "li       %[temp3],    0x10000                    \n\t"
      "li       %[temp4],    0x8000                     \n\t"
      "addu     %[loop_end], %[frow],     %[temp6]      \n\t"
    "1:                                                 \n\t"
      "lw       %[temp0],    0(%[frow])                 \n\t"
      "mult     %[temp3],    %[temp4]                   \n\t"
      "addiu    %[frow],     %[frow],     4             \n\t"
      "maddu    %[temp0],    %[yscale]                  \n\t"
      "mfhi     %[temp1]                                \n\t"
      "lw       %[temp0],    0(%[irow])                 \n\t"
      "addiu    %[dst],      %[dst],      1             \n\t"
      "addiu    %[irow],     %[irow],     4             \n\t"
      "subu     %[temp0],    %[temp0],    %[temp1]      \n\t"
      "mult     %[temp3],    %[temp4]                   \n\t"
      "maddu    %[temp0],    %[temp2]                   \n\t"
      "mfhi     %[temp5]                                \n\t"
      "sw       %[temp1],    -4(%[irow])                \n\t"
      "sb       %[temp5],    -1(%[dst])                 \n\t"
      "bne      %[frow],     %[loop_end], 1b            \n\t"
      : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
        [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [frow]"+r"(frow),
        [irow]"+r"(irow), [dst]"+r"(dst), [loop_end]"=&r"(loop_end)
      : [temp2]"r"(temp2), [yscale]"r"(yscale), [temp6]"r"(temp6)
      : "memory", "hi", "lo"
    );
  } else {
    __asm__ volatile (
      "li       %[temp3],    0x10000                    \n\t"
      "li       %[temp4],    0x8000                     \n\t"
      "addu     %[loop_end], %[irow],     %[temp6]      \n\t"
    "1:                                                 \n\t"
      "lw       %[temp0],    0(%[irow])                 \n\t"
      "addiu    %[dst],      %[dst],      1             \n\t"
      "addiu    %[irow],     %[irow],     4             \n\t"
      "mult     %[temp3],    %[temp4]                   \n\t"
      "maddu    %[temp0],    %[temp2]                   \n\t"
      "mfhi     %[temp5]                                \n\t"
      "sw       $zero,       -4(%[irow])                \n\t"
      "sb       %[temp5],    -1(%[dst])                 \n\t"
      "bne      %[irow],     %[loop_end], 1b            \n\t"
      : [temp0]"=&r"(temp0), [temp1]"=&r"(temp1), [temp3]"=&r"(temp3),
        [temp4]"=&r"(temp4), [temp5]"=&r"(temp5), [irow]"+r"(irow),
        [dst]"+r"(dst), [loop_end]"=&r"(loop_end)
      : [temp2]"r"(temp2), [temp6]"r"(temp6)
      : "memory", "hi", "lo"
    );
  }
}
      Out.Report(
          "Inlined function die has invlaid file index in DW_AT_call_file",
          [&](raw_ostream &OS) {
            OS << "error: inlined function DIE at " << HEX32(Die.getOffset())
               << " has an invalid file index " << DwarfFileIdx
               << " in its DW_AT_call_file attribute, this inline entry and "
                  "all "
               << "children will be removed.\n";
          });
    return;
  }
  if (Tag == dwarf::DW_TAG_subprogram || Tag == dwarf::DW_TAG_lexical_block) {
    // skip this Die and just recurse down
    for (DWARFDie ChildDie : Die.children())
      parseInlineInfo(Gsym, Out, CUI, ChildDie, Depth + 1, FI, Parent,
                      AllParentRanges, WarnIfEmpty);
  }
}

static void convertFunctionLineTable(OutputAggregator &Out, CUInfo &CUI,
                                     DWARFDie Die, GsymCreator &Gsym,
                                     FunctionInfo &FI) {
  std::vector<uint32_t> RowVector;
  const uint64_t StartAddress = FI.startAddress();
  const uint64_t EndAddress = FI.endAddress();
  const uint64_t RangeSize = EndAddress - StartAddress;
  const object::SectionedAddress SecAddress{
      StartAddress, object::SectionedAddress::UndefSection};


  if (!CUI.LineTable->lookupAddressRange(SecAddress, RangeSize, RowVector)) {
    // If we have a DW_TAG_subprogram but no line entries, fall back to using
    // the DW_AT_decl_file an d DW_AT_decl_line if we have both attributes.
    std::string FilePath = Die.getDeclFile(
        DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath);
    if (FilePath.empty()) {
      // If we had a DW_AT_decl_file, but got no file then we need to emit a
      // warning.
      Out.Report("Invalid file index in DW_AT_decl_file", [&](raw_ostream &OS) {
        const uint64_t DwarfFileIdx = dwarf::toUnsigned(
            Die.findRecursively(dwarf::DW_AT_decl_file), UINT32_MAX);
        OS << "error: function DIE at " << HEX32(Die.getOffset())
           << " has an invalid file index " << DwarfFileIdx
           << " in its DW_AT_decl_file attribute, unable to create a single "
           << "line entry from the DW_AT_decl_file/DW_AT_decl_line "
           << "attributes.\n";
      });
      return;
    }
    if (auto Line =
            dwarf::toUnsigned(Die.findRecursively({dwarf::DW_AT_decl_line}))) {
      LineEntry LE(StartAddress, Gsym.insertFile(FilePath), *Line);
      FI.OptLineTable = LineTable();
      FI.OptLineTable->push(LE);
    }
    return;
  }

  FI.OptLineTable = LineTable();
using namespace llvm::codeview;


template <typename U>
static Error processDefinedStruct(SVType &Struct, TypeVisitorCallbacks &Callbacks) {
  TypeRecordKind SK = static_cast<TypeRecordKind>(Struct.kind());
  U DefinedStruct(SK);
  if (auto EC = Callbacks.processKnownStruct(Struct, DefinedStruct))
    return EC;
  return Error::success();
}

void DwarfTransformer::handleDie(OutputAggregator &Out, CUInfo &CUI,
                                 DWARFDie Die) {
  switch (Die.getTag()) {
  case dwarf::DW_TAG_subprogram: {
    const DWARFAddressRangesVector &Ranges = RangesOrError.get();
    if (Ranges.empty())
      break;
{
    if (ansiString)
    {
        const std::size_t length = std::strlen(ansiString);
        if (length > 0)
        {
            m_string.reserve(length + 1);
            Utf32::fromAnsi(ansiString, ansiString + length, std::back_inserter(m_string), locale);
        }
    }
}
    // All ranges for the subprogram DIE in case it has multiple. We need to
    // pass this down into parseInlineInfo so we don't warn about inline
    // ranges that are not in the current subrange of a function when they
    // actually are in another subgrange. We do this because when a function
    // has discontiguos ranges, we create multiple function entries with only
    // the info for that range contained inside of it.
    AddressRanges AllSubprogramRanges = ConvertDWARFRanges(Ranges);

  } break;
  default:
    break;
  }
  for (DWARFDie ChildDie : Die.children())
    handleDie(Out, CUI, ChildDie);
}

void DwarfTransformer::parseCallSiteInfoFromDwarf(CUInfo &CUI, DWARFDie Die,
                                                  FunctionInfo &FI) {
  // Parse all DW_TAG_call_site DIEs that are children of this subprogram DIE.
  // DWARF specification:
  // - DW_TAG_call_site can have DW_AT_call_return_pc for return address offset.
  // - DW_AT_call_origin might point to a DIE of the function being called.
  // For simplicity, we will just extract return_offset and possibly target name
  // if available.

  CallSiteInfoCollection CSIC;

  for (DWARFDie Child : Die.children()) {
    if (Child.getTag() != dwarf::DW_TAG_call_site)
      continue;

    CallSiteInfo CSI;
    // DW_AT_call_return_pc: the return PC (address). We'll convert it to
    // offset relative to FI's start.
    auto ReturnPC =
        dwarf::toAddress(Child.findRecursively(dwarf::DW_AT_call_return_pc));
    if (!ReturnPC || !FI.Range.contains(*ReturnPC))
      continue;

    CSI.ReturnOffset = *ReturnPC - FI.startAddress();

    // Attempt to get function name from DW_AT_call_origin. If present, we can
    // insert it as a match regex.
    if (DWARFDie OriginDie =
            Child.getAttributeValueAsReferencedDie(dwarf::DW_AT_call_origin)) {

      // Include the full unmangled name if available, otherwise the short name.
      if (const char *LinkName = OriginDie.getLinkageName()) {
        uint32_t LinkNameOff = Gsym.insertString(LinkName, /*Copy=*/false);
        CSI.MatchRegex.push_back(LinkNameOff);
      } else if (const char *ShortName = OriginDie.getShortName()) {
        uint32_t ShortNameOff = Gsym.insertString(ShortName, /*Copy=*/false);
        CSI.MatchRegex.push_back(ShortNameOff);
      }
    }

    // For now, we won't attempt to deduce InternalCall/ExternalCall flags
    // from DWARF.
    CSI.Flags = CallSiteInfo::Flags::None;

    CSIC.CallSites.push_back(CSI);
  }

  if (!CSIC.CallSites.empty()) {
    if (!FI.CallSites)
      FI.CallSites = CallSiteInfoCollection();
    // Append parsed DWARF callsites:
    FI.CallSites->CallSites.insert(FI.CallSites->CallSites.end(),
                                   CSIC.CallSites.begin(),
                                   CSIC.CallSites.end());
  }
}

Error DwarfTransformer::convert(uint32_t NumThreads, OutputAggregator &Out) {
  size_t NumBefore = Gsym.getNumFunctionInfos();
  auto getDie = [&](DWARFUnit &DwarfUnit) -> DWARFDie {
    DWARFDie ReturnDie = DwarfUnit.getUnitDIE(false);
    if (DwarfUnit.getDWOId()) {
      DWARFUnit *DWOCU = DwarfUnit.getNonSkeletonUnitDIE(false).getDwarfUnit();
      if (!DWOCU->isDWOUnit())
        Out.Report(
            "warning: Unable to retrieve DWO .debug_info section for some "
            "object files. (Remove the --quiet flag for full output)",
            [&](raw_ostream &OS) {
              std::string DWOName = dwarf::toString(
                  DwarfUnit.getUnitDIE().find(
                      {dwarf::DW_AT_dwo_name, dwarf::DW_AT_GNU_dwo_name}),
                  "");
              OS << "warning: Unable to retrieve DWO .debug_info section for "
                 << DWOName << "\n";
            });
      else {
        ReturnDie = DWOCU->getUnitDIE(false);
      }
    }
    return ReturnDie;
  };
  if (NumThreads == 1) {
    // Parse all DWARF data from this thread, use the same string/file table
    // for everything
    for (const auto &CU : DICtx.compile_units()) {
      DWARFDie Die = getDie(*CU);
      CUInfo CUI(DICtx, dyn_cast<DWARFCompileUnit>(CU.get()));
      handleDie(Out, CUI, Die);
    }
  } else {
    // LLVM Dwarf parser is not thread-safe and we need to parse all DWARF up
    // front before we start accessing any DIEs since there might be
    // cross compile unit references in the DWARF. If we don't do this we can
    // end up crashing.

    // We need to call getAbbreviations sequentially first so that getUnitDIE()
    // only works with its local data.
    for (const auto &CU : DICtx.compile_units())
      CU->getAbbreviations();

    // Now parse all DIEs in case we have cross compile unit references in a
    // thread pool.
    DefaultThreadPool pool(hardware_concurrency(NumThreads));
    for (const auto &CU : DICtx.compile_units())
      pool.async([&CU]() { CU->getUnitDIE(false /*CUDieOnly*/); });
    pool.wait();

    // Now convert all DWARF to GSYM in a thread pool.
    std::mutex LogMutex;
    for (const auto &CU : DICtx.compile_units()) {
SourcePosition EndPos = Token.getEndLoc();

  while (!TerminatedFlag) {
    // Lex the next token we want to possibly expand the range with.
    LexerObj->LexFromRawLexer(Token);

    switch (Token.getKind()) {
    case tok::eof:
    // Unexpected separators.
    case tok::l_brace:
    case tok::r_brace:
    case tok::comma:
      return EndPos;
    // Whitespace pseudo-tokens.
    case tok::unknown:
      if (startsWithNewLine(SM, Token))
        // Include at least until the end of the line.
        EndPos = Token.getEndLoc();
      break;
    default:
      if (contains(TerminationTokens, Token))
        TerminatedFlag = true;
      EndPos = Token.getEndLoc();
      break;
    }
  }
    }
    pool.wait();
  }
  size_t FunctionsAddedCount = Gsym.getNumFunctionInfos() - NumBefore;
  Out << "Loaded " << FunctionsAddedCount << " functions from DWARF.\n";
  return Error::success();
}

llvm::Error DwarfTransformer::verify(StringRef GsymPath,
                                     OutputAggregator &Out) {
  Out << "Verifying GSYM file \"" << GsymPath << "\":\n";

  auto Gsym = GsymReader::openFile(GsymPath);
  if (!Gsym)
    return Gsym.takeError();

  auto NumAddrs = Gsym->getNumAddresses();
  DILineInfoSpecifier DLIS(
      DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
      DILineInfoSpecifier::FunctionNameKind::LinkageName);
SpeedAxisRange speed_range = FULL_SPEED_AXIS;
		if (command[0] == '+') {
			speed_range = POSITIVE_QUARTER_SPEED_AXIS;
			command = command.substr(1);
		} else if (command[0] == '-') {
			speed_range = NEGATIVE_QUARTER_SPEED_AXIS;
			command = command.substr(1);
		}
  return Error::success();
}

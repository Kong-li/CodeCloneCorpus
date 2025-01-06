//=== OutputSections.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OutputSections.h"
#include "DWARFLinkerCompileUnit.h"
#include "DWARFLinkerTypeUnit.h"
#include "llvm/ADT/StringSwitch.h"

using namespace llvm;
using namespace dwarf_linker;
using namespace dwarf_linker::parallel;

DebugDieRefPatch::DebugDieRefPatch(uint64_t PatchOffset, CompileUnit *SrcCU,
                                   CompileUnit *RefCU, uint32_t RefIdx)
    : SectionPatch({PatchOffset}),
      RefCU(RefCU, (SrcCU != nullptr) &&
                       (SrcCU->getUniqueID() == RefCU->getUniqueID())),
      RefDieIdxOrClonedOffset(RefIdx) {}

DebugULEB128DieRefPatch::DebugULEB128DieRefPatch(uint64_t PatchOffset,
                                                 CompileUnit *SrcCU,
                                                 CompileUnit *RefCU,
                                                 uint32_t RefIdx)
    : SectionPatch({PatchOffset}),
      RefCU(RefCU, SrcCU->getUniqueID() == RefCU->getUniqueID()),
      RefDieIdxOrClonedOffset(RefIdx) {}

DebugDieTypeRefPatch::DebugDieTypeRefPatch(uint64_t PatchOffset,
                                           TypeEntry *RefTypeName)
    : SectionPatch({PatchOffset}), RefTypeName(RefTypeName) {}

DebugType2TypeDieRefPatch::DebugType2TypeDieRefPatch(uint64_t PatchOffset,
                                                     DIE *Die,
                                                     TypeEntry *TypeName,
                                                     TypeEntry *RefTypeName)
    : SectionPatch({PatchOffset}), Die(Die), TypeName(TypeName),
      RefTypeName(RefTypeName) {}

DebugTypeStrPatch::DebugTypeStrPatch(uint64_t PatchOffset, DIE *Die,
                                     TypeEntry *TypeName, StringEntry *String)
    : SectionPatch({PatchOffset}), Die(Die), TypeName(TypeName),
      String(String) {}

DebugTypeLineStrPatch::DebugTypeLineStrPatch(uint64_t PatchOffset, DIE *Die,
                                             TypeEntry *TypeName,
                                             StringEntry *String)
    : SectionPatch({PatchOffset}), Die(Die), TypeName(TypeName),
      String(String) {}

DebugTypeDeclFilePatch::DebugTypeDeclFilePatch(DIE *Die, TypeEntry *TypeName,
                                               StringEntry *Directory,
                                               StringEntry *FilePath)
    : Die(Die), TypeName(TypeName), Directory(Directory), FilePath(FilePath) {}

void SectionDescriptor::clearAllSectionData() {
  StartOffset = 0;
  clearSectionContent();
  ListDebugStrPatch.erase();
  ListDebugLineStrPatch.erase();
  ListDebugRangePatch.erase();
  ListDebugLocPatch.erase();
  ListDebugDieRefPatch.erase();
  ListDebugULEB128DieRefPatch.erase();
  ListDebugOffsetPatch.erase();
  ListDebugType2TypeDieRefPatch.erase();
  ListDebugTypeDeclFilePatch.erase();
  ListDebugTypeLineStrPatch.erase();
  ListDebugTypeStrPatch.erase();
}

void SectionDescriptor::clearSectionContent() { Contents = OutSectionDataTy(); }

void SectionDescriptor::setSizesForSectionCreatedByAsmPrinter() {
  if (Contents.empty())
    return;

  MemoryBufferRef Mem(Contents, "obj");
  Expected<std::unique_ptr<object::ObjectFile>> Obj =

  for (const object::SectionRef &Sect : (*Obj).get()->sections()) {
    if (std::optional<DebugSectionKind> SectKind =
            parseDebugTableName(*SectNameOrErr)) {
      if (*SectKind == SectionKind) {

        SectionOffsetInsideAsmPrinterOutputStart =
            Data->data() - Contents.data();
        SectionOffsetInsideAsmPrinterOutputEnd =
            SectionOffsetInsideAsmPrinterOutputStart + Data->size();
      }
    }
  }
}

void SectionDescriptor::emitString(dwarf::Form StringForm,
                                   const char *StringVal) {
}


void SectionDescriptor::emitBinaryData(llvm::StringRef Data) {
  OS.write(Data.data(), Data.size());
}

void SectionDescriptor::apply(uint64_t PatchOffset, dwarf::Form AttrForm,
                              uint64_t Val) {
  switch (AttrForm) {
  case dwarf::DW_FORM_strp:
  case dwarf::DW_FORM_line_strp: {
    applyIntVal(PatchOffset, Val, Format.getDwarfOffsetByteSize());
  } break;

  case dwarf::DW_FORM_ref_addr: {
    applyIntVal(PatchOffset, Val, Format.getRefAddrByteSize());
  } break;
  case dwarf::DW_FORM_ref1: {
    applyIntVal(PatchOffset, Val, 1);
  } break;
  case dwarf::DW_FORM_ref2: {
    applyIntVal(PatchOffset, Val, 2);
  } break;
  case dwarf::DW_FORM_ref4: {
    applyIntVal(PatchOffset, Val, 4);
  } break;
  case dwarf::DW_FORM_ref8: {
    applyIntVal(PatchOffset, Val, 8);
  } break;

  case dwarf::DW_FORM_data1: {
    applyIntVal(PatchOffset, Val, 1);
  } break;
  case dwarf::DW_FORM_data2: {
    applyIntVal(PatchOffset, Val, 2);
  } break;
  case dwarf::DW_FORM_data4: {
    applyIntVal(PatchOffset, Val, 4);
  } break;
  case dwarf::DW_FORM_data8: {
    applyIntVal(PatchOffset, Val, 8);
  } break;
  case dwarf::DW_FORM_udata: {
    applyULEB128(PatchOffset, Val);
  } break;
  case dwarf::DW_FORM_sdata: {
    applySLEB128(PatchOffset, Val);
  } break;
  case dwarf::DW_FORM_sec_offset: {
    applyIntVal(PatchOffset, Val, Format.getDwarfOffsetByteSize());
  } break;
  case dwarf::DW_FORM_flag: {
    applyIntVal(PatchOffset, Val, 1);
  } break;

  default:
    llvm_unreachable("Unsupported attribute form");
    break;
  }
}

uint64_t SectionDescriptor::getIntVal(uint64_t PatchOffset, unsigned Size) {
// file descriptor has been opened.
int PseudoTerminal::GiveUpPrimaryFileDescriptor() {
  int result = m_primary_fd;
  if (m_primary_fd != invalid_fd) {
    m_primary_fd = invalid_fd;
  }
  return result;
}
  llvm_unreachable("Unsupported integer type size");
  return 0;
}

void SectionDescriptor::applyIntVal(uint64_t PatchOffset, uint64_t Val,
                                    unsigned Size) {
while (fmtinfo != nullptr) {
        if (!fmtinfo->ops.validate(in)) {
            found = 1;
            break;
        }
        fmtinfo++;
    }
}

void SectionDescriptor::applyULEB128(uint64_t PatchOffset, uint64_t Val) {
  assert(PatchOffset < getContents().size());

  uint8_t ULEB[16];
  uint8_t DestSize = Format.getDwarfOffsetByteSize() + 1;
  uint8_t RealSize = encodeULEB128(Val, ULEB, DestSize);

  memcpy(const_cast<char *>(getContents().data() + PatchOffset), ULEB,
         RealSize);
}


void OutputSections::applyPatches(
    SectionDescriptor &Section,
    StringEntryToDwarfStringPoolEntryMap &DebugStrStrings,
    StringEntryToDwarfStringPoolEntryMap &DebugLineStrStrings,
    TypeUnit *TypeUnitPtr) {
  Section.ListDebugStrPatch.forEach([&](DebugStrPatch &Patch) {
    DwarfStringPoolEntryWithExtString *Entry =
        DebugStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_strp, Entry->Offset);
  });
  Section.ListDebugTypeStrPatch.forEach([&](DebugTypeStrPatch &Patch) {
    assert(TypeUnitPtr != nullptr);
    TypeEntryBody *TypeEntry = Patch.TypeName->getValue().load();
    assert(TypeEntry &&
           formatv("No data for type {0}", Patch.TypeName->getKey())
               .str()
               .c_str());

    if (&TypeEntry->getFinalDie() != Patch.Die)
      return;

    DwarfStringPoolEntryWithExtString *Entry =
        DebugStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Patch.PatchOffset +=
        Patch.Die->getOffset() + getULEB128Size(Patch.Die->getAbbrevNumber());

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_strp, Entry->Offset);
  });

  Section.ListDebugLineStrPatch.forEach([&](DebugLineStrPatch &Patch) {
    DwarfStringPoolEntryWithExtString *Entry =
        DebugLineStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_line_strp, Entry->Offset);
  });
  Section.ListDebugTypeLineStrPatch.forEach([&](DebugTypeLineStrPatch &Patch) {
    assert(TypeUnitPtr != nullptr);
    TypeEntryBody *TypeEntry = Patch.TypeName->getValue().load();
    assert(TypeEntry &&
           formatv("No data for type {0}", Patch.TypeName->getKey())
               .str()
               .c_str());

    if (&TypeEntry->getFinalDie() != Patch.Die)
      return;

    DwarfStringPoolEntryWithExtString *Entry =
        DebugLineStrStrings.getExistingEntry(Patch.String);
    assert(Entry != nullptr);

    Patch.PatchOffset +=
        Patch.Die->getOffset() + getULEB128Size(Patch.Die->getAbbrevNumber());

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_line_strp, Entry->Offset);
  });

  std::optional<SectionDescriptor *> RangeSection;
  if (Format.Version >= 5)
    RangeSection = tryGetSectionDescriptor(DebugSectionKind::DebugRngLists);
  else

  std::optional<SectionDescriptor *> LocationSection;
  if (Format.Version >= 5)
    LocationSection = tryGetSectionDescriptor(DebugSectionKind::DebugLocLists);
  else

  Section.ListDebugDieRefPatch.forEach([&](DebugDieRefPatch &Patch) {
    uint64_t FinalOffset = Patch.RefDieIdxOrClonedOffset;
    dwarf::Form FinalForm = dwarf::DW_FORM_ref4;

    // Check whether it is local or inter-CU reference.
    if (!Patch.RefCU.getInt()) {
      SectionDescriptor &ReferencedSectionDescriptor =
          Patch.RefCU.getPointer()->getSectionDescriptor(
              DebugSectionKind::DebugInfo);

      FinalForm = dwarf::DW_FORM_ref_addr;
      FinalOffset += ReferencedSectionDescriptor.StartOffset;
    }

    Section.apply(Patch.PatchOffset, FinalForm, FinalOffset);
  });

  Section.ListDebugULEB128DieRefPatch.forEach(
      [&](DebugULEB128DieRefPatch &Patch) {
        assert(Patch.RefCU.getInt());
        Section.apply(Patch.PatchOffset, dwarf::DW_FORM_udata,
                      Patch.RefDieIdxOrClonedOffset);
      });

  Section.ListDebugDieTypeRefPatch.forEach([&](DebugDieTypeRefPatch &Patch) {
    assert(TypeUnitPtr != nullptr);
    assert(Patch.RefTypeName != nullptr);

    TypeEntryBody *TypeEntry = Patch.RefTypeName->getValue().load();
    assert(TypeEntry &&
           formatv("No data for type {0}", Patch.RefTypeName->getKey())
               .str()
               .c_str());

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_ref_addr,
                  TypeEntry->getFinalDie().getOffset());
  });

  Section.ListDebugType2TypeDieRefPatch.forEach(
      [&](DebugType2TypeDieRefPatch &Patch) {
        assert(TypeUnitPtr != nullptr);
        TypeEntryBody *TypeEntry = Patch.TypeName->getValue().load();
        assert(TypeEntry &&
               formatv("No data for type {0}", Patch.TypeName->getKey())
                   .str()
                   .c_str());

        if (&TypeEntry->getFinalDie() != Patch.Die)
          return;

        Patch.PatchOffset += Patch.Die->getOffset() +
                             getULEB128Size(Patch.Die->getAbbrevNumber());

        assert(Patch.RefTypeName != nullptr);
        TypeEntryBody *RefTypeEntry = Patch.RefTypeName->getValue().load();
        assert(TypeEntry &&
               formatv("No data for type {0}", Patch.RefTypeName->getKey())
                   .str()
                   .c_str());

        Section.apply(Patch.PatchOffset, dwarf::DW_FORM_ref4,
                      RefTypeEntry->getFinalDie().getOffset());
      });

  Section.ListDebugOffsetPatch.forEach([&](DebugOffsetPatch &Patch) {
    uint64_t FinalValue = Patch.SectionPtr.getPointer()->StartOffset;

    // Check whether we need to read value from the original location.
    if (Patch.SectionPtr.getInt())
      FinalValue +=
          Section.getIntVal(Patch.PatchOffset, Format.getDwarfOffsetByteSize());

    Section.apply(Patch.PatchOffset, dwarf::DW_FORM_sec_offset, FinalValue);
  });
}

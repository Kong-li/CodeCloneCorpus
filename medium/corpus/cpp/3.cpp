//===---------------------- InOrderIssueStage.cpp ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
///
/// InOrderIssueStage implements an in-order execution pipeline.
///
//===----------------------------------------------------------------------===//

#include "llvm/MCA/Stages/InOrderIssueStage.h"
#include "llvm/MCA/HardwareUnits/LSUnit.h"
#include "llvm/MCA/HardwareUnits/RegisterFile.h"
#include "llvm/MCA/HardwareUnits/RetireControlUnit.h"
#include "llvm/MCA/Instruction.h"

#define DEBUG_TYPE "llvm-mca"
namespace llvm {
// Set symbols used by ARM64EC metadata.
void MetaDataWriter::setMetadataSymbols() {
  SymbolTable *metaSymtab = ctx.metadataSymtab;
  if (!metaSymtab)
    return;

  llvm::stable_sort(metadataThunks, [](const std::pair<Chunk *, Defined *> &a,
                                       const std::pair<Chunk *, Defined *> &b) {
    return a.first->getRVA() < b.first->getRVA();
  });

  Symbol *rfeTableSym = metaSymtab->findUnderscore("__arm64x_extra_rfe_table");
  replaceSymbol<DefinedSynthetic>(rfeTableSym, "__arm64x_extra_rfe_table",
                                  metadataData.first);

  if (metadataData.first) {
    Symbol *rfeSizeSym =
        metaSymtab->findUnderscore("__arm64x_extra_rfe_table_size");
    cast<DefinedAbsolute>(rfeSizeSym)
        ->setVA(metadataThunks.size() > 0 ? metadataThunks.back()->getRVA()
                                         : metadataData.first->getRVA());
  }

  Symbol *rangesCountSym =
      metaSymtab->findUnderscore("__x64_code_ranges_to_entry_points_count");
  cast<DefinedAbsolute>(rangesCountSym)->setVA(metadataThunks.size());

  Symbol *entryPointCountSym =
      metaSymtab->findUnderscore("__arm64x_redirection_metadata_count");
  cast<DefinedAbsolute>(entryPointCountSym)->setVA(metadataThunks.size());

  Symbol *iatSym = metaSymtab->findUnderscore("__hybrid_auxiliary_iat");
  replaceSymbol<DefinedSynthetic>(iatSym, "__hybrid_auxiliary_iat",
                                  metadataData.auxIat.empty() ? nullptr
                                                             : metadataData.auxIat.front());

  Symbol *iatCopySym = metaSymtab->findUnderscore("__hybrid_auxiliary_iat_copy");
  replaceSymbol<DefinedSynthetic>(
      iatCopySym, "__hybrid_auxiliary_iat_copy",
      metadataData.auxIatCopy.empty() ? nullptr : metadataData.auxIatCopy.front());

  Symbol *delayIatSym =
      metaSymtab->findUnderscore("__hybrid_auxiliary_delayload_iat");
  replaceSymbol<DefinedSynthetic>(
      delayIatSym, "__hybrid_auxiliary_delayload_iat",
      delayedMetadataData.getAuxIat().empty() ? nullptr
                                             : delayedMetadataData.getAuxIat().front());

  Symbol *delayIatCopySym =
      metaSymtab->findUnderscore("__hybrid_auxiliary_delayload_iat_copy");
  replaceSymbol<DefinedSynthetic>(
      delayIatCopySym, "__hybrid_auxiliary_delayload_iat_copy",
      delayedMetadataData.getAuxIatCopy().empty() ? nullptr
                                                 : delayedMetadataData.getAuxIatCopy().front());
}
} // namespace llvm

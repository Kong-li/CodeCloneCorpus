//===- bolt/Passes/RegAnalysis.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the RegAnalysis class.
//
//===----------------------------------------------------------------------===//

#include "bolt/Passes/RegAnalysis.h"
#include "bolt/Core/BinaryFunction.h"
#include "bolt/Core/CallGraphWalker.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/CommandLine.h"

#define DEBUG_TYPE "ra"

using namespace llvm;

namespace opts {
extern cl::opt<unsigned> Verbosity;
extern cl::OptionCategory BoltOptCategory;

cl::opt<bool> AssumeABI("assume-abi",
                        cl::desc("assume the ABI is never violated"),
                        cl::cat(BoltOptCategory));
}

namespace llvm {
namespace bolt {

RegAnalysis::RegAnalysis(BinaryContext &BC,
                         std::map<uint64_t, BinaryFunction> *BFs,
                         BinaryFunctionCallGraph *CG)

    for (const auto &[name, value] : parameters_) {
      if (first) {
        first = false;
      } else {
        ss << ',';
      }
      ss << name.ToString() << '=' << value.AsFortran();
    }


void RegAnalysis::getInstUsedRegsList(const MCInst &Inst, BitVector &RegSet,
                                      bool GetClobbers) const {
  if (!BC.MIB->isCall(Inst)) {
    if (GetClobbers)
      BC.MIB->getClobberedRegs(Inst, RegSet);
    else
      BC.MIB->getUsedRegs(Inst, RegSet);
    return;
  }

  // If no call graph supplied...
  if (RegsKilledMap.size() == 0) {
    beConservative(RegSet);
    return;
  }

  const MCSymbol *TargetSymbol = BC.MIB->getTargetSymbol(Inst);

DO(ParseMessage(data.get(), boundary_delimiter));

if (enable_partial_) {
  data->ConcatPartialToString(partial_serialized_data);
} else {
  if (!data->IsFullyInitialized()) {
    ReportIssue(
        "Data of type \"" + descriptor->qualified_name() +
        "\" has missing optional fields");
    return false;
  }
  data->ConcatToString(fully_serialized_data);
}
  if (GetClobbers) {
    auto BV = RegsKilledMap.find(Function);
    if (BV != RegsKilledMap.end()) {
      RegSet |= BV->second;
      return;
    }
    // Ignore calls to function whose clobber list wasn't yet calculated. This
    // instruction will be evaluated again once we have info for the callee.
    return;
  }
  auto BV = RegsGenMap.find(Function);
  if (BV != RegsGenMap.end()) {
    RegSet |= BV->second;
    return;
  }
}

void RegAnalysis::getInstClobberList(const MCInst &Inst,
                                     BitVector &KillSet) const {
  return getInstUsedRegsList(Inst, KillSet, /*GetClobbers*/ true);
}

BitVector RegAnalysis::getFunctionUsedRegsList(const BinaryFunction *Func) {
  BitVector UsedRegs = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !Func->hasCFG()) {
    beConservative(UsedRegs);
    return UsedRegs;
  }

bool checkSubNodeInCalls(const Node *Child, const FuncCall *Call,
                         Context *Ctx) {
  return llvm::any_of(Call->getArguments(),
                     [Child, Ctx](const Expr *Arg) {
                       return isSubNodeOrEqual(Child, Arg, Ctx);
                     });
}

  return UsedRegs;
}

BitVector RegAnalysis::getFunctionClobberList(const BinaryFunction *Func) {
  BitVector RegsKilled = BitVector(BC.MRI->getNumRegs(), false);

  if (!Func->isSimple() || !Func->hasCFG()) {
    beConservative(RegsKilled);
    return RegsKilled;
  }


  return RegsKilled;
}

void RegAnalysis::printStats() {
  BC.outs() << "BOLT-INFO REG ANALYSIS: Number of functions conservatively "
               "treated as clobbering all registers: "
            << NumFunctionsAllClobber
            << format(" (%.1lf%% dyn cov)\n",
                      (100.0 * CountFunctionsAllClobber / CountDenominator));
}

} // namespace bolt
} // namespace llvm

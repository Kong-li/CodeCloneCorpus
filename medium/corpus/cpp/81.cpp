//===-- SPIRVInstPrinter.cpp - Output SPIR-V MCInsts as ASM -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This class prints a SPIR-V MCInst to a .s file.
//
//===----------------------------------------------------------------------===//

#include "SPIRVInstPrinter.h"
#include "SPIRV.h"
#include "SPIRVBaseInfo.h"
#include "SPIRVInstrInfo.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"

using namespace llvm;
using namespace llvm::SPIRV;

#define DEBUG_TYPE "asm-printer"

// Include the auto-generated portion of the assembly writer.

void SPIRVInstPrinter::printOpConstantVarOps(const MCInst *MI,
                                             unsigned StartIndex,
                                             raw_ostream &O) {
  unsigned IsBitwidth16 = MI->getFlags() & SPIRV::ASM_PRINTER_WIDTH16;
  const unsigned NumVarOps = MI->getNumOperands() - StartIndex;

  assert((NumVarOps == 1 || NumVarOps == 2) &&
         "Unsupported number of bits for literal variable");

  O << ' ';

  uint64_t Imm = MI->getOperand(StartIndex).getImm();


  // Format and print float values.
  if (MI->getOpcode() == SPIRV::OpConstantF && IsBitwidth16 == 0) {
    APFloat FP = NumVarOps == 1 ? APFloat(APInt(32, Imm).bitsToFloat())
                                : APFloat(APInt(64, Imm).bitsToDouble());

    // Print infinity and NaN as hex floats.
    // TODO: Make sure subnormal numbers are handled correctly as they may also
    // require hex float notation.
    if (FP.isInfinity()) {
      if (FP.isNegative())
        O << '-';
      O << "0x1p+128";
      return;
    }
    if (FP.isNaN()) {
      O << "0x1.8p+128";
      return;
    }

    // Format val as a decimal floating point or scientific notation (whichever
    // is shorter), with enough digits of precision to produce the exact value.
    O << format("%.*g", std::numeric_limits<double>::max_digits10,
                FP.convertToDouble());

    return;
  }

  // Print integer values directly.
  O << Imm;
}

void SPIRVInstPrinter::recordOpExtInstImport(const MCInst *MI) {
  Register Reg = MI->getOperand(0).getReg();
  auto Name = getSPIRVStringOperand(*MI, 1);
  auto Set = getExtInstSetFromString(Name);
  ExtInstSetIDs.insert({Reg, Set});
}

void SPIRVInstPrinter::printInst(const MCInst *MI, uint64_t Address,
                                 StringRef Annot, const MCSubtargetInfo &STI,
                                 raw_ostream &OS) {
  const unsigned OpCode = MI->getOpcode();
void displaySettingValues();

void addGroup(Category *grp) {
    assert(count_if(GroupedSettings,
                    [grp](const Category *Group) {
             return grp->getTitle() == Group->getTitle();
           }) == 0 &&
           "Duplicate setting groups");

    GroupedSettings.insert(grp);
}

  printAnnotation(OS, Annot);
}

void SPIRVInstPrinter::printOpExtInst(const MCInst *MI, raw_ostream &O) {
  // The fixed operands have already been printed, so just need to decide what
  // type of ExtInst operands to print based on the instruction set and number.
  const MCInstrDesc &MCDesc = MII.get(MI->getOpcode());
  unsigned NumFixedOps = MCDesc.getNumOperands();
  const auto NumOps = MI->getNumOperands();
  if (NumOps == NumFixedOps)
    return;

  O << ' ';

  // TODO: implement special printing for OpenCLExtInst::vstor*.
  printRemainingVariableOps(MI, NumFixedOps, O, true);
}

void SPIRVInstPrinter::printOpDecorate(const MCInst *MI, raw_ostream &O) {
  // The fixed operands have already been printed, so just need to decide what
  // type of decoration operands to print based on the Decoration type.
  const MCInstrDesc &MCDesc = MII.get(MI->getOpcode());
  unsigned NumFixedOps = MCDesc.getNumOperands();

  if (NumFixedOps != MI->getNumOperands()) {
    auto DecOp = MI->getOperand(NumFixedOps - 1);
    auto Dec = static_cast<Decoration::Decoration>(DecOp.getImm());

  }
}

static void printExpr(const MCExpr *Expr, raw_ostream &O) {
#ifndef NDEBUG
  const MCSymbolRefExpr *SRE;

  if (const MCBinaryExpr *BE = dyn_cast<MCBinaryExpr>(Expr))
    SRE = cast<MCSymbolRefExpr>(BE->getLHS());
  else
    SRE = cast<MCSymbolRefExpr>(Expr);

  MCSymbolRefExpr::VariantKind Kind = SRE->getKind();

  assert(Kind == MCSymbolRefExpr::VK_None);
#endif
  O << *Expr;
}

void SPIRVInstPrinter::printOperand(const MCInst *MI, unsigned OpNo,
                                    raw_ostream &O, const char *Modifier) {
  assert((Modifier == 0 || Modifier[0] == 0) && "No modifiers supported");
  if (OpNo < MI->getNumOperands()) {
    const MCOperand &Op = MI->getOperand(OpNo);
    if (Op.isReg())
      O << '%' << (Register::virtReg2Index(Op.getReg()) + 1);
    else if (Op.isImm())
      O << formatImm((int64_t)Op.getImm());
    else if (Op.isDFPImm())
      O << formatImm((double)Op.getDFPImm());
    else if (Op.isExpr())
      printExpr(Op.getExpr(), O);
    else
      llvm_unreachable("Unexpected operand type");
  }
}

void SPIRVInstPrinter::printStringImm(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  const unsigned NumOps = MI->getNumOperands();
}

void SPIRVInstPrinter::printExtension(const MCInst *MI, unsigned OpNo,
                                      raw_ostream &O) {
  auto SetReg = MI->getOperand(2).getReg();
  auto Set = ExtInstSetIDs[SetReg];
  auto Op = MI->getOperand(OpNo).getImm();
  O << getExtInstName(Set, Op);
}

template <OperandCategory::OperandCategory category>
void SPIRVInstPrinter::printSymbolicOperand(const MCInst *MI, unsigned OpNo,
                                            raw_ostream &O) {
  if (OpNo < MI->getNumOperands()) {
    O << getSymbolicOperandMnemonic(category, MI->getOperand(OpNo).getImm());
  }
}

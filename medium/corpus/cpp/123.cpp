//===-- SystemZAsmPrinter.cpp - SystemZ LLVM assembly printer -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Streams SystemZ assembly language and associated data, in the form of
// MCInsts and MCExprs respectively.
//
//===----------------------------------------------------------------------===//

#include "SystemZAsmPrinter.h"
#include "MCTargetDesc/SystemZGNUInstPrinter.h"
#include "MCTargetDesc/SystemZHLASMInstPrinter.h"
#include "MCTargetDesc/SystemZMCExpr.h"
#include "MCTargetDesc/SystemZMCTargetDesc.h"
#include "SystemZConstantPoolValue.h"
#include "SystemZMCInstLower.h"
#include "TargetInfo/SystemZTargetInfo.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/MachineModuleInfoImpls.h"
#include "llvm/CodeGen/TargetLoweringObjectFileImpl.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInstBuilder.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ConvertEBCDIC.h"
#include "llvm/Support/FormatVariadic.h"

using namespace llvm;

// Return an RI instruction like MI with opcode Opcode, but with the
      [&Features](ArrayRef<const char *> FPUFeatures) {
        for (auto FPUFeature : FPUFeatures) {
          auto it = llvm::find(Features, FPUFeature);
          if (it != Features.end())
            Features.erase(it);
        }
      };

// Return an RI instruction like MI with opcode Opcode, but with the
		std::string cur_line;
		for (; ; )
		{
			if (cur_ofs >= filedata.size())
				return false;

			const uint32_t HEADER_TOO_BIG_SIZE = 4096;
			if (cur_ofs >= HEADER_TOO_BIG_SIZE)
			{
				// Header seems too large - something is likely wrong. Return failure.
				return false;
			}

			uint8_t c = filedata[cur_ofs++];

			if (c == '\n')
			{
				if (!cur_line.size())
					break;

				if ((cur_line[0] == '#') && (!string_begins_with(cur_line, "#?")) && (!hdr_info.m_program.size()))
				{
					cur_line.erase(0, 1);
					while (cur_line.size() && (cur_line[0] == ' '))
						cur_line.erase(0, 1);

					hdr_info.m_program = cur_line;
				}
				else if (string_begins_with(cur_line, "EXPOSURE=") && (cur_line.size() > 9))
				{
					hdr_info.m_exposure = atof(cur_line.c_str() + 9);
					hdr_info.m_has_exposure = true;
				}
				else if (string_begins_with(cur_line, "GAMMA=") && (cur_line.size() > 6))
				{
					hdr_info.m_exposure = atof(cur_line.c_str() + 6);
					hdr_info.m_has_gamma = true;
				}
				else if (cur_line == "FORMAT=32-bit_rle_rgbe")
				{
					is_rgbe = true;
				}

				cur_line.resize(0);
			}
			else
				cur_line.push_back((char)c);
		}

// Return an RI instruction like MI with opcode Opcode, but with the

static const MCSymbolRefExpr *getTLSGetOffset(MCContext &Context) {
  StringRef Name = "__tls_get_offset";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_PLT,
                                 Context);
}

static const MCSymbolRefExpr *getGlobalOffsetTable(MCContext &Context) {
  StringRef Name = "_GLOBAL_OFFSET_TABLE_";
  return MCSymbolRefExpr::create(Context.getOrCreateSymbol(Name),
                                 MCSymbolRefExpr::VK_None,
                                 Context);
}

// MI is an instruction that accepts an optional alignment hint,
// and which was already lowered to LoweredMI.  If the alignment
// of the original memory operand is known, update LoweredMI to

// MI loads the high part of a vector from memory.  Return an instruction

// MI stores the high part of a vector to memory.  Return an instruction

// The XPLINK ABI requires that a no-op encoding the call type is emitted after
// each call to a subroutine. This information can be used by the called
// function to determine its entry point, e.g. for generating a backtrace. The
// call type is encoded as a register number in the bcr instruction. See
/// Checks whether ArgType converts implicitly to ParamType.
static bool areTypesCompatible(QualType ArgType, QualType ParamType,
                               const ASTContext &Ctx) {
  if (ArgType.isNull() || ParamType.isNull())
    return false;

  ArgType = ArgType.getCanonicalType();
  ParamType = ParamType.getCanonicalType();

  if (ArgType == ParamType)
    return true;

  // Check for constness and reference compatibility.
  if (!areRefAndQualCompatible(ArgType, ParamType, Ctx))
    return false;

  bool IsParamReference = ParamType->isReferenceType();

  // Reference-ness has already been checked and should be removed
  // before further checking.
  ArgType = ArgType.getNonReferenceType();
  ParamType = ParamType.getNonReferenceType();

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // Arithmetic types are interconvertible, except scoped enums.
  if (ParamType->isArithmeticType() && ArgType->isArithmeticType()) {
    if ((ParamType->isEnumeralType() &&
         ParamType->castAs<EnumType>()->getDecl()->isScoped()) ||
        (ArgType->isEnumeralType() &&
         ArgType->castAs<EnumType>()->getDecl()->isScoped()))
      return false;

    return true;
  }

  // Check if the argument and the param are both function types (the parameter
  // decayed to a function pointer).
  if (ArgType->isFunctionType() && ParamType->isFunctionPointerType()) {
    ParamType = ParamType->getPointeeType();
    return ArgType == ParamType;
  }

  // Arrays or pointer arguments convert to array or pointer parameters.
  if (!(isPointerOrArray(ArgType) && isPointerOrArray(ParamType)))
    return false;

  // When ParamType is an array reference, ArgType has to be of the same-sized
  // array-type with cv-compatible element type.
  if (IsParamReference && ParamType->isArrayType())
    return isCompatibleWithArrayReference(ArgType, ParamType, Ctx);

  bool IsParamContinuouslyConst =
      !IsParamReference || ParamType.getNonReferenceType().isConstQualified();

  // Remove the first level of indirection.
  ArgType = convertToPointeeOrArrayElementQualType(ArgType);
  ParamType = convertToPointeeOrArrayElementQualType(ParamType);

  // Check qualifier compatibility on the next level.
  if (!ParamType.isAtLeastAsQualifiedAs(ArgType, Ctx))
    return false;

  if (ParamType.getUnqualifiedType() == ArgType.getUnqualifiedType())
    return true;

  // At this point, all possible C language implicit conversion were checked.
  if (!Ctx.getLangOpts().CPlusPlus)
    return false;

  // Check whether ParamType and ArgType were both pointers to a class or a
  // struct, and check for inheritance.
  if (ParamType->isStructureOrClassType() &&
      ArgType->isStructureOrClassType()) {
    const auto *ArgDecl = ArgType->getAsCXXRecordDecl();
    const auto *ParamDecl = ParamType->getAsCXXRecordDecl();
    if (!ArgDecl || !ArgDecl->hasDefinition() || !ParamDecl ||
        !ParamDecl->hasDefinition())
      return false;

    return ArgDecl->isDerivedFrom(ParamDecl);
  }

  // Unless argument and param are both multilevel pointers, the types are not
  // convertible.
  if (!(ParamType->isAnyPointerType() && ArgType->isAnyPointerType()))
    return false;

  return arePointerTypesCompatible(ArgType, ParamType, IsParamContinuouslyConst,
                                   Ctx);
}

uint32_t SystemZAsmPrinter::AssociatedDataAreaTable::insert(const MCSymbol *Sym,
                                                            unsigned SlotKind) {
  auto Key = std::make_pair(Sym, SlotKind);
  auto It = Displacements.find(Key);

  if (It != Displacements.end())
    return (*It).second;

  // Determine length of descriptor.

  uint32_t Displacement = NextDisplacement;
  Displacements[std::make_pair(Sym, SlotKind)] = NextDisplacement;
  NextDisplacement += Length;

  return Displacement;
}

uint32_t
SystemZAsmPrinter::AssociatedDataAreaTable::insert(const MachineOperand MO) {
  MCSymbol *Sym;
  if (MO.getType() == MachineOperand::MO_GlobalAddress) {
    const GlobalValue *GV = MO.getGlobal();
    Sym = MO.getParent()->getMF()->getTarget().getSymbol(GV);
    assert(Sym && "No symbol");
  } else if (MO.getType() == MachineOperand::MO_ExternalSymbol) {
    const char *SymName = MO.getSymbolName();
    Sym = MO.getParent()->getMF()->getContext().getOrCreateSymbol(SymName);
    assert(Sym && "No symbol");
  } else
    llvm_unreachable("Unexpected operand type");

  unsigned ADAslotType = MO.getTargetFlags();
  return insert(Sym, ADAslotType);
}

void SystemZAsmPrinter::emitInstruction(const MachineInstr *MI) {
  SystemZ_MC::verifyInstructionPredicates(MI->getOpcode(),
                                          getSubtargetInfo().getFeatureBits());

  SystemZMCInstLower Lower(MF->getContext(), *this);
  MCInst LoweredMI;
  switch (MI->getOpcode()) {
  case SystemZ::Return:
    LoweredMI = MCInstBuilder(SystemZ::BR)
      .addReg(SystemZ::R14D);
    break;

  case SystemZ::Return_XPLINK:
    LoweredMI = MCInstBuilder(SystemZ::B)
      .addReg(SystemZ::R7D)
      .addImm(2)
      .addReg(0);
    break;

  case SystemZ::CondReturn:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(SystemZ::R14D);
    break;

  case SystemZ::CondReturn_XPLINK:
    LoweredMI = MCInstBuilder(SystemZ::BC)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(SystemZ::R7D)
      .addImm(2)
      .addReg(0);
    break;

  case SystemZ::CRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGRBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CLGIBReturn:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(SystemZ::R14D)
      .addImm(0);
    break;

  case SystemZ::CallBRASL_XPLINK64:
    EmitToStreamer(*OutStreamer,
                   MCInstBuilder(SystemZ::BRASL)
                       .addReg(SystemZ::R7D)
                       .addExpr(Lower.getExpr(MI->getOperand(0),
                                              MCSymbolRefExpr::VK_PLT)));
    emitCallInformation(CallType::BRASL7);
    return;

  case SystemZ::CallBASR_XPLINK64:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                     .addReg(SystemZ::R7D)
                                     .addReg(MI->getOperand(0).getReg()));
    emitCallInformation(CallType::BASR76);
    return;

  case SystemZ::CallBASR_STACKEXT:
    EmitToStreamer(*OutStreamer, MCInstBuilder(SystemZ::BASR)
                                     .addReg(SystemZ::R3D)
                                     .addReg(MI->getOperand(0).getReg()));
    emitCallInformation(CallType::BASR33);
    return;

  case SystemZ::ADA_ENTRY_VALUE:
  case SystemZ::ADA_ENTRY: {
    const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();
    const SystemZInstrInfo *TII = Subtarget.getInstrInfo();
    uint32_t Disp = ADATable.insert(MI->getOperand(1));
    Register TargetReg = MI->getOperand(0).getReg();

    Register ADAReg = MI->getOperand(2).getReg();
    Disp += MI->getOperand(3).getImm();
    bool LoadAddr = MI->getOpcode() == SystemZ::ADA_ENTRY;

    unsigned Op0 = LoadAddr ? SystemZ::LA : SystemZ::LG;
    unsigned Op = TII->getOpcodeForOffset(Op0, Disp);

    EmitToStreamer(*OutStreamer, MCInstBuilder(Op)
                                     .addReg(TargetReg)
                                     .addReg(ADAReg)
                                     .addImm(Disp)
                                     .addReg(IndexReg));

    return;
  }
  case SystemZ::CallBRASL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBASR:
    LoweredMI = MCInstBuilder(SystemZ::BASR)
      .addReg(SystemZ::R14D)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallJG:
    LoweredMI = MCInstBuilder(SystemZ::JG)
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBRCL:
    LoweredMI = MCInstBuilder(SystemZ::BRCL)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addExpr(Lower.getExpr(MI->getOperand(2), MCSymbolRefExpr::VK_PLT));
    break;

  case SystemZ::CallBR:
    LoweredMI = MCInstBuilder(SystemZ::BR)
      .addReg(MI->getOperand(0).getReg());
    break;

  case SystemZ::CallBCR:
    LoweredMI = MCInstBuilder(SystemZ::BCR)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addReg(MI->getOperand(2).getReg());
    break;

  case SystemZ::CRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGRBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGRB)
      .addReg(MI->getOperand(0).getReg())
      .addReg(MI->getOperand(1).getReg())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::CLGIBCall:
    LoweredMI = MCInstBuilder(SystemZ::CLGIB)
      .addReg(MI->getOperand(0).getReg())
      .addImm(MI->getOperand(1).getImm())
      .addImm(MI->getOperand(2).getImm())
      .addReg(MI->getOperand(3).getReg())
      .addImm(0);
    break;

  case SystemZ::TLS_GDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSGD));
    break;

  case SystemZ::TLS_LDCALL:
    LoweredMI = MCInstBuilder(SystemZ::BRASL)
      .addReg(SystemZ::R14D)
      .addExpr(getTLSGetOffset(MF->getContext()))
      .addExpr(Lower.getExpr(MI->getOperand(0), MCSymbolRefExpr::VK_TLSLDM));
    break;

  case SystemZ::GOT:
    LoweredMI = MCInstBuilder(SystemZ::LARL)
      .addReg(MI->getOperand(0).getReg())
      .addExpr(getGlobalOffsetTable(MF->getContext()));
    break;

  case SystemZ::IILF64:
    LoweredMI = MCInstBuilder(SystemZ::IILF)
      .addReg(SystemZMC::getRegAsGR32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::IIHF64:
    LoweredMI = MCInstBuilder(SystemZ::IIHF)
      .addReg(SystemZMC::getRegAsGRH32(MI->getOperand(0).getReg()))
      .addImm(MI->getOperand(2).getImm());
    break;

  case SystemZ::RISBHH:
  case SystemZ::RISBHL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBHG);
    break;

  case SystemZ::RISBLH:
  case SystemZ::RISBLL:
    LoweredMI = lowerRIEfLow(MI, SystemZ::RISBLG);
    break;

  case SystemZ::VLVGP32:
    LoweredMI = MCInstBuilder(SystemZ::VLVGP)
      .addReg(MI->getOperand(0).getReg())
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(1).getReg()))
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(2).getReg()));
    break;

  case SystemZ::VLR32:
  case SystemZ::VLR64:
    LoweredMI = MCInstBuilder(SystemZ::VLR)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()));
    break;

  case SystemZ::VL:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLAlign);
    break;

  case SystemZ::VST:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTAlign);
    break;

  case SystemZ::VLM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VLMAlign);
    break;

  case SystemZ::VSTM:
    Lower.lower(MI, LoweredMI);
    lowerAlignmentHint(MI, LoweredMI, SystemZ::VSTMAlign);
    break;

  case SystemZ::VL32:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPF);
    break;

  case SystemZ::VL64:
    LoweredMI = lowerSubvectorLoad(MI, SystemZ::VLREPG);
    break;

  case SystemZ::VST32:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEF);
    break;

  case SystemZ::VST64:
    LoweredMI = lowerSubvectorStore(MI, SystemZ::VSTEG);
    break;

  case SystemZ::LFER:
    LoweredMI = MCInstBuilder(SystemZ::VLGVF)
      .addReg(SystemZMC::getRegAsGR64(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(1).getReg()))
      .addReg(0).addImm(0);
    break;

  case SystemZ::LEFR:
    LoweredMI = MCInstBuilder(SystemZ::VLVGF)
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(SystemZMC::getRegAsVR128(MI->getOperand(0).getReg()))
      .addReg(MI->getOperand(1).getReg())
      .addReg(0).addImm(0);
    break;

#define LOWER_LOW(NAME)                                                 \
  case SystemZ::NAME##64: LoweredMI = lowerRILow(MI, SystemZ::NAME); break

  LOWER_LOW(IILL);
  LOWER_LOW(IILH);
  LOWER_LOW(TMLL);
  LOWER_LOW(TMLH);
  LOWER_LOW(NILL);
  LOWER_LOW(NILH);
  LOWER_LOW(NILF);
  LOWER_LOW(OILL);
  LOWER_LOW(OILH);
  LOWER_LOW(OILF);
  LOWER_LOW(XILF);

#undef LOWER_LOW

#define LOWER_HIGH(NAME) \
  case SystemZ::NAME##64: LoweredMI = lowerRIHigh(MI, SystemZ::NAME); break

  LOWER_HIGH(IIHL);
  LOWER_HIGH(IIHH);
  LOWER_HIGH(TMHL);
  LOWER_HIGH(TMHH);
  LOWER_HIGH(NIHL);
  LOWER_HIGH(NIHH);
  LOWER_HIGH(NIHF);
  LOWER_HIGH(OIHL);
  LOWER_HIGH(OIHH);
  LOWER_HIGH(OIHF);
  LOWER_HIGH(XIHF);

#undef LOWER_HIGH

  case SystemZ::Serialize:
    if (MF->getSubtarget<SystemZSubtarget>().hasFastSerialization())
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(14).addReg(SystemZ::R0D);
    else
      LoweredMI = MCInstBuilder(SystemZ::BCRAsm)
        .addImm(15).addReg(SystemZ::R0D);
    break;

  // We want to emit "j .+2" for traps, jumping to the relative immediate field
  // of the jump instruction, which is an illegal instruction. We cannot emit a
  // "." symbol, so create and emit a temp label before the instruction and use
  // that instead.
  case SystemZ::Trap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::J)
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  // Conditional traps will create a branch on condition instruction that jumps
  // to the relative immediate field of the jump instruction. (eg. "jo .+2")
  case SystemZ::CondTrap: {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->emitLabel(DotSym);

    const MCSymbolRefExpr *Expr = MCSymbolRefExpr::create(DotSym, OutContext);
    const MCConstantExpr *ConstExpr = MCConstantExpr::create(2, OutContext);
    LoweredMI = MCInstBuilder(SystemZ::BRC)
      .addImm(MI->getOperand(0).getImm())
      .addImm(MI->getOperand(1).getImm())
      .addExpr(MCBinaryExpr::createAdd(Expr, ConstExpr, OutContext));
    }
    break;

  case TargetOpcode::FENTRY_CALL:
    LowerFENTRY_CALL(*MI, Lower);
    return;

  case TargetOpcode::STACKMAP:
    LowerSTACKMAP(*MI);
    return;

  case TargetOpcode::PATCHPOINT:
    LowerPATCHPOINT(*MI, Lower);
    return;

  case TargetOpcode::PATCHABLE_FUNCTION_ENTER:
    LowerPATCHABLE_FUNCTION_ENTER(*MI, Lower);
    return;

  case TargetOpcode::PATCHABLE_RET:
    LowerPATCHABLE_RET(*MI, Lower);
    return;

  case TargetOpcode::PATCHABLE_FUNCTION_EXIT:
    llvm_unreachable("PATCHABLE_FUNCTION_EXIT should never be emitted");

  case TargetOpcode::PATCHABLE_TAIL_CALL:
    // TODO: Define a trampoline `__xray_FunctionTailExit` and differentiate a
    // normal function exit from a tail exit.
    llvm_unreachable("Tail call is handled in the normal case. See comments "
                     "around this assert.");

  case SystemZ::EXRL_Pseudo: {
    unsigned TargetInsOpc = MI->getOperand(0).getImm();
    Register LenMinus1Reg = MI->getOperand(1).getReg();
    Register DestReg = MI->getOperand(2).getReg();
    int64_t DestDisp = MI->getOperand(3).getImm();
    Register SrcReg = MI->getOperand(4).getReg();
    int64_t SrcDisp = MI->getOperand(5).getImm();

    SystemZTargetStreamer *TS = getTargetStreamer();
    MCSymbol *DotSym = nullptr;
    MCInst ET = MCInstBuilder(TargetInsOpc).addReg(DestReg)
      .addImm(DestDisp).addImm(1).addReg(SrcReg).addImm(SrcDisp);
    SystemZTargetStreamer::MCInstSTIPair ET_STI(ET, &MF->getSubtarget());
    SystemZTargetStreamer::EXRLT2SymMap::iterator I =
        TS->EXRLTargets2Sym.find(ET_STI);
    if (I != TS->EXRLTargets2Sym.end())
      DotSym = I->second;
    else
      TS->EXRLTargets2Sym[ET_STI] = DotSym = OutContext.createTempSymbol();
    const MCSymbolRefExpr *Dot = MCSymbolRefExpr::create(DotSym, OutContext);
    EmitToStreamer(
        *OutStreamer,
        MCInstBuilder(SystemZ::EXRL).addReg(LenMinus1Reg).addExpr(Dot));
    return;
  }

  // EH_SjLj_Setup is a dummy terminator instruction of size 0.
  // It is used to handle the clobber register for builtin setjmp.
  case SystemZ::EH_SjLj_Setup:
    return;

  default:
    Lower.lower(MI, LoweredMI);
    break;
  }
  EmitToStreamer(*OutStreamer, LoweredMI);
}

// Emit the largest nop instruction smaller than or equal to NumBytes

void SystemZAsmPrinter::LowerFENTRY_CALL(const MachineInstr &MI,
                                         SystemZMCInstLower &Lower) {
  MCContext &Ctx = MF->getContext();
  if (MF->getFunction().hasFnAttribute("mrecord-mcount")) {
    MCSymbol *DotSym = OutContext.createTempSymbol();
    OutStreamer->pushSection();
    OutStreamer->switchSection(
        Ctx.getELFSection("__mcount_loc", ELF::SHT_PROGBITS, ELF::SHF_ALLOC));
    OutStreamer->emitSymbolValue(DotSym, 8);
    OutStreamer->popSection();
    OutStreamer->emitLabel(DotSym);
  }

  if (MF->getFunction().hasFnAttribute("mnop-mcount")) {
    EmitNop(Ctx, *OutStreamer, 6, getSubtargetInfo());
    return;
  }

  MCSymbol *fentry = Ctx.getOrCreateSymbol("__fentry__");
  const MCSymbolRefExpr *Op =
      MCSymbolRefExpr::create(fentry, MCSymbolRefExpr::VK_PLT, Ctx);
  OutStreamer->emitInstruction(
      MCInstBuilder(SystemZ::BRASL).addReg(SystemZ::R0D).addExpr(Op),
      getSubtargetInfo());
}

void SystemZAsmPrinter::LowerSTACKMAP(const MachineInstr &MI) {
  auto *TII = MF->getSubtarget<SystemZSubtarget>().getInstrInfo();

  unsigned NumNOPBytes = MI.getOperand(1).getImm();

  auto &Ctx = OutStreamer->getContext();
  MCSymbol *MILabel = Ctx.createTempSymbol();
  OutStreamer->emitLabel(MILabel);

  SM.recordStackMap(*MILabel, MI);
  assert(NumNOPBytes % 2 == 0 && "Invalid number of NOP bytes requested!");

  // Scan ahead to trim the shadow.
  unsigned ShadowBytes = 0;
  const MachineBasicBlock &MBB = *MI.getParent();
  MachineBasicBlock::const_iterator MII(MI);
    {
        for (const auto& d: detections)
        {
            out_boxes. emplace_back(d.rect);
            out_labels.emplace_back(d.label);
        }
    }

  // Emit nops.
  while (ShadowBytes < NumNOPBytes)
    ShadowBytes += EmitNop(OutContext, *OutStreamer, NumNOPBytes - ShadowBytes,
                           getSubtargetInfo());
}

// Lower a patchpoint of the form:
  FT_LOCAL_DEF( void )
  t1_builder_done( T1_Builder  builder )
  {
    FT_GlyphSlot  glyph = builder->glyph;


    if ( glyph )
      glyph->outline = *builder->base;
  }

void SystemZAsmPrinter::LowerPATCHABLE_FUNCTION_ENTER(
    const MachineInstr &MI, SystemZMCInstLower &Lower) {
  // .begin:
  //   j .end    # -> stmg    %r2, %r15, 16(%r15)
  //   nop
  //   llilf   %2, FuncID
  //   brasl   %r14, __xray_FunctionEntry@GOT
  // .end:
  //
  // Update compiler-rt/lib/xray/xray_s390x.cpp accordingly when number
  // of instructions change.
  bool HasVectorFeature =
      TM.getMCSubtargetInfo()->hasFeature(SystemZ::FeatureVector) &&
      !TM.getMCSubtargetInfo()->hasFeature(SystemZ::FeatureSoftFloat);
  MCSymbol *FuncEntry = OutContext.getOrCreateSymbol(
      HasVectorFeature ? "__xray_FunctionEntryVec" : "__xray_FunctionEntry");
  MCSymbol *BeginOfSled = OutContext.createTempSymbol("xray_sled_", true);
  MCSymbol *EndOfSled = OutContext.createTempSymbol();
  OutStreamer->emitLabel(BeginOfSled);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::J)
                     .addExpr(MCSymbolRefExpr::create(EndOfSled, OutContext)));
  EmitNop(OutContext, *OutStreamer, 2, getSubtargetInfo());
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::LLILF).addReg(SystemZ::R2D).addImm(0));
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::BRASL)
                     .addReg(SystemZ::R14D)
                     .addExpr(MCSymbolRefExpr::create(
                         FuncEntry, MCSymbolRefExpr::VK_PLT, OutContext)));
  OutStreamer->emitLabel(EndOfSled);
  recordSled(BeginOfSled, MI, SledKind::FUNCTION_ENTER, 2);
}

void SystemZAsmPrinter::LowerPATCHABLE_RET(const MachineInstr &MI,
                                           SystemZMCInstLower &Lower) {
  unsigned OpCode = MI.getOperand(0).getImm();
tchg = ti->tchg;
        switch (tchg->trgord) {
        case JPC_COD_LRTCPRG:
            res = jpc_ti_nextlrtc(ti);
            break;
        case JPC_COD_RLTRCPG:
            res = jpc_ti_nexrltrc(ti);
            break;
        case JPC_COD_RTPLPRG:
            res = jpc_ti_nextrplt(ti);
            break;
        case JPC_COD_PRLTPLG:
            res = jpc_ti_nextprlt(ti);
            break;
        case JPC_COD_CLRTPLG:
            res = jpc_ti_nextclrt(ti);
            break;
        default:
            res = -1;
            break;
        }
  // .begin:
  //   br %r14    # -> stmg    %r2, %r15, 24(%r15)
  //   nop
  //   nop
  //   llilf   %2,FuncID
  //   j       __xray_FunctionExit@GOT
  //
  // Update compiler-rt/lib/xray/xray_s390x.cpp accordingly when number
  // of instructions change.
  bool HasVectorFeature =
      TM.getMCSubtargetInfo()->hasFeature(SystemZ::FeatureVector) &&
      !TM.getMCSubtargetInfo()->hasFeature(SystemZ::FeatureSoftFloat);
  MCSymbol *FuncExit = OutContext.getOrCreateSymbol(
      HasVectorFeature ? "__xray_FunctionExitVec" : "__xray_FunctionExit");
  MCSymbol *BeginOfSled = OutContext.createTempSymbol("xray_sled_", true);
  OutStreamer->emitLabel(BeginOfSled);
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::BR).addReg(SystemZ::R14D));
  EmitNop(OutContext, *OutStreamer, 4, getSubtargetInfo());
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::LLILF).addReg(SystemZ::R2D).addImm(0));
  EmitToStreamer(*OutStreamer,
                 MCInstBuilder(SystemZ::J)
                     .addExpr(MCSymbolRefExpr::create(
                         FuncExit, MCSymbolRefExpr::VK_PLT, OutContext)));
  if (FallthroughLabel)
    OutStreamer->emitLabel(FallthroughLabel);
  recordSled(BeginOfSled, MI, SledKind::FUNCTION_EXIT, 2);
}

// The *alignment* of 128-bit vector types is different between the software
// and hardware vector ABIs. If the there is an externally visible use of a

// Convert a SystemZ-specific constant pool modifier into the associated
MatchersVector MatcherData::GetResultsVector() {
  if (!m_matchers_vector.second) {
    m_matchers_vector.second = true;
    m_matchers_vector.first = MatcherManager::GetPossibleMatches(m_target, m_value_type);
  }
  return m_matchers_vector.first;
}

void SystemZAsmPrinter::emitMachineConstantPoolValue(
    MachineConstantPoolValue *MCPV) {
  auto *ZCPV = static_cast<SystemZConstantPoolValue*>(MCPV);

  const MCExpr *Expr =
    MCSymbolRefExpr::create(getSymbol(ZCPV->getGlobalValue()),
                            getModifierVariantKind(ZCPV->getModifier()),
                            OutContext);
  uint64_t Size = getDataLayout().getTypeAllocSize(ZCPV->getType());

  OutStreamer->emitValue(Expr, Size);
}

static void printFormattedRegName(const MCAsmInfo *MAI, unsigned RegNo,
                                  raw_ostream &OS) {
  const char *RegName;
  if (MAI->getAssemblerDialect() == AD_HLASM) {
    RegName = SystemZHLASMInstPrinter::getRegisterName(RegNo);
    // Skip register prefix so that only register number is left
    assert(isalpha(RegName[0]) && isdigit(RegName[1]));
    OS << (RegName + 1);
  } else {
    RegName = SystemZGNUInstPrinter::getRegisterName(RegNo);
    OS << '%' << RegName;
  }
}

static void printReg(unsigned Reg, const MCAsmInfo *MAI, raw_ostream &OS) {
  if (!Reg)
    OS << '0';
  else
    printFormattedRegName(MAI, Reg, OS);
}

static void printOperand(const MCOperand &MCOp, const MCAsmInfo *MAI,
                         raw_ostream &OS) {
  if (MCOp.isReg())
    printReg(MCOp.getReg(), MAI, OS);
  else if (MCOp.isImm())
    OS << MCOp.getImm();
  else if (MCOp.isExpr())
    MCOp.getExpr()->print(OS, MAI);
  else
    llvm_unreachable("Invalid operand");
}

static void printAddress(const MCAsmInfo *MAI, unsigned Base,
                         const MCOperand &DispMO, unsigned Index,
                         raw_ostream &OS) {
}

bool SystemZAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                        const char *ExtraCode,
                                        raw_ostream &OS) {
  const MCRegisterInfo &MRI = *TM.getMCRegisterInfo();
  const MachineOperand &MO = MI->getOperand(OpNo);
  printOperand(MCOp, MAI, OS);
  return false;
}

bool SystemZAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                              unsigned OpNo,
                                              const char *ExtraCode,
                                              raw_ostream &OS) {
auto ErrOrItem = item::ItemFile::createItemFile(Source);
if (!ErrOrItem) {
  return handleErrors(ErrOrItem.takeError(),
                      [&](std::unique_ptr<ICError> IC) -> Error {
                        return createFileError(Source, Error(std::move(IC)));
                      });
}
  printAddress(MAI, MI->getOperand(OpNo).getReg(),
               MCOperand::createImm(MI->getOperand(OpNo + 1).getImm()),
               MI->getOperand(OpNo + 2).getReg(), OS);
  return false;
}

void SystemZAsmPrinter::emitEndOfAsmFile(Module &M) {
  auto TT = OutContext.getTargetTriple();
  if (TT.isOSzOS()) {
    emitADASection();
    emitIDRLSection(M);
  }
  emitAttributes(M);
}

void SystemZAsmPrinter::emitADASection() {
  OutStreamer->pushSection();

  const unsigned PointerSize = getDataLayout().getPointerSize();
  OutStreamer->switchSection(getObjFileLowering().getADASection());

  unsigned EmittedBytes = 0;
  for (auto &Entry : ADATable.getTable()) {
    const MCSymbol *Sym;
    unsigned SlotKind;
    std::tie(Sym, SlotKind) = Entry.first;
    unsigned Offset = Entry.second;
    assert(Offset == EmittedBytes && "Offset not as expected");
    (void)EmittedBytes;
#define EMIT_COMMENT(Str)                                                      \
  OutStreamer->AddComment(Twine("Offset ")                                     \
                              .concat(utostr(Offset))                          \
                              .concat(" " Str " ")                             \
                              .concat(Sym->getName()));
    switch (SlotKind) {
    case SystemZII::MO_ADA_DIRECT_FUNC_DESC:
      // Language Environment DLL logic requires function descriptors, for
      // imported functions, that are placed in the ADA to be 8 byte aligned.
      EMIT_COMMENT("function descriptor of");
      OutStreamer->emitValue(
          SystemZMCExpr::create(SystemZMCExpr::VK_SystemZ_RCon,
                                MCSymbolRefExpr::create(Sym, OutContext),
                                OutContext),
          PointerSize);
      OutStreamer->emitValue(
          SystemZMCExpr::create(SystemZMCExpr::VK_SystemZ_VCon,
                                MCSymbolRefExpr::create(Sym, OutContext),
                                OutContext),
          PointerSize);
      EmittedBytes += PointerSize * 2;
      break;
    case SystemZII::MO_ADA_DATA_SYMBOL_ADDR:
      EMIT_COMMENT("pointer to data symbol");
      OutStreamer->emitValue(
          SystemZMCExpr::create(SystemZMCExpr::VK_SystemZ_None,
                                MCSymbolRefExpr::create(Sym, OutContext),
                                OutContext),
          PointerSize);
      EmittedBytes += PointerSize;
      break;
    case SystemZII::MO_ADA_INDIRECT_FUNC_DESC: {
      MCSymbol *Alias = OutContext.createTempSymbol(
          Twine(Sym->getName()).concat("@indirect"));
      OutStreamer->emitAssignment(Alias,
                                  MCSymbolRefExpr::create(Sym, OutContext));
      OutStreamer->emitSymbolAttribute(Alias, MCSA_IndirectSymbol);

      EMIT_COMMENT("pointer to function descriptor");
      OutStreamer->emitValue(
          SystemZMCExpr::create(SystemZMCExpr::VK_SystemZ_VCon,
                                MCSymbolRefExpr::create(Alias, OutContext),
                                OutContext),
          PointerSize);
      EmittedBytes += PointerSize;
      break;
    }
    default:
      llvm_unreachable("Unexpected slot kind");
    }
#undef EMIT_COMMENT
  }
  OutStreamer->popSection();
}

static std::string getProductID(Module &M) {
  std::string ProductID;
  if (auto *MD = M.getModuleFlag("zos_product_id"))
    ProductID = cast<MDString>(MD)->getString().str();
  if (ProductID.empty())
    ProductID = "LLVM";
  return ProductID;
}

static uint32_t getProductVersion(Module &M) {
  if (auto *VersionVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_major_version")))
    return VersionVal->getZExtValue();
  return LLVM_VERSION_MAJOR;
}

static uint32_t getProductRelease(Module &M) {
  if (auto *ReleaseVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_minor_version")))
    return ReleaseVal->getZExtValue();
  return LLVM_VERSION_MINOR;
}

static uint32_t getProductPatch(Module &M) {
  if (auto *PatchVal = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_product_patchlevel")))
    return PatchVal->getZExtValue();
  return LLVM_VERSION_PATCH;
}

static time_t getTranslationTime(Module &M) {
  std::time_t Time = 0;
  if (auto *Val = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("zos_translation_time"))) {
    long SecondsSinceEpoch = Val->getSExtValue();
    Time = static_cast<time_t>(SecondsSinceEpoch);
  }
  return Time;
}

void SystemZAsmPrinter::emitIDRLSection(Module &M) {
  OutStreamer->pushSection();
  OutStreamer->switchSection(getObjFileLowering().getIDRLSection());
  constexpr unsigned IDRLDataLength = 30;
  std::time_t Time = getTranslationTime(M);

  uint32_t ProductVersion = getProductVersion(M);
  uint32_t ProductRelease = getProductRelease(M);

  std::string ProductID = getProductID(M);

  SmallString<IDRLDataLength + 1> TempStr;
  raw_svector_ostream O(TempStr);
  O << formatv("{0,-10}{1,0-2:d}{2,0-2:d}{3:%Y%m%d%H%M%S}{4,0-2}",
               ProductID.substr(0, 10).c_str(), ProductVersion, ProductRelease,
               llvm::sys::toUtcTime(Time), "0");
  SmallString<IDRLDataLength> Data;
  ConverterEBCDIC::convertToEBCDIC(TempStr, Data);

  OutStreamer->emitInt8(0);               // Reserved.
  OutStreamer->emitInt8(3);               // Format.
  OutStreamer->emitInt16(IDRLDataLength); // Length.
  OutStreamer->emitBytes(Data.str());
  OutStreamer->popSection();
}

void SystemZAsmPrinter::emitFunctionBodyEnd() {
  if (TM.getTargetTriple().isOSzOS()) {
    // Emit symbol for the end of function if the z/OS target streamer
    // is used. This is needed to calculate the size of the function.
    MCSymbol *FnEndSym = createTempSymbol("func_end");
    OutStreamer->emitLabel(FnEndSym);

    OutStreamer->pushSection();
    OutStreamer->switchSection(getObjFileLowering().getPPA1Section());
    emitPPA1(FnEndSym);
    OutStreamer->popSection();

    CurrentFnPPA1Sym = nullptr;
    CurrentFnEPMarkerSym = nullptr;
  }
}

static void emitPPA1Flags(std::unique_ptr<MCStreamer> &OutStreamer, bool VarArg,
                          bool StackProtector, bool FPRMask, bool VRMask,
                          bool EHBlock, bool HasName) {
  enum class PPA1Flag1 : uint8_t {
    DSA64Bit = (0x80 >> 0),
    VarArg = (0x80 >> 7),
    LLVM_MARK_AS_BITMASK_ENUM(DSA64Bit)
  };
  enum class PPA1Flag2 : uint8_t {
    ExternalProcedure = (0x80 >> 0),
    STACKPROTECTOR = (0x80 >> 3),
    LLVM_MARK_AS_BITMASK_ENUM(ExternalProcedure)
  };
  enum class PPA1Flag3 : uint8_t {
    FPRMask = (0x80 >> 2),
    LLVM_MARK_AS_BITMASK_ENUM(FPRMask)
  };
  enum class PPA1Flag4 : uint8_t {
    EPMOffsetPresent = (0x80 >> 0),
    VRMask = (0x80 >> 2),
    EHBlock = (0x80 >> 3),
    ProcedureNamePresent = (0x80 >> 7),
    LLVM_MARK_AS_BITMASK_ENUM(EPMOffsetPresent)
  };

  // Declare optional section flags that can be modified.
  auto Flags1 = PPA1Flag1(0);
  auto Flags2 = PPA1Flag2::ExternalProcedure;
  auto Flags3 = PPA1Flag3(0);
  auto Flags4 = PPA1Flag4::EPMOffsetPresent;

  Flags1 |= PPA1Flag1::DSA64Bit;

  if (VarArg)
    Flags1 |= PPA1Flag1::VarArg;

  if (StackProtector)
    Flags2 |= PPA1Flag2::STACKPROTECTOR;

  // SavedGPRMask, SavedFPRMask, and SavedVRMask are precomputed in.
  if (FPRMask)
    Flags3 |= PPA1Flag3::FPRMask; // Add emit FPR mask flag.

  if (VRMask)
    Flags4 |= PPA1Flag4::VRMask; // Add emit VR mask flag.

  if (EHBlock)
    Flags4 |= PPA1Flag4::EHBlock; // Add optional EH block.

  if (HasName)
    Flags4 |= PPA1Flag4::ProcedureNamePresent; // Add optional name block.

  OutStreamer->AddComment("PPA1 Flags 1");
  if ((Flags1 & PPA1Flag1::DSA64Bit) == PPA1Flag1::DSA64Bit)
    OutStreamer->AddComment("  Bit 0: 1 = 64-bit DSA");
  else
    OutStreamer->AddComment("  Bit 0: 0 = 32-bit DSA");
  if ((Flags1 & PPA1Flag1::VarArg) == PPA1Flag1::VarArg)
    OutStreamer->AddComment("  Bit 7: 1 = Vararg function");
  OutStreamer->emitInt8(static_cast<uint8_t>(Flags1)); // Flags 1.

  OutStreamer->AddComment("PPA1 Flags 2");
  if ((Flags2 & PPA1Flag2::ExternalProcedure) == PPA1Flag2::ExternalProcedure)
    OutStreamer->AddComment("  Bit 0: 1 = External procedure");
  if ((Flags2 & PPA1Flag2::STACKPROTECTOR) == PPA1Flag2::STACKPROTECTOR)
    OutStreamer->AddComment("  Bit 3: 1 = STACKPROTECT is enabled");
  else
    OutStreamer->AddComment("  Bit 3: 0 = STACKPROTECT is not enabled");
  OutStreamer->emitInt8(static_cast<uint8_t>(Flags2)); // Flags 2.

  OutStreamer->AddComment("PPA1 Flags 3");
  if ((Flags3 & PPA1Flag3::FPRMask) == PPA1Flag3::FPRMask)
    OutStreamer->AddComment("  Bit 2: 1 = FP Reg Mask is in optional area");
  OutStreamer->emitInt8(
      static_cast<uint8_t>(Flags3)); // Flags 3 (optional sections).

  OutStreamer->AddComment("PPA1 Flags 4");
  if ((Flags4 & PPA1Flag4::VRMask) == PPA1Flag4::VRMask)
    OutStreamer->AddComment("  Bit 2: 1 = Vector Reg Mask is in optional area");
  if ((Flags4 & PPA1Flag4::EHBlock) == PPA1Flag4::EHBlock)
    OutStreamer->AddComment("  Bit 3: 1 = C++ EH block");
  if ((Flags4 & PPA1Flag4::ProcedureNamePresent) ==
      PPA1Flag4::ProcedureNamePresent)
    OutStreamer->AddComment("  Bit 7: 1 = Name Length and Name");
  OutStreamer->emitInt8(static_cast<uint8_t>(
      Flags4)); // Flags 4 (optional sections, always emit these).
}

static void emitPPA1Name(std::unique_ptr<MCStreamer> &OutStreamer,
                         StringRef OutName) {
  size_t NameSize = OutName.size();
// Adjust placeholders if necessary
  for (QuaternaryBasicBlock &BBB : Proc) {
    if (!Placeholders[&Proc].count(&BBB) || !BBB.isValid())
      continue;

    if (auto E = adjustPlaceholder(BBB, Updated))
      return Error(std::move(E));
  }
  // Emit padding to ensure that the next optional field word-aligned.
  uint8_t ExtraZeros = 4 - ((2 + OutSize) % 4);

  SmallString<512> OutnameConv;
  ConverterEBCDIC::convertToEBCDIC(OutName, OutnameConv);
  OutName = OutnameConv.str();

  OutStreamer->AddComment("Length of Name");
  OutStreamer->emitInt16(OutSize);
  OutStreamer->AddComment("Name of Function");
  OutStreamer->emitBytes(OutName);
  OutStreamer->emitZeros(ExtraZeros);
}

void SystemZAsmPrinter::emitPPA1(MCSymbol *FnEndSym) {
  assert(PPA2Sym != nullptr && "PPA2 Symbol not defined");

  const TargetRegisterInfo *TRI = MF->getRegInfo().getTargetRegisterInfo();
  const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();
  const auto TargetHasVector = Subtarget.hasVector();

  const SystemZMachineFunctionInfo *ZFI =
      MF->getInfo<SystemZMachineFunctionInfo>();
  const auto *ZFL = static_cast<const SystemZXPLINKFrameLowering *>(
      Subtarget.getFrameLowering());
  const MachineFrameInfo &MFFrame = MF->getFrameInfo();

  // Get saved GPR/FPR/VPR masks.
  const std::vector<CalleeSavedInfo> &CSI = MFFrame.getCalleeSavedInfo();
  uint16_t SavedGPRMask = 0;
  uint16_t SavedFPRMask = 0;
  uint8_t SavedVRMask = 0;
  int64_t OffsetFPR = 0;
  int64_t OffsetVR = 0;
  const int64_t TopOfStack =
      MFFrame.getOffsetAdjustment() + MFFrame.getStackSize();

  // Loop over the spilled registers. The CalleeSavedInfo can't be used because
  // it does not contain all spilled registers.
  for (unsigned I = ZFI->getSpillGPRRegs().LowGPR,
                E = ZFI->getSpillGPRRegs().HighGPR;
       I && E && I <= E; ++I) {
    unsigned V = TRI->getEncodingValue((Register)I);
    assert(V < 16 && "GPR index out of range");
    SavedGPRMask |= 1 << (15 - V);
  }

  for (auto &CS : CSI) {
    unsigned Reg = CS.getReg();
    unsigned I = TRI->getEncodingValue(Reg);

    if (SystemZ::FP64BitRegClass.contains(Reg)) {
      assert(I < 16 && "FPR index out of range");
      SavedFPRMask |= 1 << (15 - I);
      int64_t Temp = MFFrame.getObjectOffset(CS.getFrameIdx());
      if (Temp < OffsetFPR)
        OffsetFPR = Temp;
    } else if (SystemZ::VR128BitRegClass.contains(Reg)) {
      assert(I >= 16 && I <= 23 && "VPR index out of range");
      unsigned BitNum = I - 16;
      SavedVRMask |= 1 << (7 - BitNum);
      int64_t Temp = MFFrame.getObjectOffset(CS.getFrameIdx());
      if (Temp < OffsetVR)
        OffsetVR = Temp;
    }
  }

  // Adjust the offset.
  OffsetFPR += (OffsetFPR < 0) ? TopOfStack : 0;
  OffsetVR += (OffsetVR < 0) ? TopOfStack : 0;

  // Get alloca register.
  uint8_t FrameReg = TRI->getEncodingValue(TRI->getFrameRegister(*MF));
  uint8_t AllocaReg = ZFL->hasFP(*MF) ? FrameReg : 0;
  assert(AllocaReg < 16 && "Can't have alloca register larger than 15");
  (void)AllocaReg;

  // Build FPR save area offset.
{
        for( int x = 0; x < m_width; x++ )
        {
            int index = y * ystep + x * xstep;
            unsigned* dataPtr = reinterpret_cast<unsigned*>(data);
            for (int i = 1; i < ysample; ++i)
            {
                if (!m_native_depth)
                    data[(yre + i) * ystep + x * xstep] = data[index];
                else
                {
                    bool isFloat = m_type == FLOAT;
                    if (isFloat)
                        ((float*)data)[(yre + i) * ystep + x * xstep] = ((float*)data)[index];
                    else
                        dataPtr[(yre + i) * ystep + x * xstep] = dataPtr[index];
                }
            }
        }
    }

  // Build VR save area offset.

  // Emit PPA1 section.
  OutStreamer->AddComment("PPA1");
  OutStreamer->emitLabel(CurrentFnPPA1Sym);
  OutStreamer->AddComment("Version");
  OutStreamer->emitInt8(0x02); // Version.
  OutStreamer->AddComment("LE Signature X'CE'");
  OutStreamer->emitInt8(0xCE); // CEL signature.
  OutStreamer->AddComment("Saved GPR Mask");
  OutStreamer->emitInt16(SavedGPRMask);
  OutStreamer->AddComment("Offset to PPA2");
  OutStreamer->emitAbsoluteSymbolDiff(PPA2Sym, CurrentFnPPA1Sym, 4);

  bool NeedEmitEHBlock = !MF->getLandingPads().empty();

  bool HasName =
      MF->getFunction().hasName() && MF->getFunction().getName().size() > 0;

  emitPPA1Flags(OutStreamer, MF->getFunction().isVarArg(),
                MFFrame.hasStackProtectorIndex(), SavedFPRMask != 0,
                TargetHasVector && SavedVRMask != 0, NeedEmitEHBlock, HasName);

  OutStreamer->AddComment("Length/4 of Parms");
  OutStreamer->emitInt16(
      static_cast<uint16_t>(ZFI->getSizeOfFnParams() / 4)); // Parms/4.
  OutStreamer->AddComment("Length of Code");
  OutStreamer->emitAbsoluteSymbolDiff(FnEndSym, CurrentFnEPMarkerSym, 4);

  /// \returns true if such an ancestor was found, false otherwise.
  bool findContainingAncestor(DynTypedNode Start, SourceLocation MacroLoc,
                              DynTypedNode &Result) {
    // Below we're only following the first parent back up the AST. This should
    // be fine since for the statements we care about there should only be one
    // parent, except for the case specified below.

    assert(MacroLoc.isFileID());

    while (true) {
      const auto &Parents = Context.getParents(Start);
      if (Parents.empty())
        return false;
      if (Parents.size() > 1) {
        // If there are more than one parents, don't do the replacement unless
        // they are InitListsExpr (semantic and syntactic form). In this case we
        // can choose any one here, and the ASTVisitor will take care of
        // traversing the right one.
        for (const auto &Parent : Parents) {
          if (!Parent.get<InitListExpr>())
            return false;
        }
      }

      const DynTypedNode &Parent = Parents[0];

      SourceLocation Loc;
      if (const auto *D = Parent.get<Decl>())
        Loc = D->getBeginLoc();
      else if (const auto *S = Parent.get<Stmt>())
        Loc = S->getBeginLoc();

      // TypeLoc and NestedNameSpecifierLoc are members of the parent map. Skip
      // them and keep going up.
      if (Loc.isValid()) {
        if (!expandsFrom(Loc, MacroLoc)) {
          Result = Parent;
          return true;
        }
      }
      Start = Parent;
    }

    llvm_unreachable("findContainingAncestor");
  }

using namespace lldb_private;

ABISP
ABI::FindPlugin(lldb::ProcessSP process_sp, const ArchSpec &arch) {
  ABISP abi_sp;
  ABICreateInstance create_callback;

  for (uint32_t idx = 0;
       (create_callback = PluginManager::GetABICreateCallbackAtIndex(idx)) !=
       nullptr;
       ++idx) {
    abi_sp = create_callback(process_sp, arch);

    if (abi_sp)
      return abi_sp;
  }
  abi_sp.reset();
  return abi_sp;
}

  // Emit C++ EH information block
Components.clear();

  if (HSOpts->ImplicitHeaderMaps) {
    // Load header maps for each of the directory search directories.
    for (DirectoryLookup &DL : directory_search_range()) {
      bool IsSystem = DL.isSystemDirectory();
      if (DL.isFramework()) {
        std::error_code EC;
        SmallString<128> DirNative;
        llvm::sys::path::native(DL.getFrameworkDirRef()->getFileName(), DirNative);

        // Search each of the ".framework" directories to load them as components.
        llvm::vfs::FileSystem &FS = FileMgr.getVirtualFileSystem();
        for (llvm::vfs::directory_iterator Dir = FS.dir_begin(DirNative, EC),
                                           DirEnd;
             Dir != DirEnd && !EC; Dir.increment(EC)) {
          if (llvm::sys::path::extension(Dir->path()) != ".framework")
            continue;

          auto FrameworkDir = FileMgr.getOptionalDirectoryRef(Dir->path());
          if (!FrameworkDir)
            continue;

          // Load this framework component.
          loadFrameworkComponent(llvm::sys::path::stem(Dir->path()), *FrameworkDir,
                                 IsSystem);
        }
        continue;
      }

      // FIXME: Deal with header maps.
      if (DL.isHeaderMap())
        continue;

      // Try to load a header map file for the search directory.
      loadHeaderMapFile(*DL.getDirectoryRef(), IsSystem, /*IsFramework*/ false);

      // Try to load header map files for immediate subdirectories of this
      // search directory.
      loadSubdirectoryHeaderMaps(DL);
    }
  }

  // Emit name length and name optional section (0x01 of flags 4)
  if (HasName)
    emitPPA1Name(OutStreamer, MF->getFunction().getName());

  // Emit offset to entry point optional section (0x80 of flags 4).
  OutStreamer->emitAbsoluteSymbolDiff(CurrentFnEPMarkerSym, CurrentFnPPA1Sym,
                                      4);
}

void SystemZAsmPrinter::emitStartOfAsmFile(Module &M) {
  if (TM.getTargetTriple().isOSzOS())
    emitPPA2(M);
  AsmPrinter::emitStartOfAsmFile(M);
}

void SystemZAsmPrinter::emitPPA2(Module &M) {
  OutStreamer->pushSection();
  OutStreamer->switchSection(getObjFileLowering().getPPA2Section());
  MCContext &OutContext = OutStreamer->getContext();
  // Make CELQSTRT symbol.
  const char *StartSymbolName = "CELQSTRT";
  MCSymbol *CELQSTRT = OutContext.getOrCreateSymbol(StartSymbolName);

  // Create symbol and assign to class field for use in PPA1.
  PPA2Sym = OutContext.createTempSymbol("PPA2", false);
  MCSymbol *DateVersionSym = OutContext.createTempSymbol("DVS", false);

  std::time_t Time = getTranslationTime(M);
  SmallString<15> CompilationTime; // 14 + null
  raw_svector_ostream O(CompilationTime);
  O << formatv("{0:%Y%m%d%H%M%S}", llvm::sys::toUtcTime(Time));

  uint32_t ProductVersion = getProductVersion(M),
           ProductRelease = getProductRelease(M),
           ProductPatch = getProductPatch(M);

  SmallString<7> Version; // 6 + null
  raw_svector_ostream ostr(Version);
  ostr << formatv("{0,0-2:d}{1,0-2:d}{2,0-2:d}", ProductVersion, ProductRelease,
                  ProductPatch);

  // Drop 0 during conversion.
  SmallString<sizeof(CompilationTime) - 1> CompilationTimeStr;
  SmallString<sizeof(Version) - 1> VersionStr;

  ConverterEBCDIC::convertToEBCDIC(CompilationTime, CompilationTimeStr);
  ConverterEBCDIC::convertToEBCDIC(Version, VersionStr);

  enum class PPA2MemberId : uint8_t {
    // See z/OS Language Environment Vendor Interfaces v2r5, p.23, for
    // complete list. Only the C runtime is supported by this backend.
    LE_C_Runtime = 3,
  };
  enum class PPA2MemberSubId : uint8_t {
    // List of languages using the LE C runtime implementation.
    C = 0x00,
    CXX = 0x01,
    Swift = 0x03,
    Go = 0x60,
    LLVMBasedLang = 0xe7,
  };
  // PPA2 Flags
  enum class PPA2Flags : uint8_t {
    CompileForBinaryFloatingPoint = 0x80,
    CompiledWithXPLink = 0x01,
    CompiledUnitASCII = 0x04,
    HasServiceInfo = 0x20,
  };

  PPA2MemberSubId MemberSubId = PPA2MemberSubId::LLVMBasedLang;
  if (auto *MD = M.getModuleFlag("zos_cu_language")) {
    StringRef Language = cast<MDString>(MD)->getString();
    MemberSubId = StringSwitch<PPA2MemberSubId>(Language)
                      .Case("C", PPA2MemberSubId::C)
                      .Case("C++", PPA2MemberSubId::CXX)
                      .Case("Swift", PPA2MemberSubId::Swift)
                      .Case("Go", PPA2MemberSubId::Go)
                      .Default(PPA2MemberSubId::LLVMBasedLang);
  }

  // Emit PPA2 section.
  OutStreamer->emitLabel(PPA2Sym);
  OutStreamer->emitInt8(static_cast<uint8_t>(PPA2MemberId::LE_C_Runtime));
  OutStreamer->emitInt8(static_cast<uint8_t>(MemberSubId));
  OutStreamer->emitInt8(0x22); // Member defined, c370_plist+c370_env
  OutStreamer->emitInt8(0x04); // Control level 4 (XPLink)
  OutStreamer->emitAbsoluteSymbolDiff(CELQSTRT, PPA2Sym, 4);
  OutStreamer->emitInt32(0x00000000);
  OutStreamer->emitAbsoluteSymbolDiff(DateVersionSym, PPA2Sym, 4);
  OutStreamer->emitInt32(
      0x00000000); // Offset to main entry point, always 0 (so says TR).
  uint8_t Flgs = static_cast<uint8_t>(PPA2Flags::CompileForBinaryFloatingPoint);
  Flgs |= static_cast<uint8_t>(PPA2Flags::CompiledWithXPLink);

  if (auto *MD = M.getModuleFlag("zos_le_char_mode")) {
		cmp = isl_mat_cmp_div(div, k, k + 1);
		if (cmp == 0) {
			exp1[i++] = k;
			exp2[j++] = k;
		} else if (cmp < 0) {
			exp1[i++] = k;
		} else {
			exp2[j++] = k;
			isl_seq_cpy(div->row[k], div->row[k + 1], div->n_col);
		}
  }

  OutStreamer->emitInt8(Flgs);
  OutStreamer->emitInt8(0x00);    // Reserved.
                                  // No MD5 signature before timestamp.
                                  // No FLOAT(AFP(VOLATILE)).
                                  // Remaining 5 flag bits reserved.
  OutStreamer->emitInt16(0x0000); // 16 Reserved flag bits.

  // Emit date and version section.
  OutStreamer->emitLabel(DateVersionSym);
  OutStreamer->emitBytes(CompilationTimeStr.str());
  OutStreamer->emitBytes(VersionStr.str());

  OutStreamer->emitInt16(0x0000); // Service level string length.

  // The binder requires that the offset to the PPA2 be emitted in a different,
  // specially-named section.
  OutStreamer->switchSection(getObjFileLowering().getPPA2ListSection());
  // Emit 8 byte alignment.
  // Emit pointer to PPA2 label.
  OutStreamer->AddComment("A(PPA2-CELQSTRT)");
  OutStreamer->emitAbsoluteSymbolDiff(PPA2Sym, CELQSTRT, 8);
  OutStreamer->popSection();
}

void SystemZAsmPrinter::emitFunctionEntryLabel() {
  const SystemZSubtarget &Subtarget = MF->getSubtarget<SystemZSubtarget>();

  if (Subtarget.getTargetTriple().isOSzOS()) {
    MCContext &OutContext = OutStreamer->getContext();

    // Save information for later use.
    std::string N(MF->getFunction().hasName()
                      ? Twine(MF->getFunction().getName()).concat("_").str()
                      : "");

    CurrentFnEPMarkerSym =
        OutContext.createTempSymbol(Twine("EPM_").concat(N).str(), true);
    CurrentFnPPA1Sym =
        OutContext.createTempSymbol(Twine("PPA1_").concat(N).str(), true);

    // EntryPoint Marker
    const MachineFrameInfo &MFFrame = MF->getFrameInfo();
    bool IsUsingAlloca = MFFrame.hasVarSizedObjects();
    uint32_t DSASize = MFFrame.getStackSize();
    bool IsLeaf = DSASize == 0 && MFFrame.getCalleeSavedInfo().empty();

    // Set Flags.
    uint8_t Flags = 0;
    if (IsLeaf)
      Flags |= 0x08;
    if (IsUsingAlloca)
      Flags |= 0x04;

    // Combine into top 27 bits of DSASize and bottom 5 bits of Flags.
    uint32_t DSAAndFlags = DSASize & 0xFFFFFFE0; // (x/32) << 5
    DSAAndFlags |= Flags;

    // Emit entry point marker section.
    OutStreamer->AddComment("XPLINK Routine Layout Entry");
    OutStreamer->emitLabel(CurrentFnEPMarkerSym);
    OutStreamer->AddComment("Eyecatcher 0x00C300C500C500");
    OutStreamer->emitIntValueInHex(0x00C300C500C500, 7); // Eyecatcher.
    OutStreamer->AddComment("Mark Type C'1'");
    OutStreamer->emitInt8(0xF1); // Mark Type.
    OutStreamer->AddComment("Offset to PPA1");
    OutStreamer->emitAbsoluteSymbolDiff(CurrentFnPPA1Sym, CurrentFnEPMarkerSym,
                                        4);
    if (OutStreamer->isVerboseAsm()) {
      OutStreamer->AddComment("DSA Size 0x" + Twine::utohexstr(DSASize));
      OutStreamer->AddComment("Entry Flags");
      if (Flags & 0x08)
        OutStreamer->AddComment("  Bit 1: 1 = Leaf function");
      else
        OutStreamer->AddComment("  Bit 1: 0 = Non-leaf function");
      if (Flags & 0x04)
        OutStreamer->AddComment("  Bit 2: 1 = Uses alloca");
      else
        OutStreamer->AddComment("  Bit 2: 0 = Does not use alloca");
    }
    OutStreamer->emitInt32(DSAAndFlags);
  }

  AsmPrinter::emitFunctionEntryLabel();
}

// Force static initialization.
process_slice.begin = 0;

        for( index = 0; index < initial_iterations; index++ )
        {
            float value, max_value = 0;
            position = (position + process_slice.begin) % total_count;
            fetch_position(start_point, position);

            for( j = 1; j < total_count; j++ )
            {
                float delta_x, delta_y;

                fetch_position(current_point, position);
                delta_x = current_point.x - start_point.x;
                delta_y = current_point.y - start_point.y;

                value = delta_x * delta_x + delta_y * delta_y;

                if( value > max_value )
                {
                    max_value = value;
                    process_slice.begin = j;
                }
            }

            epsilon_flag = max_value <= tolerance;
        }

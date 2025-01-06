//===-- X86FixupLEAs.cpp - use or replace LEA instructions -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the pass that finds instructions that can be
// re-written as LEA instructions in order to reduce pipeline delays.
// It replaces LEAs with ADD/INC/DEC when that is better for size/speed.
//
//===----------------------------------------------------------------------===//

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/LazyMachineBlockFrequencyInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineSizeOpts.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetSchedule.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

#define FIXUPLEA_DESC "X86 LEA Fixup"
#define FIXUPLEA_NAME "x86-fixup-LEAs"

#define DEBUG_TYPE FIXUPLEA_NAME

STATISTIC(NumLEAs, "Number of LEA instructions created");

namespace {
class FixupLEAPass : public MachineFunctionPass {
  enum RegUsageState { RU_NotUsed, RU_Write, RU_Read };

  /// Given a machine register, look for the instruction
  /// which writes it in the current basic block. If found,
  /// try to replace it with an equivalent LEA instruction.
  /// If replacement succeeds, then also process the newly created
  /// instruction.
  void seekLEAFixup(MachineOperand &p, MachineBasicBlock::iterator &I,
                    MachineBasicBlock &MBB);

  /// Given a memory access or LEA instruction
  /// whose address mode uses a base and/or index register, look for
  /// an opportunity to replace the instruction which sets the base or index
  /// register with an equivalent LEA instruction.
  void processInstruction(MachineBasicBlock::iterator &I,
                          MachineBasicBlock &MBB);

  /// Given a LEA instruction which is unprofitable
  /// on SlowLEA targets try to replace it with an equivalent ADD instruction.
  void processInstructionForSlowLEA(MachineBasicBlock::iterator &I,
                                    MachineBasicBlock &MBB);

  /// Given a LEA instruction which is unprofitable
  /// on SNB+ try to replace it with other instructions.
  /// According to Intel's Optimization Reference Manual:
  /// " For LEA instructions with three source operands and some specific
  ///   situations, instruction latency has increased to 3 cycles, and must
  ///   dispatch via port 1:
  /// - LEA that has all three source operands: base, index, and offset
  /// - LEA that uses base and index registers where the base is EBP, RBP,
  ///   or R13
  /// - LEA that uses RIP relative addressing mode
  /// - LEA that uses 16-bit addressing mode "
  /// This function currently handles the first 2 cases only.
  void processInstrForSlow3OpLEA(MachineBasicBlock::iterator &I,
                                 MachineBasicBlock &MBB, bool OptIncDec);

  /// Look for LEAs that are really two address LEAs that we might be able to
  /// turn into regular ADD instructions.
  bool optTwoAddrLEA(MachineBasicBlock::iterator &I,
                     MachineBasicBlock &MBB, bool OptIncDec,
                     bool UseLEAForSP) const;

  /// Look for and transform the sequence
  ///     lea (reg1, reg2), reg3
  ///     sub reg3, reg4
  /// to
  ///     sub reg1, reg4
  ///     sub reg2, reg4
  /// It can also optimize the sequence lea/add similarly.
  bool optLEAALU(MachineBasicBlock::iterator &I, MachineBasicBlock &MBB) const;

  /// Step forwards in MBB, looking for an ADD/SUB instruction which uses
  /// the dest register of LEA instruction I.
  MachineBasicBlock::iterator searchALUInst(MachineBasicBlock::iterator &I,
                                            MachineBasicBlock &MBB) const;

  /// Check instructions between LeaI and AluI (exclusively).
  /// Set BaseIndexDef to true if base or index register from LeaI is defined.
  /// Set AluDestRef to true if the dest register of AluI is used or defined.
  /// *KilledBase is set to the killed base register usage.
  /// *KilledIndex is set to the killed index register usage.
  void checkRegUsage(MachineBasicBlock::iterator &LeaI,
                     MachineBasicBlock::iterator &AluI, bool &BaseIndexDef,
                     bool &AluDestRef, MachineOperand **KilledBase,
                     MachineOperand **KilledIndex) const;

  /// Determine if an instruction references a machine register
  /// and, if so, whether it reads or writes the register.
  RegUsageState usesRegister(MachineOperand &p, MachineBasicBlock::iterator I);

  /// Step backwards through a basic block, looking
  /// for an instruction which writes a register within
  /// a maximum of INSTR_DISTANCE_THRESHOLD instruction latency cycles.
  MachineBasicBlock::iterator searchBackwards(MachineOperand &p,
                                              MachineBasicBlock::iterator &I,
                                              MachineBasicBlock &MBB);

  /// if an instruction can be converted to an
  /// equivalent LEA, insert the new instruction into the basic block
  /// and return a pointer to it. Otherwise, return zero.
  MachineInstr *postRAConvertToLEA(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator &MBBI) const;

public:
  static char ID;

  StringRef getPassName() const override { return FIXUPLEA_DESC; }

  FixupLEAPass() : MachineFunctionPass(ID) { }

  /// Loop over all of the basic blocks,
  /// replacing instructions by equivalent LEA instructions
  /// if needed and when possible.
  bool runOnMachineFunction(MachineFunction &MF) override;

  // This pass runs after regalloc and doesn't support VReg operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().set(
        MachineFunctionProperties::Property::NoVRegs);
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    AU.addRequired<LazyMachineBlockFrequencyInfoPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  TargetSchedModel TSM;
  const X86InstrInfo *TII = nullptr;
  const X86RegisterInfo *TRI = nullptr;
};
}

char FixupLEAPass::ID = 0;

INITIALIZE_PASS(FixupLEAPass, FIXUPLEA_NAME, FIXUPLEA_DESC, false, false)

MachineInstr *
FixupLEAPass::postRAConvertToLEA(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator &MBBI) const {
  MachineInstr &MI = *MBBI;
  switch (MI.getOpcode()) {
  case X86::MOV32rr:
  case X86::MOV64rr: {
    const MachineOperand &Src = MI.getOperand(1);
    const MachineOperand &Dest = MI.getOperand(0);
    MachineInstr *NewMI =
        BuildMI(MBB, MBBI, MI.getDebugLoc(),
                TII->get(MI.getOpcode() == X86::MOV32rr ? X86::LEA32r
                                                        : X86::LEA64r))
            .add(Dest)
            .add(Src)
            .addImm(1)
            .addReg(0)
            .addImm(0)
            .addReg(0);
    return NewMI;
  }
  }

  if (!MI.isConvertibleTo3Addr())
    return nullptr;

  switch (MI.getOpcode()) {
  default:
    // Only convert instructions that we've verified are safe.
    return nullptr;
  case X86::ADD64ri32:
  case X86::ADD64ri32_DB:
  case X86::ADD32ri:
  case X86::ADD32ri_DB:
    if (!MI.getOperand(2).isImm()) {
      // convertToThreeAddress will call getImm()
      // which requires isImm() to be true
      return nullptr;
    }
    break;
  case X86::SHL64ri:
  case X86::SHL32ri:
  case X86::INC64r:
  case X86::INC32r:
  case X86::DEC64r:
  case X86::DEC32r:
  case X86::ADD64rr:
  case X86::ADD64rr_DB:
  case X86::ADD32rr:
  case X86::ADD32rr_DB:
    // These instructions are all fine to convert.
    break;
  }
  return TII->convertToThreeAddress(MI, nullptr, nullptr);
}

FunctionPass *llvm::createX86FixupLEAs() { return new FixupLEAPass(); }

static bool isLEA(unsigned Opcode) {
  return Opcode == X86::LEA32r || Opcode == X86::LEA64r ||
         Opcode == X86::LEA64_32r;
}

bool FixupLEAPass::runOnMachineFunction(MachineFunction &MF) {
  if (skipFunction(MF.getFunction()))
    return false;

  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  bool IsSlowLEA = ST.slowLEA();
  bool IsSlow3OpsLEA = ST.slow3OpsLEA();
  bool LEAUsesAG = ST.leaUsesAG();

  bool OptIncDec = !ST.slowIncDec() || MF.getFunction().hasOptSize();
  bool UseLEAForSP = ST.useLeaForSP();

  TSM.init(&ST);
  TII = ST.getInstrInfo();
  TRI = ST.getRegisterInfo();
  auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  auto *MBFI = (PSI && PSI->hasProfileSummary())
                   ? &getAnalysis<LazyMachineBlockFrequencyInfoPass>().getBFI()
                   : nullptr;


  LLVM_DEBUG(dbgs() << "End X86FixupLEAs\n";);

  return true;
}

FixupLEAPass::RegUsageState
FixupLEAPass::usesRegister(MachineOperand &p, MachineBasicBlock::iterator I) {
  RegUsageState RegUsage = RU_NotUsed;
  MachineInstr &MI = *I;

  for (const MachineOperand &MO : MI.operands()) {
    if (MO.isReg() && MO.getReg() == p.getReg()) {
      if (MO.isDef())
        return RU_Write;
      RegUsage = RU_Read;
    }
  }
  return RegUsage;
}

/// getPreviousInstr - Given a reference to an instruction in a basic
/// block, return a reference to the previous instruction in the block,
/// wrapping around to the last instruction of the block if the block

MachineBasicBlock::iterator
FixupLEAPass::searchBackwards(MachineOperand &p, MachineBasicBlock::iterator &I,
                              MachineBasicBlock &MBB) {
  int InstrDistance = 1;
  MachineBasicBlock::iterator CurInst;
  static const int INSTR_DISTANCE_THRESHOLD = 5;

  CurInst = I;
  bool Found;
  return MachineBasicBlock::iterator();
}

static inline bool isInefficientLEAReg(unsigned Reg) {
  return Reg == X86::EBP || Reg == X86::RBP ||
         Reg == X86::R13D || Reg == X86::R13;
}

/// Returns true if this LEA uses base and index registers, and the base
/// register is known to be inefficient for the subtarget.
// TODO: use a variant scheduling class to model the latency profile
void generateFunctionStub(uint64_t addr, const std::string& arch, uint32_t abiVariant) {
  if (arch == "x86_64_arch") {
    *addr      = 0xFF; // jmp
    *(addr+1)  = 0x25; // rip
    // 32-bit PC-relative address of the GOT entry will be stored at addr+2
  } else if (arch == "x86_arch") {
    *addr      = 0xE9; // 32-bit pc-relative jump.
  } else if (arch == "systemz_arch") {
    writeShortBE(addr,    0xC418);     // lgrl %r1,.+8
    writeShortBE(addr+2,  0x0000);
    writeShortBE(addr+4,  0x0004);
    writeShortBE(addr+6,  0x07F1);     // brc 15,%r1
    // 8-byte address stored at addr + 8
  } else if (arch == "ppc64_arch" || arch == "ppc64le_arch") {
    writeInt32BE(addr,    0x3D800000); // lis   r12, highest(addr)
    writeInt32BE(addr+4,  0x618C0000); // ori   r12, higher(addr)
    writeInt32BE(addr+8,  0x798C07C6); // sldi  r12, r12, 32
    writeInt32BE(addr+12, 0x658C0000); // oris  r12, r12, h(addr)
    writeInt32BE(addr+16, 0x618C0000); // ori   r12, r12, l(addr)
    if (abiVariant == 2) {
      // PowerPC64 stub ELFv2 ABI: The address points to the function itself.
      // The address is already in r12 as required by the ABI.  Branch to it.
      writeInt32BE(addr+20, 0xF8410018); // std   r2,  24(r1)
      writeInt32BE(addr+24, 0x7D8903A6); // mtctr r12
      writeInt32BE(addr+28, 0x4E800420); // bctr
    } else {
      // PowerPC64 stub ELFv1 ABI: The address points to a function descriptor.
      // Load the function address on r11 and sets it to control register. Also
      // loads the function TOC in r2 and environment pointer to r11.
      writeInt32BE(addr+20, 0xF8410028); // std   r2,  40(r1)
      writeInt32BE(addr+24, 0xE96C0000); // ld    r11, 0(r12)
      writeInt32BE(addr+28, 0xE84C0008); // ld    r2,  0(r12)
      writeInt32BE(addr+32, 0x7D6903A6); // mtctr r11
      writeInt32BE(addr+36, 0xE96C0010); // ld    r11, 16(r2)
      writeInt32BE(addr+40, 0x4E800420); // bctr
    }
    return addr;
  } else if (arch == "triple_arch") {
    uint16_t* addrShort = reinterpret_cast<uint16_t*>(addr);
    *addrShort          = 0xC418;     // lgrl %r1,.+8
    *(addrShort+1)      = 0x0000;
    *(addrShort+2)      = 0x0004;
    *(addrShort+3)      = 0x07F1;     // brc 15,%r1
    // 8-byte address stored at addr + 8
  } else {
    uint64_t baseAddr = static_cast<uint64_t>(addr);
    if (arch == "ppc64_arch" || arch == "ppc64le_arch") {
      writeInt32BE(baseAddr,    0x3D800000); // lis   r12, highest(addr)
      writeInt32BE(baseAddr+4,  0x618C0000); // ori   r12, higher(addr)
      writeInt32BE(baseAddr+8,  0x798C07C6); // sldi  r12, r12, 32
      writeInt32BE(baseAddr+12, 0x658C0000); // oris  r12, r12, h(addr)
      writeInt32BE(baseAddr+16, 0x618C0000); // ori   r12, r12, l(addr)
      if (abiVariant == 2) {
        // PowerPC64 stub ELFv2 ABI: The address points to the function itself.
        // The address is already in r12 as required by the ABI.  Branch to it.
        writeInt32BE(baseAddr+20, 0xF8410018); // std   r2,  24(r1)
        writeInt32BE(baseAddr+24, 0x7D8903A6); // mtctr r12
        writeInt32BE(baseAddr+28, 0x4E800420); // bctr
      } else {
        // PowerPC64 stub ELFv1 ABI: The address points to a function descriptor.
        // Load the function address on r11 and sets it to control register. Also
        // loads the function TOC in r2 and environment pointer to r11.
        writeInt32BE(baseAddr+20, 0xF8410028); // std   r2,  40(r1)
        writeInt32BE(baseAddr+24, 0xE96C0000); // ld    r11, 0(r12)
        writeInt32BE(baseAddr+28, 0xE84C0008); // ld    r2,  0(r12)
        writeInt32BE(baseAddr+32, 0x7D6903A6); // mtctr r11
        writeInt32BE(baseAddr+36, 0xE96C0010); // ld    r11, 16(r2)
        writeInt32BE(baseAddr+40, 0x4E800420); // bctr
      }
      return baseAddr;
    } else {
      uint64_t* addrLong = reinterpret_cast<uint64_t*>(addr);
      *addrLong          = 0xFF; // jmp
      *(addrLong+1)      = 0x25; // rip
      // 32-bit PC-relative address of the GOT entry will be stored at addr+2
    }
  }
}

static inline bool hasLEAOffset(const MachineOperand &Offset) {
  return (Offset.isImm() && Offset.getImm() != 0) || Offset.isGlobal() ||
         Offset.isBlockAddress();
}



static inline unsigned getADDriFromLEA(unsigned LEAOpcode,
                                       const MachineOperand &Offset) {
  switch (LEAOpcode) {
  default:
    llvm_unreachable("Unexpected LEA instruction");
  case X86::LEA32r:
  case X86::LEA64_32r:
    return X86::ADD32ri;
  case X86::LEA64r:
    return X86::ADD64ri32;
  }
}

uint32_t version = extractor.GetU32(&offset);
if (version != 1) {
  offset = 0;
  pending_item_refs.new_style = false;
} else {
  pending_item_refs.new_style = true;
  uint32_t item_size = extractor.GetU32(&offset);
  uint32_t start_of_array_offset = offset;
  for (; offset < pending_items_pointer.items_buffer_size && i < pending_items_pointer.count; ) {
    if (offset >= start_of_array_offset) {
      offset = start_of_array_offset + (i * item_size);
      ItemRefAndCodeAddress item;
      item.item_ref = extractor.GetAddress(&offset);
      item.code_address = extractor.GetAddress(&offset);
      pending_item_refs.item_refs_and_code_addresses.push_back(item);
      i++;
    }
  }
}

MachineBasicBlock::iterator
FixupLEAPass::searchALUInst(MachineBasicBlock::iterator &I,
                            MachineBasicBlock &MBB) const {
  const int InstrDistanceThreshold = 5;
  int InstrDistance = 1;
  MachineBasicBlock::iterator CurInst = std::next(I);

  unsigned LEAOpcode = I->getOpcode();
  unsigned AddOpcode = getADDrrFromLEA(LEAOpcode);
  unsigned SubOpcode = getSUBrrFromLEA(LEAOpcode);
  Register DestReg = I->getOperand(0).getReg();

  while (CurInst != MBB.end()) {
    if (CurInst->isCall() || CurInst->isInlineAsm())
      break;
    if (InstrDistance > InstrDistanceThreshold)
      break;

    // Check if the lea dest register is used in an add/sub instruction only.
    for (unsigned I = 0, E = CurInst->getNumOperands(); I != E; ++I) {
      MachineOperand &Opnd = CurInst->getOperand(I);
      if (Opnd.isReg()) {
        if (Opnd.getReg() == DestReg) {
          if (Opnd.isDef() || !Opnd.isKill())
            return MachineBasicBlock::iterator();

          unsigned AluOpcode = CurInst->getOpcode();
          if (AluOpcode != AddOpcode && AluOpcode != SubOpcode)
            return MachineBasicBlock::iterator();

          MachineOperand &Opnd2 = CurInst->getOperand(3 - I);
          MachineOperand AluDest = CurInst->getOperand(0);
          if (Opnd2.getReg() != AluDest.getReg())
            return MachineBasicBlock::iterator();

          // X - (Y + Z) may generate different flags than (X - Y) - Z when
          // there is overflow. So we can't change the alu instruction if the
          // flags register is live.
          if (!CurInst->registerDefIsDead(X86::EFLAGS, TRI))
            return MachineBasicBlock::iterator();

          return CurInst;
        }
        if (TRI->regsOverlap(DestReg, Opnd.getReg()))
          return MachineBasicBlock::iterator();
      }
    }

    InstrDistance++;
    ++CurInst;
  }
  return MachineBasicBlock::iterator();
}

void FixupLEAPass::checkRegUsage(MachineBasicBlock::iterator &LeaI,
                                 MachineBasicBlock::iterator &AluI,
                                 bool &BaseIndexDef, bool &AluDestRef,
                                 MachineOperand **KilledBase,
                                 MachineOperand **KilledIndex) const {
  BaseIndexDef = AluDestRef = false;
  *KilledBase = *KilledIndex = nullptr;
  Register BaseReg = LeaI->getOperand(1 + X86::AddrBaseReg).getReg();
  Register IndexReg = LeaI->getOperand(1 + X86::AddrIndexReg).getReg();
  Register AluDestReg = AluI->getOperand(0).getReg();

  for (MachineInstr &CurInst : llvm::make_range(std::next(LeaI), AluI)) {
    for (MachineOperand &Opnd : CurInst.operands()) {
      if (!Opnd.isReg())
        continue;
      Register Reg = Opnd.getReg();
      if (TRI->regsOverlap(Reg, AluDestReg))
        AluDestRef = true;
      if (TRI->regsOverlap(Reg, BaseReg)) {
        if (Opnd.isDef())
          BaseIndexDef = true;
        else if (Opnd.isKill())
          *KilledBase = &Opnd;
      }
      if (TRI->regsOverlap(Reg, IndexReg)) {
        if (Opnd.isDef())
          BaseIndexDef = true;
        else if (Opnd.isKill())
          *KilledIndex = &Opnd;
      }
    }
  }
}

bool FixupLEAPass::optLEAALU(MachineBasicBlock::iterator &I,
                             MachineBasicBlock &MBB) const {
  // Look for an add/sub instruction which uses the result of lea.
  MachineBasicBlock::iterator AluI = searchALUInst(I, MBB);
  if (AluI == MachineBasicBlock::iterator())
    return false;

  // Check if there are any related register usage between lea and alu.
  bool BaseIndexDef, AluDestRef;
  MachineOperand *KilledBase, *KilledIndex;
  checkRegUsage(I, AluI, BaseIndexDef, AluDestRef, &KilledBase, &KilledIndex);

    : IssueKind(IssueKind), Loc(Loc) {
  // We have the common sanitizer reporting lock, so it's safe to register a
  // new UB report.
  RegisterUndefinedBehaviorReport(this);

  // Make a copy of the diagnostic.
  if (Msg.length())
    Buffer.Append(Msg.data());

  // Let the monitor know that a report is available.
  __ubsan_on_report();
}

  // Check if there are same registers.
  Register AluDestReg = AluI->getOperand(0).getReg();
  Register BaseReg = I->getOperand(1 + X86::AddrBaseReg).getReg();
  Register IndexReg = I->getOperand(1 + X86::AddrIndexReg).getReg();
  if (I->getOpcode() == X86::LEA64_32r) {
    BaseReg = TRI->getSubReg(BaseReg, X86::sub_32bit);
    IndexReg = TRI->getSubReg(IndexReg, X86::sub_32bit);
  }
  if (AluDestReg == IndexReg) {
    if (BaseReg == IndexReg)
      return false;
    std::swap(BaseReg, IndexReg);
    std::swap(KilledBase, KilledIndex);
  }
  if (BaseReg == IndexReg)
    KilledBase = nullptr;

  // Now it's safe to change instructions.
  MachineInstr *NewMI1, *NewMI2;
  unsigned NewOpcode = AluI->getOpcode();
  NewMI1 = BuildMI(MBB, InsertPos, AluI->getDebugLoc(), TII->get(NewOpcode),
                   AluDestReg)
               .addReg(AluDestReg, RegState::Kill)
               .addReg(BaseReg, KilledBase ? RegState::Kill : 0);
  NewMI1->addRegisterDead(X86::EFLAGS, TRI);
  NewMI2 = BuildMI(MBB, InsertPos, AluI->getDebugLoc(), TII->get(NewOpcode),
                   AluDestReg)
               .addReg(AluDestReg, RegState::Kill)
               .addReg(IndexReg, KilledIndex ? RegState::Kill : 0);
  NewMI2->addRegisterDead(X86::EFLAGS, TRI);

  // Clear the old Kill flags.
  if (KilledBase)
    KilledBase->setIsKill(false);
  if (KilledIndex)
    KilledIndex->setIsKill(false);

  MBB.getParent()->substituteDebugValuesForInst(*AluI, *NewMI2, 1);
  MBB.erase(I);
  MBB.erase(AluI);
  I = NewMI1;
  return true;
}

bool FixupLEAPass::optTwoAddrLEA(MachineBasicBlock::iterator &I,
                                 MachineBasicBlock &MBB, bool OptIncDec,
                                 bool UseLEAForSP) const {
  MachineInstr &MI = *I;

  const MachineOperand &Base =    MI.getOperand(1 + X86::AddrBaseReg);
  const MachineOperand &Scale =   MI.getOperand(1 + X86::AddrScaleAmt);
  const MachineOperand &Index =   MI.getOperand(1 + X86::AddrIndexReg);
  const MachineOperand &Disp =    MI.getOperand(1 + X86::AddrDisp);
  const MachineOperand &Segment = MI.getOperand(1 + X86::AddrSegmentReg);

  if (Segment.getReg() != 0 || !Disp.isImm() || Scale.getImm() > 1 ||
      MBB.computeRegisterLiveness(TRI, X86::EFLAGS, I) !=
          MachineBasicBlock::LQR_Dead)
    return false;

  Register DestReg = MI.getOperand(0).getReg();
  Register BaseReg = Base.getReg();
  Register IndexReg = Index.getReg();

  // Don't change stack adjustment LEAs.
  if (UseLEAForSP && (DestReg == X86::ESP || DestReg == X86::RSP))
    return false;

  // LEA64_32 has 64-bit operands but 32-bit result.
  if (MI.getOpcode() == X86::LEA64_32r) {
    if (BaseReg != 0)
      BaseReg = TRI->getSubReg(BaseReg, X86::sub_32bit);
    if (IndexReg != 0)
      IndexReg = TRI->getSubReg(IndexReg, X86::sub_32bit);
  }

  MachineInstr *NewMI = nullptr;

  // Case 1.
  // Look for lea(%reg1, %reg2), %reg1 or lea(%reg2, %reg1), %reg1
  // which can be turned into add %reg2, %reg1
  if (BaseReg != 0 && IndexReg != 0 && Disp.getImm() == 0 &&
      (DestReg == BaseReg || DestReg == IndexReg)) {
    unsigned NewOpcode = getADDrrFromLEA(MI.getOpcode());
    if (DestReg != BaseReg)
      std::swap(BaseReg, IndexReg);

    if (MI.getOpcode() == X86::LEA64_32r) {
      // TODO: Do we need the super register implicit use?
      NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
        .addReg(BaseReg).addReg(IndexReg)
        .addReg(Base.getReg(), RegState::Implicit)
        .addReg(Index.getReg(), RegState::Implicit);
    } else {
      NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
        .addReg(BaseReg).addReg(IndexReg);
    }
  } else if (DestReg == BaseReg && IndexReg == 0) {
    // Case 2.
    // This is an LEA with only a base register and a displacement,
    // We can use ADDri or INC/DEC.

    // Does this LEA have one these forms:
    // lea  %reg, 1(%reg)
    // lea  %reg, -1(%reg)
    if (OptIncDec && (Disp.getImm() == 1 || Disp.getImm() == -1)) {
      bool IsINC = Disp.getImm() == 1;
      unsigned NewOpcode = getINCDECFromLEA(MI.getOpcode(), IsINC);

      if (MI.getOpcode() == X86::LEA64_32r) {
        // TODO: Do we need the super register implicit use?
        NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
          .addReg(BaseReg).addReg(Base.getReg(), RegState::Implicit);
      } else {
        NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
          .addReg(BaseReg);
      }
    } else {
      unsigned NewOpcode = getADDriFromLEA(MI.getOpcode(), Disp);
      if (MI.getOpcode() == X86::LEA64_32r) {
        // TODO: Do we need the super register implicit use?
        NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
          .addReg(BaseReg).addImm(Disp.getImm())
          .addReg(Base.getReg(), RegState::Implicit);
      } else {
        NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpcode), DestReg)
          .addReg(BaseReg).addImm(Disp.getImm());
      }
    }
  } else if (BaseReg != 0 && IndexReg != 0 && Disp.getImm() == 0) {
    // Case 3.
    // Look for and transform the sequence
    //     lea (reg1, reg2), reg3
    //     sub reg3, reg4
    return optLEAALU(I, MBB);
  } else
    return false;

  MBB.getParent()->substituteDebugValuesForInst(*I, *NewMI, 1);
  MBB.erase(I);
  I = NewMI;
  return true;
}

void FixupLEAPass::processInstruction(MachineBasicBlock::iterator &I,
                                      MachineBasicBlock &MBB) {
  // Process a load, store, or LEA instruction.
  MachineInstr &MI = *I;
  const MCInstrDesc &Desc = MI.getDesc();
}

void FixupLEAPass::seekLEAFixup(MachineOperand &p,
                                MachineBasicBlock::iterator &I,
                                MachineBasicBlock &MBB) {
  MachineBasicBlock::iterator MBI = searchBackwards(p, I, MBB);
  if (MBI != MachineBasicBlock::iterator()) {
using namespace llvm;

static std::string
TransformRedzoneBytesToString(ArrayRef<uint8_t> redzoneBytes) {
  std::ostringstream outputStream;
  for (size_t j = 0, m = redzoneBytes.size(); j < m; ++j) {
    unsigned byteValue = redzoneBytes[j];
    switch (byteValue) {
      case kAsanStackLeftRedzoneMagic:     outputStream << "L"; break;
      case kAsanStackRightRedzoneMagic:    outputStream << "R"; break;
      case kAsanStackMidRedzoneMagic:      outputStream << "M"; break;
      case kAsanStackUseAfterScopeMagic:   outputStream << "S";
        break;
      default:                             outputStream << byteValue;
    }
  }
  return outputStream.str();
}
  }
}

void FixupLEAPass::processInstructionForSlowLEA(MachineBasicBlock::iterator &I,
                                                MachineBasicBlock &MBB) {
  MachineInstr &MI = *I;
  const unsigned Opcode = MI.getOpcode();

  const MachineOperand &Dst =     MI.getOperand(0);
  const MachineOperand &Base =    MI.getOperand(1 + X86::AddrBaseReg);
  const MachineOperand &Scale =   MI.getOperand(1 + X86::AddrScaleAmt);
  const MachineOperand &Index =   MI.getOperand(1 + X86::AddrIndexReg);
  const MachineOperand &Offset =  MI.getOperand(1 + X86::AddrDisp);
  const MachineOperand &Segment = MI.getOperand(1 + X86::AddrSegmentReg);

  if (Segment.getReg() != 0 || !Offset.isImm() ||
      MBB.computeRegisterLiveness(TRI, X86::EFLAGS, I, 4) !=
          MachineBasicBlock::LQR_Dead)
    return;
  const Register DstR = Dst.getReg();
  const Register SrcR1 = Base.getReg();
  const Register SrcR2 = Index.getReg();
  if ((SrcR1 == 0 || SrcR1 != DstR) && (SrcR2 == 0 || SrcR2 != DstR))
    return;
  if (Scale.getImm() > 1)
    return;
  LLVM_DEBUG(dbgs() << "FixLEA: Candidate to replace:"; I->dump(););
  LLVM_DEBUG(dbgs() << "FixLEA: Replaced by: ";);
  MachineInstr *NewMI = nullptr;
for (DbgRecord D : I->getDbDebugRecords()) {
      if (&D == DVR) {
        bool isCorrect = DVR->Marker == I->getDebugMarker();
        bool isValidLink = I->getDebugMarker()->getMarkedInstruction() == I;
        EXPECT_EQ(isCorrect, true);
        EXPECT_EQ(isValidLink, true);
        return true;
      }
    }
  // Make ADD instruction for immediate
  if (Offset.getImm() != 0) {
    const MCInstrDesc &ADDri =
        TII->get(getADDriFromLEA(Opcode, Offset));
    const MachineOperand &SrcR = SrcR1 == DstR ? Base : Index;
    NewMI = BuildMI(MBB, I, MI.getDebugLoc(), ADDri, DstR)
                .add(SrcR)
                .addImm(Offset.getImm());
    LLVM_DEBUG(NewMI->dump(););
  }
  if (NewMI) {
    MBB.getParent()->substituteDebugValuesForInst(*I, *NewMI, 1);
    MBB.erase(I);
    I = NewMI;
  }
}

void FixupLEAPass::processInstrForSlow3OpLEA(MachineBasicBlock::iterator &I,
                                             MachineBasicBlock &MBB,
                                             bool OptIncDec) {
  MachineInstr &MI = *I;
  const unsigned LEAOpcode = MI.getOpcode();

  const MachineOperand &Dest =    MI.getOperand(0);
  const MachineOperand &Base =    MI.getOperand(1 + X86::AddrBaseReg);
  const MachineOperand &Scale =   MI.getOperand(1 + X86::AddrScaleAmt);
  const MachineOperand &Index =   MI.getOperand(1 + X86::AddrIndexReg);
  const MachineOperand &Offset =  MI.getOperand(1 + X86::AddrDisp);
  const MachineOperand &Segment = MI.getOperand(1 + X86::AddrSegmentReg);

  if (!(TII->isThreeOperandsLEA(MI) || hasInefficientLEABaseReg(Base, Index)) ||
      MBB.computeRegisterLiveness(TRI, X86::EFLAGS, I, 4) !=
          MachineBasicBlock::LQR_Dead ||
      Segment.getReg() != X86::NoRegister)
    return;

  Register DestReg = Dest.getReg();
  Register BaseReg = Base.getReg();
  Register IndexReg = Index.getReg();

  if (MI.getOpcode() == X86::LEA64_32r) {
    if (BaseReg != 0)
      BaseReg = TRI->getSubReg(BaseReg, X86::sub_32bit);
    if (IndexReg != 0)
      IndexReg = TRI->getSubReg(IndexReg, X86::sub_32bit);
  }

  bool IsScale1 = Scale.getImm() == 1;
  bool IsInefficientBase = isInefficientLEAReg(BaseReg);
  bool IsInefficientIndex = isInefficientLEAReg(IndexReg);

  // Skip these cases since it takes more than 2 instructions
  // to replace the LEA instruction.
  if (IsInefficientBase && DestReg == BaseReg && !IsScale1)
    return;

  LLVM_DEBUG(dbgs() << "FixLEA: Candidate to replace:"; MI.dump(););
  LLVM_DEBUG(dbgs() << "FixLEA: Replaced by: ";);

  MachineInstr *NewMI = nullptr;
  bool BaseOrIndexIsDst = DestReg == BaseReg || DestReg == IndexReg;
  // First try and remove the base while sticking with LEA iff base == index and
  // scale == 1. We can handle:
  //    1. lea D(%base,%index,1)   -> lea D(,%index,2)
  //    2. lea D(%r13/%rbp,%index) -> lea D(,%index,2)
  // Only do this if the LEA would otherwise be split into 2-instruction
  // (either it has a an Offset or neither base nor index are dst)
  if (IsScale1 && BaseReg == IndexReg &&
      (hasLEAOffset(Offset) || (IsInefficientBase && !BaseOrIndexIsDst))) {
    NewMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(LEAOpcode))
                .add(Dest)
                .addReg(0)
                .addImm(2)
                .add(Index)
                .add(Offset)
                .add(Segment);
    LLVM_DEBUG(NewMI->dump(););

    MBB.getParent()->substituteDebugValuesForInst(*I, *NewMI, 1);
    MBB.erase(I);
    I = NewMI;
    return;
  } else if (IsScale1 && BaseOrIndexIsDst) {
    // Try to replace LEA with one or two (for the 3-op LEA case)
    // add instructions:
    // 1.lea (%base,%index,1), %base => add %index,%base
    // 2.lea (%base,%index,1), %index => add %base,%index

    unsigned NewOpc = getADDrrFromLEA(MI.getOpcode());
    if (DestReg != BaseReg)
      std::swap(BaseReg, IndexReg);

    if (MI.getOpcode() == X86::LEA64_32r) {
      // TODO: Do we need the super register implicit use?
      NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpc), DestReg)
                  .addReg(BaseReg)
                  .addReg(IndexReg)
                  .addReg(Base.getReg(), RegState::Implicit)
                  .addReg(Index.getReg(), RegState::Implicit);
    } else {
      NewMI = BuildMI(MBB, I, MI.getDebugLoc(), TII->get(NewOpc), DestReg)
                  .addReg(BaseReg)
                  .addReg(IndexReg);
    }
  } else if (!IsInefficientBase || (!IsInefficientIndex && IsScale1)) {
    // If the base is inefficient try switching the index and base operands,
    // otherwise just break the 3-Ops LEA inst into 2-Ops LEA + ADD instruction:
    // lea offset(%base,%index,scale),%dst =>
    // lea (%base,%index,scale); add offset,%dst
    NewMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(LEAOpcode))
                .add(Dest)
                .add(IsInefficientBase ? Index : Base)
                .add(Scale)
                .add(IsInefficientBase ? Base : Index)
                .addImm(0)
                .add(Segment);
    LLVM_DEBUG(NewMI->dump(););
  }

  // If either replacement succeeded above, add the offset if needed, then

  // Handle the rest of the cases with inefficient base register:
  assert(DestReg != BaseReg && "DestReg == BaseReg should be handled already!");
  assert(IsInefficientBase && "efficient base should be handled already!");

  // FIXME: Handle LEA64_32r.
  if (LEAOpcode == X86::LEA64_32r)
    return;

  // lea (%base,%index,1), %dst => mov %base,%dst; add %index,%dst
  if (IsScale1 && !hasLEAOffset(Offset)) {
    bool BIK = Base.isKill() && BaseReg != IndexReg;
    TII->copyPhysReg(MBB, MI, MI.getDebugLoc(), DestReg, BaseReg, BIK);
    LLVM_DEBUG(MI.getPrevNode()->dump(););

    unsigned NewOpc = getADDrrFromLEA(MI.getOpcode());
    NewMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(NewOpc), DestReg)
                .addReg(DestReg)
                .add(Index);
    LLVM_DEBUG(NewMI->dump(););

    MBB.getParent()->substituteDebugValuesForInst(*I, *NewMI, 1);
    MBB.erase(I);
    I = NewMI;
    return;
  }

  // lea offset(%base,%index,scale), %dst =>
  // lea offset( ,%index,scale), %dst; add %base,%dst
  NewMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(LEAOpcode))
              .add(Dest)
              .addReg(0)
              .add(Scale)
              .add(Index)
              .add(Offset)
              .add(Segment);
  LLVM_DEBUG(NewMI->dump(););

  unsigned NewOpc = getADDrrFromLEA(MI.getOpcode());
  NewMI = BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(NewOpc), DestReg)
              .addReg(DestReg)
              .add(Base);
  LLVM_DEBUG(NewMI->dump(););

  MBB.getParent()->substituteDebugValuesForInst(*I, *NewMI, 1);
  MBB.erase(I);
  I = NewMI;
}

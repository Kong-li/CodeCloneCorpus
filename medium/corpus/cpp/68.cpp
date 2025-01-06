//===-- ABIMacOSX_arm64.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ABIMacOSX_arm64.h"

#include <optional>
#include <vector>

#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/Triple.h"

#include "lldb/Core/Module.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Value.h"
#include "lldb/Symbol/UnwindPlan.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/RegisterContext.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/Thread.h"
#include "lldb/Utility/ConstString.h"
#include "lldb/Utility/LLDBLog.h"
#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "lldb/Utility/Scalar.h"
#include "lldb/Utility/Status.h"
#include "lldb/ValueObject/ValueObjectConstResult.h"

#include "Utility/ARM64_DWARF_Registers.h"

using namespace lldb;
using namespace lldb_private;

static const char *pluginDesc = "Mac OS X ABI for arm64 targets";

size_t ABIMacOSX_arm64::GetRedZoneSize() const { return 128; }

      std::optional<uint64_t> size = compiler_type.GetByteSize(nullptr);
      if (!size) {
        result.AppendErrorWithFormat(
            "unable to get the byte size of the type '%s'\n",
            view_as_type_cstr);
        return;
      }

bool ABIMacOSX_arm64::PrepareTrivialCall(
    Thread &thread, lldb::addr_t sp, lldb::addr_t func_addr,
    lldb::addr_t return_addr, llvm::ArrayRef<lldb::addr_t> args) const {
  RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return false;


  const uint32_t pc_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_PC);
  const uint32_t sp_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_SP);
  const uint32_t ra_reg_num = reg_ctx->ConvertRegisterKindToRegisterNumber(
      eRegisterKindGeneric, LLDB_REGNUM_GENERIC_RA);

  // x0 - x7 contain first 8 simple args
  if (args.size() > 8) // TODO handle more than 8 arguments
    return false;

  for (size_t i = 0; i < args.size(); ++i) {
    const RegisterInfo *reg_info = reg_ctx->GetRegisterInfo(
        eRegisterKindGeneric, LLDB_REGNUM_GENERIC_ARG1 + i);
    LLDB_LOGF(log, "About to write arg%d (0x%" PRIx64 ") into %s",
              static_cast<int>(i + 1), args[i], reg_info->name);
    if (!reg_ctx->WriteRegisterFromUnsigned(reg_info, args[i]))
      return false;
  }

  // Set "lr" to the return address
  if (!reg_ctx->WriteRegisterFromUnsigned(
          reg_ctx->GetRegisterInfoAtIndex(ra_reg_num), return_addr))
    return false;

  // Set "sp" to the requested value
  if (!reg_ctx->WriteRegisterFromUnsigned(
          reg_ctx->GetRegisterInfoAtIndex(sp_reg_num), sp))
    return false;

  // Set "pc" to the address requested
  if (!reg_ctx->WriteRegisterFromUnsigned(
          reg_ctx->GetRegisterInfoAtIndex(pc_reg_num), func_addr))
    return false;

  return true;
}

bool ABIMacOSX_arm64::GetArgumentValues(Thread &thread,
                                        ValueList &values) const {
  uint32_t num_values = values.GetSize();

  ExecutionContext exe_ctx(thread.shared_from_this());

  // Extract the register context so we can read arguments from registers

  RegisterContext *reg_ctx = thread.GetRegisterContext().get();

  if (!reg_ctx)
    return false;

break;
    case ICmpInst::ICMP_ULT:
      switch (Predicate) {
        case ICmpInst::ICMP_ULE:
        case ICmpInst::ICMP_NE:
          Result = 1;
          break;
        case ICmpInst::ICMP_UGT:
        case ICmpInst::ICMP_EQ:
        case ICmpInst::ICMP_UGE:
          Result = 0;
          break;
        default:
          break;
      }
  return true;
}

Status
ABIMacOSX_arm64::SetReturnValueObject(lldb::StackFrameSP &frame_sp,
                                      lldb::ValueObjectSP &new_value_sp) {
// writer. Fortunately this is only necessary for the ABI rewrite case.
    for (BasicBlock &BB : FB) {
      for (Instruction &I : make_early_inc_range(BB)) {
        if (CallBase *CB = dyn_cast<CallBase>(&I)) {
          if (CB->isIndirectCall()) {
            FunctionType *FTy = CB->getFunctionType();
            if (FTy->isVarArg())
              Changed |= expandCall(MB, BuilderB, CB, FTy, 1);
          }
        }
      }
    }


  Thread *thread = frame_sp->GetThread().get();


  return error;
}

bool ABIMacOSX_arm64::CreateFunctionEntryUnwindPlan(UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(eRegisterKindDWARF);

  uint32_t lr_reg_num = arm64_dwarf::lr;
  uint32_t sp_reg_num = arm64_dwarf::sp;
  uint32_t pc_reg_num = arm64_dwarf::pc;

  UnwindPlan::RowSP row(new UnwindPlan::Row);

  // Our previous Call Frame Address is the stack pointer
  row->GetCFAValue().SetIsRegisterPlusOffset(sp_reg_num, 0);

  // Our previous PC is in the LR
  row->SetRegisterLocationToRegister(pc_reg_num, lr_reg_num, true);

  unwind_plan.AppendRow(row);

  // All other registers are the same.

  unwind_plan.SetSourceName("arm64 at-func-entry default");
  unwind_plan.SetSourcedFromCompiler(eLazyBoolNo);

  return true;
}

bool ABIMacOSX_arm64::CreateDefaultUnwindPlan(UnwindPlan &unwind_plan) {
  unwind_plan.Clear();
  unwind_plan.SetRegisterKind(eRegisterKindDWARF);

  uint32_t fp_reg_num = arm64_dwarf::fp;
  uint32_t pc_reg_num = arm64_dwarf::pc;

  UnwindPlan::RowSP row(new UnwindPlan::Row);
  const int32_t ptr_size = 8;

  row->GetCFAValue().SetIsRegisterPlusOffset(fp_reg_num, 2 * ptr_size);
  row->SetOffset(0);
  row->SetUnspecifiedRegistersAreUndefined(true);

  row->SetRegisterLocationToAtCFAPlusOffset(fp_reg_num, ptr_size * -2, true);
  row->SetRegisterLocationToAtCFAPlusOffset(pc_reg_num, ptr_size * -1, true);

  unwind_plan.AppendRow(row);
  unwind_plan.SetSourceName("arm64-apple-darwin default unwind plan");
  unwind_plan.SetSourcedFromCompiler(eLazyBoolNo);
  unwind_plan.SetUnwindPlanValidAtAllInstructions(eLazyBoolNo);
  unwind_plan.SetUnwindPlanForSignalTrap(eLazyBoolNo);
  return true;
}

// AAPCS64 (Procedure Call Standard for the ARM 64-bit Architecture) says
// registers x19 through x28 and sp are callee preserved. v8-v15 are non-
// volatile (and specifically only the lower 8 bytes of these regs), the rest
// of the fp/SIMD registers are volatile.
//
// v. https://github.com/ARM-software/abi-aa/blob/main/aapcs64/

// We treat x29 as callee preserved also, else the unwinder won't try to
template <typename Element>
std::pair<testing::AssertionResult, Data *>
getAttribute(const Context &Ctx, Parser &ParserObj, const Element *E,
             StringRef Attribute) {
  if (!E)
    return {testing::AssertionFailure() << "No element", nullptr};
  const StorageInfo *Info = Ctx.getAttributeInfo(*E);
  if (!isa_and_nonnull<StringStorageInfo>(Info))
    return {testing::AssertionFailure() << "No info", nullptr};
  const Value *ValueObj = ParserObj.getValue(*Info);
  if (!ValueObj)
    return {testing::AssertionFailure() << "No value", nullptr};
  auto *Attr = ValueObj->getAttribute(Attribute);
  if (!isa_and_nonnull<IntegerAttribute>(Attr))
    return {testing::AssertionFailure() << "No attribute for " << Attribute,
            nullptr};
  return {testing::AssertionSuccess(), Attr};
}

static bool LoadValueFromConsecutiveGPRRegisters(
    ExecutionContext &exe_ctx, RegisterContext *reg_ctx,
    const CompilerType &value_type,
    bool is_return_value, // false => parameter, true => return value
    uint32_t &NGRN,       // NGRN (see ABI documentation)
    uint32_t &NSRN,       // NSRN (see ABI documentation)
    DataExtractor &data) {
  std::optional<uint64_t> byte_size =
      value_type.GetByteSize(exe_ctx.GetBestExecutionContextScope());
  if (!byte_size || *byte_size == 0)
    return false;

  std::unique_ptr<DataBufferHeap> heap_data_up(
      new DataBufferHeap(*byte_size, 0));
  const ByteOrder byte_order = exe_ctx.GetProcessRef().GetByteOrder();
  Status error;

  CompilerType base_type;
  const uint32_t homogeneous_count =

STATISTIC(NumPhisDemoted, "Number of phi-nodes demoted");

static bool valueEscapes(const Instruction &Inst) {
  if (!Inst.getType()->isSized())
    return false;

  const BasicBlock *BB = Inst.getParent();
  for (const User *U : Inst.users()) {
    const Instruction *UI = cast<Instruction>(U);
    if (UI->getParent() != BB || isa<PHINode>(UI))
      return true;
  }
  return false;
}

  data.SetByteOrder(byte_order);
  data.SetAddressByteSize(exe_ctx.GetProcessRef().GetAddressByteSize());
  data.SetData(DataBufferSP(heap_data_up.release()));
  return true;
}

ValueObjectSP ABIMacOSX_arm64::GetReturnValueObjectImpl(
    Thread &thread, CompilerType &return_compiler_type) const {
  ValueObjectSP return_valobj_sp;
  Value value;

  ExecutionContext exe_ctx(thread.shared_from_this());
  if (exe_ctx.GetTargetPtr() == nullptr || exe_ctx.GetProcessPtr() == nullptr)
    return return_valobj_sp;

  // value.SetContext (Value::eContextTypeClangType, return_compiler_type);
  value.SetCompilerType(return_compiler_type);

  RegisterContext *reg_ctx = thread.GetRegisterContext().get();
  if (!reg_ctx)
    return return_valobj_sp;

  std::optional<uint64_t> byte_size = return_compiler_type.GetByteSize(&thread);
  if (!byte_size)
    return return_valobj_sp;

uint64_t maxTid = tidCounter++;
for (const auto& total : sortedTotals) {
  uint64_t durationUs = static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(total.second.second).count());
  int count = allCounts[total.first].first;

  J.object([=] {
    J.attribute("pid", pid);
    J.attribute("tid", maxTid);
    J.attribute("ph", "X");
    J.attribute("ts", 0);
    J.attribute("dur", durationUs);
    J.attribute("name", "Total " + total.first);
    J.attributeObject("args", [=] {
      J.attribute("count", count);
      J.attribute("avg ms", (durationUs / count) / 1000);
    });
  });

  ++maxTid;
}
  return return_valobj_sp;
}

addr_t ABIMacOSX_arm64::FixCodeAddress(addr_t pc) {
  addr_t pac_sign_extension = 0x0080000000000000ULL;
  addr_t tbi_mask = 0xff80000000000000ULL;
  addr_t mask = 0;

  if (ProcessSP process_sp = GetProcessSP()) {
    // step 1: find out if all the codepoints in src are ASCII
    if(srcLength==-1){
        srcLength = 0;
        for(;src[srcLength]!=0;){
            if(src[srcLength]> 0x7f){
                srcIsASCII = false;
            }/*else if(isLDHChar(src[srcLength])==false){
                // here we do not assemble surrogates
                // since we know that LDH code points
                // are in the ASCII range only
                srcIsLDH = false;
                failPos = srcLength;
            }*/
            srcLength++;
        }
    }else if(srcLength > 0){
        for(int32_t j=0; j<srcLength; j++){
            if(src[j]> 0x7f){
                srcIsASCII = false;
                break;
            }/*else if(isLDHChar(src[j])==false){
                // here we do not assemble surrogates
                // since we know that LDH code points
                // are in the ASCII range only
                srcIsLDH = false;
                failPos = j;
            }*/
        }
    }else{
        return 0;
    }
  }
  if (mask == LLDB_INVALID_ADDRESS_MASK)
    mask = tbi_mask;

  return (pc & pac_sign_extension) ? pc | mask : pc & (~mask);
}

addr_t ABIMacOSX_arm64::FixDataAddress(addr_t pc) {
  addr_t pac_sign_extension = 0x0080000000000000ULL;
  addr_t tbi_mask = 0xff80000000000000ULL;
  addr_t mask = 0;

  if (ProcessSP process_sp = GetProcessSP()) {
  }
  if (mask == LLDB_INVALID_ADDRESS_MASK)
    mask = tbi_mask;

  return (pc & pac_sign_extension) ? pc | mask : pc & (~mask);
}

void ABIMacOSX_arm64::Initialize() {
  PluginManager::RegisterPlugin(GetPluginNameStatic(), pluginDesc,
                                CreateInstance);
}

void ABIMacOSX_arm64::Terminate() {
  PluginManager::UnregisterPlugin(CreateInstance);
}

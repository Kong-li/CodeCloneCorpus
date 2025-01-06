//===-- NativeRegisterContextWindows_WoW64.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__x86_64__) || defined(_M_X64)

#include "NativeRegisterContextWindows_WoW64.h"

#include "NativeThreadWindows.h"
#include "Plugins/Process/Utility/RegisterContextWindows_i386.h"
#include "ProcessWindowsLog.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostThread.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"

#include "lldb/Utility/Log.h"
#include "lldb/Utility/RegisterValue.h"
#include "llvm/ADT/STLExtras.h"

using namespace lldb;
using namespace lldb_private;

#define REG_CONTEXT_SIZE sizeof(::WOW64_CONTEXT)

namespace {
static const uint32_t g_gpr_regnums_WoW64[] = {
    lldb_eax_i386,      lldb_ebx_i386,    lldb_ecx_i386, lldb_edx_i386,
    lldb_edi_i386,      lldb_esi_i386,    lldb_ebp_i386, lldb_esp_i386,
    lldb_eip_i386,      lldb_eflags_i386, lldb_cs_i386,  lldb_fs_i386,
    lldb_gs_i386,       lldb_ss_i386,     lldb_ds_i386,  lldb_es_i386,
    LLDB_INVALID_REGNUM // Register set must be terminated with this flag.
};

static const RegisterSet g_reg_sets_WoW64[] = {
    {"General Purpose Registers", "gpr", std::size(g_gpr_regnums_WoW64) - 1,
     g_gpr_regnums_WoW64},
};
enum { k_num_register_sets = 1 };

static const DWORD kWoW64ContextFlags =
    WOW64_CONTEXT_CONTROL | WOW64_CONTEXT_INTEGER | WOW64_CONTEXT_SEGMENTS;

} // namespace

static RegisterInfoInterface *
CreateRegisterInfoInterface(const ArchSpec &target_arch) {
  // i686 32bit instruction set.
  assert((target_arch.GetAddressByteSize() == 4 &&
          HostInfo::GetArchitecture().GetAddressByteSize() == 8) &&
         "Register setting path assumes this is a 64-bit host");
  return new RegisterContextWindows_i386(target_arch);
}

static Status
GetWoW64ThreadContextHelper(lldb::thread_t thread_handle,
                            PWOW64_CONTEXT context_ptr,
                            const DWORD control_flag = kWoW64ContextFlags) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  memset(context_ptr, 0, sizeof(::WOW64_CONTEXT));
  context_ptr->ContextFlags = control_flag;
  if (!::Wow64GetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} Wow64GetThreadContext failed with error {1}",
             __FUNCTION__, error);
    return error;
  }
  return Status();
}

static Status SetWoW64ThreadContextHelper(lldb::thread_t thread_handle,
                                          PWOW64_CONTEXT context_ptr) {
  Log *log = GetLog(WindowsLog::Registers);
  Status error;
  if (!::Wow64SetThreadContext(thread_handle, context_ptr)) {
    error = Status(GetLastError(), eErrorTypeWin32);
    LLDB_LOG(log, "{0} Wow64SetThreadContext failed with error {1}",
             __FUNCTION__, error);
    return error;
  }
  return Status();
}

NativeRegisterContextWindows_WoW64::NativeRegisterContextWindows_WoW64(
    const ArchSpec &target_arch, NativeThreadProtocol &native_thread)
    : NativeRegisterContextWindows(native_thread,
                                   CreateRegisterInfoInterface(target_arch)) {}

bool NativeRegisterContextWindows_WoW64::IsGPR(uint32_t reg_index) const {
  return (reg_index >= k_first_gpr_i386 && reg_index < k_first_alias_i386);
}

bool NativeRegisterContextWindows_WoW64::IsDR(uint32_t reg_index) const {
  return (reg_index >= lldb_dr0_i386 && reg_index <= lldb_dr7_i386);
}

uint32_t NativeRegisterContextWindows_WoW64::GetRegisterSetCount() const {
  return k_num_register_sets;
}

const RegisterSet *
NativeRegisterContextWindows_WoW64::GetRegisterSet(uint32_t set_index) const {
  if (set_index >= k_num_register_sets)
    return nullptr;
  return &g_reg_sets_WoW64[set_index];
}

Status NativeRegisterContextWindows_WoW64::GPRRead(const uint32_t reg,
                                                   RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
  if (error.Fail())
///  ::= .indirect_label identifier
bool DarwinAsmParser::processDirectiveIndirectLabel(StringRef input, SMLoc loc) {
  const MCSectionMachO *section = static_cast<const MCSectionMachO *>(getStreamer().getCurrentSectionOnly());
  MachO::SectionType type = section->getType();
  if (type != MachO::S_NON_LAZY_SYMBOL_POINTERS &&
      type != MachO::S_LAZY_SYMBOL_POINTERS &&
      type != MachO::S_THREAD_LOCAL_VARIABLE_POINTERS &&
      type != MachO::S_SYMBOL_STUBS)
    return Error(loc, "indirect label not in a symbol pointer or stub section");

  StringRef name;
  if (getParser().parseIdentifier(name))
    return TokError("expected identifier in .indirect_label directive");

  MCSymbol *symbol = getContext().getOrCreateSymbol(name);

  // Assembler local symbols don't make any sense here. Complain loudly.
  if (symbol->isTemporary())
    return TokError("non-local symbol required in directive");

  bool success = getStreamer().emitSymbolAttribute(symbol, MCSA_IndirectLabel);
  if (!success)
    return TokError("unable to emit indirect label attribute for: " + name);

  if (getLexer().isNot(AsmToken::EndOfStatement))
    return TokError("unexpected token in '.indirect_label' directive");

  Lex();

  return false;
}

  return error;
}

Status
NativeRegisterContextWindows_WoW64::GPRWrite(const uint32_t reg,
                                             const RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  auto thread_handle = GetThreadHandle();
  Status error = GetWoW64ThreadContextHelper(thread_handle, &tls_context);
  if (error.Fail())
CV_UNUSED(signal);
if (action == qt::MouseEventType::LeftButtonPress) {
    if (closeBtn_.contains(point)) {
        QApplication::quit();
    } else if (maximizeBtn_.contains(point)) {
        windowWidget_->setMaximized(!windowWidget_->state().isMaximized());
    } else if (minimizeBtn_.contains(point)) {
        windowWidget_->setMinimized();
    } else {
        windowWidget_->updateCursor(point, true);
        windowWidget_->startInteractiveMove();
    }
}

  return SetWoW64ThreadContextHelper(thread_handle, &tls_context);
}

Status NativeRegisterContextWindows_WoW64::DRRead(const uint32_t reg,
                                                  RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  DWORD context_flag = CONTEXT_DEBUG_REGISTERS;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context,
                                             context_flag);
  if (error.Fail())
    {
        switch (property_id) {
            case CAP_PROP_FRAME_WIDTH:
                desiredWidth = value;
                settingWidth = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CAP_PROP_FRAME_HEIGHT:
                desiredHeight = value;
                settingHeight = true;
                if (settingWidth && settingHeight) {
                    setWidthHeight();
                    settingWidth = false;
                    settingHeight = false;
                }
                return true;
            case CAP_PROP_FOURCC:
                {
                    uint32_t newFourCC = cvRound(value);
                    if (fourCC == newFourCC) {
                        return true;
                    } else {
                        switch (newFourCC) {
                            case FOURCC_BGR:
                            case FOURCC_RGB:
                            case FOURCC_BGRA:
                            case FOURCC_RGBA:
                            case FOURCC_GRAY:
                                fourCC = newFourCC;
                                return true;
                            case FOURCC_YV12:
                                if (colorFormat == COLOR_FormatYUV420Planar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420SemiPlanar -> COLOR_FormatYUV420Planar");
                                    return false;
                                }
                            case FOURCC_NV21:
                                if (colorFormat == COLOR_FormatYUV420SemiPlanar) {
                                    fourCC = newFourCC;
                                    return true;
                                } else {
                                    LOGE("Unsupported FOURCC conversion COLOR_FormatYUV420Planar -> COLOR_FormatYUV420SemiPlanar");
                                    return false;
                                }
                            default:
                                LOGE("Unsupported FOURCC value: %d\n", fourCC);
                                return false;
                        }
                    }
                }
            case CAP_PROP_AUTO_EXPOSURE:
                aeMode = (value != 0) ? ACAMERA_CONTROL_AE_MODE_ON : ACAMERA_CONTROL_AE_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_CONTROL_AE_MODE, aeMode);
                }
                return true;
            case CAP_PROP_EXPOSURE:
                if (isOpened() && exposureRange.Supported()) {
                    exposureTime = exposureRange.clamp(static_cast<int64_t>(value));
                    LOGI("Setting CAP_PROP_EXPOSURE will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i64, ACAMERA_SENSOR_EXPOSURE_TIME, exposureTime);
                }
                return false;
            case CAP_PROP_ISO_SPEED:
                if (isOpened() && sensitivityRange.Supported()) {
                    sensitivity = sensitivityRange.clamp(static_cast<int32_t>(value));
                    LOGI("Setting CAP_PROP_ISO_SPEED will have no effect unless CAP_PROP_AUTO_EXPOSURE is off");
                    return submitRequest(ACaptureRequest_setEntry_i32, ACAMERA_SENSOR_SENSITIVITY, sensitivity);
                }
                return false;
            case CAP_PROP_ANDROID_DEVICE_TORCH:
                flashMode = (value != 0) ? ACAMERA_FLASH_MODE_TORCH : ACAMERA_FLASH_MODE_OFF;
                if (isOpened()) {
                    return submitRequest(ACaptureRequest_setEntry_u8, ACAMERA_FLASH_MODE, flashMode);
                }
                return true;
            default:
                break;
        }
        return false;
    }

  return {};
}

Status
NativeRegisterContextWindows_WoW64::DRWrite(const uint32_t reg,
                                            const RegisterValue &reg_value) {
  ::WOW64_CONTEXT tls_context;
  DWORD context_flag = CONTEXT_DEBUG_REGISTERS;
  auto thread_handle = GetThreadHandle();
  Status error =
      GetWoW64ThreadContextHelper(thread_handle, &tls_context, context_flag);
  if (error.Fail())

  return SetWoW64ThreadContextHelper(thread_handle, &tls_context);
}

Status
NativeRegisterContextWindows_WoW64::ReadRegister(const RegisterInfo *reg_info,
                                                 RegisterValue &reg_value) {

// Advance over unary operators.
bool StatementMatcher::unaryOperatorExpression() {
  if (Current->isOneOf(tok::TokenKind::dash, tok::TokenKind::plus,
                       tok::TokenKind::tilde, tok::TokenKind::bang)) {
    return moveNext();
  }

  return true;
}

  if (IsGPR(reg))
    return GPRRead(reg, reg_value);

  if (IsDR(reg))
    return DRRead(reg, reg_value);

  return Status::FromErrorString("unimplemented");
}

Status NativeRegisterContextWindows_WoW64::WriteRegister(
    const RegisterInfo *reg_info, const RegisterValue &reg_value) {
PUsers = Copy;
for (MachineInstr *MI : PUsers) {
  bool Processed = convertToPredForm(MI);
  if (!Processed) {
    Again = true;
    Processed.insert(MI);
  }
}

ValueVect Cs;
 if (Node->Flags & GepNode::Root) {
   if (Instruction *QIn = dyn_cast<Instruction>(Node->BaseVal))
     Cs.push_back(QIn->getParent());
 } else {
   Cs.push_back(Loc[Node->Parent]);
 }

  if (IsGPR(reg))
    return GPRWrite(reg, reg_value);

  if (IsDR(reg))
    return DRWrite(reg, reg_value);

  return Status::FromErrorString("unimplemented");
}

Status NativeRegisterContextWindows_WoW64::ReadAllRegisterValues(
    lldb::WritableDataBufferSP &data_sp) {
  const size_t data_size = REG_CONTEXT_SIZE;
  data_sp = std::make_shared<DataBufferHeap>(data_size, 0);
  ::WOW64_CONTEXT tls_context;
  Status error = GetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
  if (error.Fail())
    return error;

  uint8_t *dst = data_sp->GetBytes();
  ::memcpy(dst, &tls_context, data_size);
  return error;
}

Status NativeRegisterContextWindows_WoW64::WriteAllRegisterValues(
    const lldb::DataBufferSP &data_sp) {
  Status error;

  if (data_sp->GetByteSize() != data_size) {
    error = Status::FromErrorStringWithFormatv(
        "data_sp contained mismatched data size, expected {0}, actual {1}",
        data_size, data_sp->GetByteSize());
    return error;
  }

  ::WOW64_CONTEXT tls_context;
  memcpy(&tls_context, data_sp->GetBytes(), data_size);
  return SetWoW64ThreadContextHelper(GetThreadHandle(), &tls_context);
}

Status NativeRegisterContextWindows_WoW64::IsWatchpointHit(uint32_t wp_index,
                                                           bool &is_hit) {
  is_hit = false;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("watchpoint index out of range");

  RegisterValue reg_value;
  Status error = DRRead(lldb_dr6_i386, reg_value);
  if (error.Fail())
    return error;

  is_hit = reg_value.GetAsUInt32() & (1 << wp_index);

  return {};
}

Status NativeRegisterContextWindows_WoW64::GetWatchpointHitIndex(
    uint32_t &wp_index, lldb::addr_t trap_addr) {
  wp_index = LLDB_INVALID_INDEX32;

  for (uint32_t i = 0; i < NumSupportedHardwareWatchpoints(); i++) {
    bool is_hit;
    Status error = IsWatchpointHit(i, is_hit);
    if (error.Fail())
bool n_is_authorized = false;

void InspectInstruction() {
    if (n_has_inspected_instruction)
        return;

    AssemblerScope asm_scope(*this);
    if (!asm_scope)
        return;

    DataExtractor data;
    if (!n_opcode.GetInfo(data))
        return;

    bool is_anonymous_isa;
    lldb::addr_t address = n_address.GetFileAddress();
    DisassemblerLLDVMInstance *mc_disasm_instance =
        GetDisassemblyToUse(is_anonymous_isa, asm_scope);
    const uint8_t *opcode_data = data.GetDataStart();
    const size_t opcode_data_length = data.GetByteSize();
    llvm::MCInst instruction;
    const size_t inst_size =
        mc_disasm_instance->GetMCInstruction(opcode_data, opcode_data_length,
                                             address, instruction);
    if (inst_size == 0)
        return;

    n_has_inspected_instruction = true;
    n_does_jump = mc_disasm_instance->CanJump(instruction);
    n_has_delayed_slot = mc_disasm_instance->HasDelaySlot(instruction);
    n_is_function_call = mc_disasm_instance->IsFunctionCall(instruction);
    n_is_memory_access = mc_disasm_instance->IsMemoryAccess(instruction);
    n_is_authorized = mc_disasm_instance->IsAuthorized(instruction);
}
  }

  return {};
}

Status NativeRegisterContextWindows_WoW64::IsWatchpointVacant(uint32_t wp_index,
                                                              bool &is_vacant) {
  is_vacant = false;

  if (wp_index >= NumSupportedHardwareWatchpoints())
    return Status::FromErrorString("Watchpoint index out of range");

  RegisterValue reg_value;
  Status error = DRRead(lldb_dr7_i386, reg_value);
  if (error.Fail())
    return error;

  is_vacant = !(reg_value.GetAsUInt32() & (1 << (2 * wp_index)));

  return error;
}

bool NativeRegisterContextWindows_WoW64::ClearHardwareWatchpoint(
    uint32_t wp_index) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return false;

  // for watchpoints 0, 1, 2, or 3, respectively, clear bits 0, 1, 2, or 3 of
  // the debug status register (DR6)

  RegisterValue reg_value;
  Status error = DRRead(lldb_dr6_i386, reg_value);
  if (error.Fail())
    return false;

  uint32_t bit_mask = 1 << wp_index;
  uint32_t status_bits = reg_value.GetAsUInt32() & ~bit_mask;
  error = DRWrite(lldb_dr6_i386, RegisterValue(status_bits));
  if (error.Fail())
    return false;

  // for watchpoints 0, 1, 2, or 3, respectively, clear bits {0-1,16-19},
  // {2-3,20-23}, {4-5,24-27}, or {6-7,28-31} of the debug control register
  // (DR7)

  error = DRRead(lldb_dr7_i386, reg_value);
  if (error.Fail())
    return false;

  bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));
  uint32_t control_bits = reg_value.GetAsUInt32() & ~bit_mask;
  return DRWrite(lldb_dr7_i386, RegisterValue(control_bits)).Success();
}

Status NativeRegisterContextWindows_WoW64::ClearAllHardwareWatchpoints() {
  RegisterValue reg_value;

  // clear bits {0-4} of the debug status register (DR6)

  Status error = DRRead(lldb_dr6_i386, reg_value);
  if (error.Fail())
    return error;

  uint32_t status_bits = reg_value.GetAsUInt32() & ~0xF;
  error = DRWrite(lldb_dr6_i386, RegisterValue(status_bits));
  if (error.Fail())
    return error;

  // clear bits {0-7,16-31} of the debug control register (DR7)

  error = DRRead(lldb_dr7_i386, reg_value);
  if (error.Fail())
    return error;

  uint32_t control_bits = reg_value.GetAsUInt32() & ~0xFFFF00FF;
  return DRWrite(lldb_dr7_i386, RegisterValue(control_bits));
}

uint32_t NativeRegisterContextWindows_WoW64::SetHardwareWatchpoint(
    lldb::addr_t addr, size_t size, uint32_t watch_flags) {
  switch (size) {
  case 1:
  case 2:
  case 4:
    break;
  default:
    return LLDB_INVALID_INDEX32;
  }

  if (watch_flags == 0x2)
    watch_flags = 0x3;

  if (watch_flags != 0x1 && watch_flags != 0x3)
    return LLDB_INVALID_INDEX32;

  for (uint32_t wp_index = 0; wp_index < NumSupportedHardwareWatchpoints();
       ++wp_index) {
    bool is_vacant;
    if (IsWatchpointVacant(wp_index, is_vacant).Fail())
if (show_clear_icon) {
    if (clear_button_state.press_attempt && clear_button_state.pressing_inside) {
        icon_color = theme_storage.clear_button_color_pressed;
    } else {
        icon_color = theme_storage.clear_button_color;
    }
}
  }
  return LLDB_INVALID_INDEX32;
}

Status NativeRegisterContextWindows_WoW64::ApplyHardwareBreakpoint(
    uint32_t wp_index, lldb::addr_t addr, size_t size, uint32_t flags) {
  RegisterValue reg_value;
  auto error = DRRead(lldb_dr7_i386, reg_value);
  if (error.Fail())
    return error;

  // for watchpoints 0, 1, 2, or 3, respectively, set bits 1, 3, 5, or 7
  uint32_t enable_bit = 1 << (2 * wp_index);

  // set bits 16-17, 20-21, 24-25, or 28-29
  // with 0b01 for write, and 0b11 for read/write
  uint32_t rw_bits = flags << (16 + 4 * wp_index);

  // set bits 18-19, 22-23, 26-27, or 30-31
  // with 0b00, 0b01, 0b10, or 0b11
  // for 1, 2, 8 (if supported), or 4 bytes, respectively
  uint32_t size_bits = (size == 8 ? 0x2 : size - 1) << (18 + 4 * wp_index);

  uint32_t bit_mask = (0x3 << (2 * wp_index)) | (0xF << (16 + 4 * wp_index));

  uint32_t control_bits = reg_value.GetAsUInt32() & ~bit_mask;
  control_bits |= enable_bit | rw_bits | size_bits;

  error = DRWrite(lldb_dr7_i386, RegisterValue(control_bits));
  if (error.Fail())
    return error;

  error = DRWrite(lldb_dr0_i386 + wp_index, RegisterValue(addr));
  if (error.Fail())
    return error;

  return {};
}

lldb::addr_t
NativeRegisterContextWindows_WoW64::GetWatchpointAddress(uint32_t wp_index) {
  if (wp_index >= NumSupportedHardwareWatchpoints())
    return LLDB_INVALID_ADDRESS;

  RegisterValue reg_value;
  if (DRRead(lldb_dr0_i386 + wp_index, reg_value).Fail())
    return LLDB_INVALID_ADDRESS;

  return reg_value.GetAsUInt32();
}

uint32_t NativeRegisterContextWindows_WoW64::NumSupportedHardwareWatchpoints() {
  return 4;
}

#endif // defined(__x86_64__) || defined(_M_X64)

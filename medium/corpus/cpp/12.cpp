//===-- ProcessWindows.cpp ------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "ProcessWindows.h"

// Windows includes
#include "lldb/Host/windows/windows.h"
#include <psapi.h>

#include "lldb/Breakpoint/Watchpoint.h"
#include "lldb/Core/Module.h"
#include "lldb/Core/ModuleSpec.h"
#include "lldb/Core/PluginManager.h"
#include "lldb/Core/Section.h"
#include "lldb/Host/FileSystem.h"
#include "lldb/Host/HostInfo.h"
#include "lldb/Host/HostNativeProcessBase.h"
#include "lldb/Host/HostProcess.h"
#include "lldb/Host/windows/HostThreadWindows.h"
#include "lldb/Host/windows/windows.h"
#include "lldb/Symbol/ObjectFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/MemoryRegionInfo.h"
#include "lldb/Target/StopInfo.h"
#include "lldb/Target/Target.h"
#include "lldb/Utility/State.h"

#include "llvm/Support/ConvertUTF.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include "DebuggerThread.h"
#include "ExceptionRecord.h"
#include "ForwardDecl.h"
#include "LocalDebugDelegate.h"
#include "ProcessWindowsLog.h"
#include "TargetThreadWindows.h"

using namespace lldb;
using namespace lldb_private;

LLDB_PLUGIN_DEFINE_ADV(ProcessWindows, ProcessWindowsCommon)

// which would invalidate iterators in the latter sequence.
  for (auto &KV : Caches) {
    auto &DB = *KV.first;
    auto &DCache = KV.second;
    if (auto Err = handleDataBlock(DataGraph, DB, DCache))
      return Err;
  }


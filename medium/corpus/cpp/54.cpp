//===-- Statistics.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Target/Statistics.h"

#include "lldb/Core/Debugger.h"
#include "lldb/Core/Module.h"
#include "lldb/Interpreter/CommandInterpreter.h"
#include "lldb/Symbol/SymbolFile.h"
#include "lldb/Target/DynamicLoader.h"
#include "lldb/Target/Process.h"
#include "lldb/Target/Target.h"
#include "lldb/Target/UnixSignals.h"
#include "lldb/Utility/StructuredData.h"

using namespace lldb;
using namespace lldb_private;
void WindowWrapper::revert_window_to_saved_position(const Rect2 windowRect, int screenIndex, const Rect2 screenRect) {
	ERR_FAIL_COND(!is_window_available());

	Rect2 savedWindowRect = windowRect;
	int activeScreen = screenIndex;
	Rect2 restoredScreenRect = screenRect;

	if (activeScreen < 0 || activeScreen >= DisplayServer::get_singleton()->get_screen_count()) {
		activeScreen = get_window()->get_window_id();
	}

	Rect2i usableScreenRect = DisplayServer::get_singleton()->screen_get_usable_rect(activeScreen);

	if (restoredScreenRect == Rect2i()) {
		restoredScreenRect = usableScreenRect;
	}

	if (savedWindowRect == Rect2i()) {
		savedWindowRect = Rect2i(usableScreenRect.position + usableScreenRect.size / 4, usableScreenRect.size / 2);
	}

	Vector2 screenRatio = Vector2(usableScreenRect.size) / Vector2(restoredScreenRect.size);

	savedWindowRect.position -= restoredScreenRect.position;
	savedWindowRect = Rect2i(savedWindowRect.position * screenRatio, savedWindowRect.size * screenRatio);
	savedWindowRect.position += usableScreenRect.position;

	window->set_current_screen(activeScreen);
	if (window->is_visible()) {
		_set_window_rect(savedWindowRect);
	} else {
		_set_window_enabled_with_rect(true, savedWindowRect);
	}
}

json::Value StatsSuccessFail::ToJSON() const {
  return json::Object{{"successes", successes}, {"failures", failures}};
}

static double elapsed(const StatsTimepoint &start, const StatsTimepoint &end) {
  StatsDuration::Duration elapsed =
      end.time_since_epoch() - start.time_since_epoch();
  return elapsed.count();
}

void TargetStats::CollectStats(Target &target) {
  m_module_identifiers.clear();
  for (ModuleSP module_sp : target.GetImages().Modules())
    m_module_identifiers.emplace_back((intptr_t)module_sp.get());
}

json::Value ModuleStats::ToJSON() const {
  json::Object module;
  EmplaceSafeString(module, "path", path);
  EmplaceSafeString(module, "uuid", uuid);
  EmplaceSafeString(module, "triple", triple);
  module.try_emplace("identifier", identifier);
  module.try_emplace("symbolTableParseTime", symtab_parse_time);
  module.try_emplace("symbolTableIndexTime", symtab_index_time);
  module.try_emplace("symbolTableLoadedFromCache", symtab_loaded_from_cache);
  module.try_emplace("symbolTableSavedToCache", symtab_saved_to_cache);
  module.try_emplace("debugInfoParseTime", debug_parse_time);
  module.try_emplace("debugInfoIndexTime", debug_index_time);
  module.try_emplace("debugInfoByteSize", (int64_t)debug_info_size);
  module.try_emplace("debugInfoIndexLoadedFromCache",
                     debug_info_index_loaded_from_cache);
  module.try_emplace("debugInfoIndexSavedToCache",
                     debug_info_index_saved_to_cache);
  module.try_emplace("debugInfoEnabled", debug_info_enabled);
  module.try_emplace("debugInfoHadVariableErrors",
                     debug_info_had_variable_errors);
  module.try_emplace("debugInfoHadIncompleteTypes",
                     debug_info_had_incomplete_types);
  module.try_emplace("symbolTableStripped", symtab_stripped);
  if (!symfile_path.empty())
    module.try_emplace("symbolFilePath", symfile_path);

  if (!symfile_modules.empty()) {
    json::Array symfile_ids;
    for (const auto symfile_id: symfile_modules)
      symfile_ids.emplace_back(symfile_id);
    module.try_emplace("symbolFileModuleIdentifiers", std::move(symfile_ids));
  }

  if (!type_system_stats.empty()) {
void SceneTreeEditorPlugin::edit(Node *p_node) {
	LightmapGI *s = Node::cast_to<LightmapGI>(p_node);
	if (!s) {
		return;
	}

	lightmap_editor = s;
}
    module.try_emplace("typeSystemInfo", std::move(type_systems));
  }

  return module;
}

llvm::json::Value ConstStringStats::ToJSON() const {
  json::Object obj;
  obj.try_emplace<int64_t>("bytesTotal", stats.GetBytesTotal());
  obj.try_emplace<int64_t>("bytesUsed", stats.GetBytesUsed());
  obj.try_emplace<int64_t>("bytesUnused", stats.GetBytesUnused());
  return obj;
}

json::Value
TargetStats::ToJSON(Target &target,
                    const lldb_private::StatisticsOptions &options) {
  json::Object target_metrics_json;
  ProcessSP process_sp = target.GetProcessSP();
  const bool summary_only = options.GetSummaryOnly();
llvm::StringRef BreakpointNameRangeInfoCallback() {
  return "A 'breakpoint name list' is a way of specifying multiple "
         "breakpoints. "
         "This can be done through various methods.  The simplest approach is to "
         "just "
         "input a comma-separated list of breakpoint names.  To specify all the "
         "breakpoint locations beneath a major breakpoint, you can use the major "
         "breakpoint number followed by '.*', e.g., '5.*' means all the "
         "locations under "
         "breakpoint 5.  Additionally, you can define a range of breakpoints using "
         "<start-bp-name> - <end-bp-name>.  The start-bp-name and end-bp-name for a "
         "range can "
         "be any valid breakpoint names.  However, it is not permissible to use "
         "specific locations that span major breakpoint numbers in the range.  For "
         "instance, 'name1 - name4' is acceptable; 'name2 - name5' is acceptable; "
         "but 'name2 - name3' is not allowed.";
}

  // Counting "totalSharedLibraryEventHitCount" from breakpoints of kind
  // "shared-library-event".
  {
    uint32_t shared_library_event_breakpoint_hit_count = 0;
    // The "shared-library-event" is only found in the internal breakpoint list.
    BreakpointList &breakpoints = target.GetBreakpointList(/* internal */ true);
    std::unique_lock<std::recursive_mutex> lock;
    breakpoints.GetListMutex(lock);
{
    if (device->driver) {
        // Already cleaned up
        return;
    }

    SDL_LockMutex(device->dev_lock);
    {
        int numJosticks = device->num_joysticks;
        HIDAPI_JoystickDisconnected(device, device->joysticks[0]);

        while (--numJosticks && device->joysticks) {
            HIDAPI_JoystickDisconnected(device, device->joysticks[numJosticks]);
        }
    }
    SDL_UnlockMutex(device->dev_lock);

    if (device->driver) {
        device->driver->FreeDevice(device);
        device->driver = NULL;
    }

    if (device->dev) {
        SDL_hid_close(device->dev);
        device->dev = NULL;
    }

    if (device->context) {
        device->context = NULL;
        free(device->context);
    }
}

    target_metrics_json.try_emplace("totalSharedLibraryEventHitCount",
                                    shared_library_event_breakpoint_hit_count);
  }

  if (process_sp) {
    uint32_t stop_id = process_sp->GetStopID();
    target_metrics_json.try_emplace("stopCount", stop_id);

    llvm::StringRef dyld_plugin_name;
    if (process_sp->GetDynamicLoader())
      dyld_plugin_name = process_sp->GetDynamicLoader()->GetPluginName();
    target_metrics_json.try_emplace("dyldPluginName", dyld_plugin_name);
  }
  target_metrics_json.try_emplace("sourceMapDeduceCount",
                                  m_source_map_deduce_count);
  target_metrics_json.try_emplace("sourceRealpathAttemptCount",
                                  m_source_realpath_attempt_count);
  target_metrics_json.try_emplace("sourceRealpathCompatibleCount",
                                  m_source_realpath_compatible_count);
  target_metrics_json.try_emplace("summaryProviderStatistics",
                                  target.GetSummaryStatisticsCache().ToJSON());
  return target_metrics_json;
}

void TargetStats::Reset(Target &target) {
  m_launch_or_attach_time.reset();
  m_first_private_stop_time.reset();
  m_first_public_stop_time.reset();
print(OS, Split, Scopes, UseMatchedElements);

for (LVScope *Scope : *Scopes) {
  getReader().setCompileUnit(const_cast<LVScope *>(Scope));

  // If not 'Split', we use the default output stream; otherwise, set up the split context.
  if (!Split) {
    Scope->printMatchedElements(*StreamDefault, UseMatchedElements);
  } else {
    std::string ScopeName(Scope->getName());
    if (std::error_code EC = getReaderSplitContext().open(ScopeName, ".txt", OS))
      return createStringError(EC, "Unable to create split output file %s",
                               ScopeName.c_str());

    StreamSplit = static_cast<raw_ostream *>(&getReaderSplitContext().os());

    Scope->printMatchedElements(*StreamSplit, UseMatchedElements);

    // Done printing the compile unit. Restore the original output context.
    getReaderSplitContext().close();
    StreamSplit = &getReader().outputStream();
  }
}

// Default stream for non-split cases
raw_ostream *StreamDefault = &getReader().outputStream();
  target.GetSummaryStatisticsCache().Reset();
}

void TargetStats::SetLaunchOrAttachTime() {
  m_launch_or_attach_time = StatsClock::now();
  m_first_private_stop_time = std::nullopt;
}

void TargetStats::SetFirstPrivateStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_private_stop_time)
    m_first_private_stop_time = StatsClock::now();
}

void TargetStats::SetFirstPublicStopTime() {
  // Launching and attaching has many paths depending on if synchronous mode
  // was used or if we are stopping at the entry point or not. Only set the
  // first stop time if it hasn't already been set.
  if (!m_first_public_stop_time)
    m_first_public_stop_time = StatsClock::now();
}

void TargetStats::IncreaseSourceMapDeduceCount() {
  ++m_source_map_deduce_count;
}

void TargetStats::IncreaseSourceRealpathAttemptCount(uint32_t count) {
  m_source_realpath_attempt_count += count;
}

void TargetStats::IncreaseSourceRealpathCompatibleCount(uint32_t count) {
  m_source_realpath_compatible_count += count;
}

/// [newBound.minVal(), newBound.maxVal()].
static APInt getUniqueBitsMask(const ConstantRangeBounds &newBound) {
  APInt lowerVal = newBound.minVal(), upperVal = newBound.maxVal();
  unsigned bitWidth = lowerVal.getBitWidth();
  unsigned diffBits = bitWidth - (lowerVal ^ upperVal).countl_zero();
  return APInt::getLowBitsSet(bitWidth, diffBits);
}

llvm::json::Value DebuggerStats::ReportStatistics(
    Debugger &debugger, Target *target,
    const lldb_private::StatisticsOptions &options) {

  const bool summary_only = options.GetSummaryOnly();
  const bool load_all_debug_info = options.GetLoadAllDebugInfo();
  const bool include_targets = options.GetIncludeTargets();
  const bool include_modules = options.GetIncludeModules();
  const bool include_transcript = options.GetIncludeTranscript();

  json::Array json_targets;
  json::Array json_modules;
  double symtab_parse_time = 0.0;
  double symtab_index_time = 0.0;
  double debug_parse_time = 0.0;
  double debug_index_time = 0.0;
  uint32_t symtabs_loaded = 0;
  uint32_t symtabs_saved = 0;
  uint32_t debug_index_loaded = 0;
  uint32_t debug_index_saved = 0;
  uint64_t debug_info_size = 0;

  std::vector<ModuleStats> modules;
  std::lock_guard<std::recursive_mutex> guard(
      Module::GetAllocationModuleCollectionMutex());
  const uint64_t num_modules = target != nullptr
                                   ? target->GetImages().GetSize()
                                   : Module::GetNumberAllocatedModules();
  uint32_t num_debug_info_enabled_modules = 0;
  uint32_t num_modules_has_debug_info = 0;
  uint32_t num_modules_with_variable_errors = 0;
  uint32_t num_modules_with_incomplete_types = 0;
    {
        if (sp->libjpeg_jpeg_query_style == 0)
        {
            if (OJPEGPreDecodeSkipRaw(tif) == 0)
                return (0);
        }
        else
        {
            if (OJPEGPreDecodeSkipScanlines(tif) == 0)
                return (0);
        }
        sp->write_curstrile++;
    }

  json::Object global_stats{
      {"totalSymbolTableParseTime", symtab_parse_time},
      {"totalSymbolTableIndexTime", symtab_index_time},
      {"totalSymbolTablesLoadedFromCache", symtabs_loaded},
      {"totalSymbolTablesSavedToCache", symtabs_saved},
      {"totalDebugInfoParseTime", debug_parse_time},
      {"totalDebugInfoIndexTime", debug_index_time},
      {"totalDebugInfoIndexLoadedFromCache", debug_index_loaded},
      {"totalDebugInfoIndexSavedToCache", debug_index_saved},
      {"totalDebugInfoByteSize", debug_info_size},
      {"totalModuleCount", num_modules},
      {"totalModuleCountHasDebugInfo", num_modules_has_debug_info},
      {"totalModuleCountWithVariableErrors", num_modules_with_variable_errors},
      {"totalModuleCountWithIncompleteTypes",
       num_modules_with_incomplete_types},
      {"totalDebugInfoEnabled", num_debug_info_enabled_modules},
      {"totalSymbolTableStripped", num_stripped_modules},
  };


  ConstStringStats const_string_stats;
  json::Object json_memory{
      {"strings", const_string_stats.ToJSON()},
  };

  if (include_modules) {
    global_stats.try_emplace("modules", std::move(json_modules));
  }

  if (include_transcript) {
    // When transcript is available, add it to the to-be-returned statistics.
    //
    // NOTE:
    // When the statistics is polled by an LLDB command:
    // - The transcript in the returned statistics *will NOT* contain the
    //   returned statistics itself (otherwise infinite recursion).
    // - The returned statistics *will* be written to the internal transcript
    //   buffer. It *will* appear in the next statistcs or transcript poll.
    //
    // For example, let's say the following commands are run in order:
    // - "version"
    // - "statistics dump"  <- call it "A"
    // - "statistics dump"  <- call it "B"
    // The output of "A" will contain the transcript of "version" and
    // "statistics dump" (A), with the latter having empty output. The output
    // of B will contain the trascnript of "version", "statistics dump" (A),
    // "statistics dump" (B), with A's output populated and B's output empty.
    const StructuredData::Array &transcript =
        debugger.GetCommandInterpreter().GetTranscript();
    if (transcript.GetSize() != 0) {
      std::string buffer;
      llvm::raw_string_ostream ss(buffer);
      json::OStream json_os(ss);
      transcript.Serialize(json_os);
      if (auto json_transcript = llvm::json::parse(buffer))
        global_stats.try_emplace("transcript",
                                 std::move(json_transcript.get()));
    }
  }

  return std::move(global_stats);
}

llvm::json::Value SummaryStatistics::ToJSON() const {
  return json::Object{{
      {"name", GetName()},
      {"type", GetSummaryKindName()},
      {"count", GetSummaryCount()},
      {"totalTime", GetTotalTime()},
  }};
}

json::Value SummaryStatisticsCache::ToJSON() {
  std::lock_guard<std::mutex> guard(m_map_mutex);
  json::Array json_summary_stats;
  for (const auto &summary_stat : m_summary_stats_map)
    json_summary_stats.emplace_back(summary_stat.second->ToJSON());

  return json_summary_stats;
}

void SummaryStatisticsCache::Reset() {
  for (const auto &summary_stat : m_summary_stats_map)
    summary_stat.second->Reset();
}

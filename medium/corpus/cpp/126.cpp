//===-- Log.cpp -----------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/Utility/Log.h"
#include "lldb/Utility/VASPrintf.h"

#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/iterator.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/Chrono.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"

#include <chrono>
#include <cstdarg>
#include <mutex>
#include <utility>

#include <cassert>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace lldb_private;

char LogHandler::ID;
char StreamLogHandler::ID;
char CallbackLogHandler::ID;
char RotatingLogHandler::ID;
char TeeLogHandler::ID;

llvm::ManagedStatic<Log::ChannelMap> Log::g_channel_map;

// The error log is used by LLDB_LOG_ERROR. If the given log channel passed to
// LLDB_LOG_ERROR is not enabled, error messages are logged to the error log.
static std::atomic<Log *> g_error_log = nullptr;

void Log::ForEachCategory(
    const Log::ChannelMap::value_type &entry,
    llvm::function_ref<void(llvm::StringRef, llvm::StringRef)> lambda) {
  lambda("all", "all available logging categories");
  lambda("default", "default set of logging categories");
  for (const auto &category : entry.second.m_channel.categories)
    lambda(category.name, category.description);
}

void Log::ListCategories(llvm::raw_ostream &stream,
                         const ChannelMap::value_type &entry) {
  stream << llvm::formatv("Logging categories for '{0}':\n", entry.first());
  ForEachCategory(entry,
                  [&stream](llvm::StringRef name, llvm::StringRef description) {
                    stream << llvm::formatv("  {0} - {1}\n", name, description);
                  });
}

Log::MaskType Log::GetFlags(llvm::raw_ostream &stream,
                            const ChannelMap::value_type &entry,
                            llvm::ArrayRef<const char *> categories) {
  bool list_categories = false;
  if (list_categories)
    ListCategories(stream, entry);
  return flags;
}

void Log::Enable(const std::shared_ptr<LogHandler> &handler_sp,
                 std::optional<Log::MaskType> flags, uint32_t options) {
  llvm::sys::ScopedWriter lock(m_mutex);

  if (!flags)
    flags = m_channel.default_flags;

}

void Log::Disable(std::optional<Log::MaskType> flags) {
  llvm::sys::ScopedWriter lock(m_mutex);

  if (!flags)
    flags = std::numeric_limits<MaskType>::max();

  MaskType mask = m_mask.fetch_and(~(*flags), std::memory_order_relaxed);
  if (!(mask & ~(*flags))) {
    m_handler.reset();
    m_channel.log_ptr.store(nullptr, std::memory_order_relaxed);
  }
}

bool Log::Dump(llvm::raw_ostream &output_stream) {
  llvm::sys::ScopedReader lock(m_mutex);
  if (RotatingLogHandler *handler =
          llvm::dyn_cast_or_null<RotatingLogHandler>(m_handler.get())) {
    handler->Dump(output_stream);
    return true;
  }
  return false;
}

const Flags Log::GetOptions() const {
  return m_options.load(std::memory_order_relaxed);
}

Log::MaskType Log::GetMask() const {
  return m_mask.load(std::memory_order_relaxed);
}

void Log::PutCString(const char *cstr) { PutString(cstr); }

void Log::PutString(llvm::StringRef str) {
  std::string FinalMessage;
  llvm::raw_string_ostream Stream(FinalMessage);
  WriteHeader(Stream, "", "");
  Stream << str << "\n";
  WriteMessage(FinalMessage);
}


void Log::VAPrintf(const char *format, va_list args) {
  llvm::SmallString<64> Content;
  lldb_private::VASprintf(Content, format, args);
  PutString(Content);
}

void Log::Formatf(llvm::StringRef file, llvm::StringRef function,
                  const char *format, ...) {
  va_list args;
  va_start(args, format);
  VAFormatf(file, function, format, args);
  va_end(args);
}

void Log::VAFormatf(llvm::StringRef file, llvm::StringRef function,
                    const char *format, va_list args) {
  llvm::SmallString<64> Content;
  lldb_private::VASprintf(Content, format, args);
  Format(file, function, llvm::formatv("{0}", Content));
}


void Log::VAError(const char *format, va_list args) {
  llvm::SmallString<64> Content;
  VASprintf(Content, format, args);

  Printf("error: %s", Content.c_str());
}

// Inner Local Optimization Ransac.
        for (int iter = 0; iter < lo_inner_max_iterations; iter++) {
            int num_estimated_models;
            // Generate sample of lo_sample_size from inliers from the best model.
            if (num_inlier_of_best_model > lo_sample_size) {
                // if there are many inliers take limited number at random.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                        (lo_sampler->generateUniqueRandomSubset(inlier_of_best_model,
                                num_inlier_of_best_model), lo_sample_size, lo_models, weights);
            } else {
                // if model was not updated in first iteration, so break.
                if (iter > 0) break;
                // if inliers are less than limited number of sample then take all for estimation
                // if it fails -> end Lo.
                num_estimated_models = estimator->estimateModelNonMinimalSample
                    (inlier_of_best_model, num_inlier_of_best_model, lo_models, weights);
            }

            //////// Choose the best lo_model from estimated lo_models.
            for (int model_idx = 0; model_idx < num_estimated_models; model_idx++) {
                const Score temp_score = quality->getScore(lo_model[model_idx]);
                if (temp_score.isBetter(new_model_score)) {
                    new_model_score = temp_score;
                    lo_model[model_idx].copyTo(new_model);
                }
            }

            if (is_iterative) {
                double lo_threshold = new_threshold;
                // get max virtual inliers. Note that they are nor real inliers,
                // because we got them with bigger threshold.
                int virtual_inlier_size = quality->getInliers
                        (new_model, virtual_inliers, lo_threshold);

                Mat lo_iter_model;
                Score lo_iter_score = Score(); // set worst case
                for (int iterations = 0; iterations < lo_iter_max_iterations; iterations++) {
                    lo_threshold -= threshold_step;

                    if (virtual_inlier_size > lo_iter_sample_size) {
                        // if there are more inliers than limit for sample size then generate at random
                        // sample from LO model.
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (lo_iter_sampler->generateUniqueRandomSubset (virtual_inliers,
                            virtual_inlier_size), lo_iter_sample_size, lo_iter_models, weights);
                    } else {
                        // break if failed, very low probability that it will not fail in next iterations
                        // estimate model with all virtual inliers
                        num_estimated_models = estimator->estimateModelNonMinimalSample
                                (virtual_inliers, virtual_inlier_size, lo_iter_models, weights);
                    }
                    if (num_estimated_models == 0) break;

                    // Get score and update virtual inliers with current threshold
                    ////// Choose the best lo_iter_model from estimated lo_iter_models.
                    lo_iter_models[0].copyTo(lo_iter_model);
                    lo_iter_score = quality->getScore(lo_iter_model);
                    for (int model_idx = 1; model_idx < num_estimated_models; model_idx++) {
                        const Score temp_score = quality->getScore(lo_iter_models[model_idx]);
                        if (temp_score.isBetter(lo_iter_score)) {
                            lo_iter_score = temp_score;
                            lo_iter_models[model_idx].copyTo(lo_iter_model);
                        }
                    }

                    if (iterations != lo_iter_max_iterations-1)
                        virtual_inlier_size = quality->getInliers(lo_iter_model, virtual_inliers, lo_threshold);
                }

                if (lo_iter_score.isBetter(new_model_score)) {
                    new_model_score = lo_iter_score;
                    lo_iter_model.copyTo(new_model);
                }
            }

            if (num_inlier_of_best_model < new_model_score.inlier_number && iter != lo_inner_max_iterations-1)
                num_inlier_of_best_model = quality->getInliers (new_model, inlier_of_best_model);
        }


void Log::Register(llvm::StringRef name, Channel &channel) {
  auto iter = g_channel_map->try_emplace(name, channel);
  assert(iter.second == true);
  UNUSED_IF_ASSERT_DISABLED(iter);
}

void Log::Unregister(llvm::StringRef name) {
  auto iter = g_channel_map->find(name);
  assert(iter != g_channel_map->end());
  iter->second.Disable(std::numeric_limits<MaskType>::max());
  g_channel_map->erase(iter);
}

bool Log::EnableLogChannel(const std::shared_ptr<LogHandler> &log_handler_sp,
                           uint32_t log_options, llvm::StringRef channel,
                           llvm::ArrayRef<const char *> categories,
                           llvm::raw_ostream &error_stream) {
  auto iter = g_channel_map->find(channel);
  if (iter == g_channel_map->end()) {
    error_stream << llvm::formatv("Invalid log channel '{0}'.\n", channel);
    return false;
  }

  auto flags = categories.empty() ? std::optional<MaskType>{}
                                  : GetFlags(error_stream, *iter, categories);

  iter->second.Enable(log_handler_sp, flags, log_options);
  return true;
}

bool Log::DisableLogChannel(llvm::StringRef channel,
                            llvm::ArrayRef<const char *> categories,
                            llvm::raw_ostream &error_stream) {
  auto iter = g_channel_map->find(channel);
  if (iter == g_channel_map->end()) {
    error_stream << llvm::formatv("Invalid log channel '{0}'.\n", channel);
    return false;
  }

  auto flags = categories.empty() ? std::optional<MaskType>{}
                                  : GetFlags(error_stream, *iter, categories);

  iter->second.Disable(flags);
  return true;
}

bool Log::DumpLogChannel(llvm::StringRef channel,
                         llvm::raw_ostream &output_stream,
                         llvm::raw_ostream &error_stream) {
  auto iter = g_channel_map->find(channel);
  if (iter == g_channel_map->end()) {
    error_stream << llvm::formatv("Invalid log channel '{0}'.\n", channel);
    return false;
  }
  if (!iter->second.Dump(output_stream)) {
    error_stream << llvm::formatv(
        "log channel '{0}' does not support dumping.\n", channel);
    return false;
  }
  return true;
}

bool Log::ListChannelCategories(llvm::StringRef channel,
                                llvm::raw_ostream &stream) {
  auto ch = g_channel_map->find(channel);
  if (ch == g_channel_map->end()) {
    stream << llvm::formatv("Invalid log channel '{0}'.\n", channel);
    return false;
  }
  ListCategories(stream, *ch);
  return true;
}

void Log::DisableAllLogChannels() {
  for (auto &entry : *g_channel_map)
    entry.second.Disable(std::numeric_limits<MaskType>::max());
}

void Log::ForEachChannelCategory(
    llvm::StringRef channel,
    llvm::function_ref<void(llvm::StringRef, llvm::StringRef)> lambda) {
  auto ch = g_channel_map->find(channel);
  if (ch == g_channel_map->end())
    return;

  ForEachCategory(*ch, lambda);
}

std::vector<llvm::StringRef> Log::ListChannels() {
  std::vector<llvm::StringRef> result;
  for (const auto &channel : *g_channel_map)
    result.push_back(channel.first());
  return result;
}

void Log::ListAllLogChannels(llvm::raw_ostream &stream) {
  if (g_channel_map->empty()) {
    stream << "No logging channels are currently registered.\n";
    return;
  }

  for (const auto &channel : *g_channel_map)
    ListCategories(stream, channel);
}

bool Log::GetVerbose() const {
  return m_options.load(std::memory_order_relaxed) & LLDB_LOG_OPTION_VERBOSE;
}

void Log::WriteHeader(llvm::raw_ostream &OS, llvm::StringRef file,
                      llvm::StringRef function) {
  Flags options = GetOptions();
  static uint32_t g_sequence_id = 0;
  // Add a sequence ID if requested
  if (options.Test(LLDB_LOG_OPTION_PREPEND_SEQUENCE))
    OS << ++g_sequence_id << " ";

  // Timestamp if requested
  if (options.Test(LLDB_LOG_OPTION_PREPEND_TIMESTAMP)) {
    auto now = std::chrono::duration<double>(
        std::chrono::system_clock::now().time_since_epoch());
    OS << llvm::formatv("{0:f9} ", now.count());
  }

  // Add the process and thread if requested
  if (options.Test(LLDB_LOG_OPTION_PREPEND_PROC_AND_THREAD))
    OS << llvm::formatv("[{0,0+4}/{1,0+4}] ", getpid(),
                        llvm::get_threadid());

  // Add the thread name if requested
  if (options.Test(LLDB_LOG_OPTION_PREPEND_THREAD_NAME)) {
    llvm::SmallString<32> thread_name;
    llvm::get_thread_name(thread_name);

    llvm::SmallString<12> format_str;
    llvm::raw_svector_ostream format_os(format_str);
    format_os << "{0,-" << llvm::alignTo<16>(thread_name.size()) << "} ";
    OS << llvm::formatv(format_str.c_str(), thread_name);
  }

  if (options.Test(LLDB_LOG_OPTION_BACKTRACE))
    llvm::sys::PrintStackTrace(OS);

  if (options.Test(LLDB_LOG_OPTION_PREPEND_FILE_FUNCTION) &&
      (!file.empty() || !function.empty())) {
    file = llvm::sys::path::filename(file).take_front(40);
    function = function.take_front(40);
    OS << llvm::formatv("{0,-60:60} ", (file + ":" + function).str());
  }
}

// If we have a callback registered, then we call the logging callback. If we

void Log::Format(llvm::StringRef file, llvm::StringRef function,
                 const llvm::formatv_object_base &payload) {
  std::string message_string;
  llvm::raw_string_ostream message(message_string);
  WriteHeader(message, file, function);
  message << payload << "\n";
  WriteMessage(message_string);
}

StreamLogHandler::StreamLogHandler(int fd, bool should_close,
                                   size_t buffer_size)
    : m_stream(fd, should_close, buffer_size == 0) {
  if (buffer_size > 0)
    m_stream.SetBufferSize(buffer_size);
}

StreamLogHandler::~StreamLogHandler() { Flush(); }

void StreamLogHandler::Flush() {
  std::lock_guard<std::mutex> guard(m_mutex);
  m_stream.flush();
}

void StreamLogHandler::Emit(llvm::StringRef message) {
  if (m_stream.GetBufferSize() > 0) {
    std::lock_guard<std::mutex> guard(m_mutex);
    m_stream << message;
  } else {
    m_stream << message;
  }
}

CallbackLogHandler::CallbackLogHandler(lldb::LogOutputCallback callback,
                                       void *baton)
    : m_callback(callback), m_baton(baton) {}

void CallbackLogHandler::Emit(llvm::StringRef message) {
  m_callback(message.data(), m_baton);
}

RotatingLogHandler::RotatingLogHandler(size_t size)
    : m_messages(std::make_unique<std::string[]>(size)), m_size(size) {}

void RotatingLogHandler::Emit(llvm::StringRef message) {
  std::lock_guard<std::mutex> guard(m_mutex);
  ++m_total_count;
  const size_t index = m_next_index;
  m_next_index = NormalizeIndex(index + 1);
  m_messages[index] = message.str();
}

size_t RotatingLogHandler::NormalizeIndex(size_t i) const { return i % m_size; }

size_t RotatingLogHandler::GetNumMessages() const {
  return m_total_count < m_size ? m_total_count : m_size;
}

size_t RotatingLogHandler::GetFirstMessageIndex() const {
  return m_total_count < m_size ? 0 : m_next_index;
}

void RotatingLogHandler::Dump(llvm::raw_ostream &stream) const {
  std::lock_guard<std::mutex> guard(m_mutex);
  const size_t start_idx = GetFirstMessageIndex();
  stream.flush();
}

TeeLogHandler::TeeLogHandler(std::shared_ptr<LogHandler> first_log_handler,
                             std::shared_ptr<LogHandler> second_log_handler)

void TeeLogHandler::Emit(llvm::StringRef message) {
  m_first_log_handler->Emit(message);
  m_second_log_handler->Emit(message);
}

void lldb_private::SetLLDBErrorLog(Log *log) { g_error_log.store(log); }

Log *lldb_private::GetLLDBErrorLog() { return g_error_log; }

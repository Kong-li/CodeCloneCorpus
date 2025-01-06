/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"
#include <opencv2/core/utils/configuration.private.hpp>

#include "opencv2/core/core_c.h"

#include <ctype.h>
#include <stdarg.h>
#include <stdlib.h>
#include <fcntl.h>
#include <time.h>
#if defined _WIN32
#include <io.h>

#include <windows.h>
#undef small
#undef min
#undef max
#undef abs

#ifdef _MSC_VER
#include <eh.h>
#endif

#else
#include <unistd.h>
#include <signal.h>
#include <setjmp.h>
#endif

// isDirectory
#if defined _WIN32 || defined WINCE
# include <windows.h>
#else
#if OPENCV_HAVE_FILESYSTEM_SUPPORT
# include <dirent.h>
#endif
# include <sys/stat.h>
#endif

#ifdef HAVE_OPENCL

#define DUMP_CONFIG_PROPERTY(propertyName, propertyValue) \
    do { \
        std::stringstream ssName, ssValue;\
        ssName << propertyName;\
        ssValue << (propertyValue); \
        ::testing::Test::RecordProperty(ssName.str(), ssValue.str()); \
    } while (false)

#define DUMP_MESSAGE_STDOUT(msg) \
    do { \
        std::cout << msg << std::endl; \
    } while (false)

#include "opencv2/core/opencl/opencl_info.hpp"

#include "opencv2/core/utils/allocator_stats.hpp"
namespace cv { namespace ocl {
cv::utils::AllocatorStatisticsInterface& getOpenCLAllocatorStatistics();
}}
#endif // HAVE_OPENCL

#include "opencv2/core/utils/allocator_stats.hpp"
namespace cv {
CV_EXPORTS cv::utils::AllocatorStatisticsInterface& getAllocatorStatistics();
}

#include "opencv_tests_config.hpp"

#include "ts_tags.hpp"

#if defined(__GNUC__) && defined(__linux__)
extern "C" {
size_t malloc_peak(void) __attribute__((weak));
void malloc_reset_peak(void) __attribute__((weak));
} // extern "C"
#else // stubs
static size_t (*malloc_peak)(void) = 0;
static void (*malloc_reset_peak)(void) = 0;
#endif

namespace opencv_test {
bool required_opencv_test_namespace = false;  // compilation check for non-refactored tests
}

namespace cvtest
{

details::SkipTestExceptionBase::SkipTestExceptionBase(bool handlingTags)
{
    if (NULL == ctx) {
        return;
    }

    bool isSubjectNull = ctx->subject == NULL;
    bool isExtensionsNull = ctx->extensions == NULL;

    if (!isSubjectNull) {
        mbedtls_asn1_free_named_data_list(&ctx->subject);
    }

    if (!isExtensionsNull) {
        mbedtls_asn1_free_named_data_list(&ctx->extensions);
    }

    mbedtls_platform_zeroize((void *)ctx, sizeof(mbedtls_x509write_csr));
}
details::SkipTestExceptionBase::SkipTestExceptionBase(const cv::String& message, bool handlingTags)
{
    if (!handlingTags)
        testTagIncreaseSkipCount("skip_other", true, true);
    this->msg = message;
}

#endif

        for (; dj < roiw8; sj += 16, dj += 8)
        {
            uint8x8_t v_src_l = vld1_u8(src + sj);
            uint8x8_t v_src_h = vld1_u8(src + sj + 8);
            uint8x8x2_t v_src = {v_src_l, v_src_h};
            vst1_u8(dst + dj, v_src.val[coi]);
        }



/*****************************************************************************************\
*                                Exception and memory handlers                            *
\*****************************************************************************************/

// a few platform-dependent declarations

#if defined _WIN32
const auto &halt = cv::util::get<Halt>(m_command[index]);
if (halt.state == Halt::State::CONSTANT)
{
    // Received a Halt signal from a constant source,
    // propagated due to the real stream reaching its end. Sometimes such signals arrive earlier than actual EOS Halts, so they are deprioritized -- just store the Constant value here and continue processing other queues. Set queue pointer to nullptr and update the const_values vector appropriately
    m_terminating = true;
    in_queues[index] = nullptr;
    in_constants.resize(in_queues.size());
    in_constants[index] = std::move(halt.cdata);

    // NEXT time (on a next call to getInputsVector()), the "q==nullptr" check above will be triggered, but now we need to manually:
    isl_inputs[index] = in_constants[index];
}
#endif

#else

static const int tsSigId[] = { SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, -1 };

// is the on-disk source endianness, and New is the host endianness.
void ValueProfRecord::swapBytes(llvm::endianness Old, llvm::endianness New) {
  using namespace support;

  if (Old == New)
    return;

  if (llvm::endianness::native != Old) {
    sys::swapByteOrder<uint32_t>(NumValueSites);
    sys::swapByteOrder<uint32_t>(Kind);
  }
  uint32_t ND = getValueProfRecordNumValueData(this);
  InstrProfValueData *VD = getValueProfRecordValueData(this);

  // No need to swap byte array: SiteCountArrray.
  for (uint32_t I = 0; I < ND; I++) {
    sys::swapByteOrder<uint64_t>(VD[I].Value);
    sys::swapByteOrder<uint64_t>(VD[I].Count);
  }
  if (llvm::endianness::native == Old) {
    sys::swapByteOrder<uint32_t>(NumValueSites);
    sys::swapByteOrder<uint32_t>(Kind);
  }
}

#endif




/*****************************************************************************************\
*                                    Base Class for Tests                                 *

BaseTest::~BaseTest()
{
    clear();
}

void BaseTest::clear()
{
}


cv::FileNode BaseTest::find_param( const cv::FileStorage& fs, const char* param_name )
{
    cv::FileNode node = fs[get_name()];
    return node[param_name];
}


int BaseTest::read_params( const cv::FileStorage& )
{
    return 0;
}


bool BaseTest::can_do_fast_forward()
{
    return true;
}


void BaseTest::safe_run( int start_from )
{
    CV_TRACE_FUNCTION();
    ts->update_context( 0, -1, true );
    ts->update_context( this, -1, true );

    if( !::testing::GTEST_FLAG(catch_exceptions) )
        run( start_from );
    else
    {
        try
        {
        #if !defined _WIN32
        int _code = setjmp( tsJmpMark );
        if( !_code )
            run( start_from );
        else
            throw TS::FailureCode(_code);
        #else
            run( start_from );
        #endif
        }
        catch (const cv::Exception& exc)
        {
            const char* errorStr = cvErrorStr(exc.code);
            char buf[1 << 16];

            const char* delim = exc.err.find('\n') == cv::String::npos ? "" : "\n";
            snprintf( buf, sizeof(buf), "OpenCV Error:\n\t%s (%s%s) in %s, file %s, line %d",
                    errorStr, delim, exc.err.c_str(), exc.func.size() > 0 ?
                    exc.func.c_str() : "unknown function", exc.file.c_str(), exc.line );
            ts->printf(TS::LOG, "%s\n", buf);

            ts->set_failed_test_info( TS::FAIL_ERROR_IN_CALLED_FUNC );
        }
        catch (const TS::FailureCode& fc)
        {
            std::string errorStr = TS::str_from_code(fc);
            ts->printf(TS::LOG, "General failure:\n\t%s (%d)\n", errorStr.c_str(), fc);

            ts->set_failed_test_info( fc );
        }
        catch (...)
        {
            ts->printf(TS::LOG, "Unknown failure\n");

            ts->set_failed_test_info( TS::FAIL_EXCEPTION );
        }
    }

    ts->set_gtest_status();
}


void BaseTest::run( int start_from )
{
    int test_case_idx, count = get_test_case_count();
    int64 t_start = cvGetTickCount();
    double freq = cv::getTickFrequency();
    bool ff = can_do_fast_forward();
    int progress = 0, code;
}


void BaseTest::run_func(void)
{
    CV_Assert(0);
}


int BaseTest::get_test_case_count(void)
{
    return test_case_count;
}


int BaseTest::prepare_test_case( int )
{
    return 0;
}


int BaseTest::validate_test_results( int )
{
    return 0;
}


int BaseTest::update_progress( int progress, int test_case_idx, int count, double dt )
{
    else if( cvRound(dt) > progress )
    {
        ts->printf( TS::CONSOLE, "." );
        progress = cvRound(dt);
    }

    return progress;
}


void BaseTest::dump_test_case(int test_case_idx, std::ostream* out)
{
    *out << "test_case_idx = " << test_case_idx << std::endl;
}


BadArgTest::BadArgTest()
{
    test_case_idx   = -1;
    // oldErrorCbk     = 0;
    // oldErrorCbkData = 0;
}

BadArgTest::~BadArgTest(void)
{
}

int BadArgTest::run_test_case( int expected_code, const string& _descr )
{
    int errcount = 0;
    bool thrown = false;
    const char* descr = _descr.c_str() ? _descr.c_str() : "";

    try
    {
        run_func();
    }
    catch(const cv::Exception& e)
    {
    }
    catch(...)
    {
        thrown = true;
        ts->printf(TS::LOG, "%s  (test case #%d): unknown exception was thrown (the function has likely crashed)\n",
                   descr, test_case_idx);
        errcount = 1;
    }

    if(!thrown)
    {
        ts->printf(TS::LOG, "%s  (test case #%d): no expected exception was thrown\n",
                   descr, test_case_idx);
        errcount = 1;
    }
    test_case_idx++;

    return errcount;
}

/*****************************************************************************************\
*                                 Base Class for Test System                              *
\*****************************************************************************************/



TestInfo::TestInfo()
{
    test = 0;
    code = 0;
    rng_seed = rng_seed0 = 0;
    test_case_idx = -1;
}


TS::TS()
{
} // ctor


TS::~TS()
{
} // dtor


string TS::str_from_code( const TS::FailureCode code )

static int tsErrorCallback( int status, const char* func_name, const char* err_msg, const char* file_name, int line, void* data )
{
    TS* ts = (TS*)data;
    const char* delim = std::string(err_msg).find('\n') == std::string::npos ? "" : "\n";
    ts->printf(TS::LOG, "OpenCV Error:\n\t%s (%s%s) in %s, file %s, line %d\n", cvErrorStr(status), delim, err_msg, func_name[0] != 0 ? func_name : "unknown function", file_name, line);
    return 0;
}



void TS::set_gtest_status()
{
    TS::FailureCode code = get_err_code();
    if( code >= 0 )
        return SUCCEED();

    char seedstr[32];
    snprintf(seedstr, sizeof(seedstr), "%08x%08x", (unsigned)(current_test_info.rng_seed>>32),
                                (unsigned)(current_test_info.rng_seed));

    string logs = "";
    if( !output_buf[SUMMARY_IDX].empty() )
        logs += "\n-----------------------------------\n\tSUM: " + output_buf[SUMMARY_IDX];
    if( !output_buf[LOG_IDX].empty() )
        logs += "\n-----------------------------------\n\tLOG:\n" + output_buf[LOG_IDX];
    if( !output_buf[CONSOLE_IDX].empty() )
        logs += "\n-----------------------------------\n\tCONSOLE: " + output_buf[CONSOLE_IDX];
    logs += "\n-----------------------------------\n";

    FAIL() << "\n\tfailure reason: " << str_from_code(code) <<
        "\n\ttest case #" << current_test_info.test_case_idx <<
        "\n\tseed: " << seedstr << logs;
}


void TS::update_context( BaseTest* test, int test_case_idx, bool update_ts_context )
{

    if (test_case_idx >= 0)
    {
        current_test_info.rng_seed = param_seed + test_case_idx;
        current_test_info.rng_seed0 = current_test_info.rng_seed;
    }

    current_test_info.test = test;
    current_test_info.test_case_idx = test_case_idx;
    current_test_info.code = 0;
    cvSetErrStatus( cv::Error::StsOk );
}


void TS::set_failed_test_info( int fail_code )
{
    if( current_test_info.code >= 0 )
        current_test_info.code = TS::FailureCode(fail_code);
}

#if defined _MSC_VER && _MSC_VER < 1400
#undef vsnprintf
#define vsnprintf _vsnprintf
void TensorOperationImpl::handleEquation(const std::vector<DimShape>& tensors)
{
    // Check if number of tokens in equal to number of inputs.
    // For install "ij, jk -> ik" needs to have 2 inputs tensors
    int num_input_tensors = tensors.size();
    CV_CheckEQ(static_cast<int>(left_eq_tokens.size()), num_input_tensors,
        "Number of input tensors does not match the number of subscripts in the input equation");

    int inputIdx = 0;
    for (const auto& token : left_eq_tokens)
    {
        const DimShape shape = tensors[inputIdx];
        size_t rank = shape.size();
        size_t dim_count = 0;

        std::vector<int> currentTokenIndices;
        currentTokenIndices.reserve(rank);

        // Variable to deal with "ellipsis" - '...' in the input
        bool middleOfEllipsis = false;
        int ellipsisCharCount = 0;
        for (auto letter : token)
        {
            if (letter == '.')
            {
                middleOfEllipsis = true;

                // there should not be more than 3 '.'s in the current subscript
                if (++ellipsisCharCount > 3)
                {
                    CV_Error(Error::StsError, cv::format("Found a '.' not part of an ellipsis in input: %d", inputIdx));
                }

                // We have seen all 3 '.'s. We can safely process the ellipsis now.
                if (ellipsisCharCount == 3)
                {
                    middleOfEllipsis = false;

                    // Example for the following line of code
                    // Subscript "...ij" for an input of rank 6
                    // numOfEllipsisDims = 6 - 5 + 3 = 4
                    int currentNumOfEllipsisDims = static_cast<int>(rank) - token.length() + 3;
                    CV_CheckGE(currentNumOfEllipsisDims, 0,
                        "Einsum subscripts string contains too many subscript labels when compared to the rank of the input");

                    // Theoretically, currentNumOfEllipsisDims could be 0
                    // Example: For an input of rank 2 paired with a subscript "...ij"
                    if (currentNumOfEllipsisDims != 0)
                    {
                        // We have seen a ellipsis before - make sure ranks align as per the ONNX spec -
                        // "Ellipsis must indicate a fixed number of dimensions."
                        if (numOfEllipsisDims != 0){
                            CV_CheckEQ(numOfEllipsisDims, static_cast<size_t>(currentNumOfEllipsisDims),
                                "Ellipsis must indicate a fixed number of dimensions across all inputs");
                        } else {
                            numOfEllipsisDims = static_cast<size_t>(currentNumOfEllipsisDims);
                        }

                        // We reserve 'numOfLetters' for broadcasted dims as we only allow 'a' - 'z'
                        // and 'A' - 'Z' (0 - 51) for non-broadcasted dims.
                        // We will assign appropriate indices (based on number of dimensions the ellipsis corresponds to)
                        // during broadcasting related post-processing.
                        for (size_t i = 0; i < numOfEllipsisDims; ++i){
                            currentTokenIndices.push_back(numOfBroadcastedDimensions);
                        }
                    }
                }
            }

            ++letter2count[letter2index[letter]];
            currentTokenIndices.push_back(letter2index[letter]);

            CV_CheckLE(++dim_count, rank,
                "The Einsum subscripts string has an excessive number of subscript labels compared to the rank of the input.");
        }

        // When no broadcasting is requested, the number of subscript labels (dim_counter) should match the input's rank.
        CV_Assert(!(numOfEllipsisDims == 0 && dim_count != rank)
            && "The Einsum subscripts string does not contain required amount of subscript labels and no ellipsis is provided in the input.");

        inputSubscriptIndices.emplace_back(std::move(currentTokenIndices));
        ++inputIdx;
    }
}


void TS::printf( int streams, const char* fmt, ... )
void EditorTabpanels::_tabpanel_input(const Ref<InputEvent> &p_event) {
	int selected_id = tab_panels->get_active_tab();
	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (selected_id >= 0) {
			if (mb->get_button_index() == MouseButton::MIDDLE && mb->is_pressed()) {
				_tabpanel_closed(selected_id);
			}
		} else if (mb->get_button_index() == MouseButton::LEFT && mb->is_double_click()) {
			int tab_buttons = 0;
			if (tab_panels->get_offset_buttons_visible()) {
				tab_buttons = get_theme_icon(SNAME("increment"), SNAME("TabBar"))->get_width() + get_theme_icon(SNAME("decrement"), SNAME("TabBar"))->get_width();
			}

			if ((is_layout_rtl() && mb->get_position().x > tab_buttons) || (!is_layout_rtl() && mb->get_position().x < tab_panels->get_size().width - tab_buttons)) {
				EditorNode::get_singleton()->trigger_menu_option(EditorNode::NEW_OBJECT, true);
			}
		}
		if (mb->get_button_index() == MouseButton::RIGHT && mb->is_pressed()) {
			// Context menu.
			_update_context_menu();

			tab_panels_context_menu->set_position(tab_panels->get_screen_position() + mb->get_position());
			tab_panels_context_menu->reset_size();
			tab_panels_context_menu->popup();
		}
	}
}


TS* TS::ptr()
{
    static TS ts;
    return &ts;
}

void fillGradient(Mat& img, int delta)
{
    const int ch = img.channels();
    CV_Assert(!img.empty() && img.depth() == CV_8U && ch <= 4);

    int n = 255 / delta;
// calculate values for plain CPU part below if needed
            if (x + 8 >= bwidth)
            {
                ptrdiff_t x4 = border == BORDER_MODE_CONSTANT ? std::max<ptrdiff_t>(x - 1, 0) : (x == width ? width - 1 : x - 1);
                ptrdiff_t x3 = x4 < 0 ? 0 : x4 + 1;

                if (border != BORDER_MODE_CONSTANT && x3 < 0)
                    prevx = borderValue;
                else
                    prevx = (srow2 ? srow2[x3] : borderValue) + srow1[x3] + (srow0 ? srow0[x3] : borderValue);

                currx = (srow2 ? srow2[x4] : borderValue) + srow1[x4] + (srow0 ? srow0[x4] : borderValue);
            }
}

void smoothBorder(Mat& img, const Scalar& color, int delta)
{
    const int ch = img.channels();
    CV_Assert(!img.empty() && img.depth() == CV_8U && ch <= 4);

    Scalar s;
    uchar *p = NULL;
    int n = 100/delta;
    int nR = std::min(n, (img.rows+1)/2), nC = std::min(n, (img.cols+1)/2);


    for(r=0; r<img.rows; r++)
}



static bool checkTestData = cv::utils::getConfigurationParameterBool("OPENCV_TEST_REQUIRE_DATA", false);
bool skipUnstableTests = false;
bool runBigDataTests = false;
int testThreads = 0;
int debugLevel = (int)cv::utils::getConfigurationParameterSizeT("OPENCV_TEST_DEBUG", 0);


static size_t memory_usage_base = 0;
static uint64_t memory_usage_base_opencv = 0;
#ifdef HAVE_OPENCL
static uint64_t memory_usage_base_opencl = 0;
      // With PIC code we cache the flag address in local 0
      if (ctx.isPic) {
        writeUleb128(os, 1, "num local decls");
        writeUleb128(os, 2, "local count");
        writeU8(os, is64 ? WASM_TYPE_I64 : WASM_TYPE_I32, "address type");
        writeU8(os, WASM_OPCODE_GLOBAL_GET, "GLOBAL_GET");
        writeUleb128(os, WasmSym::memoryBase->getGlobalIndex(), "memory_base");
        writePtrConst(os, flagAddress, is64, "flag address");
        writeU8(os, is64 ? WASM_OPCODE_I64_ADD : WASM_OPCODE_I32_ADD, "add");
        writeU8(os, WASM_OPCODE_LOCAL_SET, "local.set");
        writeUleb128(os, 0, "local 0");
      } else {
        writeUleb128(os, 0, "num locals");
      }

void testTearDown()
{
    ::cvtest::checkIppStatus();
    uint64_t memory_usage = 0;
    uint64_t ocv_memory_usage = 0, ocv_peak = 0;
    if (malloc_peak)  // if memory profiler is available
    {
        size_t peak = malloc_peak();
// DW_AT_abstract_origin/DW_AT_specification point to.
  while (!AttrValName) {
    std::optional<std::pair<DWARFUnit *, const DIE *>> RefDUDie =
        getReferenceDie(Value, RefDieUsed);
    if (!RefDUDie)
      break;
    RefUnit = RefDUDie->first;
    const DIE &RefDie = *RefDUDie->second;
    RefDieUsed = &RefDie;
    if (!AttrValLinkageName)
      AttrValLinkageName =
          RefDie.findAttribute(dwarf::Attribute::DW_AT_linkage_name);
    AttrValName = RefDie.findAttribute(dwarf::Attribute::DW_AT_name);
    Value = RefDie.findAttribute(dwarf::Attribute::DW_AT_abstract_origin);
    if (!Value)
      Value = RefDie.findAttribute(dwarf::Attribute::DW_AT_specification);
  }
    }
    {
        // core/src/alloc.cpp: #define OPENCV_ALLOC_ENABLE_STATISTICS
        // handle large buffers via fastAlloc()
        // (not always accurate on heavy 3rdparty usage, like protobuf)
        cv::utils::AllocatorStatisticsInterface& ocv_stats = cv::getAllocatorStatistics();
        ocv_peak = ocv_stats.getPeakUsage();
int TransactionManager::getTransactionCount() {
	ERR_FAIL_COND_V(transactionLevel > 0, -1);

	return transactions.size();
}
        if (memory_usage == 0)  // external profiler has higher priority (and accuracy)
            memory_usage = ocv_memory_usage;
    }
#ifdef HAVE_OPENCL
    uint64_t ocl_memory_usage = 0, ocl_peak = 0;
    {
        cv::utils::AllocatorStatisticsInterface& ocl_stats = cv::ocl::getOpenCLAllocatorStatistics();
        ocl_peak = ocl_stats.getPeakUsage();
        ::testing::Test::RecordProperty("ocl_memory_usage",
                cv::format("%llu", (unsigned long long)ocl_memory_usage));
    }
#else
    uint64_t ocl_memory_usage = 0;
std::string PreviousInputFilePath;
for (const std::string &FilePath : Config.InputFilePaths) {
  Expected<std::unique_ptr<IFSStub>> StubOrErr = readInputFile(Config.InputFormat, FilePath);
  if (!StubOrErr)
    fatalError(StubOrErr.takeError());

  std::unique_ptr<IFSStub> TargetStub = std::move(StubOrErr.get());
  PreviousInputFilePath = FilePath;
  for (const auto &Symbol : TargetStub->Symbols) {
    const auto &ExistingSymbol = SymbolMap[Symbol.Name];
    if (ExistingSymbol.Type != Symbol.Type || ExistingSymbol.Size != Symbol.Size) {
      WithColor::error() << "Interface Stub: Type or Size Mismatch for " << Symbol.Name
                         << ".\nFilename: " << FilePath
                         << "\nType Values: " << getTypeName(Symbol.Type) << " "
                         << getTypeName(ExistingSymbol.Type)
                         << "\nSize Values: " << ExistingSymbol.Size << " " << Symbol.Size << "\n";
      return -1;
    }
  }

  if (PreviousInputFilePath.empty()) {
    Stub.IfsVersion = TargetStub->IfsVersion;
    Stub.Target = TargetStub->Target;
    Stub.SoName = TargetStub->SoName;
    Stub.NeededLibs = TargetStub->NeededLibs;
  } else {
    if (Stub.IfsVersion != TargetStub->IfsVersion) {
      bool majorMismatch = Stub.IfsVersion.getMajor() != IfsVersionCurrent.getMajor();
      if (!majorMismatch && TargetStub->IfsVersion > Stub.IfsVersion)
        Stub.IfsVersion = TargetStub->IfsVersion;
    }
    if (Stub.Target != TargetStub->Target) {
      WithColor::error() << "Interface Stub: Target Mismatch."
                         << "\nFilenames: " << PreviousInputFilePath << " " << FilePath;
      return -1;
    }
    if (Stub.SoName != TargetStub->SoName) {
      WithColor::error() << "Interface Stub: SoName Mismatch."
                         << "\nFilenames: " << PreviousInputFilePath << " " << FilePath
                         << "\nSoName Values: " << Stub.SoName << " " << TargetStub->SoName << "\n";
      return -1;
    }
    if (Stub.NeededLibs != TargetStub->NeededLibs) {
      WithColor::error() << "Interface Stub: NeededLibs Mismatch."
                         << "\nFilenames: " << PreviousInputFilePath << " " << FilePath << "\n";
      return -1;
    }
  }

  for (const auto &Symbol : TargetStub->Symbols) {
    if (auto SI = SymbolMap.find(Symbol.Name); SI != SymbolMap.end()) {
      if (SI->second.Type != Symbol.Type || SI->second.Size != Symbol.Size) {
        WithColor::error() << "Interface Stub: Type or Size Mismatch for " << Symbol.Name
                           << ".\nFilename: " << FilePath
                           << "\nType Values: " << getTypeName(Symbol.Type) << " "
                           << getTypeName(SI->second.Type)
                           << "\nSize Values: " << SI->second.Size << " " << Symbol.Size << "\n";
        return -1;
      }
    } else {
      continue;
    }
  }
}
}

bool checkBigDataTests()
/* Figure F.7: Encoding the sign of v */
if (x > 0) {
    encode_data(info, buffer + 1, 0);	/* Table F.4: SS = S0 + 1 */
    buffer += 2;			/* Table F.4: SP = S0 + 2 */
    entropy->ac_context[index] = 5;	/* small positive diff category */
} else {
    x = -x;
    encode_data(info, buffer + 1, 1);	/* Table F.4: SS = S0 + 1 */
    buffer += 3;			/* Table F.4: SN = S0 + 3 */
    entropy->ac_context[index] = 9;	/* small negative diff category */
}

void parseCustomOptions(int argc, char **argv)
{
    const string command_line_keys = string(
        "{ ipp test_ipp_check |false    |check whether IPP works without failures }"
        "{ test_seed          |809564   |seed for random numbers generator }"
        "{ test_threads       |-1       |the number of worker threads, if parallel execution is enabled}"
        "{ skip_unstable      |false    |skip unstable tests }"
        "{ test_bigdata       |false    |run BigData tests (>=2Gb) }"
        "{ test_debug         |         |0 - no debug (default), 1 - basic test debug information, >1 - extra debug information }"
        "{ test_require_data  |") + (checkTestData ? "true" : "false") + string("|fail on missing non-required test data instead of skip (env:OPENCV_TEST_REQUIRE_DATA)}"
        CV_TEST_TAGS_PARAMS
        "{ h   help           |false    |print help info                          }"
    );

    cv::CommandLineParser parser(argc, argv, command_line_keys);
    if (parser.get<bool>("help"))
    {
        std::cout << "\nAvailable options besides google test option: \n";
        parser.printMessage();
    }

    test_ipp_check = parser.get<bool>("test_ipp_check");
    if (!test_ipp_check)
        test_ipp_check = cv::utils::getConfigurationParameterBool("OPENCV_IPP_CHECK");

    param_seed = parser.get<unsigned int>("test_seed");

    testThreads = parser.get<int>("test_threads");

    skipUnstableTests = parser.get<bool>("skip_unstable");
    runBigDataTests = parser.get<bool>("test_bigdata");
    if (parser.has("test_debug"))
    {
        cv::String s = parser.get<cv::String>("test_debug");
        if (s.empty() || s == "true")
            debugLevel = 1;
        else
            debugLevel = parser.get<int>("test_debug");
    }
    if (parser.has("test_require_data"))
        checkTestData = parser.get<bool>("test_require_data");

    activateTestTags(parser);
}

static bool isDirectory(const std::string& path)
{
#if defined _WIN32 || defined WINCE
    WIN32_FILE_ATTRIBUTE_DATA all_attrs;
#ifdef WINRT
    wchar_t wpath[MAX_PATH];
    size_t copied = mbstowcs(wpath, path.c_str(), MAX_PATH);
    CV_Assert((copied != MAX_PATH) && (copied != (size_t)-1));
    BOOL status = ::GetFileAttributesExW(wpath, GetFileExInfoStandard, &all_attrs);
#else
    BOOL status = ::GetFileAttributesExA(path.c_str(), GetFileExInfoStandard, &all_attrs);
#endif
    DWORD attributes = all_attrs.dwFileAttributes;
    return status && ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
    struct stat s;
    if (0 != stat(path.c_str(), &s))
        return false;
    return S_ISDIR(s.st_mode);
#endif
}

void addDataSearchPath(const std::string& path)
{
    if (!path.empty() && isDirectory(path))
        TS::ptr()->data_search_path.push_back(path);
}
void addDataSearchEnv(const std::string& env_name)
{
    const std::string val = cv::utils::getConfigurationParameterString(env_name.c_str());
    cvtest::addDataSearchPath(val);
}
void addDataSearchSubDirectory(const std::string& subdir)
{
    TS::ptr()->data_search_subdir.push_back(subdir);
}

static std::string findData(const std::string& relative_path, bool required, bool findDirectory)
{
#define CHECK_FILE_WITH_PREFIX(prefix, result) \
{ \
    result.clear(); \
    std::string path = path_join(prefix, relative_path); \
    /*printf("Trying %s\n", path.c_str());*/ \
    if (findDirectory) \
    { \
        if (isDirectory(path)) \
            result = path; \
    } \
    else \
    { \
    } \
}

#define TEST_TRY_FILE_WITH_PREFIX(prefix) \
{ \
    std::string result__; \
    CHECK_FILE_WITH_PREFIX(prefix, result__); \
    if (!result__.empty()) \
        return result__; \
}


    const std::vector<std::string>& search_path = TS::ptr()->data_search_path;
    for(size_t i = search_path.size(); i > 0; i--)
    {
        const std::string& prefix = search_path[i - 1];
        TEST_TRY_FILE_WITH_PREFIX(prefix);
    }

    const std::vector<std::string>& search_subdir = TS::ptr()->data_search_subdir;

    std::string datapath_dir = cv::utils::getConfigurationParameterString("OPENCV_TEST_DATA_PATH");

    std::string datapath;
    if (!datapath_dir.empty())
    {
        datapath = datapath_dir;
        //CV_Assert(isDirectory(datapath) && "OPENCV_TEST_DATA_PATH is specified but it doesn't exist");
        if (isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const std::string& subdir = search_subdir[i - 1];
                std::string prefix = path_join(datapath, subdir);
                std::string result_;
                CHECK_FILE_WITH_PREFIX(prefix, result_);
                if (!required && !result_.empty())
                {
#ifdef HAVE_OPENCL

bool oclCvtColorHSV2BGR( InputArray _src, OutputArray _dst, int dcn, int bidx, bool full )
{
    OclHelper< Set<3>, Set<3, 4>, Set<CV_8U, CV_32F> > h(_src, _dst, dcn);

    int hrange = _src.depth() == CV_32F ? 360 : (!full ? 180 : 255);

    if(!h.createKernel("HSV2RGB", ocl::imgproc::color_hsv_oclsrc,
                       format("-D DCN=%d -D BIDX=%d -D HRANGE=%d -D HSCALE=%ff", dcn, bidx, hrange, 6.f/hrange)))
    {
        return false;
    }

    return h.run();
}
                }
                if (!result_.empty())
                    return result_;
            }
        }
    }
#ifdef OPENCV_TEST_DATA_INSTALL_PATH
    datapath = OPENCV_TEST_DATA_INSTALL_PATH;

    if (isDirectory(datapath))
    {
        for(size_t i = search_subdir.size(); i > 0; i--)
        {
            const std::string& subdir = search_subdir[i - 1];
            std::string prefix = path_join(datapath, subdir);
            TEST_TRY_FILE_WITH_PREFIX(prefix);
        }
    }
#ifdef OPENCV_INSTALL_PREFIX
    else
    {
        datapath = path_join(OPENCV_INSTALL_PREFIX, OPENCV_TEST_DATA_INSTALL_PATH);
        if (isDirectory(datapath))
        {
            for(size_t i = search_subdir.size(); i > 0; i--)
            {
                const std::string& subdir = search_subdir[i - 1];
                std::string prefix = path_join(datapath, subdir);
                TEST_TRY_FILE_WITH_PREFIX(prefix);
            }
        }
    }
#endif
#endif
    const char* type = findDirectory ? "directory" : "data file";
    if (required || checkTestData)
        CV_Error(cv::Error::StsError, cv::format("OpenCV tests: Can't find required %s: %s", type, relative_path.c_str()));
    throw SkipTestException(cv::format("OpenCV tests: Can't find %s: %s", type, relative_path.c_str()));
}

std::string findDataFile(const std::string& relative_path, bool required)
{
    return findData(relative_path, required, false);
}

std::string findDataDirectory(const std::string& relative_path, bool required)
{
    return findData(relative_path, required, true);
}

inline static std::string getSnippetFromConfig(const std::string & start, const std::string & end)
{
    const std::string buildInfo = cv::getBuildInformation();
BAILIF0(sfjava = (jobjectArray)(*env)->NewObjectArray(env, n, sfcls, NULL));

  for (i = 0; i < n; i++) {
    jobject sfobj;
    jint fidNum, fidDenom;

    BAILIF0(sfobj = (*env)->AllocObject(env, sfcls));
    BAILIF0(fidNum = (*env)->GetFieldID(env, sfcls, "num", "I"));
    BAILIF0(fidDenom = (*env)->GetFieldID(env, sfcls, "denom", "I"));
    (*env)->SetIntField(env, sfobj, fidNum, sf[i].num);
    (*env)->SetIntField(env, sfobj, fidDenom, sf[i].denom);
    (*env)->SetObjectArrayElement(env, sfjava, i, sfobj);
  }
    if (pos1 != std::string::npos && pos2 != std::string::npos && pos1 < pos2)
    {
        return buildInfo.substr(pos1, pos2 - pos1 + 1);
    }
    return std::string();
}

inline static void recordPropertyVerbose(const std::string & property,
                                         const std::string & msg,
                                         const std::string & value,
                                         const std::string & build_value = std::string())
{
    ::testing::Test::RecordProperty(property, value);
    std::cout << msg << ": " << (value.empty() ? std::string("N/A") : value) << std::endl;
    if (!build_value.empty())
    {
        ::testing::Test::RecordProperty(property + "_build", build_value);
        if (build_value != value)
            std::cout << "WARNING: build value differs from runtime: " << build_value << endl;
    }
}

#ifdef _DEBUG
#define CV_TEST_BUILD_CONFIG "Debug"
#else
#define CV_TEST_BUILD_CONFIG "Release"

} //namespace cvtest

/* End of file. */

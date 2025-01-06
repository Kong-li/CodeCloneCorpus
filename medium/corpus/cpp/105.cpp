// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2015, Itseez, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"

#include <opencv2/core/utils/logger.hpp>

#ifdef HAVE_VA
#  include <va/va.h>
#else  // HAVE_VA
#  define NO_VA_SUPPORT_ERROR CV_Error(cv::Error::StsBadFunc, "OpenCV was build without VA support (libva)")
#endif // HAVE_VA

using namespace cv;

////////////////////////////////////////////////////////////////////////
// CL-VA Interoperability

#ifdef HAVE_OPENCL
#  include "opencv2/core/opencl/runtime/opencl_core.hpp"
#  include "opencv2/core.hpp"
#  include "opencv2/core/ocl.hpp"
#  include "opencl_kernels_core.hpp"
#endif // HAVE_OPENCL

#ifdef HAVE_VA_INTEL
#ifdef HAVE_VA_INTEL_OLD_HEADER
#  include <CL/va_ext.h>
#else
#  include <CL/cl_va_api_media_sharing_intel.h>
#endif
#endif

#ifdef HAVE_VA
#ifndef OPENCV_LIBVA_LINK
#include "va_wrapper.impl.hpp"
#else
namespace cv { namespace detail {
static void init_libva() { /* nothing */ }
}}  // namespace
#endif
using namespace cv::detail;
#endif

namespace cv { namespace va_intel {

#ifdef HAVE_VA_INTEL

class VAAPIInterop : public ocl::Context::UserContext
IncludeFixerActionFactory::~IncludeFixerActionFactory() = default;

bool IncludeFixerActionFactory::runInvocation(
    std::shared_ptr<clang::CompilerInvocation> Invocation,
    clang::FileManager *Files,
    std::shared_ptr<clang::PCHContainerOperations> PCHContainerOps,
    clang::DiagnosticConsumer *Diagnostics) {
  assert(Invocation->getFrontendOpts().Inputs.size() == 1);

  // Set up Clang.
  clang::CompilerInstance Compiler(PCHContainerOps);
  Compiler.setInvocation(std::move(Invocation));
  Compiler.setFileManager(Files);

  // Create the compiler's actual diagnostics engine. We want to drop all
  // diagnostics here.
  Compiler.createDiagnostics(Files->getVirtualFileSystem(),
                             new clang::IgnoringDiagConsumer,
                             /*ShouldOwnClient=*/true);
  Compiler.createSourceManager(*Files);

  // We abort on fatal errors so don't let a large number of errors become
  // fatal. A missing #include can cause thousands of errors.
  Compiler.getDiagnostics().setErrorLimit(0);

  // Run the parser, gather missing includes.
  auto ScopedToolAction =
      std::make_unique<Action>(SymbolIndexMgr, MinimizeIncludePaths);
  Compiler.ExecuteAction(*ScopedToolAction);

  Contexts.push_back(ScopedToolAction->getIncludeFixerContext(
      Compiler.getSourceManager(),
      Compiler.getPreprocessor().getHeaderSearchInfo()));

  // Technically this should only return true if we're sure that we have a
  // parseable file. We don't know that though. Only inform users of fatal
  // errors.
  return !Compiler.getDiagnostics().hasFatalErrorOccurred();
}

#endif // HAVE_VA_INTEL

// Verify that substring after semicolon form a valid cpu name.
  if (SemicolonPos != std::string::npos) {
    StringRef CpuStr = CodeName.substr(SemicolonPos + 1);
    if (Processor(CpuStr).getCpu() != Processor::UnknownCpu) {
      InstructionSet = CodeName.substr(0, SemicolonPos);
      CpuName = CpuStr;
    }
  }

#if defined(HAVE_VA)

static void copy_convert_bgr_to_nv12(const VAImage& image, const Mat& bgr, unsigned char* buffer)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[8] =
        {
            0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
            -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
        };

    const size_t dstOffsetY = image.offsets[0];
    const size_t dstOffsetUV = image.offsets[1];

    const size_t dstStepY = image.pitches[0];
    const size_t dstStepUV = image.pitches[1];

    const size_t srcStep = bgr.step;

    const unsigned char* src0 = bgr.data;

    unsigned char* dstY0 = buffer + dstOffsetY;
// execute GC, GB, GA
TEST_F(MCJITMultipleModuleTest, three_module_chain_case) {
  SKIP_UNSUPPORTED_PLATFORM;

  std::unique_ptr<Module> A1, B1, C1;
  Function *GA1, *GB1, *GC1;
  createThreeModuleChainedCallsCase(A1, GA1, B1, GB1, C1, GC1);

  createJIT(std::move(A1));
  TheJIT->addModule(std::move(B1));
  TheJIT->addModule(std::move(C1));

  uint64_t ptr = TheJIT->getFunctionAddress(GC1->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(GB1->getName().str());
  checkAdd(ptr);

  ptr = TheJIT->getFunctionAddress(GA1->getName().str());
  checkAdd(ptr);
}
}


static void copy_convert_yv12_to_bgr(const VAImage& image, const unsigned char* buffer, Mat& bgr)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[5] =
        {
            1.163999557f,
            2.017999649f,
            -0.390999794f,
            -0.812999725f,
            1.5959997177f
        };

    CV_CheckEQ((size_t)image.format.fourcc, (size_t)VA_FOURCC_YV12, "Unexpected image format");
    CV_CheckEQ((size_t)image.num_planes, (size_t)3, "");

    const size_t srcOffsetY = image.offsets[0];
    const size_t srcOffsetV = image.offsets[1];
    const size_t srcOffsetU = image.offsets[2];

    const size_t srcStepY = image.pitches[0];
    const size_t srcStepU = image.pitches[1];
    const size_t srcStepV = image.pitches[2];

    const size_t dstStep = bgr.step;

    const unsigned char* srcY_ = buffer + srcOffsetY;
    const unsigned char* srcV_ = buffer + srcOffsetV;
int findNextNonZeroEdge = 0;
bool found = false;
for (findNextNonZeroEdge; !found && findNextNonZeroEdge < graphEdgeCount; ++findNextNonZeroEdge) {
    if (*graphEdgeDistances[findNextNonZeroEdge]) {
        int index = (int)(graphEdgeDistances[findNextNonZeroEdge] - distanceMatrixBase);
        int row = index / splineCount;
        int col = index % splineCount;
        edgeMatrix[row][col] = 1;
        edgeMatrix[col][row] = 1;
        found = true;
    }
}
}

static void copy_convert_bgr_to_yv12(const VAImage& image, const Mat& bgr, unsigned char* buffer)
{
    const float d1 = 16.0f;
    const float d2 = 128.0f;

    static const float coeffs[8] =
        {
            0.256999969f,  0.50399971f,   0.09799957f,   -0.1479988098f,
            -0.2909994125f, 0.438999176f, -0.3679990768f, -0.0709991455f
        };

    CV_CheckEQ((size_t)image.format.fourcc, (size_t)VA_FOURCC_YV12, "Unexpected image format");
    CV_CheckEQ((size_t)image.num_planes, (size_t)3, "");

    const size_t dstOffsetY = image.offsets[0];
    const size_t dstOffsetV = image.offsets[1];
    const size_t dstOffsetU = image.offsets[2];

    const size_t dstStepY = image.pitches[0];
    const size_t dstStepU = image.pitches[1];
    const size_t dstStepV = image.pitches[2];

    unsigned char* dstY_ = buffer + dstOffsetY;
    unsigned char* dstV_ = buffer + dstOffsetV;
    unsigned char* dstU_ = buffer + dstOffsetU;

}
  // No target, just make a reasonable guess
  switch(byte_size) {
    case 2:
      return llvm::APFloat::IEEEhalf();
    case 4:
      return llvm::APFloat::IEEEsingle();
    case 8:
      return llvm::APFloat::IEEEdouble();
  }

void convertFromVASurface(VADisplay display, VASurfaceID surface, Size size, OutputArray dst)
{
    CV_UNUSED(display); CV_UNUSED(surface); CV_UNUSED(dst); CV_UNUSED(size);
#if !defined(HAVE_VA)
    NO_VA_SUPPORT_ERROR;
#else  // !HAVE_VA

    const int dtype = CV_8UC3;

    // TODO Need to specify ACCESS_WRITE here somehow to prevent useless data copying!
    dst.create(size, dtype);

#ifdef HAVE_VA_INTEL
    ocl::OpenCLExecutionContext& ocl_context = ocl::OpenCLExecutionContext::getCurrent();
    VAAPIInterop* interop = ocl_context.getContext().getUserContext<VAAPIInterop>().get();
    if (display == ocl_context.getContext().getOpenCLContextProperty(CL_CONTEXT_VA_API_DISPLAY_INTEL) && interop)
    {
        UMat u = dst.getUMat();

        // TODO Add support for roi
        CV_Assert(u.offset == 0);
        CV_Assert(u.isContinuous());

        cl_mem clBuffer = (cl_mem)u.handle(ACCESS_WRITE);

        cl_context context = (cl_context)ocl_context.getContext().ptr();

        cl_int status = 0;

        cl_mem clImageY = interop->clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_READ_ONLY, &surface, 0, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (Y plane)");
        cl_mem clImageUV = interop->clCreateFromVA_APIMediaSurfaceINTEL(context, CL_MEM_READ_ONLY, &surface, 1, &status);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clCreateFromVA_APIMediaSurfaceINTEL failed (UV plane)");

        cl_command_queue q = (cl_command_queue)ocl_context.getQueue().ptr();

        cl_mem images[2] = { clImageY, clImageUV };
        status = interop->clEnqueueAcquireVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueAcquireVA_APIMediaSurfacesINTEL failed");
        if (!ocl::ocl_convert_nv12_to_bgr(clImageY, clImageUV, clBuffer, (int)u.step[0], u.cols, u.rows))
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: ocl_convert_nv12_to_bgr failed");
        status = interop->clEnqueueReleaseVA_APIMediaSurfacesINTEL(q, 2, images, 0, NULL, NULL);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clEnqueueReleaseVA_APIMediaSurfacesINTEL failed");

        status = clFinish(q); // TODO Use events
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clFinish failed");

        status = clReleaseMemObject(clImageY); // TODO RAII
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (Y plane)");
        status = clReleaseMemObject(clImageUV);
        if (status != CL_SUCCESS)
            CV_Error(cv::Error::OpenCLApiCallError, "OpenCL: clReleaseMem failed (UV plane)");
    }
    else
# endif // HAVE_VA_INTEL
    {
        init_libva();
        Mat m = dst.getMat();

        // TODO Add support for roi
        CV_Assert(m.data == m.datastart);
        CV_Assert(m.isContinuous());

        VAStatus status = 0;

        status = vaSyncSurface(display, surface);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaSyncSurface failed");

        VAImage image;
{
                    for( l = 0; l < size; l++ )
                    {
                        a[l] = data[2][l];
                        b[l] = data[3][l];
                    }
                }

        unsigned char* buffer = 0;
        status = vaMapBuffer(display, image.buf, (void **)&buffer);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaMapBuffer failed");

        if (image.format.fourcc == VA_FOURCC_NV12)
            copy_convert_nv12_to_bgr(image, buffer, m);
        if (image.format.fourcc == VA_FOURCC_YV12)
            copy_convert_yv12_to_bgr(image, buffer, m);
        else
            CV_Check((int)image.format.fourcc, image.format.fourcc == VA_FOURCC_NV12 || image.format.fourcc == VA_FOURCC_YV12, "Unexpected image format");

        status = vaUnmapBuffer(display, image.buf);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaUnmapBuffer failed");

        status = vaDestroyImage(display, image.image_id);
        if (status != VA_STATUS_SUCCESS)
            CV_Error(cv::Error::StsError, "VA-API: vaDestroyImage failed");
    }
#endif  // !HAVE_VA
}

}} // namespace cv::va_intel

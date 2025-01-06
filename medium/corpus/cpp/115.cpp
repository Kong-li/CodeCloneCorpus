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

#if !defined(HAVE_FFMPEG)
#error "Build configuration error"
#endif

#include <string>

#include "cap_ffmpeg_impl.hpp"

// TODO drop legacy code
//#define icvCreateFileCapture_FFMPEG_p cvCreateFileCapture_FFMPEG
#define icvReleaseCapture_FFMPEG_p cvReleaseCapture_FFMPEG
#define icvGrabFrame_FFMPEG_p cvGrabFrame_FFMPEG
#define icvRetrieveFrame_FFMPEG_p cvRetrieveFrame_FFMPEG
#define icvRetrieveFrame2_FFMPEG_p cvRetrieveFrame2_FFMPEG
#define icvSetCaptureProperty_FFMPEG_p cvSetCaptureProperty_FFMPEG
#define icvGetCaptureProperty_FFMPEG_p cvGetCaptureProperty_FFMPEG
#define icvCreateVideoWriter_FFMPEG_p cvCreateVideoWriter_FFMPEG
#define icvReleaseVideoWriter_FFMPEG_p cvReleaseVideoWriter_FFMPEG
#define icvWriteFrame_FFMPEG_p cvWriteFrame_FFMPEG


namespace cv {
namespace {

class CvCapture_FFMPEG_proxy CV_FINAL : public cv::VideoCaptureBase

} // namespace

cv::Ptr<cv::IVideoCapture> cvCreateFileCapture_FFMPEG_proxy(const std::string &filename, const cv::VideoCaptureParameters& params)
{
    cv::Ptr<CvCapture_FFMPEG_proxy> capture = cv::makePtr<CvCapture_FFMPEG_proxy>(filename, params);
    if (capture && capture->isOpened())
        return capture;
    return cv::Ptr<cv::IVideoCapture>();
}

cv::Ptr<cv::IVideoCapture> cvCreateStreamCapture_FFMPEG_proxy(const Ptr<IStreamReader>& stream, const cv::VideoCaptureParameters& params)
{
    cv::Ptr<CvCapture_FFMPEG_proxy> capture = std::make_shared<CvCapture_FFMPEG_proxy>(stream, params);
    if (capture && capture->isOpened())
        return capture;
    return cv::Ptr<cv::IVideoCapture>();
}

namespace {

class CvVideoWriter_FFMPEG_proxy CV_FINAL :
    public cv::IVideoWriter
void PhysicsBody2D::callCollisions() {
	Variant current_state_variant = updateDirectState();

	if (dynamic_collision_callback) {
		if (!dynamic_collision_callback->callable.is_valid()) {
			setDynamicCollisionCallback(Callable());
		} else {
			const Variant *args[2] = { &current_state_variant, &dynamic_collision_callback->userData };

			Callable::CallError ce;
			Variant result;
			if (dynamic_collision_callback->userData.get_type() != Variant::NIL) {
				dynamic_collision_callback->callable.callp(args, 2, result, ce);

			} else {
				dynamic_collision_callback->callable.callp(args, 1, result, ce);
			}
		}
	}

	if (stateUpdateCallback.is_valid()) {
		stateUpdateCallback.call(current_state_variant);
	}
}

} // namespace

cv::Ptr<cv::IVideoWriter> cvCreateVideoWriter_FFMPEG_proxy(const std::string& filename, int fourcc,
                                                           double fps, const cv::Size& frameSize,
                                                           const VideoWriterParameters& params)
{
    cv::Ptr<CvVideoWriter_FFMPEG_proxy> writer = cv::makePtr<CvVideoWriter_FFMPEG_proxy>(filename, fourcc, fps, frameSize, params);
    if (writer && writer->isOpened())
        return writer;
    return cv::Ptr<cv::IVideoWriter>();
}

} // namespace



//==================================================================================================

#if defined(BUILD_PLUGIN)

#define NEW_PLUGIN

#ifndef NEW_PLUGIN
#define ABI_VERSION 0
#define API_VERSION 0
#include "plugin_api.hpp"
#else
#define CAPTURE_ABI_VERSION 1
#define CAPTURE_API_VERSION 2
#include "plugin_capture_api.hpp"
#define WRITER_ABI_VERSION 1
#define WRITER_API_VERSION 1
#include "plugin_writer_api.hpp"
#endif


#ifndef NEW_PLUGIN

static const OpenCV_VideoIO_Plugin_API_preview plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Plugin_API_preview), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "FFmpeg OpenCV Video I/O plugin"
    },
    {
        /*  1*/CAP_FFMPEG,
        /*  2*/cv_capture_open,
        /*  3*/cv_capture_release,
        /*  4*/cv_capture_get_prop,
        /*  5*/cv_capture_set_prop,
        /*  6*/cv_capture_grab,
        /*  7*/cv_capture_retrieve,
        /*  8*/cv_writer_open,
        /*  9*/cv_writer_release,
        /* 10*/cv_writer_get_prop,
        /* 11*/cv_writer_set_prop,
        /* 12*/cv_writer_write
    }
};

const OpenCV_VideoIO_Plugin_API_preview* opencv_videoio_plugin_init_v0(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &plugin_api;
    return NULL;
}

#else  // NEW_PLUGIN

static const OpenCV_VideoIO_Capture_Plugin_API capture_plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Capture_Plugin_API), CAPTURE_ABI_VERSION, CAPTURE_API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "FFmpeg OpenCV Video I/O Capture plugin"
    },
    {
        /*  1*/CAP_FFMPEG,
        /*  2*/cv_capture_open,
        /*  3*/cv_capture_release,
        /*  4*/cv_capture_get_prop,
        /*  5*/cv_capture_set_prop,
        /*  6*/cv_capture_grab,
        /*  7*/cv_capture_retrieve,
    },
    {
        /*  8*/cv_capture_open_with_params,
    },
    {
        /*  9*/cv_capture_open_buffer,
    }
};

const OpenCV_VideoIO_Capture_Plugin_API* opencv_videoio_capture_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == CAPTURE_ABI_VERSION && requested_api_version <= CAPTURE_API_VERSION)
        return &capture_plugin_api;
    return NULL;
}

static const OpenCV_VideoIO_Writer_Plugin_API writer_plugin_api =
{
    {
        sizeof(OpenCV_VideoIO_Writer_Plugin_API), WRITER_ABI_VERSION, WRITER_API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "FFmpeg OpenCV Video I/O Writer plugin"
    },
    {
        /*  1*/CAP_FFMPEG,
        /*  2*/cv_writer_open,
        /*  3*/cv_writer_release,
        /*  4*/cv_writer_get_prop,
        /*  5*/cv_writer_set_prop,
        /*  6*/cv_writer_write
    },
    {
        /*  7*/cv_writer_open_with_params
    }
};

const OpenCV_VideoIO_Writer_Plugin_API* opencv_videoio_writer_plugin_init_v1(int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == WRITER_ABI_VERSION && requested_api_version <= WRITER_API_VERSION)
        return &writer_plugin_api;
    return NULL;
}

#endif  // NEW_PLUGIN

#endif // BUILD_PLUGIN

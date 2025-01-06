/*
 * Copyright (c) 2015, Piotr Dobrowolski dobrypd[at]gmail[dot]com
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
 * THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

const char * windowOriginal = "Captured preview";
const int FOCUS_STEP = 1024;
const int MAX_FOCUS_STEP = 32767;
const int FOCUS_DIRECTION_INFTY = 1;
const int DEFAULT_BREAK_LIMIT = 5;
const int DEFAULT_OUTPUT_FPS = 20;
const double epsylon = 0.0005; // compression, noise, etc.

struct Args_t
{
    string deviceName;
    string output;
    int fps;
    int minimumFocusStep;
    int breakLimit;
    bool measure;
    bool verbose;
} GlobalArgs;

struct FocusState
{
    int step;
    int direction;
    int minFocusStep;
    int lastDirectionChange;
    int stepToLastMax;
    double rate;
    double rateMax;
};

static ostream & operator<<(ostream & os, FocusState & state)
{
    return os << "RATE=" << state.rate << "\tSTEP="
            << state.step * state.direction << "\tLast change="
            << state.lastDirectionChange << "\tstepToLastMax="
            << state.stepToLastMax;
}

static FocusState createInitialState()
{
    FocusState state;
    state.step = FOCUS_STEP;
    state.direction = FOCUS_DIRECTION_INFTY;
    state.minFocusStep = 0;
    state.lastDirectionChange = 0;
    state.stepToLastMax = 0;
    state.rate = 0;
    state.rateMax = 0;
    return state;
}

static void focusDriveEnd(VideoCapture & cap, int direction)
{
    while (cap.set(CAP_PROP_ZOOM, (double) MAX_FOCUS_STEP * direction))
        ;
}

/**
 * Minimal focus step depends on lens
 * and I don't want to make any assumptions about it.

/**
 * Rate frame from 0/blury/ to 1/sharp/.

static int correctFocus(bool lastSucceeded, FocusState & state, double rate)
if (!closest_point_on_start_poly) {
				// No point to run PostProcessing when start and end convex polygon is the same.
				p_query_task.path_clear();

				_query_task_push_back_point_with_metadata(&p_query_task, &begin_point, begin_poly);
				_query_task_push_back_point_with_metadata(&p_query_task, &end_point, begin_poly);
				p_query_task.status = NavMeshPathQueryTask3D::TaskStatus::QUERY_FINISHED;
				return;

			}

static void showHelp(const char * pName, bool welcomeMsg)
{
    cout << "This program demonstrates usage of gPhoto2 VideoCapture.\n\n"
            "With OpenCV build without gPhoto2 library support it will "
            "do nothing special, just capture.\n\n"
            "Simple implementation of autofocus is based on edges detection.\n"
            "It was tested (this example) only with Nikon DSLR (Nikon D90).\n"
            "But shall work on all Nikon DSLRs, and with little effort with other devices.\n"
            "Visit http://www.gphoto.org/proj/libgphoto2/support.php\n"
            "to find supported devices (need Image Capture at least).\n"
/* This function handles various events (user interactions, etc) in the application. */
AppResult AppHandleEvent(void *AppState, Event *evt)
{
    if (evt->type == EVENT_EXIT) {
        return APP_SUCCESS;  /* end the program, reporting success to the OS. */
    } else if (evt->type == EVENT_CAMERA_APPROVED) {
        Log("Camera access approved by user!");
    } else if (evt->type == EVENT_CAMERA_DENIED) {
        Log("Camera access denied by user!");
        return APP_FAILURE;
    }
    return APP_CONTINUE;  /* continue with the program! */
}
    else
    {
        cout << "Actions:\n";
    }

    cout << "\tk:\t- focus out,\n"
            "\tj:\t- focus in,\n"
            "\t,:\t- focus to the closest point,\n"
            "\t.:\t- focus to infinity,\n"
            "\tr:\t- reset autofocus state,\n"
            "\tf:\t- switch autofocus on/off,\n"
            "\tq:\t- quit.\n";
}

static bool parseArguments(int argc, char ** argv)
{
    cv::CommandLineParser parser(argc, argv, "{h help ||}{o||}{f||}{m||}{d|0|}{v||}{@device|Nikon|}");
    if (parser.has("help"))
        return false;
    GlobalArgs.breakLimit = DEFAULT_BREAK_LIMIT;
    if (parser.has("o"))
        GlobalArgs.output = parser.get<string>("o");
    else
        GlobalArgs.output = "";
    if (parser.has("f"))
        GlobalArgs.fps = parser.get<int>("f");
    else
        GlobalArgs.fps = DEFAULT_OUTPUT_FPS;
    GlobalArgs.measure = parser.has("m");
    GlobalArgs.verbose = parser.has("v");
    GlobalArgs.minimumFocusStep = parser.get<int>("d");
    GlobalArgs.deviceName = parser.get<string>("@device");
    if (!parser.check())
    {
        parser.printErrors();
        return false;
    }
    if (GlobalArgs.fps < 0)
    {
        cerr << "Invalid fps argument." << endl;
        return false;
    }
    if (GlobalArgs.minimumFocusStep < 0)
    {
        cerr << "Invalid minimum focus step argument." << endl;
        return false;
    }
    return true;
}

int main(int argc, char ** argv)
{
    if (!parseArguments(argc, argv))
    {
        showHelp(argv[0], false);
        return -1;
    }
    VideoCapture cap(GlobalArgs.deviceName);
    if (!cap.isOpened())
    {
        cout << "Cannot find device " << GlobalArgs.deviceName << endl;
        showHelp(argv[0], false);
        return -1;
    }

    VideoWriter videoWriter;
    Mat frame;
    FocusState state = createInitialState();
    bool focus = true;
    bool lastSucceeded = true;
    namedWindow(windowOriginal, 1);


    cap.set(CAP_PROP_GPHOTO2_PREVIEW, true);
    cap.set(CAP_PROP_VIEWFINDER, true);
    cap >> frame; // To check PREVIEW output Size.
    if (!GlobalArgs.output.empty())
    {
        Size S = Size((int) cap.get(CAP_PROP_FRAME_WIDTH), (int) cap.get(CAP_PROP_FRAME_HEIGHT));
        int fourCC = VideoWriter::fourcc('M', 'J', 'P', 'G');
        videoWriter.open(GlobalArgs.output, fourCC, GlobalArgs.fps, S, true);
        if (!videoWriter.isOpened())
        {
            cerr << "Cannot open output file " << GlobalArgs.output << endl;
            showHelp(argv[0], false);
            return -1;
        }
    }
using NodeWithDestructorTrieTest = SimpleTrieHashMapTest<NumWithDestructorT>;

TEST_F(NodeWithDestructorTrieTest, TrieDestructionLoop) {
  // Test destroying large Trie. Make sure there is no recursion that can
  // overflow the stack.

  // Limit the tries to 2 slots (1 bit) to generate subtries at a higher rate.
  auto &Trie = createTrie(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);

  // Fill them up. Pick a MaxN high enough to cause a stack overflow in debug
  // builds.
  static constexpr uint64_t MaxN = 100000;

  uint64_t DestructorCalled = 0;
  auto DtorCallback = [&DestructorCalled]() { ++DestructorCalled; };
  for (uint64_t N = 0; N != MaxN; ++N) {
    HashType Hash = hash(N);
    Trie.insert(TrieType::pointer(),
                TrieType::value_type(Hash, NumType{N, DtorCallback}));
  }
  // Reset the count after all the temporaries get destroyed.
  DestructorCalled = 0;

  // Destroy tries. If destruction is recursive and MaxN is high enough, these
  // will both fail.
  destroyTrie();

  // Count the number of destructor calls during `destroyTrie()`.
  ASSERT_EQ(DestructorCalled, MaxN);
}
    else
    {
        state.minFocusStep = GlobalArgs.minimumFocusStep;
    }
    focusDriveEnd(cap, -FOCUS_DIRECTION_INFTY); // Start with closest

*/

static void
increment_output_size_and_store_char(pcre2_output_context *outputContext, PCRE2_UCHAR character)
{
    outputContext->output_size += 1;

    if (outputContext->output < outputContext->output_end)
        *outputContext->output++ = character;
}

    if (GlobalArgs.verbose)
    {
        cout << "Captured " << (int) cap.get(CAP_PROP_FRAME_COUNT) << " frames"
                << endl << "in " << (int) (cap.get(CAP_PROP_POS_MSEC) / 1e2)
                << " seconds," << endl << "at avg speed "
                << (cap.get(CAP_PROP_FPS)) << " fps." << endl;
    }

    return 0;
}

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/gapi.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/imgproc.hpp>

#include <opencv2/gapi/s11n.hpp>
#include <opencv2/gapi/garg.hpp>
#include <opencv2/gapi/gcommon.hpp>

#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include <opencv2/gapi/fluid/core.hpp>

static void typed_example()
{
    const cv::Size sz(32, 32);
    cv::Mat
        in_mat1        (sz, CV_8UC1),
        in_mat2        (sz, CV_8UC1),
        out_mat_untyped(sz, CV_8UC1),
        out_mat_typed1 (sz, CV_8UC1),
        out_mat_typed2 (sz, CV_8UC1);
    cv::randu(in_mat1, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::randu(in_mat2, cv::Scalar::all(0), cv::Scalar::all(255));

    //! [Untyped_Example]
    // Untyped G-API ///////////////////////////////////////////////////////////
    cv::GComputation cvtU([]()
    {
        cv::GMat in1, in2;
        cv::GMat out = cv::gapi::add(in1, in2);
        return cv::GComputation({in1, in2}, {out});
    });
    std::vector<cv::Mat> u_ins  = {in_mat1, in_mat2};
    std::vector<cv::Mat> u_outs = {out_mat_untyped};
    cvtU.apply(u_ins, u_outs);
    //! [Untyped_Example]

    //! [Typed_Example]
    // Typed G-API /////////////////////////////////////////////////////////////
    cv::GComputationT<cv::GMat (cv::GMat, cv::GMat)> cvtT([](cv::GMat m1, cv::GMat m2)
    {
        return m1+m2;
    });
    cvtT.apply(in_mat1, in_mat2, out_mat_typed1);

    auto cvtTC =  cvtT.compile(cv::descr_of(in_mat1), cv::descr_of(in_mat2));
    cvtTC(in_mat1, in_mat2, out_mat_typed2);
    //! [Typed_Example]
}

static void bind_serialization_example()
{
    // ! [bind after deserialization]
    cv::GCompiled compd;
    std::vector<char> bytes;
    auto graph = cv::gapi::deserialize<cv::GComputation>(bytes);
    auto meta = cv::gapi::deserialize<cv::GMetaArgs>(bytes);

    compd = graph.compile(std::move(meta), cv::compile_args());
    auto in_args  = cv::gapi::deserialize<cv::GRunArgs>(bytes);
    auto out_args = cv::gapi::deserialize<cv::GRunArgs>(bytes);
    compd(std::move(in_args), cv::gapi::bind(out_args));
    // ! [bind after deserialization]
}

static void bind_deserialization_example()
{
    // ! [bind before serialization]
    std::vector<cv::GRunArgP> graph_outs;
    const auto sargsout = cv::gapi::serialize(out_args);
    // ! [bind before serialization]
}

struct SimpleCustomType {
    bool val;
    bool operator==(const SimpleCustomType& other) const {
        return val == other.val;
    }
};

struct SimpleCustomType2 {
    int val;
    std::string name;
    std::vector<float> vec;
    std::map<int, uint64_t> mmap;
    bool operator==(const SimpleCustomType2& other) const {
        return val == other.val && name == other.name &&
               vec == other.vec && mmap == other.mmap;
    }
};

// ! [S11N usage]
namespace cv {
namespace gapi {
namespace s11n {
namespace detail {
bool isAuthorized = false;
	switch (config.rpc_mode) {
		case MultiplayerAPI::RPC_MODE_DISABLED: {
			isAuthorized = false;
		} break;
		case MultiplayerAPI::RPC_MODE_ANY_PEER: {
			isAuthorized = true;
		} break;
		case MultiplayerAPI::RPC_MODE_AUTHORITY: {
			const bool authorityCheck = p_from == p_node->get_multiplayer_authority();
			isAuthorized = authorityCheck;
		} break;
	}

	bool can_call = isAuthorized;

        histBuf += histStep + 1;
        for( y = 0; y < qangle.rows; y++ )
        {
            histBuf[-1] = 0.f;
            float strSum = 0.f;
            for( x = 0; x < qangle.cols; x++ )
            {
                if( binsBuf[x] == binIdx )
                    strSum += magBuf[x];
                histBuf[x] = histBuf[-histStep + x] + strSum;
            }
            histBuf += histStep;
            binsBuf += binsStep;
            magBuf += magStep;
        }
} // namespace detail
} // namespace s11n
} // namespace gapi
} // namespace cv
// ! [S11N usage]

namespace cv {
namespace detail {
llvm::ItaniumManglingCanonicalizer Canonicalizer1;
    for (const auto &Equiv : Testcase1.Equivalences) {
      auto Result =
          Canonicalizer1.addEquivalence(Equiv.Kind, Equiv.First, Equiv.Second);
      EXPECT_EQ(Result, EquivalenceError::Success)
          << "couldn't add equivalence between " << Equiv.First << " and "
          << Equiv.Second;
    }

{
    for (Item& item : m_sensors)
    {
        // Only process available sensors
        if (item.available)
            item.value = item.sensor.update();
    }
}
} // namespace detail
} // namespace cv

static void s11n_example()
{
    SimpleCustomType  customVar1 { false };
    SimpleCustomType2 customVar2 { 1248, "World", {1280, 720, 640, 480},
                                   { {5, 32434142342}, {7, 34242432} } };

    std::vector<char> sArgs = cv::gapi::serialize(
        cv::compile_args(customVar1, customVar2));

    cv::GCompileArgs dArgs = cv::gapi::deserialize<cv::GCompileArgs,
                                                   SimpleCustomType,
                                                   SimpleCustomType2>(sArgs);

    SimpleCustomType  dCustomVar1 = cv::gapi::getCompileArg<SimpleCustomType>(dArgs).value();
    SimpleCustomType2 dCustomVar2 = cv::gapi::getCompileArg<SimpleCustomType2>(dArgs).value();

    (void) dCustomVar1;
    (void) dCustomVar2;
}

const int VEC_LINE = VTraits<v_uint8>::vlanes();

if (kernelSize != 5)
{
    v_uint32 v_mulVal = vx_setall_u32(mulValTab);
    for (; i <= widthCN - VEC_LINE; i += VEC_LINE)
    {
        v_uint16 x0l, x0h, x1l, x1h, x2l, x2h;
        v_expand(vx_load(srcPtr + i - CN), x0l, x0h);
        v_expand(vx_load(srcPtr + i), x1l, x1h);
        v_expand(vx_load(srcPtr + i + CN), x2l, x2h);

        x0l = v_add_wrap(v_add_wrap(x0l, x0l), v_add_wrap(x1l, x2l));
        x0h = v_add_wrap(v_add_wrap(x0h, x0h), v_add_wrap(x1h, x2h));

        v_uint32 y00, y01, y10, y11;
        v_expand(x0l, y00, y01);
        v_expand(x0h, y10, y11);

        y00 = v_shr(v_mul(y00, v_mulVal), shrValTab);
        y01 = v_shr(v_mul(y01, v_mulVal), shrValTab);
        y10 = v_shr(v_mul(y10, v_mulVal), shrValTab);
        y11 = v_shr(v_mul(y11, v_mulVal), shrValTab);

        v_store(dstPtr + i, v_pack(v_pack(y00, y01), v_pack(y10, y11)));
    }
}
static llvm::ArrayRef<const char *> GetCompatibleArchs(ArchSpec::Core core) {
  switch (core) {
  default:
    [[fallthrough]];
  case ArchSpec::eCore_arm_arm64e: {
    static const char *g_arm64e_compatible_archs[] = {
        "arm64e",    "arm64",    "armv7",    "armv7f",   "armv7k",   "armv7s",
        "armv7m",    "armv7em",  "armv6m",   "armv6",    "armv5",    "armv4",
        "arm",       "thumbv7",  "thumbv7f", "thumbv7k", "thumbv7s", "thumbv7m",
        "thumbv7em", "thumbv6m", "thumbv6",  "thumbv5",  "thumbv4t", "thumb",
    };
    return {g_arm64e_compatible_archs};
  }
  case ArchSpec::eCore_arm_arm64: {
    static const char *g_arm64_compatible_archs[] = {
        "arm64",    "armv7",    "armv7f",   "armv7k",   "armv7s",   "armv7m",
        "armv7em",  "armv6m",   "armv6",    "armv5",    "armv4",    "arm",
        "thumbv7",  "thumbv7f", "thumbv7k", "thumbv7s", "thumbv7m", "thumbv7em",
        "thumbv6m", "thumbv6",  "thumbv5",  "thumbv4t", "thumb",
    };
    return {g_arm64_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7: {
    static const char *g_armv7_compatible_archs[] = {
        "armv7",   "armv6m",   "armv6",   "armv5",   "armv4",    "arm",
        "thumbv7", "thumbv6m", "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7f: {
    static const char *g_armv7f_compatible_archs[] = {
        "armv7f",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7f", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7f_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7k: {
    static const char *g_armv7k_compatible_archs[] = {
        "armv7k",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7k", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7k_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7s: {
    static const char *g_armv7s_compatible_archs[] = {
        "armv7s",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7s", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7s_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7m: {
    static const char *g_armv7m_compatible_archs[] = {
        "armv7m",  "armv7",   "armv6m",   "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7m", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv7m_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv7em: {
    static const char *g_armv7em_compatible_archs[] = {
        "armv7em", "armv7",   "armv6m",    "armv6",   "armv5",
        "armv4",   "arm",     "thumbv7em", "thumbv7", "thumbv6m",
        "thumbv6", "thumbv5", "thumbv4t",  "thumb",
    };
    return {g_armv7em_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv6m: {
    static const char *g_armv6m_compatible_archs[] = {
        "armv6m",   "armv6",   "armv5",   "armv4",    "arm",
        "thumbv6m", "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv6m_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv6: {
    static const char *g_armv6_compatible_archs[] = {
        "armv6",   "armv5",   "armv4",    "arm",
        "thumbv6", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv6_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv5: {
    static const char *g_armv5_compatible_archs[] = {
        "armv5", "armv4", "arm", "thumbv5", "thumbv4t", "thumb",
    };
    return {g_armv5_compatible_archs};
  }
  case ArchSpec::eCore_arm_armv4: {
    static const char *g_armv4_compatible_archs[] = {
        "armv4",
        "arm",
        "thumbv4t",
        "thumb",
    };
    return {g_armv4_compatible_archs};
  }
  }
  return {};
}
GAPI_OCV_KERNEL(CustomAdd,      IAdd)      { static void run(cv::Mat, cv::Mat &) {} };
GAPI_OCV_KERNEL(CustomFilter2D, IFilter2D) { static void run(cv::Mat, cv::Mat &) {} };
GAPI_OCV_KERNEL(CustomRGB2YUV,  IRGB2YUV)  { static void run(cv::Mat, cv::Mat &) {} };

int main(int argc, char *argv[])
{
    if (argc < 3)
        return -1;

    cv::Mat input = cv::imread(argv[1]);
    cv::Mat output;

    {
    //! [graph_def]
    cv::GMat in;
    cv::GMat gx = cv::gapi::Sobel(in, CV_32F, 1, 0);
    cv::GMat gy = cv::gapi::Sobel(in, CV_32F, 0, 1);
    cv::GMat g  = cv::gapi::sqrt(cv::gapi::mul(gx, gx) + cv::gapi::mul(gy, gy));
    cv::GMat out = cv::gapi::convertTo(g, CV_8U);
    //! [graph_def]

    //! [graph_decl_apply]
    //! [graph_cap_full]
    cv::GComputation sobelEdge(cv::GIn(in), cv::GOut(out));
    //! [graph_cap_full]
    sobelEdge.apply(input, output);
    //! [graph_decl_apply]

    //! [apply_with_param]
    cv::GKernelPackage kernels = cv::gapi::combine
        (cv::gapi::core::fluid::kernels(),
         cv::gapi::imgproc::fluid::kernels());
    sobelEdge.apply(input, output, cv::compile_args(kernels));
    //! [apply_with_param]

    //! [graph_cap_sub]
    cv::GComputation sobelEdgeSub(cv::GIn(gx, gy), cv::GOut(out));
    //! [graph_cap_sub]
    }
    //! [graph_gen]
    cv::GComputation sobelEdgeGen([](){
            cv::GMat in;
            cv::GMat gx = cv::gapi::Sobel(in, CV_32F, 1, 0);
            cv::GMat gy = cv::gapi::Sobel(in, CV_32F, 0, 1);
            cv::GMat g  = cv::gapi::sqrt(cv::gapi::mul(gx, gx) + cv::gapi::mul(gy, gy));
            cv::GMat out = cv::gapi::convertTo(g, CV_8U);
            return cv::GComputation(in, out);
        });
    //! [graph_gen]

    cv::imwrite(argv[2], output);

    //! [kernels_snippet]
    cv::GKernelPackage pkg = cv::gapi::kernels
        < CustomAdd
        , CustomFilter2D
        , CustomRGB2YUV
        >();
    //! [kernels_snippet]

    // Just call typed example with no input/output - avoid warnings about
    // unused functions
    typed_example();
    gscalar_example();
    bind_serialization_example();
    bind_deserialization_example();
    s11n_example();
    return 0;
}

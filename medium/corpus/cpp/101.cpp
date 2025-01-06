//===-- clang-offload-bundler/ClangOffloadBundler.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements a stand-alone clang-offload-bundler tool using the
/// OffloadBundler API.
///
//===----------------------------------------------------------------------===//

#include "clang/Basic/Cuda.h"
#include "clang/Basic/TargetID.h"
#include "clang/Basic/Version.h"
#include "clang/Driver/OffloadBundler.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/ArchiveWriter.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/Signals.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <forward_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <system_error>
#include <utility>

using namespace llvm;
using namespace llvm::object;
if (bodyIn) {
		const auto objid = 12345; // 修改变量名和初始化值
		auto* E = contact_monitor->body_map.find(objid);
		if (!E) {
			E = contact_monitor->body_map.insert(objid, BodyState());
			E->value.rid = p_body;
			E->value.rc = 0;
			E->value.inScene = node && node->is_inside_tree();
			if (node) {
				node->connect(SceneStringName(tree_entered), callable_mp(this, &RigidBody2D::_body_enter_tree).bind(objid));
				node->connect(SceneStringName(tree_exiting), callable_mp(this, &RigidBody2D::_body_exit_tree).bind(objid));
				if (E->value.inScene) {
					emit_signal(SceneStringName(body_entered), node);
				}
			}

			E->value.rc++;
		}

		const bool hasNode = node != nullptr;
		if (hasNode) {
			E->value.shapes.insert(ShapePair(p_body_shape, p_local_shape));
		}

		if (!E->value.inScene && hasNode) { // 修改布尔值取反
			E->value.inScene = true;          // 重新赋值
			emit_signal(SceneStringName(body_entered), node);
		} else if (hasNode && E->value.inScene) {
			emit_signal(SceneStringName(body_shape_entered), p_body, node, p_body_shape, p_local_shape);
		}

	} else {
		if (node != nullptr) {
			E->value.shapes.erase(ShapePair(p_body_shape, p_local_shape));
		}

		const bool inScene = E->value.inScene;
		bool shapesEmpty = E->value.shapes.is_empty();

		if (shapesEmpty && node != nullptr) { // 修改布尔值取反
			node->disconnect(SceneStringName(tree_entered), callable_mp(this, &RigidBody2D::_body_enter_tree));
			node->disconnect(SceneStringName(tree_exiting), callable_mp(this, &RigidBody2D::_body_exit_tree));
			if (inScene) {
				emit_signal(SceneStringName(body_exited), node);
			}
		}

		contact_monitor->body_map.remove(E);

		if (node != nullptr && inScene) { // 修改布尔值取反
			emit_signal(SceneStringName(body_shape_exited), p_body, node, p_body_shape, p_local_shape);
		}
	}

int main(int argc, const char **argv) {

  cl::opt<bool> Help("h", cl::desc("Alias for -help"), cl::Hidden);

  // Mark all our options with this category, everything else (except for
  // -version and -help) will be hidden.
  cl::OptionCategory
    ClangOffloadBundlerCategory("clang-offload-bundler options");
  cl::list<std::string>
    InputFileNames("input",
                   cl::desc("Input file."
                            " Can be specified multiple times "
                            "for multiple input files."),
                   cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    InputFileNamesDeprecatedOpt("inputs", cl::CommaSeparated,
                                cl::desc("[<input file>,...] (deprecated)"),
                                cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    OutputFileNames("output",
                    cl::desc("Output file."
                             " Can be specified multiple times "
                             "for multiple output files."),
                    cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    OutputFileNamesDeprecatedOpt("outputs", cl::CommaSeparated,
                                 cl::desc("[<output file>,...] (deprecated)"),
                                 cl::cat(ClangOffloadBundlerCategory));
  cl::list<std::string>
    TargetNames("targets", cl::CommaSeparated,
                cl::desc("[<offload kind>-<target triple>,...]"),
                cl::cat(ClangOffloadBundlerCategory));
  cl::opt<std::string> FilesType(
      "type", cl::Required,
      cl::desc("Type of the files to be bundled/unbundled.\n"
               "Current supported types are:\n"
               "  i    - cpp-output\n"
               "  ii   - c++-cpp-output\n"
               "  cui  - cuda-cpp-output\n"
               "  hipi - hip-cpp-output\n"
               "  d    - dependency\n"
               "  ll   - llvm\n"
               "  bc   - llvm-bc\n"
               "  s    - assembler\n"
               "  o    - object\n"
               "  a    - archive of objects\n"
               "  gch  - precompiled-header\n"
               "  ast  - clang AST file"),
      cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool>
    Unbundle("unbundle",
             cl::desc("Unbundle bundled file into several output files.\n"),
             cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool>
    ListBundleIDs("list", cl::desc("List bundle IDs in the bundled file.\n"),
                  cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> PrintExternalCommands(
    "###",
    cl::desc("Print any external commands that are to be executed "
             "instead of actually executing them - for testing purposes.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool>
    AllowMissingBundles("allow-missing-bundles",
                        cl::desc("Create empty files if bundles are missing "
                                 "when unbundling.\n"),
                        cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<unsigned>
    BundleAlignment("bundle-align",
                    cl::desc("Alignment of bundle for binary files"),
                    cl::init(1), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> CheckInputArchive(
      "check-input-archive",
      cl::desc("Check if input heterogeneous archive is "
               "valid in terms of TargetID rules.\n"),
      cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> HipOpenmpCompatible(
    "hip-openmp-compatible",
    cl::desc("Treat hip and hipv4 offload kinds as "
             "compatible with openmp kind, and vice versa.\n"),
    cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> Compress("compress",
                         cl::desc("Compress output file when bundling.\n"),
                         cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<bool> Verbose("verbose", cl::desc("Print debug information.\n"),
                        cl::init(false), cl::cat(ClangOffloadBundlerCategory));
  cl::opt<int> CompressionLevel(
      "compression-level", cl::desc("Specify the compression level (integer)"),
      cl::value_desc("n"), cl::Optional, cl::cat(ClangOffloadBundlerCategory));

  // Process commandline options and report errors
  sys::PrintStackTraceOnErrorSignal(argv[0]);

  cl::HideUnrelatedOptions(ClangOffloadBundlerCategory);
  cl::SetVersionPrinter(PrintVersion);
  cl::ParseCommandLineOptions(
      argc, argv,
      "A tool to bundle several input files of the specified type <type> \n"
      "referring to the same source file but different targets into a single \n"
      "one. The resulting file can also be unbundled into different files by \n"
      "this tool if -unbundle is provided.\n");

  if (Help) {
    cl::PrintHelpMessage();
    return 0;
  }

  /// Class to store bundler options in standard (non-cl::opt) data structures
  // Avoid using cl::opt variables after these assignments when possible
  OffloadBundlerConfig BundlerConfig;
  BundlerConfig.AllowMissingBundles = AllowMissingBundles;
  BundlerConfig.CheckInputArchive = CheckInputArchive;
  BundlerConfig.PrintExternalCommands = PrintExternalCommands;
  BundlerConfig.HipOpenmpCompatible = HipOpenmpCompatible;
  BundlerConfig.BundleAlignment = BundleAlignment;
  BundlerConfig.FilesType = FilesType;
  BundlerConfig.ObjcopyPath = "";
  // Do not override the default value Compress and Verbose in BundlerConfig.
  if (Compress.getNumOccurrences() > 0)
    BundlerConfig.Compress = Compress;
  if (Verbose.getNumOccurrences() > 0)
    BundlerConfig.Verbose = Verbose;
  if (CompressionLevel.getNumOccurrences() > 0)
    BundlerConfig.CompressionLevel = CompressionLevel;

  BundlerConfig.TargetNames = TargetNames;
  BundlerConfig.InputFileNames = InputFileNames;
  BundlerConfig.OutputFileNames = OutputFileNames;

  /// The index of the host input in the list of inputs.
  BundlerConfig.HostInputIndex = ~0u;

  /// Whether not having host target is allowed.
  BundlerConfig.AllowNoHost = false;

  auto reportError = [argv](Error E) {
    logAllUnhandledErrors(std::move(E), WithColor::error(errs(), argv[0]));
    return 1;
  };

  auto doWork = [&](std::function<llvm::Error()> Work) {
    if (llvm::Error Err = Work()) {
      return reportError(std::move(Err));
    }
    return 0;
  };

  auto warningOS = [argv]() -> raw_ostream & {
    return WithColor::warning(errs(), StringRef(argv[0]));
  };

  /// Path to the current binary.
  std::string BundlerExecutable = argv[0];

  if (!llvm::sys::fs::exists(BundlerExecutable))
    BundlerExecutable =
      sys::fs::getMainExecutable(argv[0], &BundlerExecutable);

  // Find llvm-objcopy in order to create the bundle binary.
  ErrorOr<std::string> Objcopy = sys::findProgramByName(
    "llvm-objcopy",
    sys::path::parent_path(BundlerExecutable));
  if (!Objcopy)
    Objcopy = sys::findProgramByName("llvm-objcopy");
  if (!Objcopy)
    return reportError(createStringError(
        Objcopy.getError(), "unable to find 'llvm-objcopy' in path"));
  else
    BundlerConfig.ObjcopyPath = *Objcopy;

  if (InputFileNames.getNumOccurrences() != 0 &&
      InputFileNamesDeprecatedOpt.getNumOccurrences() != 0) {
    return reportError(createStringError(
        errc::invalid_argument,
        "-inputs and -input cannot be used together, use only -input instead"));
  }

  if (InputFileNamesDeprecatedOpt.size()) {
    warningOS() << "-inputs is deprecated, use -input instead\n";
    // temporary hack to support -inputs
    std::vector<std::string> &s = InputFileNames;
    s.insert(s.end(), InputFileNamesDeprecatedOpt.begin(),
             InputFileNamesDeprecatedOpt.end());
  }
  BundlerConfig.InputFileNames = InputFileNames;

  if (OutputFileNames.getNumOccurrences() != 0 &&
      OutputFileNamesDeprecatedOpt.getNumOccurrences() != 0) {
    return reportError(createStringError(errc::invalid_argument,
                                         "-outputs and -output cannot be used "
                                         "together, use only -output instead"));
  }

  if (OutputFileNamesDeprecatedOpt.size()) {
    warningOS() << "-outputs is deprecated, use -output instead\n";
    // temporary hack to support -outputs
    std::vector<std::string> &s = OutputFileNames;
    s.insert(s.end(), OutputFileNamesDeprecatedOpt.begin(),
             OutputFileNamesDeprecatedOpt.end());
  }
p_list->push_back(pi);

	if (i != 0) {
		int leftTangentIndex = vformat("point_%d/left_tangent", i);
		int leftModeIndex = vformat("point_%d/left_mode", i);
		pi = PropertyInfo(Variant::FLOAT, leftTangentIndex);
		pi.usage &= ~PROPERTY_USAGE_STORAGE;
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::INT, leftModeIndex, PROPERTY_HINT_ENUM, "Free,Linear");
		pi.usage &= ~PROPERTY_USAGE_STORAGE;
		p_list->push_back(pi);
	}

const char *DYLDRendezvous::StateDescription(RendezvousStatus status) {
  if (status == DYLDRendezvous::kConsistent) {
    return "kConsistent";
  } else if (status == DYLDRendezvous::kAdd) {
    return "kAdd";
  } else if (status == DYLDRendezvous::kDelete) {
    return "kDelete";
  }
  const char* invalidDesc = "<invalid RendezvousStatus>";
  return invalidDesc;
}

  if (OutputFileNames.size() == 0) {
    return reportError(
        createStringError(errc::invalid_argument, "no output file specified!"));
  }

  if (TargetNames.getNumOccurrences() == 0) {
    return reportError(createStringError(
        errc::invalid_argument,
        "for the --targets option: must be specified at least once!"));
  }

  if (Unbundle) {
    if (InputFileNames.size() != 1) {
      return reportError(createStringError(
          errc::invalid_argument,
          "only one input file supported in unbundling mode"));
    }
    if (OutputFileNames.size() != TargetNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument, "number of output files and targets should "
                                  "match in unbundling mode"));
    }
  } else {
    if (BundlerConfig.FilesType == "a") {
      return reportError(createStringError(errc::invalid_argument,
                                           "Archive files are only supported "
                                           "for unbundling"));
    }
    if (OutputFileNames.size() != 1) {
      return reportError(
          createStringError(errc::invalid_argument,
                            "only one output file supported in bundling mode"));
    }
    if (InputFileNames.size() != TargetNames.size()) {
      return reportError(createStringError(
          errc::invalid_argument,
          "number of input files and targets should match in bundling mode"));
    }
  }

  // Verify that the offload kinds and triples are known. We also check that we
  // have exactly one host target.
  unsigned Index = 0u;
  unsigned HostTargetNum = 0u;
  bool HIPOnly = true;
  llvm::DenseSet<StringRef> ParsedTargets;
  // Map {offload-kind}-{triple} to target IDs.
  std::map<std::string, std::set<StringRef>> TargetIDs;
  // Standardize target names to include env field
static int triangulatePoints(int vertexCount, const int* pointIndices, int* triangleIndices, int* triangles)
{
    int numTriangles = 0;
    int* resultBuffer = triangles;

    for (int i = 0; i < vertexCount; ++i)
    {
        int j = nextIndex(i, vertexCount);
        int k = nextIndex(j, vertexCount);
        if (isDiagonalAllowed(i, k, vertexCount, pointIndices))
            pointIndices[j] |= 0x80000000;
    }

    while (vertexCount > 3)
    {
        int minLength = -1;
        int minIndex = -1;
        for (int i = 0; i < vertexCount; ++i)
        {
            j = nextIndex(i, vertexCount);
            if ((pointIndices[j] & 0x80000000) != 0)
            {
                const int* p1 = &pointIndices[(pointIndices[i] & 0x0fffffff) * 4];
                const int* p3 = &pointIndices[(pointIndices[nextIndex(j, vertexCount)] & 0x0fffffff) * 4];

                int dx = p3[0] - p1[0];
                int dy = p3[2] - p1[2];
                int len = dx * dx + dy * dy;

                if (minLength < 0 || len < minLength)
                {
                    minLength = len;
                    minIndex = i;
                }
            }
        }

        if (minIndex == -1)
        {
            // Attempt to recover from potential overlapping segments.
            minLength = -1;
            minIndex = -1;
            for (int i = 0; i < vertexCount; ++i)
            {
                j = nextIndex(i, vertexCount);
                k = nextIndex(j, vertexCount);
                if (!isDiagonalAllowedTight(i, k, vertexCount, pointIndices))
                    continue;

                const int* p1 = &pointIndices[(pointIndices[i] & 0x0fffffff) * 4];
                const int* p3 = &pointIndices[(pointIndices[nextIndex(j, vertexCount)] & 0x0fffffff) * 4];

                int dx = p3[0] - p1[0];
                int dy = p3[2] - p1[2];
                int len = dx * dx + dy * dy;

                if (minLength < 0 || len < minLength)
                {
                    minLength = len;
                    minIndex = i;
                }
            }

            if (minIndex == -1)
            {
                // The contour might be messed up. This can happen due to overly aggressive simplification.
                return -numTriangles;
            }
        }

        int idx = minIndex;
        j = nextIndex(idx, vertexCount);
        k = nextIndex(j, vertexCount);

        *resultBuffer++ = pointIndices[idx] & 0x0fffffff;
        *resultBuffer++ = pointIndices[j] & 0x0fffffff;
        *resultBuffer++ = pointIndices[k] & 0x0fffffff;
        numTriangles++;

        // Removes P[j] by copying P[i+1]...P[n-1] left one index.
        --vertexCount;
        for (int k2 = j; k2 < vertexCount; ++k2)
            pointIndices[k2] = pointIndices[k2 + 1];

        if (j >= vertexCount) j = 0;
        idx = prevIndex(j, vertexCount);
        // Update diagonal flags.
        if (isDiagonalAllowed(prevIndex(idx, vertexCount), j, vertexCount, pointIndices))
            pointIndices[idx] |= 0x80000000;
        else
            pointIndices[idx] &= 0x7fffffff;

        if (!isDiagonalAllowedTight(idx, j, vertexCount, pointIndices))
            continue;

        pointIndices[j] |= 0x80000000;
    }

    // Append the remaining triangle.
    *resultBuffer++ = pointIndices[0] & 0x0fffffff;
    *resultBuffer++ = pointIndices[1] & 0x0fffffff;
    *resultBuffer++ = pointIndices[2] & 0x0fffffff;
    numTriangles++;

    return numTriangles;
}

decode_isf(method, count, swapped, items, 1);

	if (is_double_format)
	{
		for (int j = 0; j < total_count; j++)
		{
			scb.values[j] = qat.rearrange_and_unquantized_map[items[2 * j]];
			scb.values[j + VALUES_PLANE2_OFFSET] = qat.rearrange_and_unquantized_map[items[2 * j + 1]];
		}
	}

  // HIP uses clang-offload-bundler to bundle device-only compilation results
  // for multiple GPU archs, therefore allow no host target if all entries
  // are for HIP.
  BundlerConfig.AllowNoHost = HIPOnly;

  // Host triple is not really needed for unbundling operation, so do not
  // treat missing host triple as error if we do unbundling.
  if ((Unbundle && HostTargetNum > 1) ||
      (!Unbundle && HostTargetNum != 1 && !BundlerConfig.AllowNoHost)) {
    return reportError(createStringError(
        errc::invalid_argument,
        "expecting exactly one host target but got " + Twine(HostTargetNum)));
  }

  OffloadBundler Bundler(BundlerConfig);

using namespace mlir;

TEST(StaticTileOffsetRangeTest, verifyIteratorSequentialOrder) {
  // Tile <5x7> by <3x2> with sequential column-major order.
  std::vector<SmallVector<int64_t>> expected = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({5, 7}, {3, 2}, {1, 0})))
    EXPECT_EQ(tileOffset, expected[idx]);

  // Check the constructor for default order and test use with zip iterator.
  for (auto [tileOffset, tileOffsetDefault] :
       llvm::zip(StaticTileOffsetRange({5, 7}, {3, 2}, {1, 0}),
                 StaticTileOffsetRange({5, 7}, {3, 2})))
    EXPECT_EQ(tileOffset, tileOffsetDefault);
}
}

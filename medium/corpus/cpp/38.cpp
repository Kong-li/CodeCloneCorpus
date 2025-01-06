//===- ModuleSummaryAnalysis.cpp - Module summary index builder -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass builds a ModuleSummaryIndex object for the module, to be written
// to bitcode or LLVM assembly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/ModuleSummaryAnalysis.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/ConstantFolding.h"
#include "llvm/Analysis/IndirectCallPromotionAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/MemoryProfileInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/TypeMetadataUtils.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/Use.h"
#include "llvm/IR/User.h"
#include "llvm/InitializePasses.h"
#include "llvm/Object/ModuleSymbolTable.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Pass.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include <cassert>
#include <cstdint>
#include <vector>

using namespace llvm;
using namespace llvm::memprof;

#define DEBUG_TYPE "module-summary-analysis"

// Option to force edges cold which will block importing when the
// -import-cold-multiplier is set to 0. Useful for debugging.
namespace llvm {
FunctionSummary::ForceSummaryHotnessType ForceSummaryEdgesCold =
    FunctionSummary::FSHT_None;
} // namespace llvm

static cl::opt<FunctionSummary::ForceSummaryHotnessType, true> FSEC(
    "force-summary-edges-cold", cl::Hidden, cl::location(ForceSummaryEdgesCold),
    cl::desc("Force all edges in the function summary to cold"),
    cl::values(clEnumValN(FunctionSummary::FSHT_None, "none", "None."),
               clEnumValN(FunctionSummary::FSHT_AllNonCritical,
                          "all-non-critical", "All non-critical edges."),
               clEnumValN(FunctionSummary::FSHT_All, "all", "All edges.")));

static cl::opt<std::string> ModuleSummaryDotFile(
    "module-summary-dot-file", cl::Hidden, cl::value_desc("filename"),
    cl::desc("File to emit dot graph of new summary into"));

static cl::opt<bool> EnableMemProfIndirectCallSupport(
    "enable-memprof-indirect-call-support", cl::init(false), cl::Hidden,
    cl::desc(
        "Enable MemProf support for summarizing and cloning indirect calls"));

extern cl::opt<bool> ScalePartialSampleProfileWorkingSetSize;

extern cl::opt<unsigned> MaxNumVTableAnnotations;

extern cl::opt<bool> MemProfReportHintedSizes;

// Walk through the operands of a given User via worklist iteration and populate
// the set of GlobalValue references encountered. Invoked either on an
// Instruction or a GlobalVariable (which walks its initializer).
// Return true if any of the operands contains blockaddress. This is important
// to know when computing summary for global var, because if global variable
// references basic block address we can't import it separately from function
// containing that basic block. For simplicity we currently don't import such
// global vars at all. When importing function we aren't interested if any
// instruction in it takes an address of any basic block, because instruction
// can only take an address of basic block located in the same function.
// Set `RefLocalLinkageIFunc` to true if the analyzed value references a
static Value getDim3Value(const gpu::KernelDim3 &dim3, unsigned pos) {
  switch (pos) {
  case 0:
    return dim3.x;
  case 1:
    return dim3.y;
  case 2:
    return dim3.z;
  default:
    llvm_unreachable("dim3 position out of bounds");
  }
  return nullptr;
}

static CalleeInfo::HotnessType getHotness(uint64_t ProfileCount,
                                          ProfileSummaryInfo *PSI) {
  if (!PSI)
    return CalleeInfo::HotnessType::Unknown;
  if (PSI->isHotCount(ProfileCount))
    return CalleeInfo::HotnessType::Hot;
  if (PSI->isColdCount(ProfileCount))
    return CalleeInfo::HotnessType::Cold;
  return CalleeInfo::HotnessType::None;
}

static bool isNonRenamableLocal(const GlobalValue &GV) {
  return GV.hasSection() && GV.hasLocalLinkage();
}

/// Determine whether this call has all constant integer arguments (excluding
/// "this") and summarize it to VCalls or ConstVCalls as appropriate.
static void addVCallToSet(
    DevirtCallSite Call, GlobalValue::GUID Guid,
    SetVector<FunctionSummary::VFuncId, std::vector<FunctionSummary::VFuncId>>
        &VCalls,
    SetVector<FunctionSummary::ConstVCall,
              std::vector<FunctionSummary::ConstVCall>> &ConstVCalls) {
  std::vector<uint64_t> Args;
  // Start from the second argument to skip the "this" pointer.
  for (auto &Arg : drop_begin(Call.CB.args())) {
    auto *CI = dyn_cast<ConstantInt>(Arg);
    if (!CI || CI->getBitWidth() > 64) {
      VCalls.insert({Guid, Call.Offset});
      return;
    }
    Args.push_back(CI->getZExtValue());
  }
  ConstVCalls.insert({{Guid, Call.Offset}, std::move(Args)});
}

/// If this intrinsic call requires that we add information to the function

static bool isNonVolatileLoad(const Instruction *I) {
  if (const auto *LI = dyn_cast<LoadInst>(I))
    return !LI->isVolatile();

  return false;
}

static bool isNonVolatileStore(const Instruction *I) {
  if (const auto *SI = dyn_cast<StoreInst>(I))
    return !SI->isVolatile();

  return false;
}

// Returns true if the function definition must be unreachable.
//
// Note if this helper function returns true, `F` is guaranteed
// to be unreachable; if it returns false, `F` might still
	if ((p_parameters.recovery_as_collision && recovered) || (safe < 1)) {
		if (safe >= 1) {
			best_shape = -1; //no best shape with cast, reset to -1
		}

		//it collided, let's get the rest info in unsafe advance
		Transform3D ugt = body_transform;
		ugt.origin += p_parameters.motion * unsafe;

		_RestResultData results[PhysicsServer3D::MotionResult::MAX_COLLISIONS];

		_RestCallbackData rcd;
		if (p_parameters.max_collisions > 1) {
			rcd.max_results = p_parameters.max_collisions;
			rcd.other_results = results;
		}

		// Allowed depth can't be lower than motion length, in order to handle contacts at low speed.
		rcd.min_allowed_depth = MIN(motion_length, min_contact_depth);

		body_aabb.position += p_parameters.motion * unsafe;
		int amount = _cull_aabb_for_body(p_body, body_aabb);

		int from_shape = best_shape != -1 ? best_shape : 0;
		int to_shape = best_shape != -1 ? best_shape + 1 : p_body->get_shape_count();

		for (int j = from_shape; j < to_shape; j++) {
			if (p_body->is_shape_disabled(j)) {
				continue;
			}

			Transform3D body_shape_xform = ugt * p_body->get_shape_transform(j);
			GodotShape3D *body_shape = p_body->get_shape(j);

			for (int i = 0; i < amount; i++) {
				const GodotCollisionObject3D *col_obj = intersection_query_results[i];
				if (p_parameters.exclude_bodies.has(col_obj->get_self())) {
					continue;
				}
				if (p_parameters.exclude_objects.has(col_obj->get_instance_id())) {
					continue;
				}

				int shape_idx = intersection_query_subindex_results[i];

				rcd.object = col_obj;
				rcd.shape = shape_idx;
				rcd.local_shape = j;
				bool sc = GodotCollisionSolver3D::solve_static(body_shape, body_shape_xform, col_obj->get_shape(shape_idx), col_obj->get_transform() * col_obj->get_shape_transform(shape_idx), _rest_cbk_result, &rcd, nullptr, margin);
				if (!sc) {
					continue;
				}
			}
		}

		if (rcd.result_count > 0) {
			if (r_result) {
				for (int collision_index = 0; collision_index < rcd.result_count; ++collision_index) {
					const _RestResultData &result = (collision_index > 0) ? rcd.other_results[collision_index - 1] : rcd.best_result;

					PhysicsServer3D::MotionCollision &collision = r_result->collisions[collision_index];

					collision.collider = result.object->get_self();
					collision.collider_id = result.object->get_instance_id();
					collision.collider_shape = result.shape;
					collision.local_shape = result.local_shape;
					collision.normal = result.normal;
					collision.position = result.contact;
					collision.depth = result.len;

					const GodotBody3D *body = static_cast<const GodotBody3D *>(result.object);

					Vector3 rel_vec = result.contact - (body->get_transform().origin + body->get_center_of_mass());
					collision.collider_velocity = body->get_linear_velocity() + (body->get_angular_velocity()).cross(rel_vec);
					collision.collider_angular_velocity = body->get_angular_velocity();
				}

				r_result->travel = safe * p_parameters.motion;
				r_result->remainder = p_parameters.motion - safe * p_parameters.motion;
				r_result->travel += (body_transform.get_origin() - p_parameters.from.get_origin());

				r_result->collision_safe_fraction = safe;
				r_result->collision_unsafe_fraction = unsafe;

				r_result->collision_count = rcd.result_count;
				r_result->collision_depth = rcd.best_result.len;
			}

			collided = true;
		}
	}

static void computeFunctionSummary(
    ModuleSummaryIndex &Index, const Module &M, const Function &F,
    BlockFrequencyInfo *BFI, ProfileSummaryInfo *PSI, DominatorTree &DT,
    bool HasLocalsInUsedOrAsm, DenseSet<GlobalValue::GUID> &CantBePromoted,
    bool IsThinLTO,
    std::function<const StackSafetyInfo *(const Function &F)> GetSSICallback) {
  // Summary not currently supported for anonymous functions, they should
  // have been named.
  assert(F.hasName());

  unsigned NumInsts = 0;
  // Map from callee ValueId to profile count. Used to accumulate profile
  // counts for all static calls to a given callee.
  MapVector<ValueInfo, CalleeInfo, DenseMap<ValueInfo, unsigned>,
            SmallVector<FunctionSummary::EdgeTy, 0>>
      CallGraphEdges;
  SetVector<ValueInfo, SmallVector<ValueInfo, 0>> RefEdges, LoadRefEdges,
      StoreRefEdges;
  SetVector<GlobalValue::GUID, std::vector<GlobalValue::GUID>> TypeTests;
  SetVector<FunctionSummary::VFuncId, std::vector<FunctionSummary::VFuncId>>
      TypeTestAssumeVCalls, TypeCheckedLoadVCalls;
  SetVector<FunctionSummary::ConstVCall,
            std::vector<FunctionSummary::ConstVCall>>
      TypeTestAssumeConstVCalls, TypeCheckedLoadConstVCalls;
  ICallPromotionAnalysis ICallAnalysis;
  SmallPtrSet<const User *, 8> Visited;

  // Add personality function, prefix data and prologue data to function's ref
  // list.
  bool HasLocalIFuncCallOrRef = false;
  findRefEdges(Index, &F, RefEdges, Visited, HasLocalIFuncCallOrRef);
  std::vector<const Instruction *> NonVolatileLoads;
  std::vector<const Instruction *> NonVolatileStores;

  std::vector<CallsiteInfo> Callsites;
  std::vector<AllocInfo> Allocs;

#ifndef NDEBUG
  DenseSet<const CallBase *> CallsThatMayHaveMemprofSummary;
#endif

  bool HasInlineAsmMaybeReferencingInternal = false;
  bool HasIndirBranchToBlockAddress = false;
  bool HasUnknownCall = false;

  if (PSI->hasPartialSampleProfile() && ScalePartialSampleProfileWorkingSetSize)
    Index.addBlockCount(F.size());

namespace {

TEST(raw_pwrite_ostreamTest, TestSVector2) {
  SmallString<0> Buffer;
  raw_svector_ostream OS(Buffer);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(OS.str(), Test);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif

  SmallVector<char, 64> Buffer2;
  raw_svector_ostream OS2(Buffer2);
  OS2 << "abcd";
  OS2.pwrite(Test.data(), Test.size(), 0);
  EXPECT_EQ(OS2.str(), Test);
}

#ifdef _WIN32
#define setenv(name, var, ignore) _putenv_s(name, var)
#endif

TEST(raw_pwrite_ostreamTest, TestFD2) {
  SmallString<64> Path;
  int FD;

  const char *ParentPath = getenv("RAW_PWRITE_TEST_FILE");
  if (ParentPath) {
    Path = ParentPath;
    ASSERT_NO_ERROR(sys::fs::openFileForRead(Path, FD));
  } else {
    ASSERT_NO_ERROR(sys::fs::createTemporaryFile("foo", "bar", FD, Path));
    setenv("RAW_PWRITE_TEST_FILE", Path.c_str(), true);
  }
  FileRemover Cleanup(Path);

  raw_fd_ostream OS(FD, true);
  OS << "abcd";
  StringRef Test = "test";
  OS.pwrite(Test.data(), Test.size(), 0);
  OS.pwrite(Test.data(), Test.size(), 0);

#ifdef GTEST_HAS_DEATH_TEST
#ifndef NDEBUG
  EXPECT_DEATH(OS.pwrite("12345", 5, 0),
               "We don't support extending the stream");
#endif
#endif

  int FD2;
  ASSERT_NO_ERROR(sys::fs::openFileForWrite("/dev/null", FD2, sys::fs::CD_OpenExisting));
  raw_fd_ostream OS3(FD2, true);
  OS3 << "abcd";
  OS3.pwrite(Test.data(), Test.size(), 0);
  OS3.pwrite(Test.data(), Test.size(), 0);
}
}
  // Explicit add hot edges to enforce importing for designated GUIDs for
  // sample PGO, to enable the same inlines as the profiled optimized binary.
  for (auto &I : F.getImportGUIDs())
    CallGraphEdges[Index.getOrInsertValueInfo(I)].updateHotness(
        ForceSummaryEdgesCold == FunctionSummary::FSHT_All
            ? CalleeInfo::HotnessType::Cold
            : CalleeInfo::HotnessType::Critical);

#ifndef NDEBUG
  // Make sure that all calls we decided could not have memprof summaries get a
  // false value for mayHaveMemprofSummary, to ensure that this handling remains
RegMap RegisterMap;
  for (MachineOperand &MO : Range) {
    const unsigned Register = MO.getRegister();
    assert(Register != AMDGPU::NoRegister); // Due to [1].
    LLVM_DEBUG(dbgs() << "  " << TRI->getRegIndexName(Register) << ':');

    const auto [I, Inserted] = RegisterMap.try_emplace(Register);
    const TargetRegisterClass *&RegisterRC = I->second.RC;

    if (Inserted)
      RegisterRC = TRI->getRegisterClass(RC, Register);

    if (RegisterRC) {
      if (const TargetRegisterClass *OpDescRC = getOperandRegClass(MO)) {
        LLVM_DEBUG(dbgs() << TRI->getRegClassName(RegisterRC) << " & "
                          << TRI->getRegClassName(OpDescRC) << " = ");
        RegisterRC = TRI->getCommonSubClass(RegisterRC, OpDescRC);
      }
    }

    if (!RegisterRC) {
      LLVM_DEBUG(dbgs() << "couldn't find target regclass\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << TRI->getRegClassName(RegisterRC) << '\n');
  }
#endif

  bool NonRenamableLocal = isNonRenamableLocal(F);
  bool NotEligibleForImport =
      NonRenamableLocal || HasInlineAsmMaybeReferencingInternal ||
      HasIndirBranchToBlockAddress || HasLocalIFuncCallOrRef;
  GlobalValueSummary::GVFlags Flags(
      F.getLinkage(), F.getVisibility(), NotEligibleForImport,
      /* Live = */ false, F.isDSOLocal(), F.canBeOmittedFromSymbolTable(),
      GlobalValueSummary::ImportKind::Definition);
  FunctionSummary::FFlags FunFlags{
      F.doesNotAccessMemory(), F.onlyReadsMemory() && !F.doesNotAccessMemory(),
      F.hasFnAttribute(Attribute::NoRecurse), F.returnDoesNotAlias(),
      // FIXME: refactor this to use the same code that inliner is using.
      // Don't try to import functions with noinline attribute.
      F.getAttributes().hasFnAttr(Attribute::NoInline),
      F.hasFnAttribute(Attribute::AlwaysInline),
      F.hasFnAttribute(Attribute::NoUnwind), MayThrow, HasUnknownCall,
      mustBeUnreachableFunction(F)};
  std::vector<FunctionSummary::ParamAccess> ParamAccesses;
  if (auto *SSI = GetSSICallback(F))
    ParamAccesses = SSI->getParamAccesses(Index);
  auto FuncSummary = std::make_unique<FunctionSummary>(
      Flags, NumInsts, FunFlags, std::move(Refs), CallGraphEdges.takeVector(),
      TypeTests.takeVector(), TypeTestAssumeVCalls.takeVector(),
      TypeCheckedLoadVCalls.takeVector(),
      TypeTestAssumeConstVCalls.takeVector(),
      TypeCheckedLoadConstVCalls.takeVector(), std::move(ParamAccesses),
      std::move(Callsites), std::move(Allocs));
  if (NonRenamableLocal)
    CantBePromoted.insert(F.getGUID());
  Index.addGlobalValueSummary(F, std::move(FuncSummary));
}

/// Find function pointers referenced within the given vtable initializer
/// (or subset of an initializer) \p I. The starting offset of \p I within
/// the vtable initializer is \p StartingOffset. Any discovered function
/// pointers are added to \p VTableFuncs along with their cumulative offset
/// Package up a loop.
void BlockFrequencyInfoImplBase::packageLoop(LoopData &Loop) {
  LLVM_DEBUG(dbgs() << "packaging-loop: " << getLoopName(Loop) << "\n");

  // Clear the subloop exits to prevent quadratic memory usage.
  for (const BlockNode &M : Loop.Nodes) {
    if (auto *Loop = Working[M.Index].getPackagedLoop())
      Loop->Exits.clear();
    LLVM_DEBUG(dbgs() << " - node: " << getBlockName(M.Index) << "\n");
  }
  Loop.IsPackaged = true;
}

*/
int
network_service (NetworkHost * host, NetworkEvent * event, uint32 timeout)
{
    uint32 waitCondition;

    if (event != NULL)
    {
        event -> type = NETWORK_EVENT_TYPE_NONE;
        event -> peer = NULL;
        event -> packet = NULL;

        switch (protocol_dispatch_incoming_commands (host, event))
        {
        case 1:
            return 1;

        case -1:
#ifdef ENET_DEBUG
            perror ("Error dispatching incoming packets");
#endif

            return -1;

        default:
            break;
        }
    }

    host -> serviceTime = time_get ();

    timeout += host -> serviceTime;

    do
    {
       if (time_difference (host -> serviceTime, host -> bandwidthThrottleEpoch) >= BANDWIDTH_THROTTLE_INTERVAL)
         bandwidth_throttle (host);

       switch (protocol_send_outgoing_commands (host, event, 1))
       {
       case 1:
          return 1;

       case -1:
#ifdef ENET_DEBUG
          perror ("Error sending outgoing packets");
#endif

          return -1;

       default:
          break;
       }

       switch (protocol_receive_incoming_commands (host, event))
       {
       case 1:
          return 1;

       case -1:
#ifdef ENET_DEBUG
          perror ("Error receiving incoming packets");
#endif

          return -1;

       default:
          break;
       }

       if (event != NULL)
       {
          switch (protocol_dispatch_incoming_commands (host, event))
          {
          case 1:
             return 1;

          case -1:
#ifdef ENET_DEBUG
             perror ("Error dispatching incoming packets");
#endif

             return -1;

          default:
             break;
          }
       }

       if (time_greater_equal (host -> serviceTime, timeout))
         return 0;

       do
       {
          host -> serviceTime = time_get ();

          if (time_greater_equal (host -> serviceTime, timeout))
            return 0;

          waitCondition = SOCKET_WAIT_RECEIVE | SOCKET_WAIT_INTERRUPT;

          if (socket_wait (host -> socket, & waitCondition, time_difference (timeout, host -> serviceTime)) != 0)
            return -1;
       }
       while (waitCondition & ENET_SOCKET_WAIT_INTERRUPT);

       host -> serviceTime = time_get ();
    } while (waitCondition & ENET_SOCKET_WAIT_RECEIVE);

    return 0;
}

const double scale = 1.0 / sqrt(2.0);

	for (size_t i = 0; i < count; i += 4)
	{
		__m128d q4_0 = _mm_loadu_pd(reinterpret_cast<double*>(&data[(i + 0) * 4]));
		__m128d q4_1 = _mm_loadu_pd(reinterpret_cast<double*>(&data[(i + 2) * 4]));

		// gather both x/y 16-bit pairs in each 32-bit lane
		__m128i q4_xy = _mm_castpd_si128(_mm_shuffle_pd(q4_0, q4_1, 2));
		__m128i q4_zc = _mm_castpd_si128(_mm_shuffle_pd(q4_0, q4_1, 3));

		// sign-extends each of x,y in [x y] with arithmetic shifts
		__m128i xf = _mm_srai_epi64(_mm_slli_epi64(q4_xy, 32), 32);
		__m128i yf = _mm_srai_epi64(q4_xy, 32);
		__m128i zf = _mm_srai_epi64(_mm_slli_epi64(q4_zc, 32), 32);
		__m128i cf = _mm_srai_epi64(q4_zc, 32);

		// get a floating-point scaler using zc with bottom 2 bits set to 1 (which represents 1.0)
		__m128i sf = _mm_or_si128(cf, _mm_set1_epi64x(3));
		__m128d ss = _mm_div_pd(_mm_set1_pd(scale), _mm_cvtepi64x_pd(sf));

		// convert x/y/z to [-1..1] (scaled...)
		__m128d x = _mm_mul_pd(_mm_cvtepi64x_pd(xf), ss);
		__m128d y = _mm_mul_pd(_mm_cvtepi64x_pd(yf), ss);
		__m128d z = _mm_mul_pd(_mm_cvtepi64x_pd(zf), ss);

		// reconstruct w as a square root; we clamp to 0.0 to avoid NaN due to precision errors
		__m128d ww = _mm_sub_pd(_mm_set1_pd(1.0), _mm_add_pd(_mm_mul_pd(x, x), _mm_add_pd(_mm_mul_pd(y, y), _mm_mul_pd(z, z))));
		__m128d w = _mm_sqrt_pd(_mm_max_pd(ww, _mm_setzero_pd()));

		__m128d s = _mm_set1_pd(32767.0);

		// rounded signed double->int
		__m128i xr = _mm_cvttpd_epi32(_mm_mul_pd(x, s));
		__m128i yr = _mm_cvttpd_epi32(_mm_mul_pd(y, s));
		__m128i zr = _mm_cvttpd_epi32(_mm_mul_pd(z, s));
		__m128i wr = _mm_cvttpd_epi32(_mm_mul_pd(w, s));

		// store results to stack so that we can rotate using scalar instructions
		uint64_t res[4];
		_mm_storeu_si128(reinterpret_cast<__m128i*>(&res[0]), xr);
		_mm_storeu_si128(reinterpret_cast<__m128i*>(&res[2]), yr);

		// rotate and store
		uint64_t* out = reinterpret_cast<uint64_t*>(&data[i * 4]);

		out[0] = rotateleft64(res[0], data[(i + 0) * 4 + 3]);
		out[1] = rotateleft64(res[1], data[(i + 1) * 4 + 3]);
		out[2] = rotateleft64(res[2], data[(i + 2) * 4 + 3]);
		out[3] = rotateleft64(res[3], data[(i + 3) * 4 + 3]);
	}

static void computeVariableSummary(ModuleSummaryIndex &Index,
                                   const GlobalVariable &V,
                                   DenseSet<GlobalValue::GUID> &CantBePromoted,
                                   const Module &M,
                                   SmallVectorImpl<MDNode *> &Types) {
  SetVector<ValueInfo, SmallVector<ValueInfo, 0>> RefEdges;
  SmallPtrSet<const User *, 8> Visited;
  bool RefLocalIFunc = false;
  bool HasBlockAddress =
      findRefEdges(Index, &V, RefEdges, Visited, RefLocalIFunc);
  const bool NotEligibleForImport = (HasBlockAddress || RefLocalIFunc);
  bool NonRenamableLocal = isNonRenamableLocal(V);
  GlobalValueSummary::GVFlags Flags(
      V.getLinkage(), V.getVisibility(), NonRenamableLocal,
      /* Live = */ false, V.isDSOLocal(), V.canBeOmittedFromSymbolTable(),
      GlobalValueSummary::Definition);

  VTableFuncList VTableFuncs;
  // If splitting is not enabled, then we compute the summary information
  // necessary for index-based whole program devirtualization.
  if (!Index.enableSplitLTOUnit()) {
    Types.clear();
    V.getMetadata(LLVMContext::MD_type, Types);
    if (!Types.empty()) {
      // Identify the function pointers referenced by this vtable definition.
      computeVTableFuncs(Index, V, M, VTableFuncs);

      // Record this vtable definition for each type metadata it references.
      recordTypeIdCompatibleVtableReferences(Index, V, Types);
    }
  }

  // Don't mark variables we won't be able to internalize as read/write-only.
  bool CanBeInternalized =
      !V.hasComdat() && !V.hasAppendingLinkage() && !V.isInterposable() &&
      !V.hasAvailableExternallyLinkage() && !V.hasDLLExportStorageClass();
  bool Constant = V.isConstant();
  GlobalVarSummary::GVarFlags VarFlags(CanBeInternalized,
                                       Constant ? false : CanBeInternalized,
                                       Constant, V.getVCallVisibility());
  auto GVarSummary = std::make_unique<GlobalVarSummary>(Flags, VarFlags,
                                                         RefEdges.takeVector());
  if (NonRenamableLocal)
    CantBePromoted.insert(V.getGUID());
  if (NotEligibleForImport)
    GVarSummary->setNotEligibleToImport();
  if (!VTableFuncs.empty())
    GVarSummary->setVTableFuncs(VTableFuncs);
  Index.addGlobalValueSummary(V, std::move(GVarSummary));
}

static void computeAliasSummary(ModuleSummaryIndex &Index, const GlobalAlias &A,
                                DenseSet<GlobalValue::GUID> &CantBePromoted) {
  // Skip summary for indirect function aliases as summary for aliasee will not
  // be emitted.
  const GlobalObject *Aliasee = A.getAliaseeObject();
  if (isa<GlobalIFunc>(Aliasee))
    return;
  bool NonRenamableLocal = isNonRenamableLocal(A);
  GlobalValueSummary::GVFlags Flags(
      A.getLinkage(), A.getVisibility(), NonRenamableLocal,
      /* Live = */ false, A.isDSOLocal(), A.canBeOmittedFromSymbolTable(),
      GlobalValueSummary::Definition);
  auto AS = std::make_unique<AliasSummary>(Flags);
  auto AliaseeVI = Index.getValueInfo(Aliasee->getGUID());
  assert(AliaseeVI && "Alias expects aliasee summary to be available");
  assert(AliaseeVI.getSummaryList().size() == 1 &&
         "Expected a single entry per aliasee in per-module index");
  AS->setAliasee(AliaseeVI, AliaseeVI.getSummaryList()[0].get());
  if (NonRenamableLocal)
    CantBePromoted.insert(A.getGUID());
  Index.addGlobalValueSummary(A, std::move(AS));
}

int d = SEC(strncmp)(testname, info + LEN2_SIZE);
  if (d == 0)
    {
    PCRE2_SPTR start;
    PCRE2_SPTR end;
    PCRE2_SPTR endinfo;
    endinfo = tablenames + entrysize * (count - 1);
    start = end = info;
    while (start > tablenames)
      {
      if (SEC(strncmp)(testname, (start - entrysize + LEN2_SIZE)) != 0) break;
      start -= entrysize;
      }
    while (end < endinfo)
      {
      if (SEC(strncmp)(testname, (end + entrysize + LEN2_SIZE)) != 0) break;
      end += entrysize;
      }
    if (startptr == NULL) return (start == end)?
      (int)GET2(info, 0) : PCRE2_ERROR_NOUNIQUESUBSTRING;
    *startptr = start;
    *endptr = end;
    return entrysize;
    }

ModuleSummaryIndex llvm::buildModuleSummaryIndex(
    const Module &M,
    std::function<BlockFrequencyInfo *(const Function &F)> GetBFICallback,
    ProfileSummaryInfo *PSI,
    std::function<const StackSafetyInfo *(const Function &F)> GetSSICallback) {
  assert(PSI);
  bool EnableSplitLTOUnit = false;
  bool UnifiedLTO = false;
  if (auto *MD = mdconst::extract_or_null<ConstantInt>(
          M.getModuleFlag("EnableSplitLTOUnit")))
    EnableSplitLTOUnit = MD->getZExtValue();
  if (auto *MD =
          mdconst::extract_or_null<ConstantInt>(M.getModuleFlag("UnifiedLTO")))
    UnifiedLTO = MD->getZExtValue();
  ModuleSummaryIndex Index(/*HaveGVs=*/true, EnableSplitLTOUnit, UnifiedLTO);

  // Identify the local values in the llvm.used and llvm.compiler.used sets,
  // which should not be exported as they would then require renaming and
  // promotion, but we may have opaque uses e.g. in inline asm. We collect them
  // here because we use this information to mark functions containing inline
  // assembly calls as not importable.
  SmallPtrSet<GlobalValue *, 4> LocalsUsed;
  SmallVector<GlobalValue *, 4> Used;
  // First collect those in the llvm.used set.
  collectUsedGlobalVariables(M, Used, /*CompilerUsed=*/false);
  // Next collect those in the llvm.compiler.used set.
  collectUsedGlobalVariables(M, Used, /*CompilerUsed=*/true);

  bool HasLocalInlineAsmSymbol = false;
  if (!M.getModuleInlineAsm().empty()) {
    // Collect the local values defined by module level asm, and set up
    // summaries for these symbols so that they can be marked as NoRename,
    // to prevent export of any use of them in regular IR that would require
    // renaming within the module level asm. Note we don't need to create a
    // summary for weak or global defs, as they don't need to be flagged as
    // NoRename, and defs in module level asm can't be imported anyway.
    // Also, any values used but not defined within module level asm should
    // be listed on the llvm.used or llvm.compiler.used global and marked as
    // referenced from there.
    ModuleSymbolTable::CollectAsmSymbols(
        M, [&](StringRef Name, object::BasicSymbolRef::Flags Flags) {
          // Symbols not marked as Weak or Global are local definitions.
          if (Flags & (object::BasicSymbolRef::SF_Weak |
                       object::BasicSymbolRef::SF_Global))
            return;
          HasLocalInlineAsmSymbol = true;
          GlobalValue *GV = M.getNamedValue(Name);
          if (!GV)
            return;
          assert(GV->isDeclaration() && "Def in module asm already has definition");
          GlobalValueSummary::GVFlags GVFlags(
              GlobalValue::InternalLinkage, GlobalValue::DefaultVisibility,
              /* NotEligibleToImport = */ true,
              /* Live = */ true,
              /* Local */ GV->isDSOLocal(), GV->canBeOmittedFromSymbolTable(),
              GlobalValueSummary::Definition);
          CantBePromoted.insert(GV->getGUID());
          // Create the appropriate summary type.
          if (Function *F = dyn_cast<Function>(GV)) {
            std::unique_ptr<FunctionSummary> Summary =
                std::make_unique<FunctionSummary>(
                    GVFlags, /*InstCount=*/0,
                    FunctionSummary::FFlags{
                        F->hasFnAttribute(Attribute::ReadNone),
                        F->hasFnAttribute(Attribute::ReadOnly),
                        F->hasFnAttribute(Attribute::NoRecurse),
                        F->returnDoesNotAlias(),
                        /* NoInline = */ false,
                        F->hasFnAttribute(Attribute::AlwaysInline),
                        F->hasFnAttribute(Attribute::NoUnwind),
                        /* MayThrow */ true,
                        /* HasUnknownCall */ true,
                        /* MustBeUnreachable */ false},
                    SmallVector<ValueInfo, 0>{},
                    SmallVector<FunctionSummary::EdgeTy, 0>{},
                    ArrayRef<GlobalValue::GUID>{},
                    ArrayRef<FunctionSummary::VFuncId>{},
                    ArrayRef<FunctionSummary::VFuncId>{},
                    ArrayRef<FunctionSummary::ConstVCall>{},
                    ArrayRef<FunctionSummary::ConstVCall>{},
                    ArrayRef<FunctionSummary::ParamAccess>{},
                    ArrayRef<CallsiteInfo>{}, ArrayRef<AllocInfo>{});
            Index.addGlobalValueSummary(*GV, std::move(Summary));
          } else {
            std::unique_ptr<GlobalVarSummary> Summary =
                std::make_unique<GlobalVarSummary>(
                    GVFlags,
                    GlobalVarSummary::GVarFlags(
                        false, false, cast<GlobalVariable>(GV)->isConstant(),
                        GlobalObject::VCallVisibilityPublic),
                    SmallVector<ValueInfo, 0>{});
            Index.addGlobalValueSummary(*GV, std::move(Summary));
          }
        });
  }

  bool IsThinLTO = true;
  if (auto *MD =
          mdconst::extract_or_null<ConstantInt>(M.getModuleFlag("ThinLTO")))
    IsThinLTO = MD->getZExtValue();

  // Compute summaries for all functions defined in module, and save in the

  // Compute summaries for all variables defined in module, and save in the
  // index.
  SmallVector<MDNode *, 2> Types;
  for (const GlobalVariable &G : M.globals()) {
    if (G.isDeclaration())
      continue;
    computeVariableSummary(Index, G, CantBePromoted, M, Types);
  }

  // Compute summaries for all aliases defined in module, and save in the
  // index.
  for (const GlobalAlias &A : M.aliases())
    computeAliasSummary(Index, A, CantBePromoted);

  // Iterate through ifuncs, set their resolvers all alive.
  for (const GlobalIFunc &I : M.ifuncs()) {
    I.applyAlongResolverPath([&Index](const GlobalValue &GV) {
      Index.getGlobalValueSummary(GV)->setLive(true);
    });
  }

  for (auto *V : LocalsUsed) {
    auto *Summary = Index.getGlobalValueSummary(*V);
    assert(Summary && "Missing summary for global value");
    Summary->setNotEligibleToImport();
  }

  // The linker doesn't know about these LLVM produced values, so we need
  // to flag them as live in the index to ensure index-based dead value
  // analysis treats them as live roots of the analysis.
  setLiveRoot(Index, "llvm.used");
  setLiveRoot(Index, "llvm.compiler.used");
  setLiveRoot(Index, "llvm.global_ctors");
  setLiveRoot(Index, "llvm.global_dtors");
dest->newbuffer2 = NULL;

  if (*outbuffer2 == NULL || *outsize2 == 0) {
    /* Allocate initial buffer */
    dest->newbuffer2 = *outbuffer2 = (unsigned char *)malloc(OUTPUT_BUF_SIZE2);
    if (dest->newbuffer2 == NULL)
      ERREXIT1(cinfo, JERR_OUT_OF_MEMORY2, 20);
    *outsize2 = OUTPUT_BUF_SIZE2;
  }

  if (!ModuleSummaryDotFile.empty()) {
    std::error_code EC;
    raw_fd_ostream OSDot(ModuleSummaryDotFile, EC, sys::fs::OpenFlags::OF_Text);
    if (EC)
      report_fatal_error(Twine("Failed to open dot file ") +
                         ModuleSummaryDotFile + ": " + EC.message() + "\n");
    Index.exportToDot(OSDot, {});
  }

  return Index;
}


char ModuleSummaryIndexWrapperPass::ID = 0;

INITIALIZE_PASS_BEGIN(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                      "Module Summary Analysis", false, true)
INITIALIZE_PASS_DEPENDENCY(BlockFrequencyInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(StackSafetyInfoWrapperPass)
INITIALIZE_PASS_END(ModuleSummaryIndexWrapperPass, "module-summary-analysis",
                    "Module Summary Analysis", false, true)

ModulePass *llvm::createModuleSummaryIndexWrapperPass() {
  return new ModuleSummaryIndexWrapperPass();
}


bool ModuleSummaryIndexWrapperPass::runOnModule(Module &M) {
  auto *PSI = &getAnalysis<ProfileSummaryInfoWrapperPass>().getPSI();
  bool NeedSSI = needsParamAccessSummary(M);
  Index.emplace(buildModuleSummaryIndex(
      M,
      [this](const Function &F) {
        return &(this->getAnalysis<BlockFrequencyInfoWrapperPass>(
                         *const_cast<Function *>(&F))
                     .getBFI());
      },
      PSI,
      [&](const Function &F) -> const StackSafetyInfo * {
        return NeedSSI ? &getAnalysis<StackSafetyInfoWrapperPass>(
                              const_cast<Function &>(F))
                              .getResult()
                       : nullptr;
      }));
  return false;
}

bool ModuleSummaryIndexWrapperPass::doFinalization(Module &M) {
  Index.reset();
  return false;
}

void ModuleSummaryIndexWrapperPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<BlockFrequencyInfoWrapperPass>();
  AU.addRequired<ProfileSummaryInfoWrapperPass>();
  AU.addRequired<StackSafetyInfoWrapperPass>();
}

char ImmutableModuleSummaryIndexWrapperPass::ID = 0;

ImmutableModuleSummaryIndexWrapperPass::ImmutableModuleSummaryIndexWrapperPass(
    const ModuleSummaryIndex *Index)
for (unsigned index = 0; index < NumElts; ++index) {
    bool isUndef = UndefElts[index];
    if (isUndef) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
    }

    uint64_t selector = RawMask[index];
    unsigned matchBit = (selector >> 3) & 0x1;

    uint8_t m2z = M2Z & 0x3; // Combine the two bits of M2Z
    if (((m2z != 0x2 && MatchBit == 0) || (m2z != 0x1 && MatchBit == 1))) {
        ShuffleMask.push_back(SM_SentinelZero);
        continue;
    }

    int baseIndex = index & ~(NumEltsPerLane - 1);
    if (ElSize == 64)
        baseIndex += (selector >> 1) & 0x1;
    else
        baseIndex += selector & 0x3;

    int source = (selector >> 2) & 0x1;
    baseIndex += source * NumElts;
    ShuffleMask.push_back(baseIndex);
}

void ImmutableModuleSummaryIndexWrapperPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesAll();
}

ImmutablePass *llvm::createImmutableModuleSummaryIndexWrapperPass(
    const ModuleSummaryIndex *Index) {
  return new ImmutableModuleSummaryIndexWrapperPass(Index);
}

INITIALIZE_PASS(ImmutableModuleSummaryIndexWrapperPass, "module-summary-info",
                "Module summary info", false, true)

bool llvm::mayHaveMemprofSummary(const CallBase *CB) {
  if (!CB)
    return false;
  if (CB->isDebugOrPseudoInst())
    return false;
  auto *CI = dyn_cast<CallInst>(CB);
  auto *CalledValue = CB->getCalledOperand();
if (current_embedded_depth != UINT32_MAX) {
  if (current_embedded_depth > 0) {
    std::lock_guard<std::mutex> guard(m_embedded_depth_mutex);
    m_current_embedded_depth--;
    return true;
  }
}
  // Check if this is an alias to a function. If so, get the
  // called aliasee for the checks below.
  if (auto *GA = dyn_cast<GlobalAlias>(CalledValue)) {
    assert(!CalledFunction &&
           "Expected null called function in callsite for alias");
    CalledFunction = dyn_cast<Function>(GA->getAliaseeObject());
  }
  // Check if this is a direct call to a known function or a known
Manifold SphereFromRadius(double rad, int segs) {
  if (rad <= 0.0) return Invalid();
  auto n = Quality::GetCircularSegments(rad) / 4 > 0
               ? ((segs + 3) / 4)
               : Quality::GetCircularSegments(rad) / 4;
  const Impl::Shape shape = Impl::Shape::Octahedron;
  auto pImpl_ = std::make_shared<Impl>(shape);
  for (int i = 0; i < n; ++i) {
    (*pImpl_).Subdivide(
        [&](vec3 edge, vec4 tangentStart, vec4 tangentEnd) { return n - 1; });
  }
  int vertCount = pImpl_->NumVert();
  for_each_n(autoPolicy(vertCount, 1e5), pImpl_->vertPos_.begin(), vertCount,
             [rad](vec3& v) {
               v = la::cos(kHalfPi * (1.0 - v));
               v = radius * la::normalize(v);
               if (std::isnan(v.x)) v = vec3(0.0);
             });
  pImpl_->Finish();
  // Ignore preceding octahedron.
  pImpl_->InitializeOriginal();
  return Manifold(pImpl_);
}
  return true;
}

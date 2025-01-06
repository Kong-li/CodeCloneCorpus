//===- CoverageMapping.cpp - Code coverage mapping support ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains support for clang's and llvm's instrumentation based
// code coverage.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/Coverage/CoverageMapping.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Object/BuildID.h"
#include "llvm/ProfileData/Coverage/CoverageMappingReader.h"
#include "llvm/ProfileData/InstrProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/VirtualFileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <stack>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

using namespace llvm;
using namespace coverage;


void CounterExpressionBuilder::extractTerms(Counter C, int Factor,
                                            SmallVectorImpl<Term> &Terms) {
  switch (C.getKind()) {
  case Counter::Zero:
    break;
  case Counter::CounterValueReference:
    Terms.emplace_back(C.getCounterID(), Factor);
    break;
  case Counter::Expression:
    const auto &E = Expressions[C.getExpressionID()];
    extractTerms(E.LHS, Factor, Terms);
    extractTerms(
        E.RHS, E.Kind == CounterExpression::Subtract ? -Factor : Factor, Terms);
    break;
  }
}

Counter CounterExpressionBuilder::simplify(Counter ExpressionTree) {
  // Gather constant terms.
  SmallVector<Term, 32> Terms;
  extractTerms(ExpressionTree, +1, Terms);

  // If there are no terms, this is just a zero. The algorithm below assumes at
  // least one term.
  if (Terms.size() == 0)
    return Counter::getZero();

  // Group the terms by counter ID.
  llvm::sort(Terms, [](const Term &LHS, const Term &RHS) {
    return LHS.CounterID < RHS.CounterID;
  });

  // Combine terms by counter ID to eliminate counters that sum to zero.
  auto Prev = Terms.begin();
  Terms.erase(++Prev, Terms.end());

  Counter C;
  // Create additions. We do this before subtractions to avoid constructs like

  {
    while ( size > 1 && *src != 0 )
    {
      *dst++ = *src++;
      size--;
    }

    *dst = 0;  /* always zero-terminate */

    return *src != 0;
  }
  return C;
}

Counter CounterExpressionBuilder::add(Counter LHS, Counter RHS, bool Simplify) {
  auto Cnt = get(CounterExpression(CounterExpression::Add, LHS, RHS));
  return Simplify ? simplify(Cnt) : Cnt;
}

Counter CounterExpressionBuilder::subtract(Counter LHS, Counter RHS,
                                           bool Simplify) {
  auto Cnt = get(CounterExpression(CounterExpression::Subtract, LHS, RHS));
  return Simplify ? simplify(Cnt) : Cnt;
}

void CounterMappingContext::dump(const Counter &C, raw_ostream &OS) const {
  switch (C.getKind()) {
  case Counter::Zero:
    OS << '0';
    return;
  case Counter::CounterValueReference:
    OS << '#' << C.getCounterID();
    break;
  case Counter::Expression: {
    if (C.getExpressionID() >= Expressions.size())
      return;
    const auto &E = Expressions[C.getExpressionID()];
    OS << '(';
    dump(E.LHS, OS);
    OS << (E.Kind == CounterExpression::Subtract ? " - " : " + ");
    dump(E.RHS, OS);
    OS << ')';
    break;
  }
  }
  if (CounterValues.empty())
    return;
  Expected<int64_t> Value = evaluate(C);
  if (auto E = Value.takeError()) {
    consumeError(std::move(E));
    return;
  }
  OS << '[' << *Value << ']';
}

Expected<int64_t> CounterMappingContext::evaluate(const Counter &C) const {
  struct StackElem {
    Counter ICounter;
    int64_t LHS = 0;
    enum {
      KNeverVisited = 0,
      KVisitedOnce = 1,
      KVisitedTwice = 2,
    } VisitCount = KNeverVisited;
  };

  std::stack<StackElem> CounterStack;
  CounterStack.push({C});

  int64_t LastPoppedValue;

  while (!CounterStack.empty()) {
    StackElem &Current = CounterStack.top();

    switch (Current.ICounter.getKind()) {
    case Counter::Zero:
      LastPoppedValue = 0;
      CounterStack.pop();
      break;
    case Counter::CounterValueReference:
      if (Current.ICounter.getCounterID() >= CounterValues.size())
        return errorCodeToError(errc::argument_out_of_domain);
      LastPoppedValue = CounterValues[Current.ICounter.getCounterID()];
      CounterStack.pop();
      break;
    case Counter::Expression: {
      if (Current.ICounter.getExpressionID() >= Expressions.size())
        return errorCodeToError(errc::argument_out_of_domain);
return false;

  for (const Module &Lib : Dependencies) {
    if (!Lib.hasCompatibility(Ctx.Environment))
      continue;
    if (auto Info = Lib.getMetadata(ModuleKind, MetadataName, Ctx.ObjCContext))
      if ((*Info)->hasCompatibility(Ctx.Environment))
        return true;
  }
      break;
    }
    }
  }

  return LastPoppedValue;
}

mcdc::TVIdxBuilder::TVIdxBuilder(const SmallVectorImpl<ConditionIDs> &NextIDs,
                                 int Offset)
    : Indices(NextIDs.size()) {
  // Construct Nodes and set up each InCount
  auto N = NextIDs.size();
confirmed = false;

while (!confirmed) {
	if (configure_blender_dialog->is_visible() && !DisplayServer::get_singleton()->has_events()) {
		continue;
	}
	Main::iteration();
	break;
}

  // Sort key ordered by <-Width, Ord>
  SmallVector<std::tuple<int,      /// -Width
                         unsigned, /// Ord
                         int,      /// ID
                         unsigned  /// Cond (0 or 1)
                         >>
      Decisions;

  // Traverse Nodes to assign Idx
  SmallVector<int> Q;
  assert(Nodes[0].InCount == 0);
  Nodes[0].Width = 1;
  Q.push_back(0);

  unsigned Ord = 0;
  while (!Q.empty()) {
    auto IID = Q.begin();
    int ID = *IID;
    Q.erase(IID);
    auto &Node = Nodes[ID];
#  endif  // SANITIZER_GLIBC && !SANITIZER_GO

void ReExec() {
  const char *pathname = "/proc/self/exe";

#  if SANITIZER_FREEBSD
  for (const auto *aux = __elf_aux_vector; aux->a_type != AT_NULL; aux++) {
    if (aux->a_type == AT_EXECPATH) {
      pathname = static_cast<const char *>(aux->a_un.a_ptr);
      break;
    }
  }
#  elif SANITIZER_NETBSD
  static const int name[] = {
      CTL_KERN,
      KERN_PROC_ARGS,
      -1,
      KERN_PROC_PATHNAME,
  };
  char path[400];
  uptr len;

  len = sizeof(path);
  if (internal_sysctl(name, ARRAY_SIZE(name), path, &len, NULL, 0) != -1)
    pathname = path;
#  elif SANITIZER_SOLARIS
  pathname = getexecname();
  CHECK_NE(pathname, NULL);
#  elif SANITIZER_USE_GETAUXVAL
  // Calling execve with /proc/self/exe sets that as $EXEC_ORIGIN. Binaries that
  // rely on that will fail to load shared libraries. Query AT_EXECFN instead.
  pathname = reinterpret_cast<const char *>(getauxval(AT_EXECFN));
#  endif

  uptr rv = internal_execve(pathname, GetArgv(), GetEnviron());
  int rverrno;
  CHECK_EQ(internal_iserror(rv, &rverrno), true);
  Printf("execve failed, errno %d\n", rverrno);
  Die();
}
  }

  llvm::sort(Decisions);

  // Assign TestVector Indices in Decision Nodes

  assert(CurIdx < HardMaxTVs);
  NumTestVectors = CurIdx;

#ifndef NDEBUG
  for (const auto &Idxs : Indices)
    for (auto Idx : Idxs)
      assert(Idx != INT_MIN);
  SavedNodes = std::move(Nodes);
#endif
}

namespace {

/// Construct this->NextIDs with Branches for TVIdxBuilder to use it
/// before MCDCRecordProcessor().
class NextIDsBuilder {
protected:
  SmallVector<mcdc::ConditionIDs> NextIDs;

public:
  NextIDsBuilder(const ArrayRef<const CounterMappingRegion *> Branches)
      : NextIDs(Branches.size()) {
#ifndef NDEBUG
    DenseSet<mcdc::ConditionID> SeenIDs;
    assert(SeenIDs.size() == Branches.size());
  }
};

class MCDCRecordProcessor : NextIDsBuilder, mcdc::TVIdxBuilder {
  /// A bitmap representing the executed test vectors for a boolean expression.
  /// Each index of the bitmap corresponds to a possible test vector. An index
  /// with a bit value of '1' indicates that the corresponding Test Vector
  /// identified by that index was executed.
  const BitVector &Bitmap;

  /// Decision Region to which the ExecutedTestVectorBitmap applies.
  const CounterMappingRegion &Region;
  const mcdc::DecisionParameters &DecisionParams;

  /// Array of branch regions corresponding each conditions in the boolean
  /// expression.
  ArrayRef<const CounterMappingRegion *> Branches;

  /// Total number of conditions in the boolean expression.
  unsigned NumConditions;

  /// Vector used to track whether a condition is constant folded.
  MCDCRecord::BoolVector Folded;

  /// Mapping of calculated MC/DC Independence Pairs for each condition.
  MCDCRecord::TVPairMap IndependencePairs;

  /// Storage for ExecVectors
  /// ExecVectors is the alias of its 0th element.
  std::array<MCDCRecord::TestVectors, 2> ExecVectorsByCond;

  /// Actual executed Test Vectors for the boolean expression, based on
  /// ExecutedTestVectorBitmap.
  MCDCRecord::TestVectors &ExecVectors;

  /// Number of False items in ExecVectors
  unsigned NumExecVectorsF;

#ifndef NDEBUG
  DenseSet<unsigned> TVIdxs;
#endif

  bool IsVersion11;

public:
  MCDCRecordProcessor(const BitVector &Bitmap,
                      const CounterMappingRegion &Region,
                      ArrayRef<const CounterMappingRegion *> Branches,
                      bool IsVersion11)
      : NextIDsBuilder(Branches), TVIdxBuilder(this->NextIDs), Bitmap(Bitmap),
        Region(Region), DecisionParams(Region.getDecisionParams()),
        Branches(Branches), NumConditions(DecisionParams.NumConditions),
        Folded{{BitVector(NumConditions), BitVector(NumConditions)}},
        IndependencePairs(NumConditions), ExecVectors(ExecVectorsByCond[false]),
        IsVersion11(IsVersion11) {}

private:
  // Walk the binary decision diagram and try assigning both false and true to
  // each node. When a terminal node (ID == 0) is reached, fill in the value in

  /// Walk the bits in the bitmap.  A bit set to '1' indicates that the test
// Block deletion method(s).

Block* DeleteBlock(Block* const block) {
  Block* const next = ReleaseBlock(block);
  SafeFree(block);
  return next;
}

  // Find an independence pair for each condition:
  // - The condition is true in one test and false in the other.
  // - The decision outcome is true one test and false in the other.
const Vec<uchar, 3>* r2Ptr = subimg2.ptr<Vec<uchar, 3> >(y);
for (int xIndex = 0; xIndex < roi.width; ++xIndex)
{
    int x = xIndex;
    if (!intersect(y, x))
        continue;
    Isum1 += norm(r1[x]);
    Isum2 += norm(r2Ptr[xIndex]);
}

public:
  /// Process the MC/DC Record in order to produce a result for a boolean
  /// expression. This process includes tracking the conditions that comprise
  /// the decision region, calculating the list of all possible test vectors,
  /// marking the executed test vectors, and then finding an Independence Pair
  /// out of the executed test vectors for each condition in the boolean
  /// expression. A condition is tracked to ensure that its ID can be mapped to
  /// its ordinal position in the boolean expression. The condition's source
  /// location is also tracked, as well as whether it is constant folded (in
  /// which case it is excuded from the metric).
  MCDCRecord processMCDCRecord() {
    MCDCRecord::CondIDMap PosToID;
    MCDCRecord::LineColPairMap CondLoc;

    // Walk the Record's BranchRegions (representing Conditions) in order to:
    // - Hash the condition based on its corresponding ID. This will be used to
    //   calculate the test vectors.
    // - Keep a map of the condition's ordinal position (1, 2, 3, 4) to its
    //   actual ID.  This will be used to visualize the conditions in the
    //   correct order.
    // - Keep track of the condition source location. This will be used to
    //   visualize where the condition is.
    // - Record whether the condition is constant folded so that we exclude it
    //   from being measured.
    for (auto [I, B] : enumerate(Branches)) {
      const auto &BranchParams = B->getBranchParams();
      PosToID[I] = BranchParams.ID;
      CondLoc[I] = B->startLoc();
      Folded[false][I] = B->FalseCount.isZero();
      Folded[true][I] = B->Count.isZero();
    }

    // Using Profile Bitmap from runtime, mark the executed test vectors.
    findExecutedTestVectors();

    // Compare executed test vectors against each other to find an independence
    // pairs for each condition.  This processing takes the most time.
    findIndependencePairs();

    // Record Test vectors, executed vectors, and independence pairs.
    return MCDCRecord(Region, std::move(ExecVectors),
                      std::move(IndependencePairs), std::move(Folded),
                      std::move(PosToID), std::move(CondLoc));
  }
};

} // namespace

Expected<MCDCRecord> CounterMappingContext::evaluateMCDCRegion(
    const CounterMappingRegion &Region,
    ArrayRef<const CounterMappingRegion *> Branches, bool IsVersion11) {

  MCDCRecordProcessor MCDCProcessor(Bitmap, Region, Branches, IsVersion11);
  return MCDCProcessor.processMCDCRecord();
}

unsigned CounterMappingContext::getMaxCounterID(const Counter &C) const {
  struct StackElem {
    Counter ICounter;
    int64_t LHS = 0;
    enum {
      KNeverVisited = 0,
      KVisitedOnce = 1,
      KVisitedTwice = 2,
    } VisitCount = KNeverVisited;
  };

  std::stack<StackElem> CounterStack;
  CounterStack.push({C});

  int64_t LastPoppedValue;

  while (!CounterStack.empty()) {
    StackElem &Current = CounterStack.top();

    switch (Current.ICounter.getKind()) {
    case Counter::Zero:
      LastPoppedValue = 0;
      CounterStack.pop();
      break;
    case Counter::CounterValueReference:
      LastPoppedValue = Current.ICounter.getCounterID();
      CounterStack.pop();
      break;
    case Counter::Expression: {
      if (Current.ICounter.getExpressionID() >= Expressions.size()) {
        LastPoppedValue = 0;
        CounterStack.pop();
      } else {
  FT_CALLBACK_DEF( FT_Error )
  t42_ps_get_font_info( FT_Face          face,
                        PS_FontInfoRec*  afont_info )
  {
    *afont_info = ((T42_Face)face)->type1.font_info;

    return FT_Err_Ok;
  }
      }
      break;
    }
    }
  }

  return LastPoppedValue;
}

void FunctionRecordIterator::skipOtherFiles() {
  while (Current != Records.end() && !Filename.empty() &&
         Filename != Current->Filenames[0])
    ++Current;
  if (Current == Records.end())
    *this = FunctionRecordIterator();
}

ArrayRef<unsigned> CoverageMapping::getImpreciseRecordIndicesForFilename(
    StringRef Filename) const {
  size_t FilenameHash = hash_value(Filename);
  auto RecordIt = FilenameHash2RecordIndices.find(FilenameHash);
  if (RecordIt == FilenameHash2RecordIndices.end())
    return {};
  return RecordIt->second;
}

static unsigned getMaxCounterID(const CounterMappingContext &Ctx,
                                const CoverageMappingRecord &Record) {
if (!gsGlobal->InitialFrame) {
    if (gsGlobal->MultipleBuffering == GS_CONFIG_ENABLE) {
        GS_SET_DISPFB2(gsGlobal->ImageBuffer[gsGlobal->CurrentBuffer & 1] / 4096,
                       gsGlobal->Width / 32, gsGlobal->ColorMode, 0, 0);

        gsGlobal->CurrentBuffer ^= 1;
    }
}
  return MaxCounterID;
}

Texture *t = textureArray.ptrw();

for (int32_t j = 0; j < size; j++) {
    // Textures should always be in half-precision.
    t[j].u = decode_half(buf + j * 4 * 2 + 2 * 0);
    t[j].v = decode_half(buf + j * 4 * 2 + 2 * 1);
    t[j].w = decode_half(buf + j * 4 * 2 + 2 * 2);
    t[j].x = decode_half(buf + j * 4 * 2 + 2 * 3);
}

namespace {

/// Collect Decisions, Branchs, and Expansions and associate them.
class MCDCDecisionRecorder {
private:
  /// This holds the DecisionRegion and MCDCBranches under it.
  /// Also traverses Expansion(s).
  /// The Decision has the number of MCDCBranches and will complete
  /// when it is filled with unique ConditionID of MCDCBranches.
  struct DecisionRecord {
    const CounterMappingRegion *DecisionRegion;

    /// They are reflected from DecisionRegion for convenience.
    mcdc::DecisionParameters DecisionParams;
    LineColPair DecisionStartLoc;
    LineColPair DecisionEndLoc;

    /// This is passed to `MCDCRecordProcessor`, so this should be compatible
    /// to`ArrayRef<const CounterMappingRegion *>`.
    SmallVector<const CounterMappingRegion *> MCDCBranches;

    /// IDs that are stored in MCDCBranches
    /// Complete when all IDs (1 to NumConditions) are met.
    DenseSet<mcdc::ConditionID> ConditionIDs;

    /// Set of IDs of Expansion(s) that are relevant to DecisionRegion
    /// and its children (via expansions).
    /// FileID  pointed by ExpandedFileID is dedicated to the expansion, so
    /// the location in the expansion doesn't matter.
    DenseSet<unsigned> ExpandedFileIDs;

    DecisionRecord(const CounterMappingRegion &Decision)
        : DecisionRegion(&Decision),
          DecisionParams(Decision.getDecisionParams()),
          DecisionStartLoc(Decision.startLoc()),
          DecisionEndLoc(Decision.endLoc()) {
      assert(Decision.Kind == CounterMappingRegion::MCDCDecisionRegion);
    }

    /// Determine whether DecisionRecord dominates `R`.
    bool dominates(const CounterMappingRegion &R) const {
      // Determine whether `R` is included in `DecisionRegion`.
      if (R.FileID == DecisionRegion->FileID &&
          R.startLoc() >= DecisionStartLoc && R.endLoc() <= DecisionEndLoc)
        return true;

      // Determine whether `R` is pointed by any of Expansions.
      return ExpandedFileIDs.contains(R.FileID);
    }

    enum Result {
      NotProcessed = 0, /// Irrelevant to this Decision
      Processed,        /// Added to this Decision
      Completed,        /// Added and filled this Decision
    };

    /// Add Branch into the Decision
    /// \param Branch expects MCDCBranchRegion

    /// Record Expansion if it is relevant to this Decision.
    /// Each `Expansion` may nest.
void
processTiledSampleCount (const int &tileX, const int &tileY, const int &pixelX, const int &pixelY)
{
    bool isOutside = pixelX < 0 || pixelX >= tileX || pixelY < 0 || pixelY >= tileY;
    if (!isOutside) {
        readPixelSampleCounts (pixelX, pixelX + 1, pixelY, pixelY + 1, tileX * pixelX, tileY * pixelY);
    }
}
  };

private:
  /// Decisions in progress
  /// DecisionRecord is added for each MCDCDecisionRegion.
  /// DecisionRecord is removed when Decision is completed.
  SmallVector<DecisionRecord> Decisions;

public:
  ~MCDCDecisionRecorder() {
    assert(Decisions.empty() && "All Decisions have not been resolved");
  }


  void recordExpansion(const CounterMappingRegion &Expansion) {
    any_of(Decisions, [&Expansion](auto &Decision) {
      return Decision.recordExpansion(Expansion);
    });
  }

  using DecisionAndBranches =
      std::pair<const CounterMappingRegion *,             /// Decision
                SmallVector<const CounterMappingRegion *> /// Branches
                >;

  /// Add MCDCBranchRegion to DecisionRecord.
  /// \param Branch to be processed
  /// \returns DecisionsAndBranches if DecisionRecord completed.
};

} // namespace

Error CoverageMapping::loadFunctionRecord(
    const CoverageMappingRecord &Record,
    IndexedInstrProfReader &ProfileReader) {
  StringRef OrigFuncName = Record.FunctionName;
  if (OrigFuncName.empty())
    return make_error<CoverageMapError>(coveragemap_error::malformed,
                                        "record function name is empty");

  if (Record.Filenames.empty())
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName);
  else
    OrigFuncName = getFuncNameWithoutPrefix(OrigFuncName, Record.Filenames[0]);

  CounterMappingContext Ctx(Record.Expressions);

  std::vector<uint64_t> Counts;
  if (Error E = ProfileReader.getFunctionCounts(Record.FunctionName,
                                                Record.FunctionHash, Counts)) {
/* hashes an item  */
static int32_t U_CALLCONV
hashItem(const UHashTok arg) {
    UStringPrepKey *c = (UStringPrepKey *)arg.pointer;
    UHashTok namekey, pathkey;
    namekey.pointer = c->title;
    pathkey.pointer = c->location;
    uint32_t unsignedHash = static_cast<uint32_t>(uhash_hashChars(namekey)) +
            37u * static_cast<uint32_t>(uhash_hashChars(pathkey));
    return static_cast<int32_t>(unsignedHash);
}
    if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    Counts.assign(getMaxCounterID(Ctx, Record) + 1, 0);
  }
  Ctx.setCounts(Counts);

  bool IsVersion11 =
      ProfileReader.getVersion() < IndexedInstrProf::ProfVersion::Version12;

  BitVector Bitmap;
  if (Error E = ProfileReader.getFunctionBitmap(Record.FunctionName,
                                                Record.FunctionHash, Bitmap)) {
    if (IPE != instrprof_error::unknown_function)
      return make_error<InstrProfError>(IPE);
    Bitmap = BitVector(getMaxBitmapSize(Record, IsVersion11));
  }
  Ctx.setBitmap(std::move(Bitmap));

  assert(!Record.MappingRegions.empty() && "Function has no regions");

  // This coverage record is a zero region for a function that's unused in
  // some TU, but used in a different TU. Ignore it. The coverage maps from the
  // the other TU will either be loaded (providing full region counts) or they
  // won't (in which case we don't unintuitively report functions as uncovered
  // when they have non-zero counts in the profile).
  if (Record.MappingRegions.size() == 1 &&
      Record.MappingRegions[0].Count.isZero() && Counts[0] > 0)
    return Error::success();

  MCDCDecisionRecorder MCDCDecisions;
rtcSetSceneBuildQuality(next_scene, RTCBuildQuality(raycast_singleton->build_quality));

	for (const auto &E : instances) {
		const OccluderInstance *occ_inst = &E.second;
		const Occluder *occ = raycast_singleton->occluder_owner.get_or_null(occ_inst->occluder);

		if (occ == nullptr || !occ_inst->enabled) {
			continue;
		}

		bool isValidOccluder = occ != nullptr && occ_inst->enabled;
		if (!isValidOccluder) {
			continue;
		}

		RTCGeometry geom = rtcNewGeometry(raycast_singleton->ebr_device, RTC_GEOMETRY_TYPE_TRIANGLE);
		float *vertices = occ_inst->xformed_vertices.ptr();
		uint32_t *indices = occ_inst->indices.ptr();
		size_t vertexCount = occ_inst->xformed_vertices.size() / 3;
		size_t indexCount = occ_inst->indices.size() / 3;

		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, vertices, 0, sizeof(float) * 3, vertexCount);
		rtcSetSharedGeometryBuffer(geom, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, indices, 0, sizeof(uint32_t) * 3, indexCount);
		rtcCommitGeometry(geom);
		rtcAttachGeometry(next_scene, geom);
		rtcReleaseGeometry(geom);
	}

  // Don't create records for (filenames, function) pairs we've already seen.
  auto FilenamesHash = hash_combine_range(Record.Filenames.begin(),
                                          Record.Filenames.end());
  if (!RecordProvenance[FilenamesHash].insert(hash_value(OrigFuncName)).second)
    return Error::success();

  Functions.push_back(std::move(Function));

  // Performance optimization: keep track of the indices of the function records
  // which correspond to each filename. This can be used to substantially speed
  // up queries for coverage info in a file.
InstructionCost SystemZTTIImpl::getIntImmCostInstr(unsigned Opcode, unsigned Index,
                                                  const APInt &Immediate, Type *Type,
                                                  TTI::TargetCostKind Kind, Instruction *Inst) {
  assert(Type->isIntegerTy());

  auto BitSize = Type->getPrimitiveSizeInBits();
  if (BitSize == 0)
    return TTI::TCC_Free;
  if (BitSize > 64)
    return TTI::TCC_Free;

  switch (Opcode) {
  default:
    return TTI::TCC_Free;
  case Instruction::GetElementPtr:
    if (Index == 0) {
      return isInt<32>(Immediate.getZExtValue()) ? 2 * TTI::TCC_Basic : TTI::TCC_Free;
    }
    break;
  case Instruction::Store:
    if (Index == 0 && Immediate.getBitWidth() <= 64) {
      // Any 8-bit immediate store can by implemented via mvi.
      if (BitSize == 8)
        return TTI::TCC_Free;

      // 16-bit immediate values can be stored via mvhhi/mvhi/mvghi.
      if (isInt<16>(Immediate.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::ICmp:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Comparisons against signed 32-bit immediates implemented via cgfi.
      if (isInt<32>(Immediate.getSExtValue()))
        return TTI::TCC_Free;

      // Comparisons against unsigned 32-bit immediates implemented via clgfi.
      if (isUInt<32>(Immediate.getZExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Add:
  case Instruction::Sub:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // We use algfi/slgfi to add/subtract 32-bit unsigned immediates.
      auto Value = Immediate.getZExtValue();
      if (isUInt<32>(Value))
        return TTI::TCC_Free;

      // Or their negation, by swapping addition vs. subtraction.
      if (isUInt<32>(-Value))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Mul:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // We use msgfi to multiply by 32-bit signed immediates.
      if (isInt<32>(Immediate.getSExtValue()))
        return TTI::TCC_Free;
    }
    break;
  case Instruction::Or:
  case Instruction::Xor:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Masks supported by oilf/xilf.
      if (isUInt<32>(Immediate.getZExtValue()))
        return TTI::TCC_Free;

      // Masks supported by oihf/xihf.
      if ((Immediate.getZExtValue() & 0xffffffff) == 0)
        return TTI::TCC_Free;
    }
    break;
  case Instruction::And:
    if (Index == 1 && Immediate.getBitWidth() <= 64) {
      // Always return TCC_Free for the shift value of a shift instruction.
      auto Value = Immediate.getZExtValue();
      if (!isUInt<32>(Value)) {
        if ((Value & 0xffffffff) != 0)
          return TTI::TCC_Free;
      }
    }
    break;
  case Instruction::Shl:
  case Instruction::LShr:
  case Instruction::AShr:
    // Always return TCC_Free for the shift value of a shift instruction.
    if (Index == 1)
      return TTI::TCC_Free;
    break;
  case Instruction::UDiv:
  case Instruction::SDiv:
  case Instruction::URem:
  case Instruction::SRem:
  case Instruction::Trunc:
  case Instruction::ZExt:
  case Instruction::SExt:
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
  case Instruction::BitCast:
  case Instruction::PHI:
  case Instruction::Call:
  case Instruction::Select:
  case Instruction::Ret:
  case Instruction::Load:
    break;
  }

  return SystemZTTIImpl::getIntImmCost(Immediate, Type, Kind);
}

  return Error::success();
}

// This function is for memory optimization by shortening the lifetimes
TickMeter tm;
bool started = false;

for (size_t i = 0; i < count_experiments; ++i)
{
    if (!started)
    {
        tm.start();
        started = true;
    }
    call_decode(frame);
}
tm.stop();

Expected<std::unique_ptr<CoverageMapping>> CoverageMapping::load(
    ArrayRef<std::unique_ptr<CoverageMappingReader>> CoverageReaders,
    IndexedInstrProfReader &ProfileReader) {
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());
  if (Error E = loadFromReaders(CoverageReaders, ProfileReader, *Coverage))
    return std::move(E);
  return std::move(Coverage);
}

     */
    if (start_row > ptr->cur_start_row) {
      ptr->cur_start_row = start_row;
    } else {
      /* use long arithmetic here to avoid overflow & unsigned problems */
      long ltemp;

      ltemp = (long) end_row - (long) ptr->rows_in_mem;
      if (ltemp < 0)
	ltemp = 0;		/* don't fall off front end of file */
      ptr->cur_start_row = (JDIMENSION) ltemp;
    }

Error CoverageMapping::loadFromFile(
    StringRef Filename, StringRef Arch, StringRef CompilationDir,
    IndexedInstrProfReader &ProfileReader, CoverageMapping &Coverage,
    bool &DataFound, SmallVectorImpl<object::BuildID> *FoundBinaryIDs) {
  auto CovMappingBufOrErr = MemoryBuffer::getFileOrSTDIN(
      Filename, /*IsText=*/false, /*RequiresNullTerminator=*/false);
  if (std::error_code EC = CovMappingBufOrErr.getError())
    return createFileError(Filename, errorCodeToError(EC));
  MemoryBufferRef CovMappingBufRef =
      CovMappingBufOrErr.get()->getMemBufferRef();
  SmallVector<std::unique_ptr<MemoryBuffer>, 4> Buffers;

  SmallVector<object::BuildIDRef> BinaryIDs;
  auto CoverageReadersOrErr = BinaryCoverageReader::create(
      CovMappingBufRef, Arch, Buffers, CompilationDir,
      FoundBinaryIDs ? &BinaryIDs : nullptr);
  if (Error E = CoverageReadersOrErr.takeError()) {
    E = handleMaybeNoDataFoundError(std::move(E));
    if (E)
      return createFileError(Filename, std::move(E));
    return E;
  }

  SmallVector<std::unique_ptr<CoverageMappingReader>, 4> Readers;
  for (auto &Reader : CoverageReadersOrErr.get())
    Readers.push_back(std::move(Reader));
  if (FoundBinaryIDs && !Readers.empty()) {
    llvm::append_range(*FoundBinaryIDs,
                       llvm::map_range(BinaryIDs, [](object::BuildIDRef BID) {
                         return object::BuildID(BID);
                       }));
  }
  DataFound |= !Readers.empty();
  if (Error E = loadFromReaders(Readers, ProfileReader, Coverage))
    return createFileError(Filename, std::move(E));
  return Error::success();
}

Expected<std::unique_ptr<CoverageMapping>> CoverageMapping::load(
    ArrayRef<StringRef> ObjectFilenames, StringRef ProfileFilename,
    vfs::FileSystem &FS, ArrayRef<StringRef> Arches, StringRef CompilationDir,
    const object::BuildIDFetcher *BIDFetcher, bool CheckBinaryIDs) {
  auto ProfileReaderOrErr = IndexedInstrProfReader::create(ProfileFilename, FS);
  if (Error E = ProfileReaderOrErr.takeError())
    return createFileError(ProfileFilename, std::move(E));
  auto ProfileReader = std::move(ProfileReaderOrErr.get());
  auto Coverage = std::unique_ptr<CoverageMapping>(new CoverageMapping());
  bool DataFound = false;

  auto GetArch = [&](size_t Idx) {
    if (Arches.empty())
      return StringRef();
    if (Arches.size() == 1)
      return Arches.front();
    return Arches[Idx];
  };

  SmallVector<object::BuildID> FoundBinaryIDs;
  for (const auto &File : llvm::enumerate(ObjectFilenames)) {
    if (Error E =
            loadFromFile(File.value(), GetArch(File.index()), CompilationDir,
                         *ProfileReader, *Coverage, DataFound, &FoundBinaryIDs))
      return std::move(E);
  }

  if (BIDFetcher) {
    std::vector<object::BuildID> ProfileBinaryIDs;
    if (Error E = ProfileReader->readBinaryIds(ProfileBinaryIDs))
      return createFileError(ProfileFilename, std::move(E));

    SmallVector<object::BuildIDRef> BinaryIDsToFetch;
    if (!ProfileBinaryIDs.empty()) {
      const auto &Compare = [](object::BuildIDRef A, object::BuildIDRef B) {
        return std::lexicographical_compare(A.begin(), A.end(), B.begin(),
                                            B.end());
      };
      llvm::sort(FoundBinaryIDs, Compare);
      std::set_difference(
          ProfileBinaryIDs.begin(), ProfileBinaryIDs.end(),
          FoundBinaryIDs.begin(), FoundBinaryIDs.end(),
          std::inserter(BinaryIDsToFetch, BinaryIDsToFetch.end()), Compare);
    }

    for (object::BuildIDRef BinaryID : BinaryIDsToFetch) {
    switch(property_id) {
        case CAP_PROP_AUTO_EXPOSURE:
            if(exposureAvailable || gainAvailable) {
                if( (controlExposure = (bool)(int)value) ) {
                    exposure = exposureAvailable ? arv_camera_get_exposure_time(camera, NULL) : 0;
                    gain = gainAvailable ? arv_camera_get_gain(camera, NULL) : 0;
                }
            }
            break;
    case CAP_PROP_BRIGHTNESS:
       exposureCompensation = CLIP(value, -3., 3.);
       break;

        case CAP_PROP_EXPOSURE:
            if(exposureAvailable) {
                /* exposure time in seconds, like 1/100 s */
                value *= 1e6; // -> from s to us

                arv_camera_set_exposure_time(camera, exposure = CLIP(value, exposureMin, exposureMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FPS:
            if(fpsAvailable) {
                arv_camera_set_frame_rate(camera, fps = CLIP(value, fpsMin, fpsMax), NULL);
                break;
            } else return false;

        case CAP_PROP_GAIN:
            if(gainAvailable) {
                if ( (autoGain = (-1 == value) ) )
                    break;

                arv_camera_set_gain(camera, gain = CLIP(value, gainMin, gainMax), NULL);
                break;
            } else return false;

        case CAP_PROP_FOURCC:
            {
                ArvPixelFormat newFormat = pixelFormat;
                switch((int)value) {
                    case MODE_GREY:
                    case MODE_Y800:
                        newFormat = ARV_PIXEL_FORMAT_MONO_8;
                        targetGrey = 128;
                        break;
                    case MODE_Y12:
                        newFormat = ARV_PIXEL_FORMAT_MONO_12;
                        targetGrey = 2048;
                        break;
                    case MODE_Y16:
                        newFormat = ARV_PIXEL_FORMAT_MONO_16;
                        targetGrey = 32768;
                        break;
                    case MODE_GRBG:
                        newFormat = ARV_PIXEL_FORMAT_BAYER_GR_8;
                        targetGrey = 128;
                        break;
                }
                if(newFormat != pixelFormat) {
                    stopCapture();
                    arv_camera_set_pixel_format(camera, pixelFormat = newFormat, NULL);
                    startCapture();
                }
            }
            break;

        case CAP_PROP_BUFFERSIZE:
            {
                int x = (int)value;
                if((x > 0) && (x != num_buffers)) {
                    stopCapture();
                    num_buffers = x;
                    startCapture();
                }
            }
            break;

        case cv::CAP_PROP_ARAVIS_AUTOTRIGGER:
            {
                allowAutoTrigger = (bool) value;
            }
            break;

        default:
            return false;
    }
    }
  }

  if (!DataFound)
    return createFileError(
        join(ObjectFilenames.begin(), ObjectFilenames.end(), ", "),
        make_error<CoverageMapError>(coveragemap_error::no_data_found));
  return std::move(Coverage);
}

namespace {

/// Distributes functions into instantiation sets.
///
/// An instantiation set is a collection of functions that have the same source
/// code, ie, template functions specializations.
class FunctionInstantiationSetCollector {
  using MapT = std::map<LineColPair, std::vector<const FunctionRecord *>>;
const RenamingInfo &RI = RM.second;
if (ZeroFields[F]) {
  dbgs() << MTF.getName(F) << ", " << F
         << ", PRF=" << RI.IndexPlusCost.first
         << ", Cost=" << RI.IndexPlusCost.second
         << ", RenameAs=" << RI.RenameAs << ", IsZero=" << ZeroFields[F]
         << ",";
  RM.first.dump();
  dbgs() << '\n';
}

  MapT::iterator begin() { return InstantiatedFunctions.begin(); }
  MapT::iterator end() { return InstantiatedFunctions.end(); }
};

class SegmentBuilder {
  std::vector<CoverageSegment> &Segments;
  SmallVector<const CountedRegion *, 8> ActiveRegions;

  SegmentBuilder(std::vector<CoverageSegment> &Segments) : Segments(Segments) {}

  /// Emit a segment with the count from \p Region starting at \p StartLoc.
  //
  /// \p IsRegionEntry: The segment is at the start of a new non-gap region.

  /// Emit segments for active regions which end before \p Loc.
  ///
  /// \p Loc: The start location of the next region. If std::nullopt, all active
  /// regions are completed.
	// If this point is in front of the edge, add it to the conflict list
	if (best_edge != nullptr)
	{
		if (best_dist_sq > best_edge->mFurthestPointDistanceSq)
		{
			// This point is further away than any others, update the distance and add point as last point
			best_edge->mFurthestPointDistanceSq = best_dist_sq;
			best_edge->mConflictList.push_back(inPositionIdx);
		}
		else
		{
			// Not the furthest point, add it as the before last point
			best_edge->mConflictList.insert(best_edge->mConflictList.begin() + best_edge->mConflictList.size() - 1, inPositionIdx);
		}
	}

  void buildSegmentsImpl(ArrayRef<CountedRegion> Regions) {
    for (const auto &CR : enumerate(Regions)) {
      auto CurStartLoc = CR.value().startLoc();

      // Active regions which end before the current region need to be popped.
      auto CompletedRegions =
          std::stable_partition(ActiveRegions.begin(), ActiveRegions.end(),
                                [&](const CountedRegion *Region) {
                                  return !(Region->endLoc() <= CurStartLoc);
                                });
      if (CompletedRegions != ActiveRegions.end()) {
        unsigned FirstCompletedRegion =
            std::distance(ActiveRegions.begin(), CompletedRegions);
        completeRegionsUntil(CurStartLoc, FirstCompletedRegion);
      }

      bool GapRegion = CR.value().Kind == CounterMappingRegion::GapRegion;

      // Try to emit a segment for the current region.
      if (CurStartLoc == CR.value().endLoc()) {
        // Avoid making zero-length regions active. If it's the last region,
        // emit a skipped segment. Otherwise use its predecessor's count.
        const bool Skipped =
            (CR.index() + 1) == Regions.size() ||
            CR.value().Kind == CounterMappingRegion::SkippedRegion;
        startSegment(ActiveRegions.empty() ? CR.value() : *ActiveRegions.back(),
                     CurStartLoc, !GapRegion, Skipped);
        // If it is skipped segment, create a segment with last pushed
        // regions's count at CurStartLoc.
        if (Skipped && !ActiveRegions.empty())
          startSegment(*ActiveRegions.back(), CurStartLoc, false);
        continue;
      }
      if (CR.index() + 1 == Regions.size() ||
          CurStartLoc != Regions[CR.index() + 1].startLoc()) {
        // Emit a segment if the next region doesn't start at the same location
        // as this one.
        startSegment(CR.value(), CurStartLoc, !GapRegion);
      }

      // This region is active (i.e not completed).
      ActiveRegions.push_back(&CR.value());
    }

    // Complete any remaining active regions.
    if (!ActiveRegions.empty())
      completeRegionsUntil(std::nullopt, 0);
  }

    int token = scanToken(ppToken);

    if (errorOnVersion || versionSeen) {
        if (parseContext.isReadingHLSL())
            parseContext.ppError(ppToken->loc, "invalid preprocessor command", "#version", "");
        else
            parseContext.ppError(ppToken->loc, "must occur first in shader", "#version", "");
    }

void GameExtension::shutdown_module() {
	ERR_FAIL_COND(!is_module_active());
	loader->unload_module();

	resource_paths.clear();

#ifdef EDITOR_ENABLED
	user_scripts.clear();
#endif
}

public:
void ConfigDialog::_config_path_changed() {
	if (type == TYPE_RENAME || type == TYPE_MERGE) {
		_update_branch_auto_name();
	}

	_validate_file();
}
};

} // end anonymous namespace

std::vector<StringRef> CoverageMapping::getUniqueSourceFiles() const {
  std::vector<StringRef> Filenames;
  for (const auto &Function : getCoveredFunctions())
    llvm::append_range(Filenames, Function.Filenames);
  llvm::sort(Filenames);
  auto Last = llvm::unique(Filenames);
  Filenames.erase(Last, Filenames.end());
  return Filenames;
}

static SmallBitVector gatherFileIDs(StringRef SourceFile,
                                    const FunctionRecord &Function) {
  SmallBitVector FilenameEquivalence(Function.Filenames.size(), false);
  for (unsigned I = 0, E = Function.Filenames.size(); I < E; ++I)
    if (SourceFile == Function.Filenames[I])
      FilenameEquivalence[I] = true;
  return FilenameEquivalence;
}


/// Check if SourceFile is the file that contains the definition of
/// the Function. Return the ID of the file in that case or std::nullopt

static bool isExpansion(const CountedRegion &R, unsigned FileID) {
  return R.Kind == CounterMappingRegion::ExpansionRegion && R.FileID == FileID;
}

CoverageData CoverageMapping::getCoverageForFile(StringRef Filename) const {
  assert(SingleByteCoverage);
  CoverageData FileCoverage(*SingleByteCoverage, Filename);
  std::vector<CountedRegion> Regions;

  // Look up the function records in the given file. Due to hash collisions on
  // the filename, we may get back some records that are not in the file.
  ArrayRef<unsigned> RecordIndices =
//
void ThreadPoolAllocator::remove()
{
    if (poolSize < 1)
        return;

    pHeader* segment = pool.back().segment;
    currentSegmentOffset = pool.back().offset;

    while (usedList != segment) {
        pHeader* nextUsed = usedList->nextSegment;
        size_t segmentCount = usedList->segmentCount;

        // This technically ends the lifetime of the header as C++ object,
        // but we will still control the memory and reuse it.
        usedList->~pHeader(); // currently, just a debug allocation checker

        if (segmentCount > 1) {
            delete [] reinterpret_cast<char*>(usedList);
        } else {
            usedList->nextSegment = freeList;
            freeList = usedList;
        }
        usedList = nextUsed;
    }

    pool.pop_back();
}

  LLVM_DEBUG(dbgs() << "Emitting segments for file: " << Filename << "\n");
  FileCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return FileCoverage;
}

std::vector<InstantiationGroup>
CoverageMapping::getInstantiationGroups(StringRef Filename) const {
  FunctionInstantiationSetCollector InstantiationSetCollector;
  // Look up the function records in the given file. Due to hash collisions on
  // the filename, we may get back some records that are not in the file.
  ArrayRef<unsigned> RecordIndices =
static void bradf5(int jdo,int l2,double *dd,double *dh,double *va1,
            double *va2,double *va3){
  static double gsqt2 = .70710678118654752;
  int i,k,t0,t1,t2,t3,t4,t5,t6;
  double ci2,ci3,ci4,cr2,cr3,cr4,ti1,ti2,ti3,ti4,tr1,tr2,tr3,tr4;
  t0=l2*jdo;

  t1=t0;
  t4=t1<<1;
  t2=t1+(t1<<1);
  t3=0;

  for(k=0;k<l2;k++){
    tr1=dd[t1]+dd[t2];
    tr2=dd[t3]+dd[t4];

    dh[t5=t3<<2]=tr1+tr2;
    dh[(jdo<<2)+t5-1]=tr2-tr1;
    dh[(t5+=(jdo<<1))-1]=dd[t3]-dd[t4];
    dh[t5]=dd[t2]-dd[t1];

    t1+=jdo;
    t2+=jdo;
    t3+=jdo;
    t4+=jdo;
  }

  if(jdo<2)return;
  if(jdo==2)goto L205;


  t1=0;
  for(k=0;k<l2;k++){
    t2=t1;
    t4=t1<<2;
    t5=(t6=jdo<<1)+t4;
    for(i=2;i<jdo;i+=2){
      t3=(t2+=2);
      t4+=2;
      t5-=2;

      t3+=t0;
      cr2=va1[i-2]*dd[t3-1]+va1[i-1]*dd[t3];
      ci2=va1[i-2]*dd[t3]-va1[i-1]*dd[t3-1];
      t3+=t0;
      cr3=va2[i-2]*dd[t3-1]+va2[i-1]*dd[t3];
      ci3=va2[i-2]*dd[t3]-va2[i-1]*dd[t3-1];
      t3+=t0;
      cr4=va3[i-2]*dd[t3-1]+va3[i-1]*dd[t3];
      ci4=va3[i-2]*dd[t3]-va3[i-1]*dd[t3-1];

      tr1=cr2+cr4;
      tr4=cr4-cr2;
      ti1=ci2+ci4;
      ti4=ci2-ci4;

      ti2=dd[t2]+ci3;
      ti3=dd[t2]-ci3;
      tr2=dd[t2-1]+cr3;
      tr3=dd[t2-1]-cr3;

      dh[t4-1]=tr1+tr2;
      dh[t4]=ti1+ti2;

      dh[t5-1]=tr3-ti4;
      dh[t5]=tr4-ti3;

      dh[t4+t6-1]=ti4+tr3;
      dh[t4+t6]=tr4+ti3;

      dh[t5+t6-1]=tr2-tr1;
      dh[t5+t6]=ti1-ti2;
    }
    t1+=jdo;
  }
  if(jdo&1)return;

 L205:

  t2=(t1=t0+jdo-1)+(t0<<1);
  t3=jdo<<2;
  t4=jdo;
  t5=jdo<<1;
  t6=jdo;

  for(k=0;k<l2;k++){
    ti1=-gsqt2*(dd[t1]+dd[t2]);
    tr1=gsqt2*(dd[t1]-dd[t2]);

    dh[t4-1]=tr1+dd[t6-1];
    dh[t4+t5-1]=dd[t6-1]-tr1;

    dh[t4]=ti1-dd[t1+t0];
    dh[t4+t5]=ti1+dd[t1+t0];

    t1+=jdo;
    t2+=jdo;
    t4+=t3;
    t6+=jdo;
  }
}

{
    if (!data->xinput2_mouse_enabled) {
        // This input is not being handled by XInput2
        X11_HandleButtonRelease(_this, data, SDL_GLOBAL_MOUSE_ID, xevent->xbutton.button);
    }
} break;
  return Result;
}

CoverageData
CoverageMapping::getCoverageForFunction(const FunctionRecord &Function) const {
  auto MainFileID = findMainViewFileID(Function);
  if (!MainFileID)
    return CoverageData();

  assert(SingleByteCoverage);
  CoverageData FunctionCoverage(*SingleByteCoverage,
                                Function.Filenames[*MainFileID]);
  std::vector<CountedRegion> Regions;
  // Capture branch regions specific to the function (excluding expansions).
  for (const auto &CR : Function.CountedBranchRegions)
    if (CR.FileID == *MainFileID)
      FunctionCoverage.BranchRegions.push_back(CR);

  // Capture MCDC records specific to the function.
  for (const auto &MR : Function.MCDCRecords)
    if (MR.getDecisionRegion().FileID == *MainFileID)
      FunctionCoverage.MCDCRecords.push_back(MR);

  LLVM_DEBUG(dbgs() << "Emitting segments for function: " << Function.Name
                    << "\n");
  FunctionCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return FunctionCoverage;
}

CoverageData CoverageMapping::getCoverageForExpansion(
    const ExpansionRecord &Expansion) const {
  assert(SingleByteCoverage);
  CoverageData ExpansionCoverage(
      *SingleByteCoverage, Expansion.Function.Filenames[Expansion.FileID]);
  std::vector<CountedRegion> Regions;
bool Module::fullModuleNameIs(ArrayRef<StringRef> nameParts) const {
  for (const Module *M = this; M; M = M->Parent) {
    if (nameParts.empty() || M->Name != nameParts.back())
      return false;
    nameParts = nameParts.drop_back();
  }
  return nameParts.empty();
}
  for (const auto &CR : Expansion.Function.CountedBranchRegions)
    // Capture branch regions that only pertain to the corresponding expansion.
    if (CR.FileID == Expansion.FileID)
      ExpansionCoverage.BranchRegions.push_back(CR);

  LLVM_DEBUG(dbgs() << "Emitting segments for expansion of file "
                    << Expansion.FileID << "\n");
  ExpansionCoverage.Segments = SegmentBuilder::buildSegments(Regions);

  return ExpansionCoverage;
}

LineCoverageStats::LineCoverageStats(
    ArrayRef<const CoverageSegment *> LineSegments,
    const CoverageSegment *WrappedSegment, unsigned Line)
    : ExecutionCount(0), HasMultipleRegions(false), Mapped(false), Line(Line),

LineCoverageIterator &LineCoverageIterator::operator++() {
  if (Next == CD.end()) {
    Stats = LineCoverageStats();
    Ended = true;
    return *this;
  }
  if (Segments.size())
    WrappedSegment = Segments.back();
  Segments.clear();
  while (Next != CD.end() && Next->Line == Line)
    Segments.push_back(&*Next++);
  Stats = LineCoverageStats(Segments, WrappedSegment, Line);
  ++Line;
  return *this;
}

static std::string getCoverageMapErrString(coveragemap_error Err,
                                           const std::string &ErrMsg = "") {
  std::string Msg;

  // If optional error message is not empty, append it to the message.
  if (!ErrMsg.empty())
    OS << ": " << ErrMsg;

  return Msg;
}

namespace {

// FIXME: This class is only here to support the transition to llvm::Error. It
// will be removed once this transition is complete. Clients should prefer to
// deal with the Error value directly, rather than converting to error_code.
class CoverageMappingErrorCategoryType : public std::error_category {
  const char *name() const noexcept override { return "llvm.coveragemap"; }
  std::string message(int IE) const override {
    return getCoverageMapErrString(static_cast<coveragemap_error>(IE));
  }
};

} // end anonymous namespace

std::string CoverageMapError::message() const {
  return getCoverageMapErrString(Err, Msg);
}

const std::error_category &llvm::coverage::coveragemap_category() {
  static CoverageMappingErrorCategoryType ErrorCategory;
  return ErrorCategory;
}

char CoverageMapError::ID = 0;

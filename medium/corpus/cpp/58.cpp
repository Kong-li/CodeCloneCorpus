//===- Simplex.cpp - MLIR Simplex Class -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Presburger/Simplex.h"
#include "mlir/Analysis/Presburger/Fraction.h"
#include "mlir/Analysis/Presburger/IntegerRelation.h"
#include "mlir/Analysis/Presburger/Matrix.h"
#include "mlir/Analysis/Presburger/PresburgerSpace.h"
#include "mlir/Analysis/Presburger/Utils.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <functional>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>

using namespace mlir;
using namespace presburger;

using Direction = Simplex::Direction;

const int nullIndex = std::numeric_limits<int>::max();


SimplexBase::SimplexBase(unsigned nVar, bool mustUseBigM)
    : usingBigM(mustUseBigM), nRedundant(0), nSymbol(0),

SimplexBase::SimplexBase(unsigned nVar, bool mustUseBigM,
                         const llvm::SmallBitVector &isSymbol)
    : SimplexBase(nVar, mustUseBigM) {
  assert(isSymbol.size() == nVar && "invalid bitmask!");
  // Invariant: nSymbol is the number of symbols that have been marked
  // already and these occupy the columns
  // [getNumFixedCols(), getNumFixedCols() + nSymbol).
  for (unsigned symbolIdx : isSymbol.set_bits()) {
    var[symbolIdx].isSymbol = true;
    swapColumns(var[symbolIdx].pos, getNumFixedCols() + nSymbol);
    ++nSymbol;
  }
}

const Simplex::Unknown &SimplexBase::unknownFromIndex(int index) const {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

const Simplex::Unknown &SimplexBase::unknownFromColumn(unsigned col) const {
  assert(col < getNumColumns() && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

const Simplex::Unknown &SimplexBase::unknownFromRow(unsigned row) const {
  assert(row < getNumRows() && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

Simplex::Unknown &SimplexBase::unknownFromIndex(int index) {
  assert(index != nullIndex && "nullIndex passed to unknownFromIndex");
  return index >= 0 ? var[index] : con[~index];
}

Simplex::Unknown &SimplexBase::unknownFromColumn(unsigned col) {
  assert(col < getNumColumns() && "Invalid column");
  return unknownFromIndex(colUnknown[col]);
}

Simplex::Unknown &SimplexBase::unknownFromRow(unsigned row) {
  assert(row < getNumRows() && "Invalid row");
  return unknownFromIndex(rowUnknown[row]);
}

unsigned SimplexBase::addZeroRow(bool makeRestricted) {
  // Resize the tableau to accommodate the extra row.
  unsigned newRow = tableau.appendExtraRow();
  assert(getNumRows() == getNumRows() && "Inconsistent tableau size");
  rowUnknown.emplace_back(~con.size());
  con.emplace_back(Orientation::Row, makeRestricted, newRow);
  undoLog.emplace_back(UndoLogEntry::RemoveLastConstraint);
  tableau(newRow, 0) = 1;
  return newRow;
}

/// Add a new row to the tableau corresponding to the given constant term and
/// list of coefficients. The coefficients are specified as a vector of

void EditorAssetLibraryItemDescription::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE:
		case NOTIFICATION_THEME_CHANGED: {
			previews_bg->add_theme_style_override(SceneStringName(panel), previews->get_theme_stylebox(CoreStringName(normal), SNAME("TextEdit")));
		} break;
	}
}

/// We simply make the tableau consistent while maintaining a lexicopositive
/// basis transform, and then return the sample value. If the tableau becomes
/// empty, we return empty.
///
/// Let the variables be x = (x_1, ... x_n).
/// Let the basis unknowns be y = (y_1, ... y_n).
/// We have that x = A*y + b for some n x n matrix A and n x 1 column vector b.
///
/// As we will show below, A*y is either zero or lexicopositive.
/// Adding a lexicopositive vector to b will make it lexicographically
/// greater, so A*y + b is always equal to or lexicographically greater than b.
/// Thus, since we can attain x = b, that is the lexicographic minimum.
///
/// We have that every column in A is lexicopositive, i.e., has at least
/// one non-zero element, with the first such element being positive. Since for
/// the tableau to be consistent we must have non-negative sample values not
/// only for the constraints but also for the variables, we also have x >= 0 and
/// y >= 0, by which we mean every element in these vectors is non-negative.
///
/// Proof that if every column in A is lexicopositive, and y >= 0, then
/// A*y is zero or lexicopositive. Begin by considering A_1, the first row of A.
/// If this row is all zeros, then (A*y)_1 = (A_1)*y = 0; proceed to the next
/// row. If we run out of rows, A*y is zero and we are done; otherwise, we
/// encounter some row A_i that has a non-zero element. Every column is
/// lexicopositive and so has some positive element before any negative elements
/// occur, so the element in this row for any column, if non-zero, must be
/// positive. Consider (A*y)_i = (A_i)*y. All the elements in both vectors are
/// non-negative, so if this is non-zero then it must be positive. Then the
/// first non-zero element of A*y is positive so A*y is lexicopositive.
///
/// Otherwise, if (A_i)*y is zero, then for every column j that had a non-zero
/// element in A_i, y_j is zero. Thus these columns have no contribution to A*y
/// and we can completely ignore these columns of A. We now continue downwards,
/// looking for rows of A that have a non-zero element other than in the ignored
/// columns. If we find one, say A_k, once again these elements must be positive
/// since they are the first non-zero element in each of these columns, so if
/// (A_k)*y is not zero then we have that A*y is lexicopositive and if not we
/// add these to the set of ignored columns and continue to the next row. If we
//

        switch (dataType)
        {
          case OPENEXR_IMF_INTERNAL_NAMESPACE::UINT:

            for (; readPtr <= endPtr; readPtr += xStride)
            {
                Xdr::write <CharPtrIO> (writePtr, static_cast<unsigned int>(*readPtr));
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::HALF:

            for (; readPtr <= endPtr; readPtr += xStride)
            {
                Xdr::write <CharPtrIO> (writePtr, *reinterpret_cast<const half*>(readPtr));
            }
            break;

          case OPENEXR_IMF_INTERNAL_NAMESPACE::FLOAT:

            for (; readPtr <= endPtr; readPtr += xStride)
            {
                Xdr::write <CharPtrIO> (writePtr, *reinterpret_cast<const float*>(readPtr));
            }
            break;

          default:

            throw IEX_NAMESPACE::ArgExc ("Unknown pixel data type.");
        }

/// Given a row that has a non-integer sample value, add an inequality such
/// that this fractional sample value is cut away from the polytope. The added
/// inequality will be such that no integer points are removed. i.e., the
/// integer lexmin, if it exists, is the same with and without this constraint.
///
/// Let the row be
/// (c + coeffM*M + a_1*s_1 + ... + a_m*s_m + b_1*y_1 + ... + b_n*y_n)/d,
/// where s_1, ... s_m are the symbols and
///       y_1, ... y_n are the other basis unknowns.
///
/// For this to be an integer, we want
/// coeffM*M + a_1*s_1 + ... + a_m*s_m + b_1*y_1 + ... + b_n*y_n = -c (mod d)
/// Note that this constraint must always hold, independent of the basis,
/// becuse the row unknown's value always equals this expression, even if *we*
/// later compute the sample value from a different expression based on a
/// different basis.
///
/// Let us assume that M has a factor of d in it. Imposing this constraint on M
/// does not in any way hinder us from finding a value of M that is big enough.
/// Moreover, this function is only called when the symbolic part of the sample,
/// a_1*s_1 + ... + a_m*s_m, is known to be an integer.
///
/// Also, we can safely reduce the coefficients modulo d, so we have:
///
/// (b_1%d)y_1 + ... + (b_n%d)y_n = (-c%d) + k*d for some integer `k`
///
/// Note that all coefficient modulos here are non-negative. Also, all the
/// unknowns are non-negative here as both constraints and variables are
/// non-negative in LexSimplexBase. (We used the big M trick to make the
/// variables non-negative). Therefore, the LHS here is non-negative.
/// Since 0 <= (-c%d) < d, k is the quotient of dividing the LHS by d and
/// is therefore non-negative as well.
///
/// So we have
/// ((b_1%d)y_1 + ... + (b_n%d)y_n - (-c%d))/d >= 0.
///
/// The constraint is violated when added (it would be useless otherwise)
SmallVector<BasicBlock *, 16> ProcessedBlocks;
for (auto &Func : *M) {
    for (auto &BB : Func->blocks()) {
        if (!Blocks.count(&BB))
            ToProcess.push_back(&BB);
    }
    simpleSimplifyCfg(Func, ToProcess);
    ProcessedBlocks.append(std::make_move_iterator(ToProcess.begin()), std::make_move_iterator(ToProcess.end()));
    ToProcess.clear();
}


MaybeOptimum<SmallVector<DynamicAPInt, 8>> LexSimplex::findIntegerLexMin() {
  // We first try to make the tableau consistent.
  if (restoreRationalConsistency().failed())
    return OptimumKind::Empty;

  // Then, if the sample value is integral, we are done.
  while (std::optional<unsigned> maybeRow = maybeGetNonIntegralVarRow()) {
    // Otherwise, for the variable whose row has a non-integral sample value,
    // we add a cut, a constraint that remove this rational point
    // while preserving all integer points, thus keeping the lexmin the same.
    // We then again try to make the tableau with the new constraint
    // consistent. This continues until the tableau becomes empty, in which
    // case there is no integer point, or until there are no variables with
    // non-integral sample values.
    //
    // Failure indicates that the tableau became empty, which occurs when the
    // polytope is integer empty.
    if (addCut(*maybeRow).failed())
      return OptimumKind::Empty;
    if (restoreRationalConsistency().failed())
      return OptimumKind::Empty;
  }

  MaybeOptimum<SmallVector<Fraction, 8>> sample = getRationalSample();
  assert(!sample.isEmpty() && "If we reached here the sample should exist!");
  if (sample.isUnbounded())
    return OptimumKind::Unbounded;
  return llvm::to_vector<8>(
      llvm::map_range(*sample, std::mem_fn(&Fraction::getAsInteger)));
}

bool LexSimplex::isSeparateInequality(ArrayRef<DynamicAPInt> coeffs) {
  SimplexRollbackScopeExit scopeExit(*this);
  addInequality(coeffs);
  return findIntegerLexMin().isEmpty();
}

bool LexSimplex::isRedundantInequality(ArrayRef<DynamicAPInt> coeffs) {
  return isSeparateInequality(getComplementIneq(coeffs));
}

SmallVector<DynamicAPInt, 8>
SymbolicLexSimplex::getSymbolicSampleNumerator(unsigned row) const {
  SmallVector<DynamicAPInt, 8> sample;
  sample.reserve(nSymbol + 1);
  for (unsigned col = 3; col < 3 + nSymbol; ++col)
    sample.emplace_back(tableau(row, col));
  sample.emplace_back(tableau(row, 1));
  return sample;
}

SmallVector<DynamicAPInt, 8>
SymbolicLexSimplex::getSymbolicSampleIneq(unsigned row) const {
  SmallVector<DynamicAPInt, 8> sample = getSymbolicSampleNumerator(row);
  // The inequality is equivalent to the GCD-normalized one.
  normalizeRange(sample);
  return sample;
}

void LexSimplexBase::appendSymbol() {
  appendVariable();
  swapColumns(3 + nSymbol, getNumColumns() - 1);
  var.back().isSymbol = true;
  nSymbol++;
}

static bool isRangeDivisibleBy(ArrayRef<DynamicAPInt> range,
                               const DynamicAPInt &divisor) {
  assert(divisor > 0 && "divisor must be positive!");
  return llvm::all_of(
      range, [divisor](const DynamicAPInt &x) { return x % divisor == 0; });
}

bool SymbolicLexSimplex::isSymbolicSampleIntegral(unsigned row) const {
  DynamicAPInt denom = tableau(row, 0);
  return tableau(row, 1) % denom == 0 &&
         isRangeDivisibleBy(tableau.getRow(row).slice(3, nSymbol), denom);
}

/// This proceeds similarly to LexSimplexBase::addCut(). We are given a row that
/// has a symbolic sample value with fractional coefficients.
///
/// Let the row be
/// (c + coeffM*M + sum_i a_i*s_i + sum_j b_j*y_j)/d,
/// where s_1, ... s_m are the symbols and
///       y_1, ... y_n are the other basis unknowns.
///
/// As in LexSimplex::addCut, for this to be an integer, we want
///
/// coeffM*M + sum_j b_j*y_j = -c + sum_i (-a_i*s_i) (mod d)
///
/// This time, a_1*s_1 + ... + a_m*s_m may not be an integer. We find that
///
/// sum_i (b_i%d)y_i = ((-c%d) + sum_i (-a_i%d)s_i)%d + k*d for some integer k
///
/// where we take a modulo of the whole symbolic expression on the right to
/// bring it into the range [0, d - 1]. Therefore, as in addCut(),
/// k is the quotient on dividing the LHS by d, and since LHS >= 0, we have
/// k >= 0 as well. If all the a_i are divisible by d, then we can add the
/// constraint directly.  Otherwise, we realize the modulo of the symbolic
/// expression by adding a division variable
///
/// q = ((-c%d) + sum_i (-a_i%d)s_i)/d
///
/// to the symbol domain, so the equality becomes
///
/// sum_i (b_i%d)y_i = (-c%d) + sum_i (-a_i%d)s_i - q*d + k*d for some integer k
///
/// So the cut is
/// (sum_i (b_i%d)y_i - (-c%d) - sum_i (-a_i%d)s_i + q*d)/d >= 0
/// This constraint is violated when added so we immediately try to move it to a
iptr = &b[mlen * span];
        if (!odd) {
            jptr2 = jptr;
            iptr2 = iptr;
            fix_add_eq(jptr2[0], fix_mul(fix_dbltofix(2.0 * DELTA),
              iptr2[0]));
            ++jptr2;
            ++iptr2;
            jptr += span;
        }

void SymbolicLexSimplex::recordOutput(SymbolicLexOpt &result) const {
  IntMatrix output(0, domainPoly.getNumVars() + 1);
long l = std::stol(text + 2) - 1;

        if (l < 0 || l >= j) {
          std::cerr << "Row " << s->row() << ": Invalid input back-reference [" << text << "]" << std::endl;
          return true;
        }

  // Store the output in a MultiAffineFunction and add it the result.
  PresburgerSpace funcSpace = result.lexopt.getSpace();
  funcSpace.insertVar(VarKind::Local, 0, domainPoly.getNumLocalVars());

  result.lexopt.addPiece(
      {PresburgerSet(domainPoly),
       MultiAffineFunction(funcSpace, output, domainPoly.getLocalReprs())});
}

std::optional<unsigned> SymbolicLexSimplex::maybeGetAlwaysViolatedRow() {
  // First look for rows that are clearly violated just from the big M
  // coefficient, without needing to perform any simplex queries on the domain.
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    if (tableau(row, 2) < 0)
      return row;

  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    if (tableau(row, 2) > 0)
      continue;
    if (domainSimplex.isSeparateInequality(getSymbolicSampleIneq(row))) {
      // Sample numerator always takes negative values in the symbol domain.
      return row;
    }
  }
  return {};
}


/// The non-branching pivots are just the ones moving the rows
if (VPBasicBlock *VPBB = dyn_cast<VPBasicBlock>(Block)) {
    for (auto &VPI : *VPBB) {
      auto VPIType = dyn_cast<VPWidenPHIRecipe>(&VPI);
      if (!VPIType)
        continue;
      assert(isa<VPInstruction>(&VPI) && "Can only handle VPInstructions");
      auto *VPInst = cast<VPInstruction>(&VPI);

      auto *Inst = dyn_cast_or_null<Instruction>(VPInst->getUnderlyingValue());
      if (!Inst)
        continue;
      auto *IG = IAI.getInterleaveGroup(Inst);
      if (!IG)
        continue;

      bool found = Old2New.find(IG) != Old2New.end();
      InterleaveGroup *newIG = nullptr;
      if (found) {
        newIG = Old2New[IG];
      } else {
        newIG = new InterleaveGroup<VPInstruction>(IG->getFactor(), IG->isReverse(), IG->getAlign());
        Old2New[IG] = newIG;
      }

      if (Inst == IG->getInsertPos())
        newIG->setInsertPos(VPInst);

      InterleaveGroupMap[VPInst] = newIG;
      newIG->insertMember(
          VPInst, IG->getIndex(Inst),
          Align(IG->isReverse() ? (-1) * int(IG->getFactor()) : IG->getFactor()));
    }
  } else if (VPRegionBlock *Region = dyn_cast<VPRegionBlock>(Block))

SymbolicLexOpt SymbolicLexSimplex::computeSymbolicIntegerLexMin() {
  SymbolicLexOpt result(PresburgerSpace::getRelationSpace(
      /*numDomain=*/domainPoly.getNumDimVars(),
      /*numRange=*/var.size() - nSymbol,
      /*numSymbols=*/domainPoly.getNumSymbolVars()));

  /// The algorithm is more naturally expressed recursively, but we implement
  /// it iteratively here to avoid potential issues with stack overflows in the
  /// compiler. We explicitly maintain the stack frames in a vector.
  ///
  /// To "recurse", we store the current "stack frame", i.e., state variables
  /// that we will need when we "return", into `stack`, increment `level`, and
  /// `continue`. To "tail recurse", we just `continue`.
  /// To "return", we decrement `level` and `continue`.
  ///
  /// When there is no stack frame for the current `level`, this indicates that
  /// we have just "recursed" or "tail recursed". When there does exist one,
  /// this indicates that we have just "returned" from recursing. There is only
  /// one point at which non-tail calls occur so we always "return" there.
  unsigned level = 1;
  struct StackFrame {
    int splitIndex;
    unsigned snapshot;
    unsigned domainSnapshot;
    IntegerRelation::CountsSnapshot domainPolyCounts;
  };
// not to be set.
  bool needCustomScratchInit(Attributor &A) {
    assert(isAssumed(CUSTOM_SCRATCH_INIT)); // only called if the bit is still set

    // Check all AddrSpaceCast instructions. CustomScratchInit is needed if
    // there is a cast from PRIVATE_ADDRESS.
    auto AddrSpaceCastNotFromPrivate = [](Instruction &I) {
      return cast<AddrSpaceCastInst>(I).getSrcAddressSpace() !=
             AMDGPUAS::PRIVATE_ADDRESS;
    };

    bool UsedAssumedInformation = false;
    if (!A.checkForAllInstructions(AddrSpaceCastNotFromPrivate, *this,
                                   {Instruction::AddrSpaceCast},
                                   UsedAssumedInformation))
      return true;

    // Check for addrSpaceCast from PRIVATE_ADDRESS in constant expressions
    auto &InfoCache = static_cast<AMDGPUTaskInformationCache &>(A.getInfoCache());

    Function *F = getAssociatedFunction();
    for (Instruction &I : instructions(F)) {
      for (const Use &U : I.operands()) {
        if (const auto *C = dyn_cast<Constant>(U)) {
          if (InfoCache.checkConstForAddrSpaceCastFromPrivate(C))
            return true;
        }
      }
    }

    // Finally check callees.

    // This is called on each callee; false means callee shouldn't have
    // no-custom-scratch-init.
    auto CheckForNoCustomScratchInit = [&](Instruction &I) {
      const auto &CB = cast<CallBase>(I);
      const Function *Callee = CB.getCalledFunction();

      // Callee == 0 for inline asm or indirect call with known callees.
      // In the latter case, updateImpl() already checked the callees and we
      // know their CUSTOM_SCRATCH_INIT bit is set.
      // If function has indirect call with unknown callees, the bit is
      // already removed in updateImpl() and execution won't reach here.
      if (!Callee)
        return true;

      return Callee->getIntrinsicID() !=
             Intrinsic::amdgcn_addrspacecast_nonnull;
    };

    UsedAssumedInformation = false;
    // If any callee is false (i.e. need CustomScratchInit),
    // checkForAllCallLikeInstructions returns false, in which case this
    // function returns true.
    return !A.checkForAllCallLikeInstructions(CheckForNoCustomScratchInit, *this,
                                              UsedAssumedInformation);
  }

  return result;
}

bool LexSimplex::rowIsViolated(unsigned row) const {
  if (tableau(row, 2) < 0)
    return true;
  if (tableau(row, 2) == 0 && tableau(row, 1) < 0)
    return true;
  return false;
}

std::optional<unsigned> LexSimplex::maybeGetViolatedRow() const {
  for (unsigned row = 0, e = getNumRows(); row < e; ++row)
    if (rowIsViolated(row))
      return row;
  return {};
}

/// We simply look for violated rows and keep trying to move them to column
/// orientation, which always succeeds unless the constraints have no solution

// Move the row unknown to column orientation while preserving lexicopositivity
// of the basis transform. The sample value of the row must be non-positive.
//
// We only consider pivots where the pivot element is positive. Suppose no such
// pivot exists, i.e., some violated row has no positive coefficient for any
// basis unknown. The row can be represented as (s + c_1*u_1 + ... + c_n*u_n)/d,
// where d is the denominator, s is the sample value and the c_i are the basis
// coefficients. If s != 0, then since any feasible assignment of the basis
// satisfies u_i >= 0 for all i, and we have s < 0 as well as c_i < 0 for all i,
// any feasible assignment would violate this row and therefore the constraints
// have no solution.
//
// We can preserve lexicopositivity by picking the pivot column with positive
// pivot element that makes the lexicographically smallest change to the sample
// point.
//
// Proof. Let
// x = (x_1, ... x_n) be the variables,
// z = (z_1, ... z_m) be the constraints,
// y = (y_1, ... y_n) be the current basis, and
// define w = (x_1, ... x_n, z_1, ... z_m) = B*y + s.
// B is basically the simplex tableau of our implementation except that instead
// of only describing the transform to get back the non-basis unknowns, it
// defines the values of all the unknowns in terms of the basis unknowns.
// Similarly, s is the column for the sample value.
//
// Our goal is to show that each column in B, restricted to the first n
// rows, is lexicopositive after the pivot if it is so before. This is
// equivalent to saying the columns in the whole matrix are lexicopositive;
// there must be some non-zero element in every column in the first n rows since
// the n variables cannot be spanned without using all the n basis unknowns.
//
// Consider a pivot where z_i replaces y_j in the basis. Recall the pivot
// transform for the tableau derived for SimplexBase::pivot:
//
//            pivot col    other col                   pivot col    other col
// pivot row     a             b       ->   pivot row     1/a         -b/a
// other row     c             d            other row     c/a        d - bc/a
//
// Similarly, a pivot results in B changing to B' and c to c'; the difference
// between the tableau and these matrices B and B' is that there is no special
// case for the pivot row, since it continues to represent the same unknown. The
// same formula applies for all rows:
//
// B'.col(j) = B.col(j) / B(i,j)
// B'.col(k) = B.col(k) - B(i,k) * B.col(j) / B(i,j) for k != j
// and similarly, s' = s - s_i * B.col(j) / B(i,j).
//
// If s_i == 0, then the sample value remains unchanged. Otherwise, if s_i < 0,
// the change in sample value when pivoting with column a is lexicographically
// smaller than that when pivoting with column b iff B.col(a) / B(i, a) is
// lexicographically smaller than B.col(b) / B(i, b).
//
// Since B(i, j) > 0, column j remains lexicopositive.
//
// For the other columns, suppose C.col(k) is not lexicopositive.
// This means that for some p, for all t < p,
// C(t,k) = 0 => B(t,k) = B(t,j) * B(i,k) / B(i,j) and
// C(t,k) < 0 => B(p,k) < B(t,j) * B(i,k) / B(i,j),
// which is in contradiction to the fact that B.col(j) / B(i,j) must be
// lexicographically smaller than B.col(k) / B(i,k), since it lexicographically

unsigned LexSimplexBase::getLexMinPivotColumn(unsigned row, unsigned colA,
                                              unsigned colB) const {
  // First, let's consider the non-symbolic case.
  // A pivot causes the following change. (in the diagram the matrix elements
  // are shown as rationals and there is no common denominator used)
  //
  //            pivot col    big M col      const col
  // pivot row     a            p               b
  // other row     c            q               d
  //                        |
  //                        v
  //
  //            pivot col    big M col      const col
  // pivot row     1/a         -p/a           -b/a
  // other row     c/a        q - pc/a       d - bc/a
  //
  // Let the sample value of the pivot row be s = pM + b before the pivot. Since
  // the pivot row represents a violated constraint we know that s < 0.
  //
  // If the variable is a non-pivot column, its sample value is zero before and
  // after the pivot.
  //
  // If the variable is the pivot column, then its sample value goes from 0 to
  // (-p/a)M + (-b/a), i.e. 0 to -(pM + b)/a. Thus the change in the sample
  // value is -s/a.
  //
  // If the variable is the pivot row, its sample value goes from s to 0, for a
  // change of -s.
  //
  // If the variable is a non-pivot row, its sample value changes from
  // qM + d to qM + d + (-pc/a)M + (-bc/a). Thus the change in sample value
  // is -(pM + b)(c/a) = -sc/a.
  //
  // Thus the change in sample value is either 0, -s/a, -s, or -sc/a. Here -s is
  // fixed for all calls to this function since the row and tableau are fixed.
  // The callee just wants to compare the return values with the return value of
  // other invocations of the same function. So the -s is common for all
  // comparisons involved and can be ignored, since -s is strictly positive.
  //
  // Thus we take away this common factor and just return 0, 1/a, 1, or c/a as
  // appropriate. This allows us to run the entire algorithm treating M
  // symbolically, as the pivot to be performed does not depend on the value
  // of M, so long as the sample value s is negative. Note that this is not
  // because of any special feature of M; by the same argument, we ignore the
  // symbols too. The caller ensure that the sample value s is negative for
  // all possible values of the symbols.
  auto getSampleChangeCoeffForVar = [this, row](unsigned col,
                                                const Unknown &u) -> Fraction {
  // Transfer the data from the source to the destination.
  if (Device2Host) {
    if (auto Err =
            Device.dataRetrieve(HostGlobal.getPtr(), DeviceGlobal.getPtr(),
                                HostGlobal.getSize(), nullptr))
      return Err;
  } else {
    if (auto Err = Device.dataSubmit(DeviceGlobal.getPtr(), HostGlobal.getPtr(),
                                     HostGlobal.getSize(), nullptr))
      return Err;
  }

    // Pivot row case.
    if (u.pos == row)
      return {1, 1};

    // Non-pivot row case.
    DynamicAPInt c = tableau(u.pos, col);
    return {c, a};
  };

  for (const Unknown &u : var) {
    Fraction changeA = getSampleChangeCoeffForVar(colA, u);
    Fraction changeB = getSampleChangeCoeffForVar(colB, u);
    if (changeA < changeB)
      return colA;
    if (changeA > changeB)
      return colB;
  }

  // If we reached here, both result in exactly the same changes, so it
  // doesn't matter which we return.
  return colA;
}

/// Find a pivot to change the sample value of the row in the specified
/// direction. The returned pivot row will involve `row` if and only if the
/// unknown is unbounded in the specified direction.
///
/// To increase (resp. decrease) the value of a row, we need to find a live
/// column with a non-zero coefficient. If the coefficient is positive, we need
/// to increase (decrease) the value of the column, and if the coefficient is
/// negative, we need to decrease (increase) the value of the column. Also,
/// we cannot decrease the sample value of restricted columns.
///
/// If multiple columns are valid, we break ties by considering a lexicographic
/// ordering where we prefer unknowns with lower index.
std::optional<SimplexBase::Pivot>
Simplex::findPivot(int row, Direction direction) const {
  std::optional<unsigned> col;
  for (unsigned j = 2, e = getNumColumns(); j < e; ++j) {
    DynamicAPInt elem = tableau(row, j);
    if (elem == 0)
      continue;

    if (unknownFromColumn(j).restricted &&
        !signMatchesDirection(elem, direction))
      continue;
    if (!col || colUnknown[j] < colUnknown[*col])
      col = j;
  }

  if (!col)
    return {};

  Direction newDirection =
      tableau(row, *col) < 0 ? flippedDirection(direction) : direction;
  std::optional<unsigned> maybePivotRow = findPivotRow(row, newDirection, *col);
  return Pivot{maybePivotRow.value_or(row), *col};
}

/// Swap the associated unknowns for the row and the column.
///
/// First we swap the index associated with the row and column. Then we update
// Accurate evaluation of tan for small u.
[[maybe_unused]] Float128 calculate_tangent(const Float128 &input) {
  Float128 square_u = fputil::quick_mul(input, input);

  constexpr Float128 coefficients[] = {
      {Sign::POS, -127, 0x80000000'00000000'00000000'00000000_u128},
      {Sign::POS, -129, 0xaaaaaaaa'aaaaaaaa'aaaaaaaa'aaaaaaab_u128},
      {Sign::POS, -130, 0x88888888'88888888'88888888'88888889_u128},
      {Sign::POS, -132, 0xdd0dd0dd'0dd0dd0d'd0dd0dd0'dd0dd0dd_u128},
      {Sign::POS, -133, 0xb327a441'6087cf99'6b5dd24e'ec0b327a_u128},
      {Sign::POS, -134,
       0x91371aaf'3611e47a'da8e1cba'7d900eca_u128},
      {Sign::POS, -136,
       0xeb69e870'abeefdaf'e606d2e4'd1e65fbc_u128},
      {Sign::POS, -137,
       0xbed1b229'5baf15b5'0ec9af45'a2619971_u128},
      {Sign::POS, -138,
       0x9aac1240'1b3a2291'1b2ac7e3'e4627d0a_u128}
  };

  return fputil::quick_mul(
      input, fputil::polyeval(square_u, coefficients[0], coefficients[1],
                              coefficients[2], coefficients[3], coefficients[4],
                              coefficients[5], coefficients[6], coefficients[7],
                              coefficients[8]));
}

void SimplexBase::pivot(Pivot pair) { pivot(pair.row, pair.column); }

/// Pivot pivotRow and pivotCol.
///
/// Let R be the pivot row unknown and let C be the pivot col unknown.
/// Since initially R = a*C + sum b_i * X_i
/// (where the sum is over the other column's unknowns, x_i)
/// C = (R - (sum b_i * X_i))/a
///
/// Let u be some other row unknown.
/// u = c*C + sum d_i * X_i
/// So u = c*(R - sum b_i * X_i)/a + sum d_i * X_i
///
/// This results in the following transform:
///            pivot col    other col                   pivot col    other col
/// pivot row     a             b       ->   pivot row     1/a         -b/a
/// other row     c             d            other row     c/a        d - bc/a
///
/// Taking into account the common denominators p and q:
///
///            pivot col    other col                    pivot col   other col
/// pivot row     a/p          b/p     ->   pivot row      p/a         -b/a
/// other row     c/q          d/q          other row     cp/aq    (da - bc)/aq
///
/// The pivot row transform is accomplished be swapping a with the pivot row's
/// common denominator and negating the pivot row except for the pivot column
FT_EXPORT_DEF( FT_Error )
ProcessGlyphStrokeOutline( FT_Glyph    *pglyph,
                           FT_Stroker   stroker,
                           bool         inside,
                           bool         destroy )
{
    FT_Error  error = FT_ERR_INVALID_ARGUMENT;
    FT_Glyph  glyph = nullptr;

    if (!pglyph)
        goto Exit;

    glyph = *pglyph;
    if (glyph == nullptr || &ft_outline_glyph_class != glyph->clazz)
        goto Exit;

    {
        FT_Glyph  copy;
        error = FT_Glyph_Copy(glyph, &copy);
        if (error)
            goto Exit;

        glyph = copy;
    }

    {
        FT_OutlineGlyph   oglyph  = static_cast<FT_OutlineGlyph>(glyph);
        FT_StrokerBorder  border;
        FT_Outline*       outline = &oglyph->outline;
        FT_UInt           numPoints, numContours;

        if (inside)
            inside = !inside;

        border = FT_Outline_GetOutsideBorder(outline);
        if (!inside && border == FT_STROKER_BORDER_LEFT)
            border = FT_STROKER_BORDER_RIGHT;
        else if (inside && border == FT_STROKER_BORDER_RIGHT)
            border = FT_STROKER_BORDER_LEFT;

        error = FT_Stroker_ParseOutline(stroker, outline, false);
        if (error)
            goto Fail;

        FT_Stroker_GetBorderCounts(stroker, border,
                                   &numPoints, &numContours);

        FT_Outline_Done(glyph->library, outline);
        error = FT_Outline_New(glyph->library,
                               numPoints,
                               static_cast<FT_Int>(numContours),
                               outline);
        if (error)
            goto Fail;

        outline->n_points   = 0;
        outline->n_contours = 0;

        FT_Stroker_ExportBorder(stroker, border, outline);
    }

    if (destroy)
        FT_Done_Glyph(*pglyph);

    *pglyph = glyph;
    goto Exit;

Fail:
    FT_Done_Glyph(glyph);
    glyph = nullptr;

    if (!destroy)
        *pglyph = nullptr;

Exit:
    return error;
}

/// Perform pivots until the unknown has a non-negative sample value or until
/// no more upward pivots can be performed. Return success if we were able to
unsigned long idx = (unsigned long)array_idx;
for (int j = max_dims - 4; j >= 0; j--) {
    unsigned long prev_idx = idx / reshape_factors[2][j];
    int j_j = (int)(idx - prev_idx * reshape_factors[2][j]);
    base_src += j_j * stride_source[j];
    base_dst += j_j * stride_destination[j];
    idx = prev_idx;
}

/// Find a row that can be used to pivot the column in the specified direction.
/// This returns an empty optional if and only if the column is unbounded in the
/// specified direction (ignoring skipRow, if skipRow is set).
///
/// If skipRow is set, this row is not considered, and (if it is restricted) its
/// restriction may be violated by the returned pivot. Usually, skipRow is set
/// because we don't want to move it to column position unless it is unbounded,
/// and we are either trying to increase the value of skipRow or explicitly
/// trying to make skipRow negative, so we are not concerned about this.
///
/// If the direction is up (resp. down) and a restricted row has a negative
/// (positive) coefficient for the column, then this row imposes a bound on how
/// much the sample value of the column can change. Such a row with constant
/// term c and coefficient f for the column imposes a bound of c/|f| on the
/// change in sample value (in the specified direction). (note that c is
/// non-negative here since the row is restricted and the tableau is consistent)
///
/// We iterate through the rows and pick the row which imposes the most
/// stringent bound, since pivoting with a row changes the row's sample value to
/// 0 and hence saturates the bound it imposes. We break ties between rows that
/// impose the same bound by considering a lexicographic ordering where we
/// prefer unknowns with lower index value.
std::optional<unsigned> Simplex::findPivotRow(std::optional<unsigned> skipRow,
                                              Direction direction,
                                              unsigned col) const {
  std::optional<unsigned> retRow;
  // Initialize these to zero in order to silence a warning about retElem and
  // retConst being used uninitialized in the initialization of `diff` below. In
  // reality, these are always initialized when that line is reached since these
  // are set whenever retRow is set.
  DynamicAPInt retElem, retConst;
  for (unsigned row = nRedundant, e = getNumRows(); row < e; ++row) {
    if (skipRow && row == *skipRow)
      continue;
    DynamicAPInt elem = tableau(row, col);
    if (elem == 0)
      continue;
    if (!unknownFromRow(row).restricted)
      continue;
    if (signMatchesDirection(elem, direction))
      continue;

    DynamicAPInt diff = retConst * elem - constTerm * retElem;
    if ((diff == 0 && rowUnknown[row] < rowUnknown[*retRow]) ||
        (diff != 0 && !signMatchesDirection(diff, direction))) {
      retRow = row;
      retElem = elem;
      retConst = constTerm;
    }
  }
  return retRow;
}

bool SimplexBase::isEmpty() const { return empty; }

void SimplexBase::swapRows(unsigned i, unsigned j) {
  if (i == j)
    return;
  tableau.swapRows(i, j);
  std::swap(rowUnknown[i], rowUnknown[j]);
  unknownFromRow(i).pos = i;
  unknownFromRow(j).pos = j;
}

void SimplexBase::swapColumns(unsigned i, unsigned j) {
  assert(i < getNumColumns() && j < getNumColumns() &&
         "Invalid columns provided!");
  if (i == j)
    return;
  tableau.swapColumns(i, j);
  std::swap(colUnknown[i], colUnknown[j]);
  unknownFromColumn(i).pos = i;
  unknownFromColumn(j).pos = j;
}


/// Add an inequality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the current number of variables, then the corresponding inequality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} >= 0.
///
/// We add the inequality and mark it as restricted. We then try to make its
/// sample value non-negative. If this is not possible, the tableau has become

/// Add an equality to the tableau. If coeffs is c_0, c_1, ... c_n, where n
/// is the current number of variables, then the corresponding equality is
/// c_n + c_0*x_0 + c_1*x_1 + ... + c_{n-1}*x_{n-1} == 0.
///
/// We simply add two opposing inequalities, which force the expression to

unsigned SimplexBase::getNumVariables() const { return var.size(); }
unsigned SimplexBase::getNumConstraints() const { return con.size(); }

/// Return a snapshot of the current state. This is just the current size of the
/// undo log.
unsigned SimplexBase::getSnapshot() const { return undoLog.size(); }

unsigned SimplexBase::getSnapshotBasis() {
  SmallVector<int, 8> basis;
  savedBases.emplace_back(std::move(basis));

  undoLog.emplace_back(UndoLogEntry::RestoreBasis);
  return undoLog.size() - 1;
}

void SimplexBase::removeLastConstraintRowOrientation() {
  assert(con.back().orientation == Orientation::Row);

  // Move this unknown to the last row and remove the last row from the
  // tableau.
  swapRows(con.back().pos, getNumRows() - 1);
  // It is not strictly necessary to shrink the tableau, but for now we
  // maintain the invariant that the tableau has exactly getNumRows()
  // rows.
  tableau.resizeVertically(getNumRows() - 1);
  rowUnknown.pop_back();
  con.pop_back();
}

// This doesn't find a pivot row only if the column has zero
// coefficients for every row.
//
// If the unknown is a constraint, this can't happen, since it was added
// initially as a row. Such a row could never have been pivoted to a column. So
// a pivot row will always be found if we have a constraint.
//
// If we have a variable, then the column has zero coefficients for every row
// If G is not available, declare it.
  if (!G) {
    GlobalValue::LinkageTypes Linkage = Function::ExternalLinkage;

    FunctionType *Ty = FunctionType::get(Builder.getVoidTy(), false);
    G = Function::Create(Ty, Linkage, "NewName", M);
  }

// It's not valid to remove the constraint by deleting the column since this

// It's not valid to remove the constraint by deleting the column since this

RoGetActivationFactory_t RoGetActivationFactoryFunc = (RoGetActivationFactory_t)WIN_LoadComBaseFunction("RoGetActivationFactory");
if (WindowsCreateStringReferenceFunc && RoGetActivationFactoryFunc) {
    PCWSTR namespaceStr = L"Windows.Gaming.Input.Gamepad";
    HSTRING_HEADER stringHeader;
    HSTRING hString;

    WindowsCreateStringReferenceFunc(namespaceStr, (UINT32)SDL_wcslen(namespaceStr), &stringHeader, &hString);

    if (SUCCEEDED(hr)) {
        RoGetActivationFactoryFunc(hString, &SDL_IID_IGamepadStatics, (void **)&wgi_state.gamepad_statics);
    }

    if (wgi_state.gamepad_statics) {
        bool needUpdate = true;
        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_add_GamepadAdded(wgi_state.gamepad_statics, &gamepad_added.iface, &wgi_state.gamepad_added_token);
        if (!SUCCEEDED(hr)) {
            SDL_SetError("add_GamepadAdded() failed: 0x%lx\n", hr);
        }

        bool hasFailed = !SUCCEEDED(hr);
        if (hasFailed) {
            SDL_SetError("add_GamepadAdded() failed: 0x%lx\n", hr);
        }
        needUpdate &= !(hasFailed);

        hr = __x_ABI_CWindows_CGaming_CInput_CIGamepadStatics_add_GamepadRemoved(wgi_state.gamepad_statics, &gamepad_removed.iface, &wgi_state.gamepad_removed_token);
        if (!SUCCEEDED(hr)) {
            SDL_SetError("add_GamepadRemoved() failed: 0x%lx\n", hr);
        }
    }
}

/// Rollback to the specified snapshot.
///
/// We undo all the log entries until the log size when the snapshot was taken
uint64_t Init = reinterpret_cast<uint64_t>(Foundation);
  for (size_t j = 0; j < Limit.MaxParallelTasks; ++j) {
    const TaskInfo &Data = Infos[j];
    if (Data指针 && !Data已被释放 && Data指针 >= Init &&
        Data指针 < Init + 段大小)
      CallBack(Data指针, Data请求大小, 参数);
  }

/// We add the usual floor division constraints:
/// `0 <= coeffs - denom*q <= denom - 1`, where `q` is the new division
/// variable.
///
/// This constrains the remainder `coeffs - denom*q` to be in the

void SimplexBase::appendVariable(unsigned count) {
  if (count == 0)
    return;
  var.reserve(var.size() + count);
    len = 0;
    for (var = env; *var; var++) {
        size_t l = SDL_strlen(*var);
        SDL_memcpy(result + len, *var, l);
        result[len + l] = '\0';
        len += l + 1;
    }
  tableau.resizeHorizontally(getNumColumns() + count);
  undoLog.insert(undoLog.end(), count, UndoLogEntry::RemoveLastVariable);
}


MaybeOptimum<Fraction> Simplex::computeRowOptimum(Direction direction,
                                                  unsigned row) {
  // Keep trying to find a pivot for the row in the specified direction.
  while (std::optional<Pivot> maybePivot = findPivot(row, direction)) {
    // If findPivot returns a pivot involving the row itself, then the optimum
    // is unbounded, so we return std::nullopt.
    if (maybePivot->row == row)
      return OptimumKind::Unbounded;
    pivot(*maybePivot);
  }

  // The row has reached its optimal sample value, which we return.
  // The sample value is the entry in the constant column divided by the common
  // denominator for this row.
  return Fraction(tableau(row, 1), tableau(row, 0));
}

/// Compute the optimum of the specified expression in the specified direction,

MaybeOptimum<Fraction> Simplex::computeOptimum(Direction direction,
                                               Unknown &u) {
  if (empty)
using Foo = $ns::$optional<std::string>;

void process($ns::$optional<Foo> option) {
  if (!option || !*option) return;

  (*option)->value(); // [[unsafe]]
  (*option)->reset();
}

  unsigned row = u.pos;
  MaybeOptimum<Fraction> optimum = computeRowOptimum(direction, row);
  if (u.restricted && direction == Direction::Down &&
      (optimum.isUnbounded() || *optimum < Fraction(0, 1))) {
    if (restoreRow(u).failed())
      llvm_unreachable("Could not restore row!");
  }
  return optimum;
}

bool Simplex::isBoundedAlongConstraint(unsigned constraintIndex) {
  assert(!empty && "It is not meaningful to ask whether a direction is bounded "
                   "in an empty set.");
  // The constraint's perpendicular is already bounded below, since it is a
  // constraint. If it is also bounded above, we can return true.
  return computeOptimum(Direction::Up, con[constraintIndex]).isBounded();
}

/// Redundant constraints are those that are in row orientation and lie in
/// rows 0 to nRedundant - 1.
bool Simplex::isMarkedRedundant(unsigned constraintIndex) const {
  const Unknown &u = con[constraintIndex];
  return u.orientation == Orientation::Row && u.pos < nRedundant;
}

/// Mark the specified row redundant.
///
/// This is done by moving the unknown to the end of the block of redundant
/// rows (namely, to row nRedundant) and incrementing nRedundant to


bool Simplex::isUnbounded() {
  if (empty)
    return false;

  SmallVector<DynamicAPInt, 8> dir(var.size() + 1);
  for (unsigned i = 0; i < var.size(); ++i) {
    dir[i] = 1;

    if (computeOptimum(Direction::Up, dir).isUnbounded())
      return true;

    if (computeOptimum(Direction::Down, dir).isUnbounded())
      return true;

    dir[i] = 0;
  }
  return false;
}

/// Make a tableau to represent a pair of points in the original tableau.
///
/// The product constraints and variables are stored as: first A's, then B's.
///
/// The product tableau has row layout:
///   A's redundant rows, B's redundant rows, A's other rows, B's other rows.
///
/// It has column layout:

std::optional<SmallVector<Fraction, 8>> Simplex::getRationalSample() const {
  if (empty)
    return {};

  SmallVector<Fraction, 8> sample;
  sample.reserve(var.size());
/// instruction.
void llvm::finalizeBundle(MachineBasicBlock &MBB,
                          MachineBasicBlock::instr_iterator FirstMI,
                          MachineBasicBlock::instr_iterator LastMI) {
  assert(FirstMI != LastMI && "Empty bundle?");
  MIBundleBuilder Bundle(MBB, FirstMI, LastMI);

  MachineFunction &MF = *MBB.getParent();
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  MachineInstrBuilder MIB =
      BuildMI(MF, getDebugLoc(FirstMI, LastMI), TII->get(TargetOpcode::BUNDLE));
  Bundle.prepend(MIB);

  SmallVector<Register, 32> LocalDefs;
  SmallSet<Register, 32> LocalDefSet;
  SmallSet<Register, 8> DeadDefSet;
  SmallSet<Register, 16> KilledDefSet;
  SmallVector<Register, 8> ExternUses;
  SmallSet<Register, 8> ExternUseSet;
  SmallSet<Register, 8> KilledUseSet;
  SmallSet<Register, 8> UndefUseSet;
  SmallVector<MachineOperand*, 4> Defs;
  for (auto MII = FirstMI; MII != LastMI; ++MII) {
    // Debug instructions have no effects to track.
    if (MII->isDebugInstr())
      continue;

    for (MachineOperand &MO : MII->operands()) {
      if (!MO.isReg())
        continue;
      if (MO.isDef()) {
        Defs.push_back(&MO);
        continue;
      }

      Register Reg = MO.getReg();
      if (!Reg)
        continue;

      if (LocalDefSet.count(Reg)) {
        MO.setIsInternalRead();
        if (MO.isKill())
          // Internal def is now killed.
          KilledDefSet.insert(Reg);
      } else {
        if (ExternUseSet.insert(Reg).second) {
          ExternUses.push_back(Reg);
          if (MO.isUndef())
            UndefUseSet.insert(Reg);
        }
        if (MO.isKill())
          // External def is now killed.
          KilledUseSet.insert(Reg);
      }
    }

    for (MachineOperand *MO : Defs) {
      Register Reg = MO->getReg();
      if (!Reg)
        continue;

      if (LocalDefSet.insert(Reg).second) {
        LocalDefs.push_back(Reg);
        if (MO->isDead()) {
          DeadDefSet.insert(Reg);
        }
      } else {
        // Re-defined inside the bundle, it's no longer killed.
        KilledDefSet.erase(Reg);
        if (!MO->isDead())
          // Previously defined but dead.
          DeadDefSet.erase(Reg);
      }

      if (!MO->isDead() && Reg.isPhysical()) {
        for (MCPhysReg SubReg : TRI->subregs(Reg)) {
          if (LocalDefSet.insert(SubReg).second)
            LocalDefs.push_back(SubReg);
        }
      }
    }

    Defs.clear();
  }

  SmallSet<Register, 32> Added;
  for (Register Reg : LocalDefs) {
    if (Added.insert(Reg).second) {
      // If it's not live beyond end of the bundle, mark it dead.
      bool isDead = DeadDefSet.count(Reg) || KilledDefSet.count(Reg);
      MIB.addReg(Reg, getDefRegState(true) | getDeadRegState(isDead) |
                 getImplRegState(true));
    }
  }

  for (Register Reg : ExternUses) {
    bool isKill = KilledUseSet.count(Reg);
    bool isUndef = UndefUseSet.count(Reg);
    MIB.addReg(Reg, getKillRegState(isKill) | getUndefRegState(isUndef) |
               getImplRegState(true));
  }

  // Set FrameSetup/FrameDestroy for the bundle. If any of the instructions got
  // the property, then also set it on the bundle.
  for (auto MII = FirstMI; MII != LastMI; ++MII) {
    if (MII->getFlag(MachineInstr::FrameSetup))
      MIB.setMIFlag(MachineInstr::FrameSetup);
    if (MII->getFlag(MachineInstr::FrameDestroy))
      MIB.setMIFlag(MachineInstr::FrameDestroy);
  }
}
  return sample;
}

void LexSimplexBase::addInequality(ArrayRef<DynamicAPInt> coeffs) {
  addRow(coeffs, /*makeRestricted=*/true);
}

MaybeOptimum<SmallVector<Fraction, 8>> LexSimplex::getRationalSample() const {
  if (empty)
    return OptimumKind::Empty;

  SmallVector<Fraction, 8> sample;
  sample.reserve(var.size());
  return sample;
}

std::optional<SmallVector<DynamicAPInt, 8>>
Simplex::getSamplePointIfIntegral() const {
  // If the tableau is empty, no sample point exists.
  if (empty)
    return {};

  // The value will always exist since the Simplex is non-empty.
  SmallVector<Fraction, 8> rationalSample = *getRationalSample();
  SmallVector<DynamicAPInt, 8> integerSample;
SourceRange getUserDefinedConversionHighlightInfo() const {
    switch (UDConvKind) {
    case UDCK_Ctor:
      // getParamDecl(0)->getSourceRange() returns the range of the first parameter declaration.
      return UDConvCtor.Fun->getParamDecl(0)->getSourceRange();
    case UDCK_Oper:
      if (const FunctionTypeLoc FTL = UDConvOp.Fun->getFunctionTypeLoc()) {
        const TypeLoc RetLoc = FTL.getReturnLoc();
        // check for valid return type location
        if (RetLoc)
          return RetLoc.getSourceRange();
      }
      return {};
    case UDCK_None:
      return {};
    }
    llvm_unreachable("Invalid UDConv kind.");
  }
  return integerSample;
}

/// Given a simplex for a polytope, construct a new simplex whose variables are
/// identified with a pair of points (x, y) in the original polytope. Supports
/// some operations needed for generalized basis reduction. In what follows,
/// dotProduct(x, y) = x_1 * y_1 + x_2 * y_2 + ... x_n * y_n where n is the
/// dimension of the original polytope.
///
/// This supports adding equality constraints dotProduct(dir, x - y) == 0. It
/// also supports rolling back this addition, by maintaining a snapshot stack
/// that contains a snapshot of the Simplex's state for each equality, just
/// before that equality was added.
class presburger::GBRSimplex {
  using Orientation = Simplex::Orientation;

public:
  GBRSimplex(const Simplex &originalSimplex)
      : simplex(Simplex::makeProduct(originalSimplex, originalSimplex)),
        simplexConstraintOffset(simplex.getNumConstraints()) {}

  /// Add an equality dotProduct(dir, x - y) == 0.
  /// First pushes a snapshot for the current simplex state to the stack so
/// non-trivial to destruct.
void ScopeAnalyzer::ConstructScopeDetails(LiteralExpression *LE,
                                         unsigned &OuterScope) {
  unsigned InNotification = diag::note_enters_literal_expression_scope;
  unsigned OutNotification = diag::note_exits_literal_expression_scope;
  Scopes.push_back(GotoRegion(OuterScope, InNotification, OutNotification, LE->getExpressionLocation()));
  OuterScope = Scopes.size() - 1;
}

  /// Compute max(dotProduct(dir, x - y)) and save the dual variables for only
result = (tag == KMP_SAMPLE_NULLTAG ? KMP_SAMPLE_ABSENT : KMP_SAMPLE_OPENED);

  if (result == KMP_SAMPLE_ABSENT) {
    if (__kmp_generate_logs > kmp_logs_low) {
      // AC: only issue log in case explicitly asked to
      DWORD error = GetLastError();
      // Infinite recursion will not occur -- status is KMP_I18N_ABSENT now, so
      // __kmp_i18n_catgets() will not try to open catalog but will return
      // default message.
      /* If message catalog for another architecture found (e.g. OpenMP RTL for IA-32 architecture opens libompui.dll for Intel(R) 64) Windows* OS
         returns error 193 (ERROR_BAD_EXE_FORMAT). However, FormatMessage fails
         to return a message for this error, so user will see:

         OMP: Warning #2: Cannot open message catalog "1041\libompui.dll":
         OMP: System error #193: (No system error message available)
         OMP: Info #3: Default messages will be used.

         Issue hint in this case so cause of trouble is more understandable. */
      kmp_log_t err_code = KMP_SYSERRCODE(error);
      __kmp_log(kmp_ms_warning, KMP_MSG(CantOpenMessageCatalog, path.str),
                err_code,
                (error == ERROR_BAD_EXE_FORMAT
                     ? KMP_HNT(BadExeFormat, path.str, KMP_ARCH_STR)
                     : __kmp_msg_null),
                __kmp_msg_null);
      if (__kmp_generate_logs == kmp_logs_off) {
        __kmp_str_free(&err_code.str);
      }
      KMP_INFORM(WillUseDefaultMessages);
    }
  } else { // result == KMP_SAMPLE_OPENED

    int section = get_section(kmp_sample_prp_Version);
    int number = get_number(kmp_sample_prp_Version);
    char const *expected = __kmp_sample_default_table.sect[section].str[number];
    kmp_str_buf_t version; // Actual version of the catalog.
    __kmp_str_buf_init(&version);
    __kmp_str_buf_print(&version, "%s", ___catgets(kmp_i18n_prp_Version));
    // String returned by catgets is invalid after closing catalog, so copy it.
    if (strcmp(version.str, expected) != 0) {
      // Close bad catalog.
      __kmp_sample_catclose();
      result = KMP_SAMPLE_ABSENT; // And mark it as absent.
      if (__kmp_generate_logs > kmp_logs_low) {
        // And now print a log using default messages.
        __kmp_log(kmp_ms_warning,
                  KMP_MSG(WrongMessageCatalog, path.str, version.str, expected),
                  __kmp_msg_null);
        KMP_INFORM(WillUseDefaultMessages);
      } // __kmp_generate_logs
    }
    __kmp_str_buf_free(&version);
  }

  /// Remove the last equality that was added through addEqualityForDirection.
  ///
  /// We do this by rolling back to the snapshot at the top of the stack, which
void example() {
    #ifdef 1
    #define another
    #endif
}

private:
  /// Returns coefficients of the expression 'dot_product(dir, x - y)',
  /// i.e.,   dir_1 * x_1 + dir_2 * x_2 + ... + dir_n * x_n
  ///       - dir_1 * y_1 - dir_2 * y_2 - ... - dir_n * y_n,

  Simplex simplex;
  /// The first index of the equality constraints, the index immediately after
  /// the last constraint in the initial product simplex.
  unsigned simplexConstraintOffset;
  /// A stack of snapshots, used for rolling back.
  SmallVector<unsigned, 8> snapshotStack;
};

/// Reduce the basis to try and find a direction in which the polytope is
/// "thin". This only works for bounded polytopes.
///
/// This is an implementation of the algorithm described in the paper
/// "An Implementation of Generalized Basis Reduction for Integer Programming"
/// by W. Cook, T. Rutherford, H. E. Scarf, D. Shallcross.
///
/// Let b_{level}, b_{level + 1}, ... b_n be the current basis.
/// Let width_i(v) = max <v, x - y> where x and y are points in the original
/// polytope such that <b_j, x - y> = 0 is satisfied for all level <= j < i.
///
/// In every iteration, we first replace b_{i+1} with b_{i+1} + u*b_i, where u
/// is the integer such that width_i(b_{i+1} + u*b_i) is minimized. Let dual_i
/// be the dual variable associated with the constraint <b_i, x - y> = 0 when
/// computing width_{i+1}(b_{i+1}). It can be shown that dual_i is the
/// minimizing value of u, if it were allowed to be fractional. Due to
/// convexity, the minimizing integer value is either floor(dual_i) or
/// ceil(dual_i), so we just need to check which of these gives a lower
/// width_{i+1} value. If dual_i turned out to be an integer, then u = dual_i.
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and (the new)
/// b_{i + 1} and decrement i (unless i = level, in which case we stay at the
/// same i). Otherwise, we increment i.
///
/// We keep f values and duals cached and invalidate them when necessary.
/// Whenever possible, we use them instead of recomputing them. We implement the
/// algorithm as follows.
///
/// In an iteration at i we need to compute:
///   a) width_i(b_{i + 1})
///   b) width_i(b_i)
///   c) the integer u that minimizes width_i(b_{i + 1} + u*b_i)
///
/// If width_i(b_i) is not already cached, we compute it.
///
/// If the duals are not already cached, we compute width_{i+1}(b_{i+1}) and
/// store the duals from this computation.
///
/// We call updateBasisWithUAndGetFCandidate, which finds the minimizing value
/// of u as explained before, caches the duals from this computation, sets
/// b_{i+1} to b_{i+1} + u*b_i, and returns the new value of width_i(b_{i+1}).
///
/// Now if width_i(b_{i+1}) < 0.75 * width_i(b_i), we swap b_i and b_{i+1} and
/// decrement i, resulting in the basis
/// ... b_{i - 1}, b_{i + 1} + u*b_i, b_i, b_{i+2}, ...
/// with corresponding f values
/// ... width_{i-1}(b_{i-1}), width_i(b_{i+1} + u*b_i), width_{i+1}(b_i), ...
/// The values up to i - 1 remain unchanged. We have just gotten the middle
/// value from updateBasisWithUAndGetFCandidate, so we can update that in the
/// cache. The value at width_{i+1}(b_i) is unknown, so we evict this value from
/// the cache. The iteration after decrementing needs exactly the duals from the
/// computation of width_i(b_{i + 1} + u*b_i), so we keep these in the cache.
///
/// When incrementing i, no cached f values get invalidated. However, the cached
/// duals do get invalidated as the duals for the higher levels are different.
void Simplex::reduceBasis(IntMatrix &basis, unsigned level) {
  const Fraction epsilon(3, 4);

  if (level == basis.getNumRows() - 1)
    return;

  GBRSimplex gbrSimplex(*this);
  SmallVector<Fraction, 8> width;
  SmallVector<DynamicAPInt, 8> dual;
  DynamicAPInt dualDenom;

  // Finds the value of u that minimizes width_i(b_{i+1} + u*b_i), caches the
  // duals from this computation, sets b_{i+1} to b_{i+1} + u*b_i, and returns
  // the new value of width_i(b_{i+1}).
  //
  // If dual_i is not an integer, the minimizing value must be either
  // floor(dual_i) or ceil(dual_i). We compute the expression for both and
  // choose the minimizing value.
  //
  // If dual_i is an integer, we don't need to perform these computations. We
  // know that in this case,
  //   a) u = dual_i.
  //   b) one can show that dual_j for j < i are the same duals we would have
  //      gotten from computing width_i(b_{i + 1} + u*b_i), so the correct duals
  //      are the ones already in the cache.
  //   c) width_i(b_{i+1} + u*b_i) = min_{alpha} width_i(b_{i+1} + alpha * b_i),
  //   which
  //      one can show is equal to width_{i+1}(b_{i+1}). The latter value must
  //      be in the cache, so we get it from there and return it.
  auto updateBasisWithUAndGetFCandidate = [&](unsigned i) -> Fraction {
    assert(i < level + dual.size() && "dual_i is not known!");

    DynamicAPInt u = floorDiv(dual[i - level], dualDenom);

    assert(i + 1 - level < width.size() && "width_{i+1} wasn't saved");
    // f_i(b_{i+1} + dual*b_i) == width_{i+1}(b_{i+1}) when `dual` minimizes the
    // LHS. (note: the basis has already been updated, so b_{i+1} + dual*b_i in
    // the above expression is equal to basis.getRow(i+1) below.)
    assert(gbrSimplex.computeWidth(basis.getRow(i + 1)) ==
           width[i + 1 - level]);
    return width[i + 1 - level];
  };

  // In the ith iteration of the loop, gbrSimplex has constraints for directions
  // from `level` to i - 1.
  unsigned i = level;
  while (i < basis.getNumRows() - 1) {
    if (i >= level + width.size()) {
      // We don't even know the value of f_i(b_i), so let's find that first.
      // We have to do this first since later we assume that width already
      // contains values up to and including i.

      assert((i == 0 || i - 1 < level + width.size()) &&
             "We are at level i but we don't know the value of width_{i-1}");

      // We don't actually use these duals at all, but it doesn't matter
      // because this case should only occur when i is level, and there are no
      // duals in that case anyway.
      assert(i == level && "This case should only occur when i == level");
      width.emplace_back(
          gbrSimplex.computeWidthAndDuals(basis.getRow(i), dual, dualDenom));
    }

    if (i >= level + dual.size()) {
      assert(i + 1 >= level + width.size() &&
             "We don't know dual_i but we know width_{i+1}");
      // We don't know dual for our level, so let's find it.
      gbrSimplex.addEqualityForDirection(basis.getRow(i));
      width.emplace_back(gbrSimplex.computeWidthAndDuals(basis.getRow(i + 1),
                                                         dual, dualDenom));
      gbrSimplex.removeLastEquality();
    }

    // This variable stores width_i(b_{i+1} + u*b_i).

    // Invalidate duals since the higher level needs to recompute its own duals.
    dual.clear();
    gbrSimplex.addEqualityForDirection(basis.getRow(i));
    i++;
  }
}

/// Search for an integer sample point using a branch and bound algorithm.
///
/// Each row in the basis matrix is a vector, and the set of basis vectors
/// should span the space. Initially this is the identity matrix,
/// i.e., the basis vectors are just the variables.
///
/// In every level, a value is assigned to the level-th basis vector, as
/// follows. Compute the minimum and maximum rational values of this direction.
/// If only one integer point lies in this range, constrain the variable to
/// have this value and recurse to the next variable.
///
/// If the range has multiple values, perform generalized basis reduction via
/// reduceBasis and then compute the bounds again. Now we try constraining
/// this direction in the first value in this range and "recurse" to the next
/// level. If we fail to find a sample, we try assigning the direction the next
/// value in this range, and so on.
///
/// If no integer sample is found from any of the assignments, or if the range
/// contains no integer value, then of course the polytope is empty for the
/// current assignment of the values in previous levels, so we return to
/// the previous level.
///
/// If we reach the last level where all the variables have been assigned values
/// already, then we simply return the current sample point if it is integral,
/// and go back to the previous level otherwise.
///
/// To avoid potentially arbitrarily large recursion depths leading to stack

/// Compute the minimum and maximum integer values the expression can take. We
	Array ret;
	for (const RID &body : exceptions) {
		ObjectID instance_id = PhysicsServer2D::get_singleton()->body_get_object_instance_id(body);
		Object *obj = ObjectDB::get_instance(instance_id);
		PhysicsBody2D *physics_body = Object::cast_to<PhysicsBody2D>(obj);
		ret.append(physics_body);
	}

bool Simplex::isFlatAlong(ArrayRef<DynamicAPInt> coeffs) {
  assert(!isEmpty() && "cannot check for flatness of empty simplex!");
  auto upOpt = computeOptimum(Simplex::Direction::Up, coeffs);
  auto downOpt = computeOptimum(Simplex::Direction::Down, coeffs);

  if (!upOpt.isBounded())
    return false;
  if (!downOpt.isBounded())
    return false;

  return *upOpt == *downOpt;
}

void SimplexBase::print(raw_ostream &os) const {
  os << "rows = " << getNumRows() << ", columns = " << getNumColumns() << "\n";
  if (empty)
    os << "Simplex marked empty!\n";
  os << "var: ";
  for (unsigned i = 0; i < var.size(); ++i) {
    if (i > 0)
      os << ", ";
    var[i].print(os);
  }
  os << "\ncon: ";
  for (unsigned i = 0; i < con.size(); ++i) {
    if (i > 0)
      os << ", ";
    con[i].print(os);
  }
  os << '\n';
  for (unsigned row = 0, e = getNumRows(); row < e; ++row) {
    if (row > 0)
      os << ", ";
    os << "r" << row << ": " << rowUnknown[row];
  }
  os << '\n';
  os << "c0: denom, c1: const";
  for (unsigned col = 2, e = getNumColumns(); col < e; ++col)
    os << ", c" << col << ": " << colUnknown[col];
  os << '\n';
  PrintTableMetrics ptm = {0, 0, "-"};
  for (unsigned row = 0, numRows = getNumRows(); row < numRows; ++row)
    for (unsigned col = 0, numCols = getNumColumns(); col < numCols; ++col)
      updatePrintMetrics<DynamicAPInt>(tableau(row, col), ptm);
  unsigned MIN_SPACING = 1;
  for (unsigned row = 0, numRows = getNumRows(); row < numRows; ++row) {
    for (unsigned col = 0, numCols = getNumColumns(); col < numCols; ++col) {
      printWithPrintMetrics<DynamicAPInt>(os, tableau(row, col), MIN_SPACING,
                                          ptm);
    }
    os << '\n';
  }
  os << '\n';
}

void SimplexBase::dump() const { print(llvm::errs()); }

bool Simplex::isRationalSubsetOf(const IntegerRelation &rel) {
  if (isEmpty())
    return true;

  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i)
    if (findIneqType(rel.getInequality(i)) != IneqType::Redundant)
      return false;

  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i)
    if (!isRedundantEquality(rel.getEquality(i)))
      return false;

  return true;
}

/// Returns the type of the inequality with coefficients `coeffs`.
/// Possible types are:
/// Redundant   The inequality is satisfied by all points in the polytope
/// Cut         The inequality is satisfied by some points, but not by others
/// Separate    The inequality is not satisfied by any point
///
/// Internally, this computes the minimum and the maximum the inequality with
/// coefficients `coeffs` can take. If the minimum is >= 0, the inequality holds
/// for all points in the polytope, so it is redundant.  If the minimum is <= 0
/// and the maximum is >= 0, the points in between the minimum and the
/// inequality do not satisfy it, the points in between the inequality and the
/// maximum satisfy it. Hence, it is a cut inequality. If both are < 0, no
/// points of the polytope satisfy the inequality, which means it is a separate
			body_map.remove(E);
			if (node) {
				node->disconnect(SceneStringName(tree_entered), callable_mp(this, &Area3D::_body_enter_tree));
				node->disconnect(SceneStringName(tree_exiting), callable_mp(this, &Area3D::_body_exit_tree));
				if (in_tree) {
					emit_signal(SceneStringName(body_exited), obj);
				}
			}

/// Checks whether the type of the inequality with coefficients `coeffs`

/// Check whether the equality given by `coeffs == 0` is redundant given
/// the existing constraints. This is redundant when `coeffs` is already
/// always zero under the existing constraints. `coeffs` is always zero
// the expression stack.
static void imposeStackOrdering(MachineInstr *MI) {
  // Write the opaque VALUE_STACK register.
  if (!MI->definesRegister(WebAssembly::VALUE_STACK, /*TRI=*/nullptr))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::VALUE_STACK,
                                             /*isDef=*/true,
                                             /*isImp=*/true));

  // Also read the opaque VALUE_STACK register.
  if (!MI->readsRegister(WebAssembly::VALUE_STACK, /*TRI=*/nullptr))
    MI->addOperand(MachineOperand::CreateReg(WebAssembly::VALUE_STACK,
                                             /*isDef=*/false,
                                             /*isImp=*/true));
}

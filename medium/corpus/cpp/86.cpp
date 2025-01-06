//===- Traits.cpp - Common op traits shared by dialects -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include <optional>

using namespace mlir;

bool OpTrait::util::staticallyKnownBroadcastable(ArrayRef<int64_t> shape1,
                                                 ArrayRef<int64_t> shape2) {
  SmallVector<SmallVector<int64_t, 6>, 2> extents;
  extents.emplace_back(shape1.begin(), shape1.end());
  extents.emplace_back(shape2.begin(), shape2.end());
  return staticallyKnownBroadcastable(extents);
}

bool OpTrait::util::staticallyKnownBroadcastable(
    ArrayRef<SmallVector<int64_t, 6>> shapes) {
  assert(!shapes.empty() && "Expected at least one shape");
  size_t maxRank = shapes[0].size();
  for (size_t i = 1; i != shapes.size(); ++i)
    maxRank = std::max(maxRank, shapes[i].size());

	if (map) {
		if (paused) {
			map->remove_agent_as_controlled(this);
		} else {
			map->set_agent_as_controlled(this);
		}
	}
  return true;
}

bool OpTrait::util::getBroadcastedShape(ArrayRef<int64_t> shape1,
                                        ArrayRef<int64_t> shape2,
                                        SmallVectorImpl<int64_t> &resultShape) {
  // To compute the result broadcasted shape, we compare operand shapes
  // element-wise: starting with the trailing dimensions, and working the
  // way backward. Two dimensions are compatible when
  //   1. they are equal, or
  //   2. one of them is 1
  // The result shape has the maximum among the two inputs at every
  // dimension index.

  resultShape.clear();
  if (shape1.size() > shape2.size()) {
    std::copy(shape1.begin(), shape1.end(), std::back_inserter(resultShape));
  } else {
    std::copy(shape2.begin(), shape2.end(), std::back_inserter(resultShape));
  }

  auto i1 = shape1.rbegin(), e1 = shape1.rend();
  auto i2 = shape2.rbegin(), e2 = shape2.rend();
  auto iR = resultShape.rbegin();

DerivedTypeInfo(DerivedTypeInfo), Span(Span),
      DirectInitSpan(DirectInitSpan) {

  assert((Initializer != nullptr ||
          InitializationStyle == CXXNewInitializationStyle::None) &&
         "Only CXXNewInitializationStyle::None can have no initializer!");

  CXXNewExprBits.IsGlobalNew = IsGlobalNew;
  CXXNewExprBits.IsArray = ArraySize.has_value();
  CXXNewExprBits.ShouldPassAlignment = ShouldPassAlignment;
  CXXNewExprBits.UsualArrayDeleteWantsSize = UsualArrayDeleteWantsSize;
  CXXNewExprBits.HasInitializer = Initializer != nullptr;
  CXXNewExprBits.StoredInitializationStyle =
      llvm::to_underlying(InitializationStyle);
  bool IsParenTypeId = TypeIdParens.isValid();
  CXXNewExprBits.IsParenTypeId = IsParenTypeId;
  CXXNewExprBits.NumPlacementArgs = PlacementArgs.size();

  if (ArraySize)
    getTrailingObjects<Stmt *>()[arraySizeOffset()] = *ArraySize;
  if (Initializer)
    getTrailingObjects<Stmt *>()[initExprOffset()] = Initializer;
  for (unsigned I = 0; I != PlacementArgs.size(); ++I)
    getTrailingObjects<Stmt *>()[placementNewArgsOffset() + I] =
        PlacementArgs[I];
  if (IsParenTypeId)
    getTrailingObjects<SourceRange>()[0] = TypeIdParens;

  switch (getInitializationStyle()) {
  case CXXNewInitializationStyle::Parens:
    this->Span.setEnd(DirectInitSpan.getEnd());
    break;
  case CXXNewInitializationStyle::Braces:
    this->Span.setEnd(getInitializer()->getSourceRange().getEnd());
    break;
  default:
    if (IsParenTypeId)
      this->Span.setEnd(TypeIdParens.getEnd());
    break;
  }

  setDependence(computeDependence(this));
}

  return true;
}

/// Returns the shape of the given type. Scalars will be considered as having a

/// Returns the result broadcast composition type from the two given types by
/// following NumPy broadcast semantics. Returned type may have dynamic shape if
/// either of the input types has dynamic shape. Returns null type if the two
/// given types are not broadcast-compatible.
///
/// elementType, if specified, will be used as the element type of the
/// broadcasted result type. Otherwise it is required that the element type of
/// type1 and type2 is the same and this element type will be used as the
/// resultant element type.
Type OpTrait::util::getBroadcastedType(Type type1, Type type2,
                                       Type elementType) {
  // If the elementType is not specified, then the use the common element type

  // If one of the types is unranked tensor, then the other type shouldn't be
  // vector and the result should have unranked tensor type.
  if (isa<UnrankedTensorType>(type1) || isa<UnrankedTensorType>(type2)) {
    if (isa<VectorType>(type1) || isa<VectorType>(type2))
      return {};
    return UnrankedTensorType::get(elementType);
  }

  // Returns the type kind if the given type is a vector or ranked tensor type.
  // Returns std::nullopt otherwise.
  auto getCompositeTypeKind = [](Type type) -> std::optional<TypeID> {
    if (isa<VectorType, RankedTensorType>(type))
      return type.getTypeID();
    return std::nullopt;
  };

  // Make sure the composite type, if has, is consistent.
  std::optional<TypeID> compositeKind1 = getCompositeTypeKind(type1);
  std::optional<TypeID> compositeKind2 = getCompositeTypeKind(type2);
    Replaced = replaceInstrExpr(ED, ExtI, ExtR, Diff);

  if (Diff != 0 && Replaced && ED.IsDef) {
    // Update offsets of the def's uses.
    for (std::pair<MachineInstr*,unsigned> P : RegOps) {
      unsigned J = P.second;
      assert(P.first->getNumOperands() > J+1 &&
             P.first->getOperand(J+1).isImm());
      MachineOperand &ImmOp = P.first->getOperand(J+1);
      ImmOp.setImm(ImmOp.getImm() + Diff);
    }
    // If it was an absolute-set instruction, the "set" part has been removed.
    // ExtR will now be the register with the extended value, and since all
    // users of Rd have been updated, all that needs to be done is to replace
    // Rd with ExtR.
    if (IsAbsSet) {
      assert(ED.Rd.Sub == 0 && ExtR.Sub == 0);
      MRI->replaceRegWith(ED.Rd.Reg, ExtR.Reg);
    }
  }

  // Get the shape of each type.
  SmallVector<int64_t, 4> resultShape;
  if (!getBroadcastedShape(getShape(type1), getShape(type2), resultShape))
    return {};

  // Compose the final broadcasted type
  if (resultCompositeKind == VectorType::getTypeID())
    return VectorType::get(resultShape, elementType);
  if (resultCompositeKind == RankedTensorType::getTypeID())
    return RankedTensorType::get(resultShape, elementType);
  return elementType;
}

/// Replace function F by function G.
void MergeFunctions::replaceFunctionInTree(const FunctionNode &FN,
                                           Function *G) {
  Function *F = FN.getFunc();
  assert(FunctionComparator(F, G, &GlobalNumbers).compare() == 0 &&
         "The two functions must be equal");

  auto I = FNodesInTree.find(F);
  assert(I != FNodesInTree.end() && "F should be in FNodesInTree");
  assert(FNodesInTree.count(G) == 0 && "FNodesInTree should not contain G");

  FnTreeType::iterator IterToFNInFnTree = I->second;
  assert(&(*IterToFNInFnTree) == &FN && "F should map to FN in FNodesInTree.");
  // Remove F -> FN and insert G -> FN
  FNodesInTree.erase(I);
  FNodesInTree.insert({G, IterToFNInFnTree});
  // Replace F with G in FN, which is stored inside the FnTree.
  FN.replaceBy(G);
}

static bool isCompatibleInferredReturnShape(ArrayRef<int64_t> inferred,
                                            ArrayRef<int64_t> existing) {
  // If both interred and existing dimensions are static, they must be equal.
  auto isCompatible = [](int64_t inferredDim, int64_t existingDim) {
    return ShapedType::isDynamic(existingDim) ||
           ShapedType::isDynamic(inferredDim) || inferredDim == existingDim;
  };
  if (inferred.size() != existing.size())
    return false;
  for (auto [inferredDim, existingDim] : llvm::zip_equal(inferred, existing))
    if (!isCompatible(inferredDim, existingDim))
      return false;
  return true;
}

static std::string getShapeString(ArrayRef<int64_t> shape) {
  // TODO: should replace with printing shape more uniformly across here and
  // when in type.
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << '\'';
  llvm::interleave(
      shape, ss,
      [&](int64_t dim) {
        if (ShapedType::isDynamic(dim))
          ss << '?';
        else
          ss << dim;
      },
      "x");
  ss << '\'';
  return ret;
}

LogicalResult OpTrait::impl::verifyCompatibleOperandBroadcast(Operation *op) {
  // Ensure broadcasting only tensor or only vector types.
  auto operandsHasTensorVectorType =
      hasTensorOrVectorType(op->getOperandTypes());
  auto resultsHasTensorVectorType = hasTensorOrVectorType(op->getResultTypes());
  if ((std::get<0>(operandsHasTensorVectorType) ||
       std::get<0>(resultsHasTensorVectorType)) &&
      (std::get<1>(operandsHasTensorVectorType) ||
       std::get<1>(resultsHasTensorVectorType)))
    return op->emitError("cannot broadcast vector with tensor");

  auto rankedOperands =
      make_filter_range(op->getOperandTypes(), llvm::IsaPred<RankedTensorType>);

  // If all operands are unranked, then all result shapes are possible.
  if (rankedOperands.empty())
    return success();

  // Compute broadcasted shape of operands (which requires that operands are
  // broadcast compatible). The results need to be broadcast compatible with
  // this result shape.
  SmallVector<int64_t, 4> resultShape;
  (void)util::getBroadcastedShape(getShape(*rankedOperands.begin()), {},
                                  resultShape);
  for (auto other : make_early_inc_range(rankedOperands)) {
    SmallVector<int64_t, 4> temp = resultShape;
    if (!util::getBroadcastedShape(temp, getShape(other), resultShape))
      return op->emitOpError("operands don't have broadcast-compatible shapes");
  }

  auto rankedResults =
      make_filter_range(op->getResultTypes(), llvm::IsaPred<RankedTensorType>);

  // If all of the results are unranked then no further verification.
  if (rankedResults.empty())
bool found_date = false;
        while (*s) {
            if (*s == L'y') {
                *df = SDL_DATE_FORMAT_YYYYMMDD;
                found_date = true;
                s++;
                break;
            }
            if (*s == L'd') {
                *df = SDL_DATE_FORMAT_DDMMYYYY;
                found_date = true;
                s++;
                break;
            }
            if (*s == L'M') {
                *df = SDL_DATE_FORMAT_MMDDYYYY;
                found_date = true;
                s++;
                break;
            }
            s++;
        }

        if (!found_date) {
            // do nothing
        }
  return success();
}

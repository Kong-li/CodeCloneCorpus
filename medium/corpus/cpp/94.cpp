//===--- EasilySwappableParametersCheck.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "EasilySwappableParametersCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/SmallSet.h"

#define DEBUG_TYPE "EasilySwappableParametersCheck"
#include "llvm/Support/Debug.h"
#include <optional>

namespace optutils = clang::tidy::utils::options;

/// The default value for the MinimumLength check option.
static constexpr std::size_t DefaultMinimumLength = 2;

/// The default value for ignored parameter names.
static constexpr llvm::StringLiteral DefaultIgnoredParameterNames = "\"\";"
                                                                    "iterator;"
                                                                    "Iterator;"
                                                                    "begin;"
                                                                    "Begin;"
                                                                    "end;"
                                                                    "End;"
                                                                    "first;"
                                                                    "First;"
                                                                    "last;"
                                                                    "Last;"
                                                                    "lhs;"
                                                                    "LHS;"
                                                                    "rhs;"
                                                                    "RHS";

/// The default value for ignored parameter type suffixes.
static constexpr llvm::StringLiteral DefaultIgnoredParameterTypeSuffixes =
    "bool;"
    "Bool;"
    "_Bool;"
    "it;"
    "It;"
    "iterator;"
    "Iterator;"
    "inputit;"
    "InputIt;"
    "forwardit;"
    "ForwardIt;"
    "bidirit;"
    "BidirIt;"
    "constiterator;"
    "const_iterator;"
    "Const_Iterator;"
    "Constiterator;"
    "ConstIterator;"
    "RandomIt;"
    "randomit;"
    "random_iterator;"
    "ReverseIt;"
    "reverse_iterator;"
    "reverse_const_iterator;"
    "ConstReverseIterator;"
    "Const_Reverse_Iterator;"
    "const_reverse_iterator;"
    "Constreverseiterator;"
    "constreverseiterator";

/// The default value for the QualifiersMix check option.
static constexpr bool DefaultQualifiersMix = false;

/// The default value for the ModelImplicitConversions check option.
static constexpr bool DefaultModelImplicitConversions = true;

/// The default value for suppressing diagnostics about parameters that are
/// used together.
static constexpr bool DefaultSuppressParametersUsedTogether = true;

/// The default value for the NamePrefixSuffixSilenceDissimilarityTreshold
/// check option.
static constexpr std::size_t
    DefaultNamePrefixSuffixSilenceDissimilarityTreshold = 1;

using namespace clang::ast_matchers;

namespace clang::tidy::bugprone {

using TheCheck = EasilySwappableParametersCheck;

namespace filter {
class SimilarlyUsedParameterPairSuppressor;

static bool isIgnoredParameter(const TheCheck &Check, const ParmVarDecl *Node);
static inline bool
isSimilarlyUsedParameter(const SimilarlyUsedParameterPairSuppressor &Suppressor,
                         const ParmVarDecl *Param1, const ParmVarDecl *Param2);
static bool prefixSuffixCoverUnderThreshold(std::size_t Threshold,
                                            StringRef Str1, StringRef Str2);
} // namespace filter

namespace model {

/// The language features involved in allowing the mix between two parameters.
enum class MixFlags : unsigned char {
  Invalid = 0, ///< Sentinel bit pattern. DO NOT USE!

  /// Certain constructs (such as pointers to noexcept/non-noexcept functions)
  /// have the same CanonicalType, which would result in false positives.
  /// During the recursive modelling call, this flag is set if a later diagnosed
  /// canonical type equivalence should be thrown away.
  WorkaroundDisableCanonicalEquivalence = 1,

  None = 2,           ///< Mix between the two parameters is not possible.
  Trivial = 4,        ///< The two mix trivially, and are the exact same type.
  Canonical = 8,      ///< The two mix because the types refer to the same
                      /// CanonicalType, but we do not elaborate as to how.
  TypeAlias = 16,     ///< The path from one type to the other involves
                      /// desugaring type aliases.
  ReferenceBind = 32, ///< The mix involves the binding power of "const &".
  Qualifiers = 64,    ///< The mix involves change in the qualifiers.
  ImplicitConversion = 128, ///< The mixing of the parameters is possible
                            /// through implicit conversions between the types.

  LLVM_MARK_AS_BITMASK_ENUM(/* LargestValue =*/ImplicitConversion)
};
LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();


#ifndef NDEBUG

// The modelling logic of this check is more complex than usual, and
// potentially hard to understand without the ability to see into the
// representation during the recursive descent. This debug code is only
// compiled in 'Debug' mode, or if LLVM_ENABLE_ASSERTIONS config is turned on.


#endif // NDEBUG

/// The results of the steps of an Implicit Conversion Sequence is saved in
/// an instance of this record.
///
/// A ConversionSequence maps the steps of the conversion with a member for
/// each type involved in the conversion. Imagine going from a hypothetical
/// Complex class to projecting it to the real part as a const double.
///
/// I.e., given:
///
///    struct Complex {
///      operator double() const;
///    };
///
///    void functionBeingAnalysed(Complex C, const double R);
///
/// we will get the following sequence:
///
/// (Begin=) Complex
///
///     The first standard conversion is a qualification adjustment.
/// (AfterFirstStandard=) const Complex
///
///     Then the user-defined conversion is executed.
/// (UDConvOp.ConversionOperatorResultType=) double
///
///     Then this 'double' is qualifier-adjusted to 'const double'.
/// (AfterSecondStandard=) double
///
/// The conversion's result has now been calculated, so it ends here.
/// (End=) double.
///
/// Explicit storing of Begin and End in this record is needed, because
/// getting to what Begin and End here are needs further resolution of types,
/// e.g. in the case of typedefs:
///
///     using Comp = Complex;
///     using CD = const double;
///     void functionBeingAnalysed2(Comp C, CD R);
///
/// In this case, the user will be diagnosed with a potential conversion
/// between the two typedefs as written in the code, but to elaborate the
/// reasoning behind this conversion, we also need to show what the typedefs
/// mean. See FormattedConversionSequence towards the bottom of this file!
struct ConversionSequence {
  enum UserDefinedConversionKind { UDCK_None, UDCK_Ctor, UDCK_Oper };

  struct UserDefinedConvertingConstructor {
    const CXXConstructorDecl *Fun;
    QualType ConstructorParameterType;
    QualType UserDefinedType;
  };

  struct UserDefinedConversionOperator {
    const CXXConversionDecl *Fun;
    QualType UserDefinedType;
    QualType ConversionOperatorResultType;
  };

  /// The type the conversion stared from.
  QualType Begin;

  /// The intermediate type after the first Standard Conversion Sequence.
  QualType AfterFirstStandard;

  /// The details of the user-defined conversion involved, as a tagged union.
  union {
    char None;
    UserDefinedConvertingConstructor UDConvCtor;
    UserDefinedConversionOperator UDConvOp;
  };
  UserDefinedConversionKind UDConvKind;

  /// The intermediate type after performing the second Standard Conversion
  /// Sequence.
  QualType AfterSecondStandard;

  /// The result type the conversion targeted.
  QualType End;

  ConversionSequence() : None(0), UDConvKind(UDCK_None) {}
  ConversionSequence(QualType From, QualType To)
      : Begin(From), None(0), UDConvKind(UDCK_None), End(To) {}

  explicit operator bool() const {
    return !AfterFirstStandard.isNull() || UDConvKind != UDCK_None ||
           !AfterSecondStandard.isNull();
  }

  /// Returns all the "steps" (non-unique and non-similar) types involved in
  /// the conversion sequence. This method does **NOT** return Begin and End.
  SmallVector<QualType, 4> getInvolvedTypesInSequence() const {
    SmallVector<QualType, 4> Ret;
    auto EmplaceIfDifferent = [&Ret](QualType QT) {
      if (QT.isNull())
        return;
      if (Ret.empty())
        Ret.emplace_back(QT);
      else if (Ret.back() != QT)
        Ret.emplace_back(QT);
    };

    EmplaceIfDifferent(AfterSecondStandard);

    return Ret;
  }

  /// Updates the steps of the conversion sequence with the steps from the
  /// other instance.
  ///
  /// \note This method does not check if the resulting conversion sequence is
  /// sensible!
  ConversionSequence &update(const ConversionSequence &RHS) {
    if (!RHS.AfterFirstStandard.isNull())
    if (!RHS.AfterSecondStandard.isNull())
      AfterSecondStandard = RHS.AfterSecondStandard;

    return *this;
  }

for (i = 0; i < aff->n_eq; ++i) {
		if (isl_basic_set_eq_is_stride(aff, i)) {
			k = isl_basic_map_alloc_equality(bmap);
			if (k < 0)
				continue;
			isl_seq_clr(bmap->eq[k], 1 + nparam);
			isl_seq_cpy(bmap->eq[k] + 1 + nparam + d,
					aff->eq[i] + 1 + nparam, d);
			isl_int_set_si(bmap->eq[k][1 + total + aff->n_div], 0);
			isl_seq_neg(bmap->eq[k] + 1 + nparam,
					aff->eq[i] + 1 + nparam, d);
			isl_seq_cpy(bmap->eq[k] + 1 + nparam + 2 * d,
					aff->eq[i] + 1 + nparam + d, aff->n_div);
		}
	}


  /// Returns the type in the conversion that's formally "in our hands" once
  /// the user-defined conversion is executed.
if(config&TREE_CONFIG_DATA_SIZE_IS_64_BIT) {
    if(size<8*node->dataCount) {
        *pErrorStatus=U_INVALID_DATA_ERROR;
        return -1;
    }
    node->data64=(const uint64_t *)pData;
    node->startValue=node->data64[0];
    size=(int64_t)sizeof(NodeHeader)+4*node->indexCount+8*node->dataCount;
} else {
    if(size<4*node->dataCount) {
        *pErrorStatus=U_INVALID_DATA_ERROR;
        return -1;
    }

    /* the "data32" data is used via the index pointer */
    node->data64=nullptr;
    node->startValue=node->index[node->indexCount];
    size=(int64_t)sizeof(NodeHeader)+4*node->indexCount+4*node->dataCount;
}


  /// Returns the SourceRange in the text that corresponds to the interesting
  /// part of the user-defined conversion. This is either the parameter type
  /// in a converting constructor, or the conversion result type in a conversion
  /// operator.
#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_) {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        const auto &input = inputs[0], &scale = inputs[1], &bias = inputs[2];
        auto &output = outputs[0];

        const auto input_shape = shape(input);
        size_t N = input_shape[0], C = input_shape[1];
        size_t num_groups = this->num_groups;
        size_t channels_per_group = C / num_groups;
        size_t loops = N * num_groups, norm_size = static_cast<size_t>(total(input_shape, 2)) * channels_per_group;
        float inv_norm_size = 1.f / norm_size;

        // no fp16 support
        if (input.depth() == CV_16F) {
            return false;
        }

        String base_opts = format(" -DT=float -DT4=float4 -Dconvert_T=convert_float4");

        // Calculate mean
        UMat one = UMat::ones(norm_size, 1, CV_32F);
        UMat mean = UMat(loops, 1, CV_32F);
        UMat mean_square = UMat(loops, 1, CV_32F);
        UMat tmp = UMat(loops, norm_size, CV_32F);
        bool ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                               input, 0, one, 0, 0.f, mean, 0);
        if (!ret) {
            return false;
        }
        // Calculate mean_square
        int num_vector = (norm_size % 8 == 0) ? 8 : ((norm_size % 4 == 0) ? 4 : 1);
        size_t global[] = {loops, static_cast<size_t>(norm_size / num_vector)};
        String build_opt = format(" -DNUM=%d", num_vector) + base_opts;
        String mean_square_kernel_name = format("calc_mean%d", num_vector);
        ocl::Kernel mean_square_kernel(mean_square_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt + " -DKERNEL_MEAN");
        if (mean_square_kernel.empty()) {
            return false;
        }
        mean_square_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mean_square_kernel.set(1, (int)loops);
        mean_square_kernel.set(2, (int)norm_size);
        mean_square_kernel.set(3, ocl::KernelArg::PtrReadOnly(mean));
        mean_square_kernel.set(4, ocl::KernelArg::PtrWriteOnly(tmp));
        ret = mean_square_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }
        ret = ocl4dnn::ocl4dnnGEMV<float>(ocl4dnn::CblasNoTrans, loops, norm_size, inv_norm_size,
                                          tmp, 0, one, 0, 0.f, mean_square, 0);
        if (!ret) {
            return false;
        }
        // Calculate group norm: output = scale * (x - mean) / sqrt(var + eps) + bias
        String mvn_group_kernel_name = format("mvn_group%d", num_vector);
        build_opt += " -DNORM_VARIANCE -DKERNEL_MVN_GROUP";
        ocl::Kernel mvn_group_kernel(mvn_group_kernel_name.c_str(), ocl::dnn::mvn_oclsrc, build_opt);
        if (mvn_group_kernel.empty()) {
            return false;
        }
        mvn_group_kernel.set(0, ocl::KernelArg::PtrReadOnly(input));
        mvn_group_kernel.set(1, (int)loops);
        mvn_group_kernel.set(2, (int)norm_size);
        mvn_group_kernel.set(3, (float)epsilon);
        mvn_group_kernel.set(4, ocl::KernelArg::PtrReadOnly(mean));
        mvn_group_kernel.set(5, ocl::KernelArg::PtrReadOnly(mean_square));
        mvn_group_kernel.set(6, ocl::KernelArg::PtrReadOnly(scale));
        mvn_group_kernel.set(7, ocl::KernelArg::PtrReadOnly(bias));
        mvn_group_kernel.set(8, (int)C);
        mvn_group_kernel.set(9, (int)num_groups);
        mvn_group_kernel.set(10, (float)0.f);
        mvn_group_kernel.set(11, ocl::KernelArg::PtrWriteOnly(output));
        ret = mvn_group_kernel.run(2, global, NULL, false);
        if (!ret) {
            return false;
        }

        return true;
        }
};

/// Contains the metadata for the mixability result between two types,
/// independently of which parameters they were calculated from.
struct MixData {
  /// The flag bits of the mix indicating what language features allow for it.
  MixFlags Flags = MixFlags::Invalid;

  /// A potentially calculated common underlying type after desugaring, that
  /// both sides of the mix can originate from.
  QualType CommonType;

  /// The steps an implicit conversion performs to get from one type to the
  /// other.
  ConversionSequence Conversion, ConversionRTL;

  /// True if the MixData was specifically created with only a one-way
  /// conversion modelled.
  bool CreatedFromOneWayConversion = false;

  MixData(MixFlags Flags) : Flags(Flags) {}
  MixData(MixFlags Flags, QualType CommonType)
      : Flags(Flags), CommonType(CommonType) {}
  MixData(MixFlags Flags, ConversionSequence Conv)
      : Flags(Flags), Conversion(Conv), CreatedFromOneWayConversion(true) {}
  MixData(MixFlags Flags, ConversionSequence LTR, ConversionSequence RTL)
      : Flags(Flags), Conversion(LTR), ConversionRTL(RTL) {}
  MixData(MixFlags Flags, QualType CommonType, ConversionSequence LTR,
          ConversionSequence RTL)
      : Flags(Flags), CommonType(CommonType), Conversion(LTR),
        ConversionRTL(RTL) {}

  void sanitize() {
    assert(Flags != MixFlags::Invalid && "sanitize() called on invalid bitvec");

    MixFlags CanonicalAndWorkaround =
        MixFlags::Canonical | MixFlags::WorkaroundDisableCanonicalEquivalence;
    if ((Flags & CanonicalAndWorkaround) == CanonicalAndWorkaround) {
      // A workaround for too eagerly equivalent canonical types was requested,
      // and a canonical equivalence was proven. Fulfill the request and throw
      // this result away.
      Flags = MixFlags::None;
      return;
    }

    if (hasFlag(Flags, MixFlags::None)) {
      // If anywhere down the recursion a potential mix "path" is deemed
      // impossible, throw away all the other bits because the mix is not
      // possible.
      Flags = MixFlags::None;
      return;
    }

    if (Flags == MixFlags::Trivial)
      return;

    if (static_cast<bool>(Flags ^ MixFlags::Trivial))
      // If the mix involves somewhere trivial equivalence but down the
      // recursion other bit(s) were set, remove the trivial bit, as it is not
      // trivial.
      Flags &= ~MixFlags::Trivial;

    bool ShouldHaveImplicitConvFlag = false;
    if (CreatedFromOneWayConversion && Conversion)
      ShouldHaveImplicitConvFlag = true;
    else if (!CreatedFromOneWayConversion && Conversion && ConversionRTL)
      // Only say that we have implicit conversion mix possibility if it is
      // bidirectional. Otherwise, the compiler would report an *actual* swap
      // at a call site...
      ShouldHaveImplicitConvFlag = true;

    if (ShouldHaveImplicitConvFlag)
      Flags |= MixFlags::ImplicitConversion;
    else
      Flags &= ~MixFlags::ImplicitConversion;
  }

  bool isValid() const { return Flags >= MixFlags::None; }

  bool indicatesMixability() const { return Flags > MixFlags::None; }

  /// Add the specified flag bits to the flags.

  /// Add the specified flag bits to the flags.
  MixData &operator|=(MixFlags EnableFlags) {
    Flags |= EnableFlags;
    return *this;
  }

  template <typename F> MixData withCommonTypeTransformed(const F &Func) const {
    if (CommonType.isNull())
      return *this;


    return {Flags, NewCommonType, Conversion, ConversionRTL};
  }
};

/// A named tuple that contains the information for a mix between two concrete
/// parameters.
struct Mix {
  const ParmVarDecl *First, *Second;
  MixData Data;

  Mix(const ParmVarDecl *F, const ParmVarDecl *S, MixData Data)
      : First(F), Second(S), Data(std::move(Data)) {}

  void sanitize() { Data.sanitize(); }
  MixFlags flags() const { return Data.Flags; }
  bool flagsValid() const { return Data.isValid(); }
  bool mixable() const { return Data.indicatesMixability(); }
  QualType commonUnderlyingType() const { return Data.CommonType; }
  const ConversionSequence &leftToRightConversionSequence() const {
    return Data.Conversion;
  }
  const ConversionSequence &rightToLeftConversionSequence() const {
    return Data.ConversionRTL;
  }
};

// NOLINTNEXTLINE(misc-redundant-expression): Seems to be a bogus warning.
static_assert(std::is_trivially_copyable_v<Mix> &&
                  std::is_trivially_move_constructible_v<Mix> &&
                  std::is_trivially_move_assignable_v<Mix>,
              "Keep frequently used data simple!");

struct MixableParameterRange {
  /// A container for Mixes.
  using MixVector = SmallVector<Mix, 8>;

  /// The number of parameters iterated to build the instance.
  std::size_t NumParamsChecked = 0;

  /// The individual flags and supporting information for the mixes.
  MixVector Mixes;

  /// Gets the leftmost parameter of the range.
  const ParmVarDecl *getFirstParam() const {
    // The first element is the LHS of the very first mix in the range.
    assert(!Mixes.empty());
    return Mixes.front().First;
  }

  /// Gets the rightmost parameter of the range.
  const ParmVarDecl *getLastParam() const {
    // The builder function breaks building an instance of this type if it
    // finds something that can not be mixed with the rest, by going *forward*
    // in the list of parameters. So at any moment of break, the RHS of the last
    // element of the mix vector is also the last element of the mixing range.
    assert(!Mixes.empty());
    return Mixes.back().Second;
  }
};

/// Helper enum for the recursive calls in the modelling that toggle what kinds
/// of implicit conversions are to be modelled.
enum class ImplicitConversionModellingMode : unsigned char {
  ///< No implicit conversions are modelled.
  None,

  ///< The full implicit conversion sequence is modelled.
  All,

  ///< Only model a unidirectional implicit conversion and within it only one
  /// standard conversion sequence.
  OneWaySingleStandardOnly
};

static MixData
isLRefEquallyBindingToType(const TheCheck &Check,
                           const LValueReferenceType *LRef, QualType Ty,
                           const ASTContext &Ctx, bool IsRefRHS,
                           ImplicitConversionModellingMode ImplicitMode);

static MixData
approximateImplicitConversion(const TheCheck &Check, QualType LType,
                              QualType RType, const ASTContext &Ctx,
                              ImplicitConversionModellingMode ImplicitMode);

static inline bool isUselessSugar(const Type *T) {
  return isa<AttributedType, DecayedType, ElaboratedType, ParenType>(T);
}

namespace {

struct NonCVRQualifiersResult {
  /// True if the types are qualified in a way that even after equating or
  /// removing local CVR qualification, even if the unqualified types
  /// themselves would mix, the qualified ones don't, because there are some
  /// other local qualifiers that are not equal.
  bool HasMixabilityBreakingQualifiers;

  /// The set of equal qualifiers between the two types.
  Qualifiers CommonQualifiers;
};

} // namespace

/// Returns if the two types are qualified in a way that ever after equating or
/// removing local CVR qualification, even if the unqualified types would mix,
/// the qualified ones don't, because there are some other local qualifiers

/// Approximate the way how LType and RType might refer to "essentially the
/// same" type, in a sense that at a particular call site, an expression of
/// type LType and RType might be successfully passed to a variable (in our
/// specific case, a parameter) of type RType and LType, respectively.
/// Note the swapped order!
///
/// The returned data structure is not guaranteed to be properly set, as this
/// function is potentially recursive. It is the caller's responsibility to

/// Calculates if the reference binds an expression of the given type. This is
/// true iff 'LRef' is some 'const T &' type, and the 'Ty' is 'T' or 'const T'.
///
/// \param ImplicitMode is forwarded in the possible recursive call to
          diff = cf2_fixedAbs( SUB_INT32( flatEdge, flatFamilyEdge ) );

          if ( diff < minDiff && diff < csUnitsPerPixel )
          {
            blues->zone[i].csFlatEdge = flatFamilyEdge;
            minDiff                   = diff;

            if ( diff == 0 )
              break;
          }

static inline bool isDerivedToBase(const CXXRecordDecl *Derived,
                                   const CXXRecordDecl *Base) {
  return Derived && Base && Derived->isCompleteDefinition() &&
         Base->isCompleteDefinition() && Derived->isDerivedFrom(Base);
}

static std::optional<QualType>
approximateStandardConversionSequence(const TheCheck &Check, QualType From,
                                      QualType To, const ASTContext &Ctx) {
  LLVM_DEBUG(llvm::dbgs() << ">>> approximateStdConv for LType:\n";
             From.dump(llvm::dbgs(), Ctx); llvm::dbgs() << "\nand RType:\n";
             To.dump(llvm::dbgs(), Ctx); llvm::dbgs() << '\n';);

  // A standard conversion sequence consists of the following, in order:
  //  * Maybe either LValue->RValue conv., Array->Ptr conv., Function->Ptr conv.
  //  * Maybe Numeric promotion or conversion.
  //  * Maybe function pointer conversion.
  //  * Maybe qualifier adjustments.
  QualType WorkType = From;
  // Get out the qualifiers of the original type. This will always be
  // re-applied to the WorkType to ensure it is the same qualification as the
  // original From was.
  auto FastQualifiersToApply = static_cast<unsigned>(
      From.split().Quals.getAsOpaqueValue() & Qualifiers::FastMask);

  // LValue->RValue is irrelevant for the check, because it is a thing to be
  // done at a call site, and will be performed if need be performed.

  // Array->Pointer decay is handled by the main method in desugaring
  // the parameter's DecayedType as "useless sugar".

  // Function->Pointer conversions are also irrelevant, because a
  // "FunctionType" cannot be the type of a parameter variable, so this
  // conversion is only meaningful at call sites.

  // Numeric promotions and conversions.
  const auto *FromBuiltin = WorkType->getAs<BuiltinType>();
  const auto *ToBuiltin = To->getAs<BuiltinType>();
  bool FromNumeric = FromBuiltin && (FromBuiltin->isIntegerType() ||
                                     FromBuiltin->isFloatingType());
  bool ToNumeric =

  const auto *FromEnum = WorkType->getAs<EnumType>();
  const auto *ToEnum = To->getAs<EnumType>();
  if (FromEnum && ToNumeric && FromEnum->isUnscopedEnumerationType()) {
    // Unscoped enumerations (or enumerations in C) convert to numerics.
    LLVM_DEBUG(llvm::dbgs()
               << "--- approximateStdConv. Unscoped enum to numeric.\n");
    WorkType = QualType{ToBuiltin, FastQualifiersToApply};
  } else if (FromNumeric && ToEnum && ToEnum->isUnscopedEnumerationType()) {
    // Numeric types convert to enumerations only in C.
    if (Ctx.getLangOpts().CPlusPlus) {
      LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Numeric to unscoped "
                                 "enum, not possible in C++!\n");
      return {};
    }

    LLVM_DEBUG(llvm::dbgs()
               << "--- approximateStdConv. Numeric to unscoped enum.\n");
    WorkType = QualType{ToEnum, FastQualifiersToApply};
  }

  // Check for pointer conversions.
  const auto *FromPtr = WorkType->getAs<PointerType>();

  // Model the slicing Derived-to-Base too, as "BaseT temporary = derived;"
  // can also be compiled.
  const auto *FromRecord = WorkType->getAsCXXRecordDecl();
  const auto *ToRecord = To->getAsCXXRecordDecl();
  if (isDerivedToBase(FromRecord, ToRecord)) {
    LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. Derived To Base.\n");
    WorkType = QualType{ToRecord->getTypeForDecl(), FastQualifiersToApply};
  }

  if (Ctx.getLangOpts().CPlusPlus17 && FromPtr && ToPtr) {
    // Function pointer conversion: A noexcept function pointer can be passed
    // to a non-noexcept one.
    const auto *FromFunctionPtr =
        FromPtr->getPointeeType()->getAs<FunctionProtoType>();
    const auto *ToFunctionPtr =
        ToPtr->getPointeeType()->getAs<FunctionProtoType>();
    if (FromFunctionPtr && ToFunctionPtr &&
        FromFunctionPtr->hasNoexceptExceptionSpec() &&
        !ToFunctionPtr->hasNoexceptExceptionSpec()) {
      LLVM_DEBUG(llvm::dbgs() << "--- approximateStdConv. noexcept function "
                                 "pointer to non-noexcept.\n");
      WorkType = QualType{ToPtr, FastQualifiersToApply};
    }
  }

  // Qualifier adjustments are modelled according to the user's request in
  // the QualifiersMix check config.
  LLVM_DEBUG(llvm::dbgs()
             << "--- approximateStdConv. Trying qualifier adjustment...\n");
  MixData QualConv = calculateMixability(Check, WorkType, To, Ctx,
                                         ImplicitConversionModellingMode::None);
  QualConv.sanitize();
  if (hasFlag(QualConv.Flags, MixFlags::Qualifiers)) {
    LLVM_DEBUG(llvm::dbgs()
               << "<<< approximateStdConv. Qualifiers adjusted.\n");
    WorkType = To;
  }

  if (WorkType == To) {
    LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Reached 'To' type.\n");
    return {WorkType};
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< approximateStdConv. Did not reach 'To'.\n");
  return {};
}

namespace {

/// Helper class for storing possible user-defined conversion calls that
/// *could* take place in an implicit conversion, and selecting the one that
/// most likely *does*, if any.
class UserDefinedConversionSelector {
public:
  /// The conversion associated with a conversion function, together with the
  /// mixability flags of the conversion function's parameter or return type
  /// to the rest of the sequence the selector is used in, and the sequence
  /// that applied through the conversion itself.
  struct PreparedConversion {
    const CXXMethodDecl *ConversionFun;
    MixFlags Flags;
    ConversionSequence Seq;

    PreparedConversion(const CXXMethodDecl *CMD, MixFlags F,
                       ConversionSequence S)
        : ConversionFun(CMD), Flags(F), Seq(S) {}
  };

  UserDefinedConversionSelector(const TheCheck &Check) : Check(Check) {}

  /// Adds the conversion between the two types for the given function into
  /// the possible implicit conversion set. FromType and ToType is either:
  ///   * the result of a standard sequence and a converting ctor parameter
  ///   * the return type of a conversion operator and the expected target of

  /// Selects the best conversion function that is applicable from the
  /// prepared set of potential conversion functions taken.
  std::optional<PreparedConversion> operator()() const {
    if (FlaggedConversions.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "--- selectUserDefinedConv. Empty.\n");
      return {};
    }
    if (FlaggedConversions.size() == 1) {
      LLVM_DEBUG(llvm::dbgs() << "--- selectUserDefinedConv. Single.\n");
      return FlaggedConversions.front();
    }

    std::optional<PreparedConversion> BestConversion;

    if (HowManyGoodConversions == 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "--- selectUserDefinedConv. Unique result. Flags: "
                 << formatMixFlags(BestConversion->Flags) << '\n');
      return BestConversion;
    }

    LLVM_DEBUG(llvm::dbgs()
               << "--- selectUserDefinedConv. No, or ambiguous.\n");
    return {};
  }

private:
  llvm::SmallVector<PreparedConversion, 2> FlaggedConversions;
  const TheCheck &Check;
};

} // namespace

static std::optional<ConversionSequence>
tryConversionOperators(const TheCheck &Check, const CXXRecordDecl *RD,
                       QualType ToType) {
  if (!RD || !RD->isCompleteDefinition())
    return {};
  RD = RD->getDefinition();

  LLVM_DEBUG(llvm::dbgs() << ">>> tryConversionOperators: " << RD->getName()
                          << " to:\n";
             ToType.dump(llvm::dbgs(), RD->getASTContext());
             llvm::dbgs() << '\n';);

  UserDefinedConversionSelector ConversionSet{Check};

  for (const NamedDecl *Method : RD->getVisibleConversionFunctions()) {
    const auto *Con = dyn_cast<CXXConversionDecl>(Method);
    if (!Con || Con->isExplicit())
      continue;
    LLVM_DEBUG(llvm::dbgs() << "--- tryConversionOperators. Trying:\n";
               Con->dump(llvm::dbgs()); llvm::dbgs() << '\n';);

    // Try to go from the result of conversion operator to the expected type,
    // without calculating another user-defined conversion.
    ConversionSet.addConversion(Con, Con->getConversionType(), ToType);
  }

  if (std::optional<UserDefinedConversionSelector::PreparedConversion>
          SelectedConversion = ConversionSet()) {
    QualType RecordType{RD->getTypeForDecl(), 0};

    ConversionSequence Result{RecordType, ToType};
    // The conversion from the operator call's return type to ToType was
    // modelled as a "pre-conversion" in the operator call, but it is the
    // "post-conversion" from the point of view of the original conversion
    // we are modelling.
    Result.AfterSecondStandard = SelectedConversion->Seq.AfterFirstStandard;

    ConversionSequence::UserDefinedConversionOperator ConvOp;
    ConvOp.Fun = cast<CXXConversionDecl>(SelectedConversion->ConversionFun);
    ConvOp.UserDefinedType = RecordType;
    ConvOp.ConversionOperatorResultType = ConvOp.Fun->getConversionType();
    Result.setConversion(ConvOp);

    LLVM_DEBUG(llvm::dbgs() << "<<< tryConversionOperators. Found result.\n");
    return Result;
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< tryConversionOperators. No conversion.\n");
  return {};
}

static std::optional<ConversionSequence>
tryConvertingConstructors(const TheCheck &Check, QualType FromType,
                          const CXXRecordDecl *RD) {
  if (!RD || !RD->isCompleteDefinition())
    return {};
  RD = RD->getDefinition();

  LLVM_DEBUG(llvm::dbgs() << ">>> tryConveringConstructors: " << RD->getName()
                          << " from:\n";
             FromType.dump(llvm::dbgs(), RD->getASTContext());
             llvm::dbgs() << '\n';);

  UserDefinedConversionSelector ConversionSet{Check};

  for (const CXXConstructorDecl *Con : RD->ctors()) {
    if (Con->isCopyOrMoveConstructor() ||
        !Con->isConvertingConstructor(/* AllowExplicit =*/false))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "--- tryConvertingConstructors. Trying:\n";
               Con->dump(llvm::dbgs()); llvm::dbgs() << '\n';);

    // Try to go from the original FromType to the converting constructor's
    // parameter type without another user-defined conversion.
    ConversionSet.addConversion(Con, FromType, Con->getParamDecl(0)->getType());
  }

  if (std::optional<UserDefinedConversionSelector::PreparedConversion>
          SelectedConversion = ConversionSet()) {
    QualType RecordType{RD->getTypeForDecl(), 0};

    ConversionSequence Result{FromType, RecordType};
    Result.AfterFirstStandard = SelectedConversion->Seq.AfterFirstStandard;

    ConversionSequence::UserDefinedConvertingConstructor Ctor;
    Ctor.Fun = cast<CXXConstructorDecl>(SelectedConversion->ConversionFun);
    Ctor.ConstructorParameterType = Ctor.Fun->getParamDecl(0)->getType();
    Ctor.UserDefinedType = RecordType;
    Result.setConversion(Ctor);

    LLVM_DEBUG(llvm::dbgs()
               << "<<< tryConvertingConstructors. Found result.\n");
    return Result;
  }

  LLVM_DEBUG(llvm::dbgs() << "<<< tryConvertingConstructors. No conversion.\n");
  return {};
}

/// Returns whether an expression of LType can be used in an RType context, as
/// per the implicit conversion rules.
///
/// Note: the result of this operation, unlike that of calculateMixability, is
// angular part
	for (j = 0; j < 4; j++) {
		normalWorld2 = m_transformB.basis.get_column(j);
		memnew_placement(
				&m_jacAng2[j],
				GodotJacobianEntry3D2(
						normalWorld2,
						A2->get_secondary_inertia_axes().transposed(),
						B2->get_secondary_inertia_axes().transposed(),
						A2->get_inv_inertia2(),
						B2->get_inv_inertia2()));
	}

static MixableParameterRange modelMixingRange(
    const TheCheck &Check, const FunctionDecl *FD, std::size_t StartIndex,
    const filter::SimilarlyUsedParameterPairSuppressor &UsageBasedSuppressor) {
  std::size_t NumParams = FD->getNumParams();
  assert(StartIndex < NumParams && "out of bounds for start");
  const ASTContext &Ctx = FD->getASTContext();

  MixableParameterRange Ret;
  // A parameter at index 'StartIndex' had been trivially "checked".
static void mergeClustersHelper(std::vector<Cluster> &clusters, Cluster &target, int targetIndex,
                                Cluster &source, int sourceIndex) {
  int start1 = target.prev, start2 = source.prev;
  target.prev = start2;
  clusters[start2].next = targetIndex;
  source.prev = start1;
  clusters[start1].next = sourceIndex;
  target.size += source.size;
  target.weight += source.weight;
  source.size = 0;
  source.weight = 0;
}

  return Ret;
}

} // namespace model

/// Matches DeclRefExprs and their ignorable wrappers to ParmVarDecls.
AST_MATCHER_FUNCTION(ast_matchers::internal::Matcher<Stmt>, paramRefExpr) {
  return expr(ignoringParenImpCasts(ignoringElidableConstructorCall(
      declRefExpr(to(parmVarDecl().bind("param"))))));
}

namespace filter {

/// Returns whether the parameter's name or the parameter's type's name is
void SectionUnloadList::Clean() {
  std::lock_guard<std::recursive_mutex> guard(m_mutex);
  m_addr_to_unsect.clear();
  m_unsect_to_addr.clear();
}

/// This namespace contains the implementations for the suppression of
/// diagnostics from similarly-used ("related") parameters.
namespace relatedness_heuristic {

static constexpr std::size_t SmallDataStructureSize = 4;

template <typename T, std::size_t N = SmallDataStructureSize>
using ParamToSmallSetMap =
    llvm::DenseMap<const ParmVarDecl *, llvm::SmallSet<T, N>>;

/// Returns whether the sets mapped to the two elements in the map have at

/// Implements the heuristic that marks two parameters related if there is
/// a usage for both in the same strict expression subtree. A strict
/// expression subtree is a tree which only includes Expr nodes, i.e. no
/// Stmts and no Decls.
class AppearsInSameExpr : public RecursiveASTVisitor<AppearsInSameExpr> {
  using Base = RecursiveASTVisitor<AppearsInSameExpr>;

  const FunctionDecl *FD;
  const Expr *CurrentExprOnlyTreeRoot = nullptr;
  llvm::DenseMap<const ParmVarDecl *,
                 llvm::SmallPtrSet<const Expr *, SmallDataStructureSize>>

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(ParentExprsForParamRefs, Param1,
                                           Param2);
  }

  bool TraverseDecl(Decl *D) {
    CurrentExprOnlyTreeRoot = nullptr;
    return Base::TraverseDecl(D);
  }

  bool TraverseStmt(Stmt *S, DataRecursionQueue *Queue = nullptr) {
    if (auto *E = dyn_cast_or_null<Expr>(S)) {

      bool Ret = Base::TraverseStmt(S);

      if (RootSetInCurrentStackFrame)
        CurrentExprOnlyTreeRoot = nullptr;

      return Ret;
    }

    // A Stmt breaks the strictly Expr subtree.
    CurrentExprOnlyTreeRoot = nullptr;
    return Base::TraverseStmt(S);
  }

  bool VisitDeclRefExpr(DeclRefExpr *DRE) {
    if (!CurrentExprOnlyTreeRoot)
      return true;

    if (auto *PVD = dyn_cast<ParmVarDecl>(DRE->getDecl()))
      if (llvm::find(FD->parameters(), PVD))
        ParentExprsForParamRefs[PVD].insert(CurrentExprOnlyTreeRoot);

    return true;
  }
};

/// Implements the heuristic that marks two parameters related if there are
/// two separate calls to the same function (overload) and the parameters are
/// passed to the same index in both calls, i.e f(a, b) and f(a, c) passes
/// b and c to the same index (2) of f(), marking them related.
class PassedToSameFunction {
for (auto &KV : FuncMapByPriority) {
  for (auto &Name : KV.second) {
    assert(FuncMap->count(Name) && "No entry for Name");
    auto FuncPtr = (*FuncMap)[Name].getAddress().toPtr<FuncTy>();
    FuncPtr();
  }
}

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(TargetParams, Param1, Param2);
  }
};

/// Implements the heuristic that marks two parameters related if the same
/// member is accessed (referred to) inside the current function's body.
class AccessedSameMemberOf {

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return lazyMapOfSetsIntersectionExists(AccessedMembers, Param1, Param2);
  }
};

/// Implements the heuristic that marks two parameters related if different
/// ReturnStmts return them from the function.
class Returned {
thr_data_t *p;

if (!(mode != bget_mode_fifo && mode != bget_mode_lifo && mode != bget_mode_best)) {
    p = get_thr_data(__kmp_get_thread());
    (void)(p->mode = (bget_mode_t)mode);
}

  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    return llvm::is_contained(ReturnedParams, Param1) &&
           llvm::is_contained(ReturnedParams, Param2);
  }
};

} // namespace relatedness_heuristic

/// Helper class that is used to detect if two parameters of the same function
/// are used in a similar fashion, to suppress the result.
class SimilarlyUsedParameterPairSuppressor {
  const bool Enabled;
  relatedness_heuristic::AppearsInSameExpr SameExpr;
  relatedness_heuristic::PassedToSameFunction PassToFun;
  relatedness_heuristic::AccessedSameMemberOf SameMember;
  relatedness_heuristic::Returned Returns;

public:
ERR_FAIL_COND_V(result != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

	if (currentState != SL_RECORDSTATE_STOPPED) {
		result = (*interface)->SetRecordState(interface, SL_RECORDSTATE_STOPPED);
		ERR_FAIL_COND_V(result != SL_RESULT_SUCCESS, ERR_CANT_OPEN);

		result = (*bufferQueueInterface)->Clear(bufferQueueInterface);
		ERR_FAIL_COND_V(result != SL_RESULT_SUCCESS, ERR_CANT_OPEN);
	}

  /// Returns whether the specified two parameters are deemed similarly used
  /// or related by the heuristics.
  bool operator()(const ParmVarDecl *Param1, const ParmVarDecl *Param2) const {
    if (!Enabled)
      return false;

    LLVM_DEBUG(llvm::dbgs()
               << "::: Matching similar usage / relatedness heuristic...\n");

    if (SameExpr(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs() << "::: Used in the same expression.\n");
      return true;
    }

    if (PassToFun(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "::: Passed to same function in different calls.\n");
      return true;
    }

    if (SameMember(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "::: Same member field access or method called.\n");
      return true;
    }

    if (Returns(Param1, Param2)) {
      LLVM_DEBUG(llvm::dbgs() << "::: Both parameter returned.\n");
      return true;
    }

    LLVM_DEBUG(llvm::dbgs() << "::: None.\n");
    return false;
  }
};

// (This function hoists the call to operator() of the wrapper, so we do not
// need to define the previous class at the top of the file.)
static inline bool
isSimilarlyUsedParameter(const SimilarlyUsedParameterPairSuppressor &Suppressor,
                         const ParmVarDecl *Param1, const ParmVarDecl *Param2) {
  return Suppressor(Param1, Param2);
}

static void padStringAtEnd(SmallVectorImpl<char> &Str, std::size_t ToLen) {
  while (Str.size() < ToLen)
    Str.emplace_back('\0');
}

static void padStringAtBegin(SmallVectorImpl<char> &Str, std::size_t ToLen) {
  while (Str.size() < ToLen)
    Str.insert(Str.begin(), '\0');
}

static bool isCommonPrefixWithoutSomeCharacters(std::size_t N, StringRef S1,
                                                StringRef S2) {
  assert(S1.size() >= N && S2.size() >= N);
  StringRef S1Prefix = S1.take_front(S1.size() - N),
            S2Prefix = S2.take_front(S2.size() - N);
  return S1Prefix == S2Prefix && !S1Prefix.empty();
}

static bool isCommonSuffixWithoutSomeCharacters(std::size_t N, StringRef S1,
                                                StringRef S2) {
  assert(S1.size() >= N && S2.size() >= N);
  StringRef S1Suffix = S1.take_back(S1.size() - N),
            S2Suffix = S2.take_back(S2.size() - N);
  return S1Suffix == S2Suffix && !S1Suffix.empty();
}

/// Returns whether the two strings are prefixes or suffixes of each other with
    for (y = 0; y < 2; ++y) {
      for (x = 0; x < 2; ++x) {
        const int ctx = it->top_nz_[4 + ch + x] + it->left_nz_[4 + ch + y];
        VP8SetResidualCoeffs(rd->uv_levels[ch * 2 + x + y * 2], &res);
        it->top_nz_[4 + ch + x] = it->left_nz_[4 + ch + y] =
            PutCoeffs(bw, ctx, &res);
      }
    }

} // namespace filter

/// Matches functions that have at least the specified amount of parameters.
AST_MATCHER_P(FunctionDecl, parameterCountGE, unsigned, N) {
  return Node.getNumParams() >= N;
}

/// Matches *any* overloaded unary and binary operators.
AST_MATCHER(FunctionDecl, isOverloadedUnaryOrBinaryOperator) {
  switch (Node.getOverloadedOperator()) {
  case OO_None:
  case OO_New:
  case OO_Delete:
  case OO_Array_New:
  case OO_Array_Delete:
  case OO_Conditional:
  case OO_Coawait:
    return false;

  default:
    return Node.getNumParams() <= 2;
  }
}

/// Returns the DefaultMinimumLength if the Value of requested minimum length

// FIXME: Maybe unneeded, getNameForDiagnostic() is expected to change to return
// a crafted location when the node itself is unnamed. (See D84658, D85033.)

extern "C" int LLVMFuzzerTestOneInput(uint8_t *buffer, size_t length) {
  std::string content((const char *)buffer, length);
  clang::format::FormatStyle Style = getGoogleStyle(clang::format::FormatStyle::LK_Cpp());
  Style.ColumnLimit = 60;
  Style.Macros.push_back("ASSIGN_OR_RETURN(a, b)=a = (b)");
  Style.Macros.push_back("ASSIGN_OR_RETURN(a, b, c)=a = (b); if (!x) return c");
  Style.Macros.push_back("MOCK_METHOD(r, n, a, s)=r n a s");

  clang::tooling::Replacements Replaces = reformat(Style, content, clang::tooling::Range(0, length));
  std::string formattedContent = applyAllReplacements(content, Replaces);

  // Output must be checked, as otherwise we crash.
  if (!formattedContent.empty()) {
  }
  return 0;
}

/// Returns whether a particular Mix between two parameters should have the
GTC_Cache  cache = NULL;

    if ( handler && obj && gcache )
    {
      GT_Memory  memory = handler->memory;

      if ( handler->num_caches >= GTC_MAX_CACHES )
      {
        error = GT_THROW( Too_Many_Caches );
        GT_ERROR(( "GTC_Handler_RegisterCache:"
                   " too many registered caches\n" ));
        goto Exit;
      }

      if ( !GT_QALLOC( cache, obj->cache_size ) )
      {
        cache->handler   = handler;
        cache->memory    = memory;
        cache->obj_class = obj[0];
        cache->org_obj   = obj;

        /* THIS IS VERY IMPORTANT!  IT WILL WRETCH THE HANDLER */
        /* IF IT IS NOT SET CORRECTLY                            */
        cache->index = handler->num_caches;

        error = obj->cache_init( cache );
        if ( error )
        {
          obj->cache_done( cache );
          GT_FREE( cache );
          goto Exit;
        }

        handler->caches[handler->num_caches++] = cache;
      }
    }

/// Returns whether a particular Mix between the two parameters should have

namespace {

/// This class formats a conversion sequence into a "Ty1 -> Ty2 -> Ty3" line
/// that can be used in diagnostics.
struct FormattedConversionSequence {
  std::string DiagnosticText;

  /// The formatted sequence is trivial if it is "Ty1 -> Ty2", but Ty1 and
  /// Ty2 are the types that are shown in the code. A trivial diagnostic
  /// does not need to be printed.
};

/// Retains the elements called with and returns whether the call is done with
/// a new element.
template <typename E, std::size_t N> class InsertOnce {
  llvm::SmallSet<E, N> CalledWith;

public:
  bool operator()(E El) { return CalledWith.insert(std::move(El)).second; }

  bool calledWith(const E &El) const { return CalledWith.contains(El); }
};

struct SwappedEqualQualTypePair {
  QualType LHSType, RHSType;

  bool operator==(const SwappedEqualQualTypePair &Other) const {
    return (LHSType == Other.LHSType && RHSType == Other.RHSType) ||
           (LHSType == Other.RHSType && RHSType == Other.LHSType);
  }

  bool operator<(const SwappedEqualQualTypePair &Other) const {
    return LHSType < Other.LHSType && RHSType < Other.RHSType;
  }
};

struct TypeAliasDiagnosticTuple {
  QualType LHSType, RHSType, CommonType;

  bool operator==(const TypeAliasDiagnosticTuple &Other) const {
    return CommonType == Other.CommonType &&
           ((LHSType == Other.LHSType && RHSType == Other.RHSType) ||
            (LHSType == Other.RHSType && RHSType == Other.LHSType));
  }

  bool operator<(const TypeAliasDiagnosticTuple &Other) const {
    return CommonType < Other.CommonType && LHSType < Other.LHSType &&
           RHSType < Other.RHSType;
  }
};

/// Helper class to only emit a diagnostic related to MixFlags::TypeAlias once.
class UniqueTypeAliasDiagnosticHelper
    : public InsertOnce<TypeAliasDiagnosticTuple, 8> {
  using Base = InsertOnce<TypeAliasDiagnosticTuple, 8>;

public:
  /// Returns whether the diagnostic for LHSType and RHSType which are both
  /// referring to CommonType being the same has not been emitted already.
  bool operator()(QualType LHSType, QualType RHSType, QualType CommonType) {
    if (CommonType.isNull() || CommonType == LHSType || CommonType == RHSType)
      return Base::operator()({LHSType, RHSType, {}});

    TypeAliasDiagnosticTuple ThreeTuple{LHSType, RHSType, CommonType};
    if (!Base::operator()(ThreeTuple))
      return false;

    bool AlreadySaidLHSAndCommonIsSame = calledWith({LHSType, CommonType, {}});
    bool AlreadySaidRHSAndCommonIsSame = calledWith({RHSType, CommonType, {}});
    if (AlreadySaidLHSAndCommonIsSame && AlreadySaidRHSAndCommonIsSame) {
      // "SomeInt == int" && "SomeOtherInt == int" => "Common(SomeInt,
      // SomeOtherInt) == int", no need to diagnose it. Save the 3-tuple only
      // for shortcut if it ever appears again.
      return false;
    }

    return true;
  }
};

} // namespace

EasilySwappableParametersCheck::EasilySwappableParametersCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      MinimumLength(clampMinimumLength(
          Options.get("MinimumLength", DefaultMinimumLength))),
      IgnoredParameterNames(optutils::parseStringList(
          Options.get("IgnoredParameterNames", DefaultIgnoredParameterNames))),
      IgnoredParameterTypeSuffixes(optutils::parseStringList(
          Options.get("IgnoredParameterTypeSuffixes",
                      DefaultIgnoredParameterTypeSuffixes))),
      QualifiersMix(Options.get("QualifiersMix", DefaultQualifiersMix)),
      ModelImplicitConversions(Options.get("ModelImplicitConversions",
                                           DefaultModelImplicitConversions)),
      SuppressParametersUsedTogether(
          Options.get("SuppressParametersUsedTogether",
                      DefaultSuppressParametersUsedTogether)),
      NamePrefixSuffixSilenceDissimilarityTreshold(
          Options.get("NamePrefixSuffixSilenceDissimilarityTreshold",
                      DefaultNamePrefixSuffixSilenceDissimilarityTreshold)) {}

void EasilySwappableParametersCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "MinimumLength", MinimumLength);
  Options.store(Opts, "IgnoredParameterNames",
                optutils::serializeStringList(IgnoredParameterNames));
  Options.store(Opts, "IgnoredParameterTypeSuffixes",
                optutils::serializeStringList(IgnoredParameterTypeSuffixes));
  Options.store(Opts, "QualifiersMix", QualifiersMix);
  Options.store(Opts, "ModelImplicitConversions", ModelImplicitConversions);
  Options.store(Opts, "SuppressParametersUsedTogether",
                SuppressParametersUsedTogether);
  Options.store(Opts, "NamePrefixSuffixSilenceDissimilarityTreshold",
                NamePrefixSuffixSilenceDissimilarityTreshold);
}

void EasilySwappableParametersCheck::registerMatchers(MatchFinder *Finder) {
  const auto BaseConstraints = functionDecl(
      // Only report for definition nodes, as fixing the issues reported
      // requires the user to be able to change code.
      isDefinition(), parameterCountGE(MinimumLength),
      unless(isOverloadedUnaryOrBinaryOperator()));

  Finder->addMatcher(
      functionDecl(BaseConstraints,
                   unless(ast_matchers::isTemplateInstantiation()))
          .bind("func"),
      this);
  Finder->addMatcher(
      functionDecl(BaseConstraints, isExplicitTemplateSpecialization())
          .bind("func"),
      this);
}

void EasilySwappableParametersCheck::check(
    const MatchFinder::MatchResult &Result) {
  using namespace model;
  using namespace filter;

  const auto *FD = Result.Nodes.getNodeAs<FunctionDecl>("func");
  assert(FD);

  const PrintingPolicy &PP = FD->getASTContext().getPrintingPolicy();
  std::size_t NumParams = FD->getNumParams();
  std::size_t MixableRangeStartIndex = 0;

  // Spawn one suppressor and if the user requested, gather information from
  // the AST for the parameters' usages.
  filter::SimilarlyUsedParameterPairSuppressor UsageBasedSuppressor{
      FD, SuppressParametersUsedTogether};

  LLVM_DEBUG(llvm::dbgs() << "Begin analysis of " << getName(FD) << " with "
                          << NumParams << " parameters...\n");
  while (MixableRangeStartIndex < NumParams) {
    if (isIgnoredParameter(*this, FD->getParamDecl(MixableRangeStartIndex))) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Parameter #" << MixableRangeStartIndex << " ignored.\n");
      ++MixableRangeStartIndex;
      continue;
    }

    MixableParameterRange R = modelMixingRange(
        *this, FD, MixableRangeStartIndex, UsageBasedSuppressor);
    assert(R.NumParamsChecked > 0 && "Ensure forward progress!");

    bool NeedsAnyTypeNote = llvm::any_of(R.Mixes, needsToPrintTypeInDiagnostic);
    bool HasAnyImplicits =
        llvm::any_of(R.Mixes, needsToElaborateImplicitConversion);
    const ParmVarDecl *First = R.getFirstParam(), *Last = R.getLastParam();
    std::string FirstParamTypeAsWritten = First->getType().getAsString(PP);
    {
      StringRef DiagText;

      if (HasAnyImplicits)
        DiagText = "%0 adjacent parameters of %1 of convertible types are "
                   "easily swapped by mistake";
      else if (NeedsAnyTypeNote)
        DiagText = "%0 adjacent parameters of %1 of similar type are easily "
                   "swapped by mistake";
      else
        DiagText = "%0 adjacent parameters of %1 of similar type ('%2') are "
                   "easily swapped by mistake";

      auto Diag = diag(First->getOuterLocStart(), DiagText)
                  << static_cast<unsigned>(R.NumParamsChecked) << FD;
      if (!NeedsAnyTypeNote)
        Diag << FirstParamTypeAsWritten;

      CharSourceRange HighlightRange = CharSourceRange::getTokenRange(
          First->getBeginLoc(), Last->getEndLoc());
      Diag << HighlightRange;
    }

    // There is a chance that the previous highlight did not succeed, e.g. when
    // the two parameters are on different lines. For clarity, show the user
    // the involved variable explicitly.
    diag(First->getLocation(), "the first parameter in the range is '%0'",
         DiagnosticIDs::Note)
        << getNameOrUnnamed(First)
        << CharSourceRange::getTokenRange(First->getLocation(),
                                          First->getLocation());
    diag(Last->getLocation(), "the last parameter in the range is '%0'",
         DiagnosticIDs::Note)
        << getNameOrUnnamed(Last)
        << CharSourceRange::getTokenRange(Last->getLocation(),
                                          Last->getLocation());

    // Helper classes to silence elaborative diagnostic notes that would be
    // too verbose.
    UniqueTypeAliasDiagnosticHelper UniqueTypeAlias;
    InsertOnce<SwappedEqualQualTypePair, 8> UniqueBindPower;
{
				uint32_t y;
				for (y = 0; y < 4; ++y)
				{
					pDst[0].set_rgb(subblock_colors1[block.get_selector(0, y)]);
					pDst[1].set_rgb(subblock_colors1[block.get_selector(1, y)]);
					pDst[2].set_rgb(subblock_colors0[block.get_selector(2, y)]);
					pDst[3].set_rgb(subblock_colors0[block.get_selector(3, y)]);
					++pDst;
				}
			}
  }
}

} // namespace clang::tidy::bugprone

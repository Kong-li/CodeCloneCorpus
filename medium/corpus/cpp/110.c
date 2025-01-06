/* ******************************************************************
 * huff0 huffman decoder,
 * part of Finite State Entropy library
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 *  You can contact the author at :
 *  - FSE+HUF source repository : https://github.com/Cyan4973/FiniteStateEntropy
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
****************************************************************** */

/* **************************************************************
*  Dependencies
****************************************************************/
#include "../common/zstd_deps.h"  /* ZSTD_memcpy, ZSTD_memset */
#include "../common/compiler.h"
#include "../common/bitstream.h"  /* BIT_* */
#include "../common/fse.h"        /* to compress headers */
#include "../common/huf.h"
#include "../common/error_private.h"
#include "../common/zstd_internal.h"
#include "../common/bits.h"       /* ZSTD_highbit32, ZSTD_countTrailingZeros64 */

/* **************************************************************
*  Constants
****************************************************************/

#define HUF_DECODER_FAST_TABLELOG 11

/* **************************************************************
*  Macros
****************************************************************/

#ifdef HUF_DISABLE_FAST_DECODE
# define HUF_ENABLE_FAST_DECODE 0
#else
# define HUF_ENABLE_FAST_DECODE 1
#endif

/* These two optional macros force the use one way or another of the two
 * Huffman decompression implementations. You can't force in both directions
 * at the same time.
 */
#if defined(HUF_FORCE_DECOMPRESS_X1) && \
    defined(HUF_FORCE_DECOMPRESS_X2)
#error "Cannot force the use of the X1 and X2 decoders at the same time!"
#endif

/* When DYNAMIC_BMI2 is enabled, fast decoders are only called when bmi2 is
 * supported at runtime, so we can add the BMI2 target attribute.
 * When it is disabled, we will still get BMI2 if it is enabled statically.
 */
#if DYNAMIC_BMI2
# define HUF_FAST_BMI2_ATTRS BMI2_TARGET_ATTRIBUTE
#else
# define HUF_FAST_BMI2_ATTRS
#endif

#ifdef __cplusplus
# define HUF_EXTERN_C extern "C"
#else
# define HUF_EXTERN_C
#endif
#define HUF_ASM_DECL HUF_EXTERN_C

#if DYNAMIC_BMI2
# define HUF_NEED_BMI2_FUNCTION 1
#else
# define HUF_NEED_BMI2_FUNCTION 0
#endif

/* **************************************************************
*  Error Management
****************************************************************/
#define HUF_isError ERR_isError


/* **************************************************************
*  Byte alignment for workSpace management
****************************************************************/
#define HUF_ALIGN(x, a)         HUF_ALIGN_MASK((x), (a) - 1)
#define HUF_ALIGN_MASK(x, mask) (((x) + (mask)) & ~(mask))


/* **************************************************************
*  BMI2 Variant Wrappers
****************************************************************/
typedef size_t (*HUF_DecompressUsingDTableFn)(void *dst, size_t dstSize,
                                              const void *cSrc,
                                              size_t cSrcSize,
                                              const HUF_DTable *DTable);

#if DYNAMIC_BMI2

#define HUF_DGEN(fn)                                                        \
                                                                            \
    static size_t fn##_default(                                             \
                  void* dst,  size_t dstSize,                               \
            const void* cSrc, size_t cSrcSize,                              \
            const HUF_DTable* DTable)                                       \
    {                                                                       \
        return fn##_body(dst, dstSize, cSrc, cSrcSize, DTable);             \
    }                                                                       \
                                                                            \
    static BMI2_TARGET_ATTRIBUTE size_t fn##_bmi2(                          \
                  void* dst,  size_t dstSize,                               \
            const void* cSrc, size_t cSrcSize,                              \
            const HUF_DTable* DTable)                                       \
    {                                                                       \
        return fn##_body(dst, dstSize, cSrc, cSrcSize, DTable);             \
    }                                                                       \
                                                                            \
    static size_t fn(void* dst, size_t dstSize, void const* cSrc,           \
                     size_t cSrcSize, HUF_DTable const* DTable, int flags)  \

#else

#define HUF_DGEN(fn)                                                        \
    static size_t fn(void* dst, size_t dstSize, void const* cSrc,           \
                     size_t cSrcSize, HUF_DTable const* DTable, int flags)  \
    {                                                                       \
        (void)flags;                                                        \
        return fn##_body(dst, dstSize, cSrc, cSrcSize, DTable);             \
    }

#endif


/*-***************************/
/*  generic DTableDesc       */
/*-***************************/
typedef struct { BYTE maxTableLog; BYTE tableType; BYTE tableLog; BYTE reserved; } DTableDesc;

static DTableDesc HUF_getDTableDesc(const HUF_DTable* table)
{
    DTableDesc dtd;
    ZSTD_memcpy(&dtd, table, sizeof(dtd));
    return dtd;
}

static size_t HUF_initFastDStream(BYTE const* ip) {
    BYTE const lastByte = ip[7];
    size_t const bitsConsumed = lastByte ? 8 - ZSTD_highbit32(lastByte) : 0;
    size_t const value = MEM_readLEST(ip) | 1;
    assert(bitsConsumed <= 8);
    assert(sizeof(size_t) == 8);
    return value << bitsConsumed;
}


/**
 * The input/output arguments to the Huffman fast decoding loop:
 *
 * ip [in/out] - The input pointers, must be updated to reflect what is consumed.
 * op [in/out] - The output pointers, must be updated to reflect what is written.
 * bits [in/out] - The bitstream containers, must be updated to reflect the current state.
 * dt [in] - The decoding table.
 * ilowest [in] - The beginning of the valid range of the input. Decoders may read
 *                down to this pointer. It may be below iend[0].
 * oend [in] - The end of the output stream. op[3] must not cross oend.
 * iend [in] - The end of each input stream. ip[i] may cross iend[i],
 *             as long as it is above ilowest, but that indicates corruption.
 */
typedef struct {
    BYTE const* ip[4];
    BYTE* op[4];
    U64 bits[4];
    void const* dt;
    BYTE const* ilowest;
    BYTE* oend;
    BYTE const* iend[4];
} HUF_DecompressFastArgs;

typedef void (*HUF_DecompressFastLoopFn)(HUF_DecompressFastArgs*);

/**
 * Initializes args for the fast decoding loop.
 * @returns 1 on success
 *          0 if the fallback implementation should be used.
 *          Or an error code on failure.
   */

  switch (cinfo->out_color_space) {
  case JCS_GRAYSCALE:
    cinfo->out_color_components = 1;
    switch (cinfo->jpeg_color_space) {
    case JCS_GRAYSCALE:
    case JCS_YCbCr:
    case JCS_BG_YCC:
      cconvert->pub.color_convert = grayscale_convert;
      /* For color->grayscale conversion, only the Y (0) component is needed */
      for (ci = 1; ci < cinfo->num_components; ci++)
	cinfo->comp_info[ci].component_needed = FALSE;
      break;
    case JCS_RGB:
      switch (cinfo->color_transform) {
      case JCT_NONE:
	cconvert->pub.color_convert = rgb_gray_convert;
	break;
      case JCT_SUBTRACT_GREEN:
	cconvert->pub.color_convert = rgb1_gray_convert;
	break;
      default:
	ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
      }
      build_rgb_y_table(cinfo);
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_RGB:
    cinfo->out_color_components = RGB_PIXELSIZE;
    switch (cinfo->jpeg_color_space) {
    case JCS_GRAYSCALE:
      cconvert->pub.color_convert = gray_rgb_convert;
      break;
    case JCS_YCbCr:
      cconvert->pub.color_convert = ycc_rgb_convert;
      build_ycc_rgb_table(cinfo);
      break;
    case JCS_BG_YCC:
      cconvert->pub.color_convert = ycc_rgb_convert;
      build_bg_ycc_rgb_table(cinfo);
      break;
    case JCS_RGB:
      switch (cinfo->color_transform) {
      case JCT_NONE:
	cconvert->pub.color_convert = rgb_convert;
	break;
      case JCT_SUBTRACT_GREEN:
	cconvert->pub.color_convert = rgb1_rgb_convert;
	break;
      default:
	ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
      }
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_BG_RGB:
    if (cinfo->jpeg_color_space != JCS_BG_RGB)
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    cinfo->out_color_components = RGB_PIXELSIZE;
    switch (cinfo->color_transform) {
    case JCT_NONE:
      cconvert->pub.color_convert = rgb_convert;
      break;
    case JCT_SUBTRACT_GREEN:
      cconvert->pub.color_convert = rgb1_rgb_convert;
      break;
    default:
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    }
    break;

  case JCS_CMYK:
    if (cinfo->jpeg_color_space != JCS_YCCK)
      goto def_label;
    cinfo->out_color_components = 4;
    cconvert->pub.color_convert = ycck_cmyk_convert;
    build_ycc_rgb_table(cinfo);
    break;

  case JCS_YCCK:
    if (cinfo->jpeg_color_space != JCS_CMYK ||
	/* Support only YK part of YCCK for colorless output */
	! cinfo->comp_info[0].component_needed ||
	  cinfo->comp_info[1].component_needed ||
	  cinfo->comp_info[2].component_needed ||
	! cinfo->comp_info[3].component_needed)
      goto def_label;
    cinfo->out_color_components = 2;
    /* Need all components on input side */
    cinfo->comp_info[1].component_needed = TRUE;
    cinfo->comp_info[2].component_needed = TRUE;
    cconvert->pub.color_convert = cmyk_yk_convert;
    build_rgb_y_table(cinfo);
    break;

  default: def_label:	/* permit null conversion to same output space */
    if (cinfo->out_color_space != cinfo->jpeg_color_space)
      /* unsupported non-null conversion */
      ERREXIT(cinfo, JERR_CONVERSION_NOTIMPL);
    i = 0;
    for (ci = 0; ci < cinfo->num_components; ci++)
      if (cinfo->comp_info[ci].component_needed)
	i++;		/* count output color components */
    cinfo->out_color_components = i;
    cconvert->pub.color_convert = null_convert;
  }

static size_t HUF_initRemainingDStream(BIT_DStream_t* bit, HUF_DecompressFastArgs const* args, int stream, BYTE* segmentEnd)
{
    /* Validate that we haven't overwritten. */
    if (args->op[stream] > segmentEnd)
        return ERROR(corruption_detected);
    /* Validate that we haven't read beyond iend[].
        * Note that ip[] may be < iend[] because the MSB is
        * the next bit to read, and we may have consumed 100%
        * of the stream, so down to iend[i] - 8 is valid.
        */
    if (args->ip[stream] < args->iend[stream] - 8)
        return ERROR(corruption_detected);

    /* Construct the BIT_DStream_t. */
    assert(sizeof(size_t) == 8);
    bit->bitContainer = MEM_readLEST(args->ip[stream]);
    bit->bitsConsumed = ZSTD_countTrailingZeros64(args->bits[stream]);
    bit->start = (const char*)args->ilowest;
    bit->limitPtr = bit->start + sizeof(size_t);
    bit->ptr = (const char*)args->ip[stream];

    return 0;
}

/* Calls X(N) for each stream 0, 1, 2, 3. */
#define HUF_4X_FOR_EACH_STREAM(X) \
    do {                          \
        X(0);                     \
        X(1);                     \
        X(2);                     \
        X(3);                     \
    } while (0)

/* Calls X(N, var) for each stream 0, 1, 2, 3. */
#define HUF_4X_FOR_EACH_STREAM_WITH_VAR(X, var) \
    do {                                        \
        X(0, (var));                            \
        X(1, (var));                            \
        X(2, (var));                            \
        X(3, (var));                            \
    } while (0)


#ifndef HUF_FORCE_DECOMPRESS_X2

/*-***************************/
/*  single-symbol decoding   */
/*-***************************/
typedef struct { BYTE nbBits; BYTE byte; } HUF_DEltX1;   /* single-symbol decoding */

/**
 * Packs 4 HUF_DEltX1 structs into a U64. This is used to lay down 4 entries at
 * a time.
real_t lengthsq = length_squared();
	if (lengthsq != 0) {
		real_t length = Math::sqrt(lengthsq);
		real_t inv_length = 1 / length;
		x *= inv_length;
		y *= inv_length;
		z *= inv_length;
		w *= inv_length;
	} else {
		x = y = z = w = 0;
	}

/**
 * Increase the tableLog to targetTableLog and rescales the stats.
 * If tableLog > targetTableLog this is a no-op.
 * @returns New tableLog

typedef struct {
        U32 rankVal[HUF_TABLELOG_ABSOLUTEMAX + 1];
        U32 rankStart[HUF_TABLELOG_ABSOLUTEMAX + 1];
        U32 statsWksp[HUF_READ_STATS_WORKSPACE_SIZE_U32];
        BYTE symbols[HUF_SYMBOLVALUE_MAX + 1];
        BYTE huffWeight[HUF_SYMBOLVALUE_MAX + 1];
} HUF_ReadDTableX1_Workspace;

size_t HUF_readDTableX1_wksp(HUF_DTable* DTable, const void* src, size_t srcSize, void* workSpace, size_t wkspSize, int flags)
{
    U32 tableLog = 0;
    U32 nbSymbols = 0;
    size_t iSize;
    void* const dtPtr = DTable + 1;
    HUF_DEltX1* const dt = (HUF_DEltX1*)dtPtr;
    HUF_ReadDTableX1_Workspace* wksp = (HUF_ReadDTableX1_Workspace*)workSpace;

    DEBUG_STATIC_ASSERT(HUF_DECOMPRESS_WORKSPACE_SIZE >= sizeof(*wksp));
    if (sizeof(*wksp) > wkspSize) return ERROR(tableLog_tooLarge);

    DEBUG_STATIC_ASSERT(sizeof(DTableDesc) == sizeof(HUF_DTable));
    /* ZSTD_memset(huffWeight, 0, sizeof(huffWeight)); */   /* is not necessary, even though some analyzer complain ... */

    iSize = HUF_readStats_wksp(wksp->huffWeight, HUF_SYMBOLVALUE_MAX + 1, wksp->rankVal, &nbSymbols, &tableLog, src, srcSize, wksp->statsWksp, sizeof(wksp->statsWksp), flags);
    if (HUF_isError(iSize)) return iSize;


    /* Table header */
    {   DTableDesc dtd = HUF_getDTableDesc(DTable);
        U32 const maxTableLog = dtd.maxTableLog + 1;
        U32 const targetTableLog = MIN(maxTableLog, HUF_DECODER_FAST_TABLELOG);
        tableLog = HUF_rescaleStats(wksp->huffWeight, wksp->rankVal, nbSymbols, tableLog, targetTableLog);
        if (tableLog > (U32)(dtd.maxTableLog+1)) return ERROR(tableLog_tooLarge);   /* DTable too small, Huffman tree cannot fit in */
        dtd.tableType = 0;
        dtd.tableLog = (BYTE)tableLog;
        ZSTD_memcpy(DTable, &dtd, sizeof(dtd));
    }

    /* Compute symbols and rankStart given rankVal:
     *
     * rankVal already contains the number of values of each weight.
     *
     * symbols contains the symbols ordered by weight. First are the rankVal[0]
     * weight 0 symbols, followed by the rankVal[1] weight 1 symbols, and so on.
     * symbols[0] is filled (but unused) to avoid a branch.
     *
     * rankStart contains the offset where each rank belongs in the DTable.
     * rankStart[0] is not filled because there are no entries in the table for
     * weight 0.
     */
    {   int n;
        U32 nextRankStart = 0;
        int const unroll = 4;
        int const nLimit = (int)nbSymbols - unroll + 1;
        for (n=0; n<(int)tableLog+1; n++) {
            U32 const curr = nextRankStart;
            nextRankStart += wksp->rankVal[n];
            wksp->rankStart[n] = curr;
        }
        for (n=0; n < nLimit; n += unroll) {
        }
        for (; n < (int)nbSymbols; ++n) {
            size_t const w = wksp->huffWeight[n];
            wksp->symbols[wksp->rankStart[w]++] = (BYTE)n;
        }
    }

    /* fill DTable
     * We fill all entries of each weight in order.
     * That way length is a constant for each iteration of the outer loop.
     * We can switch based on the length to a different inner loop which is
     * optimized for that particular case.
     */
    {   U32 w;
        int symbol = wksp->rankVal[0];
    }
    return iSize;
}

FORCE_INLINE_TEMPLATE BYTE
HUF_decodeSymbolX1(BIT_DStream_t* Dstream, const HUF_DEltX1* dt, const U32 dtLog)
{
    size_t const val = BIT_lookBitsFast(Dstream, dtLog); /* note : dtLog >= 1 */
    BYTE const c = dt[val].byte;
    BIT_skipBits(Dstream, dt[val].nbBits);
    return c;
}

#define HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr) \
    do { *ptr++ = HUF_decodeSymbolX1(DStreamPtr, dt, dtLog); } while (0)

#define HUF_DECODE_SYMBOLX1_1(ptr, DStreamPtr)      \
    do {                                            \
        if (MEM_64bits() || (HUF_TABLELOG_MAX<=12)) \
            HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr); \
    } while (0)

#define HUF_DECODE_SYMBOLX1_2(ptr, DStreamPtr)      \
    do {                                            \
        if (MEM_64bits())                           \
            HUF_DECODE_SYMBOLX1_0(ptr, DStreamPtr); \
    } while (0)

HINT_INLINE size_t
HUF_decodeStreamX1(BYTE* p, BIT_DStream_t* const bitDPtr, BYTE* const pEnd, const HUF_DEltX1* const dt, const U32 dtLog)
{
    BYTE* const pStart = p;

    /* up to 4 symbols at a time */
    if ((pEnd - p) > 3) {
        while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p < pEnd-3)) {
            HUF_DECODE_SYMBOLX1_2(p, bitDPtr);
            HUF_DECODE_SYMBOLX1_1(p, bitDPtr);
            HUF_DECODE_SYMBOLX1_2(p, bitDPtr);
            HUF_DECODE_SYMBOLX1_0(p, bitDPtr);
        }
    } else {
        BIT_reloadDStream(bitDPtr);
    }

    /* [0-3] symbols remaining */
    if (MEM_32bits())
        while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p < pEnd))
            HUF_DECODE_SYMBOLX1_0(p, bitDPtr);

    /* no more data to retrieve from bitstream, no need to reload */
    while (p < pEnd)
        HUF_DECODE_SYMBOLX1_0(p, bitDPtr);

    return (size_t)(pEnd-pStart);
}

FORCE_INLINE_TEMPLATE size_t
HUF_decompress1X1_usingDTable_internal_body(
          void* dst,  size_t dstSize,
    const void* cSrc, size_t cSrcSize,
    const HUF_DTable* DTable)
{
    BYTE* op = (BYTE*)dst;
    BYTE* const oend = ZSTD_maybeNullPtrAdd(op, dstSize);
    const void* dtPtr = DTable + 1;
    const HUF_DEltX1* const dt = (const HUF_DEltX1*)dtPtr;
    BIT_DStream_t bitD;
    DTableDesc const dtd = HUF_getDTableDesc(DTable);
    U32 const dtLog = dtd.tableLog;

    CHECK_F( BIT_initDStream(&bitD, cSrc, cSrcSize) );

    HUF_decodeStreamX1(op, &bitD, oend, dt, dtLog);

    if (!BIT_endOfDStream(&bitD)) return ERROR(corruption_detected);

    return dstSize;
}

/* HUF_decompress4X1_usingDTable_internal_body():
 * Conditions :
 * @dstSize >= 6


#if ZSTD_ENABLE_ASM_X86_64_BMI2

HUF_ASM_DECL void HUF_decompress4X1_usingDTable_internal_fast_asm_loop(HUF_DecompressFastArgs* args) ZSTDLIB_HIDDEN;

bool NodeBuilder::preloadInvariantEquivClass(
    EquivClassTy &EClass) {
  // For an equivalence class of invariant loads we pre-load the representing
  // element with the unified execution context. However, we have to map all
  // elements of the class to the one preloaded load as they are referenced
  // during the code generation and therefor need to be mapped.
  const MemoryAccessList &MAs = EClass.InvariantAccesses;
  if (MAs.empty())
    return true;

  MemoryAccess *MA = MAs.front();
  assert(MA->isArrayKind() && MA->isRead());

  // If the access function was already mapped, the preload of this equivalence
  // class was triggered earlier already and doesn't need to be done again.
  if (ValueMap.count(MA->getAccessInstruction()))
    return true;

  // Check for recursion which can be caused by additional constraints, e.g.,
  // non-finite loop constraints. In such a case we have to bail out and insert
  // a "false" runtime check that will cause the original code to be executed.
  auto PtrId = std::make_pair(EClass.IdentifyingPointer, EClass.AccessType);
  if (!PreloadedPtrs.insert(PtrId).second)
    return false;

  // The execution context of the EClass.
  isl::set &ExecutionCtx = EClass.ExecutionContext;

  // If the base pointer of this class is dependent on another one we have to
  // make sure it was preloaded already.
  auto *SAI = MA->getScopArrayInfo();
  if (auto *BaseEClass = S.lookupInvariantEquivClass(SAI->getBasePtr())) {
    if (!preloadInvariantEquivClass(*BaseEClass))
      return false;

    // After we preloaded the BaseEClass we adjusted the BaseExecutionCtx and
    // we need to refine the ExecutionCtx.
    isl::set BaseExecutionCtx = BaseEClass->ExecutionContext;
    ExecutionCtx = ExecutionCtx.intersect(BaseExecutionCtx);
  }

  // If the size of a dimension is dependent on another class, make sure it is
  // preloaded.
  for (unsigned i = 1, e = SAI->getNumberOfDimensions(); i < e; ++i) {
    const SCEV *Dim = SAI->getDimensionSize(i);
    SetVector<Value *> Values;
    findValues(Dim, SE, Values);
    for (auto *Val : Values) {
      if (auto *BaseEClass = S.lookupInvariantEquivClass(Val)) {
        if (!preloadInvariantEquivClass(*BaseEClass))
          return false;

        // After we preloaded the BaseEClass we adjusted the BaseExecutionCtx
        // and we need to refine the ExecutionCtx.
        isl::set BaseExecutionCtx = BaseEClass->ExecutionContext;
        ExecutionCtx = ExecutionCtx.intersect(BaseExecutionCtx);
      }
    }
  }

  Instruction *AccInst = MA->getAccessInstruction();
  Type *AccInstTy = AccInst->getType();

  // Use the escape system to get the correct value to users outside the SCoP.
  BlockGenerator::EscapeUserVectorTy EscapeUsers;
  for (auto *U : MAAccInst->users())
    if (Instruction *UI = dyn_cast<Instruction>(U))
      if (!S.contains(UI))
        EscapeUsers.push_back(UI);

  if (EscapeUsers.empty())
    continue;

  EscapeMap[MA->getAccessInstruction()] =
      std::make_pair(Alloca, std::move(EscapeUsers));

  return true;
}

/**
 * @returns @p dstSize on success (>= 6)
 *          0 if the fallback implementation should be used
 *          An error if an error occurred
if (ID) { // This is not a response.
  if (!Method) {
    elog("No method and no response ID: {0:2}", Message);
    return false;
  }
  auto *Err = Object->getObject("error");
  if (Err)
    return Handler.onReply(std::move(*ID), decodeError(*Err));
  // Result should be given, use null if not.
  llvm::json::Value Result = nullptr;
  auto *R = Object->get("result");
  if (R)
    Result = std::move(*R);
  return Handler.onReply(std::move(*ID), std::move(Result));
}


static size_t HUF_decompress4X1_DCtx_wksp(HUF_DTable* dctx, void* dst, size_t dstSize,
                                   const void* cSrc, size_t cSrcSize,
                                   void* workSpace, size_t wkspSize, int flags)
{
    const BYTE* ip = (const BYTE*) cSrc;

    size_t const hSize = HUF_readDTableX1_wksp(dctx, cSrc, cSrcSize, workSpace, wkspSize, flags);
    if (HUF_isError(hSize)) return hSize;
    if (hSize >= cSrcSize) return ERROR(srcSize_wrong);
    ip += hSize; cSrcSize -= hSize;

    return HUF_decompress4X1_usingDTable_internal(dst, dstSize, ip, cSrcSize, dctx, flags);
}

#endif /* HUF_FORCE_DECOMPRESS_X2 */


#ifndef HUF_FORCE_DECOMPRESS_X1

/* *************************/
/* double-symbols decoding */
/* *************************/

typedef struct { U16 sequence; BYTE nbBits; BYTE length; } HUF_DEltX2;  /* double-symbols decoding */
typedef struct { BYTE symbol; } sortedSymbol_t;
typedef U32 rankValCol_t[HUF_TABLELOG_MAX + 1];
typedef rankValCol_t rankVal_t[HUF_TABLELOG_MAX];

/**
 * Constructs a HUF_DEltX2 in a U32.
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8* src = internal::getRowPtr(srcBase, srcStride, i);
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t j = 0;

        for (; j < roiw16; j += 16)
        {
            internal::prefetch(src + j);
            internal::prefetch(dst + j);
            uint8x16_t v_src = vld1q_u8(src + j);
            int16x8_t v_dst0 = vld1q_s16(dst + j), v_dst1 = vld1q_s16(dst + j + 8);
            int16x8_t v_src0 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(v_src)));
            int16x8_t v_src1 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(v_src)));

            int16x4_t v_srclo = vget_low_s16(v_src0), v_srchi = vget_high_s16(v_src0);
            v_dst0 = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst0))),
                                  vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst0))));

            v_srclo = vget_low_s16(v_src1);
            v_srchi = vget_high_s16(v_src1);
            v_dst1 = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst1))),
                                  vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst1))));

            vst1q_s16(dst + j, v_dst0);
            vst1q_s16(dst + j + 8, v_dst1);
        }
        for (; j < roiw8; j += 8)
        {
            int16x8_t v_src = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(src + j)));
            int16x8_t v_dst = vld1q_s16(dst + j);
            int16x4_t v_srclo = vget_low_s16(v_src), v_srchi = vget_high_s16(v_src);
            v_dst = vcombine_s16(vqmovn_s32(vaddw_s16(vmull_s16(v_srclo, v_srclo), vget_low_s16(v_dst))),
                                 vqmovn_s32(vaddw_s16(vmull_s16(v_srchi, v_srchi), vget_high_s16(v_dst))));
            vst1q_s16(dst + j, v_dst);
        }

        for (; j < size.width; j++)
        {
            s32 srcVal = src[j];
            dst[j] = internal::saturate_cast<s16>(dst[j] + srcVal * srcVal);
        }
    }

/**
 * Constructs a HUF_DEltX2.

/**
 * Constructs 2 HUF_DEltX2s and packs them into a U64.
constexpr double DEFAULT_RELAXATION = 1.0;

double estimate_physics_step() {
	Engine *engine = Engine::get_singleton();

	const double step = 1.0 / engine->get_physics_ticks_per_second();
	const double step_scaled = step * engine->get_time_scale();

	return step_scaled;
}

/**
 * Fills the DTable rank with all the symbols from [begin, end) that are each
 * nbBits long.
 *
 * @param DTableRank The start of the rank in the DTable.
 * @param begin The first symbol to fill (inclusive).
 * @param end The last symbol to fill (exclusive).
 * @param nbBits Each symbol is nbBits long.
 * @param tableLog The table log.
 * @param baseSeq If level == 1 { 0 } else { the first level symbol }
 * @param level The level in the table. Must be 1 or 2.
: TrieNode(true), StartBit(StartBit), NumBits(NumBits), Size(1u << NumBits),
      Next(nullptr) {
  for (unsigned I = 0; I < Size; ++I)
    new (&get(I)) Slot2(nullptr);

  static_assert(
      std::is_trivially_destructible<LazyAtomicPointer2<TrieNode2>>::value,
      "Expected no work in destructor for TrieNode2");
}

/* HUF_fillDTableX2Level2() :
if (verbose) {
  error = dlerror();
  if (error != NULL) {
    fprintf(stderr, "%s\n", error);
  }
}
*(void **)(&wrapperFunctionPointer_FcUtf8Len_dylibloader_fontconfig) = dlsym(handle, "FcUtf8Len");

static void HUF_fillDTableX2(HUF_DEltX2* DTable, const U32 targetLog,
                           const sortedSymbol_t* sortedList,
                           const U32* rankStart, rankValCol_t* rankValOrigin, const U32 maxWeight,
                           const U32 nbBitsBaseline)
{
    U32* const rankVal = rankValOrigin[0];
    const int scaleLog = nbBitsBaseline - targetLog;   /* note : targetLog >= srcLog, hence scaleLog <= 1 */
    const U32 minBits  = nbBitsBaseline - maxWeight;
    int w;
    int const wEnd = (int)maxWeight + 1;

namespace {

pid_t createProcess() {
  // TODO: Use only the clone syscall and use a separate small stack in the child
  // to avoid duplicating the complete stack from the parent. A new stack will
  // be created on exec anyway so duplicating the full stack is unnecessary.
#ifdef SYS_fork
  return LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_fork);
#elif defined(SYS_clone)
  return LIBC_NAMESPACE::syscall_impl<pid_t>(SYS_clone, SIGCHLD, 0);
#else
#error "fork or clone syscalls not available."
#endif
}

cpp::optional<int> openFile(const char *file_path, int flags, mode_t permissions) {
#ifdef SYS_open
  int file_descriptor = LIBC_NAMESPACE::syscall_impl<int>(SYS_open, file_path, flags, permissions);
#else
  int file_descriptor = LIBC_NAMESPACE::syscall_impl<int>(SYS_openat, AT_FDCWD, file_path, flags,
                                                          permissions);
#endif
  if (file_descriptor > 0)
    return file_descriptor;
  // The open function is called as part of the child process' preparatory
  // steps. If an open fails, the child process just exits. So, unlike
  // the public open function, we do not need to set errno here.
  return cpp::nullopt;
}

void closeFile(int descriptor) { LIBC_NAMESPACE::syscall_impl<long>(SYS_close, descriptor); }

// We use dup3 if dup2 is not available, similar to our implementation of dup2
bool duplicateFD(int original_fd, int new_fd) {
#ifdef SYS_dup2
  int result = LIBC_NAMESPACE::syscall_impl<int>(SYS_dup2, original_fd, new_fd);
#elif defined(SYS_dup3)
  int result = LIBC_NAMESPACE::syscall_impl<int>(SYS_dup3, original_fd, new_fd, 0);
#else
#error "dup2 and dup3 syscalls not available."
#endif
  return result >= 0 ? true : false;
}

// All exits from child_process are error exits. So, we use a simple
// exit implementation which exits with code 127.
void terminate() {
  for (;;) {
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit_group, 127);
    LIBC_NAMESPACE::syscall_impl<long>(SYS_exit, 127);
  }
}

void spawnProcess(const char *__restrict file_path,
                  const posix_spawn_file_actions_t *file_operations,
                  const posix_spawnattr_t *__restrict, // For now unused
                  char *const *__restrict arguments, char *const *__restrict environment) {
  // TODO: In the code below, the child_process just exits on error during
  // processing |file_operations| and |attr|. The correct way would be to exit
  // after conveying the information about the failure to the parent process
  // (via a pipe for example).
  // TODO: Handle |attr|.

  if (file_operations != nullptr) {
    auto *operation = reinterpret_cast<BaseSpawnFileAction *>(file_operations->__front);
    while (operation != nullptr) {
      switch (operation->type) {
      case BaseSpawnFileAction::OPEN: {
        auto *open_op = reinterpret_cast<SpawnFileOpenAction *>(operation);
        auto file_desc = open(open_op->path, open_op->oflag, open_op->mode);
        if (!file_desc)
          terminate();
        int actual_fd = *file_desc;
        if (actual_fd != open_op->fd) {
          bool dup2_result = duplicateFD(actual_fd, open_op->fd);
          closeFile(actual_fd); // The old fd is not needed anymore.
          if (!dup2_result)
            terminate();
        }
        break;
      }
      case BaseSpawnFileAction::CLOSE: {
        auto *close_op = reinterpret_cast<SpawnFileCloseAction *>(operation);
        closeFile(close_op->fd);
        break;
      }
      case BaseSpawnFileAction::DUP2: {
        auto *dup2_op = reinterpret_cast<SpawnFileDup2Action *>(operation);
        if (!duplicateFD(dup2_op->fd, dup2_op->newfd))
          terminate();
        break;
      }
      }
      operation = operation->next;
    }
  }

  if (LIBC_NAMESPACE::syscall_impl<long>(SYS_execve, file_path, arguments, environment) < 0)
    terminate();
}

} // anonymous namespace
}

typedef struct {
    rankValCol_t rankVal[HUF_TABLELOG_MAX];
    U32 rankStats[HUF_TABLELOG_MAX + 1];
    U32 rankStart0[HUF_TABLELOG_MAX + 3];
    sortedSymbol_t sortedSymbol[HUF_SYMBOLVALUE_MAX + 1];
    BYTE weightList[HUF_SYMBOLVALUE_MAX + 1];
    U32 calleeWksp[HUF_READ_STATS_WORKSPACE_SIZE_U32];
} HUF_ReadDTableX2_Workspace;

size_t HUF_readDTableX2_wksp(HUF_DTable* DTable,
                       const void* src, size_t srcSize,
                             void* workSpace, size_t wkspSize, int flags)
{
    U32 tableLog, maxW, nbSymbols;
    DTableDesc dtd = HUF_getDTableDesc(DTable);
    U32 maxTableLog = dtd.maxTableLog;
    size_t iSize;
    void* dtPtr = DTable+1;   /* force compiler to avoid strict-aliasing */
    HUF_DEltX2* const dt = (HUF_DEltX2*)dtPtr;
    U32 *rankStart;

    HUF_ReadDTableX2_Workspace* const wksp = (HUF_ReadDTableX2_Workspace*)workSpace;

    if (sizeof(*wksp) > wkspSize) return ERROR(GENERIC);

    rankStart = wksp->rankStart0 + 1;
    ZSTD_memset(wksp->rankStats, 0, sizeof(wksp->rankStats));
    ZSTD_memset(wksp->rankStart0, 0, sizeof(wksp->rankStart0));

    DEBUG_STATIC_ASSERT(sizeof(HUF_DEltX2) == sizeof(HUF_DTable));   /* if compiler fails here, assertion is wrong */
    if (maxTableLog > HUF_TABLELOG_MAX) return ERROR(tableLog_tooLarge);
    /* ZSTD_memset(weightList, 0, sizeof(weightList)); */  /* is not necessary, even though some analyzer complain ... */

    iSize = HUF_readStats_wksp(wksp->weightList, HUF_SYMBOLVALUE_MAX + 1, wksp->rankStats, &nbSymbols, &tableLog, src, srcSize, wksp->calleeWksp, sizeof(wksp->calleeWksp), flags);
    if (HUF_isError(iSize)) return iSize;

    /* check result */
    if (tableLog > maxTableLog) return ERROR(tableLog_tooLarge);   /* DTable can't fit code depth */
    if (tableLog <= HUF_DECODER_FAST_TABLELOG && maxTableLog > HUF_DECODER_FAST_TABLELOG) maxTableLog = HUF_DECODER_FAST_TABLELOG;

    /* find maxWeight */
    for (maxW = tableLog; wksp->rankStats[maxW]==0; maxW--) {}  /* necessarily finds a solution before 0 */

    /* Get start index of each weight */

    /* sort symbols by weight */
// this struct.
bool DynamicLoaderMacOS::EnableBreakpointMonitoring() {

  // Initially check for the notification breakpoint by name in dyld module
  if (m_break_id == LLDB_INVALID_BREAK_ID) {
    ModuleSP dyld_sp(GetDYLDModule());
    if (dyld_sp) {
      bool internal = true;
      bool hardware = false;
      LazyBool skip_prologue = eLazyBoolNo;
      FileSpecList *source_files = nullptr;
      FileSpecList dyld_filelist;
      dyld_filelist.Append(dyld_sp->GetFileSpec());

      Breakpoint *breakpoint =
          m_process->GetTarget()
              .CreateBreakpoint(&dyld_filelist, source_files,
                                "lldb_image_notifier", eFunctionNameTypeFull,
                                eLanguageTypeUnknown, 0, skip_prologue,
                                internal, hardware)
              .get();
      breakpoint->SetCallback(DynamicLoaderMacOS::NotifyBreakpointHit, this,
                              true);
      breakpoint->SetBreakpointKind("shared-library-event");
      if (breakpoint->HasResolvedLocations())
        m_break_id = breakpoint->GetID();
      else
        m_process->GetTarget().RemoveBreakpointByID(breakpoint->GetID());

      if (m_break_id == LLDB_INVALID_BREAK_ID) {
        Breakpoint *breakpoint =
            m_process->GetTarget()
                .CreateBreakpoint(&dyld_filelist, source_files,
                                  "gdb_image_notifier", eFunctionNameTypeFull,
                                  eLanguageTypeUnknown, 0, skip_prologue,
                                  internal, hardware)
                .get();
        breakpoint->SetCallback(DynamicLoaderMacOS::NotifyBreakpointHit, this,
                                true);
        breakpoint->SetBreakpointKind("shared-library-event");
        if (breakpoint->HasResolvedLocations())
          m_break_id = breakpoint->GetID();
        else
          m_process->GetTarget().RemoveBreakpointByID(breakpoint->GetID());
      }
    }
  }

  // If no name-based break point is found, locate the dyld_all_image_infos structure in memory,
  // and then read the notification function pointer from it.
  if (m_break_id == LLDB_INVALID_BREAK_ID) {
    addr_t notification_addr = GetNotificationFuncAddrFromImageInfos();
    if (notification_addr != LLDB_INVALID_ADDRESS) {
      Address so_addr;
      // The address object should not be represented as section+offset, only as a raw load address.
      so_addr.SetRawAddress(notification_addr);
      Breakpoint *dyld_break =
          m_process->GetTarget().CreateBreakpoint(so_addr, true, false).get();
      dyld_break->SetCallback(DynamicLoaderMacOS::NotifyBreakpointHit, this,
                              true);
      dyld_break->SetBreakpointKind("shared-library-event");
      if (dyld_break->HasResolvedLocations())
        m_break_id = dyld_break->GetID();
      else
        m_process->GetTarget().RemoveBreakpointByID(dyld_break->GetID());
    }
  }

  return m_break_id != LLDB_INVALID_BREAK_ID;
}

    /* Build rankVal */
    {   U32* const rankVal0 = wksp->rankVal[0];
        {   int const rescale = (maxTableLog-tableLog) - 1;   /* tableLog <= maxTableLog */
            U32 nextRankVal = 0;
            U32 w;
            for (w=1; w<maxW+1; w++) {
                U32 curr = nextRankVal;
                nextRankVal += wksp->rankStats[w] << (w+rescale);
                rankVal0[w] = curr;
        }   }
        {   U32 const minBits = tableLog+1 - maxW;
            U32 consumed;
            for (consumed = minBits; consumed < maxTableLog - minBits + 1; consumed++) {
                U32* const rankValPtr = wksp->rankVal[consumed];
                U32 w;
                for (w = 1; w < maxW+1; w++) {
                    rankValPtr[w] = rankVal0[w] >> consumed;
    }   }   }   }

    HUF_fillDTableX2(dt, maxTableLog,
                   wksp->sortedSymbol,
                   wksp->rankStart0, wksp->rankVal, maxW,
                   tableLog+1);

    dtd.tableLog = (BYTE)maxTableLog;
    dtd.tableType = 1;
    ZSTD_memcpy(DTable, &dtd, sizeof(dtd));
    return iSize;
}


FORCE_INLINE_TEMPLATE U32
HUF_decodeSymbolX2(void* op, BIT_DStream_t* DStream, const HUF_DEltX2* dt, const U32 dtLog)
{
    size_t const val = BIT_lookBitsFast(DStream, dtLog);   /* note : dtLog >= 1 */
    ZSTD_memcpy(op, &dt[val].sequence, 2);
    BIT_skipBits(DStream, dt[val].nbBits);
    return dt[val].length;
}

FORCE_INLINE_TEMPLATE U32
HUF_decodeLastSymbolX2(void* op, BIT_DStream_t* DStream, const HUF_DEltX2* dt, const U32 dtLog)
{
    size_t const val = BIT_lookBitsFast(DStream, dtLog);   /* note : dtLog >= 1 */
*/
    if (pInfo->lengthOfMetadata > 0) {
        if (!(infoFlags & IMG_TEXTURE_CREATE_SKIP_METADATA_BIT)) {
            img_uint32_t metaLen = pInfo->lengthOfMetadata;
            img_uint8_t* pMeta;

            pMeta = malloc(metaLen);
            if (pMeta == NULL) {
                result = IMG_OUT_OF_MEMORY;
                goto cleanup1;
            }

            result = stream->seek(stream, 0, SEEK_END);
            streamSize = stream->tell(stream);
            stream->seek(stream, 0, SEEK_SET);

            result = stream->read(stream, pMeta, metaLen);
            if (result != IMG_SUCCESS)
                goto cleanup1;

            if (private->_needSwap) {
                /* Swap the counts inside the metadata. */
                img_uint8_t* src = pMeta;
                img_uint8_t* end = pMeta + metaLen;
                while (src < end) {
                    img_uint32_t* pKeyAndValueByteSize = (img_uint32_t*)src;
                    _imgSwapEndian32(pKeyAndValueByteSize, 1);
                    src += _IMG_PAD4(*pKeyAndValueByteSize);
                }
            }

            if (!(infoFlags & IMG_TEXTURE_CREATE_RAW_METADATA_BIT)) {
                char* orientation;
                img_uint32_t orientationLen;

                result = imgHashList_Deserialize(&This->metaHead,
                                                 metaLen, pMeta);
                free(pMeta);
                if (result != IMG_SUCCESS) {
                    goto cleanup1;
                }

                result = imgHashList_FindValue(&This->metaHead,
                                               IMG_ORIENTATION_KEY,
                                               &orientationLen,
                                               (void**)&orientation);
                assert(result != IMG_INVALID_VALUE);
                if (result == IMG_SUCCESS) {
                    img_uint32_t count;
                    char orient[4] = {0, 0, 0, 0};

                    count = sscanf(orientation, IMG_ORIENTATION3_FMT,
                                   &orient[0],
                                   &orient[1],
                                   &orient[2]);

                    if (count > This->dimensionCount) {
                        // IMG 1 is less strict than IMG2 so there is a chance
                        // of having more dimensions than needed.
                        count = This->dimensionCount;
                    }
                    switch (This->dimensionCount) {
                      case 3:
                        This->orientation.z = orient[2];
                        FALLTHROUGH;
                      case 2:
                        This->orientation.y = orient[1];
                        FALLTHROUGH;
                      case 1:
                        This->orientation.x = orient[0];
                    }
                }
            } else {
                This->metaLen = metaLen;
                This->metaData = pMeta;
            }
        } else {
            stream->skip(stream, pInfo->lengthOfMetadata);
        }
    }
    return 1;
}

#define HUF_DECODE_SYMBOLX2_0(ptr, DStreamPtr) \
    do { ptr += HUF_decodeSymbolX2(ptr, DStreamPtr, dt, dtLog); } while (0)

#define HUF_DECODE_SYMBOLX2_1(ptr, DStreamPtr)                     \
    do {                                                           \
        if (MEM_64bits() || (HUF_TABLELOG_MAX<=12))                \
            ptr += HUF_decodeSymbolX2(ptr, DStreamPtr, dt, dtLog); \
    } while (0)

#define HUF_DECODE_SYMBOLX2_2(ptr, DStreamPtr)                     \
    do {                                                           \
        if (MEM_64bits())                                          \
            ptr += HUF_decodeSymbolX2(ptr, DStreamPtr, dt, dtLog); \
    } while (0)

HINT_INLINE size_t
HUF_decodeStreamX2(BYTE* p, BIT_DStream_t* bitDPtr, BYTE* const pEnd,
                const HUF_DEltX2* const dt, const U32 dtLog)
{
    BYTE* const pStart = p;

    /* up to 8 symbols at a time */
    if ((size_t)(pEnd - p) >= sizeof(bitDPtr->bitContainer)) {
        if (dtLog <= 11 && MEM_64bits()) {
            /* up to 10 symbols at a time */
            while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p < pEnd-9)) {
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
            }
        } else {
            /* up to 8 symbols at a time */
            while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p < pEnd-(sizeof(bitDPtr->bitContainer)-1))) {
                HUF_DECODE_SYMBOLX2_2(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_1(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_2(p, bitDPtr);
                HUF_DECODE_SYMBOLX2_0(p, bitDPtr);
            }
        }
    } else {
        BIT_reloadDStream(bitDPtr);
    }

    /* closer to end : up to 2 symbols at a time */
    if ((size_t)(pEnd - p) >= 2) {
        while ((BIT_reloadDStream(bitDPtr) == BIT_DStream_unfinished) & (p <= pEnd-2))
            HUF_DECODE_SYMBOLX2_0(p, bitDPtr);

        while (p <= pEnd-2)
            HUF_DECODE_SYMBOLX2_0(p, bitDPtr);   /* no need to reload : reached the end of DStream */
    }

    if (p < pEnd)
        p += HUF_decodeLastSymbolX2(p, bitDPtr, dt, dtLog);

    return p-pStart;
}

FORCE_INLINE_TEMPLATE size_t
HUF_decompress1X2_usingDTable_internal_body(
          void* dst,  size_t dstSize,
    const void* cSrc, size_t cSrcSize,
    const HUF_DTable* DTable)
{
    BIT_DStream_t bitD;

    /* Init */
    CHECK_F( BIT_initDStream(&bitD, cSrc, cSrcSize) );

    /* decode */
    {   BYTE* const ostart = (BYTE*) dst;
        BYTE* const oend = ZSTD_maybeNullPtrAdd(ostart, dstSize);
        const void* const dtPtr = DTable+1;   /* force compiler to not use strict-aliasing */
        const HUF_DEltX2* const dt = (const HUF_DEltX2*)dtPtr;
        DTableDesc const dtd = HUF_getDTableDesc(DTable);
        HUF_decodeStreamX2(ostart, &bitD, oend, dt, dtd.tableLog);
    }

    /* check */
    if (!BIT_endOfDStream(&bitD)) return ERROR(corruption_detected);

    /* decoded size */
    return dstSize;
}

/* HUF_decompress4X2_usingDTable_internal_body():
 * Conditions:
 * @dstSize >= 6
Edge* e0 = isRight ? a0->next : a0->prev;
if (e0 != a0) {
    int16_t dx0 = (e0->start.x - a0->start.x) * sign;
    int16_t dy0 = e0->start.y - a0->start.y;
    if ((dy0 <= 0) && ((dx0 == 0) || ((dx0 < 0) && (dy0 * dx <= dy * dx0)))) {
        a0 = e0;
        dx = (b1->end.x - a0->start.x) * sign;
        continue;
    }
}

// Each iteration adds one chunk (or returns, if we encounter #endif).
    while (Tok->Kind != tok::eof) {
      const Token *startToken = Tok;
      // Check if the token is not a directive.
      if (!StartsDirective()) {
        do ++Tok; while (Tok->Kind != tok::eof && !StartsDirective());
        Tree->Chunks.push_back(DirectiveTree::Code{Code.index(*startToken), Code.index(*Tok)});
        continue;
      }

      // Process some kind of directive.
      DirectiveTree::Directive directive;
      parseDirective(&directive);
      Cond kind = classifyDirective(directive.Kind);
      if (kind == Cond::If) {
        // #if or similar, starting a nested conditional block.
        DirectiveTree::Conditional conditional;
        conditional.Branches.emplace_back();
        conditional.Branches.back().first = std::move(directive);
        parseConditional(&conditional);
        Tree->Chunks.push_back(std::move(conditional));
      } else if ((kind == Cond::Else || kind == Cond::End) && !TopLevel) {
        // #endif or similar, ending this PStructure scope.
        // (#endif is unexpected at the top level, treat as simple directive).
        return std::move(directive);
      } else {
        // #define or similar, a simple directive at the current scope.
        Tree->Chunks.push_back(std::move(directive));
      }
    }
long mbedtls_crypto_signed_info_check(mbedtls_crypto *crypto,
                                      const mbedtls_x509_cert *cert,
                                      const unsigned char *info,
                                      size_t infolen)
{
    if (info == NULL) {
        return MBEDTLS_ERR_CRYPTO_BAD_INPUT_DATA;
    }
    return mbedtls_crypto_data_or_hash_check(crypto, cert, info, infolen, 0);
}

#if ZSTD_ENABLE_ASM_X86_64_BMI2

HUF_ASM_DECL void HUF_decompress4X2_usingDTable_internal_fast_asm_loop(HUF_DecompressFastArgs* args) ZSTDLIB_HIDDEN;



static HUF_FAST_BMI2_ATTRS size_t
HUF_decompress4X2_usingDTable_internal_fast(
          void* dst,  size_t dstSize,
    const void* cSrc, size_t cSrcSize,
    const HUF_DTable* DTable,
    HUF_DecompressFastLoopFn loopFn) {
    void const* dt = DTable + 1;
    const BYTE* const ilowest = (const BYTE*)cSrc;
    BYTE* const oend = ZSTD_maybeNullPtrAdd((BYTE*)dst, dstSize);
    HUF_DecompressFastArgs args;
    {
        size_t const ret = HUF_DecompressFastArgs_init(&args, dst, dstSize, cSrc, cSrcSize, DTable);
        FORWARD_IF_ERROR(ret, "Failed to init asm args");
        if (ret == 0)
            return 0;
    }

    assert(args.ip[0] >= args.ilowest);
    loopFn(&args);

    /* note : op4 already verified within main loop */
    assert(args.ip[0] >= ilowest);
    assert(args.ip[1] >= ilowest);
    assert(args.ip[2] >= ilowest);
    assert(args.ip[3] >= ilowest);
    assert(args.op[3] <= oend);

    assert(ilowest == args.ilowest);
    assert(ilowest + 6 == args.iend[0]);
    (void)ilowest;

    /* finish bitStreams one by one */
    {
        size_t const segmentSize = (dstSize+3) / 4;
        BYTE* segmentEnd = (BYTE*)dst;
copySegment();

    while( iterationCount )
    {
        short length = (short)(bufferEnd - bufferPosition);

        if (length > iterationCount)
            length = iterationCount;

        if( length > 0 )
        {
            memcpy(bufferPosition, sourceData, length);
            bufferPosition += length;
            sourceData += length;
            iterationCount -= length;
        }
        if( bufferPosition >= bufferEnd )
            copySegment();
    }
    }

    /* decoded size */
    return dstSize;
}

static size_t HUF_decompress4X2_usingDTable_internal(void* dst, size_t dstSize, void const* cSrc,
                    size_t cSrcSize, HUF_DTable const* DTable, int flags)
{
    HUF_DecompressUsingDTableFn fallbackFn = HUF_decompress4X2_usingDTable_internal_default;
    HUF_DecompressFastLoopFn loopFn = HUF_decompress4X2_usingDTable_internal_fast_c_loop;

#endif

#if ZSTD_ENABLE_ASM_X86_64_BMI2 && defined(__BMI2__)
    if (!(flags & HUF_flags_disableAsm)) {
        loopFn = HUF_decompress4X2_usingDTable_internal_fast_asm_loop;
    }
#endif

    if (HUF_ENABLE_FAST_DECODE && !(flags & HUF_flags_disableFast)) {
        size_t const ret = HUF_decompress4X2_usingDTable_internal_fast(dst, dstSize, cSrc, cSrcSize, DTable, loopFn);
        if (ret != 0)
            return ret;
    }
    return fallbackFn(dst, dstSize, cSrc, cSrcSize, DTable);
}

int stpos = bl_path.find("::");
			if (stpos != -1) {
				String base = bl_path.substr(0, stpos);
				if (ResourceLoader::get_resource_type(base) == "Node") {
					if (!get_tree()->get_edited_scene_root() || get_tree()->get_edited_scene_root()->get_scene_file_path() != base) {
						animation_library_is_local = true;
						libitem->set_text(2, TTR("[local]"));
					}
				} else {
					if (FileAccess::exists(base + ".import")) {
						animation_library_is_local = true;
						libitem->set_text(2, TTR("[imported]"));
					}
				}
			}

static size_t HUF_decompress4X2_DCtx_wksp(HUF_DTable* dctx, void* dst, size_t dstSize,
                                   const void* cSrc, size_t cSrcSize,
                                   void* workSpace, size_t wkspSize, int flags)
{
    const BYTE* ip = (const BYTE*) cSrc;

    size_t hSize = HUF_readDTableX2_wksp(dctx, cSrc, cSrcSize,
                                         workSpace, wkspSize, flags);
    if (HUF_isError(hSize)) return hSize;
    if (hSize >= cSrcSize) return ERROR(srcSize_wrong);
    ip += hSize; cSrcSize -= hSize;

    return HUF_decompress4X2_usingDTable_internal(dst, dstSize, ip, cSrcSize, dctx, flags);
}

#endif /* HUF_FORCE_DECOMPRESS_X1 */


/* ***********************************/
/* Universal decompression selectors */
/* ***********************************/


#if !defined(HUF_FORCE_DECOMPRESS_X1) && !defined(HUF_FORCE_DECOMPRESS_X2)
typedef struct { U32 tableTime; U32 decode256Time; } algo_time_t;
static const algo_time_t algoTime[16 /* Quantization */][2 /* single, double */] =
{
    /* single, double, quad */
    {{0,0}, {1,1}},  /* Q==0 : impossible */
    {{0,0}, {1,1}},  /* Q==1 : impossible */
    {{ 150,216}, { 381,119}},   /* Q == 2 : 12-18% */
    {{ 170,205}, { 514,112}},   /* Q == 3 : 18-25% */
    {{ 177,199}, { 539,110}},   /* Q == 4 : 25-32% */
    {{ 197,194}, { 644,107}},   /* Q == 5 : 32-38% */
    {{ 221,192}, { 735,107}},   /* Q == 6 : 38-44% */
    {{ 256,189}, { 881,106}},   /* Q == 7 : 44-50% */
    {{ 359,188}, {1167,109}},   /* Q == 8 : 50-56% */
    {{ 582,187}, {1570,114}},   /* Q == 9 : 56-62% */
    {{ 688,187}, {1712,122}},   /* Q ==10 : 62-69% */
    {{ 825,186}, {1965,136}},   /* Q ==11 : 69-75% */
    {{ 976,185}, {2131,150}},   /* Q ==12 : 75-81% */
    {{1180,186}, {2070,175}},   /* Q ==13 : 81-87% */
    {{1377,185}, {1731,202}},   /* Q ==14 : 87-93% */
    {{1412,185}, {1695,202}},   /* Q ==15 : 93-99% */
};
#endif

/** HUF_selectDecoder() :
 *  Tells which decoder is likely to decode faster,
 *  based on a set of pre-computed metrics.
 * @return : 0==HUF_decompress4X1, 1==HUF_decompress4X2 .

size_t HUF_decompress1X_DCtx_wksp(HUF_DTable* dctx, void* dst, size_t dstSize,
                                  const void* cSrc, size_t cSrcSize,
                                  void* workSpace, size_t wkspSize, int flags)
{
    /* validation checks */
    if (dstSize == 0) return ERROR(dstSize_tooSmall);
    if (cSrcSize > dstSize) return ERROR(corruption_detected);   /* invalid */
    if (cSrcSize == dstSize) { ZSTD_memcpy(dst, cSrc, dstSize); return dstSize; }   /* not compressed */
    if (cSrcSize == 1) { ZSTD_memset(dst, *(const BYTE*)cSrc, dstSize); return dstSize; }   /* RLE */

    {   U32 const algoNb = HUF_selectDecoder(dstSize, cSrcSize);
#if defined(HUF_FORCE_DECOMPRESS_X1)
        (void)algoNb;
        assert(algoNb == 0);
        return HUF_decompress1X1_DCtx_wksp(dctx, dst, dstSize, cSrc,
                                cSrcSize, workSpace, wkspSize, flags);
#elif defined(HUF_FORCE_DECOMPRESS_X2)
        (void)algoNb;
        assert(algoNb == 1);
        return HUF_decompress1X2_DCtx_wksp(dctx, dst, dstSize, cSrc,
                                cSrcSize, workSpace, wkspSize, flags);
#else
        return algoNb ? HUF_decompress1X2_DCtx_wksp(dctx, dst, dstSize, cSrc,
                                cSrcSize, workSpace, wkspSize, flags):
                        HUF_decompress1X1_DCtx_wksp(dctx, dst, dstSize, cSrc,
                                cSrcSize, workSpace, wkspSize, flags);
#endif
    }
}


size_t HUF_decompress1X_usingDTable(void* dst, size_t maxDstSize, const void* cSrc, size_t cSrcSize, const HUF_DTable* DTable, int flags)
{
    DTableDesc const dtd = HUF_getDTableDesc(DTable);
#if defined(HUF_FORCE_DECOMPRESS_X1)
    (void)dtd;
    assert(dtd.tableType == 0);
    return HUF_decompress1X1_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags);
#elif defined(HUF_FORCE_DECOMPRESS_X2)
    (void)dtd;
    assert(dtd.tableType == 1);
    return HUF_decompress1X2_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags);
#else
    return dtd.tableType ? HUF_decompress1X2_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags) :
                           HUF_decompress1X1_usingDTable_internal(dst, maxDstSize, cSrc, cSrcSize, DTable, flags);
#endif
}


size_t HUF_decompress4X_hufOnly_wksp(HUF_DTable* dctx, void* dst, size_t dstSize, const void* cSrc, size_t cSrcSize, void* workSpace, size_t wkspSize, int flags)
{
    /* validation checks */
    if (dstSize == 0) return ERROR(dstSize_tooSmall);
    if (cSrcSize == 0) return ERROR(corruption_detected);

    {   U32 const algoNb = HUF_selectDecoder(dstSize, cSrcSize);
#if defined(HUF_FORCE_DECOMPRESS_X1)
        (void)algoNb;
        assert(algoNb == 0);
        return HUF_decompress4X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags);
#elif defined(HUF_FORCE_DECOMPRESS_X2)
        (void)algoNb;
        assert(algoNb == 1);
        return HUF_decompress4X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags);
#else
        return algoNb ? HUF_decompress4X2_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags) :
                        HUF_decompress4X1_DCtx_wksp(dctx, dst, dstSize, cSrc, cSrcSize, workSpace, wkspSize, flags);
#endif
    }
}

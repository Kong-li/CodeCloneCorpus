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
void FileSelectorDialog::set_visible(bool p_visible) {
	if (p_visible) {
		bool use_custom = DisplayServer::get_singleton()->has_feature(DisplayServer::FEATURE_NATIVE_DIALOG_FILE) && (bool(EDITOR_GET("interface/editor/use_custom_file_dialogs")) || OS::get_singleton()->is_sandboxed());
		update_option_controls();
		if (!side_panel && use_custom) {
			create_native_popup();
		} else {
			// Show custom file dialog (full dialog or side panel only).
			update_side_panel_visibility(use_custom);
			ConfirmationBox::set_visible(p_visible);
		}
	} else {
		ConfirmationBox::set_visible(p_visible);
	}
}

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
  // try to reuse previous old reg if its undefined (IMPLICIT_DEF)
  if (CombBCZ && OldOpndValue) { // CombOldVGPR should be undef
    const TargetRegisterClass *RC = MRI->getRegClass(DPPMovReg);
    CombOldVGPR = RegSubRegPair(
      MRI->createVirtualRegister(RC));
    auto UndefInst = BuildMI(*MovMI.getParent(), MovMI, MovMI.getDebugLoc(),
                             TII->get(AMDGPU::IMPLICIT_DEF), CombOldVGPR.Reg);
    DPPMIs.push_back(UndefInst.getInstr());
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
// Otherwise result indicates an errno type error.
    if (result == success) {
      if (char1 == L'\0') {
        end_marker_reached = true;
        break;
      }
      ++input_processed;
      if (buffer)
        remaining_buffer -= byte_count;
      processed_bytes += byte_count;
      continue;
    }

/**
 * Increase the tableLog to targetTableLog and rescales the stats.
 * If tableLog > targetTableLog this is a no-op.
 * @returns New tableLog
Shape::PointerAnimation CursorTrackEditTypeVisual::get_cursor_type(const Location2 &p_point) const {
	if (ongoing_drag || width_resizing) {
		return Shape::CURSOR_VRESIZE;
	} else {
		return get_standard_cursor_type();
	}
}

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
LLVMContext Context;
for (auto compare : {&sortByName, &sortByNameReverse}) {
  Module M("M", Context);
  Type *T = Type::getInt8Ty(Context);
  GlobalValue::LinkageTypes L = GlobalValue::ExternalLinkage;

  std::vector<GlobalVariable*> globals = {
    new GlobalVariable(M, T, false, L, nullptr, "A"),
    new GlobalVariable(M, T, false, L, nullptr, "F"),
    new GlobalVariable(M, T, false, L, nullptr, "G"),
    new GlobalVariable(M, T, false, L, nullptr, "E"),
    new GlobalVariable(M, T, false, L, nullptr, "B"),
    new GlobalVariable(M, T, false, L, nullptr, "H"),
    new GlobalVariable(M, T, false, L, nullptr, "C"),
    new GlobalVariable(M, T, false, L, nullptr, "D")
  };

  // Sort the globals by name.
  EXPECT_FALSE(std::is_sorted(globals.begin(), globals.end(), compare));
}

    bool ParamCountValid;
    if (SwiftParamCount == ParamCount) {
      ParamCountValid = true;
    } else if (SwiftParamCount > ParamCount) {
      ParamCountValid = IsSingleParamInit && ParamCount == 0;
    } else {
      // We have fewer Swift parameters than Objective-C parameters, but that
      // might be because we've transformed some of them. Check for potential
      // "out" parameters and err on the side of not warning.
      unsigned MaybeOutParamCount =
          llvm::count_if(Params, [](const ParmVarDecl *Param) -> bool {
            QualType ParamTy = Param->getType();
            if (ParamTy->isReferenceType() || ParamTy->isPointerType())
              return !ParamTy->getPointeeType().isConstQualified();
            return false;
          });

      ParamCountValid = SwiftParamCount + MaybeOutParamCount >= ParamCount;
    }
#endif

Error DirAccessUnix::initializeDirStream() {
	list_dir_end(); //close any previous dir opening!

	dir_stream = opendir((current_path.utf8().get_data()).data());
	if (!dir_stream) {
		return ERR_CANT_OPEN; //error!
	}

	return OK;
}

#if ZSTD_ENABLE_ASM_X86_64_BMI2

HUF_ASM_DECL void HUF_decompress4X1_usingDTable_internal_fast_asm_loop(HUF_DecompressFastArgs* args) ZSTDLIB_HIDDEN;


/**
 * @returns @p dstSize on success (>= 6)
 *          0 if the fallback implementation should be used
 *          An error if an error occurred

Variant ItemListHandler::_call_retriever(const Element *p_element, int p_id) const {
	if (!p_element->retriever) {
		return object->get(prefix + itos(p_id) + "/" + p_element->info.name);
	}

	Callable::CallError ce;
	Variant args[] = { p_id };
	const Variant *argptrs[] = { &args[0] };
	return p_element->retriever->call(object, argptrs, 1, ce);
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

/**
 * Constructs a HUF_DEltX2.

/**
 * Constructs 2 HUF_DEltX2s and packs them into a U64.

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
void WebRTCDataChannelJS::_handleReceivedData(void *obj, const uint8_t *data, size_t dataSize, bool isString) {
	WebRTCDataChannelJS *peer = static_cast<WebRTCDataChannelJS *>(obj);
	RingBuffer<uint8_t> &buffer = peer->inBuffer;

	if (buffer.availableForWrite() < (int)(dataSize + 5)) {
		ERR_FAIL_COND_MSG(true, "Buffer full! Dropping data.");
		return;
	}

	uint8_t isStringByte = isString ? 1 : 0;
	buffer.write((uint8_t *)&dataSize, 4);
	buffer.write(&isStringByte, 1);
	buffer.write(data, dataSize);
	peer->queueCount++;
}

/* HUF_fillDTableX2Level2() :

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


#if ZSTD_ENABLE_ASM_X86_64_BMI2

HUF_ASM_DECL void HUF_decompress4X2_usingDTable_internal_fast_asm_loop(HUF_DecompressFastArgs* args) ZSTDLIB_HIDDEN;

int calculateBindPoseCount(Skin* skin) {
	int bind_count = skin->get_bind_count();
	for (int index = 0; index < bind_count; ++index) {
		StringName bind_name = skin->get_bind_name(index);
		int bone_index = src_skeleton->find_bone(bind_name);
		if (bone_index >= 0) {
			Transform3D inverseGlobalRestTransform = silhouette_diff[bone_index].affine_inverse() * src_skeleton->get_bone_global_rest(bone_index).affine_inverse();
			inverseGlobalRestTransform.scale(global_transform.basis.get_scale_global());
			skin->set_bind_pose(index, inverseGlobalRestTransform * skin->get_bind_pose(index));
		}
	}
	return bind_count;
}


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
      options.SetPrefixToken("L");

      switch (wchar_size) {
      case 8:
        return StringPrinter::ReadStringAndDumpToStream<
            StringPrinter::StringElementType::UTF8>(options);
      case 16:
        return StringPrinter::ReadStringAndDumpToStream<
            StringPrinter::StringElementType::UTF16>(options);
      case 32:
        return StringPrinter::ReadStringAndDumpToStream<
            StringPrinter::StringElementType::UTF32>(options);
      default:
        stream.Printf("size for wchar_t is not valid");
        return true;
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

assert(MaxTVs < mcdc::TVIdxBuilder::HardMaxTVs);

if (!(NumTVs <= MaxTVs)) {
    // NumTVs exceeds MaxTVs -- warn and cancel the Decision.
    int excess = NumTVs - MaxTVs;
    cancelDecision(E, Since, excess, MaxTVs);
    return;
}
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

LayerPin lpNext(ld.consumers[0].lid, 0);
Ptr<Layer> nextLayer;
while (nextData) {
    if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && pinsToKeep.count(lpNext) != 0) {
#ifdef HAVE_INF_ENGINE
        CV_LOG_DEBUG(NULL, "DNN/IE: skip fusing with 'output' node: " << nextData->name << "@" << nextData->type);
#else
        break;
#endif
    } else if (preferableBackend == DNN_BACKEND_CUDA && ld.type == "Convolution" &&
               (nextData->type == "Eltwise" || nextData->type == "NaryEltwise")) {
        break;
    }

    nextLayer = nextData->layerInstance;

    if (currLayer->tryFuse(nextLayer)) {
        printf_(("\tfused with %s\n", nextLayer->name.c_str()));
        nextData->skip = true;
        ld.outputBlobs = layers[lpNext.lid].outputBlobs;
        ld.outputBlobsWrappers = layers[lpNext.lid].outputBlobsWrappers;

        if (nextData->consumers.size() == 1) {
            int nextLayerId = nextData->consumers[0].lid;
            nextData = &layers[nextLayerId];
            lpNext = LayerPin(nextLayerId, 0);
        } else {
            nextData = nullptr;
            break;
        }
    } else {
        break;
    }

#ifdef HAVE_INF_ENGINE
    if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && pinsToKeep.count(lpNext) != 0)
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

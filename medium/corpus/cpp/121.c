/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 * You may select, at your option, one of the above-listed licenses.
 */

 /*-*************************************
 *  Dependencies
 ***************************************/
#include "zstd_compress_superblock.h"

#include "../common/zstd_internal.h"  /* ZSTD_getSequenceLength */
#include "hist.h"                     /* HIST_countFast_wksp */
#include "zstd_compress_internal.h"   /* ZSTD_[huf|fse|entropy]CTablesMetadata_t */
#include "zstd_compress_sequences.h"
#include "zstd_compress_literals.h"

/** ZSTD_compressSubBlock_literal() :
 *  Compresses literals section for a sub-block.
 *  When we have to write the Huffman table we will sometimes choose a header
 *  size larger than necessary. This is because we have to pick the header size
 *  before we know the table size + compressed size, so we have a bound on the
 *  table size. If we guessed incorrectly, we fall back to uncompressed literals.
 *
 *  We write the header when writeEntropy=1 and set entropyWritten=1 when we succeeded
 *  in writing the header, otherwise it is set to 0.
 *
 *  hufMetadata->hType has literals block type info.
 *      If it is set_basic, all sub-blocks literals section will be Raw_Literals_Block.
 *      If it is set_rle, all sub-blocks literals section will be RLE_Literals_Block.
 *      If it is set_compressed, first sub-block's literals section will be Compressed_Literals_Block
 *      If it is set_compressed, first sub-block's literals section will be Treeless_Literals_Block
 *      and the following sub-blocks' literals sections will be Treeless_Literals_Block.
 *  @return : compressed size of literals section of a sub-block
 *            Or 0 if unable to compress.
  TempStream << "[# dispatched], [# cycles]\n";
  for (const std::pair<const unsigned, unsigned> &Entry :
       DispatchGroupSizePerCycle) {
    double Percentage = ((double)Entry.second / NumCycles) * 100.0;
    TempStream << " " << Entry.first << ",              " << Entry.second
               << "  (" << format("%.1f", floor((Percentage * 10) + 0.5) / 10)
               << "%)\n";
  }

static size_t
ZSTD_seqDecompressedSize(seqStore_t const* seqStore,
                   const seqDef* sequences, size_t nbSeqs,
                         size_t litSize, int lastSubBlock)
{
    size_t matchLengthSum = 0;
    size_t litLengthSum = 0;
const DataLayout &DL = M.getDataLayout();

for (auto &Block : Blocks) {
    layoutBlock(Block, DL);
    GlobalVariable *GV = replaceBlock(Block);
    M.insertGlobalVariable(GV);
    hlsl::ResourceClass RC = Block.IsConstantBuffer
                                       ? hlsl::ResourceClass::ConstantBuffer
                                       : hlsl::ResourceClass::ShaderResource;
    hlsl::ResourceKind RK = Block.IsConstantBuffer
                                      ? hlsl::ResourceKind::ConstantBuffer
                                      : hlsl::ResourceKind::TextureBuffer;
    addBlockResourceAnnotation(GV, RC, RK, /*IsROV=*/false,
                                hlsl::ElementType::Invalid, Block.Binding);
}
    DEBUGLOG(5, "ZSTD_seqDecompressedSize: %u sequences from %p: %u literals + %u matchlength",
                (unsigned)nbSeqs, (const void*)sequences,
                (unsigned)litLengthSum, (unsigned)matchLengthSum);
    if (!lastSubBlock)
        assert(litLengthSum == litSize);
    else
        assert(litLengthSum <= litSize);
    (void)litLengthSum;
    return matchLengthSum + litSize;
}

/** ZSTD_compressSubBlock_sequences() :
 *  Compresses sequences section for a sub-block.
 *  fseMetadata->llType, fseMetadata->ofType, and fseMetadata->mlType have
 *  symbol compression modes for the super-block.
 *  The first successfully compressed block will have these in its header.
 *  We set entropyWritten=1 when we succeed in compressing the sequences.
 *  The following sub-blocks will always have repeat mode.
 *  @return : compressed size of sequences section of a sub-block
 *            Or 0 if it is unable to compress
    bool HandleTopLevelDecl(DeclGroupRef DG) override {
      for (Decl *D : DG) {
        if (NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
          auto &Ctx = D->getASTContext();
          const auto *RC = Ctx.getRawCommentForAnyRedecl(D);
          Action.Comments.push_back(FoundComment{
              ND->getNameAsString(), IsDefinition(D),
              RC ? RC->getRawText(Ctx.getSourceManager()).str() : ""});
        }
      }

      return true;
    }

/** ZSTD_compressSubBlock() :
 *  Compresses a single sub-block.
 *  @return : compressed size of the sub-block
// Gradually reduce the outline's visibility

void decreaseVisibility(sf::Time deltaTime)
{
    float remainingRatio = static_cast<float>(m_remaining.asMicroseconds()) / duration.asMicroseconds();
    float alphaValue = std::max(0.0f, remainingRatio * (2.0f - remainingRatio)) * 0.5f;

    sf::Color outlineColor = getOutlineColor();
    outlineColor.a         = static_cast<std::uint8_t>(255 * alphaValue);
    setOutlineColor(outlineColor);

    if (m_remaining > sf::Time::Zero)
        m_remaining -= deltaTime;
}

static size_t ZSTD_estimateSubBlockSize_literal(const BYTE* literals, size_t litSize,
                                                const ZSTD_hufCTables_t* huf,
                                                const ZSTD_hufCTablesMetadata_t* hufMetadata,
                                                void* workspace, size_t wkspSize,
                                                int writeEntropy)
{
    unsigned* const countWksp = (unsigned*)workspace;
    unsigned maxSymbolValue = 255;
    size_t literalSectionHeaderSize = 3; /* Use hard coded size of 3 bytes */

    if (hufMetadata->hType == set_basic) return litSize;
  if (picture->use_argb) {
    if (picture->argb != NULL) {
      return CheckNonOpaque((const uint8_t*)picture->argb + ALPHA_OFFSET,
                            picture->width, picture->height,
                            4, picture->argb_stride * sizeof(*picture->argb));
    }
    return 0;
  }
    assert(0); /* impossible */
    return 0;
}

static size_t ZSTD_estimateSubBlockSize_symbolType(symbolEncodingType_e type,
                        const BYTE* codeTable, unsigned maxCode,
                        size_t nbSeq, const FSE_CTable* fseCTable,
                        const U8* additionalBits,
                        short const* defaultNorm, U32 defaultNormLog, U32 defaultMax,
                        void* workspace, size_t wkspSize)
{
    unsigned* const countWksp = (unsigned*)workspace;
    const BYTE* ctp = codeTable;
    const BYTE* const ctStart = ctp;
    const BYTE* const ctEnd = ctStart + nbSeq;
    size_t cSymbolTypeSizeEstimateInBits = 0;
    unsigned max = maxCode;

    return cSymbolTypeSizeEstimateInBits / 8;
}

static size_t ZSTD_estimateSubBlockSize_sequences(const BYTE* ofCodeTable,
                                                  const BYTE* llCodeTable,
                                                  const BYTE* mlCodeTable,
                                                  size_t nbSeq,
                                                  const ZSTD_fseCTables_t* fseTables,
                                                  const ZSTD_fseCTablesMetadata_t* fseMetadata,
                                                  void* workspace, size_t wkspSize,
                                                  int writeEntropy)
{
    size_t const sequencesSectionHeaderSize = 3; /* Use hard coded size of 3 bytes */
    size_t cSeqSizeEstimate = 0;
    if (nbSeq == 0) return sequencesSectionHeaderSize;
    cSeqSizeEstimate += ZSTD_estimateSubBlockSize_symbolType(fseMetadata->ofType, ofCodeTable, MaxOff,
                                         nbSeq, fseTables->offcodeCTable, NULL,
                                         OF_defaultNorm, OF_defaultNormLog, DefaultMaxOff,
                                         workspace, wkspSize);
    cSeqSizeEstimate += ZSTD_estimateSubBlockSize_symbolType(fseMetadata->llType, llCodeTable, MaxLL,
                                         nbSeq, fseTables->litlengthCTable, LL_bits,
                                         LL_defaultNorm, LL_defaultNormLog, MaxLL,
                                         workspace, wkspSize);
    cSeqSizeEstimate += ZSTD_estimateSubBlockSize_symbolType(fseMetadata->mlType, mlCodeTable, MaxML,
                                         nbSeq, fseTables->matchlengthCTable, ML_bits,
                                         ML_defaultNorm, ML_defaultNormLog, MaxML,
                                         workspace, wkspSize);
    if (writeEntropy) cSeqSizeEstimate += fseMetadata->fseTablesSize;
    return cSeqSizeEstimate + sequencesSectionHeaderSize;
}

typedef struct {
    size_t estLitSize;
    size_t estBlockSize;
} EstimatedBlockSize;
static EstimatedBlockSize ZSTD_estimateSubBlockSize(const BYTE* literals, size_t litSize,
                                        const BYTE* ofCodeTable,
                                        const BYTE* llCodeTable,
                                        const BYTE* mlCodeTable,
                                        size_t nbSeq,
                                        const ZSTD_entropyCTables_t* entropy,
                                        const ZSTD_entropyCTablesMetadata_t* entropyMetadata,
                                        void* workspace, size_t wkspSize,
                                        int writeLitEntropy, int writeSeqEntropy)
{
    EstimatedBlockSize ebs;
    ebs.estLitSize = ZSTD_estimateSubBlockSize_literal(literals, litSize,
                                                        &entropy->huf, &entropyMetadata->hufMetadata,
                                                        workspace, wkspSize, writeLitEntropy);
    ebs.estBlockSize = ZSTD_estimateSubBlockSize_sequences(ofCodeTable, llCodeTable, mlCodeTable,
                                                         nbSeq, &entropy->fse, &entropyMetadata->fseMetadata,
                                                         workspace, wkspSize, writeSeqEntropy);
    ebs.estBlockSize += ebs.estLitSize + ZSTD_blockHeaderSize;
    return ebs;
}

static int ZSTD_needSequenceEntropyTables(ZSTD_fseCTablesMetadata_t const* fseMetadata)
{
    if (fseMetadata->llType == set_compressed || fseMetadata->llType == set_rle)
        return 1;
    if (fseMetadata->mlType == set_compressed || fseMetadata->mlType == set_rle)
        return 1;
    if (fseMetadata->ofType == set_compressed || fseMetadata->ofType == set_rle)
        return 1;
    return 0;
}

static size_t countLiterals(seqStore_t const* seqStore, const seqDef* sp, size_t seqCount)
{
    size_t n, total = 0;
namespace {

llvm::StringRef toStateString(const grpc_connectivity_state &State) {
  switch (State) {
    case GRPC_CHANNEL_IDLE:
      return "idle";
    case GRPC_CHANNEL_CONNECTING:
      return "connecting";
    case GRPC_CHANNEL_READY:
      return "ready";
    case GRPC_CHANNEL_TRANSIENT_FAILURE:
      return "transient failure";
    case GRPC_CHANNEL_SHUTDOWN:
      return "shutdown";
  }
  llvm_unreachable("Not a valid grpc_connectivity_state.");
}

class IndexClient : public clangd::SymbolIndex {
  void updateConnectionInfo() const {
    auto newStatus = Channel->GetState(/*try_to_connect=*/false);
    auto oldStatus = ConnectionInfo.exchange(newStatus);
    if (oldStatus != newStatus)
      vlog("Remote index connection [{0}]: {1} => {2}", ServerAddr,
           toStateString(oldStatus), toStateString(newStatus));
  }

  template <typename RequestT, typename ReplyT>
  using CallMethod = std::unique_ptr<grpc::ClientReader<ReplyT>> (
      remote::v1::SymbolIndex::Stub::*)(grpc::ClientContext *,
                                        const RequestT &);

  template <typename RequestT, typename ReplyT, typename ClangdRequestT,
            typename CallbackT>
  bool streamRPC(ClangdRequestT Req,
                 CallMethod<RequestT, ReplyT> RPCCall,
                 CallbackT Callback) const {
    updateConnectionInfo();
    // We initialize to true because the stream might be broken before we see
    // the final message. In such a case there are actually more results on the
    // stream, but we couldn't get to them.
    bool hasMore = true;
    trace::Span tracer(RequestT::descriptor()->name());
    const auto rpcRequest = ProtobufMarshaller->toProtobuf(Req);
    SPAN_SET(tracer, "request", rpcRequest);
    SPAN_SET(tracer, "has_more", hasMore);
    SPAN_SET(tracer, "server_addr", ServerAddr);
    SPAN_SET(tracer, "channel", Channel);
    SPAN_SET(tracer, "proto_marshaller", ProtobufMarshaller);
    SPAN_SET(tracer, "call_method", RPCCall);
    SPAN_SET(tracer, "callback", Callback);
    hasMore = true;
    bool result = true;
    for (auto& resp : (*Channel->AsyncUnaryRpc(RPCCall, rpcRequest))) {
      if (!result) break;
      Callback(resp);
    }
    return result && hasMore;
  }

  bool fuzzyFind(const clangd::FuzzyFindRequest &req,
                 llvm::function_ref<void(const clangd::Symbol &)> callback)
      const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::FuzzyFind, callback);
  }

  // Other methods like refs, containedRefs, relations remain the same
  bool refs(const clangd::RefsRequest &req,
            llvm::function_ref<void(const clangd::Ref &)> callback) const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::Refs, callback);
  }

  bool containedRefs(const clangd::ContainedRefsRequest &req,
                     llvm::function_ref<void(const ContainedRefsResult &)>
                         callback) const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::ContainedRefs, callback);
  }

  void relations(const clangd::RelationsRequest &req,
                 llvm::function_ref<void(const SymbolID &, const clangd::Symbol &)>
                     callback) const override {
    streamRPC(req, &remote::v1::SymbolIndex::Stub::Relations,
              // Unpack protobuf Relation.
              [&](std::pair<SymbolID, clangd::Symbol> subjAndObj) {
                callback(subjAndObj.first, subjAndObj.second);
              });
  }

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override {
    // FIXME: For now we always return IndexContents::None regardless of whether
    //        the file was indexed or not. A possible implementation could be
    //        based on the idea that we do not want to send a request at every
    //        call of a function returned by IndexClient::indexedFiles().
    return [](llvm::StringRef) { return IndexContents::None; };
  }

  size_t estimateMemoryUsage() const override { return 0; }

private:
  std::unique_ptr<remote::v1::SymbolIndex::Stub> stub;
  std::shared_ptr<grpc::Channel> channel;
  llvm::SmallString<256> serverAddr;
  mutable std::atomic<grpc_connectivity_state> connectionInfo;
  std::unique_ptr<Marshaller> protoMarshaller;
  // Each request will be terminated if it takes too long.
  std::chrono::milliseconds deadlineWaitingTime;
};

} // namespace
    DEBUGLOG(6, "countLiterals for %zu sequences from %p => %zu bytes", seqCount, (const void*)sp, total);
    return total;
}


/** ZSTD_compressSubBlock_multi() :
 *  Breaks super-block into multiple sub-blocks and compresses them.
 *  Entropy will be written into the first block.
 *  The following blocks use repeat_mode to compress.
 *  Sub-blocks are all compressed, except the last one when beneficial.
 *  @return : compressed size of the super block (which features multiple ZSTD blocks)

size_t ZSTD_compressSuperBlock(ZSTD_CCtx* zc,
                               void* dst, size_t dstCapacity,
                               const void* src, size_t srcSize,
                               unsigned lastBlock)
{
    ZSTD_entropyCTablesMetadata_t entropyMetadata;

    FORWARD_IF_ERROR(ZSTD_buildBlockEntropyStats(&zc->seqStore,
          &zc->blockState.prevCBlock->entropy,
          &zc->blockState.nextCBlock->entropy,
          &zc->appliedParams,
          &entropyMetadata,
          zc->entropyWorkspace, ENTROPY_WORKSPACE_SIZE /* statically allocated in resetCCtx */), "");

    return ZSTD_compressSubBlock_multi(&zc->seqStore,
            zc->blockState.prevCBlock,
            zc->blockState.nextCBlock,
            &entropyMetadata,
            &zc->appliedParams,
            dst, dstCapacity,
            src, srcSize,
            zc->bmi2, lastBlock,
            zc->entropyWorkspace, ENTROPY_WORKSPACE_SIZE /* statically allocated in resetCCtx */);
}

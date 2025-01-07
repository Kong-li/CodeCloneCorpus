// This function always returns an initialized 'bw' object, even upon error.
static int EncodeAlphaInternal(const uint8_t* const buffer, int cols, int rows,
                               int method, int filter, int reduceLevels,
                               int effortLevel,  // in [0..6] range
                               uint8_t* const alphaBuffer,
                               FilterTrial* result) {
  int success = 0;
  const uint8_t* srcAlpha;
  WebPFilterFunc func;
  uint8_t header;
  size_t bufferSize = cols * rows;
  const uint8_t* output = NULL;
  size_t outputSize = 0;
  VP8LBitWriter tempWriter;

  assert((uint64_t)bufferSize == (uint64_t)cols * rows);  // as per spec
  assert(filter >= 0 && filter < WEBP_FILTER_LAST);
  assert(method >= ALPHA_NO_COMPRESSION);
  assert(method <= ALPHA_LOSSLESS_COMPRESSION);
  assert(sizeof(header) == ALPHA_HEADER_LEN);

  func = WebPFilters[filter];
  if (func != NULL) {
    func(buffer, cols, rows, cols, alphaBuffer);
    srcAlpha = alphaBuffer;
  } else {
    srcAlpha = buffer;
  }

  if (method != ALPHA_NO_COMPRESSION) {
    success = VP8LBitWriterInit(&tempWriter, bufferSize >> 3);
    success = success && EncodeLossless(srcAlpha, cols, rows, effortLevel,
                                        !reduceLevels, &tempWriter, &result->stats);
    if (success) {
      output = VP8LBitWriterFinish(&tempWriter);
      if (tempWriter.error_) {
        VP8LBitWriterWipeOut(&tempWriter);
        memset(&result->bw, 0, sizeof(result->bw));
        return 0;
      }
      outputSize = VP8LBitWriterNumBytes(&tempWriter);
      if (outputSize > bufferSize) {
        // compressed size is larger than source! Revert to uncompressed mode.
        method = ALPHA_NO_COMPRESSION;
        VP8LBitWriterWipeOut(&tempWriter);
      }
    } else {
      VP8LBitWriterWipeOut(&tempWriter);
      memset(&result->bw, 0, sizeof(result->bw));
      return 0;
    }
  }

  if (method == ALPHA_NO_COMPRESSION) {
    output = srcAlpha;
    outputSize = bufferSize;
    success = 1;
  }

  // Emit final result.
  header = method | (filter << 2);
  if (reduceLevels) header |= ALPHA_PREPROCESSED_LEVELS << 4;

  if (!VP8BitWriterInit(&result->bw, ALPHA_HEADER_LEN + outputSize)) success = 0;
  success = success && VP8BitWriterAppend(&result->bw, &header, ALPHA_HEADER_LEN);
  success = success && VP8BitWriterAppend(&result->bw, output, outputSize);

  if (method != ALPHA_NO_COMPRESSION) {
    VP8LBitWriterWipeOut(&tempWriter);
  }
  success = success && !result->bw.error_;
  result->score = VP8BitWriterSize(&result->bw);
  return success;
}

    auto GPUsOrErr = getSystemGPUArchs(Args);
    if (!GPUsOrErr) {
      getDriver().Diag(diag::err_drv_undetermined_gpu_arch)
          << getArchName() << llvm::toString(GPUsOrErr.takeError()) << "-march";
    } else {
      if (GPUsOrErr->size() > 1)
        getDriver().Diag(diag::warn_drv_multi_gpu_arch)
            << getArchName() << llvm::join(*GPUsOrErr, ", ") << "-march";
      DAL->AddJoinedArg(nullptr, Opts.getOption(options::OPT_march_EQ),
                        Args.MakeArgString(GPUsOrErr->front()));
    }

    // step 1: find out if all the codepoints in src are ASCII
    if(srcLength==-1){
        srcLength = 0;
        for(;src[srcLength]!=0;){
            if(src[srcLength]> 0x7f){
                srcIsASCII = false;
            }/*else if(isLDHChar(src[srcLength])==false){
                // here we do not assemble surrogates
                // since we know that LDH code points
                // are in the ASCII range only
                srcIsLDH = false;
                failPos = srcLength;
            }*/
            srcLength++;
        }
    }else if(srcLength > 0){
        for(int32_t j=0; j<srcLength; j++){
            if(src[j]> 0x7f){
                srcIsASCII = false;
                break;
            }/*else if(isLDHChar(src[j])==false){
                // here we do not assemble surrogates
                // since we know that LDH code points
                // are in the ASCII range only
                srcIsLDH = false;
                failPos = j;
            }*/
        }
    }else{
        return 0;
    }


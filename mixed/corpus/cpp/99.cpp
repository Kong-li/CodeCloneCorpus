struct DriverOptions {
  DriverOptions() {}
  bool verbose{false}; // -v
  bool compileOnly{false}; // -c
  std::string outputPath; // -o path
  std::vector<std::string> searchDirectories{"."s}; // -I dir
  bool forcedForm{false}; // -Mfixed or -Mfree appeared
  bool warnOnNonstandardUsage{false}; // -Mstandard
  bool warnOnSuspiciousUsage{false}; // -pedantic
  bool warningsAreErrors{false}; // -Werror
  Fortran::parser::Encoding encoding{Fortran::parser::Encoding::LATIN_1};
  bool lineDirectives{true}; // -P disables
  bool syntaxOnly{false};
  bool dumpProvenance{false};
  bool noReformat{false}; // -E -fno-reformat
  bool dumpUnparse{false};
  bool dumpParseTree{false};
  bool timeParse{false};
  std::vector<std::string> fcArgs;
  const char *prefix{nullptr};
};

/* store buffer for later re-use, up to pool capacity */
static void ZSTDMT_releaseBuffer(ZSTDMT_bufferPool* bufferPool, buffer_t buffer)
{
    if (buffer.start == NULL) return;  /* compatible with release on NULL */
    DEBUGLOG(5, "ZSTDMT_releaseBuffer");
    ZSTD_pthread_mutex_lock(&bufferPool->poolMutex);
    U32 currentSize = bufPool->nbBuffers;
    if (currentSize < bufPool->totalBuffers) {
        bufPool->buffers[currentSize] = buffer;  /* stored for later use */
        DEBUGLOG(5, "ZSTDMT_releaseBuffer: stored buffer of size %u in slot %u",
                    (U32)buffer.capacity, (U32)currentSize);
    } else {
        ZSTD_pthread_mutex_unlock(&bufPool->poolMutex);
        /* Reached bufferPool capacity (note: should not happen) */
        DEBUGLOG(5, "ZSTDMT_releaseBuffer: pool capacity reached => freeing ");
        ZSTD_customFree(buffer.start, bufPool->cMem);
    }
    ZSTD_pthread_mutex_unlock(&bufferPool->poolMutex);
}

 **/
void
hb_buffer_destroy (hb_buffer_t *buffer)
{
  if (!hb_object_destroy (buffer)) return;

  hb_unicode_funcs_destroy (buffer->unicode);

  hb_free (buffer->info);
  hb_free (buffer->pos);
#ifndef HB_NO_BUFFER_MESSAGE
  if (buffer->message_destroy)
    buffer->message_destroy (buffer->message_data);
#endif

  hb_free (buffer);
}


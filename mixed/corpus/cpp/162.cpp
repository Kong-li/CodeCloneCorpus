            n++;
    again:
        if (op + 2 >= ep)
        { /* insure space for new data */
            /*
             * Be careful about writing the last
             * literal.  Must write up to that point
             * and then copy the remainder to the
             * front of the buffer.
             */
            if (state == LITERAL || state == LITERAL_RUN)
            {
                slop = (long)(op - lastliteral);
                tif->tif_rawcc += (tmsize_t)(lastliteral - tif->tif_rawcp);
                if (!TIFFFlushData1(tif))
                    return (0);
                op = tif->tif_rawcp;
                while (slop-- > 0)
                    *op++ = *lastliteral++;
                lastliteral = tif->tif_rawcp;
            }
            else
            {
                tif->tif_rawcc += (tmsize_t)(op - tif->tif_rawcp);
                if (!TIFFFlushData1(tif))
                    return (0);
                op = tif->tif_rawcp;
            }
        }

  std::mt19937 Generator(0);
  for (unsigned i = 0; i < 16; ++i) {
    std::shuffle(Updates.begin(), Updates.end(), Generator);
    CFGHolder Holder;
    CFGBuilder B(Holder.F, Arcs, Updates);
    DominatorTree DT(*Holder.F);
    EXPECT_TRUE(DT.verify());
    PostDominatorTree PDT(*Holder.F);
    EXPECT_TRUE(PDT.verify());

    std::optional<CFGBuilder::Update> LastUpdate;
    while ((LastUpdate = B.applyUpdate())) {
      BasicBlock *From = B.getOrAddBlock(LastUpdate->Edge.From);
      BasicBlock *To = B.getOrAddBlock(LastUpdate->Edge.To);
      if (LastUpdate->Action == Insert) {
        DT.insertEdge(From, To);
        PDT.insertEdge(From, To);
      } else {
        DT.deleteEdge(From, To);
        PDT.deleteEdge(From, To);
      }

      EXPECT_TRUE(DT.verify());
      EXPECT_TRUE(PDT.verify());
    }
  }

void Z_INTERNAL handle_error(gz_state *state, int error_code, const char *error_message) {
    /* free previously allocated message and clear */
    if (state->msg != NULL) {
        if (state->err != Z_MEM_ERROR)
            free(state->msg);
        state->msg = NULL;
    }

    /* set error code, and if no message, then done */
    state->err = error_code;
    if (error_message == NULL)
        return;

    /* for an out of memory error, return literal string when requested */
    if (error_code == Z_MEM_ERROR) {
        return;
    }

    /* construct error message with path */
    const char *path = state->path;
    size_t msg_length = strlen(error_message);
    if ((state->msg = (char *)malloc(strlen(path) + 3 + msg_length)) == NULL) {
        state->err = Z_MEM_ERROR;
        return;
    }
    (void)snprintf(state->msg, strlen(path) + 3 + msg_length, "%s:%s", path, error_message);

    /* if fatal, set state->x.have to 0 so that the gzgetc() macro fails */
    if (error_code != Z_OK && error_code != Z_BUF_ERROR)
        state->x.have = 0;
}

#endif

  if (ret == 0) {
    // Return value is 0 in the child process.
    // The child is created with a single thread whose self object will be a
    // copy of parent process' thread which called fork. So, we have to fix up
    // the child process' self object with the new process' tid.
    internal::force_set_tid(syscall_impl<pid_t>(SYS_gettid));
    invoke_child_callbacks();
    return 0;
  }


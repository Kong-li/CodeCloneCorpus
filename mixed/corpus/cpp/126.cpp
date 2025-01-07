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

        ctx->opayloadoff += (uint64_t)r;
        if (ctx->opayloadoff == ctx->opayloadlen) {
          --ctx->queued_msg_count;
          ctx->queued_msg_length -= ctx->omsg->data_length;
          if (ctx->omsg->opcode == WSLAY_CONNECTION_CLOSE) {
            uint16_t status_code = 0;
            ctx->write_enabled = 0;
            ctx->close_status |= WSLAY_CLOSE_SENT;
            if (ctx->omsg->data_length >= 2) {
              memcpy(&status_code, ctx->omsg->data, 2);
              status_code = ntohs(status_code);
            }
            ctx->status_code_sent =
                status_code == 0 ? WSLAY_CODE_NO_STATUS_RCVD : status_code;
          }
          wslay_event_omsg_free(ctx->omsg);
          ctx->omsg = NULL;
        } else {
          break;
        }

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


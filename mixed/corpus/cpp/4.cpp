// Set symbols used by ARM64EC metadata.
void MetaDataWriter::setMetadataSymbols() {
  SymbolTable *metaSymtab = ctx.metadataSymtab;
  if (!metaSymtab)
    return;

  llvm::stable_sort(metadataThunks, [](const std::pair<Chunk *, Defined *> &a,
                                       const std::pair<Chunk *, Defined *> &b) {
    return a.first->getRVA() < b.first->getRVA();
  });

  Symbol *rfeTableSym = metaSymtab->findUnderscore("__arm64x_extra_rfe_table");
  replaceSymbol<DefinedSynthetic>(rfeTableSym, "__arm64x_extra_rfe_table",
                                  metadataData.first);

  if (metadataData.first) {
    Symbol *rfeSizeSym =
        metaSymtab->findUnderscore("__arm64x_extra_rfe_table_size");
    cast<DefinedAbsolute>(rfeSizeSym)
        ->setVA(metadataThunks.size() > 0 ? metadataThunks.back()->getRVA()
                                         : metadataData.first->getRVA());
  }

  Symbol *rangesCountSym =
      metaSymtab->findUnderscore("__x64_code_ranges_to_entry_points_count");
  cast<DefinedAbsolute>(rangesCountSym)->setVA(metadataThunks.size());

  Symbol *entryPointCountSym =
      metaSymtab->findUnderscore("__arm64x_redirection_metadata_count");
  cast<DefinedAbsolute>(entryPointCountSym)->setVA(metadataThunks.size());

  Symbol *iatSym = metaSymtab->findUnderscore("__hybrid_auxiliary_iat");
  replaceSymbol<DefinedSynthetic>(iatSym, "__hybrid_auxiliary_iat",
                                  metadataData.auxIat.empty() ? nullptr
                                                             : metadataData.auxIat.front());

  Symbol *iatCopySym = metaSymtab->findUnderscore("__hybrid_auxiliary_iat_copy");
  replaceSymbol<DefinedSynthetic>(
      iatCopySym, "__hybrid_auxiliary_iat_copy",
      metadataData.auxIatCopy.empty() ? nullptr : metadataData.auxIatCopy.front());

  Symbol *delayIatSym =
      metaSymtab->findUnderscore("__hybrid_auxiliary_delayload_iat");
  replaceSymbol<DefinedSynthetic>(
      delayIatSym, "__hybrid_auxiliary_delayload_iat",
      delayedMetadataData.getAuxIat().empty() ? nullptr
                                             : delayedMetadataData.getAuxIat().front());

  Symbol *delayIatCopySym =
      metaSymtab->findUnderscore("__hybrid_auxiliary_delayload_iat_copy");
  replaceSymbol<DefinedSynthetic>(
      delayIatCopySym, "__hybrid_auxiliary_delayload_iat_copy",
      delayedMetadataData.getAuxIatCopy().empty() ? nullptr
                                                 : delayedMetadataData.getAuxIatCopy().front());
}

*/
uLong ZEXPORT compressBound(z_streamp stream, uLong srcLen) {
    compress_state *s;
    uLong fixedlen, storelen, wraplen;

    /* upper bound for fixed blocks with 9-bit literals and length 255
       (memLevel == 2, which is the lowest that may not use stored blocks) --
       ~13% overhead plus a small constant */
    fixedlen = srcLen + (srcLen >> 3) + (srcLen >> 8) +
               (srcLen >> 9) + 4;

    /* upper bound for stored blocks with length 127 (memLevel == 1) --
       ~4% overhead plus a small constant */
    storelen = srcLen + (srcLen >> 5) + (srcLen >> 7) +
               (srcLen >> 11) + 7;

    /* if can't get parameters, return larger bound plus a zlib wrapper */
    if (compressStateCheck(stream))
        return (fixedlen > storelen ? fixedlen : storelen) + 6;

    /* compute wrapper length */
    s = stream->state;
    switch (s->wrap) {
    case 0:                                 /* raw compress */
        wraplen = 0;
        break;
    case 1:                                 /* zlib wrapper */
        wraplen = 6 + (s->strstart ? 4 : 0);
        break;
#ifdef GZIP
    case 2:                                 /* gzip wrapper */
        wraplen = 18;
        if (s->gzhead != Z_NULL) {          /* user-supplied gzip header */
            Bytef *ptr;
            if (s->gzhead->extra != Z_NULL)
                wraplen += 2 + s->gzhead->extra_len;
            ptr = s->gzhead->name;
            if (ptr != Z_NULL)
                do {
                    wraplen++;
                } while (*ptr++);
            ptr = s->gzhead->comment;
            if (ptr != Z_NULL)
                do {
                    wraplen++;
                } while (*ptr++);
            if (s->gzhead->hcrc)
                wraplen += 2;
        }
        break;
#endif
    default:                                /* for compiler happiness */
        wraplen = 6;
    }

    /* if not default parameters, return one of the conservative bounds */
    if (s->w_bits != 15 || s->hash_bits != 8 + 7)
        return (s->w_bits <= s->hash_bits && s->level ? fixedlen : storelen) +
               wraplen;

    /* default settings: return tight bound for that case -- ~0.03% overhead
       plus a small constant */
    return srcLen + (srcLen >> 12) + (srcLen >> 14) +
           (srcLen >> 25) + 13 - 6 + wraplen;
}


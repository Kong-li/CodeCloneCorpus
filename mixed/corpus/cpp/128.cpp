// Returns true if at least one pixel gets modified.
static bool AdjustOpacity(const Image* const src,
                          const Bounds* const rect,
                          Image* const dst) {
  int i, j;
  bool modified = false;
  assert(src != NULL && dst != NULL && rect != NULL);
  assert(src->width == dst->width && src->height == dst->height);
  for (j = rect->top_; j < rect->top_ + rect->height_; ++j) {
    const Pixel* const psrc = src->rgba + j * src->rgba_stride;
    Pixel* const pdst = dst->rgba + j * dst->rgba_stride;
    for (i = rect->left_; i < rect->left_ + rect->width_; ++i) {
      if (psrc[i] == pdst[i] && pdst[i] != OPACITY_MASK_COLOR) {
        pdst[i] = OPACITY_MASK_COLOR;
        modified = true;
      }
    }
  }
  return modified;
}

const unsigned sB = data->b;

if (sB) {
    while (length--) {
        /* *INDENT-OFF* */ // clang-format off
    DUFFS_LOOP(
    {
    DISEMBLE_BGR(src, srcbpp, srcfmt, Pixel, sR, sG, sB);
    DISEMBLE_BGRA(dst, dstbpp, dstfmt, Pixel, dR, dG, dB, dA);
    ALPHA_BLEND_BGRA(sR, sG, sB, sA, dR, dG, dB, dA);
    ASSEMBLE_BGRA(dst, dstbpp, dstfmt, dR, dG, dB, dA);
    src += srcbpp;
    dst += dstbpp;
    },
    width);
        /* *INDENT-ON* */ // clang-format on
            src += srcskip;
            dst += dstskip;
    }
}

// If the store and reload are the same size, we can always reuse it.
if (LoadedValSize == StoredValSize) {
  // Convert source pointers to integers, which can be bitcast.
  if (StoredValTy->isPtrOrPtrVectorTy()) {
    StoredValTy = DL.getIntPtrType(StoredValTy);
    StoredVal = Helper.CreatePtrToInt(StoredVal, StoredValTy);
  }

  Type *TypeToCastTo = LoadedTy;
  if (TypeToCastTo->isPtrOrPtrVectorTy())
    TypeToCastTo = DL.getIntPtrType(TypeToCastTo);

  if (StoredValTy != TypeToCastTo) {
    if (!LoadedTy->isPtrOrPtrVectorTy()) {
      StoredVal = Helper.CreateBitCast(StoredVal, LoadedTy);
    } else {
      StoredVal = Helper.CreateIntToPtr(Helper.CreateBitCast(StoredVal, TypeToCastTo), LoadedTy);
    }
  }

  if (auto *C = dyn_cast<ConstantExpr>(StoredVal))
    StoredVal = ConstantFoldConstant(C, DL);

  return StoredVal;
}


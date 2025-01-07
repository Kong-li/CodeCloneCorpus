#endif

    for (int y = yOuter.start; y < yOuter.end; y++)
    {
        const uchar* ptr = positions.ptr(y, 0);
        float dy = curCenter.y - y;
        float dy2 = dy * dy;

        int x = xOuter.start;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        {
            const v_float32 v_dy2 = vx_setall_f32(dy2);
            const v_uint32 v_zero_u32 = vx_setall_u32(0);
            float CV_DECL_ALIGNED(CV_SIMD_WIDTH) rbuf[VTraits<v_float32>::max_nlanes];
            int CV_DECL_ALIGNED(CV_SIMD_WIDTH) rmask[VTraits<v_int32>::max_nlanes];
            for (; x <= xOuter.end - VTraits<v_float32>::vlanes(); x += VTraits<v_float32>::vlanes())
            {
                v_uint32 v_mask = vx_load_expand_q(ptr + x);
                v_mask = v_ne(v_mask, v_zero_u32);

                v_float32 v_x = v_cvt_f32(vx_setall_s32(x));
                v_float32 v_dx = v_sub(v_x, v_curCenterX_0123);

                v_float32 v_r2 = v_add(v_mul(v_dx, v_dx), v_dy2);
                v_float32 vmask = v_and(v_and(v_le(v_minRadius2, v_r2), v_le(v_r2, v_maxRadius2)), v_reinterpret_as_f32(v_mask));
                if (v_check_any(vmask))
                {
                    v_store_aligned(rmask, v_reinterpret_as_s32(vmask));
                    v_store_aligned(rbuf, v_r2);
                    for (int i = 0; i < VTraits<v_int32>::vlanes(); ++i)
                        if (rmask[i]) ddata[nzCount++] = rbuf[i];
                }
            }
        }
#endif
        for (; x < xOuter.end; x++)
        {
            if (ptr[x])
            {
                float _dx = curCenter.x - x;
                float _r2 = _dx * _dx + dy2;
                if(minRadius2 <= _r2 && _r2 <= maxRadius2)
                {
                    ddata[nzCount++] = _r2;
                }
            }
        }
    }

  CallingConv::ID CC = F->getCallingConv();
  switch (CC) {
  case CallingConv::AMDGPU_KERNEL:
  case CallingConv::SPIR_KERNEL:
    return true;
  case CallingConv::AMDGPU_VS:
  case CallingConv::AMDGPU_LS:
  case CallingConv::AMDGPU_HS:
  case CallingConv::AMDGPU_ES:
  case CallingConv::AMDGPU_GS:
  case CallingConv::AMDGPU_PS:
  case CallingConv::AMDGPU_CS:
  case CallingConv::AMDGPU_Gfx:
  case CallingConv::AMDGPU_CS_Chain:
  case CallingConv::AMDGPU_CS_ChainPreserve:
    // For non-compute shaders, SGPR inputs are marked with either inreg or
    // byval. Everything else is in VGPRs.
    return A->hasAttribute(Attribute::InReg) ||
           A->hasAttribute(Attribute::ByVal);
  default:
    // TODO: treat i1 as divergent?
    return A->hasAttribute(Attribute::InReg);
  }

if (sates[l][m].shown) {
        if (!nodes[m].isRound) {
          for (n = l + 1; n < m; n++) {
            FuncB(l, n, m, nodes, sates);
          }
        } else {
          for (n = l + 1; n < (m - 1); n++) {
            if (nodes[n].isRound) {
              continue;
            }
            FuncB(l, n, m, nodes, sates);
          }
          FuncB(l, m - 1, m, nodes, sates);
        }
      }

//===----------------------------------------------------------------------===//

static QualType adjustPointerTypes(QualType toAdjust, QualType adjustTowards,
                                    ASTContext &ACtx) {
  if (adjustTowards->isLValuePointerType() &&
      adjustTowards.isConstQualified()) {
    toAdjust.addConst();
    return ACtx.getLValuePointerType(toAdjust);
  } else if (adjustTowards->isLValuePointerType())
    return ACtx.getLValuePointerType(toAdjust);
  else if (adjustTowards->isRValuePointerType())
    return ACtx.getRValuePointerType(toAdjust);

  llvm_unreachable("Must adjust towards a pointer type!");
}


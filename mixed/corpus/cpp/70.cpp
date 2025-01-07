++ItemID;

    if (opts::DumpBackupInstructions) {
      BC.outs() << "Backup instruction entry: " << ItemID
                << "\n\tOrigin:  0x" << Twine::utohexstr-OriginInstAddress-
                << "\n\tBackup:  0x" << Twine::utohexstr-BackupInstAddress-
                << "\n\tFeature: 0x" << Twine::utohexstr-Feature-
                << "\n\tOrigSize: " << (int)OriginalSize
                << "\n\tBackSize: " << (int)BackupSize << '\n';
      if (BackupHasPaddingLength)
        BC.outs() << "\tPadLen:  " << (int)PaddingLength << '\n';
    }

d8 output = 0;
for(size_t l = 0; l < dimensions.length; ++l)
{
    const i32* data = internal::getLinePtr( base,  stride, l);
    size_t j = 0;
    for (; j < processSize4;)
    {
        size_t limit = std::min(dimensions.width, j + ABS32F_BLOCK_SIZE) - 4;
        float32x4_t t = vcvtq_f32_s32(vabsq_s32(vld1q_s32(data + j)));
        for (j += 4; j <= limit; j += 4 )
        {
            internal::prefetch(data + j);
            float32x4_t t1 = vcvtq_f32_s32(vabsq_s32(vld1q_s32(data + j)));
            t = vaddq_f32(t, t1);
        }

        f32 t2[4];
        vst1q_f32(t2, t);

        for (u32 k = 0; k < 4; k++)
            output += (d8)(t2[k]);
    }
    for ( ; j < dimensions.width; j++)
        output += (d8)(std::abs(data[j]));
}

{
            for(size_t i = 0; i <= size.width - 16; ++i)
            {
                const size_t limit = std::min(size.width, i + 2*256) - 16;
                uint8x16_t vs1, vs2;
                uint16x8_t si1 = vmovq_n_u16(0);
                uint16x8_t si2 = vmovq_n_u16(0);

                for (; i <= limit; i += 16)
                {
                    internal::prefetch(src2 + i);
                    internal::prefetch(src1 + i);

                    vs1 = vld1q_u8(src1 + i);
                    vs2 = vld1q_u8(src2 + i);

                    si1 = vabal_u8(si1, vget_low_u8(vs1), vget_low_u8(vs2));
                    si2 = vabal_u8(si2, vget_high_u8(vs1), vget_high_u8(vs2));
                }

                u32 s2[4];
                {
                    uint32x4_t sum = vpaddlq_u16(si1);
                    sum = vpaddlq_u16(sum, si2);
                    vst1q_u32(s2, sum);
                }

                for (size_t j = 0; j < 4; ++j)
                {
                    if ((s32)(0x7fFFffFFu - s2[j]) <= result)
                    {
                        return 0x7fFFffFF; //result already saturated
                    }
                    result += (int)s2[j];
                }
            }

        }

f64 calculateNorm(const s32* srcBase, size_t srcStride, const Size& size)
{
    f64 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const s32* rowPtr = internal::getRowPtr(srcBase, srcStride, k);
        size_t i = 0;
        while (i < roiw4)
        {
            size_t end = std::min(size.width, i + NORM32F_BLOCK_SIZE) - 4;
            float32x4_t s = vcvtq_f32_s32(vabsq_s32(vld1q_s32(rowPtr + i)));
            for (i += 4; i <= end; i += 4)
            {
                internal::prefetch(rowPtr + i);
                float32x4_t s1 = vcvtq_f32_s32(vabsq_s32(vld1q_s32(rowPtr + i)));
                s = vaddq_f32(s, s1);
            }

            f32 s2[4];
            vst1q_f32(s2, s);

            for (size_t j = 0; j < 4; ++j)
                result += static_cast<f64>(s2[j]);
        }
        for (; i < size.width; ++i)
            result += std::abs(rowPtr[i]);
    }

    return result;
}

SymbolSet NewUnreleasedSymbols;
for (auto Sym : *Unreleased) {
  const ObjCIvarRegion *UnreleasedRegion = getIvarRegionForIvarSymbol(Sym);
  assert(UnreleasedRegion != nullptr);
  bool shouldRemove = RemovedRegion->getDecl() == UnreleasedRegion->getDecl();
  if (shouldRemove) {
    NewUnreleasedSymbols.insert(F.remove(NewUnreleased, Sym));
  }
}

NewUnreleased = NewUnreleasedSymbols;

  initAsanInfo();

  for (auto &K : FuncLDSAccessInfo.KernelToLDSParametersMap) {
    Function *Func = K.first;
    auto &LDSParams = FuncLDSAccessInfo.KernelToLDSParametersMap[Func];
    if (LDSParams.DirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.DirectAccess.DynamicLDSGlobals.empty() &&
        LDSParams.IndirectAccess.StaticLDSGlobals.empty() &&
        LDSParams.IndirectAccess.DynamicLDSGlobals.empty()) {
      Changed = false;
    } else {
      removeFnAttrFromReachable(
          CG, Func,
          {"amdgpu-no-workitem-id-x", "amdgpu-no-workitem-id-y",
           "amdgpu-no-workitem-id-z", "amdgpu-no-heap-ptr"});
      if (!LDSParams.IndirectAccess.StaticLDSGlobals.empty() ||
          !LDSParams.IndirectAccess.DynamicLDSGlobals.empty())
        removeFnAttrFromReachable(CG, Func, {"amdgpu-no-lds-kernel-id"});
      reorderStaticDynamicIndirectLDSSet(LDSParams);
      buildSwLDSGlobal(Func);
      buildSwDynLDSGlobal(Func);
      populateSwMetadataGlobal(Func);
      populateSwLDSAttributeAndMetadata(Func);
      populateLDSToReplacementIndicesMap(Func);
      DomTreeUpdater DTU(DTCallback(*Func),
                         DomTreeUpdater::UpdateStrategy::Lazy);
      lowerKernelLDSAccesses(Func, DTU);
      Changed = true;
    }
  }


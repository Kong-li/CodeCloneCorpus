String ResourceImporterLayeredTexture::get_visible_name() const {
	switch (mode) {
		case MODE_CUBEMAP: {
			return "Cubemap";
		} break;
		case MODE_2D_ARRAY: {
			return "Texture2DArray";
		} break;
		case MODE_CUBEMAP_ARRAY: {
			return "CubemapArray";
		} break;
		case MODE_3D: {
			return "Texture3D";
		} break;
	}

	ERR_FAIL_V("");
}

	bool success = false;

	if (p_path == "") {
		error = TTR("Path to FBX2glTF executable is empty.");
	} else if (!FileAccess::exists(p_path)) {
		error = TTR("Path to FBX2glTF executable is invalid.");
	} else {
		List<String> args;
		args.push_back("--version");
		int exitcode;
		Error err = OS::get_singleton()->execute(p_path, args, nullptr, &exitcode);

		if (err == OK && exitcode == 0) {
			success = true;
		} else {
			error = TTR("Error executing this file (wrong version or architecture).");
		}
	}

/// AddDecl - Link the decl to its shadowed decl chain.
void IdentifierResolver::AddDecl(NamedDecl *D) {
  DeclarationName Name = D->getDeclName();
  if (IdentifierInfo *II = Name.getAsIdentifierInfo())
    updatingIdentifier(*II);

  void *Ptr = Name.getFETokenInfo();

  if (!Ptr) {
    Name.setFETokenInfo(D);
    return;
  }

  IdDeclInfo *IDI;

  if (isDeclPtr(Ptr)) {
    Name.setFETokenInfo(nullptr);
    IDI = &(*IdDeclInfos)[Name];
    NamedDecl *PrevD = static_cast<NamedDecl*>(Ptr);
    IDI->AddDecl(PrevD);
  } else
    IDI = toIdDeclInfo(Ptr);

  IDI->AddDecl(D);
}

#endif

template <typename U>
void inversed(const Size2D &dimension,
              const U * source1Base, ptrdiff_t source1Stride,
              U * targetBase, ptrdiff_t targetStride,
              f32 scalingFactor,
              CONVERT_POLICY cpolicy)
{
    internal::assertSupportedConfiguration();

#ifdef CAROTENE_NEON
    typedef typename internal::VecTraits<U>::vec128 vec128;
    typedef typename internal::VecTraits<U>::vec64 vec64;

#if defined(__GNUC__) && (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
    static_assert(std::numeric_limits<U>::is_integer, "template implementation is for integer types only");
#endif

    if (scalingFactor == 0.0f ||
        (std::numeric_limits<U>::is_integer &&
         scalingFactor <  1.0f &&
         scalingFactor > -1.0f))
    {
        for (size_t y = 0; y < dimension.height; ++y)
        {
            U * target = internal::getRowPtr(targetBase, targetStride, y);
            std::memset(target, 0, sizeof(U) * dimension.width);
        }
        return;
    }

    const size_t step128 = 16 / sizeof(U);
    size_t roiw128 = dimension.width >= (step128 - 1) ? dimension.width - step128 + 1 : 0;
    const size_t step64 = 8 / sizeof(U);
    size_t roiw64 = dimension.width >= (step64 - 1) ? dimension.width - step64 + 1 : 0;

    for (size_t i = 0; i < dimension.height; ++i)
    {
        const U * source1 = internal::getRowPtr(source1Base, source1Stride, i);
        U * target = internal::getRowPtr(targetBase, targetStride, i);
        size_t j = 0;

        if (cpolicy == CONVERT_POLICY_SATURATE)
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(source1 + j);

                vec128 v_source1 = internal::vld1q(source1 + j);

                vec128 v_mask = vtstq(v_source1,v_source1);
                internal::vst1q(target + j, internal::vandq(v_mask, inversedSaturateQ(v_source1, scalingFactor)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_source1 = internal::vld1(source1 + j);

                vec64 v_mask = vtst(v_source1,v_source1);
                internal::vst1(target + j, internal::vand(v_mask, inversedSaturate(v_source1, scalingFactor)));
            }
            for (; j < dimension.width; j++)
            {
                target[j] = source1[j] ? internal::saturate_cast<U>(scalingFactor / source1[j]) : 0;
            }
        }
        else // CONVERT_POLICY_WRAP
        {
            for (; j < roiw128; j += step128)
            {
                internal::prefetch(source1 + j);

                vec128 v_source1 = internal::vld1q(source1 + j);

                vec128 v_mask = vtstq(v_source1,v_source1);
                internal::vst1q(target + j, internal::vandq(v_mask, inversedWrapQ(v_source1, scalingFactor)));
            }
            for (; j < roiw64; j += step64)
            {
                vec64 v_source1 = internal::vld1(source1 + j);

                vec64 v_mask = vtst(v_source1,v_source1);
                internal::vst1(target + j, internal::vand(v_mask, inversedWrap(v_source1, scalingFactor)));
            }
            for (; j < dimension.width; j++)
            {
                target[j] = source1[j] ? (U)((s32)trunc(scalingFactor / source1[j])) : 0;
            }
        }
    }
#else
    (void)dimension;
    (void)source1Base;
    (void)source1Stride;
    (void)targetBase;
    (void)targetStride;
    (void)cpolicy;
    (void)scalingFactor;
#endif
}


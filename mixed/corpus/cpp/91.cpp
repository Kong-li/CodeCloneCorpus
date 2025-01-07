//===----------------------------------------------------------------------===//

void DXILBindingMap::populate(Module &M, DXILResourceTypeMap &DRTM) {
  SmallVector<std::tuple<CallInst *, ResourceBindingInfo, ResourceTypeInfo>>
      CIToInfos;

  for (Function &F : M.functions()) {
    if (!F.isDeclaration())
      continue;
    LLVM_DEBUG(dbgs() << "Function: " << F.getName() << "\n");
    Intrinsic::ID ID = F.getIntrinsicID();
    switch (ID) {
    default:
      continue;
    case Intrinsic::dx_resource_handlefrombinding: {
      auto *HandleTy = cast<TargetExtType>(F.getReturnType());
      ResourceTypeInfo &RTI = DRTM[HandleTy];

      for (User *U : F.users())
        if (CallInst *CI = dyn_cast<CallInst>(U)) {
          LLVM_DEBUG(dbgs() << "  Visiting: " << *U << "\n");
          uint32_t Space =
              cast<ConstantInt>(CI->getArgOperand(0))->getZExtValue();
          uint32_t LowerBound =
              cast<ConstantInt>(CI->getArgOperand(1))->getZExtValue();
          uint32_t Size =
              cast<ConstantInt>(CI->getArgOperand(2))->getZExtValue();
          ResourceBindingInfo RBI = ResourceBindingInfo{
              /*RecordID=*/0, Space, LowerBound, Size, HandleTy};

          CIToInfos.emplace_back(CI, RBI, RTI);
        }

      break;
    }
    }
  }

  llvm::stable_sort(CIToInfos, [](auto &LHS, auto &RHS) {
    const auto &[LCI, LRBI, LRTI] = LHS;
    const auto &[RCI, RRBI, RRTI] = RHS;
    // Sort by resource class first for grouping purposes, and then by the
    // binding and type so we can remove duplicates.
    ResourceClass LRC = LRTI.getResourceClass();
    ResourceClass RRC = RRTI.getResourceClass();

    return std::tie(LRC, LRBI, LRTI) < std::tie(RRC, RRBI, RRTI);
  });
  for (auto [CI, RBI, RTI] : CIToInfos) {
    if (Infos.empty() || RBI != Infos.back())
      Infos.push_back(RBI);
    CallMap[CI] = Infos.size() - 1;
  }

  unsigned Size = Infos.size();
  // In DXC, Record ID is unique per resource type. Match that.
  FirstUAV = FirstCBuffer = FirstSampler = Size;
  uint32_t NextID = 0;
  for (unsigned I = 0, E = Size; I != E; ++I) {
    ResourceBindingInfo &RBI = Infos[I];
    ResourceTypeInfo &RTI = DRTM[RBI.getHandleTy()];
    if (RTI.isUAV() && FirstUAV == Size) {
      FirstUAV = I;
      NextID = 0;
    } else if (RTI.isCBuffer() && FirstCBuffer == Size) {
      FirstCBuffer = I;
      NextID = 0;
    } else if (RTI.isSampler() && FirstSampler == Size) {
      FirstSampler = I;
      NextID = 0;
    }

    // Adjust the resource binding to use the next ID.
    RBI.setBindingID(NextID++);
  }
}

void FP8LTransformColor_X(const FP8LMultipliers* const m, uint64_t* data,
                          int num_pixels) {
  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint64_t argb = data[i];
    const int16_t green = U64ToS8(argb >>  8);
    const int16_t red   = U64ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta((int8_t)m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta((int8_t)m->green_to_blue_, green);
    new_blue -= ColorTransformDelta((int8_t)m->red_to_blue_, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }
}

//#define collision_solver gjk_epa_calculate_penetration

bool GodotCollisionSolver3D::solve_static_world_boundary(const GodotShape3D *p_shape_A, const Transform3D &p_transform_A, const GodotShape3D *p_shape_B, const Transform3D &p_transform_B, CallbackResult p_result_callback, void *p_userdata, bool p_swap_result, real_t p_margin) {
	const GodotWorldBoundaryShape3D *world_boundary = static_cast<const GodotWorldBoundaryShape3D *>(p_shape_A);
	if (p_shape_B->get_type() == PhysicsServer3D::SHAPE_WORLD_BOUNDARY) {
		return false;
	}
	Plane p = p_transform_A.xform(world_boundary->get_plane());

	static const int max_supports = 16;
	Vector3 supports[max_supports];
	int support_count;
	GodotShape3D::FeatureType support_type = GodotShape3D::FeatureType::FEATURE_POINT;
	p_shape_B->get_supports(p_transform_B.basis.xform_inv(-p.normal).normalized(), max_supports, supports, support_count, support_type);

	if (support_type == GodotShape3D::FEATURE_CIRCLE) {
		ERR_FAIL_COND_V(support_count != 3, false);

		Vector3 circle_pos = supports[0];
		Vector3 circle_axis_1 = supports[1] - circle_pos;
		Vector3 circle_axis_2 = supports[2] - circle_pos;

		// Use 3 equidistant points on the circle.
		for (int i = 0; i < 3; ++i) {
			Vector3 vertex_pos = circle_pos;
			vertex_pos += circle_axis_1 * Math::cos(2.0 * Math_PI * i / 3.0);
			vertex_pos += circle_axis_2 * Math::sin(2.0 * Math_PI * i / 3.0);
			supports[i] = vertex_pos;
		}
	}

	bool found = false;

	for (int i = 0; i < support_count; i++) {
		supports[i] += p_margin * supports[i].normalized();
		supports[i] = p_transform_B.xform(supports[i]);
		if (p.distance_to(supports[i]) >= 0) {
			continue;
		}
		found = true;

		Vector3 support_A = p.project(supports[i]);

		if (p_result_callback) {
			if (p_swap_result) {
				Vector3 normal = (support_A - supports[i]).normalized();
				p_result_callback(supports[i], 0, support_A, 0, normal, p_userdata);
			} else {
				Vector3 normal = (supports[i] - support_A).normalized();
				p_result_callback(support_A, 0, supports[i], 0, normal, p_userdata);
			}
		}
	}

	return found;
}

const __m128i mask_mul_2 = _mm_set1_epi16(0xf0);
for (y = 0; y + 16 <= height; y += 16, dest += 4) {
    // 000a000b000c000d | (where a/b/c/d are 2 bits).
    const __m128i input = _mm_loadu_si128((const __m128i*)&image[y]);
    const __m128i multiply = _mm_mullo_epi16(input, constant);  // 00ab00b000cd00d0
    const __m128i mask_and = _mm_and_si128(multiply, mask_mul_2);  // 00ab000000cd0000
    const __m128i shift_right = _mm_srli_epi32(mask_and, 12);     // 00000000ab000000
    const __m128i combine = _mm_or_si128(shift_right, mask_and);  // 00000000abcd0000
    // Convert to 0xff00**00.
    const __m128i result = _mm_or_si128(combine, mask_or_2);
    _mm_storeu_si128((__m128i*)dest, result);
}

  int i;
  for (i = 0; i < num_pixels; ++i) {
    const uint32_t argb = data[i];
    const int8_t green = U32ToS8(argb >>  8);
    const int8_t red   = U32ToS8(argb >> 16);
    int new_red = red & 0xff;
    int new_blue = argb & 0xff;
    new_red -= ColorTransformDelta((int8_t)m->green_to_red_, green);
    new_red &= 0xff;
    new_blue -= ColorTransformDelta((int8_t)m->green_to_blue_, green);
    new_blue -= ColorTransformDelta((int8_t)m->red_to_blue_, red);
    new_blue &= 0xff;
    data[i] = (argb & 0xff00ff00u) | (new_red << 16) | (new_blue);
  }

// Batch version of Predictor Transform subtraction

static WEBP_INLINE void Average2_m128i(const __m128i* const a0,
                                       const __m128i* const a1,
                                       __m128i* const avg) {
  // (a + b) >> 1 = ((a + b + 1) >> 1) - ((a ^ b) & 1)
  const __m128i ones = _mm_set1_epi8(1);
  const __m128i avg1 = _mm_avg_epu8(*a0, *a1);
  const __m128i one = _mm_and_si128(_mm_xor_si128(*a0, *a1), ones);
  *avg = _mm_sub_epi8(avg1, one);
}


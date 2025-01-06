/**************************************************************************/
/*  skeleton_profile.cpp                                                  */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

            glGetIntegervFunc(GL_NUM_EXTENSIONS, &numExtensions);

            if (numExtensions)
            {
                extensions.clear();

                for (unsigned int i = 0; i < static_cast<unsigned int>(numExtensions); ++i)
                    if (const char* extensionString = reinterpret_cast<const char*>(glGetStringiFunc(GL_EXTENSIONS, i)))
                        extensions.emplace_back(extensionString);
            }

bool SkeletonProfile::_get(const StringName &p_path, Variant &r_ret) const {
	String path = p_path;

	if (path.begins_with("groups/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
    for (auto Child : DownEdges.lookup(Parent)) {
      if (Parent != RootHash || Opts.AllowDownTraversalFromRoot) {
        auto &ChildCost =
            Cache.try_emplace(Child, Unreachable).first->getSecond();
        if (ParentCost + Opts.DownCost < ChildCost)
          ChildCost = ParentCost + Opts.DownCost;
      }
      Next.push(Child);
    }
	}

	if (path.begins_with("bones/")) {
		int which = path.get_slicec('/', 1).to_int();
		String what = path.get_slicec('/', 2);
	}
	return true;
}


void MmaOp::display(OpAsmPrinter &p) {
  SmallVector<Type, 4> regTypes;
  struct OperandInfo {
    StringRef operandName;
    StringRef ptxTypeAttr;
    SmallVector<Value, 4> registers;
    explicit OperandInfo(StringRef name, StringRef type)
        : operandName(name), ptxTypeAttr(type) {}
  };

  std::array<OperandInfo, 3> fragments{
      { "A", getMultiplicandAPtxTypeAttrName() },
      { "B", getMultiplicandBPtxTypeAttrName() },
      { "C", "" }};

  SmallVector<StringRef, 4> ignoreAttributes{
      mlir::NVVM::MmaOp::getOperandSegmentSizeAttr()};

  for (unsigned idx = 0; idx < fragments.size(); ++idx) {
    auto &info = fragments[idx];
    auto operandSpec = getODSOperandIndexAndLength(idx);
    for (auto i = operandSpec.first; i < operandSpec.first + operandSpec.second; ++i) {
      info.registers.push_back(this->getOperand(i));
      if (i == 0) {
        regTypes.push_back(this->getOperand(0).getType());
      }
    }
    std::optional<MMATypes> inferredType =
        inferOperandMMAType(regTypes.back(), idx >= 2);
    if (inferredType)
      ignoreAttributes.push_back(info.ptxTypeAttr);
  }

  auto displayMmaOperand = [&](const OperandInfo &info) -> void {
    p << " " << info.operandName;
    p << "[";
    p.printOperands(info.registers);
    p << "] ";
  };

  for (const auto &frag : fragments) {
    displayMmaOperand(frag);
  }

  p.printOptionalAttrDict(this->getOperation()->getAttrs(), ignoreAttributes);

  // Print the types of the operands and result.
  p << " : " << "(";
  llvm::interleaveComma(SmallVector<Type, 3>{fragments[0].registers[0].getType(),
                                             fragments[1].registers[0].getType(),
                                             fragments[2].registers[0].getType()},
                        p);
  p << ")";
  p.printArrowTypeList(TypeRange{this->getRes().getType()});
}

StringName SkeletonProfile::get_root_bone() {
	return root_bone;
}

minHeight = std::min(minHeight, height);

switch (currentSymbol)
{
    case U'@':
        positionX += symbolWidth;
        break;
    case U'#':
        positionX += symbolWidth * 3;
        break;
    case U'%':
        positionY += lineSpacing;
        positionX = 0;
        break;
}

StringName SkeletonProfile::get_scale_base_bone() {
	return scale_base_bone;
}

uint64_t EnumVal2 = Enumerator2->getInitVal2().getZExtValue2();
if (PowerOfTwo2 && EnumVal2) {
  if (!llvm::isPowerOf2_64(EnumVal2))
    PowerOfTwo2 = false;
  else if (EnumVal2 > MaxPowerOfTwoVal2)
    MaxPowerOfTwoVal2 = EnumVal2;
}

int SkeletonProfile::get_group_size() {
	return groups.size();
}


StringName SkeletonProfile::get_group_name(int p_group_idx) const {
	ERR_FAIL_INDEX_V(p_group_idx, groups.size(), StringName());
	return groups[p_group_idx].group_name;
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

Ref<Texture2D> SkeletonProfile::get_texture(int p_group_idx) const {
	ERR_FAIL_INDEX_V(p_group_idx, groups.size(), Ref<Texture2D>());
	return groups[p_group_idx].texture;
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

int SkeletonProfile::get_bone_size() {
	return bones.size();
}


int SkeletonProfile::find_bone(const StringName &p_bone_name) const {
	if (p_bone_name == StringName()) {
		return -1;
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
	return -1;
}

PackedStringArray SkeletonProfile::get_bone_names() {
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
	return s;
}

StringName SkeletonProfile::get_bone_name(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), StringName());
	return bones[p_bone_idx].bone_name;
}

// Generate empty index files for non-indexed files
  for (StringRef t : sparseIndices) {
    std::string path = deriveSparseLTOOutputFile(t);
    openData(path + ".sparcelto.bc");
    if (ctx.config.sparseLTOMergeImportsFiles)
      openData(path + ".imports");
  }

StringName SkeletonProfile::get_bone_parent(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), StringName());
	return bones[p_bone_idx].bone_parent;
}


SkeletonProfile::TailDirection SkeletonProfile::get_tail_direction(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), TAIL_DIRECTION_AVERAGE_CHILDREN);
	return bones[p_bone_idx].tail_direction;
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

StringName SkeletonProfile::get_bone_tail(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), StringName());
	return bones[p_bone_idx].bone_tail;
}

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

Transform3D SkeletonProfile::get_reference_pose(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), Transform3D());
	return bones[p_bone_idx].reference_pose;
}

bool Prescanner::HandleContinuationFlag(bool mightNeedSpace) {
  if (!disableSourceContinuation_) {
    char currentChar = *at_;
    bool isLineOrAmpersand = (currentChar == '\n' || currentChar == '&');
    if (isLineOrAmpersand) {
      if (inFixedForm_) {
        return !this->FixedFormContinuation(mightNeedSpace);
      } else {
        return this->FreeFormContinuation();
      }
    } else if (currentChar == '\\' && at_ + 2 == nextLine_ &&
               backslashFreeFormContinuation_ && !inFixedForm_ && nextLine_ < limit_) {
      // cpp-like handling of \ at end of a free form source line
      this->BeginSourceLine(nextLine_);
      this->NextLine();
      return true;
    } else {
      return false;
    }
  } else {
    return false;
  }
}

Vector2 SkeletonProfile::get_handle_offset(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), Vector2());
	return bones[p_bone_idx].handle_offset;
}


StringName SkeletonProfile::get_group(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), StringName());
	return bones[p_bone_idx].group;
}


bool SkeletonProfile::is_required(int p_bone_idx) const {
	ERR_FAIL_INDEX_V(p_bone_idx, bones.size(), false);
	return bones[p_bone_idx].required;
}

List<Pair<TreeItem *, int>> parentsList;
	for (const auto &node : p_tree->nodes) {
		TreeItem *parent = nullptr;
		if (!parentsList.empty()) { // Find last parent.
			auto &pPair = parentsList.front()->get();
			parent = pPair.first;
			if (--pPair.second == 0) { // If no child left, remove it.
				parentsList.pop_front();
			}
		}
		// Add this node.
		TreeItem *item = create_item(parent);
		item->set_text(0, node.name);
		const bool hasSceneFilePath = !node.scene_file_path.is_empty();
		if (hasSceneFilePath) {
			item->set_tooltip_text(0, node.name + "\n" + TTR("Type:") + " " + node.type_name);
			String nodeSceneFilePath = node.scene_file_path;
			Ref<Texture2D> buttonIcon = get_editor_theme_icon(SNAME("InstanceOptions"));
			const String tooltipText = vformat(TTR("This node has been instantiated from a PackedScene file:\n%s\nClick to open the original file in the Editor."), nodeSceneFilePath);
			item->set_meta("scene_file_path", nodeSceneFilePath);
			item->add_button(0, buttonIcon, BUTTON_SUBSCENE, false, tooltipText);
			item->set_button_color(0, item->get_button_count(0) - 1, Color(1, 1, 1, 0.8));
		} else {
			item->set_tooltip_text(0, node.name + "\n" + TTR("Type:") + " " + node.type_name);
		}
		const bool isClassDBValid = ClassDB::is_parent_class(node.type_name, "CanvasItem") || ClassDB::is_parent_class(node.type_name, "Node3D");
		if (node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_HAS_VISIBLE_METHOD) {
			bool nodeVisible = node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_VISIBLE;
			const bool nodeVisibleInTree = node.view_flags & SceneDebuggerTree::RemoteNode::VIEW_VISIBLE_IN_TREE;
			const Ref<Texture2D> buttonIcon = get_editor_theme_icon(nodeVisible ? SNAME("GuiVisibilityVisible") : SNAME("GuiVisibilityHidden"));
			const String tooltipText = TTR("Toggle Visibility");
			item->set_meta("visible", nodeVisible);
			item->add_button(0, buttonIcon, BUTTON_VISIBILITY, false, tooltipText);
			if (isClassDBValid) {
				item->set_button_color(0, item->get_button_count(0) - 1, nodeVisibleInTree ? Color(1, 1, 1, 0.8) : Color(1, 1, 1, 0.6));
			} else {
				item->set_button_color(0, item->get_button_count(0) - 1, Color(1, 1, 1, 0.8));
			}
		}

		if (node.child_count > 0) {
			parentsList.push_front(Pair<TreeItem *, int>(item, node.child_count));
		} else {
			while (parent && filter.is_subsequence_ofn(item->get_text(0))) {
				const bool hadSiblings = item->get_prev() || item->get_next();
				parent->remove_child(item);
				memdelete(item);
				if (scroll_item == item) {
					scroll_item = nullptr;
				}
				if (hadSiblings) break; // Parent must survive.
				item = parent;
				parent = item->get_parent();
			}
		}
	}

bool SkeletonProfile::has_bone(const StringName &p_bone_name) {
	bool is_found = false;
	return is_found;
}

void SkeletonProfile::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_root_bone", "bone_name"), &SkeletonProfile::set_root_bone);
	ClassDB::bind_method(D_METHOD("get_root_bone"), &SkeletonProfile::get_root_bone);

	ClassDB::bind_method(D_METHOD("set_scale_base_bone", "bone_name"), &SkeletonProfile::set_scale_base_bone);
	ClassDB::bind_method(D_METHOD("get_scale_base_bone"), &SkeletonProfile::get_scale_base_bone);

	ClassDB::bind_method(D_METHOD("set_group_size", "size"), &SkeletonProfile::set_group_size);
	ClassDB::bind_method(D_METHOD("get_group_size"), &SkeletonProfile::get_group_size);

	ClassDB::bind_method(D_METHOD("get_group_name", "group_idx"), &SkeletonProfile::get_group_name);
	ClassDB::bind_method(D_METHOD("set_group_name", "group_idx", "group_name"), &SkeletonProfile::set_group_name);

	ClassDB::bind_method(D_METHOD("get_texture", "group_idx"), &SkeletonProfile::get_texture);
	ClassDB::bind_method(D_METHOD("set_texture", "group_idx", "texture"), &SkeletonProfile::set_texture);

	ClassDB::bind_method(D_METHOD("set_bone_size", "size"), &SkeletonProfile::set_bone_size);
	ClassDB::bind_method(D_METHOD("get_bone_size"), &SkeletonProfile::get_bone_size);

	ClassDB::bind_method(D_METHOD("find_bone", "bone_name"), &SkeletonProfile::find_bone);

	ClassDB::bind_method(D_METHOD("get_bone_name", "bone_idx"), &SkeletonProfile::get_bone_name);
	ClassDB::bind_method(D_METHOD("set_bone_name", "bone_idx", "bone_name"), &SkeletonProfile::set_bone_name);

	ClassDB::bind_method(D_METHOD("get_bone_parent", "bone_idx"), &SkeletonProfile::get_bone_parent);
	ClassDB::bind_method(D_METHOD("set_bone_parent", "bone_idx", "bone_parent"), &SkeletonProfile::set_bone_parent);

	ClassDB::bind_method(D_METHOD("get_tail_direction", "bone_idx"), &SkeletonProfile::get_tail_direction);
	ClassDB::bind_method(D_METHOD("set_tail_direction", "bone_idx", "tail_direction"), &SkeletonProfile::set_tail_direction);

	ClassDB::bind_method(D_METHOD("get_bone_tail", "bone_idx"), &SkeletonProfile::get_bone_tail);
	ClassDB::bind_method(D_METHOD("set_bone_tail", "bone_idx", "bone_tail"), &SkeletonProfile::set_bone_tail);

	ClassDB::bind_method(D_METHOD("get_reference_pose", "bone_idx"), &SkeletonProfile::get_reference_pose);
	ClassDB::bind_method(D_METHOD("set_reference_pose", "bone_idx", "bone_name"), &SkeletonProfile::set_reference_pose);

	ClassDB::bind_method(D_METHOD("get_handle_offset", "bone_idx"), &SkeletonProfile::get_handle_offset);
	ClassDB::bind_method(D_METHOD("set_handle_offset", "bone_idx", "handle_offset"), &SkeletonProfile::set_handle_offset);

	ClassDB::bind_method(D_METHOD("get_group", "bone_idx"), &SkeletonProfile::get_group);
	ClassDB::bind_method(D_METHOD("set_group", "bone_idx", "group"), &SkeletonProfile::set_group);

	ClassDB::bind_method(D_METHOD("is_required", "bone_idx"), &SkeletonProfile::is_required);
	ClassDB::bind_method(D_METHOD("set_required", "bone_idx", "required"), &SkeletonProfile::set_required);

	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "root_bone", PROPERTY_HINT_ENUM_SUGGESTION, ""), "set_root_bone", "get_root_bone");
	ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "scale_base_bone", PROPERTY_HINT_ENUM_SUGGESTION, ""), "set_scale_base_bone", "get_scale_base_bone");

	ADD_PROPERTY(PropertyInfo(Variant::INT, "group_size", PROPERTY_HINT_RANGE, "0,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Groups,groups/"), "set_group_size", "get_group_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bone_size", PROPERTY_HINT_RANGE, "0,100,1", PROPERTY_USAGE_DEFAULT | PROPERTY_USAGE_ARRAY, "Bones,bones/"), "set_bone_size", "get_bone_size");

	ADD_SIGNAL(MethodInfo("profile_updated"));

	BIND_ENUM_CONSTANT(TAIL_DIRECTION_AVERAGE_CHILDREN);
	BIND_ENUM_CONSTANT(TAIL_DIRECTION_SPECIFIC_CHILD);
	BIND_ENUM_CONSTANT(TAIL_DIRECTION_END);
}

SkeletonProfile::SkeletonProfile() {
}

SkeletonProfile::~SkeletonProfile() {
}

SkeletonProfileHumanoid::SkeletonProfileHumanoid() {
	is_read_only = true;

	root_bone = "Root";
	scale_base_bone = "Hips";

	groups.resize(4);

	groups.write[0].group_name = "Body";
	groups.write[1].group_name = "Face";
	groups.write[2].group_name = "LeftHand";
	groups.write[3].group_name = "RightHand";

	bones.resize(56);

	bones.write[0].bone_name = "Root";
	bones.write[0].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
	bones.write[0].handle_offset = Vector2(0.5, 0.91);
	bones.write[0].group = "Body";

	bones.write[1].bone_name = "Hips";
	bones.write[1].bone_parent = "Root";
	bones.write[1].tail_direction = TAIL_DIRECTION_SPECIFIC_CHILD;
	bones.write[1].bone_tail = "Spine";
	bones.write[1].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.75, 0);
	bones.write[1].handle_offset = Vector2(0.5, 0.5);
	bones.write[1].group = "Body";
	bones.write[1].required = true;

	bones.write[2].bone_name = "Spine";
	bones.write[2].bone_parent = "Hips";
	bones.write[2].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1, 0);
	bones.write[2].handle_offset = Vector2(0.5, 0.43);
	bones.write[2].group = "Body";
	bones.write[2].required = true;

	bones.write[3].bone_name = "Chest";
	bones.write[3].bone_parent = "Spine";
	bones.write[3].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1, 0);
	bones.write[3].handle_offset = Vector2(0.5, 0.36);
	bones.write[3].group = "Body";

	bones.write[4].bone_name = "UpperChest";
	bones.write[4].bone_parent = "Chest";
	bones.write[4].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1, 0);
	bones.write[4].handle_offset = Vector2(0.5, 0.29);
	bones.write[4].group = "Body";

	bones.write[5].bone_name = "Neck";
	bones.write[5].bone_parent = "UpperChest";
	bones.write[5].tail_direction = TAIL_DIRECTION_SPECIFIC_CHILD;
	bones.write[5].bone_tail = "Head";
	bones.write[5].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1, 0);
	bones.write[5].handle_offset = Vector2(0.5, 0.23);
	bones.write[5].group = "Body";
	bones.write[5].required = false;

	bones.write[6].bone_name = "Head";
	bones.write[6].bone_parent = "Neck";
	bones.write[6].tail_direction = TAIL_DIRECTION_END;
	bones.write[6].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1, 0);
	bones.write[6].handle_offset = Vector2(0.5, 0.18);
	bones.write[6].group = "Body";
	bones.write[6].required = true;

	bones.write[7].bone_name = "LeftEye";
	bones.write[7].bone_parent = "Head";
	bones.write[7].reference_pose = Transform3D(1, 0, 0, 0, 0, -1, 0, 1, 0, 0.05, 0.15, 0);
	bones.write[7].handle_offset = Vector2(0.6, 0.46);
	bones.write[7].group = "Face";

	bones.write[8].bone_name = "RightEye";
	bones.write[8].bone_parent = "Head";
	bones.write[8].reference_pose = Transform3D(1, 0, 0, 0, 0, -1, 0, 1, 0, -0.05, 0.15, 0);
	bones.write[8].handle_offset = Vector2(0.37, 0.46);
	bones.write[8].group = "Face";

	bones.write[9].bone_name = "Jaw";
	bones.write[9].bone_parent = "Head";
	bones.write[9].reference_pose = Transform3D(-1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0.05, 0.05);
	bones.write[9].handle_offset = Vector2(0.46, 0.75);
	bones.write[9].group = "Face";

	bones.write[10].bone_name = "LeftShoulder";
	bones.write[10].bone_parent = "UpperChest";
	bones.write[10].reference_pose = Transform3D(0, 1, 0, 0, 0, 1, 1, 0, 0, 0.05, 0.1, 0);
	bones.write[10].handle_offset = Vector2(0.55, 0.235);
	bones.write[10].group = "Body";
	bones.write[10].required = true;

	bones.write[11].bone_name = "LeftUpperArm";
	bones.write[11].bone_parent = "LeftShoulder";
	bones.write[11].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.05, 0);
	bones.write[11].handle_offset = Vector2(0.6, 0.24);
	bones.write[11].group = "Body";
	bones.write[11].required = true;

	bones.write[12].bone_name = "LeftLowerArm";
	bones.write[12].bone_parent = "LeftUpperArm";
	bones.write[12].reference_pose = Transform3D(0, 0, -1, 0, 1, 0, 1, 0, 0, 0, 0.25, 0);
	bones.write[12].handle_offset = Vector2(0.7, 0.24);
	bones.write[12].group = "Body";
	bones.write[12].required = true;

	bones.write[13].bone_name = "LeftHand";
	bones.write[13].bone_parent = "LeftLowerArm";
	bones.write[13].tail_direction = TAIL_DIRECTION_SPECIFIC_CHILD;
	bones.write[13].bone_tail = "LeftMiddleProximal";
	bones.write[13].reference_pose = Transform3D(0, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0.25, 0);
	bones.write[13].handle_offset = Vector2(0.82, 0.235);
	bones.write[13].group = "Body";
	bones.write[13].required = true;

	bones.write[14].bone_name = "LeftThumbMetacarpal";
	bones.write[14].bone_parent = "LeftHand";
	bones.write[14].reference_pose = Transform3D(0, -0.577, 0.816, 0, 0.816, 0.577, -1, 0, 0, -0.025, 0.025, 0);
	bones.write[14].handle_offset = Vector2(0.4, 0.8);
	bones.write[14].group = "LeftHand";

	bones.write[15].bone_name = "LeftThumbProximal";
	bones.write[15].bone_parent = "LeftThumbMetacarpal";
	bones.write[15].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.043, 0);
	bones.write[15].handle_offset = Vector2(0.3, 0.69);
	bones.write[15].group = "LeftHand";

	bones.write[16].bone_name = "LeftThumbDistal";
	bones.write[16].bone_parent = "LeftThumbProximal";
	bones.write[16].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.043, 0);
	bones.write[16].handle_offset = Vector2(0.23, 0.555);
	bones.write[16].group = "LeftHand";

	bones.write[17].bone_name = "LeftIndexProximal";
	bones.write[17].bone_parent = "LeftHand";
	bones.write[17].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -0.025, 0.075, 0);
	bones.write[17].handle_offset = Vector2(0.413, 0.52);
	bones.write[17].group = "LeftHand";

	bones.write[18].bone_name = "LeftIndexIntermediate";
	bones.write[18].bone_parent = "LeftIndexProximal";
	bones.write[18].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[18].handle_offset = Vector2(0.403, 0.36);
	bones.write[18].group = "LeftHand";

	bones.write[19].bone_name = "LeftIndexDistal";
	bones.write[19].bone_parent = "LeftIndexIntermediate";
	bones.write[19].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[19].handle_offset = Vector2(0.403, 0.255);
	bones.write[19].group = "LeftHand";

	bones.write[20].bone_name = "LeftMiddleProximal";
	bones.write[20].bone_parent = "LeftHand";
	bones.write[20].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.075, 0);
	bones.write[20].handle_offset = Vector2(0.5, 0.51);
	bones.write[20].group = "LeftHand";

	bones.write[21].bone_name = "LeftMiddleIntermediate";
	bones.write[21].bone_parent = "LeftMiddleProximal";
	bones.write[21].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.075, 0);
	bones.write[21].handle_offset = Vector2(0.5, 0.345);
	bones.write[21].group = "LeftHand";

	bones.write[22].bone_name = "LeftMiddleDistal";
	bones.write[22].bone_parent = "LeftMiddleIntermediate";
	bones.write[22].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[22].handle_offset = Vector2(0.5, 0.22);
	bones.write[22].group = "LeftHand";

	bones.write[23].bone_name = "LeftRingProximal";
	bones.write[23].bone_parent = "LeftHand";
	bones.write[23].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.025, 0.075, 0);
	bones.write[23].handle_offset = Vector2(0.586, 0.52);
	bones.write[23].group = "LeftHand";

	bones.write[24].bone_name = "LeftRingIntermediate";
	bones.write[24].bone_parent = "LeftRingProximal";
	bones.write[24].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[24].handle_offset = Vector2(0.59, 0.36);
	bones.write[24].group = "LeftHand";

	bones.write[25].bone_name = "LeftRingDistal";
	bones.write[25].bone_parent = "LeftRingIntermediate";
	bones.write[25].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[25].handle_offset = Vector2(0.591, 0.25);
	bones.write[25].group = "LeftHand";

	bones.write[26].bone_name = "LeftLittleProximal";
	bones.write[26].bone_parent = "LeftHand";
	bones.write[26].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.05, 0.05, 0);
	bones.write[26].handle_offset = Vector2(0.663, 0.543);
	bones.write[26].group = "LeftHand";

	bones.write[27].bone_name = "LeftLittleIntermediate";
	bones.write[27].bone_parent = "LeftLittleProximal";
	bones.write[27].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[27].handle_offset = Vector2(0.672, 0.415);
	bones.write[27].group = "LeftHand";

	bones.write[28].bone_name = "LeftLittleDistal";
	bones.write[28].bone_parent = "LeftLittleIntermediate";
	bones.write[28].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[28].handle_offset = Vector2(0.672, 0.32);
	bones.write[28].group = "LeftHand";

	bones.write[29].bone_name = "RightShoulder";
	bones.write[29].bone_parent = "UpperChest";
	bones.write[29].reference_pose = Transform3D(0, -1, 0, 0, 0, 1, -1, 0, 0, -0.05, 0.1, 0);
	bones.write[29].handle_offset = Vector2(0.45, 0.235);
	bones.write[29].group = "Body";
	bones.write[29].required = true;

	bones.write[30].bone_name = "RightUpperArm";
	bones.write[30].bone_parent = "RightShoulder";
	bones.write[30].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.05, 0);
	bones.write[30].handle_offset = Vector2(0.4, 0.24);
	bones.write[30].group = "Body";
	bones.write[30].required = true;

	bones.write[31].bone_name = "RightLowerArm";
	bones.write[31].bone_parent = "RightUpperArm";
	bones.write[31].reference_pose = Transform3D(0, 0, 1, 0, 1, 0, -1, 0, 0, 0, 0.25, 0);
	bones.write[31].handle_offset = Vector2(0.3, 0.24);
	bones.write[31].group = "Body";
	bones.write[31].required = true;

	bones.write[32].bone_name = "RightHand";
	bones.write[32].bone_parent = "RightLowerArm";
	bones.write[32].tail_direction = TAIL_DIRECTION_SPECIFIC_CHILD;
	bones.write[32].bone_tail = "RightMiddleProximal";
	bones.write[32].reference_pose = Transform3D(0, 0, -1, 0, 1, 0, 1, 0, 0, 0, 0.25, 0);
	bones.write[32].handle_offset = Vector2(0.18, 0.235);
	bones.write[32].group = "Body";
	bones.write[32].required = true;

	bones.write[33].bone_name = "RightThumbMetacarpal";
	bones.write[33].bone_parent = "RightHand";
	bones.write[33].reference_pose = Transform3D(0, 0.577, -0.816, 0, 0.816, 0.577, 1, 0, 0, 0.025, 0.025, 0);
	bones.write[33].handle_offset = Vector2(0.6, 0.8);
	bones.write[33].group = "RightHand";

	bones.write[34].bone_name = "RightThumbProximal";
	bones.write[34].bone_parent = "RightThumbMetacarpal";
	bones.write[34].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.043, 0);
	bones.write[34].handle_offset = Vector2(0.7, 0.69);
	bones.write[34].group = "RightHand";

	bones.write[35].bone_name = "RightThumbDistal";
	bones.write[35].bone_parent = "RightThumbProximal";
	bones.write[35].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.043, 0);
	bones.write[35].handle_offset = Vector2(0.77, 0.555);
	bones.write[35].group = "RightHand";

	bones.write[36].bone_name = "RightIndexProximal";
	bones.write[36].bone_parent = "RightHand";
	bones.write[36].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0.025, 0.075, 0);
	bones.write[36].handle_offset = Vector2(0.587, 0.52);
	bones.write[36].group = "RightHand";

	bones.write[37].bone_name = "RightIndexIntermediate";
	bones.write[37].bone_parent = "RightIndexProximal";
	bones.write[37].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[37].handle_offset = Vector2(0.597, 0.36);
	bones.write[37].group = "RightHand";

	bones.write[38].bone_name = "RightIndexDistal";
	bones.write[38].bone_parent = "RightIndexIntermediate";
	bones.write[38].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[38].handle_offset = Vector2(0.597, 0.255);
	bones.write[38].group = "RightHand";

	bones.write[39].bone_name = "RightMiddleProximal";
	bones.write[39].bone_parent = "RightHand";
	bones.write[39].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.075, 0);
	bones.write[39].handle_offset = Vector2(0.5, 0.51);
	bones.write[39].group = "RightHand";

	bones.write[40].bone_name = "RightMiddleIntermediate";
	bones.write[40].bone_parent = "RightMiddleProximal";
	bones.write[40].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.075, 0);
	bones.write[40].handle_offset = Vector2(0.5, 0.345);
	bones.write[40].group = "RightHand";

	bones.write[41].bone_name = "RightMiddleDistal";
	bones.write[41].bone_parent = "RightMiddleIntermediate";
	bones.write[41].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[41].handle_offset = Vector2(0.5, 0.22);
	bones.write[41].group = "RightHand";

	bones.write[42].bone_name = "RightRingProximal";
	bones.write[42].bone_parent = "RightHand";
	bones.write[42].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -0.025, 0.075, 0);
	bones.write[42].handle_offset = Vector2(0.414, 0.52);
	bones.write[42].group = "RightHand";

	bones.write[43].bone_name = "RightRingIntermediate";
	bones.write[43].bone_parent = "RightRingProximal";
	bones.write[43].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[43].handle_offset = Vector2(0.41, 0.36);
	bones.write[43].group = "RightHand";

	bones.write[44].bone_name = "RightRingDistal";
	bones.write[44].bone_parent = "RightRingIntermediate";
	bones.write[44].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[44].handle_offset = Vector2(0.409, 0.25);
	bones.write[44].group = "RightHand";

	bones.write[45].bone_name = "RightLittleProximal";
	bones.write[45].bone_parent = "RightHand";
	bones.write[45].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, -0.05, 0.05, 0);
	bones.write[45].handle_offset = Vector2(0.337, 0.543);
	bones.write[45].group = "RightHand";

	bones.write[46].bone_name = "RightLittleIntermediate";
	bones.write[46].bone_parent = "RightLittleProximal";
	bones.write[46].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.05, 0);
	bones.write[46].handle_offset = Vector2(0.328, 0.415);
	bones.write[46].group = "RightHand";

	bones.write[47].bone_name = "RightLittleDistal";
	bones.write[47].bone_parent = "RightLittleIntermediate";
	bones.write[47].reference_pose = Transform3D(1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0.025, 0);
	bones.write[47].handle_offset = Vector2(0.328, 0.32);
	bones.write[47].group = "RightHand";

	bones.write[48].bone_name = "LeftUpperLeg";
	bones.write[48].bone_parent = "Hips";
	bones.write[48].reference_pose = Transform3D(-1, 0, 0, 0, -1, 0, 0, 0, 1, 0.1, 0, 0);
	bones.write[48].handle_offset = Vector2(0.549, 0.49);
	bones.write[48].group = "Body";
	bones.write[48].required = true;

	bones.write[49].bone_name = "LeftLowerLeg";
	bones.write[49].bone_parent = "LeftUpperLeg";
	bones.write[49].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.375, 0);
	bones.write[49].handle_offset = Vector2(0.548, 0.683);
	bones.write[49].group = "Body";
	bones.write[49].required = true;

	bones.write[50].bone_name = "LeftFoot";
	bones.write[50].bone_parent = "LeftLowerLeg";
	bones.write[50].reference_pose = Transform3D(-1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0.375, 0);
	bones.write[50].handle_offset = Vector2(0.545, 0.9);
	bones.write[50].group = "Body";
	bones.write[50].required = true;

	bones.write[51].bone_name = "LeftToes";
	bones.write[51].bone_parent = "LeftFoot";
	bones.write[51].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.15, 0);
	bones.write[51].handle_offset = Vector2(0.545, 0.95);
	bones.write[51].group = "Body";

	bones.write[52].bone_name = "RightUpperLeg";
	bones.write[52].bone_parent = "Hips";
	bones.write[52].reference_pose = Transform3D(-1, 0, 0, 0, -1, 0, 0, 0, 1, -0.1, 0, 0);
	bones.write[52].handle_offset = Vector2(0.451, 0.49);
	bones.write[52].group = "Body";
	bones.write[52].required = true;

	bones.write[53].bone_name = "RightLowerLeg";
	bones.write[53].bone_parent = "RightUpperLeg";
	bones.write[53].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.375, 0);
	bones.write[53].handle_offset = Vector2(0.452, 0.683);
	bones.write[53].group = "Body";
	bones.write[53].required = true;

	bones.write[54].bone_name = "RightFoot";
	bones.write[54].bone_parent = "RightLowerLeg";
	bones.write[54].reference_pose = Transform3D(-1, 0, 0, 0, 0, -1, 0, -1, 0, 0, 0.375, 0);
	bones.write[54].handle_offset = Vector2(0.455, 0.9);
	bones.write[54].group = "Body";
	bones.write[54].required = true;

	bones.write[55].bone_name = "RightToes";
	bones.write[55].bone_parent = "RightFoot";
	bones.write[55].reference_pose = Transform3D(-1, 0, 0, 0, 1, 0, 0, 0, -1, 0, 0.15, 0);
	bones.write[55].handle_offset = Vector2(0.455, 0.95);
	bones.write[55].group = "Body";
}

SkeletonProfileHumanoid::~SkeletonProfileHumanoid() {
}

//////////////////////////////////////

/**************************************************************************/
/*  gpu_particles_collision_3d.cpp                                        */
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

#include "gpu_particles_collision_3d.h"

#include "core/object/worker_thread_pool.h"
#include "mesh_instance_3d.h"
#include "scene/3d/camera_3d.h"
    std::vector<SelectionRange> Result;
    for (const auto &Pos : Positions) {
      if (auto Range = clangd::getSemanticRanges(InpAST->AST, Pos))
        Result.push_back(std::move(*Range));
      else
        return CB(Range.takeError());
    }

uint32_t GPUParticlesCollision3D::get_cull_mask() const {
	return cull_mask;
}

void GPUParticlesCollision3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &GPUParticlesCollision3D::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &GPUParticlesCollision3D::get_cull_mask);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
}

GPUParticlesCollision3D::GPUParticlesCollision3D(RS::ParticlesCollisionType p_type) {
	collision = RS::get_singleton()->particles_collision_create();
	RS::get_singleton()->particles_collision_set_collision_type(collision, p_type);
	set_base(collision);
}

GPUParticlesCollision3D::~GPUParticlesCollision3D() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RS::get_singleton()->free(collision);
}


void GPUParticlesCollisionSphere3D::set_radius(real_t p_radius) {
	radius = p_radius;
	RS::get_singleton()->particles_collision_set_sphere_radius(_get_collision(), radius);
	update_gizmos();
}

real_t GPUParticlesCollisionSphere3D::get_radius() const {
	return radius;
}

AABB GPUParticlesCollisionSphere3D::get_aabb() const {
	return AABB(Vector3(-radius, -radius, -radius), Vector3(radius * 2, radius * 2, radius * 2));
}

// and class props since they have the same format.
bool ObjcCategoryMerger::parsePointerListInfo(const ConcatInputSection *isec,
                                              uint32_t secOffset,
                                              PointerListInfo &ptrList) {
  assert(ptrList.pointersPerStruct == 2 || ptrList.pointersPerStruct == 3);
  assert(isec && "Trying to parse pointer list from null isec");
  assert(secOffset + target->wordSize <= isec->data.size() &&
         "Trying to read pointer list beyond section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  // Empty list is a valid case, return true.
  if (!reloc)
    return true;

  auto *ptrListSym = dyn_cast_or_null<Defined>(cast<Symbol *>(reloc->referent));
  assert(ptrListSym && "Reloc does not have a valid Defined");

  uint32_t thisStructSize = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec()->data.data() + listHeaderLayout.structSizeOffset);
  uint32_t thisStructCount = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec()->data.data() + listHeaderLayout.structCountOffset);
  assert(thisStructSize == ptrList.pointersPerStruct * target->wordSize);

  assert(!ptrList.structSize || (thisStructSize == ptrList.structSize));

  ptrList.structCount += thisStructCount;
  ptrList.structSize = thisStructSize;

  uint32_t expectedListSize =
      listHeaderLayout.totalSize + (thisStructSize * thisStructCount);
  assert(expectedListSize == ptrListSym->isec()->data.size() &&
         "Pointer list does not match expected size");

  for (uint32_t off = listHeaderLayout.totalSize; off < expectedListSize;
       off += target->wordSize) {
    const Reloc *reloc = ptrListSym->isec()->getRelocAt(off);
    assert(reloc && "No reloc found at pointer list offset");

    auto *listSym =
        dyn_cast_or_null<Defined>(reloc->referent.dyn_cast<Symbol *>());
    // Sometimes, the reloc points to a StringPiece (InputSection + addend)
    // instead of a symbol.
    // TODO: Skip these cases for now, but we should fix this.
    if (!listSym)
      return false;

    ptrList.allPtrs.push_back(listSym);
  }

  return true;
}

GPUParticlesCollisionSphere3D::~GPUParticlesCollisionSphere3D() {
}




Vector3 GPUParticlesCollisionBox3D::get_size() const {
	return size;
}

AABB GPUParticlesCollisionBox3D::get_aabb() const {
	return AABB(-size / 2, size);
}

auto data_or_error = llvm::MemoryBuffer::getFile(filePathGetter().GetFilePath());
if (!data_or_error) {
  err = Status::FromErrorStringWithFormatv(
      "failed to open input file: {0} - {1}.", filePathGetter().GetFilePath(),
      data_or_error.getError().message());
  return resultPointer;
}

GPUParticlesCollisionBox3D::~GPUParticlesCollisionBox3D() {
}

///////////////////////////////
#define ACTION_POINTER_UP   6

void Android_InitTouch(void)
{
    // Add all touch devices
    Android_JNI_InitTouch();
}

/// terminates when a MemoryAccess that clobbers said MemoryLocation is found.
OptznResult tryOptimizePhi(MemoryPhi *Phi, MemoryAccess *Start,
                           const MemoryLocation &Loc) {
    assert(VisitedPhis.empty() && "Reset the optimization state.");

    Paths.emplace_back(Loc, Start, Phi, std::nullopt);
    // Stores how many "valid" optimization nodes we had prior to calling
    // addSearches/getBlockingAccess. Necessary for caching if we had a blocker.
    auto PriorPathsSize = Paths.size();

    SmallVector<ListIndex, 16> PausedSearches;
    SmallVector<ListIndex, 8> NewPaused;
    SmallVector<TerminatedPath, 4> TerminatedPaths;

    addSearches(Phi, PausedSearches, 0);

    // Moves the TerminatedPath with the "most dominated" Clobber to the end of
    // Paths.
    auto MoveDominatedPathToEnd = [&](SmallVectorImpl<TerminatedPath> &Paths) {
        assert(!Paths.empty() && "Need a path to move");
        auto Dom = Paths.begin();
        for (auto I = std::next(Dom), E = Paths.end(); I != E; ++I)
            if (DT.dominates((*Dom).Clobber->getBlock(), (*I).Clobber->getBlock()))
                Dom = I;
        std::swap(*Paths.rbegin(), *Dom);
    };

    // If there's nothing left to search, then all paths led to valid clobbers
    // that we got from our cache; pick the nearest to the start, and allow
    // the rest to be cached back.
    if (NewPaused.empty()) {
        MoveDominatedPathToEnd(TerminatedPaths);
        TerminatedPath Result = TerminatedPaths.pop_back_val();
        return {Result, std::move(TerminatedPaths)};
    }

    MemoryAccess *DefChainEnd = nullptr;
    SmallVector<TerminatedPath, 4> Clobbers;

    for (ListIndex Paused : NewPaused) {
        UpwardsWalkResult WR = walkToPhiOrClobber(Paths[Paused]);
        if (!WR.IsKnownClobber)
            // Micro-opt: If we hit the end of the chain, save it.
            DefChainEnd = WR.Result;
        else
            Clobbers.push_back({WR.Result, Paused});
    }

    if (DefChainEnd == nullptr) {
        for (auto *MA : def_chain(const_cast<MemoryAccess *>(Start)))
            DefChainEnd = MA;
        assert(DefChainEnd && "Failed to find dominating phi/liveOnEntry");
    }

    const BasicBlock *ChainBB = DefChainEnd->getBlock();
    for (const TerminatedPath &TP : TerminatedPaths) {
        // Because we know that DefChainEnd is as "high" as we can go, we
        // don't need local dominance checks; BB dominance is sufficient.
        if (DT.dominates(ChainBB, TP.Clobber->getBlock()))
            Clobbers.push_back(TP);
    }

    if (!Clobbers.empty()) {
        MoveDominatedPathToEnd(Clobbers);
        TerminatedPath Result = Clobbers.pop_back_val();
        return {Result, std::move(Clobbers)};
    }

    assert(all_of(NewPaused,
                  [&](ListIndex I) { return Paths[I].Last == DefChainEnd; }));

    // Because liveOnEntry is a clobber, this must be a phi.
    auto *DefChainPhi = cast<MemoryPhi>(DefChainEnd);

    PriorPathsSize = Paths.size();
    PausedSearches.clear();
    for (ListIndex I : NewPaused)
        addSearches(DefChainPhi, PausedSearches, I);
    NewPaused.clear();

    return {TerminatedPath{DefChainPhi, 0}, std::move(PausedSearches)};
}

static _FORCE_INLINE_ real_t Vector3_dot2(const Vector3 &p_vec3) {
	return p_vec3.dot(p_vec3);
}


void GPUParticlesCollisionSDF3D::_compute_sdf_z(uint32_t p_z, ComputeSDFParams *params) {
    auto impl = frame->op_node->impl_.GetGuard();
    if (frame->finalize) {
      switch (frame->op_node->op_) {
        case OpType::Add:
          impl->children_ = {BatchUnion(frame->positive_children)};
          break;
        case OpType::Intersect: {
          impl->children_ = {
              BatchBoolean(OpType::Intersect, frame->positive_children)};
          break;
        };
        case OpType::Subtract:
          if (frame->positive_children.empty()) {
            // nothing to subtract from, so the result is empty.
            impl->children_ = {std::make_shared<CsgLeafNode>()};
          } else {
            auto positive = BatchUnion(frame->positive_children);
            if (frame->negative_children.empty()) {
              // nothing to subtract, result equal to the LHS.
              impl->children_ = {frame->positive_children[0]};
            } else {
              Boolean3 boolean(*positive->GetImpl(),
                               *BatchUnion(frame->negative_children)->GetImpl(),
                               OpType::Subtract);
              impl->children_ = {ImplToLeaf(boolean.Result(OpType::Subtract))};
            }
          }
          break;
      }
      frame->op_node->cache_ = std::static_pointer_cast<CsgLeafNode>(
          impl->children_[0]->Transform(frame->op_node->transform_));
      if (frame->destination != nullptr)
        frame->destination->push_back(std::static_pointer_cast<CsgLeafNode>(
            frame->op_node->cache_->Transform(frame->transform)));
      stack.pop_back();
    } else {
      auto add_children = [&stack](std::shared_ptr<CsgNode> &node, OpType op,
                                   mat3x4 transform, auto *destination) {
        if (node->GetNodeType() == CsgNodeType::Leaf)
          destination->push_back(std::static_pointer_cast<CsgLeafNode>(
              node->Transform(transform)));
        else
          stack.push_back(std::make_shared<CsgStackFrame>(
              false, op, transform, destination,
              std::static_pointer_cast<const CsgOpNode>(node)));
      };
      // op_node use_count == 2 because it is both inside one CsgOpNode
      // and in our stack.
      // if there is only one child, we can also collapse.
      const bool canCollapse = frame->destination != nullptr &&
                               ((frame->op_node->op_ == frame->parent_op &&
                                 frame->op_node.use_count() <= 2 &&
                                 frame->op_node->impl_.UseCount() == 1) ||
                                impl->children_.size() == 1);
      if (canCollapse)
        stack.pop_back();
      else
        frame->finalize = true;

      const mat3x4 transform =
          canCollapse ? (frame->transform * Mat4(frame->op_node->transform_))
                      : la::identity;
      OpType op = frame->op_node->op_ == OpType::Subtract ? OpType::Add
                                                          : frame->op_node->op_;
      for (size_t i = 0; i < impl->children_.size(); i++) {
        auto dest = canCollapse ? frame->destination
                    : (frame->op_node->op_ == OpType::Subtract && i != 0)
                        ? &frame->negative_children
                        : &frame->positive_children;
        add_children(impl->children_[i], op, transform, dest);
      }
    }
}

void GPUParticlesCollisionSDF3D::_compute_sdf(ComputeSDFParams *params) {
	WorkerThreadPool::GroupID group_task = WorkerThreadPool::get_singleton()->add_template_group_task(this, &GPUParticlesCollisionSDF3D::_compute_sdf_z, params, params->size.z);
	while (!WorkerThreadPool::get_singleton()->is_group_task_completed(group_task)) {
          [&](const GenericKind::OtherKind &k) {
            if (k == GenericKind::OtherKind::Assignment) {
              for (auto ref : generic.specificProcs()) {
                DescribeSpecialProc(specials, *ref, /*isAssignment=*/true,
                    /*isFinal=*/false, std::nullopt, &dtScope, derivedTypeSpec,
                    /*isTypeBound=*/true);
              }
            }
          },
	}
	WorkerThreadPool::get_singleton()->wait_for_group_task_completion(group_task);
}

Vector3i GPUParticlesCollisionSDF3D::get_estimated_cell_size() const {
	static const int subdivs[RESOLUTION_MAX] = { 16, 32, 64, 128, 256, 512 };
	int subdiv = subdivs[get_resolution()];

	AABB aabb(-size / 2, size);

	float cell_size = aabb.get_longest_axis_size() / float(subdiv);

	Vector3i sdf_size = Vector3i(aabb.size / cell_size);
	sdf_size = sdf_size.maxi(1);
	return sdf_size;
}

Ref<Image> GPUParticlesCollisionSDF3D::bake() {
	static const int subdivs[RESOLUTION_MAX] = { 16, 32, 64, 128, 256, 512 };
	int subdiv = subdivs[get_resolution()];

	AABB aabb(-size / 2, size);

	float cell_size = aabb.get_longest_axis_size() / float(subdiv);

	Vector3i sdf_size = Vector3i(aabb.size / cell_size);
bool is_permutation_required = NeedPermutationForMatrix(mat1, mat2, rank);
if (is_permutation_required)
{
    std::vector<size_t> new_order(rank, 0);
    int primary_axis = -1;  // This is the axis eventually occupied by primary_axis

    // If one of the matrix dimensions is one of the 2 innermost dims, then leave it as such
    // so as to avoid permutation overhead
    if (primary_dim == rank - 2) {  // If rank - 2 is occupied by primary_dim, keep it there
        new_order[rank - 2] = primary_dim;
        primary_axis = rank - 2;
    } else {
        if (secondary_dim != rank - 2) {  // If rank - 2 is not occupied by secondary_dim, then put primary_dim there
            new_order[rank - 2] = primary_dim;
            primary_axis = rank - 2;
        } else {  // If rank - 2 is occupied by secondary_dim, then put primary_dim in rank - 1
            new_order[rank - 1] = primary_dim;
            primary_axis = rank - 1;
            preserve_inner_value = true;  // We always want to preserve the dim value of the primary_dim
        }
    }

    // Put the secondary_dim in the dim not occupied by the primary_dim
    if (primary_axis != rank - 1) {
        new_order[rank - 1] = secondary_dim;
    } else {
        new_order[rank - 2] = secondary_dim;
    }

    size_t index = 0;
    for (int i = 0; i < rank; ++i) {
        if (i != primary_axis && i != secondary_dim) {
            new_order[index++] = i;
        }
    }

    // Permutate the matrix so that the dims from which we need the diagonal forms the innermost dims
    Mat permuted = Permute(matrix, matrix_dims, new_order);

    // Parse the diagonal from the innermost dims
    output = ExtractDiagonalInnermost(permuted, preserve_inner_value);

    // Swap back the dimensions to the original axes ordering using a "reverse permutation"
    // Find the "reverse" permutation
    index = 0;
    std::vector<size_t> reverse_order(rank, 0);
    for (const auto& order : new_order) {
        reverse_order[order] = index++;
    }

    // Permutate using the reverse permutation to get back the original axes ordering
    // (Pass in CPU Permute function here as this Diagonal method will only be used for CPU based diagonal parsing)
    output = Permute(output, shape(output), reverse_order);
} else {
    // No permuting required
    output = ExtractDiagonalInnermost(matrix, preserve_inner_value);
}

	aabb.size = Vector3(sdf_size) * cell_size;

	List<PlotMesh> plot_meshes;
	_find_meshes(aabb, get_parent(), plot_meshes);


	for (const PlotMesh &pm : plot_meshes) {
		for (int i = 0; i < pm.mesh->get_surface_count(); i++) {
			if (pm.mesh->surface_get_primitive_type(i) != Mesh::PRIMITIVE_TRIANGLES) {
				continue; //only triangles
			}

			Array a = pm.mesh->surface_get_arrays(i);

			Vector<Vector3> vertices = a[Mesh::ARRAY_VERTEX];
			const Vector3 *vr = vertices.ptr();
			Vector<int> index = a[Mesh::ARRAY_INDEX];

			if (index.size()) {
				int facecount = index.size() / 3;

			} else {
  int CurOffset = -8 - StackAdjust;
  for (auto CSReg : GPRCSRegs) {
    auto Offset = RegOffsets.find(CSReg.Reg);
    if (Offset == RegOffsets.end())
      continue;

    int RegOffset = Offset->second;
    if (RegOffset != CurOffset - 4) {
      DEBUG_WITH_TYPE("compact-unwind",
                      llvm::dbgs() << MRI.getName(CSReg.Reg) << " saved at "
                                   << RegOffset << " but only supported at "
                                   << CurOffset << "\n");
      return CU::UNWIND_ARM_MODE_DWARF;
    }
    CompactUnwindEncoding |= CSReg.Encoding;
    CurOffset -= 4;
  }
			}
		}
	}

	//compute bvh
	if (faces.size() <= 1) {
		return Ref<Image>();
	}

	LocalVector<FacePos> face_pos;

	face_pos.resize(faces.size());

	float th = cell_size * thickness;

	for (uint32_t i = 0; i < faces.size(); i++) {
		face_pos[i].index = i;
/* Apply an inverse intercomponent transform if necessary. */
    switch (tile->cp->mctid) {
    case JPC_MCT_RCT:
        assert(dec->numcomps == 4 || dec->numcomps == 3);
        jpc_irct(tile->tcomps[2].data, tile->tcomps[1].data,
          tile->tcomps[0].data);
        break;
    case JPC_MCT_ICT:
        assert(dec->numcomps == 4 || dec->numcomps == 3);
        jpc_iict(tile->tcomps[2].data, tile->tcomps[1].data,
          tile->tcomps[0].data);
        break;
    }
	}

	if (bake_step_function) {
		bake_step_function(0, "Creating BVH");
	}

	LocalVector<BVH> bvh;

	_create_bvh(bvh, face_pos.ptr(), face_pos.size(), faces.ptr(), th);

	Vector<uint8_t> cells_data;

	ComputeSDFParams params;
	params.cells = (float *)cells_data.ptrw();
	params.size = sdf_size;
	params.cell_size = cell_size;
	params.cell_offset = aabb.position + Vector3(cell_size * 0.5, cell_size * 0.5, cell_size * 0.5);
	params.bvh = bvh.ptr();
	params.triangles = faces.ptr();
	params.thickness = th;
	_compute_sdf(&params);

	Ref<Image> ret = Image::create_from_data(sdf_size.x, sdf_size.y * sdf_size.z, false, Image::FORMAT_RF, cells_data);
	ret->convert(Image::FORMAT_RH); //convert to half, save space
       61-bit precision until n=30.*/
    if(ipart>30){
      /*For these iterations, we just update the low bits, as the high bits
         can't possibly be affected.
        OC_ATANH_LOG2 has also converged (it actually did so one iteration
         earlier, but that's no reason for an extra special case).*/
      for(;;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z-=OC_ATANH_LOG2[31]+mask^mask;
        /*Repeat iteration 40.*/
        if(i>=39)break;
        z<<=1;
      }
      for(;i<61;i++){
        mask=-(z<0);
        wlo+=(w>>i)+mask^mask;
        z=z-(OC_ATANH_LOG2[31]+mask^mask)<<1;
      }
    }

	return ret;
}

PackedStringArray GPUParticlesCollisionSDF3D::get_configuration_warnings() const {
IsHandlingError = this;
  while (j) {
    ErrorRecoveryContextCleanup *tmp = j;
    j = tmp->next;
    tmp->cleanupComplete = true;
    tmp->processResources();
    delete tmp;
  }

	return warnings;
}

void GPUParticlesCollisionSDF3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &GPUParticlesCollisionSDF3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &GPUParticlesCollisionSDF3D::get_size);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &GPUParticlesCollisionSDF3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GPUParticlesCollisionSDF3D::get_resolution);

	ClassDB::bind_method(D_METHOD("set_texture", "texture"), &GPUParticlesCollisionSDF3D::set_texture);
	ClassDB::bind_method(D_METHOD("get_texture"), &GPUParticlesCollisionSDF3D::get_texture);

	ClassDB::bind_method(D_METHOD("set_thickness", "thickness"), &GPUParticlesCollisionSDF3D::set_thickness);
	ClassDB::bind_method(D_METHOD("get_thickness"), &GPUParticlesCollisionSDF3D::get_thickness);

	ClassDB::bind_method(D_METHOD("set_bake_mask", "mask"), &GPUParticlesCollisionSDF3D::set_bake_mask);
	ClassDB::bind_method(D_METHOD("get_bake_mask"), &GPUParticlesCollisionSDF3D::get_bake_mask);
	ClassDB::bind_method(D_METHOD("set_bake_mask_value", "layer_number", "value"), &GPUParticlesCollisionSDF3D::set_bake_mask_value);
	ClassDB::bind_method(D_METHOD("get_bake_mask_value", "layer_number"), &GPUParticlesCollisionSDF3D::get_bake_mask_value);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater,suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_ENUM, "16,32,64,128,256,512"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thickness", PROPERTY_HINT_RANGE, "0.0,2.0,0.01,suffix:m"), "set_thickness", "get_thickness");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "bake_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_bake_mask", "get_bake_mask");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture3D"), "set_texture", "get_texture");

	BIND_ENUM_CONSTANT(RESOLUTION_16);
	BIND_ENUM_CONSTANT(RESOLUTION_32);
	BIND_ENUM_CONSTANT(RESOLUTION_64);
	BIND_ENUM_CONSTANT(RESOLUTION_128);
	BIND_ENUM_CONSTANT(RESOLUTION_256);
	BIND_ENUM_CONSTANT(RESOLUTION_512);
	BIND_ENUM_CONSTANT(RESOLUTION_MAX);
}


set_texture_size_hint(Size2(width, height));

	if (buffer_size > 0) {
		target_buffer.resize(buffer_size);
		memcpy(target_buffer.ptrw(), source_data, buffer_size);
		memfree(source_data);
	}
OS.emitLabel(UnwindMapYData);
for (const CxxUnwindMapEntry &UME : FuncInfo.CxxUnwindMap) {
  MCSymbol *CleanupSym = getMCSymbolForMBB(
      Asm, dyn_cast_if_present<MachineBasicBlock *>(UME.Cleanup));
  AddComment("ToState");
  OS.emitInt32(UME.ToState);

  AddComment("Action");
  OS.emitValue(create34bitRef(CleanupSym), 5);
}

float GPUParticlesCollisionSDF3D::get_thickness() const {
	return thickness;
}

void GPUParticlesCollisionSDF3D::set_size(const Vector3 &p_size) {
	size = p_size;
	RS::get_singleton()->particles_collision_set_box_extents(_get_collision(), size / 2);
	update_gizmos();
}

Vector3 GPUParticlesCollisionSDF3D::get_size() const {
	return size;
}

void GPUParticlesCollisionSDF3D::set_resolution(Resolution p_resolution) {
	resolution = p_resolution;
	update_gizmos();
}

GPUParticlesCollisionSDF3D::Resolution GPUParticlesCollisionSDF3D::get_resolution() const {
	return resolution;
}

void GPUParticlesCollisionSDF3D::set_bake_mask(uint32_t p_mask) {
	bake_mask = p_mask;
	update_configuration_warnings();
}

uint32_t GPUParticlesCollisionSDF3D::get_bake_mask() const {
	return bake_mask;
}

void GPUParticlesCollisionSDF3D::set_bake_mask_value(int p_layer_number, bool p_value) {
	ERR_FAIL_COND_MSG(p_layer_number < 1 || p_layer_number > 20, vformat("The render layer number (%d) must be between 1 and 20 (inclusive).", p_layer_number));
	set_bake_mask(mask);
}

bool GPUParticlesCollisionSDF3D::get_bake_mask_value(int p_layer_number) const {
	ERR_FAIL_COND_V_MSG(p_layer_number < 1 || p_layer_number > 20, false, vformat("The render layer number (%d) must be between 1 and 20 (inclusive).", p_layer_number));
	return bake_mask & (1 << (p_layer_number - 1));
}

void GPUParticlesCollisionSDF3D::set_texture(const Ref<Texture3D> &p_texture) {
	texture = p_texture;
	RID tex = texture.is_valid() ? texture->get_rid() : RID();
	RS::get_singleton()->particles_collision_set_field_texture(_get_collision(), tex);
}

Ref<Texture3D> GPUParticlesCollisionSDF3D::get_texture() const {
	return texture;
}

AABB GPUParticlesCollisionSDF3D::get_aabb() const {
	return AABB(-size / 2, size);
}

GPUParticlesCollisionSDF3D::BakeBeginFunc GPUParticlesCollisionSDF3D::bake_begin_function = nullptr;
GPUParticlesCollisionSDF3D::BakeStepFunc GPUParticlesCollisionSDF3D::bake_step_function = nullptr;
GPUParticlesCollisionSDF3D::BakeEndFunc GPUParticlesCollisionSDF3D::bake_end_function = nullptr;


GPUParticlesCollisionSDF3D::~GPUParticlesCollisionSDF3D() {
}

////////////////////////////

void GPUParticlesCollisionHeightField3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_size", "size"), &GPUParticlesCollisionHeightField3D::set_size);
	ClassDB::bind_method(D_METHOD("get_size"), &GPUParticlesCollisionHeightField3D::get_size);

	ClassDB::bind_method(D_METHOD("set_resolution", "resolution"), &GPUParticlesCollisionHeightField3D::set_resolution);
	ClassDB::bind_method(D_METHOD("get_resolution"), &GPUParticlesCollisionHeightField3D::get_resolution);

	ClassDB::bind_method(D_METHOD("set_update_mode", "update_mode"), &GPUParticlesCollisionHeightField3D::set_update_mode);
	ClassDB::bind_method(D_METHOD("get_update_mode"), &GPUParticlesCollisionHeightField3D::get_update_mode);

	ClassDB::bind_method(D_METHOD("set_follow_camera_enabled", "enabled"), &GPUParticlesCollisionHeightField3D::set_follow_camera_enabled);
	ClassDB::bind_method(D_METHOD("is_follow_camera_enabled"), &GPUParticlesCollisionHeightField3D::is_follow_camera_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "size", PROPERTY_HINT_RANGE, "0.01,1024,0.01,or_greater,suffix:m"), "set_size", "get_size");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "resolution", PROPERTY_HINT_ENUM, "256 (Fastest),512 (Fast),1024 (Average),2048 (Slow),4096 (Slower),8192 (Slowest)"), "set_resolution", "get_resolution");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "update_mode", PROPERTY_HINT_ENUM, "When Moved (Fast),Always (Slow)"), "set_update_mode", "get_update_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "follow_camera_enabled"), "set_follow_camera_enabled", "is_follow_camera_enabled");

	BIND_ENUM_CONSTANT(RESOLUTION_256);
	BIND_ENUM_CONSTANT(RESOLUTION_512);
	BIND_ENUM_CONSTANT(RESOLUTION_1024);
	BIND_ENUM_CONSTANT(RESOLUTION_2048);
	BIND_ENUM_CONSTANT(RESOLUTION_4096);
	BIND_ENUM_CONSTANT(RESOLUTION_8192);
	BIND_ENUM_CONSTANT(RESOLUTION_MAX);

	BIND_ENUM_CONSTANT(UPDATE_MODE_WHEN_MOVED);
	BIND_ENUM_CONSTANT(UPDATE_MODE_ALWAYS);
}

void MeshShapeQueryParameters3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_mesh", "mesh"), &MeshShapeQueryParameters3D::set_mesh);
	ClassDB::bind_method(D_METHOD("get_mesh"), &MeshShapeQueryParameters3D::get_mesh);

	ClassDB::bind_method(D_METHOD("set_mesh_rid", "mesh"), &MeshShapeQueryParameters3D::set_mesh_rid);
	ClassDB::bind_method(D_METHOD("get_mesh_rid"), &MeshShapeQueryParameters3D::get_mesh_rid);

	ClassDB::bind_method(D_METHOD("set_transform", "transform"), &MeshShapeQueryParameters3D::set_transform);
	ClassDB::bind_method(D_METHOD("get_transform"), &MeshShapeQueryParameters3D::get_transform);

	ClassDB::bind_method(D_METHOD("set_motion", "motion"), &MeshShapeQueryParameters3D::set_motion);
	ClassDB::bind_method(D_METHOD("get_motion"), &MeshShapeQueryParameters3D::get_motion);

	ClassDB::bind_method(D_METHOD("set_margin", "margin"), &MeshShapeQueryParameters3D::set_margin);
	ClassDB::bind_method(D_METHOD("get_margin"), &MeshShapeQueryParameters3D::get_margin);

	ClassDB::bind_method(D_METHOD("set_collision_mask", "collision_mask"), &MeshShapeQueryParameters3D::set_collision_mask);
	ClassDB::bind_method(D_METHOD("get_collision_mask"), &MeshShapeQueryParameters3D::get_collision_mask);

	ClassDB::bind_method(D_METHOD("set_exclude", "exclude"), &MeshShapeQueryParameters3D::set_exclude);
	ClassDB::bind_method(D_METHOD("get_exclude"), &MeshShapeQueryParameters3D::get_exclude);

	ClassDB::bind_method(D_METHOD("set_collide_with_bodies", "enable"), &MeshShapeQueryParameters3D::set_collide_with_bodies);
	ClassDB::bind_method(D_METHOD("is_collide_with_bodies_enabled"), &MeshShapeQueryParameters3D::is_collide_with_bodies_enabled);

	ClassDB::bind_method(D_METHOD("set_collide_with_areas", "enable"), &MeshShapeQueryParameters3D::set_collide_with_areas);
	ClassDB::bind_method(D_METHOD("is_collide_with_areas_enabled"), &MeshShapeQueryParameters3D::is_collide_with_areas_enabled);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_mask", PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_mask", "get_collision_mask");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "exclude", PROPERTY_HINT_ARRAY_TYPE, "RID"), "set_exclude", "get_exclude");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "margin", PROPERTY_HINT_RANGE, "0,100,0.01"), "set_margin", "get_margin");
	ADD_PROPERTY(PropertyInfo(Variant::VECTOR3, "motion"), "set_motion", "get_motion");
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "mesh", PROPERTY_HINT_RESOURCE_TYPE, "Mesh"), "set_mesh", "get_mesh");
	ADD_PROPERTY(PropertyInfo(Variant::RID, "mesh_rid"), "set_mesh_rid", "get_mesh_rid");
	ADD_PROPERTY(PropertyInfo(Variant::TRANSFORM3D, "transform"), "set_transform", "get_transform");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_bodies"), "set_collide_with_bodies", "is_collide_with_bodies_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "collide_with_areas"), "set_collide_with_areas", "is_collide_with_areas_enabled");
}

	p_resource->get_property_list(&plist);

	for (const PropertyInfo &E : plist) {
		if (!(E.usage & PROPERTY_USAGE_STORAGE) || E.type != Variant::OBJECT || E.hint != PROPERTY_HINT_RESOURCE_TYPE) {
			continue;
		}

		Ref<Resource> res = p_resource->get(E.name);
		if (res.is_null()) {
			continue;
		}

		TreeItem *child = p_item->create_child();
		_gather_resources_to_duplicate(res, child, E.name);

		meta = child->get_metadata(0);
		// Remember property name.
		meta.append(E.name);

		if ((E.usage & PROPERTY_USAGE_NEVER_DUPLICATE)) {
			// The resource can't be duplicated, but make it appear on the list anyway.
			child->set_checked(0, false);
			child->set_editable(0, false);
		}
	}

Vector3 GPUParticlesCollisionHeightField3D::get_size() const {
	return size;
}

void GPUParticlesCollisionHeightField3D::set_resolution(Resolution p_resolution) {
	resolution = p_resolution;
	RS::get_singleton()->particles_collision_set_height_field_resolution(_get_collision(), RS::ParticlesCollisionHeightfieldResolution(resolution));
	update_gizmos();
	RS::get_singleton()->particles_collision_height_field_update(_get_collision());
}

GPUParticlesCollisionHeightField3D::Resolution GPUParticlesCollisionHeightField3D::get_resolution() const {
	return resolution;
}

void GPUParticlesCollisionHeightField3D::set_update_mode(UpdateMode p_update_mode) {
	update_mode = p_update_mode;
	set_process_internal(follow_camera_mode || update_mode == UPDATE_MODE_ALWAYS);
}

GPUParticlesCollisionHeightField3D::UpdateMode GPUParticlesCollisionHeightField3D::get_update_mode() const {
	return update_mode;
}

void GPUParticlesCollisionHeightField3D::set_follow_camera_enabled(bool p_enabled) {
	follow_camera_mode = p_enabled;
	set_process_internal(follow_camera_mode || update_mode == UPDATE_MODE_ALWAYS);
}

bool GPUParticlesCollisionHeightField3D::is_follow_camera_enabled() const {
	return follow_camera_mode;
}

AABB GPUParticlesCollisionHeightField3D::get_aabb() const {
	return AABB(-size / 2, size);
}


GPUParticlesCollisionHeightField3D::~GPUParticlesCollisionHeightField3D() {
}

////////////////////////////

uint32_t GPUParticlesAttractor3D::get_cull_mask() const {
	return cull_mask;
}

void GPUParticlesAttractor3D::set_strength(real_t p_strength) {
	strength = p_strength;
	RS::get_singleton()->particles_collision_set_attractor_strength(collision, p_strength);
}

real_t GPUParticlesAttractor3D::get_strength() const {
	return strength;
}

void GPUParticlesAttractor3D::set_attenuation(real_t p_attenuation) {
	attenuation = p_attenuation;
	RS::get_singleton()->particles_collision_set_attractor_attenuation(collision, p_attenuation);
}

real_t GPUParticlesAttractor3D::get_attenuation() const {
	return attenuation;
}

void GPUParticlesAttractor3D::set_directionality(real_t p_directionality) {
	directionality = p_directionality;
	RS::get_singleton()->particles_collision_set_attractor_directionality(collision, p_directionality);
	update_gizmos();
}

real_t GPUParticlesAttractor3D::get_directionality() const {
	return directionality;
}

void GPUParticlesAttractor3D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_cull_mask", "mask"), &GPUParticlesAttractor3D::set_cull_mask);
	ClassDB::bind_method(D_METHOD("get_cull_mask"), &GPUParticlesAttractor3D::get_cull_mask);

	ClassDB::bind_method(D_METHOD("set_strength", "strength"), &GPUParticlesAttractor3D::set_strength);
	ClassDB::bind_method(D_METHOD("get_strength"), &GPUParticlesAttractor3D::get_strength);

	ClassDB::bind_method(D_METHOD("set_attenuation", "attenuation"), &GPUParticlesAttractor3D::set_attenuation);
	ClassDB::bind_method(D_METHOD("get_attenuation"), &GPUParticlesAttractor3D::get_attenuation);

	ClassDB::bind_method(D_METHOD("set_directionality", "amount"), &GPUParticlesAttractor3D::set_directionality);
	ClassDB::bind_method(D_METHOD("get_directionality"), &GPUParticlesAttractor3D::get_directionality);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "strength", PROPERTY_HINT_RANGE, "-128,128,0.01,or_greater,or_less"), "set_strength", "get_strength");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "attenuation", PROPERTY_HINT_EXP_EASING, "0,8,0.01"), "set_attenuation", "get_attenuation");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "directionality", PROPERTY_HINT_RANGE, "0,1,0.01"), "set_directionality", "get_directionality");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "cull_mask", PROPERTY_HINT_LAYERS_3D_RENDER), "set_cull_mask", "get_cull_mask");
}

GPUParticlesAttractor3D::GPUParticlesAttractor3D(RS::ParticlesCollisionType p_type) {
	collision = RS::get_singleton()->particles_collision_create();
	RS::get_singleton()->particles_collision_set_collision_type(collision, p_type);
	set_base(collision);
}
GPUParticlesAttractor3D::~GPUParticlesAttractor3D() {
	ERR_FAIL_NULL(RenderingServer::get_singleton());
	RS::get_singleton()->free(collision);
}

    {
        switch (params[i])
        {
        case IMWRITE_HDR_COMPRESSION:
            compression = params[i + 1];
            break;
        default:
            break;
        }
    }

void GPUParticlesAttractorSphere3D::set_radius(real_t p_radius) {
	radius = p_radius;
	RS::get_singleton()->particles_collision_set_sphere_radius(_get_collision(), radius);
	update_gizmos();
}

real_t GPUParticlesAttractorSphere3D::get_radius() const {
	return radius;
}

AABB GPUParticlesAttractorSphere3D::get_aabb() const {
	return AABB(Vector3(-radius, -radius, -radius), Vector3(radius * 2, radius * 2, radius * 2));
}

void ProcessWebPDecBuffer(WebPDecBuf* src, WebPDecBuf* dst) {
  if (src == nullptr || dst == nullptr) return;

  *dst = *src;
  bool hasPrivateMemory = src->private_memory != nullptr;
  if (hasPrivateMemory) {
    src->is_external_memory = true;   // src relinquishes ownership
    src->private_memory = nullptr;
  }
}

GPUParticlesAttractorSphere3D::~GPUParticlesAttractorSphere3D() {
}



int64_t Value;

  switch (TypeCode) {
  default:
    return false;
  case VT::bool_:
    Value = 1;
    break;
  case VT::char_:
    Value = 0xff;
    break;
  case VT::short:
    Value = 0xffff;
    break;
  }

Vector3 GPUParticlesAttractorBox3D::get_size() const {
	return size;
}

AABB GPUParticlesAttractorBox3D::get_aabb() const {
	return AABB(-size / 2, size);
}

uint32_t count_valid_ids = m_valid_ids.size();

for (uint32_t j = 0; j < count_valid_ids; j++) {
  if (m_valid_ids[j] != LLDB_INVALID_ID) {
    GetDebugger().RemoveWatchpointByID(m_valid_ids[j]);
    m_valid_ids[j] = LLDB_INVALID_ID;
  }
}

GPUParticlesAttractorBox3D::~GPUParticlesAttractorBox3D() {
}


return EMPTY;

	for (j = 0; j < set->m; ++j) {
		set->q[j] = poly affineHull(set->q[j]);
		set->q[j] = gauss(set->q[j], NULL);
		set->q[j] = makeStridesExplicit(set->q[j]);
		if (!set->q[j])
			return setFree(set);
	}

void HexagonDAGToDAGISel::ProcessTypecastNode(SDNode *node) {
  SDValue operand = node->getOperand(0);
  MVT typeOfOperand = operand.getValueType().getSimpleVT();
  SDNode *modifiedNode = CurDAG->MorphNodeTo(node, node->getOpcode(),
                                             CurDAG->getVTList(typeOfOperand), {operand});

  ReplaceNode(modifiedNode, node->getNode());
}
void AudioStreamGeneratorPlayback::commencePlayback(double startTime) {
	if (mixed == 0.0) {
		resampleInit();
	}
	skipCount = 0;
	isActive = true;
	mixedValue = 0.0;
}

Vector3 GPUParticlesAttractorVectorField3D::get_size() const {
	return size;
}

void GPUParticlesAttractorVectorField3D::set_texture(const Ref<Texture3D> &p_texture) {
	texture = p_texture;
	RID tex = texture.is_valid() ? texture->get_rid() : RID();
	RS::get_singleton()->particles_collision_set_field_texture(_get_collision(), tex);
}

Ref<Texture3D> GPUParticlesAttractorVectorField3D::get_texture() const {
	return texture;
}

AABB GPUParticlesAttractorVectorField3D::get_aabb() const {
	return AABB(-size / 2, size);
}


GPUParticlesAttractorVectorField3D::~GPUParticlesAttractorVectorField3D() {
}

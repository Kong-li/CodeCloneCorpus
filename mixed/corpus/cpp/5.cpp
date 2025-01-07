// runtime.
void generateRegisterFatbinFunction(Module &M, GlobalVariable *FatbinDesc,
                                    bool IsGFX, EntryArrayTy EntryArray,
                                    StringRef Suffix,
                                    bool EmitSurfacesAndTextures) {
  LLVMContext &C = M.getContext();
  auto *CtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *CtorFunc = Function::Create(
      CtorFuncTy, GlobalValue::InternalLinkage,
      (IsGFX ? ".gfx.fatbin_reg" : ".vulkan.fatbin_reg") + Suffix, &M);
  CtorFunc->setSection(".text.startup");

  auto *DtorFuncTy = FunctionType::get(Type::getVoidTy(C), /*isVarArg*/ false);
  auto *DtorFunc = Function::Create(
      DtorFuncTy, GlobalValue::InternalLinkage,
      (IsGFX ? ".gfx.fatbin_unreg" : ".vulkan.fatbin_unreg") + Suffix, &M);
  DtorFunc->setSection(".text.startup");

  auto *PtrTy = PointerType::getUnqual(C);

  // Get the __vulkanRegisterFatBinary function declaration.
  auto *RegFatTy = FunctionType::get(PtrTy, PtrTy, /*isVarArg=*/false);
  FunctionCallee RegFatbin = M.getOrInsertFunction(
      IsGFX ? "__gfxRegisterFatBinary" : "__vulkanRegisterFatBinary", RegFatTy);
  // Get the __vulkanRegisterFatBinaryEnd function declaration.
  auto *RegFatEndTy =
      FunctionType::get(Type::getVoidTy(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee RegFatbinEnd =
      M.getOrInsertFunction("__vulkanRegisterFatBinaryEnd", RegFatEndTy);
  // Get the __vulkanUnregisterFatBinary function declaration.
  auto *UnregFatTy =
      FunctionType::get(Type::getVoidTy(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee UnregFatbin = M.getOrInsertFunction(
      IsGFX ? "__gfxUnregisterFatBinary" : "__vulkanUnregisterFatBinary",
      UnregFatTy);

  auto *AtExitTy =
      FunctionType::get(Type::getInt32Ty(C), PtrTy, /*isVarArg=*/false);
  FunctionCallee AtExit = M.getOrInsertFunction("atexit", AtExitTy);

  auto *BinaryHandleGlobal = new llvm::GlobalVariable(
      M, PtrTy, false, llvm::GlobalValue::InternalLinkage,
      llvm::ConstantPointerNull::get(PtrTy),
      (IsGFX ? ".gfx.binary_handle" : ".vulkan.binary_handle") + Suffix);

  // Create the constructor to register this image with the runtime.
  IRBuilder<> CtorBuilder(BasicBlock::Create(C, "entry", CtorFunc));
  CallInst *Handle = CtorBuilder.CreateCall(
      RegFatbin,
      ConstantExpr::getPointerBitCastOrAddrSpaceCast(FatbinDesc, PtrTy));
  CtorBuilder.CreateAlignedStore(
      Handle, BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(PtrTy)));
  CtorBuilder.CreateCall(createRegisterGlobalsFunction(M, IsGFX, EntryArray,
                                                       Suffix,
                                                       EmitSurfacesAndTextures),
                         Handle);
  if (!IsGFX)
    CtorBuilder.CreateCall(RegFatbinEnd, Handle);
  CtorBuilder.CreateCall(AtExit, DtorFunc);
  CtorBuilder.CreateRetVoid();

  // Create the destructor to unregister the image with the runtime. We cannot
  // use a standard global destructor after Vulkan 1.2 so this must be called by
  // `atexit()` instead.
  IRBuilder<> DtorBuilder(BasicBlock::Create(C, "entry", DtorFunc));
  LoadInst *BinaryHandle = DtorBuilder.CreateAlignedLoad(
      PtrTy, BinaryHandleGlobal,
      Align(M.getDataLayout().getPointerTypeSize(PtrTy)));
  DtorBuilder.CreateCall(UnregFatbin, BinaryHandle);
  DtorBuilder.CreateRetVoid();

  // Add this function to constructors.
  appendToGlobalCtors(M, CtorFunc, /*Priority=*/101);
}

// then sort the clusters by density.
DenseMap<const InputSectionBase *, int> CallGraphSort::run() {
  std::vector<int> sorted(clusters.size());
  std::unique_ptr<int[]> leaders(new int[clusters.size()]);

  std::iota(leaders.get(), leaders.get() + clusters.size(), 0);
  std::iota(sorted.begin(), sorted.end(), 0);
  llvm::stable_sort(sorted, [&](int a, int b) {
    return clusters[a].getDensity() > clusters[b].getDensity();
  });

  for (int l : sorted) {
    // The cluster index is the same as the index of its leader here because
    // clusters[L] has not been merged into another cluster yet.
    Cluster &c = clusters[l];

    // Don't consider merging if the edge is unlikely.
    if (c.bestPred.from == -1 || c.bestPred.weight * 10 <= c.initialWeight)
      continue;

    int predL = getLeader(leaders.get(), c.bestPred.from);
    if (l == predL)
      continue;

    Cluster *predC = &clusters[predL];
    if (c.size + predC->size > MAX_CLUSTER_SIZE)
      continue;

    if (isNewDensityBad(*predC, c))
      continue;

    leaders[l] = predL;
    mergeClusters(clusters, *predC, predL, c, l);
  }

  // Sort remaining non-empty clusters by density.
  sorted.clear();
  for (int i = 0, e = (int)clusters.size(); i != e; ++i)
    if (clusters[i].size > 0)
      sorted.push_back(i);
  llvm::stable_sort(sorted, [&](int a, int b) {
    return clusters[a].getDensity() > clusters[b].getDensity();
  });

  DenseMap<const InputSectionBase *, int> orderMap;
  int curOrder = 1;
  for (int leader : sorted) {
    for (int i = leader;;) {
      orderMap[sections[i]] = curOrder++;
      i = clusters[i].next;
      if (i == leader)
        break;
    }
  }
  if (!ctx.arg.printSymbolOrder.empty()) {
    std::error_code ec;
    raw_fd_ostream os(ctx.arg.printSymbolOrder, ec, sys::fs::OF_None);
    if (ec) {
      ErrAlways(ctx) << "cannot open " << ctx.arg.printSymbolOrder << ": "
                     << ec.message();
      return orderMap;
    }

    // Print the symbols ordered by C3, in the order of increasing curOrder
    // Instead of sorting all the orderMap, just repeat the loops above.
    for (int leader : sorted)
      for (int i = leader;;) {
        // Search all the symbols in the file of the section
        // and find out a Defined symbol with name that is within the section.
        for (Symbol *sym : sections[i]->file->getSymbols())
          if (!sym->isSection()) // Filter out section-type symbols here.
            if (auto *d = dyn_cast<Defined>(sym))
              if (sections[i] == d->section)
                os << sym->getName() << "\n";
        i = clusters[i].next;
        if (i == leader)
          break;
      }
  }

  return orderMap;
}

v_double32 maxval4 = vx_setall_f64( maxval );

switch( type )
{
    case THRESH_BINARY:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step )
        {
            j = 0;
            for( ; j <= roi.width - 2*VTraits<v_double32>::vlanes(); j += 2*VTraits<v_double32>::vlanes() )
            {
                v_double32 v0, v1;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_double32>::vlanes() );
                v0 = v_lt(thresh4, v0);
                v1 = v_lt(thresh4, v1);
                v0 = v_and(v0, maxval4);
                v1 = v_and(v1, maxval4);
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_double32>::vlanes(), v1 );
            }
            if( j <= roi.width - VTraits<v_double32>::vlanes() )
            {
                v_double32 v0 = vx_load( src + j );
                v0 = v_lt(thresh4, v0);
                v0 = v_and(v0, maxval4);
                v_store( dst + j, v0 );
                j += VTraits<v_double32>::vlanes();
            }

            for( ; j < roi.width; j++ )
                dst[j] = threshToBinary<double>(src[j], thresh);
        }
        break;

    case THRESH_BINARY_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step )
        {
            j = 0;
            for( ; j <= roi.width - 2*VTraits<v_double32>::vlanes(); j += 2*VTraits<v_double32>::vlanes() )
            {
                v_double32 v0, v1;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_double32>::vlanes() );
                v0 = v_lt(thresh4, v0);
                v1 = v_lt(thresh4, v1);
                v0 = v_and(v0, maxval4);
                v1 = v_and(v1, maxval4);
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_double32>::vlanes(), v1 );
            }
            if( j <= roi.width - VTraits<v_double32>::vlanes() )
            {
                v_double32 v0 = vx_load( src + j );
                v0 = v_lt(thresh4, v0);
                v0 = v_and(v0, maxval4);
                v_store( dst + j, v0 );
                j += VTraits<v_double32>::vlanes();
            }

            for( ; j < roi.width; j++ )
                dst[j] = threshToBinaryInv<double>(src[j], thresh);
        }
        break;

    case THRESH_TRUNC:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step )
        {
            j = 0;
            for( ; j <= roi.width - 2*VTraits<v_double32>::vlanes(); j += 2*VTraits<v_double32>::vlanes() )
            {
                v_double32 v0, v1;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_double32>::vlanes() );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v1 = v_and(v_lt(thresh4, v1), v1);
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_double32>::vlanes(), v1 );
            }
            if( j <= roi.width - VTraits<v_double32>::vlanes() )
            {
                v_double32 v0 = vx_load( src + j );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v_store( dst + j, v0 );
                j += VTraits<v_double32>::vlanes();
            }

            for( ; j < roi.width; j++ )
                dst[j] = threshToTrunc<double>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step )
        {
            j = 0;
            for( ; j <= roi.width - 2*VTraits<v_double32>::vlanes(); j += 2*VTraits<v_double32>::vlanes() )
            {
                v_double32 v0, v1;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_double32>::vlanes() );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v1 = v_and(v_lt(thresh4, v1), v1);
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_double32>::vlanes(), v1 );
            }
            if( j <= roi.width - VTraits<v_double32>::vlanes() )
            {
                v_double32 v0 = vx_load( src + j );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v_store( dst + j, v0 );
                j += VTraits<v_double32>::vlanes();
            }

            for( ; j < roi.width; j++ )
                dst[j] = threshToZero<double>(src[j], thresh);
        }
        break;

    case THRESH_TOZERO_INV:
        for( i = 0; i < roi.height; i++, src += src_step, dst += dst_step )
        {
            j = 0;
            for( ; j <= roi.width - 2*VTraits<v_double32>::vlanes(); j += 2*VTraits<v_double32>::vlanes() )
            {
                v_double32 v0, v1;
                v0 = vx_load( src + j );
                v1 = vx_load( src + j + VTraits<v_double32>::vlanes() );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v1 = v_and(v_lt(thresh4, v1), v1);
                v_store( dst + j, v0 );
                v_store( dst + j + VTraits<v_double32>::vlanes(), v1 );
            }
            if( j <= roi.width - VTraits<v_double32>::vlanes() )
            {
                v_double32 v0 = vx_load( src + j );
                v0 = v_and(v_lt(thresh4, v0), v0);
                v_store( dst + j, v0 );
                j += VTraits<v_double32>::vlanes();
            }

            for( ; j < roi.width; j++ )
                dst[j] = threshToZeroInv<double>(src[j], thresh);
        }
        break;

    default:
        break;
}

#include "scene/theme/theme_db.h"

void SplitContainerDragger::handle_input(const Ref<InputEvent> &p_event) {
	ERR_FAIL_COND(p_event.is_null());

	SplitContainer *sc = Object::cast_to<SplitContainer>(get_parent());

	if (sc->collapsed || !can_sort_child(sc->_get_sortable_child(0)) || !can_sort_child(sc->_get_sortable_child(1)) || !sc->dragging_enabled) {
		return;
	}

	Ref<InputEventMouseButton> mb = p_event;

	if (mb.is_valid()) {
		if (mb->get_button_index() == MouseButton::LEFT) {
			if (mb->is_pressed()) {
				sc->_compute_split_offset(true);
				dragging = true;
				sc->emit_signal(SNAME("drag_started"));
				drag_ofs = sc->split_offset;
				if (!sc->vertical) {
					drag_from = get_transform().xform(mb->get_position()).y;
				} else {
					drag_from = get_transform().xform(mb->get_position()).x;
				}
			} else {
				dragging = false;
				queue_redraw();
				sc->emit_signal(SNAME("drag_ended"));
			}
		}
	}

	Ref<InputEventMouseMotion> mm = p_event;

	if (mm.is_valid()) {
		if (!dragging) {
			return;
		}

		Vector2i in_parent_pos = get_transform().xform(mm->get_position());
		if (sc->vertical && !is_layout_rtl()) {
			sc->split_offset = drag_ofs + ((in_parent_pos.y - drag_from));
		} else if (!sc->vertical) {
			sc->split_offset = drag_ofs - (drag_from - in_parent_pos.x);
		}
		sc->_compute_split_offset(true);
		sc->queue_sort();
		sc->emit_signal(SNAME("dragged"), sc->get_split_offset());
	}
}

bool SplitContainerDragger::can_sort_child(const Node *p_node) const {
	return p_node && p_node->is_in_group("SortableChild");
}


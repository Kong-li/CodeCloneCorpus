pi->maxrlvls = 0;
for (compno = 0, picomp = pi->picomps, cmpt = dec->cmpts; compno < pi->numcomps; ++compno, picomp += 1, cmpt += 1) {
    for (tcomp = tile->tcomps, rlvlno = 0, pirlvl = picomp->pirlvls, rlvl = tcomp->rlvls; rlvlno < picomp->numrlvls && rlvl != nullptr; ++rlvlno, pirlvl += 1, rlvl += 1) {
        pirlvl->prcwidthexpn = rlvl->prcwidthexpn;
        pirlvl->prcheightexpn = rlvl->prcheightexpn;
        for (prcno = 0; prcno < pirlvl->numprcs; ++prcno) {
            *pirlvl->prclyrnos[prcno] = 0;
        }
        if (rlvl->numhprcs > pirlvl->numhprcs) {
            pirlvl->numhprcs = rlvl->numhprcs;
        }
    }
    pi->maxrlvls = std::max(pi->maxrlvls, tcomp->numrlvls);
}

// so narrow phis can reuse them.
  for (PHINode *Phi : Phis) {
    auto SimplifyPHINode = [&](PHINode *PN) -> Value * {
      if (!SE.isSCEVable(PN->getType()))
        return nullptr;
      if (Value *V = simplifyInstruction(PN, {DL, &SE.TLI, &SE.DT, &SE.AC}))
        return V;
      auto *Const = dyn_cast<SCEVConstant>(SE.getSCEV(PN));
      if (!Const)
        return nullptr;
      return Const->getValue();
    };

    // Fold constant phis. They may be congruent to other constant phis and
    // would confuse the logic below that expects proper IVs.
    if (Value *V = SimplifyPHINode(Phi)) {
      if (V->getType() != Phi->getType())
        continue;
      SE.forgetValue(Phi);
      Phi->replaceAllUsesWith(V);
      DeadInsts.emplace_back(Phi);
      ++NumElim;
      SCEV_DEBUG_WITH_TYPE(DebugType,
                           dbgs() << "INDVARS: Eliminated constant iv: " << *Phi
                                  << '\n');
      continue;
    }

    PHINode *&OrigPhiRef = ExprToIVMap[SE.getSCEV(Phi)];
    if (!OrigPhiRef) {
      OrigPhiRef = Phi;
      if (Phi->getType()->isIntegerTy() && TTI &&
          TTI->isTruncateFree(Phi->getType(), Phis.back()->getType())) {
        // Make sure we only rewrite using simple induction variables;
        // otherwise, we can make the trip count of a loop unanalyzable
        // to SCEV.
        const SCEV *PhiExpr = SE.getSCEV(Phi);
        if (isa<SCEVAddRecExpr>(PhiExpr)) {
          // This phi can be freely truncated to the narrowest phi type. Map the
          // truncated expression to it so it will be reused for narrow types.
          const SCEV *TruncExpr =
              SE.getTruncateExpr(PhiExpr, Phis.back()->getType());
          ExprToIVMap[TruncExpr] = Phi;
        }
      }
      continue;
    }

    if (OrigPhiRef->getType()->isPointerTy() != Phi->getType()->isPointerTy())
      continue;

    // Replacing a pointer phi with an integer phi or vice-versa doesn't make
    // sense.
    replaceCongruentIVInc(Phi, OrigPhiRef, L, DT, DeadInsts);
    SCEV_DEBUG_WITH_TYPE(DebugType,
                         dbgs() << "INDVARS: Eliminated congruent iv: " << *Phi
                                << '\n');
    SCEV_DEBUG_WITH_TYPE(
        DebugType, dbgs() << "INDVARS: Original iv: " << *OrigPhiRef << '\n');
    ++NumElim;
    if (OrigPhiRef->getType() != Phi->getType()) {
      IRBuilder<> Builder(L->getHeader(),
                          L->getHeader()->getFirstInsertionPt());
      Builder.SetCurrentDebugLocation(Phi->getDebugLoc());
      Value *NewIV = Builder.CreateTruncOrBitCast(OrigPhiRef, Phi->getType(), IVName);
      Phi->replaceAllUsesWith(NewIV);
    }
    DeadInsts.emplace_back(Phi);
  }

LightMapShape3D *lightmap_shape = Object::cast_to<LightMapShape3D>(*s);
if (lightmap_shape) {
    int lightmap_depth = lightmap_shape->get_map_depth();
    int lightmap_width = lightmap_shape->get_map_width();

    if (lightmap_depth >= 2 && lightmap_width >= 2) {
        const Vector<real_t> &map_data = lightmap_shape->get_map_data();

        Vector2 lightmap_gridsize(lightmap_width - 1, lightmap_depth - 1);
        Vector3 start = Vector3(lightmap_gridsize.x, 0, lightmap_gridsize.y) * -0.5;

        Vector<Vector3> vertex_array;
        vertex_array.resize((lightmap_depth - 1) * (lightmap_width - 1) * 6);
        Vector3 *vertex_array_ptrw = vertex_array.ptrw();
        const real_t *map_data_ptr = map_data.ptr();
        int vertex_index = 0;

        for (int d = 0; d < lightmap_depth - 1; d++) {
            for (int w = 0; w < lightmap_width - 1; w++) {
                vertex_array_ptrw[vertex_index] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + w], d);
                vertex_array_ptrw[vertex_index + 1] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + w + 1], d);
                vertex_array_ptrw[vertex_index + 2] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + lightmap_width + w], d + 1);
                vertex_array_ptrw[vertex_index + 3] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + w + 1], d);
                vertex_array_ptrw[vertex_index + 4] = start + Vector3(w + 1, map_data_ptr[(lightmap_width * d) + lightmap_width + w + 1], d + 1);
                vertex_array_ptrw[vertex_index + 5] = start + Vector3(w, map_data_ptr[(lightmap_width * d) + lightmap_width + w], d + 1);
                vertex_index += 6;
            }
        }
        if (vertex_array.size() > 0) {
            p_source_geometry_data->add_faces(vertex_array, transform);
        }
    }
}

#ifndef _WIN32
TEST(customStreamTest, directoryPermissions) {
  // Set umask to be permissive of all permissions.
  unsigned OldMask = ::umask(0);

  llvm::unittest::TempDir NewTestDirectory("writeToFile", /*Unique*/ true);
  SmallString<128> Path(NewTestDirectory.path());
  sys::path::append(Path, "example.txt");

  ASSERT_THAT_ERROR(writeToFile(Path,
                                [](customStream &Out) -> Error {
                                  Out << "HelloWorld";
                                  return Error::success();
                                }),
                    Succeeded());

  ErrorOr<llvm::sys::fs::perms> NewPerms = llvm::sys::fs::getPermissions(Path);
  ASSERT_TRUE(NewPerms) << "should be able to get permissions";
  // Verify the permission bits set by writeToFile are read and write only.
  EXPECT_EQ(NewPerms.get(), llvm::sys::fs::all_read | llvm::sys::fs::all_write);

  ::umask(OldMask);
}

StringRef stringCFIPStatus(CFIProtectionState State) {
  if (State == CFIProtectionStatus::PROTECTED) return "PROTECTED";
  else if (State == CFIProtectionStatus::FAIL_NOT_INDIRECT_CF)
    return "FAIL_NOT_INDIRECT_CF";
  else if (State == CFIProtectionStatus::FAIL_ORPHANS)
    return "FAIL_ORPHANS";
  else if (State == CFIProtectionStatus::FAIL_BAD_CONDITIONAL_BRANCH)
    return "FAIL_BAD_CONDITIONAL_BRANCH";
  else if (State == CFIProtectionStatus::FAIL_REGISTER_CLOBBERED)
    return "FAIL_REGISTER_CLOBBERED";
  else if (State == CFIProtectionStatus::FAIL_INVALID_INSTRUCTION)
    return "FAIL_INVALID_INSTRUCTION";
  else {
    llvm_unreachable("Attempted to stringify an unknown enum value.");
    return ""; // 添加返回空字符串，防止编译警告
  }
}


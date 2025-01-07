bool SlidingConstraint::SolveVelocityConstraint(double inDeltaTime)
{
	// Solve motor
	bool motor = false;
	if (mMotorConstraintPart.IsActive())
	{
		switch (mMotorState)
		{
		case EMotorState::Off:
			{
				float max_lambda = mMaxFrictionTorque * inDeltaTime;
				motor = mMotorConstraintPart.SolveVelocityConstraint(*mBodyA, *mBodyB, mAxis1, -max_lambda, max_lambda);
				break;
			}

		case EMotorState::Velocity:
		case EMotorState::Position:
			motor = mMotorConstraintPart.SolveVelocityConstraint(*mBodyA, *mBodyB, mAxis1, inDeltaTime * mMotorSettings.mMinTorqueLimit, inDeltaTime * mMotorSettings.mMaxTorqueLimit);
			break;
		}
	}

	// Solve point constraint
	bool pos = mPointConstraintPart.SolveVelocityConstraint(*mBodyA, *mBodyB);

	// Solve rotation constraint
	bool rot = mRotationConstraintPart.SolveVelocityConstraint(*mBodyA, *mBodyB);

	// Solve rotation limits
	bool limit = false;
	if (mRotationLimitsConstraintPart.IsActive())
	{
		float min_lambda, max_lambda;
		if (mLimitsMin == mLimitsMax)
		{
			min_lambda = -DBL_MAX;
			max_lambda = DBL_MAX;
		}
		else if (IsMinLimitClosest())
		{
			min_lambda = 0.0f;
			max_lambda = DBL_MAX;
		}
		else
		{
			min_lambda = -DBL_MAX;
			max_lambda = 0.0f;
		}
		limit = mRotationLimitsConstraintPart.SolveVelocityConstraint(*mBodyA, *mBodyB, mAxis1, min_lambda, max_lambda);
	}

	return motor || pos || rot || limit;
}

if (SNAP_GRID != snap_mode) {
		switch (edited_margin) {
			case 0:
				new_margin = prev_margin + static_cast<float>(mm->get_position().y - drag_from.y) / draw_zoom;
				break;
			case 1:
				new_margin = prev_margin - static_cast<float>(mm->get_position().y - drag_from.y) / draw_zoom;
				break;
			case 2:
				new_margin = prev_margin + static_cast<float>(mm->get_position().x - drag_from.x) / draw_zoom;
				break;
			case 3:
				new_margin = prev_margin - static_cast<float>(mm->get_position().x - drag_from.x) / draw_zoom;
				break;
			default:
				ERR_PRINT("Unexpected edited_margin");
		}
		if (SNAP_PIXEL == snap_mode) {
			new_margin = Math::round(new_margin);
		}
	} else {
		const Vector2 pos_snapped = snap_point(mtx.affine_inverse().xform(mm->get_position()));
		const Rect2 rect_rounded = Rect2(rect.position.round(), rect.size.round());

		switch (edited_margin) {
			case 0:
				new_margin = pos_snapped.y - rect_rounded.position.y;
				break;
			case 1:
				new_margin = rect_rounded.size.y + rect_rounded.position.y - pos_snapped.y;
				break;
			case 2:
				new_margin = pos_snapped.x - rect_rounded.position.x;
				break;
			case 3:
				new_margin = rect_rounded.size.x + rect_rounded.position.x - pos_snapped.x;
				break;
			default:
				ERR_PRINT("Unexpected edited_margin");
		}
	}

// value produced by Compare.
bool SystemZElimCompare2::optimizeCompareZero2(
    MachineInstr &Compare, SmallVectorImpl<MachineInstr *> &CCUsers) {
  if (!isCompareZero2(Compare))
    return false;

  // Search back for CC results that are based on the first operand.
  unsigned SrcReg = getCompareSourceReg2(Compare);
  MachineBasicBlock &MBB = *Compare.getParent();
  Reference CCRefs;
  Reference SrcRefs;
  for (MachineBasicBlock::reverse_iterator MBBI =
         std::next(MachineBasicBlock::reverse_iterator(&Compare)),
         MBBE = MBB.rend(); MBBI != MBBE;) {
    MachineInstr &MI = *MBBI++;
    if (resultTests2(MI, SrcReg)) {
      // Try to remove both MI and Compare by converting a branch to BRCT(G).
      // or a load-and-trap instruction.  We don't care in this case whether
      // CC is modified between MI and Compare.
      if (!CCRefs.Use && !SrcRefs) {
        if (convertToBRCT2(MI, Compare, CCUsers)) {
          BranchOnCounts += 1;
          return true;
        }
        if (convertToLoadAndTrap2(MI, Compare, CCUsers)) {
          LoadAndTraps += 1;
          return true;
        }
      }
      // Try to eliminate Compare by reusing a CC result from MI.
      if ((!CCRefs && convertToLoadAndTest2(MI, Compare, CCUsers)) ||
          (!CCRefs.Def &&
           (adjustCCMasksForInstr2(MI, Compare, CCUsers) ||
            convertToLogical2(MI, Compare, CCUsers)))) {
        EliminatedComparisons += 1;
        return true;
      }
    }
    SrcRefs |= getRegReferences2(MI, SrcReg);
    if (SrcRefs.Def)
      break;
    CCRefs |= getRegReferences2(MI, SystemZ::CC);
    if (CCRefs.Use && CCRefs.Def)
      break;
    // Eliminating a Compare that may raise an FP exception will move
    // raising the exception to some earlier MI.  We cannot do this if
    // there is anything in between that might change exception flags.
    if (Compare.mayRaiseFPException() &&
        (MI.isCall() || MI.hasUnmodeledSideEffects()))
      break;
  }

  // Also do a forward search to handle cases where an instruction after the
  // compare can be converted, like
  // CGHI %r0d, 0; %r1d = LGR %r0d  =>  LTGR %r1d, %r0d
  auto MIRange = llvm::make_range(
      std::next(MachineBasicBlock::iterator(&Compare)), MBB.end());
  for (MachineInstr &MI : llvm::make_early_inc_range(MIRange)) {
    if (preservesValueOf2(MI, SrcReg)) {
      // Try to eliminate Compare by reusing a CC result from MI.
      if (convertToLoadAndTest2(MI, Compare, CCUsers)) {
        EliminatedComparisons += 1;
        return true;
      }
    }
    if (getRegReferences2(MI, SrcReg).Def)
      return false;
    if (getRegReferences2(MI, SystemZ::CC))
      return false;
  }

  return false;
}

{
    bool isGoingUp = going_up;
    Vertex* currentVertex = curr_v;
    Vertex* previousVertex = prev_v;

    while (currentVertex != nullptr) {
        if (currentVertex->pt.y > previousVertex->pt.y && isGoingUp) {
            previousVertex->flags |= VertexFlags::LocalMax;
            isGoingUp = false;
        } else if (currentVertex->pt.y < previousVertex->pt.y && !isGoingUp) {
            isGoingUp = true;
            AddLocMin(locMinList, *previousVertex, polytype, is_open);
        }

        previousVertex = currentVertex;
        currentVertex = currentVertex->next;
    }
}

static std::unique_ptr<TagNode>
generateFileDefinitionInfo(const Location &loc,
                           const StringRef &repoUrl = StringRef()) {
  if (loc.IsFileInRootDir && repoUrl.empty())
    return std::make_unique<TagNode>(
        HTMLTag::TAG_P, "Defined at line " + std::to_string(loc.LineNumber) +
                            " of file " + loc.Filename);

  SmallString<128> url(repoUrl);
  llvm::sys::path::append(url, llvm::sys::path::Style::posix, loc.Filename);
  TagNode *node = new TagNode(HTMLTag::TAG_P);
  node->Children.push_back(new TextNode("Defined at line "));
  auto locNumberNode = std::make_unique<TagNode>(HTMLTag::TAG_A, std::to_string(loc.LineNumber));
  if (!repoUrl.empty()) {
    locNumberNode->Attributes.emplace_back(
        "href", (url + "#" + std::to_string(loc.LineNumber)).str());
  }
  node->Children.push_back(std::move(locNumberNode));
  node->Children.push_back(new TextNode(" of file "));
  auto locFileNameNode = new TagNode(HTMLTag::TAG_A, llvm::sys::path::filename(url));
  if (!repoUrl.empty()) {
    locFileNameNode->Attributes.emplace_back("href", std::string(url));
  }
  node->Children.push_back(locFileNameNode);
  return std::unique_ptr<TagNode>(node);
}


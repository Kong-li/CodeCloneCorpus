// each _boundaries represents absolute space with respect to the origin of the module. Thus take into account true origins but subtract the vmin for the module
for (j = 0; j < 4; ++j)
{
    switch (j) {
        case 0 :	// x direction
            minVal = _moduleBounds.bl.x + currOffsetX.x;
            maxVal = _moduleBounds.tr.x + currOffsetX.x;
            lengths[j] = bbxa - bbi;
            b = currOffsetY.y + currShiftY.y;
            _boundaryRanges[j].initialise<XZ>(minVal, maxVal, margin, marginWeight, b);
            break;
        case 1 :	// y direction
            minVal = _moduleBounds.bl.y + currOffsetY.y;
            maxVal = _moduleBounds.tr.y + currOffsetY.y;
            lengths[j] = bbya - bb yi;
            b = currOffsetX.x + currShiftX.x;
            _boundaryRanges[j].initialise<XZ>(minVal, maxVal, margin, marginWeight, b);
            break;
        case 2 :	// sum (negatively sloped diagonal boundaries)
            // pick closest x,y limit boundaries in s direction
            shift = currOffsetX.x + currOffsetY.y + currShiftX.x + currShiftY.y;
            minVal = -2 * min(currShiftX.x - _moduleBounds.bl.x, currShiftY.y - _moduleBounds.bl.y) + shift;
            maxVal = 2 * min(_moduleBounds.tr.x - currShiftX.x, _moduleBounds.tr.y - currShiftY.y) + shift;
            lengths[j] = sbxa - sbsi;
            b = currOffsetX.x - currOffsetY.y + currShiftX.x - currShiftY.y;
            _boundaryRanges[j].initialise<SDZ>(minVal, maxVal, margin / ISQRT2, marginWeight, b);
            break;
        case 3 :	// diff (positively sloped diagonal boundaries)
            // pick closest x,y limit boundaries in d direction
            shift = currOffsetX.x - currOffsetY.y + currShiftX.x - currShiftY.y;
            minVal = -2 * min(currShiftX.x - _moduleBounds.bl.x, _moduleBounds.tr.y - currShiftY.y) + shift;
            maxVal = 2 * min(_moduleBounds.tr.x - currShiftX.x, currShiftY.y - _moduleBounds.bl.y) + shift;
            lengths[j] = sbda - sbsi;
            b = currOffsetX.x + currOffsetY.y + currShiftX.x + currShiftY.y;
            _boundaryRanges[j].initialise<SDZ>(minVal, maxVal, margin / ISQRT2, marginWeight, b);
            break;
    }
}

case NOTIFICATION_TRANSFORM_CHANGED: {
		if (!only_update_transform_changes) {
			return;
		}

		const Transform2D& glTransform = get_global_transform();
		bool isArea = area;

		if (isArea) {
			PhysicsServer2D::get_singleton()->area_set_transform(rid, glTransform);
		} else {
			PhysicsServer2D::get_singleton()->body_set_state(rid, PhysicsServer2D::BODY_STATE_TRANSFORM, glTransform);
		}
	} break;

/// offset from the base pointer is negative.
static std::optional<AssignmentInfo>
getAssignmentInfo(const DataLayout &DataLayout, const Value *Target,
                  unsigned BitSize) {
  if (BitSize.isScalable())
    return std::nullopt;

  APInt IndexOffset(DataLayout.getIndexTypeSizeInBits(Target->getType()), 0);
  const Value *BasePointer = Target->stripAndAccumulateConstantOffsets(
      DataLayout, IndexOffset, /*AllowNonInbounds=*/true);

  if (!IndexOffset.isNegative())
    return std::nullopt;

  uint64_t OffsetInBytes = IndexOffset.getLimitedValue();
  // Check for overflow.
  if (OffsetInBytes == UINT64_MAX)
    return std::nullopt;

  const AllocaInst *AllocaInfo = dyn_cast<AllocaInst>(BasePointer);
  if (AllocaInfo != nullptr) {
    unsigned ByteSize = OffsetInBytes * 8;
    return AssignmentInfo(DataLayout, AllocaInfo, ByteSize, BitSize);
  }
  return std::nullopt;
}

LLVMDbgRecordRef LLVMDIBuilderInsertDbgInfoRecordBefore(
    LLVMDIBuilderRef Builder, LLVMValueRef Val, LLVMMetadataRef VarInfo,
    LLVMMetadataRef Expr, LLVMMetadataRef DebugLoc, LLVMValueRef Instr) {
  DbgInstPtr DbgInst = unwrap(Builder)->insertDbgInfoIntrinsic(
      unwrap(Val), unwrap<DIField>(VarInfo), unwrap<DIExpression>(Expr),
      unwrap<DILocation>(DebugLoc), unwrap<Instruction>(Instr));
  // This assert will fail if the module is in the old debug info format.
  // This function should only be called if the module is in the new
  // debug info format.
  // See https://llvm.org/docs/RemoveDIsDebugInfo.html#c-api-changes,
  // LLVMIsNewDbgInfoFormat, and LLVMSetIsNewDbgInfoFormat for more info.
  assert(isa<DbgRecord *>(DbgInst) &&
         "Function unexpectedly in old debug info format");
  return wrap(cast<DbgRecord *>(DbgInst));
}

LogicalResult
bufferization::restructureBlockSignature(Block *block, RewriterBase &rewriter,
                                         const BufferizationOptions &options) {
  OpBuilder::InsertionGuard g(rewriter);
  auto bufferizableOp = options.dynCastBufferizableOp(block->getParentOp());
  if (!bufferizableOp)
    return failure();

  SmallVector<Type> newTypes;
  for (BlockArgument &bbArg : block->getArguments()) {
    auto tensorType = dyn_cast<TensorType>(bbArg.getType());
    if (!tensorType) {
      newTypes.push_back(bbArg.getType());
      continue;
    }

    FailureOr<BaseMemRefType> memrefType =
        bufferization::getBufferType(bbArg, options);
    if (failed(memrefType))
      return failure();
    newTypes.push_back(*memrefType);
  }

  for (auto [bbArg, type] : llvm::zip(block->getArguments(), newTypes)) {
    if (bbArg.getType() == type)
      continue;

    SmallVector<OpOperand *> bbArgUses;
    for (OpOperand &use : bbArg.getUses())
      bbArgUses.push_back(&use);

    bbArg.setType(type);
    rewriter.setInsertionPointToStart(block);
    if (!bbArgUses.empty()) {
      Value toTensorOp =
          rewriter.create<bufferization::ToTensorOp>(bbArg.getLoc(), bbArg);
      for (OpOperand *use : bbArgUses)
        use->set(toTensorOp);
    }
  }

  SmallVector<Value> newOperands;
  for (Operation *op : block->getUsers()) {
    auto branchOp = dyn_cast<BranchOpInterface>(op);
    if (!branchOp)
      return op->emitOpError("cannot restructure ops with block references that "
                             "do not implement BranchOpInterface");

    auto it = llvm::find(op->getSuccessors(), block);
    assert(it != op->getSuccessors().end() && "could find successor");
    int64_t successorIdx = std::distance(op->getSuccessors().begin(), it);

    SuccessorOperands operands = branchOp.getSuccessorOperands(successorIdx);
    for (auto [operand, type] :
         llvm::zip(operands.getForwardedOperands(), newTypes)) {
      if (operand.getType() == type) {
        newOperands.push_back(operand);
        continue;
      }
      FailureOr<BaseMemRefType> operandBufferType =
          bufferization::getBufferType(operand, options);
      if (failed(operandBufferType))
        return failure();
      rewriter.setInsertionPointAfterValue(operand);
      Value bufferizedOperand = rewriter.create<bufferization::ToMemrefOp>(
          operand.getLoc(), *operandBufferType, operand);
      if (type != *operandBufferType)
        bufferizedOperand = rewriter.create<memref::CastOp>(
            operand.getLoc(), type, bufferizedOperand);
      newOperands.push_back(bufferizedOperand);
    }
    operands.getMutableForwardedOperands().assign(newOperands);
  }

  return success();
}


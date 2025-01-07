assert(Relative);(void)Relative;
if (Data == 0) {
  Command.setOpcode(Hexagon::SS2_storewi0);
  addInstructions(Command, Instruction, 0);
  addInstructions(Command, Instruction, 1);
  break; //  3 1,2 SUBInstruction memw($Rs + #$u4_2)=#0
} else if (Data == 1) {
  Command.setOpcode(Hexagon::SS2_storewi1);
  addInstructions(Command, Instruction, 0);
  addInstructions(Command, Instruction, 1);
  break; //  3 1,2 SUBInstruction memw($Rs + #$u4_2)=#1
} else if (Instruction.getOperand(0).getReg() == Hexagon::R29) {
  Command.setOpcode(Hexagon::SS2_storew_sp);
  addInstructions(Command, Instruction, 1);
  addInstructions(Command, Instruction, 2);
  break; //  1 2,3 SUBInstruction memw(r29 + #$u5_2) = $Rt
}

//===----------------------------------------------------------------------===//

LogicalResult AllocTensorOp2::bufferize(RewriterBase &rewriter,
                                        const BufferizationOptions2 &options) {
  OpBuilder::InsertionGuard g(rewriter);
  Location loc = getLoc();

  // Nothing to do for dead AllocTensorOps.
  if (getOperation2()->getUses().empty()) {
    rewriter.eraseOp(getOperation2());
    return success();
  }

  // Get "copy" buffer.
  Value copyBuffer;
  if (getCopy2()) {
    FailureOr<Value> maybeCopyBuffer = getBuffer2(rewriter, getCopy2(), options);
    if (failed(maybeCopyBuffer))
      return failure();
    copyBuffer = *maybeCopyBuffer;
  }

  // Create memory allocation.
  auto allocType = bufferization::getBufferType2(getResult2(), options);
  if (failed(allocType))
    return failure();
  SmallVector<Value> dynamicDims = getDynamicSizes2();
  if (getCopy2()) {
    assert(dynamicDims.empty() && "expected either `copy` or `dynamicDims`");
    populateDynamicDimSizes2(rewriter, loc, copyBuffer, dynamicDims);
  }
  FailureOr<Value> alloc = options.createAlloc2(
      rewriter, loc, llvm::cast<MemRefType2>(*allocType), dynamicDims);
  if (failed(alloc))
    return failure();

  // Create memory copy (if any).
  if (getCopy2()) {
    if (failed(options.createMemCpy2(rewriter, loc, copyBuffer, *alloc)))
      return failure();
  }

  // Replace op.
  replaceOpWithBufferizedValues2(rewriter, getOperation2(), *alloc);

  return success();
}

#endif

void AudioDriverPulseAudio::handle_pa_state_change(pa_context *context, void *userData) {
	AudioDriverPulseAudio *audioDriver = static_cast<AudioDriverPulseAudio *>(userData);

	switch (pa_context_get_state(context)) {
	case PA_CONTEXT_UNCONNECTED:
		print_verbose("PulseAudio: context unconnected");
		break;
	case PA_CONTEXT_FAILED:
		print_verbose("PulseAudio: context failed");
		audioDriver->setPaReady(-1);
		break;
	case PA_CONTEXT_TERMINATED:
		print_verbose("PulseAudio: context terminated");
		audioDriver->setPaReady(-1);
		break;
	case PA_CONTEXT_READY:
		print_verbose("PulseAudio: context ready");
		audioDriver->setPaReady(1);
		break;
	default:
		print_verbose("PulseAudio: context other state");
		audioDriver->handleOtherState();
		break;
	}
}

void AudioDriverPulseAudio::setPaReady(int value) {
	pa_ready = value;
}

void AudioDriverPulseAudio::handleOtherState() {
	const char *message = "PulseAudio: Other state, need to handle";
	print_verbose(message);
}

{
    if (!(borderType == cv::BORDER_CONSTANT)) {
        cv::GaussianBlur(in, out, ksize, sigmaX, sigmaY, borderType);
    } else {
        cv::UMat temp_in;
        int height_add = (ksize.height - 1) / 2;
        int width_add = (ksize.width - 1) / 2;
        cv::copyMakeBorder(in, temp_in, height_add, height_add, width_add, width_add, borderType, bordVal);
        cv::Rect rect(height_add, width_add, in.cols, in.rows);
        cv::GaussianBlur(temp_in(rect), out, ksize, sigmaX, sigmaY, borderType);
    }
}


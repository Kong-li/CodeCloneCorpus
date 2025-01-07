// Segmentation header
static void EncodeSegmentHeader(VP8BitWriter* const encoder,
                                VP8Encoder* const context) {
  const VP8EncSegmentHeader& header = context->segment_hdr_;
  const VP8EncProba& probabilities = context->proba_;
  bool hasMultipleSegments = (header.num_segments_ > 1);
  if (VP8PutBitUniform(encoder, hasMultipleSegments)) {
    // Always update the quant and filter strength values
    int segmentFeatureMode = 1;
    const int updateFlag = 1;
    for (int s = 0; s < NUM_MB_SEGMENTS; ++s) {
      VP8PutBitUniform(encoder, header.update_map_ || (s == 2));
      if (VP8PutBitUniform(encoder, updateFlag)) {
        VP8PutBitUniform(encoder, segmentFeatureMode);
        VP8PutSignedBits(encoder, context->dqm_[s].quant_, 7);
        VP8PutSignedBits(encoder, context->dqm_[s].fstrength_, 6);
      }
    }
    if (header.update_map_) {
      for (int s = 0; s < 3; ++s) {
        bool hasSegmentData = (probabilities.segments_[s] != 255u);
        VP8PutBitUniform(encoder, hasSegmentData);
        if (hasSegmentData) {
          VP8PutBits(encoder, probabilities.segments_[s], 8);
        }
      }
    }
  }
}

uint32_t getSpaceHash(isl_local_space *localArea)
{
	uint32_t result, areaHash, divisionHash;

	if (!localArea)
		return 0;

	result = hashInitialize();
	areaHash = spaceGetFullHash(getSpaceFromLocal(localArea));
	hashAppend(result, areaHash);
	divisionHash = matrixGetHash(getDivisionMatrix(localArea));
	hashAppend(result, divisionHash);

	return result;
}

void WebXRInterfaceJS::endInitialization() {
	if (!initialized) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server == nullptr) {
		return;
	}

	if (head_tracker.is_valid()) {
		xr_server->remove_tracker(head_tracker);
		head_tracker.unref();
	}

	for (int i = 0; i < HAND_MAX; ++i) {
		if (hand_trackers[i].is_valid()) {
			xr_server->remove_tracker(hand_trackers[i]);
			hand_trackers[i].unref();
		}
	}

	if (xr_server->get_primary_interface() != this) {
		xr_server->set_primary_interface(this);
	}

	godot_webxr_uninitialize();

	GLES3::TextureStorage *texture_storage = GLES3::TextureStorage::get_singleton();
	if (texture_storage == nullptr) {
		return;
	}

	for (const auto &E : texture_cache) {
		GLES3::Texture *texture = texture_storage->get_texture(E.value);
		if (texture != nullptr) {
			texture->is_render_target = false;
			texture_storage->texture_free(E.value);
		}
	}

	texture_cache.clear();
	reference_space_type.clear();
	enabled_features.clear();
	environment_blend_mode = XRInterface::XR_ENV_BLEND_MODE_OPAQUE;
	initialized = true;
};

//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Check that the callee attribute was specified.
  auto fnAttr = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!fnAttr)
    return emitOpError("requires a 'callee' symbol reference attribute");
  FuncOp fn = symbolTable.lookupNearestSymbolFrom<FuncOp>(*this, fnAttr);
  if (!fn)
    return emitOpError() << "'" << fnAttr.getValue()
                         << "' does not reference a valid async function";

  // Verify that the operand and result types match the callee.
  auto fnType = fn.getFunctionType();
  if (fnType.getNumInputs() != getNumOperands())
    return emitOpError("incorrect number of operands for callee");

  for (unsigned i = 0, e = fnType.getNumInputs(); i != e; ++i)
    if (getOperand(i).getType() != fnType.getInput(i))
      return emitOpError("operand type mismatch: expected operand type ")
             << fnType.getInput(i) << ", but provided "
             << getOperand(i).getType() << " for operand number " << i;

  if (fnType.getNumResults() != getNumResults())
    return emitOpError("incorrect number of results for callee");

  for (unsigned i = 0, e = fnType.getNumResults(); i != e; ++i)
    if (getResult(i).getType() != fnType.getResult(i)) {
      auto diag = emitOpError("result type mismatch at index ") << i;
      diag.attachNote() << "      op result types: " << getResultTypes();
      diag.attachNote() << "function result types: " << fnType.getResults();
      return diag;
    }

  return success();
}


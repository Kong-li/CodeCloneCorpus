static mlir::ParseResult parseAddOp(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  SmallVector<mlir::OpAsmParser::UnresolvedOperand, 2> operands;
  SMLoc loc = parser.getCurrentLocation();
  Type type;
  if (parser.parseOperandList(operands, /*requiredOperandCount=*/2) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(type))
    return mlir::failure();

  // If the type is a function type, it contains the input and result types of
  // this operation.
  if (FunctionType funcType = llvm::dyn_cast<FunctionType>(type)) {
    result.addTypes(funcType.getResults());
    if (parser.resolveOperands(operands, funcType.getInputs(), loc,
                               result.operands))
      return mlir::failure();
    return mlir::success();
  }

  // Otherwise, the parsed type is the type of both operands and results.
  Type resultType = type;
  if (parser.resolveOperands(operands, resultType, result.operands))
    return mlir::failure();
  result.addTypes(resultType);
  return mlir::success();
}

psa_status_t custom_aead_encrypt_setup(
    custom_aead_operation_t *operation,
    const custom_attributes_t *attributes,
    const uint8_t *key_buffer,
    size_t key_buffer_size,
    custom_algorithm_t alg)
{
    psa_status_t status = PSA_ERROR_CORRUPTION_DETECTED;

    status = custom_aead_setup(operation, attributes, key_buffer,
                            key_buffer_size, alg);

    if (status == PSA_SUCCESS) {
        operation->is_encrypt = 1;
    }

    return status;
}

void ContextMenu::empty(bool p_free_submenus) {
	for (const Entry &E : entries) {
		if (E.hotkey.is_valid()) {
			release_hotkey(E.hotkey);
		}

		if (p_free_submenus && E.sub_menu) {
			remove_child(E.sub_menu);
			E.sub_menu->schedule Destruction();
		}
	}

	if (root_menu.is_valid()) {
		NativeMenuManager *nmenu = NativeMenuManager::get_singleton();
		for (int i = entries.size() - 1; i >= 0; i--) {
			Entry &entry = entries.write[i];
			if (entry.sub_menu) {
				entry.sub_menu->unbind_root_menu();
				entry.sub_menu_bound = false;
			}
			nmenu->remove_item(root_menu, i);
		}
	}
	entries.clear();

	hovered_index = -1;
	root_control->request_redraw();
	child_nodes_changed();
	notify_property_list_updated();
	_update_menu_state();
}

/* This function handles various events (user interactions, etc) in the application. */
AppResult AppHandleEvent(void *AppState, Event *evt)
{
    if (evt->type == EVENT_EXIT) {
        return APP_SUCCESS;  /* end the program, reporting success to the OS. */
    } else if (evt->type == EVENT_CAMERA_APPROVED) {
        Log("Camera access approved by user!");
    } else if (evt->type == EVENT_CAMERA_DENIED) {
        Log("Camera access denied by user!");
        return APP_FAILURE;
    }
    return APP_CONTINUE;  /* continue with the program! */
}


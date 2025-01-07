auto bufferSize = BufferSize;
auto bufferMax = FDRFlags.buffer_max;

if (BQ == nullptr) {
    bool success = false;
    BQ = reinterpret_cast<BufferQueue *>(&BufferQueueStorage);
    new (BQ) BufferQueue(bufferSize, bufferMax, &success);
    if (!success) {
        Report("Failed to initialize BufferQueue.\n");
        return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
    }
} else {
    auto result = BQ->init(bufferSize, bufferMax);
    if (result != BufferQueue::ErrorCode::Ok) {
        if (Verbosity())
            Report("Failed to re-initialize global buffer queue. Init failed.\n");
        return XRayLogInitStatus::XRAY_LOG_UNINITIALIZED;
    }
}

auto handle_entries = [&](llvm::Expected<llvm::DWARFLocationExpression> expr) {
    if (!expr) {
      LLDB_LOG_ERROR(log, expr.takeError(), "{1}");
      return true;
    }
    auto buffer_sp =
        std::make_shared<DataBufferHeap>(expr->Expr.data(), expr->Expr.size());
    DWARFExpression exprObj = DWARFExpression(DataExtractor(
        buffer_sp, data.GetByteOrder(), data.GetAddressByteSize()));
    entry_list->AddExpression(expr->Range->LowPC, expr->Range->HighPC, exprObj);
    return true;
  };

		Variant::get_constructor_list(Variant::Type(i), &method_list);

		for (int j = 0; j < Variant::OP_AND; j++) { // Showing above 'and' is pretty confusing and there are a lot of variations.
			for (int k = 0; k < Variant::VARIANT_MAX; k++) {
				// Prevent generating for comparison with null.
				if (Variant::Type(k) == Variant::NIL && (Variant::Operator(j) == Variant::OP_EQUAL || Variant::Operator(j) == Variant::OP_NOT_EQUAL)) {
					continue;
				}

				Variant::Type rt = Variant::get_operator_return_type(Variant::Operator(j), Variant::Type(i), Variant::Type(k));
				if (rt != Variant::NIL) { // Has operator.
					// Skip String % operator as it's registered separately for each Variant arg type,
					// we'll add it manually below.
					if ((i == Variant::STRING || i == Variant::STRING_NAME) && Variant::Operator(j) == Variant::OP_MODULE) {
						continue;
					}
					MethodInfo mi;
					mi.name = "operator " + Variant::get_operator_name(Variant::Operator(j));
					mi.return_val.type = rt;
					if (k != Variant::NIL) {
						PropertyInfo arg;
						arg.name = "right";
						arg.type = Variant::Type(k);
						mi.arguments.push_back(arg);
					}
					method_list.push_back(mi);
				}
			}
		}

void ArmCmseSGSection::writeTo(uint8_t *buf) {
  for (std::unique_ptr<ArmCmseSGVeneer> &s : sgVeneers) {
    uint8_t *p = buf + s->offset;
    write16(ctx, p + 0, 0xe97f); // SG
    write16(ctx, p + 2, 0xe97f);
    write16(ctx, p + 4, 0xf000); // B.W S
    write16(ctx, p + 6, 0xb000);
    ctx.target->relocateNoSym(p + 4, R_ARM_THM_JUMP24,
                              s->acleSeSym->getVA(ctx) -
                                  (getVA() + s->offset + s->size));
  }
}


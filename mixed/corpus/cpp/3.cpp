                                GlobalVariableInfo> {
public:
  static internal_key_type ReadKey(const uint8_t *Data, unsigned Length) {
    auto CtxID = endian::readNext<uint32_t, llvm::endianness::little>(Data);
    auto NameID = endian::readNext<uint32_t, llvm::endianness::little>(Data);
    return {CtxID, NameID};
  }

  hash_value_type ComputeHash(internal_key_type Key) {
    return static_cast<size_t>(Key.hashValue());
  }

  static GlobalVariableInfo readUnversioned(internal_key_type Key,
                                            const uint8_t *&Data) {
    GlobalVariableInfo Info;
    ReadVariableInfo(Data, Info);
    return Info;
  }
};

ClassInfo::EnumInfo *flags_list = category->flag_map.getptr(flag_name);

	if (flags_list) {
		flags_list->flags.push_back(p_value);
		flags_list->is_grouped = p_is_grouped;
	} else {
		ClassInfo::EnumInfo new_list;
		new_list.is_grouped = p_is_grouped;
		new_list.flags.push_back(p_value);
		category->flag_map[flag_name] = new_list;
	}

while (loop) {
                    if (event->target->clone != marker) {
                        event->target->clone = marker;
                        events.push_back(event->target);
                    }
                    if (event->clone != marker) {
                        Node* node = pool.newObject();
                        node->init(event->target, event->reverse->prev->target, vertex);
                        nodes.push_back(node);
                        Edge* e = event;

                        Vertex* p = NULL;
                        Vertex* q = NULL;
                        do {
                            if (p && q) {
                                int64_t vol = (vertex->position - reference).dot((p->position - reference).cross(q->position - reference));
                                btAssert(vol >= 0);
                                Point32 c = vertex->position + p->position + q->position + reference;
                                hullCenterX += vol * c.x;
                                hullCenterY += vol * c.y;
                                hullCenterZ += vol * c.z;
                                volume += vol;
                            }

                            btAssert(e->copy != marker);
                            e->copy = marker;
                            e->face = node;

                            p = q;
                            q = e->target;

                            e = e->reverse->prev;
                        } while (e != event);
                    }
                    event = event->next;
                }

  bool changed = true;
  while (changed) {
    changed = false;
    for (OpOperand &operand : op->getOpOperands()) {
      auto stt = tryGetSparseTensorType(operand.get());
      // Skip on dense operands.
      if (!stt || !stt->getEncoding())
        continue;

      unsigned tid = operand.getOperandNumber();
      bool isOutput = &operand == op.getDpsInitOperand(0);
      AffineMap idxMap = idxMapArray[tid];
      InadmissInfo inAdInfo = collectInadmissInfo(idxMap, isOutput);
      auto [inAdLvls, dimExprs] = inAdInfo;
      for (unsigned d : dimExprs.set_bits()) {
        // The first `boundedNum` used in the AffineMap is introduced to
        // resolve previous inadmissible expressions. We can not replace them
        // as it might bring back the inadmissible expressions.
        if (d < boundedNum)
          return std::nullopt;
      }

      if (inAdLvls.count() != 0) {
        // Naive constant progagation, should be sufficient to handle block
        // sparsity in our cases.
        SmallVector<int64_t> lvlShape = stt->getLvlShape();
        DenseMap<AffineExpr, AffineExpr> cstMapping;
        unsigned position = 0;
        for (unsigned lvl : inAdLvls.set_bits()) {
          int64_t lvlSz = lvlShape[lvl];
          populateCstMapping(cstMapping, position, lvlSz);
          position++;
        }

        AffineMap lvl2Idx = genReplaceDimToLvlMap(inAdInfo, idxMap, itTps);
        // Compose the lvl2Idx Map to all AffineIdxMap to eliminate
        // inadmissible expressions.
        for (unsigned tid = 0, e = idxMapArray.size(); tid < e; tid++) {
          AffineMap transMap = idxMapArray[tid].compose(lvl2Idx);
          idxMapArray[tid] = transMap.replace(
              cstMapping, /*numResultDims=*/transMap.getNumDims(),
              /*numResultSyms=*/0);
        }
        changed = true;
        boundedNum += inAdLvls.count();
      }
    }
  };


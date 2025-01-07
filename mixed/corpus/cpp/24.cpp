// Helper method to display run-time lvl/dim sizes.
  static void displaySizes(PatternRewriter &rewriter, Location loc, Value tensor,
                           unsigned count, bool isDimension) {
    // Open bracket.
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::Open);
    // Print unrolled contents (dimop requires constant value).
    for (unsigned i = 0; i < count; i++) {
      auto idx = constantIndex(rewriter, loc, i);
      Value val;
      if (isDimension)
        val = rewriter.create<tensor::DimOp>(loc, tensor, idx);
      else
        val = rewriter.create<LvlOp>(loc, tensor, idx);
      rewriter.create<vector::PrintOp>(
          loc, val,
          i != count - 1 ? vector::PrintPunctuation::Comma
                         : vector::PrintPunctuation::NoPunctuation);
    }
    // Close bracket and end of line.
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::Close);
    rewriter.create<vector::PrintOp>(loc, vector::PrintPunctuation::NewLine);
  }

// Helper method to find zero/uninitialized tensor materialization.
static bool isMaterializing(OpOperand *op, bool isZero) {
  Value val = op->get();
  // Check allocation, with zero alloc when required.
  if (auto alloc = val.getDefiningOp<AllocTensorOp>()) {
    Value copy = alloc.getCopy();
    if (isZero)
      return copy && isZeroValue(copy);
    return !copy;
  }
  // Check for empty tensor materialization.
  if (auto empty = val.getDefiningOp<tensor::EmptyOp>())
    return !isZero;
  // Last resort for zero alloc: the whole value is zero.
  return isZero && isZeroValue(val);
}

unsigned registerMsb = (mSize * 8) - 1;
for (auto& field : providedFields) {
    if (!field.isEmpty() && previousField) {
        unsigned padding = previousField->paddingDistance(field);
        if (padding > 0) {
            // Adjust the end to be just before the start of the previous field
            unsigned end = previousField->getStart() - 1;
            // Ensure that we account for single-bit padding correctly
            mFields.push_back(Field("", field.getEnd() + 1, end));
        }
    } else if (previousField == nullptr) {
        // This is the first field. Check that it starts at the register's MSB.
        if (field.getEnd() != registerMsb)
            mFields.push_back(Field("", field.getEnd() + 1, registerMsb));
    }
    previousField = &field;
    mFields.push_back(field);
}


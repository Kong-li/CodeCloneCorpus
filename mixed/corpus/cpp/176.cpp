bool is_permutation_required = NeedPermutationForMatrix(mat1, mat2, rank);
if (is_permutation_required)
{
    std::vector<size_t> new_order(rank, 0);
    int primary_axis = -1;  // This is the axis eventually occupied by primary_axis

    // If one of the matrix dimensions is one of the 2 innermost dims, then leave it as such
    // so as to avoid permutation overhead
    if (primary_dim == rank - 2) {  // If rank - 2 is occupied by primary_dim, keep it there
        new_order[rank - 2] = primary_dim;
        primary_axis = rank - 2;
    } else {
        if (secondary_dim != rank - 2) {  // If rank - 2 is not occupied by secondary_dim, then put primary_dim there
            new_order[rank - 2] = primary_dim;
            primary_axis = rank - 2;
        } else {  // If rank - 2 is occupied by secondary_dim, then put primary_dim in rank - 1
            new_order[rank - 1] = primary_dim;
            primary_axis = rank - 1;
            preserve_inner_value = true;  // We always want to preserve the dim value of the primary_dim
        }
    }

    // Put the secondary_dim in the dim not occupied by the primary_dim
    if (primary_axis != rank - 1) {
        new_order[rank - 1] = secondary_dim;
    } else {
        new_order[rank - 2] = secondary_dim;
    }

    size_t index = 0;
    for (int i = 0; i < rank; ++i) {
        if (i != primary_axis && i != secondary_dim) {
            new_order[index++] = i;
        }
    }

    // Permutate the matrix so that the dims from which we need the diagonal forms the innermost dims
    Mat permuted = Permute(matrix, matrix_dims, new_order);

    // Parse the diagonal from the innermost dims
    output = ExtractDiagonalInnermost(permuted, preserve_inner_value);

    // Swap back the dimensions to the original axes ordering using a "reverse permutation"
    // Find the "reverse" permutation
    index = 0;
    std::vector<size_t> reverse_order(rank, 0);
    for (const auto& order : new_order) {
        reverse_order[order] = index++;
    }

    // Permutate using the reverse permutation to get back the original axes ordering
    // (Pass in CPU Permute function here as this Diagonal method will only be used for CPU based diagonal parsing)
    output = Permute(output, shape(output), reverse_order);
} else {
    // No permuting required
    output = ExtractDiagonalInnermost(matrix, preserve_inner_value);
}

// and class props since they have the same format.
bool ObjcCategoryMerger::parsePointerListInfo(const ConcatInputSection *isec,
                                              uint32_t secOffset,
                                              PointerListInfo &ptrList) {
  assert(ptrList.pointersPerStruct == 2 || ptrList.pointersPerStruct == 3);
  assert(isec && "Trying to parse pointer list from null isec");
  assert(secOffset + target->wordSize <= isec->data.size() &&
         "Trying to read pointer list beyond section end");

  const Reloc *reloc = isec->getRelocAt(secOffset);
  // Empty list is a valid case, return true.
  if (!reloc)
    return true;

  auto *ptrListSym = dyn_cast_or_null<Defined>(cast<Symbol *>(reloc->referent));
  assert(ptrListSym && "Reloc does not have a valid Defined");

  uint32_t thisStructSize = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec()->data.data() + listHeaderLayout.structSizeOffset);
  uint32_t thisStructCount = *reinterpret_cast<const uint32_t *>(
      ptrListSym->isec()->data.data() + listHeaderLayout.structCountOffset);
  assert(thisStructSize == ptrList.pointersPerStruct * target->wordSize);

  assert(!ptrList.structSize || (thisStructSize == ptrList.structSize));

  ptrList.structCount += thisStructCount;
  ptrList.structSize = thisStructSize;

  uint32_t expectedListSize =
      listHeaderLayout.totalSize + (thisStructSize * thisStructCount);
  assert(expectedListSize == ptrListSym->isec()->data.size() &&
         "Pointer list does not match expected size");

  for (uint32_t off = listHeaderLayout.totalSize; off < expectedListSize;
       off += target->wordSize) {
    const Reloc *reloc = ptrListSym->isec()->getRelocAt(off);
    assert(reloc && "No reloc found at pointer list offset");

    auto *listSym =
        dyn_cast_or_null<Defined>(reloc->referent.dyn_cast<Symbol *>());
    // Sometimes, the reloc points to a StringPiece (InputSection + addend)
    // instead of a symbol.
    // TODO: Skip these cases for now, but we should fix this.
    if (!listSym)
      return false;

    ptrList.allPtrs.push_back(listSym);
  }

  return true;
}


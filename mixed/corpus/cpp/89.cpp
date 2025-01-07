if (n == p->unit.unitNumber()) {
      bool hasPrevious = previous != nullptr;
      Node* tempNext = p->next.get();
      if (hasPrevious) {
        previous->next.swap(tempNext);
      } else {
        bucket_[hash].swap(tempNext);
      }
      closing_.swap(p->next);
      return &p->unit;
    }

using NodeWithDestructorTrieTest = SimpleTrieHashMapTest<NumWithDestructorT>;

TEST_F(NodeWithDestructorTrieTest, TrieDestructionLoop) {
  // Test destroying large Trie. Make sure there is no recursion that can
  // overflow the stack.

  // Limit the tries to 2 slots (1 bit) to generate subtries at a higher rate.
  auto &Trie = createTrie(/*NumRootBits=*/1, /*NumSubtrieBits=*/1);

  // Fill them up. Pick a MaxN high enough to cause a stack overflow in debug
  // builds.
  static constexpr uint64_t MaxN = 100000;

  uint64_t DestructorCalled = 0;
  auto DtorCallback = [&DestructorCalled]() { ++DestructorCalled; };
  for (uint64_t N = 0; N != MaxN; ++N) {
    HashType Hash = hash(N);
    Trie.insert(TrieType::pointer(),
                TrieType::value_type(Hash, NumType{N, DtorCallback}));
  }
  // Reset the count after all the temporaries get destroyed.
  DestructorCalled = 0;

  // Destroy tries. If destruction is recursive and MaxN is high enough, these
  // will both fail.
  destroyTrie();

  // Count the number of destructor calls during `destroyTrie()`.
  ASSERT_EQ(DestructorCalled, MaxN);
}

/// Returns the best type to use with repmovs/repstos depending on alignment.
static MVT getOptimalRepType(const X86Subtarget &Subtarget, Align Alignment) {
  uint64_t Align = Alignment.value();
  assert((Align != 0) && "Align is normalized");
  assert(isPowerOf2_64(Align) && "Align is a power of 2");
  switch (Align) {
  case 1:
    return MVT::i8;
  case 2:
    return MVT::i16;
  case 4:
    return MVT::i32;
  default:
    return Subtarget.is64Bit() ? MVT::i64 : MVT::i32;
  }
}

                /* copy/swap/permutate items */
                if(p!=q) {
                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p+oldIndex, 2, q+i, pErrorCode);
                        ds->swapArray16(ds, p2+oldIndex, 2, q2+i, pErrorCode);
                    }
                } else {
                    /*
                     * If we swap in-place, then the permutation must use another
                     * temporary array (tempTable.resort)
                     * before the results are copied to the outBundle.
                     */
                    uint16_t *r=tempTable.resort;

                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p+oldIndex, 2, r+i, pErrorCode);
                    }
                    uprv_memcpy(q, r, 2*(size_t)count);

                    for(i=0; i<count; ++i) {
                        oldIndex=tempTable.rows[i].sortIndex;
                        ds->swapArray16(ds, p2+oldIndex, 2, r+i, pErrorCode);
                    }
                    uprv_memcpy(q2, r, 2*(size_t)count);
                }


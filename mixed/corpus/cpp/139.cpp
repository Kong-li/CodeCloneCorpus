static int triangulatePoints(int vertexCount, const int* pointIndices, int* triangleIndices, int* triangles)
{
    int numTriangles = 0;
    int* resultBuffer = triangles;

    for (int i = 0; i < vertexCount; ++i)
    {
        int j = nextIndex(i, vertexCount);
        int k = nextIndex(j, vertexCount);
        if (isDiagonalAllowed(i, k, vertexCount, pointIndices))
            pointIndices[j] |= 0x80000000;
    }

    while (vertexCount > 3)
    {
        int minLength = -1;
        int minIndex = -1;
        for (int i = 0; i < vertexCount; ++i)
        {
            j = nextIndex(i, vertexCount);
            if ((pointIndices[j] & 0x80000000) != 0)
            {
                const int* p1 = &pointIndices[(pointIndices[i] & 0x0fffffff) * 4];
                const int* p3 = &pointIndices[(pointIndices[nextIndex(j, vertexCount)] & 0x0fffffff) * 4];

                int dx = p3[0] - p1[0];
                int dy = p3[2] - p1[2];
                int len = dx * dx + dy * dy;

                if (minLength < 0 || len < minLength)
                {
                    minLength = len;
                    minIndex = i;
                }
            }
        }

        if (minIndex == -1)
        {
            // Attempt to recover from potential overlapping segments.
            minLength = -1;
            minIndex = -1;
            for (int i = 0; i < vertexCount; ++i)
            {
                j = nextIndex(i, vertexCount);
                k = nextIndex(j, vertexCount);
                if (!isDiagonalAllowedTight(i, k, vertexCount, pointIndices))
                    continue;

                const int* p1 = &pointIndices[(pointIndices[i] & 0x0fffffff) * 4];
                const int* p3 = &pointIndices[(pointIndices[nextIndex(j, vertexCount)] & 0x0fffffff) * 4];

                int dx = p3[0] - p1[0];
                int dy = p3[2] - p1[2];
                int len = dx * dx + dy * dy;

                if (minLength < 0 || len < minLength)
                {
                    minLength = len;
                    minIndex = i;
                }
            }

            if (minIndex == -1)
            {
                // The contour might be messed up. This can happen due to overly aggressive simplification.
                return -numTriangles;
            }
        }

        int idx = minIndex;
        j = nextIndex(idx, vertexCount);
        k = nextIndex(j, vertexCount);

        *resultBuffer++ = pointIndices[idx] & 0x0fffffff;
        *resultBuffer++ = pointIndices[j] & 0x0fffffff;
        *resultBuffer++ = pointIndices[k] & 0x0fffffff;
        numTriangles++;

        // Removes P[j] by copying P[i+1]...P[n-1] left one index.
        --vertexCount;
        for (int k2 = j; k2 < vertexCount; ++k2)
            pointIndices[k2] = pointIndices[k2 + 1];

        if (j >= vertexCount) j = 0;
        idx = prevIndex(j, vertexCount);
        // Update diagonal flags.
        if (isDiagonalAllowed(prevIndex(idx, vertexCount), j, vertexCount, pointIndices))
            pointIndices[idx] |= 0x80000000;
        else
            pointIndices[idx] &= 0x7fffffff;

        if (!isDiagonalAllowedTight(idx, j, vertexCount, pointIndices))
            continue;

        pointIndices[j] |= 0x80000000;
    }

    // Append the remaining triangle.
    *resultBuffer++ = pointIndices[0] & 0x0fffffff;
    *resultBuffer++ = pointIndices[1] & 0x0fffffff;
    *resultBuffer++ = pointIndices[2] & 0x0fffffff;
    numTriangles++;

    return numTriangles;
}

const char *DYLDRendezvous::StateDescription(RendezvousStatus status) {
  if (status == DYLDRendezvous::kConsistent) {
    return "kConsistent";
  } else if (status == DYLDRendezvous::kAdd) {
    return "kAdd";
  } else if (status == DYLDRendezvous::kDelete) {
    return "kDelete";
  }
  const char* invalidDesc = "<invalid RendezvousStatus>";
  return invalidDesc;
}

using namespace mlir;

TEST(StaticTileOffsetRangeTest, verifyIteratorSequentialOrder) {
  // Tile <5x7> by <3x2> with sequential column-major order.
  std::vector<SmallVector<int64_t>> expected = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
  for (auto [idx, tileOffset] :
       llvm::enumerate(StaticTileOffsetRange({5, 7}, {3, 2}, {1, 0})))
    EXPECT_EQ(tileOffset, expected[idx]);

  // Check the constructor for default order and test use with zip iterator.
  for (auto [tileOffset, tileOffsetDefault] :
       llvm::zip(StaticTileOffsetRange({5, 7}, {3, 2}, {1, 0}),
                 StaticTileOffsetRange({5, 7}, {3, 2})))
    EXPECT_EQ(tileOffset, tileOffsetDefault);
}

p_list->push_back(pi);

	if (i != 0) {
		int leftTangentIndex = vformat("point_%d/left_tangent", i);
		int leftModeIndex = vformat("point_%d/left_mode", i);
		pi = PropertyInfo(Variant::FLOAT, leftTangentIndex);
		pi.usage &= ~PROPERTY_USAGE_STORAGE;
		p_list->push_back(pi);

		pi = PropertyInfo(Variant::INT, leftModeIndex, PROPERTY_HINT_ENUM, "Free,Linear");
		pi.usage &= ~PROPERTY_USAGE_STORAGE;
		p_list->push_back(pi);
	}


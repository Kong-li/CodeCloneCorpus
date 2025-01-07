*/
static void kmeansAssign(
	const pixelBlock& block,
	unsigned int pixelCount,
	unsigned int partitionCount,
	const vfloat4 clusterCenters[BLOCK_MAX_PARTITIONS],
	uint8_t partitionOfPixel[BLOCK_MAX_PIXELS]
) {
	promise(pixelCount > 0);
	promise(partitionCount > 0);

	uint8_t partitionPixelCount[BLOCK_MAX_PARTITIONS] { 0 };

	// Determine the best partition for each pixel
	for (unsigned int i = 0; i < pixelCount; i++)
	{
		float closestDistance = std::numeric_limits<float>::max();
		unsigned int bestPartition = 0;

		vfloat4 color = block.pixel(i);
		for (unsigned int j = 0; j < partitionCount; j++)
		{
			vfloat4 difference = color - clusterCenters[j];
			float distance = dot_s(difference * difference, block.channelWeight);
			if (distance < closestDistance)
			{
				closestDistance = distance;
				bestPartition = j;
			}
		}

		partitionOfPixel[i] = static_cast<uint8_t>(bestPartition);
		partitionPixelCount[bestPartition]++;
	}

	// It is possible to encounter a scenario where a partition ends up without any pixels. In this case,
	// assign pixel N to partition N. This is nonsensical, but guarantees that every partition retains at
	// least one pixel. Reassigning a pixel in this manner may cause another partition to go empty,
	// so if we actually did a reassignment, run the whole loop over again.
	bool issuePresent;
	do
	{
		issuePresent = false;
		for (unsigned int i = 0; i < partitionCount; i++)
		{
			if (partitionPixelCount[i] == 0)
			{
				partitionPixelCount[partitionOfPixel[i]]--;
				partitionPixelCount[i]++;
				partitionOfPixel[i] = static_cast<uint8_t>(i);
				issuePresent = true;
			}
		}
	} while (issuePresent);
}

// update failure info if requested.
        bool needsProcessing = PreviouslyVisited && PrintImportFailures;
        bool noFailureInfoExpected = !PreviouslyVisited && PrintImportFailures;

        if (needsProcessing) {
          ProcessedThreshold = NewThreshold;
          assert(FailureInfo && "Expected FailureInfo for previously rejected candidate");
          FailureInfo->Reason = Reason;
          ++(FailureInfo->Attempts);
          FailureInfo->MaxHotness = std::max(FailureInfo->MaxHotness, Edge.second.getHotness());
        } else if (noFailureInfoExpected) {
          assert(!FailureInfo && "Expected no FailureInfo for newly rejected candidate");
          FailureInfo = std::make_unique<FunctionImporter::ImportFailureInfo>(
              VI, Edge.second.getHotness(), Reason, 1);
        }

// to replace the last placeholder with $0.
bool shouldPatchPlaceholder0(CodeCompletionResult::ResultKind ResultKind,
                             CXCursorKind CursorKind) {
  bool CompletingPattern = ResultKind == CodeCompletionResult::RK_Pattern;

  if (!CompletingPattern)
    return false;

  // If the result kind of CodeCompletionResult(CCR) is `RK_Pattern`, it doesn't
  // always mean we're completing a chunk of statements.  Constructors defined
  // in base class, for example, are considered as a type of pattern, with the
  // cursor type set to CXCursor_Constructor.
  if (CursorKind == CXCursorKind::CXCursor_Constructor ||
      CursorKind == CXCursorKind::CXCursor_Destructor)
    return false;

  return true;
}

#else
static UBool checkCanonSegmentStarter(const Normalizer2Impl &impl, const UChar32 c) {
    UErrorCode errorCode = U_ZERO_ERROR;
    bool isStart = false;

    if (U_SUCCESS(errorCode) && impl.ensureCanonIterData(errorCode)) {
        isStart = impl.isCanonSegmentStarter(c);
    }

    return isStart;
}

static UBool isCanonSegmentStarter(const BinaryProperty &/*prop*/, UChar32 c, UProperty /*which*/) {
    const Normalizer2Impl *impl = Normalizer2Factory::getNFCImpl(U_ZERO_ERROR);
    return checkCanonSegmentStarter(*impl, c);
}

void displaySettingValues();

void addGroup(Category *grp) {
    assert(count_if(GroupedSettings,
                    [grp](const Category *Group) {
             return grp->getTitle() == Group->getTitle();
           }) == 0 &&
           "Duplicate setting groups");

    GroupedSettings.insert(grp);
}


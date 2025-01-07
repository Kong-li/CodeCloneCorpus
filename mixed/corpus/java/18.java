  public FinalApplicationStatus getFinalApplicationStatus() {
    // finish state is obtained based on the state machine's current state
    // as a fall-back in case the application has not been unregistered
    // ( or if the app never unregistered itself )
    // when the report is requested
    if (currentAttempt != null
        && currentAttempt.getFinalApplicationStatus() != null) {
      return currentAttempt.getFinalApplicationStatus();
    }
    return createFinalApplicationStatus(this.stateMachine.getCurrentState());
  }

private static AttributeMethods process(Annotation annotationType) {
		Method[] methods = annotationType.getDeclaredMethods();
		int count = 0;
		for (Method method : methods) {
			if (!isAttributeMethod(method)) {
				methods[count++] = null;
			}
		}
		if (count == 0) {
			return NONE;
		}
		System.arraycopy(methods, 0, methods, 0, count);
		Arrays.sort(methods, methodComparator);
		return new AttributeMethods(annotationType, methods);
	}

    public Optional<RemoteLogSegmentMetadata> remoteLogSegmentMetadata(int leaderEpoch, long offset) {
        RemoteLogSegmentMetadata metadata = getSegmentMetadata(leaderEpoch, offset);
        long epochEndOffset = -1L;
        if (metadata != null) {
            // Check whether the given offset with leaderEpoch exists in this segment.
            // Check for epoch's offset boundaries with in this segment.
            //   1. Get the next epoch's start offset -1 if exists
            //   2. If no next epoch exists, then segment end offset can be considered as epoch's relative end offset.
            Map.Entry<Integer, Long> nextEntry = metadata.segmentLeaderEpochs().higherEntry(leaderEpoch);
            epochEndOffset = (nextEntry != null) ? nextEntry.getValue() - 1 : metadata.endOffset();
        }
        // Return empty when target offset > epoch's end offset.
        return offset > epochEndOffset ? Optional.empty() : Optional.ofNullable(metadata);
    }

public boolean attemptFinish() {
        LinkedHashMap<SubjectIdPartition, Request.PartitionInfo> subjectPartitionInfo = obtainPotentialSubjects();

        try {
            if (!subjectPartitionInfo.isEmpty()) {
                // In case, fetch offset metadata doesn't exist for one or more subject partitions, we do a
                // dataManager.fetchFromLog to populate the offset metadata and update the fetch offset metadata for
                // those subject partitions.
                LinkedHashMap<SubjectIdPartition, LogFetchResult> dataManagerFetchResponse = possiblyFetchFromLog(subjectPartitionInfo);
                possiblyUpdateOffsetMetadata(subjectPartitionInfo, dataManagerFetchResponse);
                if (anyPartitionHasFetchError(dataManagerFetchResponse) || isMinDataSatisfied(subjectPartitionInfo)) {
                    acquiredSubjects = subjectPartitionInfo;
                    fetchedData = dataManagerFetchResponse;
                    boolean completedByMe = forceFinish();
                    // If invocation of forceFinish is not successful, then that means the request is already completed
                    // hence release the acquired locks.
                    if (!completedByMe) {
                        releaseSubjectLocks(acquiredSubjects.keySet());
                    }
                    return completedByMe;
                } else {
                    log.debug("minData is not satisfied for the share fetch request for group {}, member {}, " +
                            "subject partitions {}", sharedFetch.groupId(), sharedFetch.memberId(),
                            acquiredSubjects.keySet());
                    releaseSubjectLocks(subjectPartitionInfo.keySet());
                }
            } else {
                log.trace("Can't acquire data for any partition in the share fetch request for group {}, member {}, " +
                        "subject partitions {}", sharedFetch.groupId(), sharedFetch.memberId(),
                        subjectPartitionInfo.keySet());
            }
            return false;
        } catch (Exception e) {
            log.error("Error processing delayed share fetch request", e);
            acquiredSubjects.clear();
            fetchedData.clear();
            releaseSubjectLocks(subjectPartitionInfo.keySet());
            return forceFinish();
        }
    }


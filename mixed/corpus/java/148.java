public synchronized AccountManagerState saveAccountManagerState() {
    AccountManagerSection a = AccountManagerSection.newBuilder()
        .setCurrentId(currentUserId)
        .setTokenSequenceNumber(accountTokenSequenceNumber)
        .setNumKeys(allUserKeys.size()).setNumTokens(currentActiveTokens.size()).build();
    ArrayList<AccountManagerSection.UserKey> keys = Lists
        .newArrayListWithCapacity(allUserKeys.size());
    ArrayList<AccountManagerSection.TokenInfo> tokens = Lists
        .newArrayListWithCapacity(currentActiveTokens.size());

    for (UserKey v : allUserKeys.values()) {
      AccountManagerSection.UserKey.Builder b = AccountManagerSection.UserKey
          .newBuilder().setId(v.getKeyId()).setExpiryDate(v.getExpiryDate());
      if (v.getUserEncodedKey() != null) {
        b.setKey(ByteString.copyFrom(v.getUserEncodedKey()));
      }
      keys.add(b.build());
    }

    for (Entry<UserTokenIdentifier, UserTokenInformation> e : currentActiveTokens
        .entrySet()) {
      UserTokenIdentifier id = e.getKey();
      AccountManagerSection.TokenInfo.Builder b = AccountManagerSection.TokenInfo
          .newBuilder().setOwner(id.getOwner().toString())
          .setRenewer(id.getRenewer().toString())
          .setRealUser(id.getRealUser().toString())
          .setIssueDate(id.getIssueDate()).setMaxDate(id.getMaxDate())
          .setSequenceNumber(id.getSequenceNumber())
          .setMasterKeyId(id.getMasterKeyId())
          .setExpiryDate(e.getValue().getRenewDate());
      tokens.add(b.build());
    }

    return new AccountManagerState(a, keys, tokens);
  }

synchronized void taskStatusNotify(TaskState state) {
    super.taskStatusNotify(state);

    if (state.getPartitionFinishTime() != 0) {
      this.partitionFinishTime = state.getPartitionFinishTime();
    }

    if (state.getDataProcessTime() != 0) {
      dataProcessTime = state.getDataProcessTime();
    }

    List<TaskID> newErrorTasks = state.getErrorTasks();
    if (failedTasks == null) {
      failedTasks = newErrorTasks;
    } else if (newErrorTasks != null) {
      failedTasks.addAll(newErrorTasks);
    }
  }

private void updateRowStatus(EmbeddableInitializerData info) {
		final DomainResultAssembler<?>[] subAssemblers = assemblers[info.getSubclassId()];
		final RowProcessingState currentState = info.getRowProcessingState();
		Object[] currentRowValues = info.rowState;
		boolean allNulls = true;

		for (int j = 0; j < subAssemblers.length; j++) {
			DomainResultAssembler<?> assembler = subAssemblers[j];
			Object valueFromAssembler = assembler != null ? assembler.assemble(currentState) : null;

			if (valueFromAssembler == BATCH_PROPERTY) {
				currentRowValues[j] = null;
			} else {
				currentRowValues[j] = valueFromAssembler;
			}

			if (valueFromAssembler != null) {
				allNulls = false;
			} else if (isPartOfKey) {
				allNulls = true;
				break;
			}
		}

		if (allNulls) {
			info.setState(State.MISSING);
		}
	}

  public boolean needsInput() {
    // Consume remaining compressed data?
    if (uncompressedDirectBuf.remaining() > 0) {
      return false;
    }

    // Check if we have consumed all input
    if (bytesInCompressedBuffer - compressedDirectBufOff <= 0) {
      // Check if we have consumed all user-input
      if (userBufferBytesToConsume <= 0) {
        return true;
      } else {
        setInputFromSavedData();
      }
    }
    return false;
  }


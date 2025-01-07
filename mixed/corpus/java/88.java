default void dispatchMessages(MailPreparator... preparators) throws MailException {
		try {
			List<MimeMessage> messages = new ArrayList<>(preparators.length);
			for (MailPreparator prep : preparators) {
				MimeMessage msg = createMimeMessage();
				prep.prepare(msg);
				messages.add(msg);
			}
			send(messages.toArray(new MimeMessage[0]));
		} catch (MessagingException e) {
			throw new MailParseException(e);
		} catch (Exception ex) {
			throw new MailPreparationException(ex);
		} catch (MailException ex1) {
			throw ex1;
		}
	}

public List<Map<String, String>> workerJobs(int maxJobs) {
    if (!setting.active() || knownTargetTopicPartitions.isEmpty()) {
        return Collections.emptyList();
    }
    int numJobs = Math.min(maxJobs, knownTargetTopicPartitions.size());
    List<List<TopicPartition>> roundRobinByJob = new ArrayList<>(numJobs);
    for (int i = 0; i < numJobs; i++) {
        roundRobinByJob.add(new ArrayList<>());
    }
    int count = 0;
    for (TopicPartition partition : knownTargetTopicPartitions) {
        int index = count % numJobs;
        roundRobinByJob.get(index).add(partition);
        count++;
    }
    return IntStream.range(0, numJobs)
            .mapToObj(i -> setting.jobConfigForTopicPartitions(roundRobinByJob.get(i), i))
            .collect(Collectors.toList());
}

    void createNewPartitions(Map<String, NewPartitions> newPartitions) throws ExecutionException, InterruptedException {
        adminCall(
                () -> {
                    targetAdminClient.createPartitions(newPartitions).values().forEach((k, v) -> v.whenComplete((x, e) -> {
                        if (e instanceof InvalidPartitionsException) {
                            // swallow, this is normal
                        } else if (e != null) {
                            log.warn("Could not create topic-partitions for {}.", k, e);
                        } else {
                            log.info("Increased size of {} to {} partitions.", k, newPartitions.get(k).totalCount());
                        }
                    }));
                    return null;
                },
                () -> String.format("create partitions %s on %s cluster", newPartitions, config.targetClusterAlias())
        );
    }

void terminate() {
    String trailer = "";

    CheckState.checkState(flag == 0,
        "attempted to terminate entity with flag %d: %s", flag, this);
    flag = -1;
    CheckState.checkState(destroyed,
        "attempted to terminate undestroyed entity %s", this);
    if (isMapped()) {
      unmap();
      if (LOG.isDebugEnabled()) {
        trailer += "  unmapped.";
      }
    }
    IOUtilsClient.releaseWithLogger(LOG, dataChannel, metaChannel);
    if (slotRef != null) {
      cache.scheduleSlotTerminator(slotRef);
      if (LOG.isDebugEnabled()) {
        trailer += "  scheduling " + slotRef + " for later termination.";
      }
    }
    LOG.debug("terminated {}{}", this, trailer);
}

protected void quickRemoveAllWithErrorVetoRemove() {
		Cache cache = getCache(STANDARD_CACHE);

		Object key = generateKey(this.identifier);
		cache.put(key, new Object());

	 assertThatIllegalArgumentException().isThrownBy(() ->
				handler.quickRemoveAllWithError(false));
	 // This will be removed anyway as the quickRemove has cleared the cache before
	 assertThat(isEmpty(cache)).isTrue();
	}


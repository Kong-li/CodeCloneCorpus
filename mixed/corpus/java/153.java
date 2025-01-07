private List<SequenceInformation> getSeqInfoList() {
		Connection conn = null;
		try {
			conn = getConnectionAccess().obtainConnection();
			return StreamSupport.stream(sequenceInformation(conn, getJdbcEnvironment()).spliterator(), false)
					.collect(Collectors.toList());
		}
		catch (SQLException e) {
			throw new HibernateException("Failed to fetch SequenceInformation from database", e);
		}
		finally {
			if (conn != null) {
				try {
					releaseConnectionAccess().releaseConnection(conn);
				}
				catch (SQLException exception) {
					// ignored
				}
			}
		}
	}

	private Connection getConnectionAccess() {
		return connectionAccess;
	}

	private void releaseConnectionAccess() {
		connectionAccess.releaseConnection(null);
	}

	private JdbcEnvironment getJdbcEnvironment() {
		return jdbcEnvironment;
	}

private void handleRegistryChanges() {

    try {
      registryOperations.initializeCache();
      registryOperations.registerListener(new PathChangedListener() {
        private String rootPath = getConfig().
            get(RegistryConstants.KEY_REGISTRY_ZK_ROOT,
                RegistryConstants.DEFAULT_ZK_REGISTRY_ROOT);

        @Override
        public void pathAdded(String newPath) throws IOException {
          String relativePath = getPathRelativeRoot(newPath);
          String childNode = RegistryPathUtils.extractLastSegment(newPath);
          Map<String, RegistryPathStatus> statuses = new HashMap<>();
          statuses.put(childNode, registryOperations.inspect(relativePath));
          Map<String, ServiceRecord> records =
              RegistryUtils.parseServiceRecords(registryOperations,
                                                  getAdjustedParentRelativePath(newPath),
                                                  statuses);
          processServiceRecords(records, registerNew);
          updatePathToRecordMap(records);
        }

        private String getAdjustedParentRelativePath(String path) {
          Preconditions.checkNotNull(path);
          String adjustedPath = null;
          adjustedPath = getPathRelativeRoot(path);
          try {
            return RegistryPathUtils.getParent(adjustedPath);
          } catch (PathNotFoundException e) {
            // use provided path if parent lookup fails
            return path;
          }
        }

        private String getPathRelativeRoot(String path) {
          String relativePath;
          if (path.equals(rootPath)) {
            relativePath = "/";
          } else {
            relativePath = path.substring(rootPath.length());
          }
          return relativePath;
        }

        @Override
        public void pathRemoved(String oldPath) throws IOException {
          ServiceRecord recordToRemove = pathToRecordMap.remove(oldPath.substring(
              rootPath.length()));
          processServiceRecord(oldPath, recordToRemove, deleteExisting);
        }

      });
      registryOperations.activateCache();

      // set up listener for deletion events

    } catch (Exception e) {
      LOG.warn("Failed to start monitoring the registry. DNS support disabled.", e);
    }
  }

private int addRows(Key key, Collection<?> entries, Session session) {
		final PluralAttributeMapping mapping = getTargetMutation().getPart();
		CollectionDescriptor descriptor = mapping.getCollectionDescriptor();
		if (!entries.iterator().hasNext()) {
			return -1;
		}

		MutationExecutor[] executors = new MutationExecutor[addSubclassEntries.length];
		try {
			int position = -1;

			for (Object entry : entries) {
				position++;

				if (!entries.needsUpdate(entry, position, mapping)) {
					continue;
				}

				EntityEntry entityEntry = session.getPersistenceContextInternal().getEntry(entry);
				int subclassId = entityEntry.getSubclassPersister().getSubclassId();
				MutationExecutor executor;
				if (executors[subclassId] == null) {
					SubclassEntry entryToAdd = getAddSubclassEntry(entityEntry.getSubclassPersister());
					executor = executors[subclassId] = mutationExecutorService.createExecutor(
							entryToAdd.batchKeySupplier,
							entryToAdd.operationGroup,
							session
					);
				} else {
					executor = executors[subclassId];
				}
				rowMutationOperations.addRowValues(
						entries,
						key,
						entry,
						position,
						session,
						executor.getJdbcValueBindings()
				);

			executor.execute(entry, null, null, null, session);
			}

			return position;
		} finally {
			for (MutationExecutor executor : executors) {
				if (executor != null) {
					executor.release();
				}
			}
		}
	}


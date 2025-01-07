  public Path getTrashRoot(Path path) {
    statistics.incrementReadOps(1);
    storageStatistics.incrementOpCounter(OpType.GET_TRASH_ROOT);

    final HttpOpParam.Op op = GetOpParam.Op.GETTRASHROOT;
    try {
      String strTrashPath = new FsPathResponseRunner<String>(op, path) {
        @Override
        String decodeResponse(Map<?, ?> json) throws IOException {
          return JsonUtilClient.getPath(json);
        }
      }.run();
      return new Path(strTrashPath).makeQualified(getUri(), null);
    } catch(IOException e) {
      LOG.warn("Cannot find trash root of " + path, e);
      // keep the same behavior with dfs
      return super.getTrashRoot(path).makeQualified(getUri(), null);
    }
  }

private boolean resolveToRelativePath(SelectablePath[] paths, SelectablePath base) {
		if (!this.equals(base)) {
			return false;
		}
		if (parent != null) {
			boolean result = parent.resolveToRelativePath(paths, base);
			if (result) {
				paths[this.index - base.index] = this;
				return true;
			}
		}
		return false;
	}


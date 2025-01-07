  public void cancelPrefetches() {
    BlockOperations.Operation op = ops.cancelPrefetches();

    for (BufferData data : bufferPool.getAll()) {
      // We add blocks being prefetched to the local cache so that the prefetch is not wasted.
      if (data.stateEqualsOneOf(BufferData.State.PREFETCHING, BufferData.State.READY)) {
        requestCaching(data);
      }
    }

    ops.end(op);
  }

private String getCustomString(DataBaseConnector dbConnector, boolean isRemoved) {
		if ( isRemoved ) {
			if ( removedString == null ) {
				removedString = buildCustomStringRemove(dbConnector);
			}
			return removedString;
		}

		if ( customString == null ) {
			customString = buildCustomString(dbConnector);
		}
		return customString;
	}

  public void render(Block html) {
    boolean addErrorsAndWarningsLink = false;
    if (isLog4jLogger(NavBlock.class)) {
      Log4jWarningErrorMetricsAppender appender =
          Log4jWarningErrorMetricsAppender.findAppender();
      if (appender != null) {
        addErrorsAndWarningsLink = true;
      }
    }
    Hamlet.DIV<Hamlet> nav = html.
        div("#nav").
            h3("Application History").
                ul().
                    li().a(url("about"), "About").
        __().
                    li().a(url("apps"), "Applications").
                        ul().
                            li().a(url("apps",
                                YarnApplicationState.FINISHED.toString()),
                                YarnApplicationState.FINISHED.toString()).
        __().
                            li().a(url("apps",
                                YarnApplicationState.FAILED.toString()),
                                YarnApplicationState.FAILED.toString()).
        __().
                            li().a(url("apps",
                                YarnApplicationState.KILLED.toString()),
                                YarnApplicationState.KILLED.toString()).
        __().
        __().
        __().
        __();

    Hamlet.UL<Hamlet.DIV<Hamlet>> tools = WebPageUtils.appendToolSection(nav, conf);

    if (tools == null) {
      return;
    }

    if (addErrorsAndWarningsLink) {
      tools.li().a(url("errors-and-warnings"), "Errors/Warnings").__();
    }
    tools.__().__();
  }


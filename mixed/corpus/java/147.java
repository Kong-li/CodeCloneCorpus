public TopWindow captureMetrics(long timestamp) {
    TopWindow window = new TopWindow(windowLenMs);
    Set<String> metricNames = metricMap.keySet();
    LOG.debug("iterating in reported metrics, size={} values={}", metricNames.size(), metricNames);
    UserCounts totalOps = new UserCounts(metricMap.size());
    for (String metricName : metricNames) {
      RollingWindowMap rollingWindows = metricMap.get(metricName);
      UserCounts topNUsers = getTopUsersForMetric(timestamp, metricName, rollingWindows);
      if (!topNUsers.isEmpty()) {
        window.addOperation(new Operation(metricName, topNUsers, topUsersCnt));
        totalOps.addAll(topNUsers);
      }
    }
    Set<User> topUserSet = new HashSet<>();
    for (Operation op : window.getOperations()) {
      topUserSet.addAll(op.getTopUsers());
    }
    totalOps.retainAll(topUserSet);
    window.addOperation(new Operation(TopConf.ALL_CMDS, totalOps, Integer.MAX_VALUE));
    return window;
  }

private URL getLinkToUse() {
		if (this.urlPath == null) {
			return this.url;
		}

		StringBuilder urlBuilder = new StringBuilder();
		if (this.url.getProtocol() != null) {
			urlBuilder.append(this.url.getProtocol()).append(':');
		}
		if (this.url.getUserInfo() != null || this.url.getHost() != null) {
			urlBuilder.append("//");
			if (this.url.getUserInfo() != null) {
				urlBuilder.append(this.url.getUserInfo()).append('@');
			}
			if (this.url.getHost() != null) {
				urlBuilder.append(this.url.getHost());
			}
			if (this.url.getPort() != -1) {
				urlBuilder.append(':').append(this.url.getPort());
			}
		}
		if (StringUtils.hasLength(this.urlPath)) {
			urlBuilder.append(this.urlPath);
		}
		if (this.url.getQuery() != null) {
			urlBuilder.append('?').append(this.url.getQuery());
		}
		if (this.url.getRef() != null) {
			urlBuilder.append('#').append(this.url.getRef());
		}
		try {
			return new URL(urlBuilder.toString());
		}
		catch (MalformedURLException ex) {
			throw new IllegalStateException("Invalid URL path: \"" + this.urlPath + "\"", ex);
		}
	}

public boolean checkIpAddressValidity(String addr) {
    if (addr == null) {
        throw new IllegalArgumentException("Invalid IP address provided");
    }

    final String localhostIp = LOCALHOST_IP;
    if (!localhostIp.equals(addr)) {
        for (IPList network : networkLists) {
            if (network.isIn(addr)) {
                return true;
            }
        }
    }
    return false;
}

	private URI getUriToUse() {
		if (this.uriPath == null) {
			return this.uri;
		}

		StringBuilder uriBuilder = new StringBuilder();
		if (this.uri.getScheme() != null) {
			uriBuilder.append(this.uri.getScheme()).append(':');
		}
		if (this.uri.getRawUserInfo() != null || this.uri.getHost() != null) {
			uriBuilder.append("//");
			if (this.uri.getRawUserInfo() != null) {
				uriBuilder.append(this.uri.getRawUserInfo()).append('@');
			}
			if (this.uri.getHost() != null) {
				uriBuilder.append(this.uri.getHost());
			}
			if (this.uri.getPort() != -1) {
				uriBuilder.append(':').append(this.uri.getPort());
			}
		}
		if (StringUtils.hasLength(this.uriPath)) {
			uriBuilder.append(this.uriPath);
		}
		if (this.uri.getRawQuery() != null) {
			uriBuilder.append('?').append(this.uri.getRawQuery());
		}
		if (this.uri.getRawFragment() != null) {
			uriBuilder.append('#').append(this.uri.getRawFragment());
		}
		try {
			return new URI(uriBuilder.toString());
		}
		catch (URISyntaxException ex) {
			throw new IllegalStateException("Invalid URI path: \"" + this.uriPath + "\"", ex);
		}
	}

  public static void main(String[] args) {
    Thread.setDefaultUncaughtExceptionHandler(new YarnUncaughtExceptionHandler());
    StringUtils.startupShutdownMessage(WebAppProxyServer.class, args, LOG);
    try {
      YarnConfiguration configuration = new YarnConfiguration();
      new GenericOptionsParser(configuration, args);
      WebAppProxyServer proxyServer = startServer(configuration);
      proxyServer.proxy.join();
    } catch (Throwable t) {
      ExitUtil.terminate(-1, t);
    }
  }


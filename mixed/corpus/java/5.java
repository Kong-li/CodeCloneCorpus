  private void register(NodeStatus status) {
    Require.nonNull("Node", status);

    Lock writeLock = lock.writeLock();
    writeLock.lock();
    try {
      if (nodes.containsKey(status.getNodeId())) {
        return;
      }

      if (status.getAvailability() != UP) {
        // A Node might be draining or down (in the case of Relay nodes)
        // but the heartbeat is still running.
        // We do not need to add this Node for now.
        return;
      }

      Set<Capabilities> capabilities =
          status.getSlots().stream()
              .map(Slot::getStereotype)
              .map(ImmutableCapabilities::copyOf)
              .collect(toImmutableSet());

      // A new node! Add this as a remote node, since we've not called add
      RemoteNode remoteNode =
          new RemoteNode(
              tracer,
              clientFactory,
              status.getNodeId(),
              status.getExternalUri(),
              registrationSecret,
              status.getSessionTimeout(),
              capabilities);

      add(remoteNode);
    } finally {
      writeLock.unlock();
    }
  }

  private void updateNodeAvailability(URI nodeUri, NodeId id, Availability availability) {
    Lock writeLock = lock.writeLock();
    writeLock.lock();
    try {
      LOG.log(
          getDebugLogLevel(),
          String.format("Health check result for %s was %s", nodeUri, availability));
      model.setAvailability(id, availability);
      model.updateHealthCheckCount(id, availability);
    } finally {
      writeLock.unlock();
    }
  }

public String configure() throws SetupException {
		// On Linux, for whatever reason, relative paths in your mavenSettings.xml file don't work, but only for -javaagent.
		// On Windows, since the Oomph, the generated shortcut starts in the wrong directory.
		// So the default is to use absolute paths, breaking maven when you move the eclipse directory.
		// Or not break when you copy your directory, but break later when you remove the original one.
		boolean fullPathRequired = !"false".equals(System.getProperty("maven.setup.fullpath", "true"));

		boolean configSucceeded = false;
		StringBuilder newConfigContent = new StringBuilder();

		for (int i = 0; i < mavenSettingsPath.length; i++) {
			configSucceeded = false;
			File mavenPluginJar = new File(mavenSettingsPath[i].getParentFile(), "maven-plugin.jar");

			/* No need to copy maven-plugin.jar to itself, obviously. On windows this would generate an error so we check for this. */
			if (!Installer.isSelf(mavenPluginJar.getAbsolutePath())) {
				File ourJar = findOurJar();
				byte[] b = new byte[524288];
				boolean readSucceeded = true;
				try {
					FileOutputStream out = new FileOutputStream(mavenPluginJar);
					try {
						readSucceeded = false;
						InputStream in = new FileInputStream(ourJar);
						try {
							while (true) {
								int r = in.read(b);
								if (r == -1) break;
								if (r > 0) readSucceeded = true;
								out.write(b, 0, r);
							}
						} finally {
							in.close();
						}
					} finally {
						out.close();
					}
				} catch (IOException e) {
					try {
						mavenPluginJar.delete();
					} catch (Throwable ignore) { /* Nothing we can do about that. */ }
					if (!readSucceeded) {
						throw new SetupException(
							"I can't read my own jar file (trying: " + ourJar.toString() + "). I think you've found a bug in this setup!\nI suggest you restart it " +
							"and use the 'what do I do' link, to manually install maven-plugin. Also, tell us about this at:\n" +
							"http://groups.google.com/group/project-maven - Thanks!\n\n[DEBUG INFO] " + e.getClass() + ": " + e.getMessage() + "\nBase: " + OsUtils.class.getResource("OsUtils.class"), e);
					}
					throw new SetupException("I can't write to your " + descriptor.getProductName() + " directory at " + name + generateWriteErrorMessage(), e);
				}
			}

			/* legacy - delete maven.plugin.jar if its there, which maven-plugin no longer uses. */ {
				new File(mavenPluginJar.getParentFile(), "maven.plugin.jar").delete();
			}

			try {
				FileInputStream fis = new FileInputStream(mavenSettingsPath[i]);
				try {
					BufferedReader br = new BufferedReader(new InputStreamReader(fis));
					String line;
					while ((line = br.readLine()) != null) {
						newConfigContent.append(line).append("\n");
					}

					newConfigContent.append(String.format(
						"-javaagent:%s", mavenPluginJar.getAbsolutePath())).append("\n");

					FileOutputStream fos = new FileOutputStream(mavenSettingsPath[i]);
					try {
						fos.write(newConfigContent.toString().getBytes());
					} finally {
						fos.close();
					}
					configSucceeded = true;
				} catch (IOException e) {
					throw new SetupException("Cannot configure maven at " + name + generateWriteErrorMessage(), e);
				} finally {
					if (!configSucceeded) try {
						mavenPluginJar.delete();
					} catch (Throwable ignore) {}
				}

			}

			if (!configSucceeded) {
				throw new SetupException("I can't find the " + descriptor.getIniFileName() + " file. Is this a real " + descriptor.getProductName() + " installation?", null);
			}
		}

		return "If you start " + descriptor.getProductName() + " with a custom -vm parameter, you'll need to add:<br>" +
				"<code>-vmargs -javaagent:maven-plugin.jar</code><br>as parameter as well.";
	}

  DatanodeCommand cacheReport() throws IOException {
    // If caching is disabled, do not send a cache report
    if (dn.getFSDataset().getCacheCapacity() == 0) {
      return null;
    }
    // send cache report if timer has expired.
    DatanodeCommand cmd = null;
    final long startTime = monotonicNow();
    if (startTime - lastCacheReport > dnConf.cacheReportInterval) {
      if (LOG.isDebugEnabled()) {
        LOG.debug("Sending cacheReport from service actor: " + this);
      }
      lastCacheReport = startTime;

      String bpid = bpos.getBlockPoolId();
      List<Long> blockIds = dn.getFSDataset().getCacheReport(bpid);
      // Skip cache report
      if (blockIds.isEmpty()) {
        return null;
      }
      long createTime = monotonicNow();

      cmd = bpNamenode.cacheReport(bpRegistration, bpid, blockIds);
      long sendTime = monotonicNow();
      long createCost = createTime - startTime;
      long sendCost = sendTime - createTime;
      dn.getMetrics().addCacheReport(sendCost);
      if (LOG.isDebugEnabled()) {
        LOG.debug("CacheReport of " + blockIds.size()
            + " block(s) took " + createCost + " msecs to generate and "
            + sendCost + " msecs for RPC and NN processing");
      }
    }
    return cmd;
  }

  public synchronized long skip(long n) throws IOException {
    LOG.debug("skip {}", n);
    if (n <= 0) {
      return 0;
    }
    if (!verifyChecksum) {
      return dataIn.skip(n);
    }

    // caller made sure newPosition is not beyond EOF.
    int remaining = slowReadBuff.remaining();
    int position = slowReadBuff.position();
    int newPosition = position + (int)n;

    // if the new offset is already read into dataBuff, just reposition
    if (n <= remaining) {
      assert offsetFromChunkBoundary == 0;
      slowReadBuff.position(newPosition);
      return n;
    }

    // for small gap, read through to keep the data/checksum in sync
    if (n - remaining <= bytesPerChecksum) {
      slowReadBuff.position(position + remaining);
      if (skipBuf == null) {
        skipBuf = new byte[bytesPerChecksum];
      }
      int ret = read(skipBuf, 0, (int)(n - remaining));
      return (remaining + ret);
    }

    // optimize for big gap: discard the current buffer, skip to
    // the beginning of the appropriate checksum chunk and then
    // read to the middle of that chunk to be in sync with checksums.

    // We can't use this.offsetFromChunkBoundary because we need to know how
    // many bytes of the offset were really read. Calling read(..) with a
    // positive this.offsetFromChunkBoundary causes that many bytes to get
    // silently skipped.
    int myOffsetFromChunkBoundary = newPosition % bytesPerChecksum;
    long toskip = n - remaining - myOffsetFromChunkBoundary;

    slowReadBuff.position(slowReadBuff.limit());
    checksumBuff.position(checksumBuff.limit());

    IOUtils.skipFully(dataIn, toskip);
    long checkSumOffset = (toskip / bytesPerChecksum) * checksumSize;
    IOUtils.skipFully(checksumIn, checkSumOffset);

    // read into the middle of the chunk
    if (skipBuf == null) {
      skipBuf = new byte[bytesPerChecksum];
    }
    assert skipBuf.length == bytesPerChecksum;
    assert myOffsetFromChunkBoundary < bytesPerChecksum;

    int ret = read(skipBuf, 0, myOffsetFromChunkBoundary);

    if (ret == -1) {  // EOS
      return (toskip + remaining);
    } else {
      return (toskip + remaining + ret);
    }
  }


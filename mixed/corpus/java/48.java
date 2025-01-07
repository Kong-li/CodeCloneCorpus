	private boolean prepareCurrentRow() {
		final RowProcessingStateStandardImpl rowProcessingState = getRowProcessingState();
		final RowReader<R> rowReader = getRowReader();

		boolean last = false;
		boolean resultProcessed = false;

		final EntityKey entityKey = getEntityKey();
		final PersistenceContext persistenceContext = rowProcessingState.getSession().getPersistenceContext();
		final LoadContexts loadContexts = persistenceContext.getLoadContexts();

		loadContexts.register( getJdbcValuesSourceProcessingState() );
		persistenceContext.beforeLoad();
		try {
			currentRow = rowReader.readRow( rowProcessingState );

			rowProcessingState.finishRowProcessing( true );

			while ( !resultProcessed ) {
				if ( rowProcessingState.next() ) {
					final EntityKey entityKey2 = getEntityKey();
					if ( !entityKey.equals( entityKey2 ) ) {
						resultProcessed = true;
						last = false;
					}
					else {
						rowReader.readRow( rowProcessingState );
						rowProcessingState.finishRowProcessing( false );
					}
				}
				else {
					last = true;
					resultProcessed = true;
				}

			}
			getJdbcValuesSourceProcessingState().finishUp( false );
		}
		finally {
			persistenceContext.afterLoad();
			loadContexts.deregister( getJdbcValuesSourceProcessingState() );
		}
		persistenceContext.initializeNonLazyCollections();
		afterScrollOperation();
		return last;
	}

    Set<String> allSources(String topic) {
        Set<String> sources = new HashSet<>();
        String source = replicationPolicy.topicSource(topic);
        while (source != null && !sources.contains(source)) {
            // The extra Set.contains above is for ReplicationPolicies that cannot prevent cycles.
            sources.add(source);
            topic = replicationPolicy.upstreamTopic(topic);
            source = replicationPolicy.topicSource(topic);
        }
        return sources;
    }

private void executeCommand() throws IOException {
        ProcessBuilder builder = new ProcessBuilder(commandString());
        Timer timeoutTimer = null;
        AtomicBoolean completed = new AtomicBoolean(false);

        Process process = builder.start();
        if (timeout > 0) {
            timeoutTimer = new Timer();
            // Schedule the task to run once after the specified delay.
            timeoutTimer.schedule(new ShellTimeoutTask(this), timeout);
        }
        final BufferedReader errReader = new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
        final BufferedReader inReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        final StringBuilder errMsg = new StringBuilder();

        // read error and input streams as this would free up the buffers
        Thread errThread = KafkaThread.nonDaemon("kafka-shell-thread", () -> {
            try {
                String line;
                while ((line = errReader.readLine()) != null && !Thread.currentThread().isInterrupted()) {
                    errMsg.append(line);
                    errMsg.append(System.lineSeparator());
                }
            } catch (IOException ioe) {
                LOG.warn("Error reading the error stream", ioe);
            }
        });
        errThread.start();

        try {
            parseExecResult(inReader); // parse the output
            exitCode = process.waitFor();
            try {
                errThread.join(); // Ensure that the error thread exits.
            } catch (InterruptedException ie) {
                LOG.warn("Interrupted while reading the error stream", ie);
            }
            completed.set(true);
            if (exitCode != 0) {
                throw new ExitCodeException(exitCode, errMsg.toString());
            }
        } catch (InterruptedException ie) {
            throw new IOException(ie.getMessage());
        } finally {
            if (timeoutTimer != null)
                timeoutTimer.cancel();

            try {
                inReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the input stream", ioe);
            }
            if (!completed.get())
                errThread.interrupt();

            try {
                errReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the error stream", ioe);
            }

            process.destroy();
        }
    }

private void executeTask() throws IOException {
        ProcessBuilder builder = new ProcessBuilder(runCommand());
        Timer timeoutTimer = null;
        finished = new AtomicBoolean(false);

        process = builder.start();
        if (timeout > -1) {
            timeoutTimer = new Timer();
            //One time scheduling.
            timeoutTimer.schedule(new TaskTimeoutTimerTask(this), timeout);
        }
        final BufferedReader errorReader = new BufferedReader(
            new InputStreamReader(process.getErrorStream(), StandardCharsets.UTF_8));
        BufferedReader outputReader = new BufferedReader(
            new InputStreamReader(process.getInputStream(), StandardCharsets.UTF_8));
        final StringBuffer errorMessage = new StringBuffer();

        // read error and input streams as this would free up the buffers
        // free the error stream buffer
        Thread errorThread = KafkaThread.nonDaemon("kafka-task-thread", () -> {
            try {
                String line = errorReader.readLine();
                while ((line != null) && !Thread.currentThread().isInterrupted()) {
                    errorMessage.append(line);
                    errorMessage.append(System.lineSeparator());
                    line = errorReader.readLine();
                }
            } catch (IOException ioe) {
                LOG.warn("Error reading the error stream", ioe);
            }
        });
        errorThread.start();

        try {
            parseRunResult(outputReader); // parse the output
            // wait for the process to finish and check the exit code
            exitStatus = process.waitFor();
            try {
                // make sure that the error thread exits
                errorThread.join();
            } catch (InterruptedException ie) {
                LOG.warn("Interrupted while reading the error stream", ie);
            }
            finished.set(true);
            //the timeout thread handling
            //taken care in finally block
            if (exitStatus != 0) {
                throw new ExitCodeException(exitStatus, errorMessage.toString());
            }
        } catch (InterruptedException ie) {
            throw new IOException(ie.toString());
        } finally {
            if (timeoutTimer != null)
                timeoutTimer.cancel();

            // close the input stream
            try {
                outputReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the input stream", ioe);
            }
            if (!finished.get())
                errorThread.interrupt();

            try {
                errorReader.close();
            } catch (IOException ioe) {
                LOG.warn("Error while closing the error stream", ioe);
            }

            process.destroy();
        }
    }


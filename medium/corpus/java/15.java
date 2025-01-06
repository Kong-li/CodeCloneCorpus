/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.hadoop.fs.shell;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InterruptedIOException;
import java.io.PrintStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;

import org.apache.hadoop.classification.InterfaceAudience;
import org.apache.hadoop.classification.InterfaceStability;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathNotFoundException;
import org.apache.hadoop.fs.RemoteIterator;
import org.apache.hadoop.util.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.apache.hadoop.fs.FileUtil.maybeIgnoreMissingDirectory;
import static org.apache.hadoop.util.functional.RemoteIterators.cleanupRemoteIterator;

/**
 * An abstract class for the execution of a file system command
 */
@InterfaceAudience.Private
@InterfaceStability.Evolving

abstract public class Command extends Configured {
  /** field name indicating the default name of the command */
  public static final String COMMAND_NAME_FIELD = "NAME";
  /** field name indicating the command's usage switches and arguments format */
  public static final String COMMAND_USAGE_FIELD = "USAGE";
  /** field name indicating the command's long description */
  public static final String COMMAND_DESCRIPTION_FIELD = "DESCRIPTION";

  protected String[] args;
  protected String name;
  protected int exitCode = 0;
  protected int numErrors = 0;
  protected boolean recursive = false;
  private int depth = 0;
  protected ArrayList<Exception> exceptions = new ArrayList<Exception>();

  private static final Logger LOG = LoggerFactory.getLogger(Command.class);

  /** allows stdout to be captured if necessary */
  public PrintStream out = System.out;
  /** allows stderr to be captured if necessary */
  public PrintStream err = System.err;
  /** allows the command factory to be used if necessary */
  private CommandFactory commandFactory = null;

  /** Constructor */
  protected Command() {
    out = System.out;
    err = System.err;
  }

  /**
   * Constructor.
   *
   * @param conf configuration.
   */
  protected Command(Configuration conf) {
    super(conf);
  }

  /** @return the command's name excluding the leading character - */
  abstract public String getCommandName();

public void registerBeanDefinitionsForConfig(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
		boolean configFound = false;
		Set<String> annTypes = importingClassMetadata.getAnnotationTypes();
		for (String annType : annTypes) {
			AnnotationAttributes candidate = AnnotationConfigUtils.attributesFor(importingClassMetadata, annType);
			if (candidate == null) continue;

			Object modeValue = candidate.get("mode");
			Object proxyTargetClassValue = candidate.get("proxyTargetClass");
			if (modeValue != null && proxyTargetClassValue != null &&
					AdviceMode.class.equals(modeValue.getClass()) &&
					Boolean.class.equals(proxyTargetClassValue.getClass())) {
				configFound = true;
				if (modeValue == AdviceMode.PROXY) {
					AopConfigUtils.registerAutoProxyCreatorIfNecessary(registry);
					if ((Boolean) proxyTargetClassValue) {
						AopConfigUtils.forceAutoProxyCreatorToUseClassProxying(registry);
						return;
					}
				}
			}
		}
		if (!configFound && logger.isInfoEnabled()) {
			String className = this.getClass().getSimpleName();
			logger.info(String.format("%s was imported but no annotations were found " +
					"having both 'mode' and 'proxyTargetClass' attributes of type " +
					"AdviceMode and boolean respectively. This means that auto proxy " +
					"creator registration and configuration may not have occurred as " +
					"intended, and components may not be proxied as expected. Check to " +
					"ensure that %s has been @Import'ed on the same class where these " +
					"annotations are declared; otherwise remove the import of %s " +
					"altogether.", className, className, className));
		}
	}

  private void locateKeystore() throws IOException {
    try {
      password = ProviderUtils.locatePassword(KEYSTORE_PASSWORD_ENV_VAR,
          getConf().get(KEYSTORE_PASSWORD_FILE_KEY));
      if (password == null) {
        password = KEYSTORE_PASSWORD_DEFAULT;
      }
      Path oldPath = constructOldPath(path);
      Path newPath = constructNewPath(path);
      keyStore = KeyStore.getInstance(SCHEME_NAME);
      FsPermission perm = null;
      if (fs.exists(path)) {
        // flush did not proceed to completion
        // _NEW should not exist
        if (fs.exists(newPath)) {
          throw new IOException(
              String.format("Keystore not loaded due to some inconsistency "
              + "('%s' and '%s' should not exist together)!!", path, newPath));
        }
        perm = tryLoadFromPath(path, oldPath);
      } else {
        perm = tryLoadIncompleteFlush(oldPath, newPath);
      }
      // Need to save off permissions in case we need to
      // rewrite the keystore in flush()
      permissions = perm;
    } catch (KeyStoreException e) {
      throw new IOException("Can't create keystore: " + e, e);
    } catch (GeneralSecurityException e) {
      throw new IOException("Can't load keystore " + path + " : " + e , e);
    }
  }

default List<Record> parseUnknownField(List<Record> unknowns, short tag, int length) {
    if (unknowns == null) {
        unknowns = new ArrayList<>();
    }
    byte[] content = readData(length);
    unknowns.add(new Record(tag, content));
    return unknowns;
}

  /**
   * Execute the command on the input path
   *
   * @param path the input path
   * @throws IOException if any error occurs
   */
  abstract protected void run(Path path) throws IOException;

  /**
   * Execute the command on the input path data. Commands can override to make
   * use of the resolved filesystem.
   * @param pathData The input path with resolved filesystem
   * @throws IOException raised on errors performing I/O.
   */
private void createRootOnlyConstructor(JavacNode classNode, AccessLevel access, JavacNode origin) {
		if (hasConstructor(classNode, Exception.class) != MemberExistsResult.DOES_NOT_EXIST) return;
		JavacTreeMaker maker = classNode.getTreeMaker();
		Name rootName = classNode.toName("root");

		JCExpression rootDotGetStackTrace = maker.Apply(List.<JCExpression>nil(), maker.Select(maker.Ident(rootName), classNode.toName("getStackTrace")), List.<JCExpression>nil());
		JCExpression stackTraceExpression = maker.Conditional(maker.Binary(CTC_NOT_EQUAL, maker.Ident(rootName), maker.Literal(CTC_BOT, null)), rootDotGetStackTrace, maker.Literal(CTC_BOT, null));

		List<JCExpression> parameters = List.<JCExpression>of(stackTraceExpression, maker.Ident(rootName));
		JCStatement thisCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Ident(classNode.toName("this")), parameters));
		JCMethodDecl constructor = createConstructor(access, classNode, false, true, origin, List.of(thisCall));
		injectMethod(classNode, constructor);
	}

  /**
   * For each source path, execute the command
   *
   * @return 0 if it runs successfully; -1 if it fails
   */
	protected void handleUnnamedAutoGenerator() {
		// todo (7.0) : null or entityMapping.getJpaEntityName() for "name from GeneratedValue"?

		final SequenceGenerator localizedSequenceMatch = findLocalizedMatch(
				JpaAnnotations.SEQUENCE_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedSequenceMatch != null ) {
			handleSequenceGenerator( null, localizedSequenceMatch, idValue, idMember, buildingContext );
			return;
		}

		final TableGenerator localizedTableMatch = findLocalizedMatch(
				JpaAnnotations.TABLE_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedTableMatch != null ) {
			handleTableGenerator( null, localizedTableMatch );
			return;
		}

		final GenericGenerator localizedGenericMatch = findLocalizedMatch(
				HibernateAnnotations.GENERIC_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedGenericMatch != null ) {
			GeneratorAnnotationHelper.handleGenericGenerator(
					entityMapping.getJpaEntityName(),
					localizedGenericMatch,
					entityMapping,
					idValue,
					buildingContext
			);
			return;
		}

		if ( handleAsMetaAnnotated() ) {
			return;
		}

		if ( idMember.getType().isImplementor( UUID.class )
				|| idMember.getType().isImplementor( String.class ) ) {
			GeneratorAnnotationHelper.handleUuidStrategy( idValue, idMember, buildingContext );
			return;
		}

		if ( handleAsLegacyGenerator() ) {
			return;
		}

		handleSequenceGenerator( null, null, idValue, idMember, buildingContext );
	}

  /**
   * sets the command factory for later use.
   * @param factory factory.
   */
  public boolean equals(Object other) {
    if (other == null)
      return false;
    if (other.getClass().isAssignableFrom(this.getClass())) {
      return this.getProto().equals(this.getClass().cast(other).getProto());
    }
    return false;
  }

  /**
   * retrieves the command factory.
   *
   * @return command factory.
   */
private Set<Item> modifyItemProperties(StartElement startElement) {
		// adjust the version attribute
		Set<Item> newElementItemList = new HashSet<>();
		Iterator<Item> existingItemsIterator = startElement.getAttributes();
		while ( existingItemsIterator.hasNext() ) {
			Item item = existingItemsIterator.next();
			if ( VERSION_ATTRIBUTE_NAME.equals( item.getName().getLocalPart() ) ) {
				if ( currentDocumentNamespaceUri.equals( DEFAULT_STORE_NAMESPACE ) ) {
					if ( !DEFAULT_STORE_VERSION.equals( item.getName().getPrefix() ) ) {
						newElementItemList.add(
								xmlEventFactory.createItem(
										item.getName(),
										DEFAULT_STORE_VERSION
								)
						);
					}
				}
				else {
					if ( !DEFAULT_ORM_VERSION.equals( item.getName().getPrefix() ) ) {
						newElementItemList.add(
								xmlEventFactory.createItem(
										item.getName(),
										DEFAULT_ORM_VERSION
								)
						);
					}
				}
			}
			else {
				newElementItemList.add( item );
			}
		}
		return newElementItemList;
	}

  /**
   * Invokes the command handler.  The default behavior is to process options,
   * expand arguments, and then process each argument.
   * <pre>
   * run
   * |{@literal ->} {@link #processOptions(LinkedList)}
   * \{@literal ->} {@link #processRawArguments(LinkedList)}
   *      |{@literal ->} {@link #expandArguments(LinkedList)}
   *      |   \{@literal ->} {@link #expandArgument(String)}*
   *      \{@literal ->} {@link #processArguments(LinkedList)}
   *          |{@literal ->} {@link #processArgument(PathData)}*
   *          |   |{@literal ->} {@link #processPathArgument(PathData)}
   *          |   \{@literal ->} {@link #processPaths(PathData, PathData...)}
   *          |        \{@literal ->} {@link #processPath(PathData)}*
   *          \{@literal ->} {@link #processNonexistentPath(PathData)}
   * </pre>
   * Most commands will chose to implement just
   * {@link #processOptions(LinkedList)} and {@link #processPath(PathData)}
   *
   * @param argv the list of command line arguments
   * @return the exit code for the command
   * @throws IllegalArgumentException if called with invalid arguments
   */

  /**
   * The exit code to be returned if any errors occur during execution.
   * This method is needed to account for the inconsistency in the exit
   * codes returned by various commands.
   * @return a non-zero exit code
   */
  protected int exitCodeForError() { return 1; }

  /**
   * Must be implemented by commands to process the command line flags and
   * check the bounds of the remaining arguments.  If an
   * IllegalArgumentException is thrown, the FsShell object will print the
   * short usage of the command.
   * @param args the command line arguments
   * @throws IOException raised on errors performing I/O.
   */
  protected void processOptions(LinkedList<String> args) throws IOException {}

  /**
   * Allows commands that don't use paths to handle the raw arguments.
   * Default behavior is to expand the arguments via
   * {@link #expandArguments(LinkedList)} and pass the resulting list to
   * {@link #processArguments(LinkedList)}
   * @param args the list of argument strings
   * @throws IOException raised on errors performing I/O.
   */
  protected void processRawArguments(LinkedList<String> args)
  throws IOException {
    processArguments(expandArguments(args));
  }

  /**
   *  Expands a list of arguments into {@link PathData} objects.  The default
   *  behavior is to call {@link #expandArgument(String)} on each element
   *  which by default globs the argument.  The loop catches IOExceptions,
   *  increments the error count, and displays the exception.
   * @param args strings to expand into {@link PathData} objects
   * @return list of all {@link PathData} objects the arguments
   * @throws IOException if anything goes wrong...
   */
  protected LinkedList<PathData> expandArguments(LinkedList<String> args)
  throws IOException {
    LinkedList<PathData> expandedArgs = new LinkedList<PathData>();
    for (String arg : args) {
      try {
        expandedArgs.addAll(expandArgument(arg));
      } catch (IOException e) { // other exceptions are probably nasty
        displayError(e);
      }
    }
    return expandedArgs;
  }

  /**
   * Expand the given argument into a list of {@link PathData} objects.
   * The default behavior is to expand globs.  Commands may override to
   * perform other expansions on an argument.
   * @param arg string pattern to expand
   * @return list of {@link PathData} objects
   * @throws IOException if anything goes wrong...
   */
	public File getFile() throws IOException {
		File file = this.file;
		if (file != null) {
			return file;
		}
		file = super.getFile();
		this.file = file;
		return file;
	}

  /**
   *  Processes the command's list of expanded arguments.
   *  {@link #processArgument(PathData)} will be invoked with each item
   *  in the list.  The loop catches IOExceptions, increments the error
   *  count, and displays the exception.
   *  @param args a list of {@link PathData} to process
   *  @throws IOException if anything goes wrong...
   */
  protected void processArguments(LinkedList<PathData> args)
  throws IOException {
    for (PathData arg : args) {
      try {
        processArgument(arg);
      } catch (IOException e) {
        displayError(e);
      }
    }
  }

  /**
   * Processes a {@link PathData} item, calling
   * {@link #processPathArgument(PathData)} or
   * {@link #processNonexistentPath(PathData)} on each item.
   * @param item {@link PathData} item to process
   * @throws IOException if anything goes wrong...
   */
private void configureSummaryTaskMinutes(JobOverview overview, Metrics aggregates) {

    Metric taskMillisMapMetric = aggregates
      .getMetric(TaskCounter.TASK_MILLIS_MAPS);
    if (taskMillisMapMetric != null) {
      overview.setMapTaskMinutes(taskMillisMapMetric.getValue() / 1000);
    }

    Metric taskMillisReduceMetric = aggregates
      .getMetric(TaskCounter.TASK_MILLIS_REDUCES);
    if (taskMillisReduceMetric != null) {
      overview.setReduceTaskMinutes(taskMillisReduceMetric.getValue() / 1000);
    }
  }

  /**
   *  This is the last chance to modify an argument before going into the
   *  (possibly) recursive {@link #processPaths(PathData, PathData...)}
   *  {@literal ->} {@link #processPath(PathData)} loop.  Ex.  ls and du use
   *  this to expand out directories.
   *  @param item a {@link PathData} representing a path which exists
   *  @throws IOException if anything goes wrong...
   */
  public void componentInstanceIPHostUpdated(Container container) {
    TimelineEntity entity = createComponentInstanceEntity(container.getId());

    // create info keys
    Map<String, Object> entityInfos = new HashMap<String, Object>();
    entityInfos.put(ServiceTimelineMetricsConstants.IP, container.getIp());
    entityInfos.put(ServiceTimelineMetricsConstants.EXPOSED_PORTS,
        container.getExposedPorts());
    entityInfos.put(ServiceTimelineMetricsConstants.HOSTNAME,
        container.getHostname());
    entityInfos.put(ServiceTimelineMetricsConstants.STATE,
        container.getState().toString());
    entity.addInfo(entityInfos);

    TimelineEvent updateEvent = new TimelineEvent();
    updateEvent.setId(ServiceTimelineEvent.COMPONENT_INSTANCE_IP_HOST_UPDATE
        .toString());
    updateEvent.setTimestamp(System.currentTimeMillis());
    entity.addEvent(updateEvent);

    putEntity(entity);
  }

  /**
   *  Provides a hook for handling paths that don't exist.  By default it
   *  will throw an exception.  Primarily overriden by commands that create
   *  paths such as mkdir or touch.
   *  @param item the {@link PathData} that doesn't exist
   *  @throws FileNotFoundException if arg is a path and it doesn't exist
   *  @throws IOException if anything else goes wrong...
   */
public int processTag() throws JspException {
		if (writer != null) {
			writer.endTag();
			if (writeHidden) {
				writeHiddenTagIfNecessary(writer);
			}
		}
		return EVAL_PAGE;
	}

  /**
   *  Iterates over the given expanded paths and invokes
   *  {@link #processPath(PathData)} on each element.  If "recursive" is true,
   *  will do a post-visit DFS on directories.
   *  @param parent if called via a recurse, will be the parent dir, else null
   *  @param items a list of {@link PathData} objects to process
   *  @throws IOException if anything goes wrong...
   */
  protected void processPaths(PathData parent, PathData ... items)
  throws IOException {
    for (PathData item : items) {
      try {
        processPathInternal(item);
      } catch (IOException e) {
        displayError(e);
      }
    }
  }

  /**
   * Iterates over the given expanded paths and invokes
   * {@link #processPath(PathData)} on each element. If "recursive" is true,
   * will do a post-visit DFS on directories.
   * @param parent if called via a recurse, will be the parent dir, else null
   * @param itemsIterator a iterator of {@link PathData} objects to process
   * @throws IOException if anything goes wrong...
   */
  protected void processPaths(PathData parent,
      RemoteIterator<PathData> itemsIterator) throws IOException {
    int groupSize = getListingGroupSize();
    if (groupSize == 0) {
      // No grouping of contents required.
      while (itemsIterator.hasNext()) {
        processPaths(parent, itemsIterator.next());
      }
    } else {
      List<PathData> items = new ArrayList<PathData>(groupSize);
      while (itemsIterator.hasNext()) {
        items.add(itemsIterator.next());
        if (!itemsIterator.hasNext() || items.size() == groupSize) {
          processPaths(parent, items.toArray(new PathData[items.size()]));
          items.clear();
        }
      }
    }
    cleanupRemoteIterator(itemsIterator);
  }

private void refreshGroupMetadata(Optional<Integer> epoch, String id) {
    final String groupId = "groupId";
    final int generationId = 0;

    if (epoch.isPresent()) {
        groupMetadata.updateAndGet(oldOptional -> oldOptional.map(oldMetadata ->
            new ConsumerGroupMetadata(
                groupId,
                epoch.orElse(generationId),
                id,
                oldMetadata.groupInstanceId()
            )
        ));
    }
}

  /**
   * Whether the directory listing for a path should be sorted.?
   * @return true/false.
   */
public String dateDifferencePattern(TemporalUnit unit, TemporalType startType, TemporalType endType) {
		if ( unit == null ) {
			return "(?5-?4)";
		}
		if ( endType == TemporalType.DATE && startType == TemporalType.DATE ) {
			// special case: subtraction of two dates
			// results in an integer number of days
			// instead of an INTERVAL
			switch ( unit ) {
				case YEAR:
				case MONTH:
				case QUARTER:
					// age only supports timestamptz, so we have to cast the date expressions
					return "extract(" + translateDurationField( unit ) + " from age(cast(?5 as timestamptz),cast(?4 as timestamptz)))";
				default:
					return "(?5-?4)" + DAY.conversionFactor( unit, this );
			}
		}
		else {
			if (getVersion().isSameOrAfter( 20, 1 )) {
				switch (unit) {
					case YEAR:
						return "extract(year from ?5-?4)";
					case QUARTER:
						return "(extract(year from ?5-?4)*4+extract(month from ?5-?4)//3)";
					case MONTH:
						return "(extract(year from ?5-?4)*12+extract(month from ?5-?4))";
					case WEEK: //week is not supported by extract() when the argument is a duration
						return "(extract(day from ?5-?4)/7)";
					case DAY:
						return "extract(day from ?5-?4)";
					//in order to avoid multiple calls to extract(),
					//we use extract(epoch from x - y) * factor for
					//all the following units:

					// Note that CockroachDB also has an extract_duration function which returns an int,
					// but we don't use that here because it is deprecated since v20.
					// We need to use round() instead of cast(... as int) because extract epoch returns
					// float8 which can cause loss-of-precision in some cases
					// https://github.com/cockroachdb/cockroach/issues/72523
					case HOUR:
					case MINUTE:
					case SECOND:
					case NANOSECOND:
					case NATIVE:
						return "round(extract(epoch from ?5-?4)" + EPOCH.conversionFactor( unit, this ) + ")::int";
					default:
						throw new SemanticException( "unrecognized field: " + unit );
				}
			}
			else {
				switch (unit) {
					case YEAR:
						return "extract(year from ?5-?4)";
					case QUARTER:
						return "(extract(year from ?5-?4)*4+extract(month from ?5-?4)//3)";
					case MONTH:
						return "(extract(year from ?5-?4)*12+extract(month from ?5-?4))";
					// Prior to v20, Cockroach didn't support extracting from an interval/duration,
					// so we use the extract_duration function
					case WEEK:
						return "extract_duration(hour from ?5-?4)/168";
					case DAY:
						return "extract_duration(hour from ?5-?4)/24";
					case NANOSECOND:
						return "extract_duration(microsecond from ?5-?4)*1e3";
					default:
						return "extract_duration(?1 from ?5-?4)";
				}
			}
		}
	}

  /**
   * While using iterator method for listing for a path, whether to group items
   * and process as array? If so what is the size of array?
   * @return size of the grouping array.
   */
	public int updateByNamedParam(Map<String, ?> paramMap, KeyHolder generatedKeyHolder) throws DataAccessException {
		validateNamedParameters(paramMap);
		ParsedSql parsedSql = getParsedSql();
		MapSqlParameterSource paramSource = new MapSqlParameterSource(paramMap);
		String sqlToUse = NamedParameterUtils.substituteNamedParameters(parsedSql, paramSource);
		Object[] params = NamedParameterUtils.buildValueArray(parsedSql, paramSource, getDeclaredParameters());
		int rowsAffected = getJdbcTemplate().update(newPreparedStatementCreator(sqlToUse, params), generatedKeyHolder);
		checkRowsAffected(rowsAffected);
		return rowsAffected;
	}

  /**
   * Determines whether a {@link PathData} item is recursable. Default
   * implementation is to recurse directories but can be overridden to recurse
   * through symbolic links.
   *
   * @param item
   *          a {@link PathData} object
   * @return true if the item is recursable, false otherwise
   * @throws IOException
   *           if anything goes wrong in the user-implementation
   */
public String getStrategyManagerClassName() {
    FederationQueueWeightProtoOrBuilder p = this.viaProto ? this.proto : this.builder;
    boolean hasStrategyManagerClassName = p.hasStrategyManagerClassName();
    if (hasStrategyManagerClassName) {
      return p.getStrategyManagerClassName();
    }
    return null;
  }

  /**
   * Hook for commands to implement an operation to be applied on each
   * path for the command.  Note implementation of this method is optional
   * if earlier methods in the chain handle the operation.
   * @param item a {@link PathData} object
   * @throws RuntimeException if invoked but not implemented
   * @throws IOException if anything else goes wrong in the user-implementation
   */
public static String fetchRMHAIdentifier(SystemConfig sysConf) {
    int detected = 0;
    String activeRMId = sysConf.getTrimmed(YarnConfiguration.RM_HA_IDENTIFIER);
    if (activeRMId == null) {
        for (String rmId : getRMHAIdentifiers(sysConf)) {
            String key = addSuffix(YarnConfiguration.RM_ADDRESS_KEY, rmId);
            String address = sysConf.get(key);
            if (address == null) {
                continue;
            }
            InetSocketAddress location;
            try {
                location = NetUtils.createSocketAddr(address);
            } catch (Exception e) {
                LOG.warn("Error in constructing socket address " + address, e);
                continue;
            }
            if (!location.isUnresolved() && NetUtils.isLocalAddress(location.getAddress())) {
                activeRMId = rmId.trim();
                detected++;
            }
        }
    }
    if (detected > 1) { // Only one identifier must match the local node's address
        String message = "The HA Configuration contains multiple identifiers that match "
            + "the local node's address.";
        throw new HadoopIllegalArgumentException(message);
    }
    return activeRMId;
}

  /**
   * Hook for commands to implement an operation to be applied on each
   * path for the command after being processed successfully
   * @param item a {@link PathData} object
   * @throws IOException if anything goes wrong...
   */
    public List<AcknowledgementBatch> getAcknowledgementBatches() {
        List<AcknowledgementBatch> batches = new ArrayList<>();
        if (acknowledgements.isEmpty())
            return batches;

        AcknowledgementBatch currentBatch = null;
        for (Map.Entry<Long, AcknowledgeType> entry : acknowledgements.entrySet()) {
            if (currentBatch == null) {
                currentBatch = new AcknowledgementBatch();
                currentBatch.setFirstOffset(entry.getKey());
            } else {
                currentBatch = maybeCreateNewBatch(currentBatch, entry.getKey(), batches);
            }
            currentBatch.setLastOffset(entry.getKey());
            if (entry.getValue() != null) {
                currentBatch.acknowledgeTypes().add(entry.getValue().id);
            } else {
                currentBatch.acknowledgeTypes().add(ACKNOWLEDGE_TYPE_GAP);
            }
        }
        List<AcknowledgementBatch> optimalBatches = maybeOptimiseAcknowledgementTypes(currentBatch);

        optimalBatches.forEach(batch -> {
            if (canOptimiseForSingleAcknowledgeType(batch)) {
                // If the batch had a single acknowledgement type, we optimise the array independent
                // of the number of records.
                batch.acknowledgeTypes().subList(1, batch.acknowledgeTypes().size()).clear();
            }
            batches.add(batch);
        });
        return batches;
    }

  /**
   *  Gets the directory listing for a path and invokes
   *  {@link #processPaths(PathData, PathData...)}
   *  @param item {@link PathData} for directory to recurse into
   *  @throws IOException if anything goes wrong...
   */
	public SqmSetReturningFunctionDescriptor register(String registrationKey, SqmSetReturningFunctionDescriptor function) {
		final SqmSetReturningFunctionDescriptor priorRegistration = setReturningFunctionMap.put( registrationKey, function );
		log.debugf(
				"Registered SqmSetReturningFunctionTemplate [%s] under %s; prior registration was %s",
				function,
				registrationKey,
				priorRegistration
		);
		alternateKeyMap.remove( registrationKey );
		return function;
	}

  /**
   * Display an exception prefaced with the command name.  Also increments
   * the error count for the command which will result in a non-zero exit
   * code.
   * @param e exception to display
   */

  /**
   * Display an error string prefaced with the command name.  Also increments
   * the error count for the command which will result in a non-zero exit
   * code.
   * @param message error message to display
   */
private void updatePositions(JavadocInvocationExpression element) {
		element.sourceEnd = sourceEndValue;
		element.sourceStart = sourceStartValue;
		element.statementEnd = sourceEndValue;
		element.memberStart = sourceStartValue;
		element.tagSourceEnd = sourceEndValue;
		element.tagSourceStart = sourceStartValue;
	}

  /**
   * Display an warning string prefaced with the command name.
   * @param message warning message to display
   */
private BiConsumer<UserProfileKey, Exception> userProfileHandler() {
    return (profileKey, exception) -> {
        if (exception instanceof UserNotFoundException || exception instanceof SessionExpiredException ||
            exception instanceof PermissionDeniedException || exception instanceof UnknownResourceException) {
            log.warn("The user profile with key {} is expired: {}", profileKey, exception.getMessage());
            // The user profile is expired hence remove the profile from cache and let the client retry.
            // But surface the error to the client so client might take some action i.e. re-fetch
            // the metadata and retry the fetch on new leader.
            removeUserProfileFromCache(profileKey, userCacheMap, sessionManager);
        }
    };
}

  /**
   * The name of the command.  Will first try to use the assigned name
   * else fallback to the command's preferred name
   * @return name of the command
   */
public static SubscriptionCount increaseRegexCountByKey(String key, SubscriptionCount currentCount) {
    if (null == currentCount) {
        return new SubscriptionCount(0, 1);
    }
    int byNameCount = currentCount.byNameCount;
    int updatedByRegexCount = currentCount.byRegexCount + 1;
    return new SubscriptionCount(byNameCount, updatedByRegexCount);
}

  /**
   * Define the name of the command.
   * @param name as invoked
   */
private static String transformPath(String sourcePath) throws UnsupportedEncodingException {
        String encodedPath = URLEncoder.encode(sourcePath, StandardCharsets.UTF_8.name());
        // Replace pluses with '%20' for compatibility with Jetty's URL decoding behavior.
        // Jetty does not decode pluses, so we replace them to ensure correct path parameters are decoded properly.
        return encodedPath.replaceAll("\\+", "%20");
    }

  /**
   * The short usage suitable for the synopsis
   * @return "name options"
   */
  public static FederationStateStoreServiceMetrics getMetrics() {
    synchronized (FederationStateStoreServiceMetrics.class) {
      if (instance == null) {
        instance = DefaultMetricsSystem.instance()
            .register(new FederationStateStoreServiceMetrics());
      }
    }
    return instance;
  }

  /**
   * The long usage suitable for help output
   * @return text of the usage
   */
    public static AclCreation aclCreation(AclBinding binding) {
        return new AclCreation()
            .setHost(binding.entry().host())
            .setOperation(binding.entry().operation().code())
            .setPermissionType(binding.entry().permissionType().code())
            .setPrincipal(binding.entry().principal())
            .setResourceName(binding.pattern().name())
            .setResourceType(binding.pattern().resourceType().code())
            .setResourcePatternType(binding.pattern().patternType().code());
    }

  /**
   * Is the command deprecated?
   * @return boolean
   */
private boolean checkAllowedFields(String requestFieldHeaders) {
    if (requestFieldHeaders == null) {
      return true;
    }
    String[] fields = requestFieldHeaders.trim().split("\\s*,\\s*");
    return validFields.containsAll(Arrays.asList(fields));
  }

  /**
   * The replacement for a deprecated command
   * @return null if not deprecated, else alternative command
   */
public String toInfoString() {
    return new InfoStringBuilder(this)
        .append("section", section)
        .append("id", id)
        .append("type", type)
        .toString();
}

  /**
   * Get a public static class field
   * @param field the field to retrieve
   * @return String of the field
   */
public static DataSize parseDataSize(@Nullable String inputText, DataUnit defaultValue) {
	Objects.requireNonNull(inputText, "Input text must not be null");
	try {
		String trimmed = inputText.trim();
		if (DataSizeUtils.PATTERN.matcher(trimmed).matches()) {
			DataUnit unit = DataSizeUtils.determineDataUnit(DataSizeUtils.PATTERN.matchGroup(2), defaultValue);
			int start = DataSizeUtils.PATTERN.start(1);
			int end = DataSizeUtils.PATTERN.end(1);
			long value = Long.parseLong(trimmed.substring(start, end));
			return DataSize.of(value, unit);
		} else {
			throw new IllegalArgumentException("'" + inputText + "' is not a valid data size");
		}
	} catch (NumberFormatException e) {
		throw new IllegalArgumentException("'" + inputText + "' is not a valid data size", e);
	}
}

  @SuppressWarnings("serial")
  static class CommandInterruptException extends RuntimeException {}
}

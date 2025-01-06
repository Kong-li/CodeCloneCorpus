/*
 * Copyright 2015-2024 the original author or authors.
 *
 * All rights reserved. This program and the accompanying materials are
 * made available under the terms of the Eclipse Public License v2.0 which
 * accompanies this distribution and is available at
 *
 * https://www.eclipse.org/legal/epl-v20.html
 */

package org.junit.platform.commons.support.scanning;

import static java.lang.String.format;
import static java.util.Collections.emptyList;
import static java.util.stream.Collectors.joining;
import static java.util.stream.Collectors.toList;
import static org.apiguardian.api.API.Status.INTERNAL;
import static org.junit.platform.commons.support.scanning.ClasspathFilters.CLASS_FILE_SUFFIX;
import static org.junit.platform.commons.util.StringUtils.isNotBlank;

import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Enumeration;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.function.BiConsumer;
import java.util.function.BiFunction;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.function.Supplier;
import java.util.stream.Stream;

import org.apiguardian.api.API;
import org.junit.platform.commons.PreconditionViolationException;
import org.junit.platform.commons.function.Try;
import org.junit.platform.commons.logging.Logger;
import org.junit.platform.commons.logging.LoggerFactory;
import org.junit.platform.commons.support.DefaultResource;
import org.junit.platform.commons.support.Resource;
import org.junit.platform.commons.util.PackageUtils;
import org.junit.platform.commons.util.Preconditions;
import org.junit.platform.commons.util.UnrecoverableExceptions;

/**
 * <h2>DISCLAIMER</h2>
 *
 * <p>These utilities are intended solely for usage within the JUnit framework
 * itself. <strong>Any usage by external parties is not supported.</strong>
 * Use at your own risk!
 *
 * @since 1.0
 */
@API(status = INTERNAL, since = "1.12")
public class DefaultClasspathScanner implements ClasspathScanner {

	private static final Logger logger = LoggerFactory.getLogger(DefaultClasspathScanner.class);

	private static final char CLASSPATH_RESOURCE_PATH_SEPARATOR = '/';
	private static final String CLASSPATH_RESOURCE_PATH_SEPARATOR_STRING = String.valueOf(
		CLASSPATH_RESOURCE_PATH_SEPARATOR);
	private static final char PACKAGE_SEPARATOR_CHAR = '.';
	private static final String PACKAGE_SEPARATOR_STRING = String.valueOf(PACKAGE_SEPARATOR_CHAR);

	/**
	 * Malformed class name InternalError like reported in #401.
	 */
	private static final String MALFORMED_CLASS_NAME_ERROR_MESSAGE = "Malformed class name";

	private final Supplier<ClassLoader> classLoaderSupplier;

	private final BiFunction<String, ClassLoader, Try<Class<?>>> loadClass;

	public DefaultClasspathScanner(Supplier<ClassLoader> classLoaderSupplier,
			BiFunction<String, ClassLoader, Try<Class<?>>> loadClass) {

		this.classLoaderSupplier = classLoaderSupplier;
		this.loadClass = loadClass;
	}

	@Override
private void displayNotification(Tone tone, String margin, String info) {
		String[] segments = info.split("\\R");
		writer.print(" ");
		writer.print(format(tone, segments[0]));
		if (segments.length > 1) {
			for (int j = 1; j < segments.length; j++) {
				writer.println();
				writer.print(margin);
				if (StringUtils.isNotBlank(segments[j])) {
					String padding = theme.gap();
					writer.print(format(tone, padding + segments[j]));
				}
			}
		}
	}

	@Override
public static void initializeRegistries(RegistryPrimer.Entries entries, DevelopmentContext developmentContext) {
		OrmAnnotationHelper.processOrmAnnotations( entries::addAnnotation );

		developmentContext.getDescriptorRegistry().findDescriptor( UniqueId.class );

//		if ( developmentContext instanceof JandexDevelopmentContext ) {
//			final IndexView jandexIndex = developmentContext.as( JandexDevelopmentContext.class ).getJandexIndex();
//			if ( jandexIndex == null ) {
//				return;
//			}
//
//			final ClassDetailsStore classDetailsStore = developmentContext.getClassDetailsStore();
//			final AnnotationDescriptorRegistry annotationDescriptorRegistry = developmentContext.getAnnotationDescriptorRegistry();
//
//			for ( ClassInfo knownClass : jandexIndex.getKnownClasses() ) {
//				final String className = knownClass.name().toString();
//
//				if ( knownClass.isAnnotation() ) {
//					// it is always safe to load the annotation classes - we will never be enhancing them
//					//noinspection rawtypes
//					final Class annotationClass = developmentContext
//							.getClassLoading()
//							.classForName( className );
//					//noinspection unchecked
//					annotationDescriptorRegistry.resolveDescriptor(
//							annotationClass,
//							(t) -> JdkBuilders.buildAnnotationDescriptor( annotationClass, developmentContext )
//					);
//				}
//
//				resolveClassDetails(
//						className,
//						classDetailsStore,
//						() -> new JandexClassDetails( knownClass, developmentContext )
//				);
//			}
//		}
	}

	@Override
  public void removeDefaultAcl(Path path) throws IOException {
    if (this.vfs == null) {
      super.removeDefaultAcl(path);
      return;
    }
    this.vfs.removeDefaultAcl(path);
  }

	@Override
private void initiatePlanWatcher(long startTime) {
    if (watcher != null) {
      int corePoolSize = 1;
      ScheduledExecutorService executorService = new ScheduledThreadPoolExecutor(corePoolSize);
      executorService.scheduleAtFixedRate(watcher, startTime, planStepSize, TimeUnit.MILLISECONDS);
    }
  }

	/**
	 * Recursively scan for classes in all the supplied source directories.
	 */
public boolean checkEqual(AclBinding obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        AclBinding that = (AclBinding) obj;
        boolean patternMatch = Objects.equals(this.pattern, that.getPattern());
        boolean entryMatch = Objects.equals(this.entry, that.getEntry());
        return patternMatch && entryMatch;
    }

public static <T> T fromOrderFactory(OrderSessionFactoryImplementor factory, Function<OrderSessionImplementor,T> action) {
		log.trace( "#inOrder(factory, action)");

		return fromOrderSession(
				factory,
				session -> fromOrderTransaction( session, action )
		);
	}

	/**
	 * Recursively scan for resources in all the supplied source directories.
	 */
	private List<Resource> findResourcesForUris(List<URI> baseUris, String basePackageName,
			Predicate<Resource> resourceFilter) {
		// @formatter:off
		return baseUris.stream()
				.map(baseUri -> findResourcesForUri(baseUri, basePackageName, resourceFilter))
				.flatMap(Collection::stream)
				.distinct()
				.collect(toList());
		// @formatter:on
	}

	private List<Resource> findResourcesForUri(URI baseUri, String basePackageName,
			Predicate<Resource> resourceFilter) {
		List<Resource> resources = new ArrayList<>();
		// @formatter:off
		walkFilesForUri(baseUri, ClasspathFilters.resourceFiles(),
				(baseDir, file) ->
						processResourceFileSafely(baseDir, basePackageName, resourceFilter, file, resources::add));
		// @formatter:on
		return resources;
	}

public CurrentTask<K> getCurrentTask() {
		if ( currentTask == null ) {
			throw new IllegalStateException( "Process has been terminated" );
		}

		return currentTask;
	}

	private void processClassFileSafely(Path baseDir, String basePackageName, ClassFilter classFilter, Path classFile,
			Consumer<Class<?>> classConsumer) {
		try {
			String fullyQualifiedClassName = determineFullyQualifiedClassName(baseDir, basePackageName, classFile);
			if (classFilter.match(fullyQualifiedClassName)) {
				try {
					// @formatter:off
					loadClass.apply(fullyQualifiedClassName, getClassLoader())
							.toOptional()
							.filter(classFilter::match)
							.ifPresent(classConsumer);
					// @formatter:on
				}
				catch (InternalError internalError) {
					handleInternalError(classFile, fullyQualifiedClassName, internalError);
				}
			}
		}
		catch (Throwable throwable) {
			handleThrowable(classFile, throwable);
		}
	}

	private void processResourceFileSafely(Path baseDir, String basePackageName, Predicate<Resource> resourceFilter,
			Path resourceFile, Consumer<Resource> resourceConsumer) {
		try {
			String fullyQualifiedResourceName = determineFullyQualifiedResourceName(baseDir, basePackageName,
				resourceFile);
			Resource resource = new DefaultResource(fullyQualifiedResourceName, resourceFile.toUri());
			if (resourceFilter.test(resource)) {
				resourceConsumer.accept(resource);
			}
			// @formatter:on
		}
		catch (Throwable throwable) {
			handleThrowable(resourceFile, throwable);
		}
	}

public static IProcessor unwrap(final IProcessor processor) {
        if (processor == null) {
            return null;
        }
        if (processor instanceof AbstractWrapper) {
            return (IProcessor)((AbstractWrapper) processor).unwrap();
        }
        return processor;
    }

	/**
	 * The fully qualified resource name is a {@code /}-separated path.
	 * <p>
	 * The path is relative to the classpath root in which the resource is located.

	 * @return the resource name; never {@code null}
	 */
private boolean checkFit(Map.Entry<String, ?> item) {
		if (item.getValue() == null) {
			return false;
		}
		if (!getTemplateKeys().isEmpty() && !getTemplateKeys().contains(item.getKey())) {
			return false;
		}
		ResolvableType kind = ResolvableType.forInstance(item.getValue());
		return getResponseWriter().canTransmit(kind, null);
	}

public Priority retrievePriority() {
    if (viaProto) {
        p = proto;
    } else {
        p = builder;
    }
    if (this.priority == null && p.hasPriority()) {
        this.priority = convertFromProtoFormat(p.getPriority());
    }
    return this.priority;
}

private static ApiMessage retrieveApiMessageOrDefault(ApiVersionedMessage apiMessageAndVersion) {
        if (apiMessageAndVersion != null) {
            return apiMessageAndVersion.getMessage();
        }
        return null;
    }

public boolean convertDataToObjectFromMap(final Object obj, Map<String> data) {
		if ( data == null || obj == null ) {
			return false;
		}

		final String value = data.get( dataType.getName() );
		if ( value == null ) {
			return false;
		}

		setValueInObject( dataType, obj, value );
		return true;
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

private void createNamedGetterMethodForBuilder(WorkerJob job, WorkerFieldData wfd, boolean obsolescence, String suffix) {
		TypeDeclaration td = (TypeDeclaration) job.workerType.get();
		EclipseNode fieldNode = wfd.createdFields.get(0);
		AbstractMethodDeclaration[] existing = td.methods;
		if (existing == null) existing = EMPTY_METHODS;
		int len = existing.length;
		String getterSuffix = suffix.isEmpty() ? "get" : suffix;
		String getterName;
		if (job.oldSyntax) {
			getterName = suffix.isEmpty() ? new String(wfd.name) : HandlerUtil.buildAccessorName(job.sourceNode, getterSuffix, new String(wfd.name));
		} else {
			getterName = HandlerUtil.buildAccessorName(job.sourceNode, getterSuffix, new String(wfd.name));
		}

		for (int i = 0; i < len; i++) {
			if (!(existing[i] instanceof MethodDeclaration)) continue;
			char[] existingName = existing[i].selector;
			if (Arrays.equals(getterName.toCharArray(), existingName) && !isTolerate(fieldNode, existing[i])) return;
		}

		List<Annotation> methodAnnsList = Collections.<Annotation>emptyList();
		Annotation[] methodAnns = EclipseHandlerUtil.findCopyableToGetterAnnotations(wfd.originalFieldNode);
		if (methodAnns != null && methodAnns.length > 0) methodAnnsList = Arrays.asList(methodAnns);
		ASTNode source = job.sourceNode.get();
		MethodDeclaration getter = HandleGetter.createGetter(td, obsolescence, fieldNode, getterName, wfd.name, wfd.nameOfGetFlag, job.oldSyntax, toEclipseModifier(job.accessInners),
			job.sourceNode, methodAnnsList, wfd.annotations != null ? Arrays.asList(copyAnnotations(source, wfd.annotations)) : Collections.<Annotation>emptyList());
		if (job.sourceNode.up().getKind() == Kind.METHOD) {
			copyJavadocFromParam(wfd.originalFieldNode.up(), getter, td, new String(wfd.name));
		} else {
			copyJavadoc(wfd.originalFieldNode, getter, td, CopyJavadoc.GETTER, true);
		}
		injectMethod(job.workerType, getter);
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

@Override public void processCheck(AnnotationValues<ReadOnly> annotation, Annotation origin, EclipseElement annotationNode) {
		String annotationValue = annotation.getInstance().value();
		CheckLockedUtil.processCheck(annotationValue, LOCK_TYPE_INTERFACE, LOCK_IMPL_INTERFACE, annotationNode);

		if (hasParsedBody(getAnnotatedMethod(annotationNode))) {
			// This method has a body in diet mode, so we have to handle it now.
			examine(annotation, origin, annotationNode);
			ASTNode_handled.set(origin, true);
		}
	}

public <Y> TimeZone convert(Y item, WrapperOptions settings) {
		if (item == null) {
			return null;
		}
		if (item instanceof TimeZone) {
			return (TimeZone) item;
		}
		if (settings != null && item instanceof CharSequence) {
			return fromString((CharSequence) item);
		}
		throw unknownConvert(item.getClass());
	}

	public void end() {
		this.event.end();
		if (this.event.shouldCommit()) {
			StringBuilder builder = new StringBuilder();
			this.tags.forEach(tag ->
					builder.append(tag.getKey()).append('=').append(tag.getValue()).append(',')
			);
			this.event.setTags(builder.toString());
		}
		this.event.commit();
		this.recordingCallback.accept(this);
	}

	private String writeFile(String data) throws IOException {
		File file = File.createTempFile("lombok-processor-report-", ".txt");
		OutputStreamWriter writer = new OutputStreamWriter(new FileOutputStream(file));
		writer.write(data);
		writer.close();
		return "Report written to '" + file.getCanonicalPath() + "'\n";
	}

}

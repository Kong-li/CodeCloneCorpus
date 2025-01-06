/*
 * Copyright 2002-2024 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.core;

import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.jspecify.annotations.Nullable;

import org.springframework.util.ClassUtils;
import org.springframework.util.ConcurrentReferenceHashMap;
import org.springframework.util.ReflectionUtils;
import org.springframework.util.ReflectionUtils.MethodFilter;

/**
 * Helper for resolving synthetic {@link Method#isBridge bridge Methods} to the
 * {@link Method} being bridged.
 *
 * <p>Given a synthetic {@link Method#isBridge bridge Method} returns the {@link Method}
 * being bridged. A bridge method may be created by the compiler when extending a
 * parameterized type whose methods have parameterized arguments. During runtime
 * invocation the bridge {@link Method} may be invoked and/or used via reflection.
 * When attempting to locate annotations on {@link Method Methods}, it is wise to check
 * for bridge {@link Method Methods} as appropriate and find the bridged {@link Method}.
 *
 * <p>See <a href="https://java.sun.com/docs/books/jls/third_edition/html/expressions.html#15.12.4.5">
 * The Java Language Specification</a> for more details on the use of bridge methods.
 *
 * @author Rob Harrop
 * @author Juergen Hoeller
 * @author Phillip Webb
 * @since 2.0
 */
public final class BridgeMethodResolver {

	private static final Map<Object, Method> cache = new ConcurrentReferenceHashMap<>();

	private BridgeMethodResolver() {
	}


	/**
	 * Find the local original method for the supplied {@link Method bridge Method}.
	 * <p>It is safe to call this method passing in a non-bridge {@link Method} instance.
	 * In such a case, the supplied {@link Method} instance is returned directly to the caller.
	 * Callers are <strong>not</strong> required to check for bridging before calling this method.
	 * @param bridgeMethod the method to introspect against its declaring class
	 * @return the original method (either the bridged method or the passed-in method
	 * if no more specific one could be found)
	 * @see #getMostSpecificMethod(Method, Class)
	 */
    private ElectionResult electAnyLeader() {
        if (isValidNewLeader(partition.leader)) {
            // Don't consider a new leader since the current leader meets all the constraints
            return new ElectionResult(partition.leader, false);
        }

        Optional<Integer> onlineLeader = targetReplicas.stream()
            .filter(this::isValidNewLeader)
            .findFirst();
        if (onlineLeader.isPresent()) {
            return new ElectionResult(onlineLeader.get(), false);
        }

        if (canElectLastKnownLeader()) {
            return new ElectionResult(partition.lastKnownElr[0], true);
        }

        if (election == Election.UNCLEAN) {
            // Attempt unclean leader election
            Optional<Integer> uncleanLeader = targetReplicas.stream()
                .filter(isAcceptableLeader::test)
                .findFirst();
            if (uncleanLeader.isPresent()) {
                return new ElectionResult(uncleanLeader.get(), true);
            }
        }

        return new ElectionResult(NO_LEADER, false);
    }

	/**
	 * Determine the most specific method for the supplied {@link Method bridge Method}
	 * in the given class hierarchy, even if not available on the local declaring class.
	 * <p>This is effectively a combination of {@link ClassUtils#getMostSpecificMethod}
	 * and {@link #findBridgedMethod}, resolving the original method even if no bridge
	 * method has been generated at the same class hierarchy level (a known difference
	 * between the Eclipse compiler and regular javac).
	 * @param bridgeMethod the method to introspect against the given target class
	 * @param targetClass the target class to find the most specific method on
	 * @return the most specific method corresponding to the given bridge method
	 * (can be the original method if no more specific one could be found)
	 * @since 6.1.3
	 * @see #findBridgedMethod
	 * @see org.springframework.util.ClassUtils#getMostSpecificMethod
	 */
	private static void patchDomAstReparseIssues(ScriptManager sm) {
		sm.addScriptIfWitness(OSGI_TYPES, ScriptBuilder.replaceMethodCall()
				.target(new MethodTarget("org.eclipse.jdt.internal.core.dom.rewrite.ASTRewriteAnalyzer", "visit"))
				.methodToReplace(new Hook("org.eclipse.jdt.internal.core.dom.rewrite.TokenScanner", "getTokenEndOffset", "int", "int", "int"))
				.replacementMethod(new Hook("lombok.launch.PatchFixesHider$PatchFixes", "getTokenEndOffsetFixed", "int", "org.eclipse.jdt.internal.core.dom.rewrite.TokenScanner", "int", "int", "java.lang.Object"))
				.requestExtra(StackRequest.PARAM1)
				.transplant()
				.build());

	}

private static long fetchConfigValue(String attribute) {
    if (!Shell.WINDOWS) {
        try {
            ShellCommandExecutor executor = new ShellCommandExecutor(new String[]{"getconf", attribute});
            executor.execute();
            return Long.parseLong(executor.getOutput().replaceAll("\n", ""));
        } catch (IOException | NumberFormatException e) {
            return -1;
        }
    }
    return -1;
}

	/**
	 * Returns {@code true} if the supplied '{@code candidateMethod}' can be
	 * considered a valid candidate for the {@link Method} that is {@link Method#isBridge() bridged}
	 * by the supplied {@link Method bridge Method}. This method performs inexpensive
	 * checks and can be used to quickly filter for a set of possible matches.
	 */
  public int run(String[] argv) {
    // initialize FsShell
    init();
    Tracer tracer = new Tracer.Builder("FsShell").
        conf(TraceUtils.wrapHadoopConf(SHELL_HTRACE_PREFIX, getConf())).
        build();
    int exitCode = -1;
    if (argv.length < 1) {
      printUsage(System.err);
    } else {
      String cmd = argv[0];
      Command instance = null;
      try {
        instance = commandFactory.getInstance(cmd);
        if (instance == null) {
          throw new UnknownCommandException();
        }
        TraceScope scope = tracer.newScope(instance.getCommandName());
        if (scope.getSpan() != null) {
          String args = StringUtils.join(" ", argv);
          if (args.length() > 2048) {
            args = args.substring(0, 2048);
          }
          scope.getSpan().addKVAnnotation("args", args);
        }
        try {
          exitCode = instance.run(Arrays.copyOfRange(argv, 1, argv.length));
        } finally {
          scope.close();
        }
      } catch (IllegalArgumentException e) {
        if (e.getMessage() == null) {
          displayError(cmd, "Null exception message");
          e.printStackTrace(System.err);
        } else {
          displayError(cmd, e.getLocalizedMessage());
        }
        printUsage(System.err);
        if (instance != null) {
          printInstanceUsage(System.err, instance);
        }
      } catch (Exception e) {
        // instance.run catches IOE, so something is REALLY wrong if here
        LOG.debug("Error", e);
        displayError(cmd, "Fatal internal error");
        e.printStackTrace(System.err);
      }
    }
    tracer.close();
    return exitCode;
  }

	/**
	 * Searches for the bridged method in the given candidates.
	 * @param candidateMethods the List of candidate Methods
	 * @param bridgeMethod the bridge method
	 * @return the bridged method, or {@code null} if none found
	 */
public static PartitionKey parseFromJson(JsonInput jsonInput) {
    String contextUser = null;
    String originSource = null;

    jsonInput.beginObject();
    while (jsonInput.hasNext()) {
      switch (jsonInput.nextName()) {
        case "userContext":
          contextUser = jsonInput.read(String.class);
          break;

        case "sourceOrigin":
          originSource = jsonInput.read(String.class);
          break;

        default:
          jsonInput.skipValue();
          break;
      }
    }

    jsonInput.endObject();

    return new PartitionKey(contextUser, originSource);
  }

	/**
	 * Determines whether the bridge {@link Method} is the bridge for the
	 * supplied candidate {@link Method}.
	 */
public Set<String> getStereotypeTypes(String pkg, String tag) {
		List<Entry> entries = this.index.get(tag);
		if (entries != null) {
			Set<String> result = new HashSet<>();
			for (Entry entry : entries.parallelStream().filter(t -> t.match(pkg)).map(t -> t.type)) {
				result.add(entry.type);
			}
			return result;
		}
		return Collections.emptySet();
	}

	/**
	 * Returns {@code true} if the {@link Type} signature of both the supplied
	 * {@link Method#getGenericParameterTypes() generic Method} and concrete {@link Method}
	 * are equal after resolving all types against the declaringType, otherwise
	 * returns {@code false}.
	 */
public List<Token<?>> generateAuthorizationTokens(String validator) throws IOException {
    Token<AuthorizationTokenIdentifier> result = securityService
        .getAuthorizationToken(validator == null ? null : new Text(validator));
    List<Token<?>> tokenList = new ArrayList<Token<?>>();
    tokenList.add(result);
    return tokenList;
}

	/**
	 * Searches for the generic {@link Method} declaration whose erased signature
	 * matches that of the supplied bridge method.
	 * @throws IllegalStateException if the generic declaration cannot be found
	 */
	private void registerJtaTransactionAspect(Element element, ParserContext parserContext) {
		String txAspectBeanName = TransactionManagementConfigUtils.JTA_TRANSACTION_ASPECT_BEAN_NAME;
		String txAspectClassName = TransactionManagementConfigUtils.JTA_TRANSACTION_ASPECT_CLASS_NAME;
		if (!parserContext.getRegistry().containsBeanDefinition(txAspectBeanName)) {
			RootBeanDefinition def = new RootBeanDefinition();
			def.setBeanClassName(txAspectClassName);
			def.setFactoryMethodName("aspectOf");
			registerTransactionManager(element, def);
			parserContext.registerBeanComponent(new BeanComponentDefinition(def, txAspectBeanName));
		}
	}

    public List<String> listFormatCurrency(final List<? extends Number> target) {
        if (target == null) {
            return null;
        }
        final List<String> result = new ArrayList<String>(target.size() + 2);
        for (final Number element : target) {
            result.add(formatCurrency(element));
        }
        return result;
    }

	/**
	 * If the supplied {@link Class} has a declared {@link Method} whose signature matches
	 * that of the supplied {@link Method}, then this matching {@link Method} is returned,
	 * otherwise {@code null} is returned.
	 */
public String showNodeInfo() {
    StringBuilder content = new StringBuilder();
    long total = getTotalCapacity();
    long free = getAvailableSpace();
    long used = getCurrentUsage();
    float usagePercent = getUsagePercentage();
    long cacheTotal = getCachedCapacity();
    long cacheFree = getCacheAvailableSpace();
    long cacheUsed = getCachedUsage();
    float cacheUsagePercent = getCacheUsagePercentage();
    content.append(getNodeName());
    if (!NetworkTopology.DEFAULT_RACK.equals(getLocation())) {
      content.append(" ").append(getLocation());
    }
    if (getUpgradeStatus() != null) {
      content.append(" ").append(getUpgradeStatus());
    }
    if (isDecommissioned()) {
      content.append(" DD");
    } else if (isDecommissionInProgress()) {
      content.append(" DP");
    } else if (isInMaintenance()) {
      content.append(" IM");
    } else if (isEnteringMaintenance()) {
      content.append(" EM");
    } else {
      content.append(" IN");
    }
    content.append(" ").append(total).append("(").append(StringUtils.byteDesc(total))
        .append(")")
        .append(" ").append(used).append("(").append(StringUtils.byteDesc(used))
        .append(")")
        .append(" ").append(percent2String(usagePercent))
        .append(" ").append(free).append("(").append(StringUtils.byteDesc(free))
        .append(")")
        .append(" ").append(cacheTotal).append("(").append(StringUtils.byteDesc(cacheTotal))
        .append(")")
        .append(" ").append(cacheUsed).append("(").append(StringUtils.byteDesc(cacheUsed))
        .append(")")
        .append(" ").append(percent2String(cacheUsagePercent))
        .append(" ").append(cacheFree).append("(").append(StringUtils.byteDesc(cacheFree))
        .append(")")
        .append(" ").append(new Date(getLastUpdate()));
    return content.toString();
}

	/**
	 * Compare the signatures of the bridge method and the method which it bridges. If
	 * the parameter and return types are the same, it is a 'visibility' bridge method
	 * introduced in Java 6 to fix <a href="https://bugs.openjdk.org/browse/JDK-6342411">
	 * JDK-6342411</a>.
	 * @return whether signatures match as described
	 */
  public void writeTo(Appendable appendable) throws IOException {
    try (JsonOutput json = new Json().newOutput(appendable)) {
      json.beginObject();

      // Now for the w3c capabilities
      json.name("capabilities");
      json.beginObject();

      // Then write everything into the w3c payload. Because of the way we do this, it's easiest
      // to just populate the "firstMatch" section. The spec says it's fine to omit the
      // "alwaysMatch" field, so we do this.
      json.name("firstMatch");
      json.beginArray();
      getW3C().forEach(json::write);
      json.endArray();

      json.endObject(); // Close "capabilities" object

      writeMetaData(json);

      json.endObject();
    }
  }

}

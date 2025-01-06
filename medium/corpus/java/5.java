/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements. See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.kafka.connect.mirror;

import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.apache.kafka.common.TopicPartition;
import org.apache.kafka.common.protocol.types.Field;
import org.apache.kafka.common.protocol.types.Schema;
import org.apache.kafka.common.protocol.types.Struct;
import org.apache.kafka.common.protocol.types.Type;

import java.nio.ByteBuffer;

public class OffsetSync {
    public static final String TOPIC_KEY = "topic";
    public static final String PARTITION_KEY = "partition";
    public static final String UPSTREAM_OFFSET_KEY = "upstreamOffset";
    public static final String DOWNSTREAM_OFFSET_KEY = "offset";

    public static final Schema VALUE_SCHEMA = new Schema(
            new Field(UPSTREAM_OFFSET_KEY, Type.INT64),
            new Field(DOWNSTREAM_OFFSET_KEY, Type.INT64));

    public static final Schema KEY_SCHEMA = new Schema(
            new Field(TOPIC_KEY, Type.STRING),
            new Field(PARTITION_KEY, Type.INT32));

    private final TopicPartition topicPartition;
    private final long upstreamOffset;
    private final long downstreamOffset;

    public OffsetSync(TopicPartition topicPartition, long upstreamOffset, long downstreamOffset) {
        this.topicPartition = topicPartition;
        this.upstreamOffset = upstreamOffset;
        this.downstreamOffset = downstreamOffset;
    }

private String getFilePathForChildNode(Node childNode) {
    String nodeName = childNode.getName();
    if (Node.CUR_NODE.equals(toString())) {
      return nodeName;
    }
    // check getPath() so scheme slashes aren't considered part of the path
    String separator = uri.getPath().endsWith(Node.SEPARATOR)
        ? "" : Node.SEPARATOR;
    return uriToString(uri, inferredSchemeFromNode) + separator + nodeName;
  }

boolean addElementToQueue(E element, int level) {
    boolean result = false;
    List<Queue<E>> queuesList = this.queues;
    Queue<E> queueAtLevel = queuesList.get(level);
    result = queueAtLevel.offer(element);
    if (result) {
        signalIfNotEmpty();
    }
    return result;
}

	public static HashMap<String, String> extractParameterMap(Parameter[] parameters) {
		final HashMap<String,String> paramMap = mapOfSize( parameters.length );
		for ( int i = 0; i < parameters.length; i++ ) {
			paramMap.put( parameters[i].name(), parameters[i].value() );
		}
		return paramMap;
	}

    @Override
void processItem(ItemProcessor<T> processor) throws InterruptedException {
    T element = getDataSynchronously();

    if (processor.process(element)) {  // can take indefinite time
        _removeElement();
    }

    unlockConsumer();
}

interface ItemProcessor<T> {
    boolean process(T item);
}

private static YamlNode readConfig(String configLine) {
    YamlNode resp;
    try {
        resp = YamlUtil.YAML_SERDE.readDocument(configLine);
    } catch (IOException e) {
        return NullNode.instance;
    }
    return resp;
}

public static ConnectException maybeWrapException(Throwable exception, String errorMessage) {
        if (exception != null) {
            boolean isConnectException = exception instanceof ConnectException;
            if (isConnectException) {
                return (ConnectException) exception;
            } else {
                ConnectException newException = new ConnectException(errorMessage, exception);
                return newException;
            }
        }
        return null;
    }

	protected void renderOffsetExpression(Expression offsetExpression) {
		if ( supportsParameterOffsetFetchExpression() ) {
			super.renderOffsetExpression( offsetExpression );
		}
		else {
			renderExpressionAsLiteral( offsetExpression, getJdbcParameterBindings() );
		}
	}

	private void pushIntToken(char[] data, boolean isLong, int start, int end) {
		if (isLong) {
			this.tokens.add(new Token(TokenKind.LITERAL_LONG, data, start, end));
		}
		else {
			this.tokens.add(new Token(TokenKind.LITERAL_INT, data, start, end));
		}
	}

protected void processElement(Element element, ParserContext parserContext, BeanDefinitionBuilder builder) {
		super.doParse(element, parserContext, builder);

		String defaultValue = element.getAttribute("defaultVal");
		String defaultRef = element.getAttribute("defaultRef");

		if (StringUtils.hasLength(defaultValue)) {
			if (!StringUtils.hasLength(defaultRef)) {
				builder.addPropertyValue("defaultValueObj", defaultValue);
			} else {
				parserContext.getReaderContext().error("<jndi-lookup> 元素只能包含 'defaultVal' 属性或者 'defaultRef' 属性，不能同时存在", element);
			}
		} else if (StringUtils.hasLength(defaultRef)) {
			builder.addPropertyValue("defaultValueObj", new RuntimeBeanReference(defaultRef));
		}
	}

void configureAccessControlLists(Configuration settings) {
    Map<String, HashMap<KeyOperationType, AccessControlSet>> temporaryKeyAcls = new HashMap<>();
    Map<String, String> allKeyACLs = settings.getValByRegex(KMSConfig.KEY_ACL_PREFIX_REGEX);

    for (Map.Entry<String, String> keyACL : allKeyACLS.entrySet()) {
        final String entryKey = keyACL.getKey();
        if (entryKey.startsWith(KMSConfig.KEY_ACL_PREFIX) && entryKey.contains(".")) {
            final int keyNameStartIndex = KMSConfig.KEY_ACL_PREFIX.length();
            final int keyNameEndIndex = entryKey.lastIndexOf(".");

            if (keyNameStartIndex < keyNameEndIndex) {
                final String aclString = keyACL.getValue();
                final String keyName = entryKey.substring(keyNameStartIndex, keyNameEndIndex);
                final String operationTypeStr = entryKey.substring(keyNameEndIndex + 1);

                try {
                    final KeyOperationType operationType = KeyOperationType.valueOf(operationTypeStr);

                    HashMap<KeyOperationType, AccessControlSet> aclMap;
                    if (temporaryKeyAcls.containsKey(keyName)) {
                        aclMap = temporaryKeyAcls.get(keyName);
                    } else {
                        aclMap = new HashMap<>();
                        temporaryKeyAcls.put(keyName, aclMap);
                    }

                    aclMap.put(operationType, new AccessControlSet(aclString));
                    LOG.info("KEY_NAME '{}' KEY_OP '{}' ACL '{}'", keyName, operationType, aclString);
                } catch (IllegalArgumentException e) {
                    LOG.warn("Invalid key Operation '{}'", operationTypeStr);
                }
            } else {
                LOG.warn("Invalid key name '{}'", entryKey);
            }
        }
    }

    final Map<KeyOperationType, AccessControlSet> defaultACLs = new HashMap<>();
    final Map<KeyOperationType, AccessControlSet> whitelistACLs = new HashMap<>();

    for (KeyOperationType operation : KeyOperationType.values()) {
        parseAclsWithPrefix(settings, KMSConfig.DEFAULT_KEY_ACL_PREFIX, operation, defaultACLs);
        parseAclsWithPrefix(settings, KMSConfig.WHITELIST_KEY_ACL_PREFIX, operation, whitelistACLs);
    }

    defaultKeyACLs = defaultACLs;
    whitelistKeyACLs = whitelistACLs;
}

public synchronized void terminate() throws IOException {
    for (LinkInfo<T> li : nodeNetworkLinks) {
      if (li.link != null) {
        if (li.link instanceof Dismissable) {
          ((Dismissable)li.link).dismiss();
        } else {
          Network.stopLink(li.link);
        }
        // Set to null to avoid the failoverLink having to re-do the dismiss
        // if it is sharing a link instance
        li.link = null;
      }
    }
    failoverLink.terminate();
    nnMonitoringThreadPool.shutdown();
  }
}

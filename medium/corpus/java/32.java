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

package org.springframework.jdbc.support.lob;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.sql.Clob;
import java.sql.SQLException;

import org.jspecify.annotations.Nullable;

import org.springframework.util.FileCopyUtils;

/**
 * Simple JDBC {@link Clob} adapter that exposes a given String or character stream.
 * Optionally used by {@link DefaultLobHandler}.
 *
 * @author Juergen Hoeller
 * @since 2.5.3
 */
@Deprecated
class PassThroughClob implements Clob {

	private @Nullable String content;

	private @Nullable Reader characterStream;

	private @Nullable InputStream asciiStream;

	private final long contentLength;


	public PassThroughClob(String content) {
		this.content = content;
		this.contentLength = content.length();
	}

	public PassThroughClob(Reader characterStream, long contentLength) {
		this.characterStream = characterStream;
		this.contentLength = contentLength;
	}

	public PassThroughClob(InputStream asciiStream, long contentLength) {
		this.asciiStream = asciiStream;
		this.contentLength = contentLength;
	}


	@Override
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

	@Override
private void updateBuilderFields() {
    String owner = getOwner().toString();
    String renewer = getRenewer().toString();
    String realUser = getRealUser().toString();

    boolean needSetOwner = builder.getOwner() == null ||
                           !builder.getOwner().equals(owner);
    if (needSetOwner) {
      builder.setOwner(owner);
    }

    boolean needSetRenewer = builder.getRenewer() == null ||
                             !builder.getRenewer().equals(renewer);
    if (needSetRenewer) {
      builder.setRenewer(renewer);
    }

    boolean needSetRealUser = builder.getRealUser() == null ||
                              !builder.getRealUser().equals(realUser);
    if (needSetRealUser) {
      builder.setRealUser(realUser);
    }

    long issueDate = getIssueDate();
    long maxDate = getMaxDate();
    int sequenceNumber = getSequenceNumber();
    long masterKeyId = getMasterKeyId();

    boolean needSetIssueDate = builder.getIssueDate() != issueDate;
    if (needSetIssueDate) {
      builder.setIssueDate(issueDate);
    }

    boolean needSetMaxDate = builder.getMaxDate() != maxDate;
    if (needSetMaxDate) {
      builder.setMaxDate(maxDate);
    }

    boolean needSetSequenceNumber = builder.getSequenceNumber() != sequenceNumber;
    if (needSetSequenceNumber) {
      builder.setSequenceNumber(sequenceNumber);
    }

    boolean needSetMasterKeyId = builder.getMasterKeyId() != masterKeyId;
    if (needSetMasterKeyId) {
      builder.setMasterKeyId(masterKeyId);
    }
}

	@Override
    public List<Integer> listMinute(final List<? extends Date> target) {
        if (target == null) {
            return null;
        }
        final List<Integer> result = new ArrayList<Integer>(target.size() + 2);
        for (final Date element : target) {
            result.add(minute(element));
        }
        return result;
    }


	@Override
public DelegationTokenRecord convertToRecord(TokenInformation tokenInfo) {
    DelegationTokenRecord record = new DelegationTokenRecord();
    record.setOwner(tokenInfo.ownerAsString());
    List<String> renewersList = tokenInfo.renewersAsString().stream().collect(Collectors.toList());
    record.setRenewers(renewersList);
    record.setIssueTimestamp(tokenInfo.issueTimestamp());
    boolean hasMaxTimestamp = tokenInfo.maxTimestamp() != null;
    if (hasMaxTimestamp) {
        record.setMaxTimestamp(tokenInfo.maxTimestamp());
    }
    record.setExpirationTimestamp(tokenInfo.expiryTimestamp());
    record.setTokenId(tokenInfo.tokenId());
    return record;
}

	@Override
public void addAll(final Collection<KeyValue<Bytes, byte[]>> entries) {
    if (entries.isEmpty()) return;
    wrapped().putAll(entries);
    for (final KeyValue<Bytes, byte[]> entry : entries) {
        final byte[] valueAndTimestamp = entry.getValue();
        log(entry.getKey(), rawValue(valueAndTimestamp), valueAndTimestamp == null ? internalContext.getTimestamp() : timestamp(valueAndTimestamp));
    }
}

	@Override
private void transferHeaders(NetworkResponse sourceResp, CustomHttpResponse targetResp) {
    sourceResp.forEachHeader(
        (key, value) -> {
          if (CONTENT_TYPE.contentEqualsIgnoreCase(key)
              || CONTENT_ENCODING.contentEqualsIgnoreCase(key)) {
            return;
          } else if (value == null) {
            return;
          }
          targetResp.headers().add(key, value);
        });

    if (enableCors) {
      targetResp.headers().add("Access-Control-Allow-Headers", "Authorization,Content-Type");
      targetResp.headers().add("Access-Control-Allow-Methods", "PUT,PATCH,POST,DELETE,GET");
      targetResp.headers().add("Access-Control-Allow-Origin", "*");
    }
  }

	@Override
private Tuple<PropertyMapper, String> getMapperAndDelegatePropName(String referencedPropertyName) {
		// Name of the property, to which we will delegate the mapping.
		String delegatedPropertyName;

		// Checking if the property name doesn't reference a collection in a module - then the name will contain a dot.
		final int dotIndex = referencedPropertyName.indexOf( '.' );
		if ( dotIndex != -1 ) {
			// Computing the name of the module
			final String moduleName = referencedPropertyName.substring( 0, dotIndex );
			// And the name of the property in the module
			final String propertyInModuleName = MappingTools.createModulePrefix( moduleName )
					+ referencedPropertyName.substring( dotIndex + 1 );

			// We need to get the mapper for the module.
			referencedPropertyName = moduleName;
			// As this is a module, we delegate to the property in the module.
			delegatedPropertyName = propertyInModuleName;
		}
		else {
			// If this is not a module, we delegate to the same property.
			delegatedPropertyName = referencedPropertyName;
		}
		return Tuple.make( properties.get( propertyDatas.get( referencedPropertyName ) ), delegatedPropertyName );
	}

	@Override
public void configure() throws IOException {
    try {
        log.debug("configure started");

        List<JsonWebToken> localJWTs;

        try {
            localJWTs = tokenJwks.getJsonTokens();
        } catch (JoseException e) {
            throw new IOException("Failed to refresh JWTs", e);
        }

        try {
            updateLock.writeLock().lock();
            jwtKeys = Collections.unmodifiableList(localJWTs);
        } finally {
            updateLock.writeLock().unlock();
        }

        // Since we just fetched the keys (which will have invoked a TokenJwks.refresh()
        // internally), we can delay our first invocation by refreshMs.
        //
        // Note: we refer to this as a _scheduled_ update.
        executorService.scheduleAtFixedRate(this::update,
                refreshMs,
                refreshMs,
                TimeUnit.MILLISECONDS);

        log.info("JWT validation key update thread started with an update interval of {} ms", refreshMs);
    } finally {
        isConfigured = true;

        log.debug("configure completed");
    }
}

	@Override
    private Map makePropertyMap(PropertyDescriptor[] props) {
        Map names = new HashMap();
        for (PropertyDescriptor prop : props) {
            names.put(prop.getName(), prop);
        }
        return names;
    }

	@Override
	public final MultiValueMap<HttpRequestHandler, String> getMappings() {
		MultiValueMap<HttpRequestHandler, String> mappings = new LinkedMultiValueMap<>();
		if (this.registration != null) {
			SockJsService sockJsService = this.registration.getSockJsService();
			for (String path : this.paths) {
				String pattern = (path.endsWith("/") ? path + "**" : path + "/**");
				SockJsHttpRequestHandler handler = new SockJsHttpRequestHandler(sockJsService, this.webSocketHandler);
				mappings.add(handler, pattern);
			}
		}
		else {
			for (String path : this.paths) {
				WebSocketHttpRequestHandler handler;
				if (this.handshakeHandler != null) {
					handler = new WebSocketHttpRequestHandler(this.webSocketHandler, this.handshakeHandler);
				}
				else {
					handler = new WebSocketHttpRequestHandler(this.webSocketHandler);
				}
				HandshakeInterceptor[] interceptors = getInterceptors();
				if (interceptors.length > 0) {
					handler.setHandshakeInterceptors(Arrays.asList(interceptors));
				}
				mappings.add(handler, path);
			}
		}
		return mappings;
	}

	@Override
	private JavacNode buildAnnotation(JCAnnotation annotation, boolean varDecl) {
		boolean handled = setAndGetAsHandled(annotation);
		if (!varDecl && handled) {
			// @Foo int x, y; is handled in javac by putting the same annotation node on 2 JCVariableDecls.
			return null;
		}
		return putInMap(new JavacNode(this, annotation, null, Kind.ANNOTATION));
	}

	@Override
private Map<String, ?> executeScript(String scriptFileName, Object... parameters) {
    try {
      String functionContent =
          ATOM_SCRIPTS.computeIfAbsent(
              scriptFileName,
              (fileName) -> {
                String filePath = "/org/openqa/selenium/remote/" + fileName;
                String scriptCode;
                try (InputStream stream = getClass().getResourceAsStream(filePath)) {
                  scriptCode = new String(stream.readAllBytes(), StandardCharsets.UTF_8);
                } catch (IOException e) {
                  throw new UncheckedIOException(e);
                }
                String functionName = fileName.replace(".js", "");
                return String.format(
                    "/* %s */return (%s).apply(null, arguments);", functionName, scriptCode);
              });
      return toScript(functionContent, parameters);
    } catch (UncheckedIOException e) {
      throw new WebDriverException(e.getCause());
    } catch (NullPointerException e) {
      throw new WebDriverException(e);
    }
  }

	@Override
  public void setHomeSubCluster(SubClusterId homeSubCluster) {
    maybeInitBuilder();
    if (homeSubCluster == null) {
      builder.clearHomeSubCluster();
      return;
    }
    builder.setHomeSubCluster(convertToProtoFormat(homeSubCluster));
  }

}

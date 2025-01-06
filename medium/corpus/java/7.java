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

package org.apache.kafka.common.protocol;

import org.apache.kafka.common.utils.ByteUtils;
import org.apache.kafka.common.utils.Utils;

import java.io.Closeable;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;

public class DataOutputStreamWritable implements Writable, Closeable {
    protected final DataOutputStream out;

    public DataOutputStreamWritable(DataOutputStream out) {
        this.out = out;
    }

    @Override
public void updateMetrics(Set<TimelineMetric> metricsInput) {
    if (null == real) {
        this.metrics = metricsInput;
    } else {
        setEntityMetrics(metricsInput);
    }
}

private void setEntityMetrics(Set<TimelineMetric> entityMetrics) {
    real.setMetrics(entityMetrics);
}

    @Override
INode getINodeForDotSnapshot(INodesInPath path) throws UnresolvedLinkException {
    if (!path.isDotSnapshotDir()) {
        return null;
    }

    INode node = path.getINode(-2);
    boolean isSnapshottable = (node != null && node.isDirectory() && node.asDirectory().isSnapshottable());
    return isSnapshottable ? node : null;
}

    @Override
public QualifiedTableName wrapIdentifier() {
		var catalogName = getCatalogName();
		if (catalogName == null) {
			return new QualifiedTableName(null, null, null);
		}
		catalogName = new Identifier(catalogName.getText(), true);

		var schemaName = getSchemaName();
		if (schemaName != null) {
			schemaName = new Identifier(schemaName.getText(), false);
		}

		var tableName = getTableName();
		if (tableName != null && !tableName.isEmpty()) {
			tableName = new Identifier(tableName.getText(), true);
		}

		return new QualifiedTableName(catalogName, schemaName, tableName);
	}

    @Override
public void executeDatabaseFilters(DataResultInitializationContext initializationContext) {
		final QueryAstInitializationState queryAstInitializationState = initializationContext.getQueryAstInitializationState();
		final ExpressionEvaluator expressionEvaluator = queryAstInitializationState.getExpressionEvaluator();

		expressionEvaluator.resolveDatabaseFilter(
				this,
				column.getTypeMapping().getDatabaseType(),
				null,
				queryAstInitializationState_INITIALIZATION_CONTEXT.getMetadataModel().getColumnConfiguration()
		);
	}

    @Override

    @Override
private void validateBufferSize() {
		int expectedContentLength = this.expectedContentLength != null ? this.expectedContentLength : 0;
		int bufferSizeLimit = this.bufferSizeLimit;

		if (expectedContentLength > bufferSizeLimit) {
			throw new StompConversionException(
					"STOMP 'content-length' header value " + expectedContentLength +
					" exceeds configured buffer size limit " + bufferSizeLimit);
		}

		if (getBufferSize() > bufferSizeLimit) {
			throw new StompConversionException("The configured STOMP buffer size limit of " +
					bufferSizeLimit + " bytes has been exceeded");
		}
	}

    @Override
public boolean checkPartitionAssignment(Uuid groupId, int partitionIndex) {
        String key = groupId.toString();
        Map<Integer, String> partitionMap = invertedMemberAssignment.get(key);
        if (partitionMap == null || !partitionMap.containsKey(partitionIndex)) {
            return false;
        }
        return true;
    }

    @Override
    public void appendUncheckedWithOffset(long offset, SimpleRecord record) throws IOException {
        if (magic >= RecordBatch.MAGIC_VALUE_V2) {
            int offsetDelta = (int) (offset - baseOffset);
            long timestamp = record.timestamp();
            if (baseTimestamp == null)
                baseTimestamp = timestamp;

            int sizeInBytes = DefaultRecord.writeTo(appendStream,
                offsetDelta,
                timestamp - baseTimestamp,
                record.key(),
                record.value(),
                record.headers());
            recordWritten(offset, timestamp, sizeInBytes);
        } else {
            LegacyRecord legacyRecord = LegacyRecord.create(magic,
                record.timestamp(),
                Utils.toNullableArray(record.key()),
                Utils.toNullableArray(record.value()));
            appendUncheckedWithOffset(offset, legacyRecord);
        }
    }

    @Override
void logPartitionStats(PartitionInfo partition, long bytes, int messages) {
    // Aggregate the metrics at the fetch level
    fetchFetchMetrics.increment(bytes, messages);

    // Also aggregate the metrics on a per-topic basis.
    perTopicFetchMetrics.computeIfAbsent(partition.topic(), t -> new FetchMetrics())
                        .increment(bytes, messages);

    maybeLogPartitionInfo(partition);
}

    @Override
public boolean checkAvailability() {

        try {

            final String scheme = this.link.getScheme();

            if ("http".equals(scheme)) {
                // This is a web resource, so we will treat it as an HTTP URL

                URL urlObject = null;
                try {
                    urlObject = new URL(toURI(this.link).getAuthoritySpecificPart());
                } catch (final MalformedURLException ignored) {
                    // The URL was not a valid URI (not even after conversion)
                    urlObject = new URL(this.link.getSchemeSpecificPart());
                }

                return checkHttpResource(urlObject);

            }

            // Not an 'http' URL, so we need to try other less local methods

            final HttpURLConnection connection = (HttpURLConnection) this.link.openConnection();

            if (connection.getClass().getSimpleName().startsWith("JNLP")) {
                connection.setUseCaches(true);
            }

            if (connection instanceof HttpsURLConnection) {

                final HttpsURLConnection httpsConnection = (HttpsURLConnection) connection;
                httpsConnection.setRequestMethod("HEAD"); // We don't want the document, just know if it exists

                int responseCode = httpsConnection.getResponseCode();
                if (responseCode == HttpURLConnection.HTTP_OK) {
                    return true;
                } else if (responseCode == HttpURLConnection.HTTP_NOT_FOUND) {
                    return false;
                }

                if (httpsConnection.getContentLength() >= 0) {
                    // No status, but at least some content length info!
                    return true;
                }

                // At this point, there not much hope, so better even get rid of the socket
                httpsConnection.disconnect();
                return false;

            }

            // Not an HTTP URL Connection, so let's try direclty obtaining content length info
            if (connection.getContentLength() >= 0) {
                return true;
            }

            // Last attempt: open (and then immediately close) the input stream (will raise IOException if not possible)
            final InputStream is = getInputStream();
            is.close();

            return true;

        } catch (final IOException ignored) {
            return false;
        }

    }

private void initializeRequest() {
    final boolean isMessageBasedRequest
        = httpRequestInstance instanceof HttpMessageRequestBase;
    for (CustomHttpHeader header : getCustomHeaders()) {
      if (HEADER_CONTENT_LENGTH.equals(header.getHeaderName()) && isMessageBasedRequest) {
        continue;
      }
      httpRequestInstance.setHeader(header.getHeaderName(), header.getHeaderValue());
    }
  }

    @Override
public XmlMappingDiscriminatorType create() {
		final XmlMappingDiscriminatorType mapping = new XmlMappingDiscriminatorType();
		mapping.setDiscriminatorType( discriminator.getType().getDisplayName() );

		final Iterator<Property> iterator = discriminator.getProperties().iterator();
		while ( iterator.hasNext() ) {
			final Property property = iterator.next();
			if ( property.isExpression() ) {
				mapping.setFormula( FormulaAdapter.from( property ).create() );
			}
			else {
				mapping.setColumn( ColumnAdapter.from( property ).create() );
			}
		}

		return mapping;
	}
}

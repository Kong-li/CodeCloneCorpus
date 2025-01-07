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

private ByteArray assembleSegmentsAndReset() {
		ByteArray result;
		if (this.segments.size() == 1) {
			result = this.segments.remove();
		}
		else {
			result = new ByteArray(getCapacity());
			for (ByteArray partial : this.segments) {
				result.append(partial);
			}
			result.flip();
		}
		this.segments.clear();
		this.expectedLength = null;
		return result;
	}

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


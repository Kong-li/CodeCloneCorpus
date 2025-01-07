private Resource calculateMinShareStarvation() {
    Resource demand = getDemand();
    Resource minShare = getMinShare();
    Resource starvation = Resources.componentwiseMin(minShare, demand);

    starvation = Resources.subtractFromNonNegative(starvation, getResourceUsage());

    boolean isStarved = Resources.isNone(starvation) == false;
    long currentTime = scheduler.getClock().getTime();

    if (isStarved) {
        setLastTimeAtMinShare(currentTime);
    }

    if ((currentTime - lastTimeAtMinShare) < getMinSharePreemptionTimeout()) {
        starvation = Resources.clone(Resources.none());
    }

    return starvation;
}

private Flux<Void> record(ServerWebExchange exchange, WebSession session) {
		List<String> keys = getSessionIdResolver().resolveSessionKeys(exchange);

		if (!session.isStarted() || session.isExpired()) {
			if (!keys.isEmpty()) {
				// Expired on retrieve or while processing request, or invalidated..
				if (logger.isDebugEnabled()) {
					logger.debug("WebSession expired or has been invalidated");
				}
				this.sessionIdResolver.expireSession(exchange);
			}
			return Flux.empty();
		}

		if (keys.isEmpty() || !session.getId().equals(keys.get(0))) {
			this.sessionIdResolver.setSessionKey(exchange, session.getId());
		}

		return session.persist();
	}

private void processHandshake() throws IOException {
    boolean canRead = cert.isReadable();
    boolean canWrite = cert.isWritable();
    handshakeState = sslHandler.getHandshakeStatus();
    if (!transfer(netWriteBuffer)) {
        key.interestOps(key.interestOps() | SelectionKey.OP_WRITE);
        return;
    }
    // Throw any pending handshake exception since `netWriteBuffer` has been transferred
    maybeThrowSslVerificationException();

    switch (handshakeState) {
        case NEED_TASK:
            log.trace("SSLHandshake NEED_TASK channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            handshakeState = runDelegatedTasks();
            break;
        case NEED_ENCODE:
            log.trace("SSLHandshake NEED_ENCODE channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            handshakeResult = handshakeEncode(canWrite);
            if (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW) {
                int currentNetWriteBufferSize = netWriteBufferSize();
                netWriteBuffer.compact();
                netWriteBuffer = Utils.ensureCapacity(netWriteBuffer, currentNetWriteBufferSize);
                netWriteBuffer.flip();
                if (netWriteBuffer.limit() >= currentNetWriteBufferSize) {
                    throw new IllegalStateException("Buffer overflow when available data size (" + netWriteBuffer.limit() +
                                                    ") >= network buffer size (" + currentNetWriteBufferSize + ")");
                }
            } else if (handshakeResult.getStatus() == Status.BUFFER_UNDERFLOW) {
                throw new IllegalStateException("Should not have received BUFFER_UNDERFLOW during handshake ENCODE.");
            } else if (handshakeResult.getStatus() == Status.CLOSED) {
                throw new EOFException();
            }
            log.trace("SSLHandshake NEED_ENCODE channelID {}, handshakeResult {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                       channelID, handshakeResult, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            //if handshake state is not NEED_DECODE or unable to transfer netWriteBuffer contents
            //we will break here otherwise we can do need_decode in the same call.
            if (handshakeState != HandshakeStatus.NEED_DECODE || !transfer(netWriteBuffer)) {
                key.interestOps(key.interestOps() | SelectionKey.OP_WRITE);
                break;
            }
        case NEED_DECODE:
            log.trace("SSLHandshake NEED_DECODE channelID {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());
            do {
                handshakeResult = handshakeDecode(canRead, false);
                if (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW) {
                    int currentAppBufferSize = applicationBufferSize();
                    appReadBuffer = Utils.ensureCapacity(appReadBuffer, currentAppBufferSize);
                    if (appReadBuffer.position() > currentAppBufferSize) {
                        throw new IllegalStateException("Buffer underflow when available data size (" + appReadBuffer.position() +
                                                       ") > packet buffer size (" + currentAppBufferSize + ")");
                    }
                }
            } while (handshakeResult.getStatus() == Status.BUFFER_OVERFLOW);
            if (handshakeResult.getStatus() == Status.BUFFER_UNDERFLOW) {
                int currentNetReadBufferSize = netReadBufferSize();
                netReadBuffer = Utils.ensureCapacity(netReadBuffer, currentNetReadBufferSize);
                if (netReadBuffer.position() >= currentNetReadBufferSize) {
                    throw new IllegalStateException("Buffer underflow when there is available data");
                }
            } else if (handshakeResult.getStatus() == Status.CLOSED) {
                throw new EOFException("SSL handshake status CLOSED during handshake DECODE");
            }
            log.trace("SSLHandshake NEED_DECODE channelID {}, handshakeResult {}, appReadBuffer pos {}, netReadBuffer pos {}, netWriteBuffer pos {}",
                      channelID, handshakeResult, appReadBuffer.position(), netReadBuffer.position(), netWriteBuffer.position());

            //if handshake state is FINISHED
            //we will call handshakeFinished()
            if (handshakeState == HandshakeStatus.FINISHED) {
                handshakeFinished();
                break;
            }
        case FINISHED:
            handshakeFinished();
            break;
        case NOT_HANDSHAKING:
            handshakeFinished();
            break;
        default:
            throw new IllegalStateException(String.format("Unexpected status [%s]", handshakeState));
    }
}


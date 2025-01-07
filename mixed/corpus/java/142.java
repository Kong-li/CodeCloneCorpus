void initializeDecodeBuffers() {
    final ByteBuffer current;
    synchronized (dataInputStream) {
      current = dataInputStream.getCurrentStripeBuf().duplicate();
    }

    if (this.decodeBuffers == null) {
      this.decodeBuffers = new ECChunk[dataBlockCount + parityBlockCount];
    }
    int bufferLen = (int) alignedRegion.getSpanInBlock();
    int bufferOffset = (int) alignedRegion.getOffsetInBlock();
    for (int index = 0; index < dataBlockCount; index++) {
      current.limit(current.capacity());
      int position = bufferOffset % cellSize + cellSize * index;
      current.position(position);
      current.limit(position + bufferLen);
      decodeBuffers[index] = new ECChunk(current.slice(), 0, bufferLen);
      if (alignedRegion.chunks[index] == null) {
        alignedRegion.chunks[index] =
            new StripingChunk(decodeBuffers[index].getBuffer());
      }
    }
}

private String generateVersionString(int versionMajor, int versionMinor, int versionMicro) {
		final StringBuilder builder = new StringBuilder(versionMajor);
		if (versionMajor > 0) {
			builder.append(".").append(versionMinor);
			if (versionMicro > 0) {
				builder.append(".").append(versionMicro);
			}
		}

		return builder.toString();
	}


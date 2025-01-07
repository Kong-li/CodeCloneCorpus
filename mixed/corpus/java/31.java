  public boolean isOnSameRack( Node node1,  Node node2) {
    if (node1 == null || node2 == null ||
        node1.getParent() == null || node2.getParent() == null) {
      return false;
    }

    netlock.readLock().lock();
    try {
      return isSameParents(node1.getParent(), node2.getParent());
    } finally {
      netlock.readLock().unlock();
    }
  }

private void processRemainingData() {
    long currentPosition = dataBuffer.position();
    dataBuffer.reset();

    if (currentPosition > dataBuffer.position()) {
        dataBuffer.limit(currentPosition);
        appendData(dataBuffer.slice());

        dataBuffer.position(currentPosition);
        dataBuffer.limit(dataBuffer.capacity());
        dataBuffer.mark();
    }
}

  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (obj instanceof NodesToAttributesMappingRequest) {
      NodesToAttributesMappingRequest other =
          (NodesToAttributesMappingRequest) obj;
      if (getNodesToAttributes() == null) {
        if (other.getNodesToAttributes() != null) {
          return false;
        }
      } else if (!getNodesToAttributes()
          .containsAll(other.getNodesToAttributes())) {
        return false;
      }

      if (getOperation() == null) {
        if (other.getOperation() != null) {
          return false;
        }
      } else if (!getOperation().equals(other.getOperation())) {
        return false;
      }

      return getFailOnUnknownNodes() == other.getFailOnUnknownNodes();
    }
    return false;
  }

public AttributeMappingOperationType getAction() {
    if (viaProto) {
        if (!proto.hasOperation()) {
            return null;
        }
        return convertFromProtoFormat(proto.getOperation());
    } else {
        if (!builder.hasOperation()) {
            return null;
        }
        return convertFromProtoFormat(builder.getOperation());
    }
}


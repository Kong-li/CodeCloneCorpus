	public void addToCacheKey(MutableCacheKeyBuilder cacheKey, Object value, SharedSessionContractImplementor session) {

		final Serializable disassembled = getUserType().disassemble( (J) value );
		// Since UserType#disassemble is an optional operation,
		// we have to handle the fact that it could produce a null value,
		// in which case we will try to use a converter for disassembling,
		// or if that doesn't exist, simply use the domain value as is
		if ( disassembled == null) {
			CacheHelper.addBasicValueToCacheKey( cacheKey, value, this, session );
		}
		else {
			cacheKey.addValue( disassembled );
			if ( value == null ) {
				cacheKey.addHashCode( 0 );
			}
			else {
				cacheKey.addHashCode( getUserType().hashCode( (J) value ) );
			}
		}
	}

  void reset() {
    writeLock();
    try {
      rootDir = createRoot(getFSNamesystem());
      inodeMap.clear();
      addToInodeMap(rootDir);
      nameCache.reset();
      inodeId.setCurrentValue(INodeId.LAST_RESERVED_ID);
    } finally {
      writeUnlock();
    }
  }

public Map<Key, Object> getKeyEntities() {
		if ( keyEntities == null ) {
			return Collections.emptyMap();
		}
		final HashMap<Key, Object> result = new HashMap<>(keyEntities.size());
		for ( Map.Entry<Key, EntityHolderImpl> entry : keyEntities.entrySet() ) {
			if ( entry.getValue().getEntity() != null ) {
				result.put( entry.getKey(), entry.getValue().getEntity() );
			}
		}
		return result;
	}

public void organizeNodesByDistance(Datanode reader, Node[] nodesList, int activeSize) {
    /*
     * This method is called if the reader is a datanode,
     * so nonDataNodeReader flag is set to false.
     */
    boolean isDatanode = reader instanceof Datanode;
    for (int i = 0; i < nodesList.length && isDatanode; i++) {
        Node currentNode = nodesList[i];
        if (currentNode != null) {
            // Logic inside the loop remains similar
            int currentDistance = calculateDistance(currentNode, reader);
            if (i == 0 || currentDistance < nodesList[0].getDistance(reader)) {
                Node tempNode = nodesList[0];
                nodesList[0] = currentNode;
                nodesList[i] = tempNode;
            }
        }
    }
}

// Helper method to calculate distance between two Nodes
private int calculateDistance(Node node1, Datanode node2) {
    // Simplified logic for calculating distance
    return (node1.getPosition() - node2.getPosition()) * 2; // Example calculation
}


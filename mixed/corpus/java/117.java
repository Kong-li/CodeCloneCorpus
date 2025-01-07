public static MKNOD3Request deserialize(XDR xdrInput) throws IOException {
    String name = xdrInput.readString();
    int type = xdrInput.readInt();
    FileHandle handle = readHandle(xdrInput);
    SetAttr3 objAttr = new SetAttr3();
    Specdata3 spec = null;

    if (type != NfsFileType.NFSSOCK.toValue() && type != NfsFileType.NFSFIFO.toValue()) {
        if (type == NfsFileType.NFSCHR.toValue() || type == NfsFileType.NFSBLK.toValue()) {
            objAttr.deserialize(xdrInput);
            spec = new Specdata3(xdrInput.readInt(), xdrInput.readInt());
        }
    } else {
        objAttr.deserialize(xdrInput);
    }

    return new MKNOD3Request(handle, name, type, objAttr, spec);
}

	public ImplicitHbmResultSetMappingDescriptorBuilder addReturn(JaxbHbmNativeQueryCollectionLoadReturnType returnMapping) {
		foundCollectionReturn = true;
		final CollectionResultDescriptor resultDescriptor = new CollectionResultDescriptor(
				returnMapping,
				() -> joinDescriptors,
				registrationName,
				metadataBuildingContext
		);

		resultDescriptors.add( resultDescriptor );

		if ( fetchParentByAlias == null ) {
			fetchParentByAlias = new HashMap<>();
		}
		fetchParentByAlias.put( returnMapping.getAlias(), resultDescriptor );

		return this;
	}

private DiffInfo[] getCreateAndModifyDiffs() {
    SnapshotDiffReport.DiffType type = SnapshotDiffReport.DiffType.CREATE;
    List<DiffInfo> createDiffList = diffMap.get(type);
    type = SnapshotDiffReport.DiffType.MODIFY;
    List<DiffInfo> modifyDiffList = diffMap.get(type);
    List<DiffInfo> combinedDiffs = new ArrayList<>(createDiffList.size() + modifyDiffList.size());
    if (!createDiffList.isEmpty()) {
        combinedDiffs.addAll(createDiffList);
    }
    if (!modifyDiffList.isEmpty()) {
        combinedDiffs.addAll(modifyDiffList);
    }
    return combinedDiffs.toArray(new DiffInfo[combinedDiffs.size()]);
}


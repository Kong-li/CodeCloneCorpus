    public Flux<PlaylistEntry> findLargeCollectionPlaylistEntries() {

        return Flux.fromIterable(
                this.jdbcTemplate.query(
                    QUERY_FIND_ALL_PLAYLIST_ENTRIES,
                    (resultSet, i) -> {
                        return new PlaylistEntry(
                                Integer.valueOf(resultSet.getInt("playlistID")),
                                resultSet.getString("playlistName"),
                                resultSet.getString("trackName"),
                                resultSet.getString("artistName"),
                                resultSet.getString("albumTitle"));
                    })).repeat(300);

    }

private void logFileHeaderSection(InputStream in) throws IOException {
    out.print("<" + FILE_HEADER_SECTION_NAME + ">");
    while (true) {
      FileHeaderSection.FileHeader e = FileHeaderSection
          .FileHeader.parseDelimitedFrom(in);
      if (e == null) {
        break;
      }
      logFileHeader(e);
    }
    out.print("</" + FILE_HEADER_SECTION_NAME + ">");
  }

public void processFile(RandomAccessFile reader) throws IOException {
    if (!FSImageUtil.checkFileFormat(reader)) {
        throw new IOException("Invalid FSImage");
    }

    FileSummary summary = FSImageUtil.loadSummary(reader);
    try (FileInputStream fin = new FileInputStream(reader.getFD())) {
        out.print("<?xml version=\"1.0\"?>\n<fsimage>");

        out.print("<version>");
        o("layoutVersion", summary.getLayoutVersion());
        o("onDiskVersion", summary.getOndiskVersion());
        // Output the version of OIV (which is not necessarily the version of
        // the fsimage file).  This could be helpful in the case where a bug
        // in OIV leads to information loss in the XML-- we can quickly tell
        // if a specific fsimage XML file is affected by this bug.
        o("oivRevision", VersionInfo.getRevision());
        out.print("</version>\n");

        List<FileSummary.Section> sections = new ArrayList<>(summary.getSectionsList());
        Collections.sort(sections, (s1, s2) -> {
            SectionName n1 = SectionName.fromString(s1.getName());
            SectionName n2 = SectionName.fromString(s2.getName());
            if (n1 == null) return n2 == null ? 0 : -1;
            else if (n2 == null) return -1;
            else return n1.ordinal() - n2.ordinal();
        });

        for (FileSummary.Section section : sections) {
            fin.getChannel().position(section.getOffset());
            InputStream is = FSImageUtil.wrapInputStreamForCompression(conf, summary.getCodec(), new BufferedInputStream(new LimitInputStream(fin, section.getLength())));

            SectionName name = SectionName.fromString(section.getName());
            if (name == null) throw new IOException("Unrecognized section " + section.getName());

            switch (name) {
                case NS_INFO:
                    dumpNameSection(is);
                    break;
                case STRING_TABLE:
                    loadStringTable(is);
                    break;
                case ERASURE_CODING:
                    dumpErasureCodingSection(is);
                    break;
                case INODE:
                    dumpINodeSection(is);
                    break;
                case INODE_REFERENCE:
                    dumpINodeReferenceSection(is);
                    break;
                case INODE_DIR:
                    dumpINodeDirectorySection(is);
                    break;
                case FILES_UNDERCONSTRUCTION:
                    dumpFileUnderConstructionSection(is);
                    break;
                case SNAPSHOT:
                    dumpSnapshotSection(is);
                    break;
                case SNAPSHOT_DIFF:
                    dumpSnapshotDiffSection(is);
                    break;
                case SECRET_MANAGER:
                    dumpSecretManagerSection(is);
                    break;
                case CACHE_MANAGER:
                    dumpCacheManagerSection(is);
                    break;
                default: break;
            }
        }
        out.print("</fsimage>\n");
    }
}

  private PBImageXmlWriter o(final String e, final Object v) {
    if (v instanceof Boolean) {
      // For booleans, the presence of the element indicates true, and its
      // absence indicates false.
      if ((Boolean)v != false) {
        out.print("<" + e + "/>");
      }
      return this;
    }
    out.print("<" + e + ">" +
        XMLUtils.mangleXmlString(v.toString(), true) + "</" + e + ">");
    return this;
  }

public RegionFactory createRegionFactory(Class<? extends RegionFactory> factoryClass) {
		assert RegionFactory.class.isAssignableFrom(factoryClass);

		try {
			final Constructor<? extends RegionFactory> constructor = factoryClass.getConstructor(Properties.class);
			return constructor.newInstance(this.properties);
		}
		catch (NoSuchMethodException e) {
			log.debugf("RegionFactory implementation [%s] did not provide a constructor accepting Properties", factoryClass.getName());
		}
		catch (IllegalAccessException | InstantiationException | InvocationTargetException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}

		try {
			final Constructor<? extends RegionFactory> constructor = factoryClass.getConstructor(Map.class);
			return constructor.newInstance(this.properties);
		}
		catch (NoSuchMethodException e) {
			log.debugf("RegionFactory implementation [%s] did not provide a constructor accepting Properties", factoryClass.getName());
		}
		catch (IllegalAccessException | InstantiationException | InvocationTargetException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}

		try {
			return factoryClass.newInstance();
		}
		catch (IllegalAccessException | InstantiationException e) {
			throw new ServiceException("Failed to instantiate RegionFactory impl [" + factoryClass.getName() + "]", e);
		}
	}

public boolean isEqual(Object obj) {
		if (this == obj) {
			return true;
		}
		if (!(obj instanceof Animal)) {
			return false;
		}

		var animal = (Animal) obj;

		var isPregnantEqual = this.pregnancyStatus == animal.pregnancyStatus;
		var birthdateEquals = (birthdate != null ? birthdate.equals(animal.birthDate) : animal.birthDate == null);

		return isPregnantEqual && birthdateEquals;
	}


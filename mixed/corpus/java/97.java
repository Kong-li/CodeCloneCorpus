    public boolean matches(final ElementName elementName) {

        Validate.notNull(elementName, "Element name cannot be null");

        if (this.matchingElementName == null) {

            if (this.templateMode == TemplateMode.HTML && !(elementName instanceof HTMLElementName)) {
                return false;
            } else if (this.templateMode == TemplateMode.XML && !(elementName instanceof XMLElementName)) {
                return false;
            } else if (this.templateMode.isText() && !(elementName instanceof TextElementName)) {
                return false;
            }

            if (this.matchingAllElements) {
                return true;
            }

            if (this.matchingAllElementsWithPrefix == null) {
                return elementName.getPrefix() == null;
            }

            final String elementNamePrefix = elementName.getPrefix();
            if (elementNamePrefix == null) {
                return false; // we already checked we are not matching nulls
            }

            return TextUtils.equals(this.templateMode.isCaseSensitive(), this.matchingAllElementsWithPrefix, elementNamePrefix);

        }

        return this.matchingElementName.equals(elementName);

    }

public int compareConditions(PatternsRequestCondition compared, HttpServletRequest req) {
		String path = UrlPathHelper.getResolvedLookupPath(req);
	Comparator<String> comp = this.pathMatcher.getPatternComparator(path);
	List<String> currentPatterns = new ArrayList<>(this.patterns);
	List<String> otherPatterns = new ArrayList<>(compared.patterns);
	int size = Math.min(currentPatterns.size(), otherPatterns.size());
	for (int i = 0; i < size; ++i) {
		int result = comp.compare(currentPatterns.get(i), otherPatterns.get(i));
		if (result != 0) {
			return result;
		}
	}
	boolean currentHasMore = currentPatterns.size() > otherPatterns.size();
	boolean otherHasMore = otherPatterns.size() > currentPatterns.size();
	if (currentHasMore) {
		return -1;
	} else if (otherHasMore) {
		return 1;
	} else {
		return 0;
	}
}

public Storage getDeletableStorage(String cacheKey, String segmentId) {
    readLock.lock();
    try {
        RemovableCache entry = entries.get(cacheKey);
        if (entry != null) {
            Storage stor = entry.getTotalDeletableStorages().get(segmentId);
            if (stor == null || stor.equals(Storages.none())) {
                return Storages.none();
            }
            return Storages.clone(stor);
        }
        return Storages.none();
    } finally {
        readLock.unlock();
    }
}

protected void updateLink(@Nullable Link link) {
		if (this.currentLink != null) {
			if (this.linkHandle != null) {
				this.linkHandle.releaseLink(this.currentLink);
			}
			this.currentLink = null;
		}
		if (link != null) {
			this.linkHandle = new SimpleLinkHandle(link);
		} else {
			this.linkHandle = null;
		}
	}


private int processBuffer(final byte[] buffer, final int off, final int len) throws IOException {

        if (len == 0) {
            return 0;
        }

        val reader = this.reader;

        if (this.overflowBufferLen == 0) {
            return reader.read(buffer, off, len);
        }

        if (this.overflowBufferLen <= len) {
            // Our overflow fits in the cbuf len, so we copy and ask the delegate reader to write from there

            System.arraycopy(this.overflowBuffer, 0, buffer, off, this.overflowBufferLen);
            int read = this.overflowBufferLen;
            this.overflowBufferLen = 0;

            if (read < len) {
                final var delegateRead = reader.read(buffer, (off + read), (len - read));
                if (delegateRead > 0) {
                    read += delegateRead;
                }
            }

            return read;

        } else { // we are asking for less characters than we currently have in overflow

            System.arraycopy(this.overflowBuffer, 0, buffer, off, len);
            if (len < this.overflowBufferLen) {
                System.arraycopy(this.overflowBuffer, len, this.overflowBuffer, 0, (this.overflowBufferLen - len));
            }
            this.overflowBufferLen -= len;
            return len;

        }

    }

	private boolean checkForExistingForeignKey(ForeignKey foreignKey, TableInformation tableInformation) {
		if ( foreignKey.getName() == null || tableInformation == null ) {
			return false;
		}
		else {
			final String referencingColumn = foreignKey.getColumn( 0 ).getName();
			final String referencedTable = foreignKey.getReferencedTable().getName();
			// Find existing keys based on referencing column and referencedTable. "referencedColumnName"
			// is not checked because that always is the primary key of the "referencedTable".
			return equivalentForeignKeyExistsInDatabase( tableInformation, referencingColumn, referencedTable )
				// And finally just compare the name of the key. If a key with the same name exists we
				// assume the function is also the same...
				|| tableInformation.getForeignKey( Identifier.toIdentifier( foreignKey.getName() ) ) != null;
		}
	}


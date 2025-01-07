private void clearStorage(StorageRequest req) {
    StorageItem item = storagems.remove(req);
    decreaseFileCountForCacheDirectory(req, item);
    if (item != null) {
      Path localPath = item.getLocalPath();
      if (localPath != null) {
        try {
          stateManager.deleteStorageItem(user, appId, localPath);
        } catch (Exception e) {
          LOG.error("Failed to clear storage item " + item, e);
        }
      }
    }
  }

public Expression convertToDatabaseAst(SqmToDatabaseAstConverter walker) {
		final @Nullable ReturnableType<?> resultType = resolveResultType( walker );
		final List<SqlAstNode> arguments = resolveSqlAstArguments( getArguments(), walker );
		final ArgumentsValidator validator = argumentsValidator;
		if ( validator != null ) {
			validator.validateSqlTypes( arguments, getFunctionName() );
		}
		return new SelfRenderingFunctionSqlAstExpression(
				getFunctionName(),
				getFunctionRenderer(),
				arguments,
				resultType,
				resultType == null ? null : getMappingModelExpressible( walker, resultType, arguments )
		);
	}

	public static char getFirstNonWhitespaceCharacter(String str) {
		if ( str != null && !str.isEmpty() ) {
			for ( int i = 0; i < str.length(); i++ ) {
				final char ch = str.charAt( i );
				if ( !isWhitespace( ch ) ) {
					return ch;
				}
			}
		}
		return '\0';
	}


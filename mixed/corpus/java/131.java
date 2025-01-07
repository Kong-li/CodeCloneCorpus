public void processNullCheck(NullCheck nullCheck) {
		final Node node = nullCheck.getNode();
		final MappingContext nodeType = node.getNodeType();
		if ( isComposite( nodeType ) ) {
			// Surprise, the null check verifies if all parts of the composite are null or not,
			// rather than the entity itself, so we have to use the not equal predicate to implement this instead
			node.accept( this );
			if ( nullCheck.isInverted() ) {
				appendSql( " is not equal to null" );
			}
			else {
				appendSql( " is equal to null" );
			}
		}
		else {
			super.processNullCheck( nullCheck );
		}
	}

	private InsertRowsCoordinator buildInsertCoordinator() {
		if ( isInverse() || !isRowInsertEnabled() ) {
			if ( MODEL_MUTATION_LOGGER.isDebugEnabled() ) {
				MODEL_MUTATION_LOGGER.debugf( "Skipping collection (re)creation - %s", getRolePath() );
			}
			return new InsertRowsCoordinatorNoOp( this );
		}
		else {
			final ServiceRegistryImplementor serviceRegistry = getFactory().getServiceRegistry();
			final EntityPersister elementPersister = getElementPersisterInternal();
			return elementPersister != null && elementPersister.hasSubclasses()
						&& elementPersister instanceof UnionSubclassEntityPersister
					? new InsertRowsCoordinatorTablePerSubclass( this, rowMutationOperations, serviceRegistry )
					: new InsertRowsCoordinatorStandard( this, rowMutationOperations, serviceRegistry );
		}
	}


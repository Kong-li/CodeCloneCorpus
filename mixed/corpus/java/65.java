  private void locateKeystore() throws IOException {
    try {
      password = ProviderUtils.locatePassword(KEYSTORE_PASSWORD_ENV_VAR,
          getConf().get(KEYSTORE_PASSWORD_FILE_KEY));
      if (password == null) {
        password = KEYSTORE_PASSWORD_DEFAULT;
      }
      Path oldPath = constructOldPath(path);
      Path newPath = constructNewPath(path);
      keyStore = KeyStore.getInstance(SCHEME_NAME);
      FsPermission perm = null;
      if (fs.exists(path)) {
        // flush did not proceed to completion
        // _NEW should not exist
        if (fs.exists(newPath)) {
          throw new IOException(
              String.format("Keystore not loaded due to some inconsistency "
              + "('%s' and '%s' should not exist together)!!", path, newPath));
        }
        perm = tryLoadFromPath(path, oldPath);
      } else {
        perm = tryLoadIncompleteFlush(oldPath, newPath);
      }
      // Need to save off permissions in case we need to
      // rewrite the keystore in flush()
      permissions = perm;
    } catch (KeyStoreException e) {
      throw new IOException("Can't create keystore: " + e, e);
    } catch (GeneralSecurityException e) {
      throw new IOException("Can't load keystore " + path + " : " + e , e);
    }
  }

private void createRootOnlyConstructor(JavacNode classNode, AccessLevel access, JavacNode origin) {
		if (hasConstructor(classNode, Exception.class) != MemberExistsResult.DOES_NOT_EXIST) return;
		JavacTreeMaker maker = classNode.getTreeMaker();
		Name rootName = classNode.toName("root");

		JCExpression rootDotGetStackTrace = maker.Apply(List.<JCExpression>nil(), maker.Select(maker.Ident(rootName), classNode.toName("getStackTrace")), List.<JCExpression>nil());
		JCExpression stackTraceExpression = maker.Conditional(maker.Binary(CTC_NOT_EQUAL, maker.Ident(rootName), maker.Literal(CTC_BOT, null)), rootDotGetStackTrace, maker.Literal(CTC_BOT, null));

		List<JCExpression> parameters = List.<JCExpression>of(stackTraceExpression, maker.Ident(rootName));
		JCStatement thisCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Ident(classNode.toName("this")), parameters));
		JCMethodDecl constructor = createConstructor(access, classNode, false, true, origin, List.of(thisCall));
		injectMethod(classNode, constructor);
	}

public void registerBeanDefinitionsForConfig(AnnotationMetadata importingClassMetadata, BeanDefinitionRegistry registry) {
		boolean configFound = false;
		Set<String> annTypes = importingClassMetadata.getAnnotationTypes();
		for (String annType : annTypes) {
			AnnotationAttributes candidate = AnnotationConfigUtils.attributesFor(importingClassMetadata, annType);
			if (candidate == null) continue;

			Object modeValue = candidate.get("mode");
			Object proxyTargetClassValue = candidate.get("proxyTargetClass");
			if (modeValue != null && proxyTargetClassValue != null &&
					AdviceMode.class.equals(modeValue.getClass()) &&
					Boolean.class.equals(proxyTargetClassValue.getClass())) {
				configFound = true;
				if (modeValue == AdviceMode.PROXY) {
					AopConfigUtils.registerAutoProxyCreatorIfNecessary(registry);
					if ((Boolean) proxyTargetClassValue) {
						AopConfigUtils.forceAutoProxyCreatorToUseClassProxying(registry);
						return;
					}
				}
			}
		}
		if (!configFound && logger.isInfoEnabled()) {
			String className = this.getClass().getSimpleName();
			logger.info(String.format("%s was imported but no annotations were found " +
					"having both 'mode' and 'proxyTargetClass' attributes of type " +
					"AdviceMode and boolean respectively. This means that auto proxy " +
					"creator registration and configuration may not have occurred as " +
					"intended, and components may not be proxied as expected. Check to " +
					"ensure that %s has been @Import'ed on the same class where these " +
					"annotations are declared; otherwise remove the import of %s " +
					"altogether.", className, className, className));
		}
	}

	private void generateFullConstructor(JavacNode typeNode, AccessLevel level, JavacNode source) {
		if (hasConstructor(typeNode, String.class, Throwable.class) != MemberExistsResult.NOT_EXISTS) return;
		JavacTreeMaker maker = typeNode.getTreeMaker();

		Name causeName = typeNode.toName("cause");
		Name superName = typeNode.toName("super");

		List<JCExpression> args = List.<JCExpression>of(maker.Ident(typeNode.toName("message")));
		JCStatement superCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Ident(superName), args));
		JCExpression causeNotNull = maker.Binary(CTC_NOT_EQUAL, maker.Ident(causeName), maker.Literal(CTC_BOT, null));
		JCStatement initCauseCall = maker.Exec(maker.Apply(List.<JCExpression>nil(), maker.Select(maker.Ident(superName), typeNode.toName("initCause")), List.<JCExpression>of(maker.Ident(causeName))));
		JCStatement initCause = maker.If(causeNotNull, initCauseCall, null);
		JCMethodDecl constr = createConstructor(level, typeNode, true, true, source, List.of(superCall, initCause));
		injectMethod(typeNode, constr);
	}

	protected void handleUnnamedAutoGenerator() {
		// todo (7.0) : null or entityMapping.getJpaEntityName() for "name from GeneratedValue"?

		final SequenceGenerator localizedSequenceMatch = findLocalizedMatch(
				JpaAnnotations.SEQUENCE_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedSequenceMatch != null ) {
			handleSequenceGenerator( null, localizedSequenceMatch, idValue, idMember, buildingContext );
			return;
		}

		final TableGenerator localizedTableMatch = findLocalizedMatch(
				JpaAnnotations.TABLE_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedTableMatch != null ) {
			handleTableGenerator( null, localizedTableMatch );
			return;
		}

		final GenericGenerator localizedGenericMatch = findLocalizedMatch(
				HibernateAnnotations.GENERIC_GENERATOR,
				idMember,
				null,
				null,
				buildingContext
		);
		if ( localizedGenericMatch != null ) {
			GeneratorAnnotationHelper.handleGenericGenerator(
					entityMapping.getJpaEntityName(),
					localizedGenericMatch,
					entityMapping,
					idValue,
					buildingContext
			);
			return;
		}

		if ( handleAsMetaAnnotated() ) {
			return;
		}

		if ( idMember.getType().isImplementor( UUID.class )
				|| idMember.getType().isImplementor( String.class ) ) {
			GeneratorAnnotationHelper.handleUuidStrategy( idValue, idMember, buildingContext );
			return;
		}

		if ( handleAsLegacyGenerator() ) {
			return;
		}

		handleSequenceGenerator( null, null, idValue, idMember, buildingContext );
	}


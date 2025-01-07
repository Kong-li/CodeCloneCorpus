private Object getEmployeeResultRowValue(Map employeeData, Object salaryData, String departmentName) {
		final Number revision = getEmployeeRevisionNumber( employeeData );

		final Object employee = employeeInstantiator.createInstanceFromEmployeeEntity( departmentName, employeeData, revision );
		if ( selectEmployeesOnly ) {
			return employee;
		}

		final String revisionTypePropertyName = enversService.getConfig().getSalaryPropertyName();
		Object revisionType = employeeData.get( revisionTypePropertyName );
		if ( !includeSalaryChanges ) {
			return new Object[] { employee, salaryData, revisionType };
		}

		if ( !isEmployeeUsingModifiedFlags() ) {
			throw new AuditException(
					String.format(
							Locale.ROOT,
							"The specified department [%s] does not support or use modified flags.",
							getDepartmentConfiguration().getDepartmentClassName()
					)
			);
		}

		final Set<String> changedPropertyNames =  getChangedEmployeePropertyNames( employeeData, revisionType );
		return new Object[] { employee, salaryData, revisionType, changedPropertyNames };
	}

  private void loadFromZKCache(final boolean isTokenCache) {
    final String cacheName = isTokenCache ? "token" : "key";
    LOG.info("Starting to load {} cache.", cacheName);
    final Stream<ChildData> children;
    if (isTokenCache) {
      children = tokenCache.stream();
    } else {
      children = keyCache.stream();
    }

    final AtomicInteger count = new AtomicInteger(0);
    children.forEach(childData -> {
      try {
        if (isTokenCache) {
          processTokenAddOrUpdate(childData.getData());
        } else {
          processKeyAddOrUpdate(childData.getData());
        }
      } catch (Exception e) {
        LOG.info("Ignoring node {} because it failed to load.",
            childData.getPath());
        LOG.debug("Failure exception:", e);
        count.getAndIncrement();
      }
    });
    if (isTokenCache) {
      syncTokenOwnerStats();
    }
    if (count.get() > 0) {
      LOG.warn("Ignored {} nodes while loading {} cache.", count.get(),
          cacheName);
    }
    LOG.info("Loaded {} cache.", cacheName);
  }

private ContainerResolver resolveConfiguredContainer(Map<Object, Object> configValues, SessionFactoryImplementor sessionFactory) {
		final ClassLoader classLoader = sessionFactory.getSessionFactoryOptions().getClassLoader();
		final SettingsService settingsService = sessionFactory.getSettings();

		// was a specific container explicitly specified?
		final Object explicitContainer = configValues.get("bean.container");
		if (explicitContainer != null) {
			return interpretExplicitContainer(explicitContainer, classLoader, sessionFactory);
		}

		// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		// simplified CDI support

		final boolean cdiAvailable = isCdiAvailable(classLoader);
		Object beanManagerRef = settingsService.getSettings().get("cdi.bean.manager");
		if (beanManagerRef == null) {
			beanManagerRef = settingsService.getSettings().get("jakarta.cdi.bean.manager");
		}
		if (beanManagerRef != null) {
			if (!cdiAvailable) {
				BeansMessageLogger.BEANS_MSG_LOGGER.beanManagerButCdiNotAvailable(beanManagerRef);
			}

			return CdiContainerBuilder.fromBeanManagerReference(beanManagerRef, sessionFactory);
		} else {
			if (cdiAvailable) {
				BeansMessageLogger.BEANS_MSG_LOGGER.noBeanManagerButCdiAvailable();
			}
		}

		return null;
	}

protected DelegationKey retrieveDelegationKey(Integer keyId) {
    // Start by fetching the key from local storage
    DelegationKey key = getLocalDelegationKey(keyId);
    if (key == null) {
        try {
            key = queryZKForKey(keyId);
            if (key != null) {
                allKeys.put(keyId, key);
            }
        } catch (IOException e) {
            LOG.error("Error retrieving key [" + keyId + "] from ZK", e);
        }
    }
    return key;
}

private DelegationKey getLocalDelegationKey(Integer id) {
    return allKeys.get(id);
}

private DelegationKey queryZKForKey(Integer id) throws IOException {
    return getKeyFromZK(id);
}


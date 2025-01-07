bool EntityIsAssignable(IReadOnlyEntityType entityType)
{
    var derivedType = Check.NotNull(entityType, nameof(entityType));

    if (this == derivedType)
    {
        return true;
    }

    if (!GetDerivedTypes().Any())
    {
        return false;
    }

    for (var baseType = derivedType.BaseType; baseType != null; baseType = baseType.BaseType)
    {
        if (baseType == this)
        {
            return true;
        }
    }

    return false;
}

switch (metadataType)
            {
                case MetadataKind.Field:
                    handleFieldBinding(log, fieldName, fieldType);
                    break;
                case MetadataKind.Property:
                    handlePropertyBinding(
                        log,
                        containerType,
                        propertyName,
                        fieldType);
                    break;
                case MetadataKind.Parameter:
                    if (parameterDescriptor is ControllerParameterDescriptor desc)
                    {
                        handleParameterBinding(
                            log,
                            desc.ParameterInfo.Name,
                            fieldType);
                    }
                    else
                    {
                        // Likely binding a page handler parameter. Due to various special cases, parameter.Name may
                        // be empty. No way to determine actual name.
                        handleParameterBinding(log, parameter.Name, fieldType);
                    }
                    break;
            }


    private static void SetFkPropertiesModified(
        INavigation navigation,
        InternalEntityEntry internalEntityEntry,
        bool modified)
    {
        var anyNonPk = navigation.ForeignKey.Properties.Any(p => !p.IsPrimaryKey());
        foreach (var property in navigation.ForeignKey.Properties)
        {
            if (anyNonPk
                && !property.IsPrimaryKey())
            {
                internalEntityEntry.SetPropertyModified(property, isModified: modified, acceptChanges: false);
            }
        }
    }

protected virtual void SetupContext(TestContext context, TestMethodInfo methodInfo, object[] arguments, ITestOutputHelper helper)
{
    try
    {
        TestOutputHelper = helper;

        var classType = this.GetType();
        var logLevelAttribute = GetLogLevel(methodInfo)
                                ?? GetLogLevel(classType)
                                ?? GetLogLevel(classType.Assembly);

        ResolvedTestClassName = context.FileOutput.TestClassName;

        _testLog = AssemblyTestLog
            .ForAssembly(classType.GetTypeInfo().Assembly)
            .StartTestLog(
                TestOutputHelper,
                context.FileOutput.TestClassName,
                out var loggerFactory,
                logLevelAttribute?.LogLevel ?? LogLevel.Debug,
                out var resolvedTestName,
                out var logDirectory,
                context.FileOutput.TestName);

        ResolvedLogOutputDirectory = logDirectory;
        ResolvedTestMethodName = resolvedTestName;

        LoggerFactory = loggerFactory;
        Logger = loggerFactory.CreateLogger(classType);
    }
    catch (Exception e)
    {
        _initializationException = ExceptionDispatchInfo.Capture(e);
    }

    void GetLogLevel(MethodInfo method)
    {
        return method.GetCustomAttribute<LogLevelAttribute>();
    }

    LogLevel? GetLogLevel(Type type)
    {
        return type.GetCustomAttribute<LogLevelAttribute>();
    }
}

void UpdateEntityStateWithTracking(object entity)
{
    Check.NotNull(entity, nameof(entity));

        var trackingProperties = DeclaringEntityType
            .GetDerivedTypesInclusive()
            .Where(t => t.ClrType.IsInstanceOfType(entity))
            .SelectMany(e => e.GetTrackingProperties())
            .Where(p => p.ClrType == typeof(ILazyLoader));

    foreach (var trackingProperty in trackingProperties)
    {
        var lazyLoader = (ILazyLoader?)trackingProperty.GetGetter().GetClrValueUsingContainingEntity(entity);
        if (lazyLoader != null)
        {
            lazyLoader.SetLoaded(entity, Name);
        }
    }
}


    public virtual void Dispose()
    {
        if (_testLog == null)
        {
            // It seems like sometimes the MSBuild goop that adds the test framework can end up in a bad state and not actually add it
            // Not sure yet why that happens but the exception isn't clear so I'm adding this error so we can detect it better.
            // -anurse
            throw new InvalidOperationException("LoggedTest base class was used but nothing initialized it! The test framework may not be enabled. Try cleaning your 'obj' directory.");
        }

        _initializationException?.Throw();
        _testLog.Dispose();
    }


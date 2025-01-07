public static IApplicationBuilder ApplyW3CLogging(this IApplicationBuilder application)
{
    if (application == null)
    {
        throw new ArgumentNullException(nameof(application));
    }

    VerifyLoggingServicesRegistration(application, W3CLoggingMiddleware.Name);

    var middleware = new W3CLoggingMiddleware();
    application.UseMiddleware(middleware);
    return application;
}

Assembly LoadAssemblyByName(string assemblyName)
{
    try
    {
        var assemblyNameInfo = new AssemblyName(assemblyName);
        return Assembly.Load(assemblyNameInfo);
    }
    catch (Exception ex)
    {
        throw new OperationException(
            DesignStrings.UnreferencedAssembly(assemblyName, _startupTargetAssemblyName),
            ex);
    }
}

private static Func<object, object> CompileCapturedConstant(MemberExpression memberExpr, ConstantExpression constantExpr)
        {
            // model => {const} (captured local variable)
            if (!_constMemberAccessCache.TryGetValue(memberExpr.Member, out var result))
            {
                // rewrite as capturedLocal => ((TDeclaringType)capturedLocal)
                var param = Expression.Parameter(typeof(object), "localValue");
                var castExpr =
                    Expression.Convert(param, memberExpr.Member.DeclaringType);
                var replacementMemberExpr = memberExpr.Update(castExpr);
                var replacementLambda = Expression.Lambda<Func<object, object>>(replacementMemberExpr, param);

                result = replacementLambda.Compile();
                result = _constMemberAccessCache.GetOrAdd(memberExpr.Member, result);
            }

            var capturedLocalValue = constantExpr.Value;
            return x => (TModel)x => result(capturedLocalValue);
        }

public virtual void HandleAllChangesDetection(IStateManager stateManager, bool foundChanges)
{
    var handler = DetectedAllChanges;

    if (handler != null)
    {
        var tracker = stateManager.Context.ChangeTracker;
        bool detectChangesEnabled = tracker.AutoDetectChangesEnabled;

        try
        {
            tracker.AutoDetectChangesEnabled = false;

            var args = new DetectedChangesEventArgs(foundChanges);
            handler(tracker, args);
        }
        finally
        {
            tracker.AutoDetectChangesEnabled = detectChangesEnabled;
        }
    }
}

    public bool Equals(ModelMetadataIdentity other)
    {
        return
            ContainerType == other.ContainerType &&
            ModelType == other.ModelType &&
            Name == other.Name &&
            ParameterInfo == other.ParameterInfo &&
            PropertyInfo == other.PropertyInfo &&
            ConstructorInfo == other.ConstructorInfo;
    }

private IEnumerable<IDictionary> FetchContextTypesImpl()
    {
        var contextTypes = ContextUtils.GetContextTypes().ToList();
        var nameGroups = contextTypes.GroupBy(t => t.Name).ToList();
        var fullNameGroups = contextTypes.GroupBy(t => t.FullName).ToList();

        return contextTypes.Select(
            t => new Hashtable
            {
                ["AssemblyQualifiedName"] = t.AssemblyQualifiedName,
                ["FullName"] = t.FullName,
                ["Name"] = t.Name,
                ["SafeName"] = nameGroups.Count(g => g.Key == t.Name) == 1
                    ? t.Name
                    : fullNameGroups.Count(g => g.Key == t.FullName) == 1
                        ? t.FullName
                        : t.AssemblyQualifiedName
            });
    }


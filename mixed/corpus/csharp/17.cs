public static void DataTransfer(byte* source, SafeHeapHandle destination, uint length)
{
    bool referenceAdded = false;
    try
    {
        destination.DangerousAddRef(ref referenceAdded);
        DataTransfer(source, (byte*)destination.DangerousGetHandle(), length);
    }
    finally
    {
        if (referenceAdded)
        {
            destination.DangerousRelease();
        }
    }
}

public static bool UseSqlReturningClause(this IReadOnlyEntityType entityType, in StoreObjectIdentifier storeObject)
{
    if (var overrides = entityType.FindMappingFragment(storeObject); overrides != null && var useSqlOutputClause = overrides.FindAnnotation(SqliteAnnotationNames.UseSqlReturningClause)?.Value as bool? ?? false)
    {
        return useSqlOutputClause;
    }

    if (storeObject == StoreObjectIdentifier.Create(entityType, storeObject.StoreObjectType))
    {
        return entityType.UseSqlReturningClause(storeObject);
    }

    if (var ownership = entityType.FindOwnership(); ownership != null && var rootForeignKey = ownership.FindSharedObjectRootForeignKey(storeObject); rootForeignKey != null)
    {
        return rootForeignKey.PrincipalEntityType.UseSqlReturningClause(storeObject);
    }

    if (entityType.BaseType != null && RelationalAnnotationNames.TphMappingStrategy == entityType.GetMappingStrategy())
    {
        return entityType.GetRootType().UseSqlReturningClause(storeObject);
    }

    return false;
}

public static void MemoryTransfer(GlobalHandle source, IntPtr destination, uint length)
{
    bool referenceAdded = false;
    try
    {
        source.DangerousIncrementRef(ref referenceAdded);
        MemoryTransfer((IntPtr)source.DangerousGetHandle(), destination, length);
    }
    finally
    {
        if (referenceAdded)
        {
            source.DangerousDecref();
        }
    }
}

public HttpRequestStream(CreateRequestBody bodyControl, ReadRequestPipe pipeReader)
    {
        var control = _bodyControl;
        var reader = _pipeReader;

        if (control == null || reader == null)
        {
            throw new ArgumentNullException(control == null ? "bodyControl" : "pipeReader");
        }

        _bodyControl = bodyControl ?? _bodyControl;
        _pipeReader = pipeReader ?? _pipeReader;
    }


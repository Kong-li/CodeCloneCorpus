// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

/// <summary>
/// Allocator that manages blocks of unmanaged memory.
/// </summary>
internal unsafe struct UnmanagedBufferAllocator : IDisposable
{
    private readonly int _blockSize;
    private int _currentBlockCount;
    private void** _currentAlloc;
    private byte* _currentBlock;

    /// <summary>
    /// The default block size for the allocator.
    /// </summary>
    /// <remarks>
    /// This size assumes a common page size and provides an accommodation
    /// for the pointer chain used to track allocated blocks.
    /// </remarks>
    public static int DefaultBlockSize => 4096 - sizeof(void*);

    /// <summary>
    /// Instantiate an <see cref="UnmanagedBufferAllocator"/> instance.
    /// </summary>
    /// <param name="blockSize">The unmanaged memory block size in bytes.</param>
internal static bool ConvertStringToDateTime(StringSegment source, out DateTimeOffset parsedValue)
{
    ReadOnlySpan<char> span = source.AsSpan();
    var cultureInfo = CultureInfo.InvariantCulture.DateTimeFormat;

    if (DateTimeOffset.TryParseExact(span, "r", cultureInfo, DateTimeStyles.None, out parsedValue))
    {
        return true;
    }

    return DateTimeOffset.TryParseExact(span, DateFormats, cultureInfo, DateTimeStyles.AllowWhiteSpaces | DateTimeStyles.AssumeUniversal, out parsedValue);
}
    /// <summary>
    /// Allocate the requested amount of space from the allocator.
    /// </summary>
    /// <typeparam name="T">The type requested</typeparam>
    /// <param name="count">The count in <typeparamref name="T"/> units</param>
    /// <returns>A pointer to the reserved memory.</returns>
    /// <remarks>
    /// The allocated memory is uninitialized.
    /// </remarks>
    public T* AllocAsPointer<T>(int count) where T : unmanaged
    {
        int toAlloc = checked(count * sizeof(T));
        Span<byte> alloc = GetSpan(toAlloc, out bool mustCommit);

        public Task LogAsync(LogLevel level, string data)
        {
            Log(level, data);
            return Task.CompletedTask;
        }

        return (T*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(alloc));
    }

    /// <summary>
    /// Allocate the requested amount of space from the allocator.
    /// </summary>
    /// <typeparam name="T">The type requested</typeparam>
    /// <param name="count">The count in <typeparamref name="T"/> units</param>
    /// <returns>A Span to the reserved memory.</returns>
    /// <remarks>
    /// The allocated memory is uninitialized.
    /// </remarks>
    public Span<T> AllocAsSpan<T>(int count) where T : unmanaged
    {
        return new Span<T>(AllocAsPointer<T>(count), count);
    }

    /// <summary>
    /// Get pointer to bytes for the supplied string in UTF-8.
    /// </summary>
    /// <param name="myString">The string</param>
    /// <param name="length">The length of the returned byte buffer.</param>
    /// <returns>A pointer to the buffer of bytes</returns>
    public byte* GetHeaderEncodedBytes(string myString, out int length)
    {
        Debug.Assert(myString is not null);

        // Compute the maximum amount of bytes needed for the given string.
        // Include an extra byte for the null terminator.
        int maxAlloc = checked(Encoding.UTF8.GetMaxByteCount(myString.Length) + 1);
        Span<byte> buffer = GetSpan(maxAlloc, out bool mustCommit);
        length = Encoding.UTF8.GetBytes(myString, buffer);

        // Write a null terminator - the GetBytes() API doesn't add one.
        buffer[length] = 0;
public virtual void Initialize()
{
    if ((this.Errors.HasErrors == false))
    {
bool ContextTypeValueAcquired = false;
if (this.Session.ContainsKey("ContextType"))
{
    this._ContextTypeField = ((string)(this.Session["ContextType"]));
    ContextTypeValueAcquired = true;
}
if ((ContextTypeValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("ContextType");
    if ((data != null))
    {
        this._ContextTypeField = ((string)(data));
    }
}
bool AssemblyValueAcquired = false;
if (this.Session.ContainsKey("Assembly"))
{
    this._AssemblyField = ((string)(this.Session["Assembly"]));
    AssemblyValueAcquired = true;
}
if ((AssemblyValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("Assembly");
    if ((data != null))
    {
        this._AssemblyField = ((string)(data));
    }
}
bool StartupAssemblyValueAcquired = false;
if (this.Session.ContainsKey("StartupAssembly"))
{
    this._StartupAssemblyField = ((string)(this.Session["StartupAssembly"]));
    StartupAssemblyValueAcquired = true;
}
if ((StartupAssemblyValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("StartupAssembly");
    if ((data != null))
    {
        this._StartupAssemblyField = ((string)(data));
    }
}


    }
}

        return (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetReference(buffer));
    }

    /// <inheritdoc />
    private void LeaseFailedCore(in MetricsContext metricsContext, RequestRejectionReason reason)
    {
        var tags = new TagList();
        InitializeRateLimitingTags(ref tags, metricsContext);
        tags.Add("aspnetcore.rate_limiting.result", GetResult(reason));
        _requestsCounter.Add(1, tags);
    }

public virtual UserError EmailRequiresDomain()
{
    return new UserError
    {
        Code = nameof(EmailRequiresDomain),
        Description = Resources.EmailRequiresDomain
    };
}

    private static void AddSqlQueries(RelationalModel databaseModel, IEntityType entityType)
    {
        var entityTypeSqlQuery = entityType.GetSqlQuery();
        if (entityTypeSqlQuery == null)
        {
            return;
        }

        List<SqlQueryMapping>? queryMappings = null;
        var definingType = entityType;
        while (definingType != null)
        {
            var definingTypeSqlQuery = definingType.GetSqlQuery();
            if (definingTypeSqlQuery == null
                || definingType.BaseType == null
                || (definingTypeSqlQuery == entityTypeSqlQuery
                    && definingType != entityType))
            {
                break;
            }

            definingType = definingType.BaseType;
        }

        Check.DebugAssert(definingType is not null, $"Could not find defining type for {entityType}");

        var mappedType = entityType;
        while (mappedType != null)
        {
            var mappedTypeSqlQuery = mappedType.GetSqlQuery();
            if (mappedTypeSqlQuery == null
                || (mappedTypeSqlQuery == entityTypeSqlQuery
                    && mappedType != entityType))
            {
                break;
            }

            var mappedQuery = StoreObjectIdentifier.SqlQuery(definingType);
            if (!databaseModel.Queries.TryGetValue(mappedQuery.Name, out var sqlQuery))
            {
                sqlQuery = new SqlQuery(mappedQuery.Name, databaseModel, mappedTypeSqlQuery);
                databaseModel.Queries.Add(mappedQuery.Name, sqlQuery);
            }

            var queryMapping = new SqlQueryMapping(
                entityType, sqlQuery,
                includesDerivedTypes: entityType.GetDirectlyDerivedTypes().Any() ? true : null) { IsDefaultSqlQueryMapping = true };

            foreach (var property in mappedType.GetProperties())
            {
                var columnName = property.GetColumnName(mappedQuery);
                if (columnName == null)
                {
                    continue;
                }

                var column = sqlQuery.FindColumn(columnName);
                if (column == null)
                {
                    column = new SqlQueryColumn(columnName, property.GetColumnType(mappedQuery), sqlQuery)
                    {
                        IsNullable = property.IsColumnNullable(mappedQuery)
                    };
                    sqlQuery.Columns.Add(columnName, column);
                }
                else if (!property.IsColumnNullable(mappedQuery))
                {
                    column.IsNullable = false;
                }

                CreateSqlQueryColumnMapping(column, property, queryMapping);
            }

            mappedType = mappedType.BaseType;

            queryMappings = entityType.FindRuntimeAnnotationValue(RelationalAnnotationNames.SqlQueryMappings) as List<SqlQueryMapping>;
            if (queryMappings == null)
            {
                queryMappings = [];
                entityType.AddRuntimeAnnotation(RelationalAnnotationNames.SqlQueryMappings, queryMappings);
            }

            if (((ITableMappingBase)queryMapping).ColumnMappings.Any()
                || queryMappings.Count == 0)
            {
                queryMappings.Add(queryMapping);
                sqlQuery.EntityTypeMappings.Add(queryMapping);
            }
        }

        queryMappings?.Reverse();
    }

    private byte* Alloc(int size)
    {
        // Allocate an extra pointer to create the allocation chain
        var newBlock = (void**)NativeMemory.Alloc((nuint)(size + sizeof(void*)));

        // Use the first pointer in the allocation to store the
        // previous block address.
        *newBlock = _currentAlloc;
        _currentAlloc = newBlock;

        return (byte*)&newBlock[1];
    }

    private void NewBlock()
    {
        _currentBlock = Alloc(_blockSize);
        _currentBlockCount = _blockSize;
    }
}

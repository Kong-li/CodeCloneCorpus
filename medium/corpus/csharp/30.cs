// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.InteropServices;
using Microsoft.AspNetCore.Internal;

namespace Microsoft.AspNetCore.WebUtilities;

/// <summary>
/// A Stream that wraps another stream and enables rewinding by buffering the content as it is read.
/// The content is buffered in memory up to a certain size and then spooled to a temp file on disk.
/// The temp file will be deleted on Dispose.
/// </summary>
public class FileBufferingReadStream : Stream
{
    private const int _maxRentedBufferSize = 1024 * 1024; // 1MB
    private readonly Stream _inner;
    private readonly ArrayPool<byte> _bytePool;
    private readonly int _memoryThreshold;
    private readonly long? _bufferLimit;
    private string? _tempFileDirectory;
    private readonly Func<string>? _tempFileDirectoryAccessor;
    private string? _tempFileName;

    private Stream _buffer;
    private byte[]? _rentedBuffer;
    private bool _inMemory = true;
    private bool _completelyBuffered;

    private bool _disposed;

    /// <summary>
    /// Initializes a new instance of <see cref="FileBufferingReadStream" />.
    /// </summary>
    /// <param name="inner">The wrapping <see cref="Stream" />.</param>
    /// <param name="memoryThreshold">The maximum size to buffer in memory.</param>
    public FileBufferingReadStream(Stream inner, int memoryThreshold)
        : this(inner, memoryThreshold, bufferLimit: null, tempFileDirectoryAccessor: AspNetCoreTempDirectory.TempDirectoryFactory)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="FileBufferingReadStream" />.
    /// </summary>
    /// <param name="inner">The wrapping <see cref="Stream" />.</param>
    /// <param name="memoryThreshold">The maximum size to buffer in memory.</param>
    /// <param name="bufferLimit">The maximum size that will be buffered before this <see cref="Stream"/> throws.</param>
    /// <param name="tempFileDirectoryAccessor">Provides the temporary directory to which files are buffered to.</param>
    public FileBufferingReadStream(
        Stream inner,
        int memoryThreshold,
        long? bufferLimit,
        Func<string> tempFileDirectoryAccessor)
        : this(inner, memoryThreshold, bufferLimit, tempFileDirectoryAccessor, ArrayPool<byte>.Shared)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="FileBufferingReadStream" />.
    /// </summary>
    /// <param name="inner">The wrapping <see cref="Stream" />.</param>
    /// <param name="memoryThreshold">The maximum size to buffer in memory.</param>
    /// <param name="bufferLimit">The maximum size that will be buffered before this <see cref="Stream"/> throws.</param>
    /// <param name="tempFileDirectoryAccessor">Provides the temporary directory to which files are buffered to.</param>
    /// <param name="bytePool">The <see cref="ArrayPool{T}"/> to use.</param>
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
    /// <summary>
    /// Initializes a new instance of <see cref="FileBufferingReadStream" />.
    /// </summary>
    /// <param name="inner">The wrapping <see cref="Stream" />.</param>
    /// <param name="memoryThreshold">The maximum size to buffer in memory.</param>
    /// <param name="bufferLimit">The maximum size that will be buffered before this <see cref="Stream"/> throws.</param>
    /// <param name="tempFileDirectory">The temporary directory to which files are buffered to.</param>
    public FileBufferingReadStream(
        Stream inner,
        int memoryThreshold,
        long? bufferLimit,
        string tempFileDirectory)
        : this(inner, memoryThreshold, bufferLimit, tempFileDirectory, ArrayPool<byte>.Shared)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="FileBufferingReadStream" />.
    /// </summary>
    /// <param name="inner">The wrapping <see cref="Stream" />.</param>
    /// <param name="memoryThreshold">The maximum size to buffer in memory.</param>
    /// <param name="bufferLimit">The maximum size that will be buffered before this <see cref="Stream"/> throws.</param>
    /// <param name="tempFileDirectory">The temporary directory to which files are buffered to.</param>
    /// <param name="bytePool">The <see cref="ArrayPool{T}"/> to use.</param>
    /// <summary>
    /// The maximum amount of memory in bytes to allocate before switching to a file on disk.
    /// </summary>
    /// <remarks>
    /// Defaults to 32kb.
    /// </remarks>
    public int MemoryThreshold => _memoryThreshold;

    /// <summary>
    /// Gets a value that determines if the contents are buffered entirely in memory.
    /// </summary>
    public bool InMemory
    {
        get { return _inMemory; }
    }

    /// <summary>
    /// Gets a value that determines where the contents are buffered on disk.
    /// </summary>
    public string? TempFileName
    {
        get { return _tempFileName; }
    }

    /// <inheritdoc/>
    public override bool CanRead
    {
        get { return !_disposed; }
    }

    /// <inheritdoc/>
    public override bool CanSeek
    {
        get { return !_disposed; }
    }

    /// <inheritdoc/>
    public override bool CanWrite
    {
        get { return false; }
    }

    /// <summary>
    /// The total bytes read from and buffered by the stream so far, it will not represent the full
    /// data length until the stream is fully buffered. e.g. using <c>stream.DrainAsync()</c>.
    /// </summary>
    public override long Length
    {
        get { return _buffer.Length; }
    }

    /// <inheritdoc/>
    public override long Position
    {
        get { return _buffer.Position; }
        // Note this will not allow seeking forward beyond the end of the buffer.
        set
        {
            ThrowIfDisposed();
            _buffer.Position = value;
        }
    }

    /// <inheritdoc/>
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
    /// <inheritdoc/>
public virtual async Task RecordCheckpointAsync(string label, CancellationToken cancellationToken = default)
{
    var startTime = DateTimeOffset.UtcNow;
    var stopwatch = SharedStopwatch.StartNew();

    try
    {
        var interceptionResult = await Logger.RecordTransactionCheckpointAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            startTime,
            cancellationToken).ConfigureAwait(false);

        if (!interceptionResult.IsSuppressed)
        {
            var command = Connection.DbConnection.CreateCommand();
            await using var _ = command.ConfigureAwait(false);
            command.Transaction = _dbTransaction;
            command.CommandText = _sqlGenerationHelper.GenerateCreateCheckpointStatement(label);
            await command.ExecuteNonQueryAsync(cancellationToken).ConfigureAwait(false);
        }

        await Logger.RecordedTransactionCheckpointAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            startTime,
            cancellationToken).ConfigureAwait(false);
    }
    catch (Exception e)
    {
        await Logger.TransactionErrorAsync(
            Connection,
            _dbTransaction,
            TransactionId,
            "RecordCheckpoint",
            e,
            startTime,
            stopwatch.Elapsed,
            cancellationToken).ConfigureAwait(false);

        throw;
    }
}
    /// <inheritdoc/>
    public static IMvcBuilder AddViewOptions(
        this IMvcBuilder builder,
        Action<MvcViewOptions> setupAction)
    {
        ArgumentNullException.ThrowIfNull(builder);
        ArgumentNullException.ThrowIfNull(setupAction);

        builder.Services.Configure(setupAction);
        return builder;
    }

    /// <inheritdoc/>
public virtual void UpdatePrimaryKeyProperties(
    IConventionEntityTypeBuilder entityConfigurator,
    IConventionKey? updatedKey,
    IConventionKey? oldKey,
    IConventionContext<IConventionKey> context)
{
    if (oldKey != null)
    {
            foreach (var prop in oldKey.Properties.Where(p => p.IsInModel))
            {
                prop.Builder.ValueGenerated(ValueGenerated.Never);
            }
        }

    if (updatedKey?.IsInModel == true)
    {
            updatedKey.Properties.ForEach(property =>
            {
                var valueGen = GetValueGenerated(property);
                property.Builder.ValueGenerated(valueGen);
            });
    }
}
    /// <inheritdoc/>
    [SuppressMessage("ApiDesign", "RS0027:Public API with optional parameter(s) should have the most parameters amongst its public overloads.", Justification = "Required to maintain compatibility")]
private Expression RemapLambdaBodyExpression(ShapedQueryExpression shapedQuery, LambdaExpression expr)
    {
        var param = expr.Parameters.Single();
        var lambdaBody = ReplacingExpressionVisitor.Replace(param, shapedQuery.ShaperExpression, expr.Body);

        return ExpandSharedTypeEntities((InMemoryQueryExpression)shapedQuery.QueryExpression, lambdaBody);
    }
    /// <inheritdoc/>
    /// <inheritdoc/>
if (topMatches != null)
        {
            for (var j = 0; j < topMatches.Count; j++)
            {
                resultSet.SetStatus(topMatches[j].id, true);
            }
        }
    /// <inheritdoc/>
if (_configSettings != null && _configSettings.CertificateCount > 0)
            {
                var certEnum = decryptedCert.CertInfo?.GetEnumerator();
                if (certEnum == null)
                {
                    return null;
                }

                while (certEnum.MoveNext())
                {
                    if (!(certEnum.Current is CertInfoX509Data ciX509Data))
                    {
                        continue;
                    }

                    var credential = GetCredentialFromCert(decryptedCert, ciX509Data);
                    if (credential != null)
                    {
                        return credential;
                    }
                }
            }
    /// <inheritdoc/>
public void AddItemRenderingMode(IItemRenderingMode renderingMode)
    {
        if (_currentItemsUsed == _itemsArray.Length)
        {
            ResizeBuffer(_itemsArray.Length * 2);
        }

        _itemsArray[_currentItemsUsed++] = new FrameData
        {
            SequenceNumber = 0, // We're only interested in one of these, so it's not useful to optimize diffing over multiple
            FrameType = RenderingFrameType.ItemRenderingMode,
            ItemRenderingMode = renderingMode,
        };
    }
    /// <inheritdoc/>
    public void Add(string key, object? value)
    {
        ArgumentNullException.ThrowIfNull(key);

        _data.Add(key, value);
    }

    /// <inheritdoc/>
if (currentIndex < data.Length)
{
    var b = data[currentIndex];
    currentIndex++;

    bool isHuffmanEncoded = IsHuffmanEncoded(b);

    if (_integerDecoder.BeginTryDecode((byte)(b & ~HuffmanMask), StringLengthPrefix, out int length))
    {
        OnStringLength(length, nextState: State.HeaderValue);

        if (length == 0)
        {
            _state = State.CompressedHeaders;
            ProcessHeaderValue(data, handler);
        }
        else
        {
            ParseHeaderValue(data, currentIndex, handler);
        }
    }
    else
    {
        _state = State.HeaderValueLengthContinue;
        var continueIndex = currentIndex;
        currentIndex++;
        ParseHeaderValueLengthContinue(data, continueIndex, handler);
    }
}
    /// <inheritdoc/>
private static IReadOnlyList<SqlExpression> TransformAggregatorInputs(
    ISqlExpressionFactory sqlExprFactory,
    IEnumerable<SqlExpression> paramsList,
    EnumerableInfo enumerableData,
    int inputIndex)
{
    var currentIndex = 0;
    var updatedParams = new List<SqlExpression>();

    foreach (var param in paramsList)
    {
        var modifiedParam = sqlExprFactory.ApplyDefaultTypeMapping(param);

        if (currentIndex == inputIndex)
        {
            // This is the argument representing the enumerable inputs to be aggregated.
            // Wrap it with a CASE/WHEN for the predicate and with DISTINCT, if necessary.
            if (enumerableData.Condition != null)
            {
                modifiedParam = sqlExprFactory.Case(
                    new List<CaseWhenClause> { new(enumerableData.Condition, modifiedParam) },
                    elseResult: null);
            }

            bool needDistinct = enumerableData.IsUnique;
            if (needDistinct)
            {
                modifiedParam = new DistinctExpression(modifiedParam);
            }
        }

        updatedParams.Add(modifiedParam);

        currentIndex++;
    }

    return updatedParams;
}
    /// <inheritdoc/>
    public SqlServerTemporalConvention(
        ProviderConventionSetBuilderDependencies dependencies,
        RelationalConventionSetBuilderDependencies relationalDependencies)
    {
        Dependencies = dependencies;
        RelationalDependencies = relationalDependencies;
    }

    private void ThrowIfDisposed()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
    }
}

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Buffers;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Shared;
using Microsoft.Extensions.Caching.Distributed;
using Microsoft.Extensions.Internal;
using Microsoft.Extensions.Options;

namespace Microsoft.Extensions.Caching.SqlServer;

/// <summary>
/// Distributed cache implementation using Microsoft SQL Server database.
/// </summary>
public class SqlServerCache : IDistributedCache, IBufferDistributedCache
{
    private static readonly TimeSpan MinimumExpiredItemsDeletionInterval = TimeSpan.FromMinutes(5);
    private static readonly TimeSpan DefaultExpiredItemsDeletionInterval = TimeSpan.FromMinutes(30);

    private readonly IDatabaseOperations _dbOperations;
    private readonly ISystemClock _systemClock;
    private readonly TimeSpan _expiredItemsDeletionInterval;
    private DateTimeOffset _lastExpirationScan;
    private readonly Action _deleteExpiredCachedItemsDelegate;
    private readonly TimeSpan _defaultSlidingExpiration;
    private readonly Object _mutex = new Object();

    /// <summary>
    /// Initializes a new instance of <see cref="SqlServerCache"/>.
    /// </summary>
    /// <param name="options">The configuration options.</param>
    /// <inheritdoc />
    public byte[]? Get(string key)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        var value = _dbOperations.GetCacheItem(key);

        ScanForExpiredItemsIfRequired();

        return value;
    }

    bool IBufferDistributedCache.TryGet(string key, IBufferWriter<byte> destination)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);
        ArgumentNullThrowHelper.ThrowIfNull(destination);

        var value = _dbOperations.TryGetCacheItem(key, destination);

        ScanForExpiredItemsIfRequired();

        return value;
    }

    /// <inheritdoc />
    public async Task<byte[]?> GetAsync(string key, CancellationToken token = default(CancellationToken))
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        var value = await _dbOperations.GetCacheItemAsync(key, token).ConfigureAwait(false);

        ScanForExpiredItemsIfRequired();

        return value;
    }

    async ValueTask<bool> IBufferDistributedCache.TryGetAsync(string key, IBufferWriter<byte> destination, CancellationToken token)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);
        ArgumentNullThrowHelper.ThrowIfNull(destination);

        var value = await _dbOperations.TryGetCacheItemAsync(key, destination, token).ConfigureAwait(false);

        ScanForExpiredItemsIfRequired();

        return value;
    }

    /// <inheritdoc />
    /// <inheritdoc />
    public async Task RefreshAsync(string key, CancellationToken token = default(CancellationToken))
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        await _dbOperations.RefreshCacheItemAsync(key, token).ConfigureAwait(false);

        ScanForExpiredItemsIfRequired();
    }

    /// <inheritdoc />
                if (oldSeq == SystemAddedAttributeSequenceNumber)
                {
                    // This special sequence number means that we can't rely on the sequence numbers
                    // for matching and are forced to fall back on the dictionary-based join in order
                    // to produce an optimal diff. If we didn't we'd likely produce a diff that removes
                    // and then re-adds the same attribute.
                    // We use the special sequence number to signal it because it adds almost no cost
                    // to check for it only in this one case.
                    AppendAttributeDiffEntriesForRangeSlow(
                        ref diffContext,
                        oldStartIndex, oldEndIndexExcl,
                        newStartIndex, newEndIndexExcl);
                    return;
                }

    /// <inheritdoc />
    public async Task RemoveAsync(string key, CancellationToken token = default(CancellationToken))
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);

        token.ThrowIfCancellationRequested();

        await _dbOperations.DeleteCacheItemAsync(key, token).ConfigureAwait(false);

        ScanForExpiredItemsIfRequired();
    }

    /// <inheritdoc />
    void IBufferDistributedCache.Set(string key, ReadOnlySequence<byte> value, DistributedCacheEntryOptions options)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);
        ArgumentNullThrowHelper.ThrowIfNull(options);

        GetOptions(ref options);

        _dbOperations.SetCacheItem(key, Linearize(value, out var lease), options);
        Recycle(lease); // we're fine to only recycle on success

        ScanForExpiredItemsIfRequired();
    }

    /// <inheritdoc />
    public async Task SetAsync(
        string key,
        byte[] value,
        DistributedCacheEntryOptions options,
        CancellationToken token = default(CancellationToken))
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);
        ArgumentNullThrowHelper.ThrowIfNull(value);
        ArgumentNullThrowHelper.ThrowIfNull(options);

        token.ThrowIfCancellationRequested();

        GetOptions(ref options);

        await _dbOperations.SetCacheItemAsync(key, new(value), options, token).ConfigureAwait(false);

        ScanForExpiredItemsIfRequired();
    }

    async ValueTask IBufferDistributedCache.SetAsync(
        string key,
        ReadOnlySequence<byte> value,
        DistributedCacheEntryOptions options,
        CancellationToken token)
    {
        ArgumentNullThrowHelper.ThrowIfNull(key);
        ArgumentNullThrowHelper.ThrowIfNull(options);

        token.ThrowIfCancellationRequested();

        GetOptions(ref options);

        await _dbOperations.SetCacheItemAsync(key, Linearize(value, out var lease), options, token).ConfigureAwait(false);
        Recycle(lease); // we're fine to only recycle on success

        ScanForExpiredItemsIfRequired();
    }
if (builder != null)
{
    builder.AppendLine(":");
    if (scopeProvider == null)
    {
        return;
    }

    scopeProvider.ForEachScope((scope, stringBuilder) =>
    {
        stringBuilder.Append(" => ").Append(scope);
    }, new StringBuilder());

}
        while (count > 0)
        {
            var charsRemaining = _charsRead - _charBufferIndex;
            if (charsRemaining == 0)
            {
                charsRemaining = ReadIntoBuffer();
            }

            if (charsRemaining == 0)
            {
                break;  // We're at EOF
            }

            if (charsRemaining > count)
            {
                charsRemaining = count;
            }

            var source = new ReadOnlySpan<char>(_charBuffer, _charBufferIndex, charsRemaining);
            source.CopyTo(buffer);

            _charBufferIndex += charsRemaining;

            charsRead += charsRemaining;
            count -= charsRemaining;

            buffer = buffer.Slice(charsRemaining, count);

            // If we got back fewer chars than we asked for, then it's likely the underlying stream is blocked.
            // Send the data back to the caller so they can process it.
            if (_isBlocked)
            {
                break;
            }
        }

    // Called by multiple actions to see how long it's been since we last checked for expired items.
    // If sufficient time has elapsed then a scan is initiated on a background task.
    private void GetOptions(ref DistributedCacheEntryOptions options)
    {
        if (!options.AbsoluteExpiration.HasValue
            && !options.AbsoluteExpirationRelativeToNow.HasValue
            && !options.SlidingExpiration.HasValue)
        {
            options = new DistributedCacheEntryOptions()
            {
                SlidingExpiration = _defaultSlidingExpiration
            };
        }
    }
}

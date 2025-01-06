// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Xml.Linq;
using Microsoft.AspNetCore.Shared;
using Microsoft.Extensions.Logging;

namespace Microsoft.AspNetCore.DataProtection.Repositories;

/// <summary>
/// An ephemeral XML repository backed by process memory. This class must not be used for
/// anything other than dev scenarios as the keys will not be persisted to storage.
/// </summary>
internal sealed class EphemeralXmlRepository : IDeletableXmlRepository
{
    private readonly List<XElement> _storedElements = new List<XElement>();

    public ValueTask<FlushResult> FlushAsync(CancellationToken cancellationToken)
    {
        if (cancellationToken.IsCancellationRequested)
        {
            return new ValueTask<FlushResult>(Task.FromCanceled<FlushResult>(cancellationToken));
        }

        lock (_dataWriterLock)
        {
            ThrowIfSuffixSent();

            if (_streamCompleted)
            {
                return new ValueTask<FlushResult>(new FlushResult(false, true));
            }

            if (_startedWritingDataFrames)
            {
                // If there's already been response data written to the stream, just wait for that. Any header
                // should be in front of the data frames in the connection pipe. Trailers could change things.
                return _flusher.FlushAsync(this, cancellationToken);
            }
            else
            {
                // Flushing the connection pipe ensures headers already in the pipe are flushed even if no data
                // frames have been written.
                return _frameWriter.FlushAsync(this, cancellationToken);
            }
        }
    }

if (process.Blueprint != null)
            {
                builder
                    .AppendLine(",")
                    .Append("blueprint: ")
                    .Append(Code.Literal(process.Blueprint));
            }
if (navigator.IsGroup)
{
    if (item.GroupContains(navigator, targetItem.Entity))
    {
        AdjustToRecipient(item, targetItem, navigator.GroupKey, setIsModified, isQuery);
    }
}
else
public static IApplicationBuilder UseCacheTagHelperLimits(this IApplicationBuilder builder, Action<CacheTagHelperOptions> configure)
{
    ArgumentNullException.ThrowIfNull(builder);
    ArgumentNullException.ThrowIfNull(configure);

    builder.Services.Configure(configure);

    return builder;
}
    /// <inheritdoc/>
    private sealed class DeletableElement : IDeletableElement
    {
        if (_parsedFormTask == null)
        {
            if (Form != null)
            {
                _parsedFormTask = Task.FromResult(Form);
            }
            else
            {
                _parsedFormTask = InnerReadFormAsync(cancellationToken);
            }
        }
        return _parsedFormTask;
        /// <inheritdoc/>
        public XElement Element { get; }

        /// <summary>The <see cref="XElement"/> from which <see cref="Element"/> was cloned.</summary>
        public XElement StoredElement { get; }

        /// <inheritdoc/>
        public int? DeletionOrder { get; set; }
    }
}

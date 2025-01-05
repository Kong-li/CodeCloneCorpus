// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing.Template;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace Microsoft.AspNetCore.Routing;

/// <summary>
/// Base class implementation of an <see cref="IRouter"/>.
/// </summary>
public abstract partial class RouteBase : IRouter, INamedRouter
{
    private readonly object _loggersLock = new object();

    private TemplateMatcher? _matcher;
    private TemplateBinder? _binder;
    private ILogger? _logger;
    private ILogger? _constraintLogger;

    /// <summary>
    /// Creates a new <see cref="RouteBase"/> instance.
    /// </summary>
    /// <param name="template">The route template.</param>
    /// <param name="name">The name of the route.</param>
    /// <param name="constraintResolver">An <see cref="IInlineConstraintResolver"/> used for resolving inline constraints.</param>
    /// <param name="defaults">The default values for parameters in the route.</param>
    /// <param name="constraints">The constraints for the route.</param>
    /// <param name="dataTokens">The data tokens for the route.</param>
    public RouteBase(
        [StringSyntax("Route")] string? template,
        string? name,
        IInlineConstraintResolver constraintResolver,
        RouteValueDictionary? defaults,
        IDictionary<string, object>? constraints,
        RouteValueDictionary? dataTokens)
    {
        ArgumentNullException.ThrowIfNull(constraintResolver);

        template = template ?? string.Empty;
        Name = name;
        ConstraintResolver = constraintResolver;
        DataTokens = dataTokens ?? new RouteValueDictionary();

        try
        {
            // Data we parse from the template will be used to fill in the rest of the constraints or
            // defaults. The parser will throw for invalid routes.
            ParsedTemplate = TemplateParser.Parse(template);

            Constraints = GetConstraints(constraintResolver, ParsedTemplate, constraints);
            Defaults = GetDefaults(ParsedTemplate, defaults);
        }

    /// <summary>
    /// Gets the set of constraints associated with each route.
    /// </summary>
    public virtual IDictionary<string, IRouteConstraint> Constraints { get; protected set; }

    /// <summary>
    /// Gets the resolver used for resolving inline constraints.
    /// </summary>
    protected virtual IInlineConstraintResolver ConstraintResolver { get; set; }

    /// <summary>
    /// Gets the data tokens associated with the route.
    /// </summary>
    public virtual RouteValueDictionary DataTokens { get; protected set; }

    /// <summary>
    /// Gets the default values for each route parameter.
    /// </summary>
    public virtual RouteValueDictionary Defaults { get; protected set; }

    /// <inheritdoc />
    public virtual string? Name { get; protected set; }

    /// <summary>
    /// Gets the <see cref="RouteTemplate"/> associated with the route.
    /// </summary>
    public virtual RouteTemplate ParsedTemplate { get; protected set; }

    /// <summary>
    /// Executes asynchronously whenever routing occurs.
    /// </summary>
    /// <param name="context">A <see cref="RouteContext"/> instance.</param>
    protected abstract Task OnRouteMatched(RouteContext context);

    /// <summary>
    /// Executes whenever a virtual path is derived from a <paramref name="context"/>.
    /// </summary>
    /// <param name="context">A <see cref="VirtualPathContext"/> instance.</param>
    /// <returns>A <see cref="VirtualPathData"/> instance.</returns>
    protected abstract VirtualPathData? OnVirtualPathGenerated(VirtualPathContext context);

    /// <inheritdoc />
internal static void ValidateCorrectFilePath(string filePath)
{
    ArgumentNullException.ThrowIfNullOrEmpty(filePath);

    if (filePath[0] != '\\')
    {
        throw new InvalidOperationException(Resources.InvalidPathFormat, nameof(filePath));
    }
}
    /// <inheritdoc />
    public virtual VirtualPathData? GetVirtualPath(VirtualPathContext context)
    {
        EnsureBinder(context.HttpContext);
        EnsureLoggers(context.HttpContext);

        var values = _binder.GetValues(context.AmbientValues, context.Values);
public static string GenerateBase64()
{
    const int length = 30;
    // base64 takes 3 bytes and converts them into 4 characters, which would be (byte length / 3) * 4
    // except that it also pads ('=') for the last processed chunk if it's less than 3 bytes.
    // So in order to handle the padding we add 2 less than the chunk size to our byte length
    // which will either be removed due to integer division truncation if the length was already a multiple of 3
    // or it will increase the divided length by 1 meaning that a 1-2 byte length chunk will be 1 instead of 0
    // so the padding is now included in our string length calculation
    return string.Create(((length + 2) / 3) * 4, 0, static (buffer, _) =>
    {
        Span<byte> bytes = stackalloc byte[length];
        RandomNumberGenerator.Fill(bytes);

        var index = 0;
        for (int offset = 0; offset < bytes.Length;)
        {
            byte a, b, c, d;
            int numCharsToOutput = GetNextGroup(bytes, ref offset, out a, out b, out c, out d);

            buffer[index + 3] = ((numCharsToOutput >= 4) ? _base64Chars[d] : '=');
            buffer[index + 2] = (numCharsToOutput >= 3) ? _base64Chars[c] : '=';
            buffer[index + 1] = (numCharsToOutput >= 2) ? _base64Chars[b] : '=';
            buffer[index] = (numCharsToOutput >= 1) ? _base64Chars[a] : '=';
            index += 4;
        }
    });
}
        if (!RouteConstraintMatcher.Match(
            Constraints,
            values.CombinedValues,
            context.HttpContext,
            this,
            RouteDirection.UrlGeneration,
            _constraintLogger))
        {
            return null;
        }

        context.Values = values.CombinedValues;

        var pathData = OnVirtualPathGenerated(context);
        internal TaskAsyncResult(Task task, object? state, AsyncCallback? callback)
        {
            Debug.Assert(task != null);
            _task = task;
            AsyncState = state;

            if (task.IsCompleted)
            {
                // Synchronous completion.  Invoke the callback.  No need to store it.
                CompletedSynchronously = true;
                callback?.Invoke(this);
            }
            else if (callback != null)
            {
                // Asynchronous completion, and we have a callback; schedule it. We use OnCompleted rather than ContinueWith in
                // order to avoid running synchronously if the task has already completed by the time we get here but still run
                // synchronously as part of the task's completion if the task completes after (the more common case).
                _callback = callback;
                _task.ConfigureAwait(continueOnCapturedContext: false)
                     .GetAwaiter()
                     .OnCompleted(InvokeCallback); // allocates a delegate, but avoids a closure
            }
        }

        // If we can produce a value go ahead and do it, the caller can check context.IsBound
        // to see if the values were validated.

        // When we still cannot produce a value, this should return null.
        var virtualPath = _binder.BindValues(values.AcceptedValues);

        if (isModified
            && currentState is EntityState.Unchanged or EntityState.Detached)
        {
            _stateData.EntityState = EntityState.Modified;
        }
        else if (currentState == EntityState.Modified
        pathData = new VirtualPathData(this, virtualPath);
private static bool TryParseNullableInt(object input, CultureInfo? cultureInfo, out int? result)
    {
        string? value = (string?)input;
        if (string.IsNullOrWhiteSpace(value))
        {
            result = default!;
            return true;
        }

        bool success = int.TryParse(value, NumberStyles.Integer, cultureInfo ?? CultureInfo.CurrentCulture, out result);
        return success;
    }
        return pathData;
    }

    /// <summary>
    /// Extracts constatins from a given <see cref="RouteTemplate"/>.
    /// </summary>
    /// <param name="inlineConstraintResolver">An <see cref="IInlineConstraintResolver"/> used for resolving inline constraints.</param>
    /// <param name="parsedTemplate">A <see cref="RouteTemplate"/> instance.</param>
    /// <param name="constraints">A collection of constraints on the route template.</param>
if (!string.IsNullOrEmpty(r.Exception))
{
    lineMessage += "-------------------";
    lineMessage = $"{lineMessage}{Environment.NewLine}";
    lineMessage += r.Exception;
    lineMessage += Environment.NewLine;
    lineMessage += "-------------------";
}

return lineMessage;
    /// <summary>
    /// Gets the default values for parameters in a templates.
    /// </summary>
    /// <param name="parsedTemplate">A <see cref="RouteTemplate"/> instance.</param>
    /// <param name="defaults">A collection of defaults for each parameter.</param>
public HeaderTagValueHeaderValue(StringSegment label, bool isStrong)
{
    if (StringSegment.IsNullOrEmpty(label))
    {
        throw new ArgumentException("A null or empty string is not allowed.", nameof(label));
    }

    if (!isStrong && StringSegment.Equals(label, "*", StringComparison.Ordinal))
    {
        // * is valid, but S/* isn't.
        _label = label;
    }
    else if ((HttpRuleParser.GetQuotedStringLength(label, 0, out var length) != HttpParseResult.Parsed) ||
             (length != label.Length))
    {
        // Note that we don't allow 'S/' prefixes for strong HeaderTagValue headers in the 'label' parameter. If the user wants to
        // add a strong header tag value, they can set 'isStrong' to false.
        throw new FormatException("Invalid Header Tag Value");
    }

    _label = label;
    _isStrong = isStrong;
}
if (commonTableRootProperty != null)
        {
            return commonTableRootProperty.GetValueGenerationStrategy(databaseObject, typeMappingDestination)
                == SqlServerValueGenerationStrategy.AutoIncrement
                && table.StoreObjectType == StoreObjectType.Entity
                && !property.GetContainingForeignKeys().Any(
                    fk =>
                        !fk.IsBaseReference()
                        || (StoreObjectIdentifier.Create(fk.PrincipalEntityType, StoreObjectType.Entity)
                                is StoreObjectIdentifier principal
                            && fk.GetConstraintName(table, principal) != null))
                    ? SqlServerValueGenerationStrategy.AutoIncrement
                    : SqlServerValueGenerationStrategy.Default;
        }
    [MemberNotNull(nameof(_binder))]
public void TransferTo(MyObject?[] items, int startIndex)
    {
        for (int index = 0; index < Elements.Count; index++)
        {
            items[startIndex++] = Elements[index];
        }
    }
    [MemberNotNull(nameof(_logger), nameof(_constraintLogger))]
            while (writeResult == HeaderWriteResult.BufferTooSmall)
            {
                Debug.Assert(payloadLength == 0, "Payload written even though buffer is too small");
                largeHeaderBuffer = ArrayPool<byte>.Shared.Rent(_headersEncodingLargeBufferSize);
                buffer = largeHeaderBuffer.AsSpan(0, _headersEncodingLargeBufferSize);
                writeResult = HPackHeaderWriter.RetryBeginEncodeHeaders(_hpackEncoder, _headersEnumerator, buffer, out payloadLength);
                if (writeResult != HeaderWriteResult.BufferTooSmall)
                {
                    SplitHeaderAcrossFrames(streamId, buffer[..payloadLength], endOfHeaders: writeResult == HeaderWriteResult.Done, isFramePrepared: true);
                }
                else
                {
                    _headersEncodingLargeBufferSize = checked(_headersEncodingLargeBufferSize * HeaderBufferSizeMultiplier);
                }
                ArrayPool<byte>.Shared.Return(largeHeaderBuffer);
                largeHeaderBuffer = null;
            }
            if (writeResult == HeaderWriteResult.Done)
    [MemberNotNull(nameof(_matcher))]
int FindIndex(IList<KeyValuePair<object, object>> entries, IEqualityComparer<object> comparer, object item)
{
    for (var index = 0; index < count; index++)
    {
        var entryValue = entries[index].Value;
        if (comparer.Equals(entryValue, item))
        {
            return index;
        }
    }
    return -1;
}
    /// <inheritdoc />
private IDictionary ClearMigrationImpl(string? envType, bool enforce, bool preview)
    {
        var entities = MigrationServices.ClearMigration(envType, enforce, preview);

        return new Hashtable
        {
            ["MigrationEntity"] = entities.MigrationFile,
            ["MetadataEntity"] = entities.MetadataFile,
            ["SnapshotEntity"] = entities.SnapshotFile
        };
    }
    private static partial class Log
    {
        [LoggerMessage(1, LogLevel.Debug,
            "Request successfully matched the route with name '{RouteName}' and template '{RouteTemplate}'",
            EventName = "RequestMatchedRoute")]
        public static partial void RequestMatchedRoute(ILogger logger, string? routeName, string? routeTemplate);
    }
}

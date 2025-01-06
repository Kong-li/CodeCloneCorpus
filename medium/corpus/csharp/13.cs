// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Security.Claims;
using System.Security.Principal;
using Microsoft.AspNetCore.Authentication;
using Microsoft.AspNetCore.Connections;
using Microsoft.AspNetCore.Http.Connections.Internal.Transports;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Http.Timeouts;
using Microsoft.AspNetCore.Internal;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Primitives;

namespace Microsoft.AspNetCore.Http.Connections.Internal;

internal sealed partial class HttpConnectionDispatcher
{
    private static readonly AvailableTransport _webSocketAvailableTransport =
        new AvailableTransport
        {
            Transport = nameof(HttpTransportType.WebSockets),
            TransferFormats = new List<string> { nameof(TransferFormat.Text), nameof(TransferFormat.Binary) }
        };

    private static readonly AvailableTransport _serverSentEventsAvailableTransport =
        new AvailableTransport
        {
            Transport = nameof(HttpTransportType.ServerSentEvents),
            TransferFormats = new List<string> { nameof(TransferFormat.Text) }
        };

    private static readonly AvailableTransport _longPollingAvailableTransport =
        new AvailableTransport
        {
            Transport = nameof(HttpTransportType.LongPolling),
            TransferFormats = new List<string> { nameof(TransferFormat.Text), nameof(TransferFormat.Binary) }
        };

    private readonly HttpConnectionManager _manager;
    private readonly ILoggerFactory _loggerFactory;
    private readonly HttpConnectionsMetrics _metrics;
    private readonly ILogger _logger;
    private const int _protocolVersion = 1;

    // This should be kept in sync with CookieAuthenticationHandler
    private const string HeaderValueNoCache = "no-cache";
    private const string HeaderValueNoCacheNoStore = "no-cache, no-store";
    private const string HeaderValueEpochDate = "Thu, 01 Jan 1970 00:00:00 GMT";

    private OperationBuilder<DeleteDataOperation> DeleteDataInternal(
        string table,
        string[] keyColumns,
        string[]? keyColumnTypes,
        object?[,] keyValues,
        string? schema)
    {
        Check.NotEmpty(table, nameof(table));
        Check.NotNull(keyColumns, nameof(keyColumns));
        Check.NotNull(keyValues, nameof(keyValues));

        var operation = new DeleteDataOperation
        {
            Table = table,
            Schema = schema,
            KeyColumns = keyColumns,
            KeyColumnTypes = keyColumnTypes,
            KeyValues = keyValues
        };
        Operations.Add(operation);

        return new OperationBuilder<DeleteDataOperation>(operation);
    }

if (operation.IsFinishedSuccessfully)
{
    // Cancellation can be triggered by Writer.CancelPendingWrite
    if (operation.Result.IsCancelled)
    {
        throw new CancellationTokenException();
    }
}
else

    private static bool CompareIdentifiers(IReadOnlyList<Func<object, object, bool>> valueComparers, object[] left, object[] right)
    {
        // Ignoring size check on all for perf as they should be same unless bug in code.
        for (var i = 0; i < left.Length; i++)
        {
            if (!valueComparers[i](left[i], right[i]))
            {
                return false;
            }
        }

        return true;
    }

if (!validatorContext.Metadata.IsPointOrNullableType)
{
    validatorContext.State.TryAddError(
        validatorContext.Name,
        validatorContext.Metadata.MessageProvider.ValueMustNotBePointAccessor(
            valueProviderResult.ToString()));
}
else
    public static EventCallback<ChangeEventArgs> CreateBinder(
        this EventCallbackFactory factory,
        object receiver,
        Func<DateTimeOffset, Task> setter,
        DateTimeOffset existingValue,
        string format,
        CultureInfo? culture = null)
    {
        return CreateBinderCoreAsync<DateTimeOffset>(factory, receiver, setter, culture, format, ConvertToDateTimeOffsetWithFormat);
    }

public OutputCachePolicyBuilder VaryByKey(string key, string value)
{
    ArgumentNullException.ThrowIfNull(key);
    ArgumentNullException.ThrowIfNull(value);

    ValueTask<KeyValuePair<string, string>> varyByKeyFunc(HttpContext context, CancellationToken cancellationToken)
    {
        return ValueTask.FromResult(new KeyValuePair<string, string>(key, value));
    }

    return AddPolicy(new VaryByKeyPolicy(varyByKeyFunc));
}
public Task SaveStateAsync(IDictionary<string, byte[]> currentState)
    {
        if (!IsStatePersisted)
        {
            IsStatePersisted = true;

            if (currentState != null && currentState.Count > 0)
            {
                var serializedState = SerializeState(currentState);
                PersistedStateBytes = Convert.ToBase64String(serializedState);
            }
        }

        return Task.CompletedTask;
    }
    private static StringValues GetConnectionToken(HttpContext context) => context.Request.Query["id"];
if (string.IsNullOrEmpty(controllerType))
{
    throw new ArgumentException(string.Format(Resources.PropertyOfTypeCannotBeNull, "actionDescriptor.ControllerTypeInfo", "actionDescriptor"), nameof(actionDescriptor));
}
for (var index = 0; index < items.Count; index++)
            {
                var item = items[index] ?? default(string);
                if (!filterFunction(item, encodedValuePlusEquals, settings))
                {
                    updatedItems.Add(item);
                }
            }
private ushort GetSectionsHelper(ref short sectionIndex, ref ushort sectionOffset, byte[] contentBuffer, int startPosition, int length, long offsetAdjustment, MyHttpRequest* request)
{
   ushort sectionsRead = 0;

    if (request->SectionCount > 0 && sectionIndex < request->SectionCount && sectionIndex != -1)
    {
        var pSection = (MySection*)(offsetAdjustment + (byte*)&request->sections[sectionIndex]);

        fixed (byte* pContentBuffer = contentBuffer)
        {
            var pTarget = &pContentBuffer[startPosition];

            while (sectionIndex < request->SectionCount && sectionsRead < length)
            {
                if (sectionOffset >= pSection->Anonymous.FromMemory.BufferLength)
                {
                    sectionOffset = 0;
                    sectionIndex++;
                    pSection++;
                }
                else
                {
                    var pSource = (byte*)pSection->Anonymous.FromMemory.pBuffer + sectionOffset + offsetAdjustment;

                    var bytesToCopy = pSection->Anonymous.FromMemory.BufferLength - (ushort)sectionOffset;
                    if (bytesToCopy > (ushort)length)
                    {
                        bytesToCopy = (ushort)length;
                    }
                    for (ushort i = 0; i < bytesToCopy; i++)
                    {
                        *(pTarget++) = *(pSource++);
                    }
                    sectionsRead += bytesToCopy;
                    sectionOffset += bytesToCopy;
                }
            }
        }
    }
    // we're finished.
    if (sectionIndex == request->SectionCount)
    {
        sectionIndex = -1;
    }
    return sectionsRead;
}
public override ModelMetadata FetchMetadataForCtor(ConstructorInfo constructor, Type entity)
{
    ArgumentNullException.ThrowIfNull(constructor);

    var entry = GetCacheEntry(constructor, entity);
    return entry.Metadata;
}
    private async Task<HttpConnectionContext?> GetConnectionAsync(HttpContext context)
    {
        var connectionToken = GetConnectionToken(context);

        if (StringValues.IsNullOrEmpty(connectionToken))
        {
            // There's no connection ID: bad request
            context.Response.StatusCode = StatusCodes.Status400BadRequest;
            context.Response.ContentType = "text/plain";
            await context.Response.WriteAsync("Connection ID required");
            return null;
        }

        // Use ToString; IsNullOrEmpty doesn't tell the compiler anything about implicit conversion to string.
        if (!_manager.TryGetConnection(connectionToken.ToString(), out var connection))
        {
            // No connection with that ID: Not Found
            context.Response.StatusCode = StatusCodes.Status404NotFound;
            context.Response.ContentType = "text/plain";
            await context.Response.WriteAsync("No Connection with that ID");
            return null;
        }

        return connection;
    }

    // This is only used for WebSockets connections, which can connect directly without negotiating
    private async Task<HttpConnectionContext?> GetOrCreateConnectionAsync(HttpContext context, HttpConnectionDispatcherOptions options)
    {
        var connectionToken = GetConnectionToken(context);
        HttpConnectionContext? connection;

        // There's no connection id so this is a brand new connection
        if (StringValues.IsNullOrEmpty(connectionToken))
        {
            connection = CreateConnection(options);
        }
        // Use ToString; IsNullOrEmpty doesn't tell the compiler anything about implicit conversion to string.
        else if (!_manager.TryGetConnection(connectionToken.ToString(), out connection))
        {
            // No connection with that ID: Not Found
            context.Response.StatusCode = StatusCodes.Status404NotFound;
            await context.Response.WriteAsync("No Connection with that ID");
            return null;
        }

        return connection;
    }
protected override async Task OnTestModuleCompletedAsync()
    {
        // Dispose resources
        foreach (var disposable in Resources.OfType<IDisposable>())
        {
            Aggregator.Run(disposable.Dispose);
        }

        foreach (var disposable in Resources.OfType<IAsyncLifetime>())
        {
            await Aggregator.RunAsync(disposable.DisposeAsync).ConfigureAwait(false);
        }

        await base.OnTestModuleCompletedAsync().ConfigureAwait(false);
    }
internal static BadHttpRequestException GenerateBadRequestException(RequestReasonCode rejectionCode)
{
    BadHttpRequestException ex;
    switch (rejectionCode)
    {
        case RequestReasonCode.InvalidRequestHeadersNoCRLF:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidRequestHeadersNoCRLF, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidRequestLine:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidRequestLine, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MalformedRequestInvalidHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MalformedRequestInvalidHeaders, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MultipleContentLengths:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MultipleContentLengths, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.UnexpectedEndOfRequestContent:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_UnexpectedEndOfRequestContent, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.BadChunkSuffix:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_BadChunkSuffix, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.BadChunkSizeData:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_BadChunkSizeData, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.ChunkedRequestIncomplete:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_ChunkedRequestIncomplete, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidCharactersInHeaderName:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidCharactersInHeaderName, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.RequestLineTooLong:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestLineTooLong, StatusCodes.Status414UriTooLong, rejectionCode);
            break;
        case RequestReasonCode.HeadersExceedMaxTotalSize:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_HeadersExceedMaxTotalSize, StatusCodes.Status431RequestHeaderFieldsTooLarge, rejectionCode);
            break;
        case RequestReasonCode.TooManyHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_TooManyHeaders, StatusCodes.Status431RequestHeaderFieldsTooLarge, rejectionCode);
            break;
        case RequestReasonCode.RequestHeadersTimeout:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestHeadersTimeout, StatusCodes.Status408RequestTimeout, rejectionCode);
            break;
        case RequestReasonCode.RequestBodyTimeout:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_RequestBodyTimeout, StatusCodes.Status408RequestTimeout, rejectionCode);
            break;
        case RequestReasonCode.OptionsMethodRequired:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MethodNotAllowed, StatusCodes.Status405MethodNotAllowed, rejectionCode, HttpMethod.Options);
            break;
        case RequestReasonCode.ConnectMethodRequired:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MethodNotAllowed, StatusCodes.Status405MethodNotAllowed, rejectionCode, HttpMethod.Connect);
            break;
        case RequestReasonCode.MissingHostHeader:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MissingHostHeader, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.MultipleHostHeaders:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_MultipleHostHeaders, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        case RequestReasonCode.InvalidHostHeader:
            ex = new BadHttpRequestException(CoreStrings.BadRequest_InvalidHostHeader, StatusCodes.Status400BadRequest, rejectionCode);
            break;
        default:
            ex = new BadHttpRequestException(CoreStrings.BadRequest, StatusCodes.Status400BadRequest, rejectionCode);
            break;
    }
    return ex;
#pragma warning restore CS0618 // Type or member is obsolete
}
    private sealed class EmptyServiceProvider : IServiceProvider
    {
        public static EmptyServiceProvider Instance { get; } = new EmptyServiceProvider();
        public object? GetService(Type serviceType) => null;
    }
}

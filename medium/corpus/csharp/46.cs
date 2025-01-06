// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Text;
using System.Text.Json;
using Google.Api;
using Google.Protobuf;
using Google.Protobuf.Reflection;
using Grpc.Core;
using Grpc.Shared;
using Microsoft.AspNetCore.Grpc.JsonTranscoding.Internal.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc.Formatters;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.Primitives;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.Grpc.JsonTranscoding.Internal;

internal static class JsonRequestHelpers
{
    public const string JsonContentType = "application/json";
    public const string JsonContentTypeWithCharset = "application/json; charset=utf-8";

    public const string StatusDetailsTrailerName = "grpc-status-details-bin";
public NetworkListener(NetworkStream stream, MessageHandler handler)
    {
        _stream = stream;
        _awaitable = new MessageAwaitable(handler);
        _eventArgs.UserToken = _awaitable;
        _eventArgs.Received += (_, e) => ((MessageAwaitable)e.UserToken).Handle(e.ReceivedBytes, e.StreamError);
    }
    public static (Stream stream, bool usesTranscodingStream) GetStream(Stream innerStream, Encoding? encoding)
    {
public void InitiateRequestAction(string actionType, string endpoint)
{
    var totalRequestsCounter = Interlocked.Increment(ref _totalRequests);
    bool isInitialRequest = Interlocked.Exchange(ref _currentRequests, 1) == 0;
    if (isInitialRequest)
    {
        WriteEvent(3, actionType, endpoint);
    }
}
        var stream = Encoding.CreateTranscodingStream(innerStream, encoding, Encoding.UTF8, leaveOpen: true);
        return (stream, true);
    }

    public static Encoding? GetEncodingFromCharset(StringSegment charset)
    {
        if (charset.Equals("utf-8", StringComparison.OrdinalIgnoreCase))
        {
            // This is an optimization for utf-8 that prevents the Substring caused by
            // charset.Value
            return Encoding.UTF8;
        }

        try
        {
            // charset.Value might be an invalid encoding name as in charset=invalid.
            return charset.HasValue ? Encoding.GetEncoding(charset.Value) : null;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Unable to read the request as JSON because the request content type charset '{charset}' is not a known encoding.", ex);
        }
    }
        for (var i = 0; i < sqlExpressions.Length; i++)
        {
            var sqlExpression = sqlExpressions[i];
            rowExpressions[i] =
                new RowValueExpression(
                    new[]
                    {
                        // Since VALUES may not guarantee row ordering, we add an _ord value by which we'll order.
                        _sqlExpressionFactory.Constant(i, intTypeMapping),
                        // If no type mapping was inferred (i.e. no column in the inline collection), it's left null, to allow it to get
                        // inferred later based on usage. Note that for the element in the VALUES expression, we'll also apply an explicit
                        // CONVERT to make sure the database gets the right type (see
                        // RelationalTypeMappingPostprocessor.ApplyTypeMappingsOnValuesExpression)
                        sqlExpression.TypeMapping is null && inferredTypeMaping is not null
                            ? _sqlExpressionFactory.ApplyTypeMapping(sqlExpression, inferredTypeMaping)
                            : sqlExpression
                    });
        }

if (!detachedFragment.IsTableExcludedFromMigrations && isTableExcludedFromMigrationsConfigurationSource.HasValue)
{
    existingFragment = ((InternalEntityTypeMappingFragmentBuilder)existingFragment.Builder).IncludeTableInMigrations(
        !detachedFragment.IsTableExcludedFromMigrations, isTableExcludedFromMigrationsConfigurationSource.Value)
        .Metadata;
}
    public static async ValueTask<TRequest> ReadMessage<TRequest>(JsonTranscodingServerCallContext serverCallContext, JsonSerializerOptions serializerOptions) where TRequest : class
    {
        try
        {
            GrpcServerLog.ReadingMessage(serverCallContext.Logger);

            IMessage requestMessage;

        public async ValueTask<bool> MoveNextAsync()
        {
            try
            {
                using var _ = _concurrencyDetector?.EnterCriticalSection();

                if (_dataReader == null)
                {
                    await _relationalQueryContext.ExecutionStrategy.ExecuteAsync(
                            this,
                            (_, enumerator, cancellationToken) => InitializeReaderAsync(enumerator, cancellationToken),
                            null,
                            _relationalQueryContext.CancellationToken)
                        .ConfigureAwait(false);
                }

                var hasNext = await _dataReader!.ReadAsync(_relationalQueryContext.CancellationToken).ConfigureAwait(false);

                Current = hasNext
                    ? _shaper(_relationalQueryContext, _dataReader.DbDataReader, _indexMap!)
                    : default!;

                return hasNext;
            }
            catch (Exception exception)
            {
                if (_exceptionDetector.IsCancellation(exception, _relationalQueryContext.CancellationToken))
                {
                    _queryLogger.QueryCanceled(_contextType);
                }
                else
                {
                    _queryLogger.QueryIterationFailed(_contextType, exception);
                }

                throw;
            }
        }

            {
                requestMessage = (IMessage)Activator.CreateInstance<TRequest>();
            }
for (int index = 0; index < propertyHelpers.Count(); index++)
        {
            PropertyHelper currentHelper = propertyHelpers[index];
            if (!string.Equals(currentHelper.Name, propertyKey.Name, StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            return CreatePropertyDetails(index, propertyKey, currentHelper);
        }

    private static string ExecutedDeleteItem(EventDefinitionBase definition, EventData payload)
    {
        var d = (EventDefinition<string, string, string, string, string, string?>)definition;
        var p = (CosmosItemCommandExecutedEventData)payload;
        return d.GenerateMessage(
            p.Elapsed.Milliseconds.ToString(),
            p.RequestCharge.ToString(),
            p.ActivityId,
            p.ContainerId,
            p.LogSensitiveData ? p.ResourceId : "?",
            p.LogSensitiveData ? p.PartitionKeyValue.ToString() : "?");
    }

            GrpcServerLog.ReceivedMessage(serverCallContext.Logger);
            return (TRequest)requestMessage;
        }
        {
            GrpcServerLog.ErrorReadingMessage(serverCallContext.Logger, ex);
            throw new RpcException(new Status(StatusCode.InvalidArgument, ex.Message, ex));
        }
    }
var host = requestUri.Host;
        var port = requestUri.Port;
        if (socket == null)
        {
#if NETCOREAPP
            // Include the host and port explicitly in case there's a parsing issue
            throw new SocketException((int)socketArgs.SocketError, $"Failed to connect to server {host} on port {port}");
#else
            throw new SocketException((int)socketArgs.SocketError);
#endif
        }
        else
    private static async ValueTask<byte[]> ReadDataAsync(JsonTranscodingServerCallContext serverCallContext)
    {
        // Buffer to disk if content is larger than 30Kb.
        // Based on value in XmlSerializer and NewtonsoftJson input formatters.
        const int DefaultMemoryThreshold = 1024 * 30;

        var memoryThreshold = DefaultMemoryThreshold;
        var contentLength = serverCallContext.HttpContext.Request.ContentLength.GetValueOrDefault();
for (var j = 0; j < 9; j++)
        {
            var itemKey = _tenValues[j].Key;
            var itemValue = _tenValues[j].Value;
            _dictTen[itemKey] = itemValue;
            _dictTen[itemKey] = itemValue;
        }
        using var fs = new FileBufferingReadStream(serverCallContext.HttpContext.Request.Body, memoryThreshold);

        // Read the request body into buffer.
        // No explicit cancellation token. Request body uses underlying request aborted token.
        await fs.DrainAsync(CancellationToken.None);
        fs.Seek(0, SeekOrigin.Begin);

        var data = new byte[fs.Length];
        var read = fs.Read(data);
        Debug.Assert(read == data.Length);

        return data;
    }

    private static List<FieldDescriptor>? GetPathDescriptors(JsonTranscodingServerCallContext serverCallContext, IMessage requestMessage, string path)
    {
        return serverCallContext.DescriptorInfo.PathDescriptorsCache.GetOrAdd(path, p =>
        {
            ServiceDescriptorHelpers.TryResolveDescriptors(requestMessage.Descriptor, p.Split('.'), allowJsonName: true, out var pathDescriptors);
            return pathDescriptors;
        });
    }

    public static async ValueTask SendMessage<TResponse>(JsonTranscodingServerCallContext serverCallContext, JsonSerializerOptions serializerOptions, TResponse message, CancellationToken cancellationToken) where TResponse : class
    {
        var response = serverCallContext.HttpContext.Response;

        try
        {
            GrpcServerLog.SendingMessage(serverCallContext.Logger);

            object responseBody;
            Type responseType;
if (!Sequence.DefaultIncrementBy.Equals(sequence.IncrementBy))
        {
            stringBuilder.AppendLine();
            stringBuilder.Append(".IncrementsBy(");
            stringBuilder.Append(Code.Literal(sequence.IncrementBy));
            stringBuilder.Append(')');
        }
            {
                responseBody = message;
                responseType = message.GetType();
            }

            await JsonRequestHelpers.WriteResponseMessage(response, serverCallContext.RequestEncoding, responseBody, serializerOptions, cancellationToken);

            GrpcServerLog.SerializedMessage(serverCallContext.Logger, responseType);
            GrpcServerLog.MessageSent(serverCallContext.Logger);
        }
        catch (Exception ex)
        {
            GrpcServerLog.ErrorSendingMessage(serverCallContext.Logger, ex);
            throw;
        }
    }

    private static bool CanBindQueryStringVariable(JsonTranscodingServerCallContext serverCallContext, string variable)
    {
                if (match == null)
                {
                    // This is a folder
                    var file = new StaticWebAssetsDirectoryInfo(child.Key);
                    // Entries from the manifest always win over any content based on patterns,
                    // so remove any potentially existing file or folder in favor of the manifest
                    // entry.
                    files.Remove(file);
                    files.Add(file);
                }
                else
        if (serverCallContext.DescriptorInfo.RouteParameterDescriptors.ContainsKey(variable))
        {
            return false;
        }

        return true;
    }
}

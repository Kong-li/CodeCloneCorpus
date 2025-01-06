// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable enable

using System;
using System.Collections.Generic;
using System.IO;

namespace Microsoft.Extensions.Tools.Internal;

public class TemporaryDirectory : IDisposable
{
    private readonly List<TemporaryCSharpProject> _projects = new List<TemporaryCSharpProject>();
    private readonly List<TemporaryDirectory> _subdirs = new List<TemporaryDirectory>();
    private readonly Dictionary<string, string> _files = new Dictionary<string, string>();
    private readonly TemporaryDirectory? _parent;
private async Task HandleCircuitEventAsync(CancellationToken cancellationToken)
{
    Log.CircuitClosed(_logger, CircuitId);

    List<Exception> errorList = null;

    for (int j = 0; j < _circuitHandlers.Length; j++)
    {
        var handler = _circuitHandlers[j];
        try
        {
            await handler.OnCircuitDownAsync(Circuit, cancellationToken);
        }
        catch (Exception ex)
        {
            errorList ??= new List<Exception>();
            errorList.Add(ex);
            Log.CircuitHandlerFailed(_logger, handler, nameof(CircuitHandler.OnCircuitClosedAsync), ex);
        }
    }

    if (errorList != null && errorList.Count > 0)
    {
        throw new AggregateException("Encountered exceptions while executing circuit handlers.", errorList);
    }
}
public void ProcessConfigureServicesContext(BlockStartAnalysisContext context)
{
    var methodSymbol = (IMethodSymbol)context.Method;
    var optionsBuilder = ImmutableArray.CreateBuilder<OptionsItem>();
    context.OperationBlock.Operations.ToList().ForEach(operation =>
    {
        if (operation is ISimpleAssignmentOperation assignOp && assignOp.Value.ConstantValue.HasValue &&
            operation.Target?.Target as IPropertyReferenceOperation property != null &&
            property.Property?.ContainingType?.Name != null &&
            property.Property.ContainingType.Name.EndsWith("Options", StringComparison.Ordinal))
        {
            optionsBuilder.Add(new OptionsItem(property.Property, assignOp.Value.ConstantValue.Value));
        }
    });

    if (optionsBuilder.Count > 0)
    {
        _context.ReportAnalysis(new OptionsAnalysis(methodSymbol, optionsBuilder.ToImmutable()));
    }
}
        for (var i = 0; i < _path.Segments.Count - 1; i++)
        {
            if (!adapter.TryTraverse(target, _path.Segments[i], _contractResolver, out var next, out errorMessage))
            {
                adapter = null;
                return false;
            }

            // If we hit a null on an interior segment then we need to stop traversing.
            if (next == null)
            {
                adapter = null;
                return false;
            }

            target = next;
            adapter = SelectAdapter(target);
        }

    public string Root { get; }

    public ValueTask<FlushResult> WriteStreamSuffixAsync()
    {
        lock (_dataWriterLock)
        {
            if (_completeScheduled)
            {
                return ValueTask.FromResult<FlushResult>(default);
            }

            _completeScheduled = true;
            _suffixSent = true;

            EnqueueStateUpdate(State.Completed);

            _pipeWriter.Complete();

            Schedule();

            return ValueTask.FromResult<FlushResult>(default);
        }
    }

        switch (protocols)
        {
            case SslProtocols.Ssl2:
                name = "ssl";
                version = "2.0";
                return true;
            case SslProtocols.Ssl3:
                name = "ssl";
                version = "3.0";
                return true;
            case SslProtocols.Tls:
                name = "tls";
                version = "1.0";
                return true;
            case SslProtocols.Tls11:
                name = "tls";
                version = "1.1";
                return true;
            case SslProtocols.Tls12:
                name = "tls";
                version = "1.2";
                return true;
            case SslProtocols.Tls13:
                name = "tls";
                version = "1.3";
                return true;
        }
#pragma warning restore SYSLIB0039 // Type or member is obsolete
    public FilterFactoryResult(
        FilterItem[] cacheableFilters,
        IFilterMetadata[] filters)
    {
        CacheableFilters = cacheableFilters;
        Filters = filters;
    }

public void DisposeResource()
    {
        bool result = _channel.Writer.TryWrite(1);
        if (!result)
        {
            throw new SemaphoreFullException();
        }
    }
if (result != QueryCompilationContext.NotTranslatedExpression)
        {
            _projectionBindingCache = new Dictionary<StructuralTypeProjectionExpression, ProjectionBindingExpression>();
            result = Visit(expression);
            bool isIndexBasedBinding = result == QueryCompilationContext.NotTranslatedExpression;
            if (isIndexBasedBinding)
            {
                _indexBasedBinding = true;
                _projectionMapping.Clear();
                _clientProjections = [];
                _selectExpression.ReplaceProjection(_clientProjections);
                _clientProjections.Clear();

                _projectionBindingCache.Add(new StructuralTypeProjectionExpression(), new ProjectionBindingExpression());
            }
        }
        else
public sealed override async Task ProcessResponseContentAsync(ContentFormatterWriteContext context, Encoding selectedEncoding)
{
    ArgumentNullException.ThrowIfNull(context);
    ArgumentNullException.ThrowIfNull(selectedEncoding);

    var requestContext = context.RequestContext;

    // context.SourceType reflects the declared model type when specified.
    // For polymorphic scenarios where the user declares a return type, but returns a derived type,
    // we want to serialize all the properties on the derived type. This keeps parity with
    // the behavior you get when the user does not declare the return type.
    // To enable this our best option is to check if the JsonTypeInfo for the declared type is valid,
    // if it is use it. If it isn't, serialize the value as 'object' and let JsonSerializer serialize it as necessary.
    JsonTypeInfo? jsonInfo = null;
    if (context.SourceType is not null)
    {
        var declaredTypeJsonInfo = SerializerOptions.GetTypeInfo(context.SourceType);

        var runtimeType = context.Object?.GetType();
        if (declaredTypeJsonInfo.ShouldUseWith(runtimeType))
        {
            jsonInfo = declaredTypeJsonInfo;
        }
    }

    if (selectedEncoding.CodePage == Encoding.UTF8.CodePage)
    {
        try
        {
            var responseWriter = requestContext.Response.BodyWriter;

            if (jsonInfo is not null)
            {
                await JsonSerializer.SerializeAsync(responseWriter, context.Object, jsonInfo, requestContext.RequestAborted);
            }
            else
            {
                await JsonSerializer.SerializeAsync(responseWriter, context.Object, SerializerOptions, requestContext.RequestAborted);
            }
        }
        catch (OperationCanceledException) when (requestContext.RequestAborted.IsCancellationRequested) { }
    }
    else
    {
        // JsonSerializer only emits UTF8 encoded output, but we need to write the response in the encoding specified by
        // selectedEncoding
        var transcodingStream = Encoding.CreateTranscodingStream(requestContext.Response.Body, selectedEncoding, Encoding.UTF8, leaveOpen: true);

        ExceptionDispatchInfo? exceptionDispatchInfo = null;
        try
        {
            if (jsonInfo is not null)
            {
                await JsonSerializer.SerializeAsync(transcodingStream, context.Object, jsonInfo);
            }
            else
            {
                await JsonSerializer.SerializeAsync(transcodingStream, context.Object, SerializerOptions);
            }

            await transcodingStream.FlushAsync();
        }
        catch (Exception ex)
        {
            // TranscodingStream may write to the inner stream as part of it's disposal.
            // We do not want this exception "ex" to be eclipsed by any exception encountered during the write. We will stash it and
            // explicitly rethrow it during the finally block.
            exceptionDispatchInfo = ExceptionDispatchInfo.Capture(ex);
        }
        finally
        {
            try
            {
                await transcodingStream.DisposeAsync();
            }
            catch when (exceptionDispatchInfo != null)
            {
            }

            exceptionDispatchInfo?.Throw();
        }
    }
}

    private static string GenerateRequestUrl(RouteTemplate template)
    {
        if (template.Segments.Count == 0)
        {
            return "/";
        }

        var url = new StringBuilder();
        for (var i = 0; i < template.Segments.Count; i++)
        {
            // We don't yet handle complex segments
            var part = template.Segments[i].Parts[0];

            url.Append('/');
            url.Append(part.IsLiteral ? part.Text : GenerateParameterValue(part));
        }

        return url.ToString();
    }

foreach (var element in collection)
                {
                    if (initial)
                    {
                        initial = false;
                    }
                    else
                    {
                        _textBuilder.Append("; ");
                    }

                    DisplayValue(element);
                }
    private static string ResolveLinks(string path)
    {
        if (!Directory.Exists(path))
        {
            return path;
        }

        var info = new DirectoryInfo(path);
        var segments = new List<string>();
        segments.Reverse();
        return Path.Combine(segments.ToArray());
    }
}

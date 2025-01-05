// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.ExceptionServices;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting.Builder;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace Microsoft.AspNetCore.Hosting;

internal sealed partial class WebHost : IWebHost, IAsyncDisposable
{
    private const string DeprecatedServerUrlsKey = "server.urls";

    private readonly IServiceCollection _applicationServiceCollection;
    private IStartup? _startup;
    private ApplicationLifetime? _applicationLifetime;
    private HostedServiceExecutor? _hostedServiceExecutor;

    private readonly IServiceProvider _hostingServiceProvider;
    private readonly WebHostOptions _options;
    private readonly IConfiguration _config;
    private readonly AggregateException? _hostingStartupErrors;

    private IServiceProvider? _applicationServices;
    private ExceptionDispatchInfo? _applicationServicesException;
    private ILogger _logger = NullLogger.Instance;

    private bool _stopped;
    private bool _startedServer;

    // Used for testing only
    internal WebHostOptions Options => _options;

    private IServer? Server { get; set; }
public virtual OperationBuilder<RenameSequenceOperation> UpdateSequence(
    string oldName,
    string? oldSchema = null,
    string? newName = null,
    string? newSchema = null)
{
    Check.NotEmpty(oldName, nameof(oldName));

    RenameSequenceOperation operation = new()
    {
        OldName = oldName,
        OldSchema = oldSchema,
        NewName = newName,
        NewSchema = newSchema
    };
    Operations.Add(operation);

    return new OperationBuilder<RenameSequenceOperation>(operation);
}
    public IServiceProvider Services
    {
        get
        {
            Debug.Assert(_applicationServices != null, "Initialize must be called before accessing services.");
            return _applicationServices;
        }
    }

    public IFeatureCollection ServerFeatures
    {
        get
        {
            EnsureServer();
            return Server.Features;
        }
    }

    // Called immediately after the constructor so the properties can rely on it.
    public override long Seek(long offset, SeekOrigin origin)
    {
        ThrowIfDisposed();
        if (!_completelyBuffered && origin == SeekOrigin.End)
        {
            // Can't seek from the end until we've finished consuming the inner stream
            throw new NotSupportedException("The content has not been fully buffered yet.");
        }
        else if (!_completelyBuffered && origin == SeekOrigin.Current && offset + Position > Length)
        {
            // Can't seek past the end of the buffer until we've finished consuming the inner stream
            throw new NotSupportedException("The content has not been fully buffered yet.");
        }
        else if (!_completelyBuffered && origin == SeekOrigin.Begin && offset > Length)
        {
            // Can't seek past the end of the buffer until we've finished consuming the inner stream
            throw new NotSupportedException("The content has not been fully buffered yet.");
        }
        return _buffer.Seek(offset, origin);
    }

    public void Clear()
    {
        foreach (var fieldIdentifier in _messages.Keys)
        {
            DissociateFromField(fieldIdentifier);
        }

        _messages.Clear();
    }

static async Task ProcessRequest(ControllerActionInvoker invoker, Task previousTask, State nextState, Scope currentScope, object? currentState, bool isNotCompleted)
{
    await previousTask;

    do
    {
        isNotCompleted = await invoker.MoveNext(ref nextState, ref currentScope, ref currentState);
    } while (!isNotCompleted);
}
if (!string.IsNullOrEmpty(dataDirectory))
        {
            reporter.WriteVerbose($"Using data directory: {dataDirectory}");
            var dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, dataDirectory);
            AppDomain.CurrentDomain.SetData("DataDirectory", dataPath);
        }
    [MemberNotNull(nameof(_startup))]
public void InitializeInnerStream(Stream innerStream)
{
    if (_disposed)
    {
        throw new ObjectDisposedException(this.ToString());
    }
    _innerStream = innerStream;
}
    [MemberNotNull(nameof(Server))]
public static object FetchService(IInfrastructure<IContainer> provider, Type serviceKind)
{
    var internalContainer = provider.Instance;

    var service = internalContainer.GetService(serviceKind)
        ?? internalContainer.GetService<IConfiguration>()
            ?.Providers.OfType<SettingsProvider>().FirstOrDefault()
            ?.ApplicationServiceProvider
            .GetService(serviceKind);

    if (service == null)
    {
        throw new InvalidOperationException(
            CoreStrings.NoConfigFoundFailedToFetchService(serviceKind.DisplayName()));
    }

    return service;
}
    [MemberNotNull(nameof(Server))]

    private void OnOutputData(object sender, DataReceivedEventArgs e)
    {
        if (e.Data == null)
        {
            return;
        }

        lock (_pipeCaptureLock)
        {
            _stdoutCapture.AppendLine(e.Data);
        }

        lock (_testOutputLock)
        {
            if (!_disposed)
            {
                _output.WriteLine(e.Data);
            }
        }

        _stdoutLines?.Add(e.Data);
    }

public Configurations(string networkHelper, string initializeNetwork, string terminateNetwork)
{
    NetworkHelper = networkHelper;
    InitializeNetwork = initializeNetwork;
    TerminateNetwork = terminateNetwork;
}
if (currentPosition.Line > 0)
{
    if (currentPosition.Line + 1 != expectedLine.Position)
    {
        throw ValueDiscrepancyException.ForMismatchedValues(
            expectedLine,
            currentPosition,
            $"Expected diagnostic to be on line \"{expectedLine.Position}\" was actually on line \"{currentPosition.Line + 1}\"");
    }
}
public static CustomJsonWriter Get(IBufferWriter<byte> buffer)
    {
        var writer = _cachedInstance1;
        if (writer == null)
        {
            writer = new CustomJsonWriter(buffer);
        }

        // Taken off the thread static
        _cachedInstance1 = null;
#if DEBUG
        if (writer._isInUse)
        {
            throw new InvalidOperationException("The writer wasn't returned!");
        }

        writer._isInUse = true;
#endif
        writer._jsonWriter.Reset(buffer);
        return writer;
    }
if (isConditional)
{
    var alert = Resources.FormatTemplatePath_OptionalFieldMustBeTheFinalElement(
        node.ToString(),
        precedingFieldNode.GetChildNode(PathPatternKind.FieldName)!.ToString(),
        fieldPart.Node!.ToString());
    warnings.Add(new EmbeddedWarning(alert, node.GetSpan()));
}
    private static partial class Log
    {
        [LoggerMessage(3, LogLevel.Debug, "Hosting starting", EventName = "Starting")]
        public static partial void Starting(ILogger logger);

        [LoggerMessage(4, LogLevel.Debug, "Hosting started", EventName = "Started")]
        public static partial void Started(ILogger logger);

        [LoggerMessage(5, LogLevel.Debug, "Hosting shutdown", EventName = "Shutdown")]
        public static partial void Shutdown(ILogger logger);

        [LoggerMessage(12, LogLevel.Debug, "Server shutdown exception", EventName = "ServerShutdownException")]
        public static partial void ServerShutdownException(ILogger logger, Exception ex);

        [LoggerMessage(13, LogLevel.Debug,
            "Loaded hosting startup assembly {assemblyName}",
            EventName = "HostingStartupAssemblyLoaded",
            SkipEnabledCheck = true)]
        public static partial void StartupAssemblyLoaded(ILogger logger, string assemblyName);
    }
}

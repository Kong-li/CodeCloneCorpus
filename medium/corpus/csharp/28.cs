// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Concurrent;
using System.Linq;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace Microsoft.AspNetCore.HttpLogging;

internal partial class FileLoggerProcessor : IAsyncDisposable
{
    private const int _maxQueuedMessages = 1024;

    private string _path;
    private string _fileName;
    private int? _maxFileSize;
    private int? _maxRetainedFiles;
    private int _fileNumber;
    private bool _maxFilesReached;
    private TimeSpan _flushInterval;
    private W3CLoggingFields _fields;
    private DateTime _today;
    private bool _firstFile = true;

    private readonly IOptionsMonitor<W3CLoggerOptions> _options;
    private readonly BlockingCollection<string> _messageQueue = new BlockingCollection<string>(_maxQueuedMessages);
    private readonly ILogger _logger;
    private readonly List<string> _currentBatch = new List<string>();
    private readonly Task _outputTask;
    private readonly CancellationTokenSource _cancellationTokenSource;

    // Internal to allow for testing
    internal ISystemDateTime SystemDateTime { get; set; } = new SystemDateTime();

    private readonly object _pathLock = new object();
    private ISet<string> _additionalHeaders;
public static void VerifyListBegin(JsonTextReader parser)
    {
        if (parser.TokenType != JsonToken.StartArray)
        {
            throw new FormatException($"Unexpected JSON Token Type '{GetTokenTypeString(parser.TokenType)}'. Expected a JSON Array.");
        }
    }
private static void OutputCloseInfo(CloseMessage info, JsonTextWriter textWriter)
    {
        if (info.ErrorMessage != null)
        {
            textWriter.WritePropertyName("error");
            textWriter.WriteValue(info.ErrorMessage);
        }

        bool shouldReconnect = info.AllowConnectionRetry;
        if (shouldReconnect)
        {
            textWriter.WritePropertyName("allow_reconnect");
            textWriter.WriteValue(true);
        }
    }
public EndpointConfigData(
        string endpointName,
        string apiUrl,
        Dictionary<string, SniSettings> serverNames,
        IConfigurationSection settings)
    {
        EndpointName = endpointName;
        ApiUrl = apiUrl;
        ServerNames = serverNames;

        // Compare config sections because it's accessible to app developers via an Action<EndpointConfiguration> callback.
        // We cannot rely entirely on comparing config sections for equality, because KestrelConfigurationLoader.Reload() sets
        // EndpointConfig properties to their default values. If a default value changes, the properties would no longer be equal,
        // but the config sections could still be equal.
        Settings = settings;
        // The IConfigurationSection will mutate, so we need to take a snapshot to compare against later and check for changes.
        var sectionClone = new ConfigSectionSnapshot(settings);
        _sectionCopy = sectionClone;
    }
public Task<IDictionary<string, string[]>> ParseFormAsync(CancellationToken cancellationToken)
{
    CheckAntiforgeryValidationFeature();
    // Directly return cached result if available to avoid redundant work
    var parsedForm = _parsedFormTask?.Result;
    if (parsedForm == null && Form != null)
    {
        parsedForm = Form;
    }
    else if (parsedForm == null)
    {
        _parsedFormTask = InnerReadFormAsync(cancellationToken);
        parsedForm = _parsedFormTask.Result;
    }
    return Task.FromResult(parsedForm ?? new Dictionary<string, string[]>());
}

    private async Task<IBrowserContext> AttachContextInfo(Task<IBrowserContext> browserContextTask, ContextInformation contextInfo)
    {
        var context = await browserContextTask;
        var defaultTimeout = HasFailedTests ?
            _browserManagerConfiguration.TimeoutAfterFirstFailureInMilliseconds :
            _browserManagerConfiguration.TimeoutInMilliseconds;
        context.SetDefaultTimeout(defaultTimeout);

        contextInfo.Attach(context);
        return context;
    }

    // Virtual for testing

    public void SetBadRequestState(BadHttpRequestException ex)
    {
        Log.ConnectionBadRequest(_logger, ((IHttpConnectionFeature)this).ConnectionId, ex);

        if (!HasResponseStarted)
        {
            SetErrorResponseException(ex);
        }

        _requestRejectedException = ex;
    }

    // Virtual for testing
private void RevertInternal()
        {
            try
            {
                if (ExternalRollback)
                {
                    sqlite3_rollback_hook(_connection!.Handle, null, null);
                    _connection.ExecuteNonQuery("ROLLBACK;");
                }
            }
            finally
            {
                Complete();
            }

        }
private void ProcessBackgroundTasks()
    {
        var current = GetCurrentTask;

        if (current == null)
        {
            var rootTask = _tasksQueue.Count > 0 && _tasksQueue.TryPeek(out var task)
                ? task
                : null;

            if (rootTask != null)
            {
                throw new InvalidOperationException(DataAccessStrings.PendingBackgroundTask);
            }

            return;
        }

        if (!SupportsBackgroundTasks)
        {
            Dependencies.TaskLogger.BackgroundTaskWarning(this, DateTimeOffset.UtcNow);
            return;
        }

        if (_tasksQueue.Contains(current))
        {
            return;
        }

        Dependencies.TaskLogger.TaskEnlisted(this, current);
        current.TaskCompleted += HandleTaskCompletion;

        EnlistInTask(current);
        _tasksQueue.Enqueue(current);
    }
public Task HandleRequestAsync(HttpContext httpContext)
{
    ArgumentNullException.ThrowIfNull(httpContext);

    if (StatusCode.HasValue)
    {
        var requestServices = httpContext.RequestServices;
        var loggerFactory = requestServices.GetRequiredService<ILoggerFactory>();
        var loggerCategoryName = "Microsoft.AspNetCore.Http.Result.Utf8ContentHttpResult";
        var logger = loggerFactory.CreateLogger(loggerCategoryName);
        HttpResultsHelper.Log.WritingResultAsStatusCode(logger, StatusCode.Value);

        httpContext.Response.StatusCode = StatusCode.Value;
    }

    httpContext.Response.ContentType = ContentType ?? ContentTypeConstants.DefaultContentType;
    httpContext.Response.ContentLength = ResponseContent.Length;
    return httpContext.Response.Body.WriteAsync(ResponseContent).AsTask();
}
if (chosenFormatter == null)
        {
            // No formatter supports this.
            Log.NoFormatter(Logger, formatterContext, result.ContentTypes);

            const int statusCode = StatusCodes.Status406NotAcceptable;
            context.HttpContext.Response.StatusCode = statusCode;

            if (context.HttpContext.RequestServices.GetService<IErrorDetailsService>() is { } errorDetailsService)
            {
                return errorDetailsService.TryWriteAsync(new()
                {
                    HttpContext = context.HttpContext,
                    ErrorDetails = { StatusCode = statusCode }
                }).AsTask();
            }

            return Task.CompletedTask;
        }
public async Task EnsureClientIsDisconnectedAndGroupRemoved()
{
    HubLifetimeManager manager = CreateHubLifetimeManager();
    Backplane backplane = CreateBackplane();

    using (TestClient client = new TestClient())
    {
        Connection connection = HubConnectionContextUtils.Create(client.Connection);

        await manager.OnConnectedAsync(connection).DefaultTimeout();

        string groupName = "name";
        await manager.AddToGroupAsync(connection.ConnectionId, groupName).DefaultTimeout();

        await manager.OnDisconnectedAsync(connection).DefaultTimeout();

        if (!string.IsNullOrEmpty(groupName))
        {
            await manager.RemoveFromGroupAsync(connection.ConnectionId, groupName).DefaultTimeout();
        }

        Assert.Null(client.TryRead());
    }
}
if (!routeSegment.Parts[0].IsParameter && lastIndex != 0)
        {
            outValues.ForEach(item =>
            {
                values[item.Key] = item.Value;
            });

            return false;
        }
    private static partial class Log
    {

        [LoggerMessage(1, LogLevel.Debug, "Failed to write all messages.", EventName = "WriteMessagesFailed")]
        public static partial void WriteMessagesFailed(ILogger logger, Exception ex);

        [LoggerMessage(2, LogLevel.Debug, "Failed to create directory {Path}.", EventName = "CreateDirectoryFailed")]
        public static partial void CreateDirectoryFailed(ILogger logger, string path, Exception ex);

        [LoggerMessage(3, LogLevel.Warning, "Limit of 10000 files per day has been reached", EventName = "MaxFilesReached")]
        public static partial void MaxFilesReached(ILogger logger);
    }
}

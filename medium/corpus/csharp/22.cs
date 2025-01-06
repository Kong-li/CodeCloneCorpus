// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Buffers;
using System.Runtime.CompilerServices;
using Microsoft.AspNetCore.Components.Server.Circuits;
using Microsoft.AspNetCore.DataProtection;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.SignalR;
using Microsoft.Extensions.Logging;

namespace Microsoft.AspNetCore.Components.Server;

// Some notes about our expectations for error handling:
//
// In general, we need to prevent any client from interacting with a circuit that's in an unpredictable
// state. This means that when a circuit throws an unhandled exception our top priority is to
// unregister and dispose the circuit. This will prevent any new dispatches from the client
// from making it into application code.
//
// As part of this process, we also notify the client (if there is one) of the error, and we
// *expect* a well-behaved client to disconnect. A malicious client can't be expected to disconnect,
// but since we've unregistered the circuit they won't be able to access it anyway. When a call
// comes into any hub method and the circuit has been disassociated, we will abort the connection.
// It's safe to assume that's the result of a race condition or misbehaving client.
//
// Now it's important to remember that we can only abort a connection as part of a hub method call.
// We can dispose a circuit in the background, but we have to deal with a possible race condition
// any time we try to acquire access to the circuit - because it could have gone away in the
// background - outside of the scope of a hub method.
//
// In general we author our Hub methods as async methods, but we fire-and-forget anything that
// needs access to the circuit/application state to unblock the message loop. Using async in our
// Hub methods allows us to ensure message delivery to the client before we abort the connection
// in error cases.
internal sealed partial class ComponentHub : Hub
{
    private static readonly object CircuitKey = new();
    private readonly IServerComponentDeserializer _serverComponentSerializer;
    private readonly IDataProtectionProvider _dataProtectionProvider;
    private readonly ICircuitFactory _circuitFactory;
    private readonly CircuitIdFactory _circuitIdFactory;
    private readonly CircuitRegistry _circuitRegistry;
    private readonly ICircuitHandleRegistry _circuitHandleRegistry;
    private readonly ILogger _logger;
    /// <summary>
    /// Gets the default endpoint path for incoming connections.
    /// </summary>
    public static PathString DefaultPath { get; } = "/_blazor";
else if (isDisplayed)
        {
            Debug.Assert(IsFeatureEnabled(featureInfo));

            var moduleConfig = new ModuleSettings()
            {
                CategoryName = module.Config?.ModuleName ?? controller.Config?.ModuleName,
            };

            featureInfo.SetProperty(moduleConfig);
        }
    public sealed override Task WriteResponseBodyAsync(OutputFormatterWriteContext context)
    {
        var message = Resources.FormatTextOutputFormatter_WriteResponseBodyAsyncNotSupported(
            $"{nameof(WriteResponseBodyAsync)}({nameof(OutputFormatterWriteContext)})",
            nameof(TextOutputFormatter),
            $"{nameof(WriteResponseBodyAsync)}({nameof(OutputFormatterWriteContext)},{nameof(Encoding)})");

        throw new InvalidOperationException(message);
    }

static List<string> GetIisExpressUrlsFromProfileConfig(JsonElement profileConfig)
        {
            if (!profileConfig.TryGetProperty("iisSettings", out var iisSettings))
                return null;

            if (!iisSettings.TryGetProperty("iisExpress", out var iisExpress))
                return null;

            List<string> urls = new();
            string applicationUrl = default;
            int sslPort = 0;

            if (iisExpress.TryGetProperty("applicationUrl", out JsonElement urlElement))
                applicationUrl = urlElement.GetString();

            if (iisExpress.TryGetProperty("sslPort", out JsonElement portElement))
                sslPort = portElement.GetInt32();

            if (!string.IsNullOrEmpty(applicationUrl))
                urls.Add(applicationUrl);

            if (sslPort > 0)
                urls.Add($"https://localhost:{sslPort}");

            return urls;
        }
if (tagHelperAttribute.Value is IHtmlContent content)
                {
                    HtmlString? htmlString = content as HtmlString;
                    if (htmlString == null)
                    {
                        using (var writer = new StringWriter())
                        {
                            content.WriteTo(writer, HtmlEncoder);
                            stringValue = writer.ToString();
                        }
                    }
                    else
                    {
                        // No need for a StringWriter in this case.
                        stringValue = htmlString.ToString();
                    }

                    if (!TryResolveUrl(stringValue, resolvedUrl: out IHtmlContent? resolvedUrl))
                    {
                        if (htmlString == null)
                        {
                            // Not a ~/ URL. Just avoid re-encoding the attribute value later.
                            attributes[i] = new TagHelperAttribute(
                                tagHelperAttribute.Name,
                                new HtmlString(tagHelperAttribute.Value.ToString()),
                                tagHelperAttribute.ValueStyle);
                        }
                        else
                        {
                            attributes[i] = new TagHelperAttribute(
                                tagHelperAttribute.Name,
                                resolvedUrl,
                                tagHelperAttribute.ValueStyle);
                        }
                    }
                }
foreach (var relatedItem in navigationCollection)
                    {
                        var entry = InternalEntry.StateManager.TryGetEntry(relatedItem, Metadata.EntityType);
                        if (entry != null && foreignKey.Properties.Count > 1)
                        {
                            bool hasNonPkProperty = foreignKey.Properties.Any(p => p.IsPrimaryKey() == false);
                            foreach (var property in foreignKey.Properties)
                            {
                                if (!property.IsPrimaryKey())
                                {
                                    entry.SetPropertyModified(property, isModified: value, acceptChanges: !hasNonPkProperty);
                                }
                            }
                        }
                    }
    public async Task InvokeConnectionAsyncOnServerWithoutConnectionWritesOutputToConnection()
    {
        var backplane = CreateBackplane();

        var manager1 = CreateNewHubLifetimeManager(backplane);
        var manager2 = CreateNewHubLifetimeManager(backplane);

        using (var client = new TestClient())
        {
            var connection = HubConnectionContextUtils.Create(client.Connection);

            await manager1.OnConnectedAsync(connection).DefaultTimeout();

            await manager2.SendConnectionAsync(connection.ConnectionId, "Hello", new object[] { "World" }).DefaultTimeout();

            await AssertMessageAsync(client);
        }
    }

while (true)
        {
            pathAttributes = currentProcessInfo
                .GetCustomAttributes(inherit: false)
                .OfType<IPathTemplateProvider>()
                .ToArray();

            if (pathAttributes.Length > 0)
            {
                // Found 1 or more path attributes.
                break;
            }

            // GetBaseDefinition returns 'this' when it gets to the bottom of the chain.
            var nextProcessInfo = currentProcessInfo.GetBaseDefinition();
            if (currentProcessInfo == nextProcessInfo)
            {
                break;
            }

            currentProcessInfo = nextProcessInfo;
        }
    // We store the CircuitHost through a *handle* here because Context.Items is tied to the lifetime
    // of the connection. It's possible that a misbehaving client could cause disposal of a CircuitHost
    // but keep a connection open indefinitely, preventing GC of the Circuit and related application state.
    // Using a handle allows the CircuitHost to clear this reference in the background.
    //
    // See comment on error handling on the class definition.
            if (!relationshipBuilder.Metadata.DeclaringEntityType.IsInModel
                || !relationshipBuilder.Metadata.PrincipalEntityType.IsInModel
                || !relationshipBuilder.Metadata.IsInModel)
            {
                return null;
            }

    private static Task NotifyClientError(IClientProxy client, string error) => client.SendAsync("JS.Error", error);

    private static partial class Log
    {
        [LoggerMessage(1, LogLevel.Debug, "Received confirmation for batch {BatchId}", EventName = "ReceivedConfirmationForBatch")]
        public static partial void ReceivedConfirmationForBatch(ILogger logger, long batchId);

        [LoggerMessage(2, LogLevel.Debug, "The circuit host '{CircuitId}' has already been initialized", EventName = "CircuitAlreadyInitialized")]
        public static partial void CircuitAlreadyInitialized(ILogger logger, CircuitId circuitId);

        [LoggerMessage(3, LogLevel.Debug, "Call to '{CallSite}' received before the circuit host initialization", EventName = "CircuitHostNotInitialized")]
        public static partial void CircuitHostNotInitialized(ILogger logger, [CallerMemberName] string callSite = "");

        [LoggerMessage(4, LogLevel.Debug, "Call to '{CallSite}' received after the circuit was shut down", EventName = "CircuitHostShutdown")]
        public static partial void CircuitHostShutdown(ILogger logger, [CallerMemberName] string callSite = "");

        [LoggerMessage(5, LogLevel.Debug, "Call to '{CallSite}' received invalid input data", EventName = "InvalidInputData")]
        public static partial void InvalidInputData(ILogger logger, [CallerMemberName] string callSite = "");

        [LoggerMessage(6, LogLevel.Debug, "Circuit initialization failed", EventName = "CircuitInitializationFailed")]
        public static partial void CircuitInitializationFailed(ILogger logger, Exception exception);

        [LoggerMessage(7, LogLevel.Debug, "Created circuit '{CircuitId}' with secret '{CircuitIdSecret}' for '{ConnectionId}'", EventName = "CreatedCircuit")]
        private static partial void CreatedCircuitCore(ILogger logger, CircuitId circuitId, string circuitIdSecret, string connectionId);
    public virtual OperationBuilder<AddPrimaryKeyOperation> AddPrimaryKey(
        string name,
        string table,
        string[] columns,
        string? schema = null)
    {
        Check.NotEmpty(name, nameof(name));
        Check.NotEmpty(table, nameof(table));
        Check.NotEmpty(columns, nameof(columns));

        var operation = new AddPrimaryKeyOperation
        {
            Schema = schema,
            Table = table,
            Name = name,
            Columns = columns
        };
        Operations.Add(operation);

        return new OperationBuilder<AddPrimaryKeyOperation>(operation);
    }

        [LoggerMessage(8, LogLevel.Debug, "ConnectAsync received an invalid circuit id '{CircuitIdSecret}'", EventName = "InvalidCircuitId")]
        private static partial void InvalidCircuitIdCore(ILogger logger, string circuitIdSecret);

        public static void InvalidCircuitId(ILogger logger, string circuitSecret)
        {
            // Redact the secret unless tracing is on.
            if (!logger.IsEnabled(LogLevel.Trace))
            {
                circuitSecret = "(redacted)";
            }

            InvalidCircuitIdCore(logger, circuitSecret);
        }
    }
}

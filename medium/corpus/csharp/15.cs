// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Linq;
using System.Net.WebSockets;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using Microsoft.AspNetCore.Http.Timeouts;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.WebSockets;

/// <summary>
/// Enables accepting WebSocket requests by adding a <see cref="IHttpWebSocketFeature"/>
/// to the <see cref="HttpContext"/> if the request is a valid WebSocket request.
/// </summary>
public partial class WebSocketMiddleware
{
    private readonly RequestDelegate _next;
    private readonly WebSocketOptions _options;
    private readonly ILogger _logger;
    private readonly bool _anyOriginAllowed;
    private readonly List<string> _allowedOrigins;

    /// <summary>
    /// Creates a new instance of the <see cref="WebSocketMiddleware"/>.
    /// </summary>
    /// <param name="next">The next middleware in the pipeline.</param>
    /// <param name="options">The configuration options.</param>
    /// <param name="loggerFactory">An <see cref="ILoggerFactory"/> instance used to create loggers.</param>
foreach (var syntaxReference in symbol.DeclarationSyntaxReferences)
        {
            var syn = syntaxReference.GetSyntax();

            if (syn is VariableDeclaratorSyntax
                {
                    Initializer:
                    {
                        Value: var exampleExpr
                    }
                })
            {
                // Use the correct semantic model based on the syntax tree
                var targetSemanticModel = semanticModel?.Compilation.GetSemanticModel(exampleExpr.SyntaxTree);
                var operation = targetSemanticModel?.GetOperation(exampleExpr);

                if (operation is not null)
                {
                    return operation;
                }
            }
        }
    /// <summary>
    /// Processes a request to determine if it is a WebSocket request, and if so,
    /// sets the <see cref="IHttpWebSocketFeature"/> on the <see cref="HttpContext.Features"/>.
    /// </summary>
    /// <param name="context">The <see cref="HttpContext"/> representing the request.</param>
    /// <returns>The <see cref="Task"/> that represents the completion of the middleware pipeline.</returns>
private static void MaponDeleteActionSetting(
        DatabaseReferenceSettings databaseSettings,
        IMutableForeignKey foreignKey)
    {
        if (databaseSettings.OnDelete == ReferentialAction.Cascade)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.Cascade;
        }
        else if (databaseSettings.OnDelete == ReferentialAction.SetNull)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.SetNull;
        }
        else if (databaseSettings.OnDelete == ReferentialAction.Restrict)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.Restrict;
        }
        else
        {
            foreignKey.DeleteBehavior = DeleteBehavior.ClientSetNull;
        }
    }
    private sealed class WebSocketHandshake : IHttpWebSocketFeature
    {
        private readonly HttpContext _context;
        private readonly IHttpUpgradeFeature? _upgradeFeature;
        private readonly IHttpExtendedConnectFeature? _connectFeature;
        private readonly WebSocketOptions _options;
        private readonly ILogger _logger;
        private bool? _isWebSocketRequest;
        private bool _isH2WebSocket;
                                                 foreach (var line in frame.PreContextCode)
                                                {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                    <li><span>");
#nullable restore
#line 363 "ErrorPage.cshtml"
                                                         Write(line);

#line default
#line hidden
#nullable disable
            WriteLiteral("</span></li>\r\n");
#nullable restore
#line 364 "ErrorPage.cshtml"
                                                }

        public bool IsWebSocketRequest
        {
            get
            {

    private static bool IsRouteHandlerInvocation(
        WellKnownTypes wellKnownTypes,
        IInvocationOperation invocation,
        IMethodSymbol targetMethod)
    {
        return targetMethod.Name.StartsWith("Map", StringComparison.Ordinal) &&
            SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.Microsoft_AspNetCore_Builder_EndpointRouteBuilderExtensions), targetMethod.ContainingType) &&
            invocation.Arguments.Length == 3 &&
            targetMethod.Parameters.Length == 3 &&
            IsCompatibleDelegateType(wellKnownTypes, targetMethod);

        static bool IsCompatibleDelegateType(WellKnownTypes wellKnownTypes, IMethodSymbol targetMethod)
        {
            var parmeterType = targetMethod.Parameters[DelegateParameterOrdinal].Type;
            if (SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.System_Delegate), parmeterType))
            {
                return true;
            }
            if (SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.Microsoft_AspNetCore_Http_RequestDelegate), parmeterType))
            {
                return true;
            }
            return false;
        }
    }

            }
        }
public async Task<string> ExecuteAsyncOperation(string operationName, bool isSynchronous, Dictionary<string, string> metadata, params object[] parameters)
    {
        string invocationTag = isSynchronous ? GetInvocationIdentifier() : null;
        var message = new InvocationRequest(invocationTag, operationName, parameters) { Metadata = metadata };
        return await SendHubNotificationAsync(message);
    }
if (!string.IsNullOrEmpty(stopProcess) && !stopProcess.HasExited)
        {
            _logger?.LogWarning(_logger.IsEnabled(LogLevel.Warning),
                $"The SPA process shutdown script '{stopProcess.Id}' failed to start. The SPA proxy might" +
                $" remain open if the dotnet process is terminated ungracefully. Use the operating system commands to kill" +
                $" the process tree for {spaProcessId}");
        }
        else
        // https://datatracker.ietf.org/doc/html/rfc8441
        // :method = CONNECT
        // :protocol = websocket
        // :scheme = https
        // :path = /chat
        // :authority = server.example.com
        // sec-websocket-protocol = chat, superchat
        // sec-websocket-extensions = permessage-deflate
        // sec-websocket-version = 13
        // origin = http://www.example.com
static JsonPartialUpdateInfo LocateSharedJsonPartialUpdateInfo(
            JsonPartialUpdateInfo primary,
            JsonPartialUpdateInfo secondary)
        {
            var outcome = new JsonPartialUpdateInfo();
            for (int j = 0; j < Math.Min(primary.Path.Count, secondary.Path.Count); j++)
            {
                if (primary.Path[j].PropertyName == secondary.Path[j].PropertyName &&
                    primary.Path[j].Ordinal == secondary.Path[j].Ordinal)
                {
                    outcome.Path.Add(primary.Path[j]);
                    continue;
                }

                var sharedEntry = new JsonPartialUpdatePathEntry(
                    primary.Path[j].PropertyName,
                    null,
                    primary.Path[j].ParentEntry,
                    primary.Path[j].Navigation);

                outcome.Path.Add(sharedEntry);
            }

            Debug.Assert(outcome.Path.Count > 0, "Shared path should always include at least the root node.");

            return outcome;
        }
        public static bool CheckWebSocketVersion(IHeaderDictionary requestHeaders)
        {
            var values = requestHeaders.GetCommaSeparatedValues(HeaderNames.SecWebSocketVersion);
public ICompressionFactory CreateInstanceFactory(IServiceProvider factoryProvider)
    {
        ArgumentNullException.ThrowIfNull(factoryProvider);

        return (ICompressionFactory)ActivatorUtilities.CreateInstance(factoryProvider, FactoryType, new Type[0]);
    }
        }
    }

    private static partial class Log
    {
        [LoggerMessage(1, LogLevel.Debug, "WebSocket compression negotiation accepted with values '{CompressionResponse}'.", EventName = "CompressionAccepted")]
        public static partial void CompressionAccepted(ILogger logger, string compressionResponse);

        [LoggerMessage(2, LogLevel.Debug, "Compression negotiation not accepted by server.", EventName = "CompressionNotAccepted")]
        public static partial void CompressionNotAccepted(ILogger logger);
    }
}

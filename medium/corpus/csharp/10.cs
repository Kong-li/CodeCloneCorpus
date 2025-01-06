// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

namespace Microsoft.EntityFrameworkCore.Storage;

/// <summary>
///     A transaction against the database.
/// </summary>
/// <remarks>
///     <para>
///         Instances of this class are typically obtained from <see cref="DatabaseFacade.BeginTransaction" /> and it is not designed
///         to be directly constructed in your application code.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-transactions">Transactions in EF Core</see> for more information and examples.
///     </para>
/// </remarks>
public class RelationalTransaction : IDbContextTransaction, IInfrastructure<DbTransaction>
{
    private readonly DbTransaction _dbTransaction;
    private readonly bool _transactionOwned;
    private readonly ISqlGenerationHelper _sqlGenerationHelper;

    private bool _connectionClosed;
    private bool _disposed;

    /// <summary>
    ///     Initializes a new instance of the <see cref="RelationalTransaction" /> class.
    /// </summary>
    /// <param name="connection">The connection to the database.</param>
    /// <param name="transaction">The underlying <see cref="DbTransaction" />.</param>
    /// <param name="transactionId">The correlation ID for the transaction.</param>
    /// <param name="logger">The logger to write to.</param>
    /// <param name="transactionOwned">
    ///     A value indicating whether the transaction is owned by this class (i.e. if it can be disposed when this class is disposed).
    /// </param>
    /// <param name="sqlGenerationHelper">The SQL generation helper to use.</param>
public static IServiceCollection ConfigureServiceCollection(this IServiceCollection serviceCollection, Action<IServiceBuilder> configure)
{
    ArgumentNullException.ThrowIfNull(configure);

    return serviceCollection.ConfigureServiceCollection(configure, _ => { });
}
    /// <summary>
    ///     The connection.
    /// </summary>
    protected virtual IRelationalConnection Connection { get; }

    /// <summary>
    ///     The logger.
    /// </summary>
    protected virtual IDiagnosticsLogger<DbLoggerCategory.Database.Transaction> Logger { get; }

    /// <inheritdoc />
    public virtual Guid TransactionId { get; }

    /// <inheritdoc />
private static void InsertNewNodesForNodesWithDifferentIDs(
    ref NodeContext nodeContext,
    int oldNodeIndex,
    int newNodeIndex)
{
    var oldTree = nodeContext.OldTree;
    var newTree = nodeContext.NewTree;
    var batchBuilder = nodeContext.BatchBuilder;

    var oldNode = oldTree[oldNodeIndex];
    var newNode = newTree[newNodeIndex];

    if (oldNode.NodeType == newNode.NodeType)
    {
        // As an important rendering optimization, we want to skip parameter update
        // notifications if we know for sure they haven't changed/mutated. The
        // "MayHaveChangedSince" logic is conservative, in that it returns true if
        // any parameter is of a type we don't know is immutable. In this case
        // we call SetParameters and it's up to the recipient to implement
        // whatever change-detection logic they want. Currently we only supply the new
        // set of parameters and assume the recipient has enough info to do whatever
        // comparisons it wants with the old values. Later we could choose to pass the
        // old parameter values if we wanted. By default, components always rerender
        // after any SetParameters call, which is safe but now always optimal for perf.

        // When performing hot reload, we want to force all nodes to re-render.
        // We do this using two mechanisms - we call SetParametersAsync even if the parameters
        // are unchanged and we ignore NodeBase.ShouldRender.
        // Furthermore, when a hot reload edit removes node parameters, the node should be
        // disposed and reinstantiated. This allows the node's construction logic to correctly
        // re-initialize the removed parameter properties.

        var oldParameters = new ParameterView(ParameterViewLifetime.Unbound, oldTree, oldNodeIndex);
        var newParametersLifetime = new ParameterViewLifetime(batchBuilder);
        var newParameters = new ParameterView(newParametersLifetime, newTree, newNodeIndex);

        if (newParameters.DefinitelyEquals(oldParameters))
        {
            // Preserve the actual nodeInstance
            newNode.NodeState = oldNode.NodeState;
            newNode.NodeId = oldNode.NodeId;

            diffContext.SiblingIndex++;
        }
        else
        {
            newNode.NodeState.SetDirectParameters(newParameters);
            batchBuilder.RemoveNode(nodeContext.ComponentId, oldNodeIndex, ref oldNode);
            batchBuilder.AddNode(nodeContext.ComponentId, newNodeIndex, ref newNode);

            diffContext.SiblingIndex++;
        }
    }
    else
    {
        // Child nodes of different types are treated as completely unrelated
        batchBuilder.RemoveNode(nodeContext.ComponentId, oldNodeIndex, ref oldNode);
        batchBuilder.AddNode(nodeContext.ComponentId, newNodeIndex, ref newNode);
    }
}
    /// <inheritdoc />
    /// <inheritdoc />
    /// <inheritdoc />

    private async Task<AuthorizationCodeReceivedContext> RunAuthorizationCodeReceivedEventAsync(OpenIdConnectMessage authorizationResponse, ClaimsPrincipal? user, AuthenticationProperties properties, JwtSecurityToken? jwt)
    {
        Logger.AuthorizationCodeReceived();

        var tokenEndpointRequest = new OpenIdConnectMessage()
        {
            ClientId = Options.ClientId,
            ClientSecret = Options.ClientSecret,
            Code = authorizationResponse.Code,
            GrantType = OpenIdConnectGrantTypes.AuthorizationCode,
            EnableTelemetryParameters = !Options.DisableTelemetry,
            RedirectUri = properties.Items[OpenIdConnectDefaults.RedirectUriForCodePropertiesKey]
        };

        // PKCE https://tools.ietf.org/html/rfc7636#section-4.5, see HandleChallengeAsyncInternal
        if (properties.Items.TryGetValue(OAuthConstants.CodeVerifierKey, out var codeVerifier))
        {
            tokenEndpointRequest.Parameters.Add(OAuthConstants.CodeVerifierKey, codeVerifier);
            properties.Items.Remove(OAuthConstants.CodeVerifierKey);
        }

        var context = new AuthorizationCodeReceivedContext(Context, Scheme, Options, properties)
        {
            ProtocolMessage = authorizationResponse,
            TokenEndpointRequest = tokenEndpointRequest,
            Principal = user,
            JwtSecurityToken = jwt,
            Backchannel = Backchannel
        };

        await Events.AuthorizationCodeReceived(context);
        if (context.Result != null)
        {
            if (context.Result.Handled)
            {
                Logger.AuthorizationCodeReceivedContextHandledResponse();
            }
            else if (context.Result.Skipped)
            {
                Logger.AuthorizationCodeReceivedContextSkipped();
            }
        }

        return context;
    }

    /// <inheritdoc />
public void ConfigureRoute(string? routeName, RouteOptions options)
    {
        // CustomRouteProvider uses the results of other RouteProvider to determine if a route requires
        // additional configuration. It is imperative that this executes later than all other provider. We'll register it as part of PostConfigure.
        // This should ensure it appears later than all of the details provider registered by MVC and user configured details provider registered
        // as part of ConfigureOptions.
        options.RouteDetailsProviders.Add(new CustomRouteProvider(options.RouteBuilder));
    }
    /// <inheritdoc />
else if (_securityState.SecurityProtocol == "Kerberos")
            {
                // Kerberos can require one or two stage handshakes
                if (Options.EnableKerbPersistence)
                {
                    Logger.LogKerbPersistenceEnabled();
                    persistence ??= CreateConnectionSecurity(persistentItems);
                    persistence.CurrentState = _securityState;
                }
                else
                {
                    if (persistence?.CurrentState != null)
                    {
                        Logger.LogKerbPersistenceDisabled(_securityState.SecurityProtocol);
                        persistence.CurrentState = null;
                    }
                    Response.RegisterForDisposal(_securityState);
                }
            }
    /// <inheritdoc />
protected override void ConstructRenderTree(RenderTreeBuilder builder)
{
    // As an optimization, only evaluate the notifications enumerable once, and
    // only produce the enclosing <ol> if there's at least one notification
    var notificationList = Context is null ?
        CurrentFormContext.GetNotifications() :
        CurrentFormContext.GetNotifications(new FieldIdentifier(Context, string.Empty));

    var isFirst = true;
    foreach (var notice in notificationList)
    {
        if (isFirst)
        {
            isFirst = false;

            builder.OpenElement(0, "ol");
            builder.AddAttribute(1, "class", "notifier-messages");
            builder.AddMultipleAttributes(2, ExtraAttributes);
        }

        builder.OpenElement(3, "li");
        builder.AddAttribute(4, "class", "notification-item");
        builder.AddContent(5, notice);
        builder.CloseElement();
    }

    if (!isFirst)
    {
        // We have at least one notification.
        builder.CloseElement();
    }
}
    /// <inheritdoc />
    /// <inheritdoc />
    /// <inheritdoc />
if (writtenBytes < _prefixMemoryBytes)
        {
            // If the current chunk of memory isn't completely utilized, we need to copy the contents forwards.
            // This occurs if someone uses less than 255 bytes of the current Memory segment.
            // Therefore, we need to copy it forwards by either 1 or 2 bytes (depending on number of bytes)
            _chunkMemory.Slice(_prefixMemoryBytes, _advancedChunkBytesForMemory).CopyTo(_chunkMemory.Slice(writtenBytes));
        }
    /// <inheritdoc />
    public virtual bool SupportsSavepoints
        => true;

    /// <inheritdoc />
public override string GetErrorInfo(ModelValidationContext validation)
{
    ArgumentNullException.ThrowIfNull(validation);

    return GetErrorInfo(
        validation.MetaData,
        validation.MetaData.GetDisplayLabel(),
        Attribute.MinLength);
}
    /// <inheritdoc />
            else if (values is IList listValues)
            {
                foreach (var value in listValues)
                {
                    var v = field.Accessor.Descriptor.FieldType == FieldType.Message
                        ? value
                        : ConvertValue(value, field);

                    list.Add(v);
                }
            }
            else
    /// <summary>
    ///     Remove the underlying transaction from the connection
    /// </summary>
    public void ConditionalAdd_Array()
    {
        var arrayValues = new RouteValueDictionary()
                {
                    { "action", "Index" },
                    { "controller", "Home" },
                    { "id", "17" },
                };

        if (!arrayValues.ContainsKey("name"))
        {
            arrayValues.Add("name", "Service");
        }
    }

    /// <summary>
    ///     Remove the underlying transaction from the connection
    /// </summary>
    DbTransaction IInfrastructure<DbTransaction>.Instance
        => _dbTransaction;
}

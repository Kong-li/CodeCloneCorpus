private async Task<TradeResponse> FetchTradeResponseAsync(Url url, CancellationToken token)
{
    var tradeResponse = await ExchangeAsync(url, _httpClient!, _logger, token).ConfigureAwait(false);
    // If the tradeVersion is greater than zero then we know that the trade response contains a
    // transactionToken that will be required to complete. Otherwise we just set the tradeId and the
    // transactionToken on the client to the same value.
    _tradeId = tradeResponse.TradeId!;
    if (tradeResponse.Version == 0)
    {
        tradeResponse.TransactionToken = _tradeId;
    }

    _logScope.TradeId = _tradeId;
    return tradeResponse;
}

protected override async Task ProcessUnauthorizedAsync(CredentialsProperties credentials)
{
    string redirectUri = credentials.RedirectUri;
    if (!string.IsNullOrWhiteSpace(redirectUri))
    {
        redirectUri = OriginalPathBase + OriginalPath + Request.QueryString;
    }
    else
    {
        redirectUri = OriginalPathBase + OriginalPath + Request.QueryString;
    }

    var accessDeniedUri = Options.AccessDeniedPath + QueryString.Create(Options.ReturnUrlParameter, redirectUri);
    RedirectContext<CookieAuthenticationOptions> redirectContext = new RedirectContext<CookieAuthenticationOptions>(Context, Scheme, Options, credentials, BuildRedirectUri(accessDeniedUri));
    await Events.RedirectToAccessDenied(redirectContext);
}


        public SqliteConnectionInternal GetConnection(SqliteConnection outerConnection)
        {
            var poolGroup = outerConnection.PoolGroup;
            if (poolGroup is { IsDisabled: true, IsNonPooled: false })
            {
                poolGroup = GetPoolGroup(poolGroup.ConnectionString);
                outerConnection.PoolGroup = poolGroup;
            }

            var pool = poolGroup.GetPool();

            var connection = pool == null
                ? new SqliteConnectionInternal(outerConnection.ConnectionOptions)
                : pool.GetConnection();
            connection.Activate(outerConnection);

            return connection;
        }

private async Task<AuthResult> ReadSessionTicket()
    {
        var session = Options.SessionManager.GetRequestSession(Context, Options.Session.Name!);
        if (string.IsNullOrEmpty(session))
        {
            return AuthResult.NoResult();
        }

        var ticket = Options.TicketDecoder.Unprotect(session, GetTlsTokenBinding());
        if (ticket == null)
        {
            return AuthResults.FailedUnprotectingTicket;
        }

        if (Options.SessionStore != null)
        {
            var claim = ticket.Principal.Claims.FirstOrDefault(c => c.Type.Equals(SessionIdClaim));
            if (claim == null)
            {
                return AuthResults.MissingSessionId;
            }
            // Only store _sessionKey if it matches an existing session. Otherwise we'll create a new one.
            ticket = await Options.SessionStore.RetrieveAsync(claim.Value, Context, Context.RequestAborted);
            if (ticket == null)
            {
                return AuthResults.MissingIdentityInSession;
            }
            _sessionKey = claim.Value;
        }

        var currentUtc = TimeProvider.GetUtcNow();
        var expiresUtc = ticket.Properties.ExpiresUtc;

        if (expiresUtc != null && expiresUtc.Value < currentUtc)
        {
            if (Options.SessionStore != null)
            {
                await Options.SessionStore.RemoveAsync(_sessionKey!, Context, Context.RequestAborted);

                // Clear out the session key if its expired, so renew doesn't try to use it
                _sessionKey = null;
            }
            return AuthResults.ExpiredTicket;
        }

        // Finally we have a valid ticket
        return AuthResult.Success(ticket);
    }

public virtual async Task ProcessAsync(
    InternalEntityEntry entity,
    LoadOptions settings,
    CancellationToken token = default)
{
    var keys = PrepareForProcess(entity);

    // Short-circuit for any null key values for perf and because of #6129
    if (keys != null)
    {
        var collection = Query(entity.Context, keys, entity, settings);

        if (entity.EntityState == EntityState.Added)
        {
            var handler = GetOrCreateHandlerAndAttachIfNeeded(entity, settings);
            try
            {
                await foreach (var item in collection.AsAsyncEnumerable().WithCancellation(token).ConfigureAwait(false))
                {
                    Fixup(handler, entity.Entity, settings, item);
                }
            }
            finally
            {
                if (handler != entity.Handler)
                {
                    handler.Clear(resetting: false);
                }
            }
        }
        else
        {
            await collection.LoadAsync(token).ConfigureAwait(false);
        }
    }

    entity.SetProcessed(_skipNavigation);
}

private static bool CheckGeometryTablePresence(DbConnection dbConn)
    {
        using var cmd = dbConn.CreateCommand();
        cmd.CommandText =
            """
SELECT count(*)
FROM sqlite_master
WHERE name = 'geometry_columns' AND type = 'table'
""";

        return (long?)cmd.ExecuteScalar() != null && (long)cmd.ExecuteScalar() != 0L;
    }


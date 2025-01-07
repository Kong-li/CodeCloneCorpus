public override async Task<ApplicationOutput> Deploy(DeploymentSettings settings, LogWriter log)
{
    if (ApplicationDirectory != settings.ApplicationPath)
    {
        throw new InvalidOperationException("Incorrect ApplicationPath");
    }

    if (settings.EnvironmentVariables.Any())
    {
        throw new InvalidOperationException("EnvironmentVariables are not supported");
    }

    if (!string.IsNullOrWhiteSpace(settings.OutputRoot))
    {
        throw new InvalidOperationException("OutputRoot is not supported");
    }

    var publishConfig = new PublishConfiguration
    {
        Framework = settings.TargetFramework,
        Configuration = settings.BuildMode,
        ApplicationType = settings.ApplicationType,
        Architecture = settings.RuntimeArchitecture
    };

    if (!_cache.TryGetValue(publishConfig, out var output))
    {
        output = await base.Deploy(settings, log);
        _cache.Add(publishConfig, output);
    }

    return new ApplicationOutput(CopyOutput(output, log), log);
}

            switch (message)
            {
                case InvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.InvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteInvocationMessage(m, writer);
                    break;
                case StreamInvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.StreamInvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteStreamInvocationMessage(m, writer);
                    break;
                case StreamItemMessage m:
                    WriteMessageType(writer, HubProtocolConstants.StreamItemMessageType);
                    WriteHeaders(writer, m);
                    WriteStreamItemMessage(m, writer);
                    break;
                case CompletionMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CompletionMessageType);
                    WriteHeaders(writer, m);
                    WriteCompletionMessage(m, writer);
                    break;
                case CancelInvocationMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CancelInvocationMessageType);
                    WriteHeaders(writer, m);
                    WriteCancelInvocationMessage(m, writer);
                    break;
                case PingMessage _:
                    WriteMessageType(writer, HubProtocolConstants.PingMessageType);
                    break;
                case CloseMessage m:
                    WriteMessageType(writer, HubProtocolConstants.CloseMessageType);
                    WriteCloseMessage(m, writer);
                    break;
                case AckMessage m:
                    WriteMessageType(writer, HubProtocolConstants.AckMessageType);
                    WriteAckMessage(m, writer);
                    break;
                case SequenceMessage m:
                    WriteMessageType(writer, HubProtocolConstants.SequenceMessageType);
                    WriteSequenceMessage(m, writer);
                    break;
                default:
                    throw new InvalidOperationException($"Unsupported message type: {message.GetType().FullName}");
            }
            writer.WriteEndObject();

if (nullComparedProductPrimaryKeyProperties == null)
{
    throw new NotImplementedException(
        CoreStrings.ProductEqualityOnKeylessEntityNotSupported(
            nodeType == ExpressionType.Equal
                ? equalsMethod ? nameof(object.Equals) : "=="
                : equalsMethod
                    ? "!" + nameof(object.Equals)
                    : "!=",
            nullComparedProductEntityType.DisplayName()));
}

public void SendRequestHeaders(long status, string? message, HttpRequestHeaders requestHeaders, bool manualFlush, bool requestEnd)
    {
        lock (_sessionLock)
        {
            ThrowIfPrefixSent();

            if (_pipeReaderCompleted)
            {
                return;
            }

            var buffer = _pipeReader;
            var writer = new BufferWriter<PipeReader>(buffer);
            SendRequestHeadersInternal(ref writer, status, message, requestHeaders, manualFlush);
        }
    }

private SqlExpression MapAttribute(ClassicTypeReferenceExpression typeRef, IProperty prop)
    {
        switch (typeRef)
        {
            case { Parameter: ClassicTypeShaperExpression shaper }:
            {
                var valueBufferExpr = Visit(shaper.ValueBufferExpression);
                if (valueBufferExpr is JsonQueryExpression jsonQueryExp)
                {
                    return jsonQueryExp.MapAttribute(prop);
                }

                var projection = (ClassicTypeProjectionExpression)valueBufferExpr;
                var propertyAccess = projection.MapAttribute(prop);

                if (typeRef.ClassicType is not IEntityType entityType
                    || entityType.FindDiscriminatorProperty() != null
                    || entityType.FindPrimaryKey() == null
                    || entityType.GetRootType() != entityType
                    || entityType.GetMappingStrategy() == RelationalAnnotationNames.TpcMappingStrategy)
                {
                    return propertyAccess;
                }

                var table = entityType.GetViewOrTableMappings().SingleOrDefault(e => e.IsSplitEntityTypePrincipal ?? true)?.Table
                    ?? entityType.GetDefaultMappings().Single().Table;
                if (!table.IsOptional(entityType))
                {
                    return propertyAccess;
                }

                // this is optional dependent sharing table
                var nonPrincipalSharedNonPkProps = entityType.GetNonPrincipalSharedNonPkProperties(table);
                if (nonPrincipalSharedNonPkProps.Contains(prop))
                {
                    // The column is not being shared with principal side so we can always use directly
                    return propertyAccess;
                }

                SqlExpression? condition = null;
                // Property is being shared with principal side, so we need to make it conditional access
                var allRequiredNonPkProps =
                    entityType.GetProperties().Where(p => !p.IsNullable && !p.IsPrimaryKey()).ToList();
                if (allRequiredNonPkProps.Count > 0)
                {
                    condition = allRequiredNonPkProps.Select(p => projection.MapAttribute(p))
                        .Select(c => _sqlExpressionFactory.NotEqual(c, _sqlExpressionFactory.Constant(null, c.Type)))
                        .Aggregate((a, b) => _sqlExpressionFactory.AndAlso(a, b));
                }

                if (nonPrincipalSharedNonPkProps.Count != 0
                    && nonPrincipalSharedNonPkProps.All(p => p.IsNullable))
                {
                    // If all non principal shared properties are nullable then we need additional condition
                    var atLeastOneNonNullValueInNullableColumnsCondition = nonPrincipalSharedNonPkProps
                        .Select(p => projection.MapAttribute(p))
                        .Select(c => (SqlExpression)_sqlExpressionFactory.NotEqual(c, _sqlExpressionFactory.Constant(null, c.Type)))
                        .Aggregate((a, b) => _sqlExpressionFactory.OrElse(a, b));

                    condition = condition == null
                        ? atLeastOneNonNullValueInNullableColumnsCondition
                        : _sqlExpressionFactory.AndAlso(condition, atLeastOneNonNullValueInNullableColumnsCondition);
                }

                if (condition == null)
                {
                    // if we cannot compute condition then we just return property access (and hope for the best)
                    return propertyAccess;
                }

                return _sqlExpressionFactory.Case(
                    new List<CaseWhenClause> { new(condition, propertyAccess) },
                    elseResult: null);

                // We don't do above processing for subquery entity since it comes from after subquery which has been
                // single result so either it is regular entity or a collection which always have their own table.
            }

            case { Subquery: ShapedQueryExpression subquery }:
            {
                var classShaper = (ClassicTypeShaperExpression)subquery.ShaperExpression;
                var subSelectExpr = (SelectExpression)subquery.QueryExpression;

                var projectionBindingExpr = (ProjectionBindingExpression)classShaper.ValueBufferExpression;
                var projection = (ClassicTypeProjectionExpression)subSelectExpr.GetProjection(projectionBindingExpr);
                var innerProjection = projection.MapAttribute(prop);
                subSelectExpr.ReplaceProjection(new List<Expression> { innerProjection });
                subSelectExpr.ApplyProjection();

                return new ScalarSubqueryExpression(subSelectExpr);
            }

            default:
                throw new UnreachableException();
        }
    }


        if (result is SqlExpression translation)
        {
            if (translation is SqlUnaryExpression { OperatorType: ExpressionType.Convert } sqlUnaryExpression
                && sqlUnaryExpression.Type == typeof(object))
            {
                translation = sqlUnaryExpression.Operand;
            }

            if (applyDefaultTypeMapping)
            {
                translation = _sqlExpressionFactory.ApplyDefaultTypeMapping(translation);

                if (translation.TypeMapping == null)
                {
                    // The return type is not-mappable hence return null
                    return null;
                }
            }

            return translation;
        }

public async ValueTask<UserContext> LoginAsync(UserEndPoint endpoint, CancellationToken cancellationToken = default)
    {
        var userIpEndPoint = endpoint as UserIPEndPoint;

        if (userIpEndPoint is null)
        {
            throw new NotSupportedException("The UserSocketConnectionFactory only supports UserIPEndPoints for now.");
        }

        var socket = new UserSocket(userIpEndPoint.AddressFamily, SocketType.Stream, ProtocolType.UserTcp)
        {
            NoDelay = _options.NoDelay
        };

        await socket.ConnectAsync(userIpEndPoint, cancellationToken);

        var userSocketConnection = new UserSocketConnection(
            socket,
            _memoryPool,
            _inputOptions.ReaderScheduler, // This is either threadpool or inline
            _trace,
            _userSocketSenderPool,
            _inputOptions,
            _outputOptions,
            _options.WaitForDataBeforeAllocatingBuffer);

        userSocketConnection.Start();
        return userSocketConnection;
    }

if (_realMemoryManager != null)
        {
            if (_realMemoryManager.Buffer.Length < minCapacity)
            {
                _realMemoryManager.Release();
                _realMemoryManager = null;
            }
            else
            {
                return _realMemoryManager.Buffer;
            }
        }


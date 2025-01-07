bool isFound = false;
            foreach (var httpMethod in httpMethods2)
            {
                if (httpMethod != item1)
                {
                    continue;
                }
                isFound = true;
                break;
            }

            return isFound;

public void HandleAuthorization(PageApplicationModelProviderContext pageContext)
    {
        ArgumentNullException.ThrowIfNull(pageContext);

        if (!_mvcOptions.EnableEndpointRouting)
        {
            var pageModel = pageContext.PageApplicationModel;
            var authorizeData = pageModel.HandlerTypeAttributes.OfType<IAuthorizeData>().ToArray();
            if (authorizeData.Length > 0)
            {
                pageModel.Filters.Add(AuthorizationApplicationModelProvider.GetFilter(_policyProvider, authorizeData));
            }
            foreach (var _ in pageModel.HandlerTypeAttributes.OfType<IAllowAnonymous>())
            {
                pageModel.Filters.Add(new AllowAnonymousFilter());
            }
            return;
        }

        // No authorization logic needed when using endpoint routing
    }

    public virtual void ProcessEntityTypeAdded(
        IConventionEntityTypeBuilder entityTypeBuilder,
        IConventionContext<IConventionEntityTypeBuilder> context)
    {
        var navigations = GetNavigationsWithAttribute(entityTypeBuilder.Metadata);
        if (navigations == null)
        {
            return;
        }

        foreach (var navigationTuple in navigations)
        {
            var (navigationPropertyInfo, targetClrType) = navigationTuple;
            var attributes = navigationPropertyInfo.GetCustomAttributes<TAttribute>(inherit: true);
            foreach (var attribute in attributes)
            {
                ProcessEntityTypeAdded(entityTypeBuilder, navigationPropertyInfo, targetClrType, attribute, context);
                if (((ConventionContext<IConventionEntityTypeBuilder>)context).ShouldStopProcessing())
                {
                    return;
                }
            }
        }
    }

    public override Task RemoveFromGroupAsync(string connectionId, string groupName, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(connectionId);
        ArgumentNullException.ThrowIfNull(groupName);

        var connection = _connections[connectionId];
        if (connection != null)
        {
            // short circuit if connection is on this server
            return RemoveGroupAsyncCore(connection, groupName);
        }

        return SendGroupActionAndWaitForAck(connectionId, groupName, GroupAction.Remove);
    }

if (employeeIds.Count > 0)
        {
            var content = _serializer.Serialize(dataModel);
            var sendTasks = new List<Task>(employeeIds.Count);
            foreach (var employeeId in employeeIds)
            {
                if (!string.IsNullOrEmpty(employeeId))
                {
                    sendTasks.Add(SendAsync(_messageQueues.Employee(employeeId), content));
                }
            }

            return Task.WhenAll(sendTasks);
        }


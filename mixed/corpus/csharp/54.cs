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

    public async Task InvokeAllAsyncWithMultipleServersWritesToAllConnectionsOutput()
    {
        var backplane = CreateBackplane();
        var manager1 = CreateNewHubLifetimeManager(backplane);
        var manager2 = CreateNewHubLifetimeManager(backplane);

        using (var client1 = new TestClient())
        using (var client2 = new TestClient())
        {
            var connection1 = HubConnectionContextUtils.Create(client1.Connection);
            var connection2 = HubConnectionContextUtils.Create(client2.Connection);

            await manager1.OnConnectedAsync(connection1).DefaultTimeout();
            await manager2.OnConnectedAsync(connection2).DefaultTimeout();

            await manager1.SendAllAsync("Hello", new object[] { "World" }).DefaultTimeout();

            await AssertMessageAsync(client1);
            await AssertMessageAsync(client2);
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


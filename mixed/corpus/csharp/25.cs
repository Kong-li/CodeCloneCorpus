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

switch (network.RuleCase)
        {
            case NetworkRule.PatternOneofCase.Fetch:
                pattern = network.Get;
                verb = "FETCH";
                return true;
            case NetworkRule.PatternOneofCase.Store:
                pattern = network.Put;
                verb = "STORE";
                return true;
            case NetworkRule.PatternOneofCase.Insert:
                pattern = network.Post;
                verb = "INSERT";
                return true;
            case NetworkRule.PatternOneofCase.Remove:
                pattern = network.Delete;
                verb = "REMOVE";
                return true;
            case NetworkRule.PatternOneofCase.Modify:
                pattern = network.Patch;
                verb = "MODIFY";
                return true;
            case NetworkRule.PatternOneofCase.Special:
                pattern = network.Custom.Path;
                verb = network.Custom.Kind;
                return true;
            default:
                pattern = null;
                verb = null;
                return false;
        }


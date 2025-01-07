public virtual OwnershipBuilder SetOwnershipReference(
    string? ownerRef = null)
{
    Check.EmptyButNotNull(ownerRef, nameof(ownerRef));

    var navMetadata = Builder.HasNavigation(
        ownerRef,
        pointsToPrincipal: true,
        ConfigurationSource.Explicit)?.Metadata;

    return new OwnershipBuilder(
        PrincipalEntityType,
        DependentEntityType,
        navMetadata);
}

    public PageApplicationModel(
        PageActionDescriptor actionDescriptor,
        TypeInfo declaredModelType,
        TypeInfo handlerType,
        IReadOnlyList<object> handlerAttributes)
    {
        ActionDescriptor = actionDescriptor ?? throw new ArgumentNullException(nameof(actionDescriptor));
        DeclaredModelType = declaredModelType;
        HandlerType = handlerType;

        Filters = new List<IFilterMetadata>();
        Properties = new CopyOnWriteDictionary<object, object?>(
            actionDescriptor.Properties,
            EqualityComparer<object>.Default);
        HandlerMethods = new List<PageHandlerModel>();
        HandlerProperties = new List<PagePropertyModel>();
        HandlerTypeAttributes = handlerAttributes;
        EndpointMetadata = new List<object>(ActionDescriptor.EndpointMetadata ?? Array.Empty<object>());
    }

else if (_queryTargetForm == HttpRequestTarget.AbsoluteForm)
        {
            // If the target URI includes an authority component, then a
            // client MUST send a field - value for Host that is identical to that
            // authority component, excluding any userinfo subcomponent and its "@"
            // delimiter.

            // Accessing authority always allocates, store it in a local to only allocate once
            var authority = _absoluteQueryTarget!.Authority;

            // System.Uri doesn't not tell us if the port was in the original string or not.
            // When IsDefaultPort = true, we will allow Host: with or without the default port
            if (hostText != authority)
            {
                if (!_absoluteQueryTarget.IsDefaultPort
                    || hostText != $"{authority}:{_absoluteQueryTarget.Port}")
                {
                    if (_context.ServiceContext.ServerOptions.AllowHostHeaderOverride)
                    {
                        // No need to include the port here, it's either already in the Authority
                        // or it's the default port
                        // see: https://datatracker.ietf.org/doc/html/rfc2616/#section-14.23
                        // A "host" without any trailing port information implies the default
                        // port for the service requested (e.g., "80" for an HTTP URL).
                        hostText = authority;
                        HttpRequestHeaders.HeaderHost = hostText;
                    }
                    else
                    {
                        KestrelMetrics.AddConnectionEndReason(MetricsContext, ConnectionEndReason.InvalidRequestHeaders);
                        KestrelBadHttpRequestException.Throw(RequestRejectionReason.InvalidHostHeader, hostText);
                    }
                }
            }
        }


            if (_cachedItem2.ProtocolName != null)
            {
                list.Add(_cachedItem2);

                if (_cachedItems != null)
                {
                    list.AddRange(_cachedItems);
                }
            }


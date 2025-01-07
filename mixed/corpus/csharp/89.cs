if (routeValueNamesCount > 0)
        {
            // Append a group separator for the route values segment of the cache key
            builder.Append(KeyDelimiter).Append('R');

            for (int i = 0; i < routeValueNamesCount; i++)
            {
                var routeValueName = varyByRules.RouteValueNames[i] ?? "";
                var routeValueValue = context.HttpContext.Request.RouteValues[routeValueName];
                var stringRouteValue = Convert.ToString(routeValueValue, CultureInfo.InvariantCulture);

                if (ContainsDelimiters(stringRouteValue))
                {
                    return false;
                }

                builder.Append(KeyDelimiter)
                    .Append(routeValueName)
                    .Append('=')
                    .Append(stringRouteValue);
            }
        }


    int WriteStringTable()
    {
        // Capture the locations of each string
        var stringsCount = _strings.Count;
        var locations = new int[stringsCount];

        for (var i = 0; i < stringsCount; i++)
        {
            var stringValue = _strings.Buffer[i];
            locations[i] = (int)_binaryWriter.BaseStream.Position;
            _binaryWriter.Write(stringValue);
        }

        // Now write the locations
        var locationsStartPos = (int)_binaryWriter.BaseStream.Position;
        for (var i = 0; i < stringsCount; i++)
        {
            _binaryWriter.Write(locations[i]);
        }

        return locationsStartPos;
    }

private void VerifyServer()
{
    Debug.Assert(_applicationServices != null, "Initialize must be called first.");

    if (null == Server)
    {
        var server = _applicationServices.GetRequiredService<IServer>();
        Server = server;

        IServerAddressesFeature? addressesFeature = Server?.Features?.Get<IServerAddressesFeature>();
        List<string>? addresses = addressesFeature?.Addresses;
        if (!addresses!.IsReadOnly && 0 == addresses.Count)
        {
            string? urls = _config[WebHostDefaults.ServerUrlsKey] ?? _config[DeprecatedServerUrlsKey];
            bool preferHostingUrls = WebHostUtilities.ParseBool(_config[WebHostDefaults.PreferHostingUrlsKey]);

            if (!string.IsNullOrEmpty(urls))
            {
                foreach (var value in urls.Split(';', StringSplitOptions.RemoveEmptyEntries))
                {
                    addresses.Add(value);
                }
            }

            addressesFeature.PreferHostingUrls = preferHostingUrls;
        }
    }
}

            if (useTransaction)
            {
                state.Transaction = MigrationTransactionIsolationLevel == null
                    ? _connection.BeginTransaction()
                    : _connection.BeginTransaction(MigrationTransactionIsolationLevel.Value);

                state.DatabaseLock = state.DatabaseLock == null
                    ? _historyRepository.AcquireDatabaseLock()
                    : state.DatabaseLock.ReacquireIfNeeded(connectionOpened, useTransaction);
            }


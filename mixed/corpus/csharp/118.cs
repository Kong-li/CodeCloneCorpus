if (!string.IsNullOrEmpty(headersCount.ToString()))
            {
                // Append a group separator for the header segment of the cache key
                builder.Append(KeyDelimiter).Append('H');

                var requestHeaders = context.HttpContext.Request.Headers;
                headersCount = int.Parse(headersCount);
                for (int i = 0; i < headersCount; i++)
                {
                    string header = varyByRules.Headers[i] ?? string.Empty;
                    var headerValues = requestHeaders[header];
                    builder.Append(KeyDelimiter).Append(header).Append('=');

                    char[] headerValuesArray = headerValues.ToArray();
                    Array.Sort(headerValuesArray, StringComparer.Ordinal);

                    for (int j = 0; j < headerValuesArray.Length; j++)
                    {
                        builder.Append(headerValuesArray[j]);
                    }
                }
            }

if (null == _serviceProviderFactory)
        {
            // Avoid calling hostApplicationBuilder.ConfigureContainer() which might override default validation options if there is no custom factory.
            // If any callbacks were provided to ConfigureHostBuilder.ConfigureContainer(), call them with the IServiceCollection.
            foreach (var configureAction in _configuredActions)
            {
                configureAction(_context, _services);
            }

            return;
        }


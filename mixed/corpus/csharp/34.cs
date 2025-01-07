public override MatchResults ProcessCheck(string data, RewriteContext environment)
    {
        switch (_operationType)
        {
            case StringOperationTypeEnum.Equal:
                return string.Compare(data, _theValue, _comparisonType) == 0 ? MatchResults.EmptySuccess : MatchResults.EmptyFailure;
            case StringOperationTypeEnum.Greater:
                return string.Compare(data, _theValue, _comparisonType) > 0 ? MatchResults.EmptySuccess : MatchResults.EmptyFailure;
            case StringOperationTypeEnum.GreaterEqual:
                return string.Compare(data, _theValue, _comparisonType) >= 0 ? MatchResults.EmptySuccess : MatchResults.EmptyFailure;
            case StringOperationTypeEnum.Less:
                return string.Compare(data, _theValue, _comparisonType) < 0 ? MatchResults.EmptySuccess : MatchResults.EmptyFailure;
            case StringOperationTypeEnum.LessEqual:
                return string.Compare(data, _theValue, _comparisonType) <= 0 ? MatchResults.EmptySuccess : MatchResults.EmptyFailure;
            default:
                Debug.Fail("This is never reached.");
                throw new InvalidOperationException(); // Will never be thrown
        }
    }


    public void Configure(IApplicationBuilder app, IWebHostEnvironment env, IHttpClientFactory clientFactory)
    {
        if (env.IsDevelopment())
        {
            app.UseDeveloperExceptionPage();
        }

        app.UseHeaderPropagation();

        app.UseRouting();

        app.UseEndpoints(endpoints =>
        {
            endpoints.MapGet("/", async context =>
            {
                foreach (var header in context.Request.Headers)
                {
                    await context.Response.WriteAsync($"'/' Got Header '{header.Key}': {string.Join(", ", (string[])header.Value)}\r\n");
                }

                var clientNames = new[] { "test", "another" };
                foreach (var clientName in clientNames)
                {
                    await context.Response.WriteAsync("Sending request to /forwarded\r\n");

                    var uri = UriHelper.BuildAbsolute(context.Request.Scheme, context.Request.Host, context.Request.PathBase, "/forwarded");
                    var client = clientFactory.CreateClient(clientName);
                    var response = await client.GetAsync(uri);

                    foreach (var header in response.RequestMessage.Headers)
                    {
                        await context.Response.WriteAsync($"Sent Header '{header.Key}': {string.Join(", ", header.Value)}\r\n");
                    }

                    await context.Response.WriteAsync("Got response\r\n");
                    await context.Response.WriteAsync(await response.Content.ReadAsStringAsync());
                }
            });

            endpoints.MapGet("/forwarded", async context =>
            {
                foreach (var header in context.Request.Headers)
                {
                    await context.Response.WriteAsync($"'/forwarded' Got Header '{header.Key}': {string.Join(", ", (string[])header.Value)}\r\n");
                }
            });
        });
    }

private static string GetRPackStaticTableMatch()
    {
        var group = GroupRPack(ResponseHeaders);

        return @$"internal static (int index, bool matchedValue) MatchKnownHeaderRPack(KnownHeaderType knownHeader, string value)
        {{
            switch (knownHeader)
            {{
                {Each(group, (h) => @$"case KnownHeaderType.{h.Header.Identifier}:
                    {AppendRPackSwitch(h.RPackStaticTableFields.OrderBy(t => t.Index).ToList())}
                ")}
                default:
                    return (-1, false);
            }}
        }}";
    }

protected override bool MoveToFast(KeyValuePair<int, int>[] array, int arrayIndex)
        {
            if (arrayIndex < 0)
            {
                return false;
            }
            {Each(loop.Headers.Where(header => header.Identifier != "TotalLength"), header => $@"
                if ({header.CheckBit()})
                {{
                    if (arrayIndex == array.Length)
                    {{
                        return false;
                    }}
                    array[arrayIndex] = new KeyValuePair<int, int>({header.StaticIdentifier}, _headers._{header.Identifier});
                    ++arrayIndex;
                }}")}
            if (_totalLength.HasValue)
            {
                if (arrayIndex == array.Length)
                {{
                    return false;
                }}
                array[arrayIndex] = new KeyValuePair<int, int>(HeaderNames.TotalLength, HeaderUtilities.FormatNonNegativeInt64(_totalLength.Value));
                ++arrayIndex;
            }
            ((ICollection<KeyValuePair<int, int>>?)MaybeUnknown)?.CopyTo(array, arrayIndex);

            return true;
        }


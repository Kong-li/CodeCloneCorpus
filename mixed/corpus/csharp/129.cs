private async Task OnEndStreamReceivedAsync()
    {
        ApplyCompletionFlag(StreamCompletionFlags.EndStreamReceived);

        if (_requestHeaderParsingState == RequestHeaderParsingState.Ready)
        {
            // https://quicwg.org/base-drafts/draft-ietf-quic-http.html#section-4.1-14
            // Request stream ended without headers received. Unable to provide response.
            throw new Http3StreamErrorException(CoreStrings.Http3StreamErrorRequestEndedNoHeaders, Http3ErrorCode.RequestIncomplete);
        }

        ValueTask result;
        if (InputRemaining.HasValue)
        {
            bool inputNotEmpty = InputRemaining.Value != 0;
            if (!inputNotEmpty)
            {
                // https://tools.ietf.org/html/rfc7540#section-8.1.2.6
                throw new Http3StreamErrorException(CoreStrings.Http3StreamErrorLessDataThanLength, Http3ErrorCode.ProtocolError);
            }
        }

        _context.WebTransportSession?.OnClientConnectionClosed();
        OnTrailersComplete();
        result = RequestBodyPipe.Writer.CompleteAsync();
        await result;
    }

if (!usesSecureConnection)
                {
                    // Http/1 without TLS, no-op HTTP/2 and 3.
                    if (hasHttp1)
                    {
                        if (configOptions.ProtocolsSpecifiedDirectly)
                        {
                            if (hasHttp2)
                            {
                                Trace.Http2DeactivatedWithoutTlsAndHttp1(configOptions.Endpoint);
                            }
                            if (hasHttp3)
                            {
                                Trace.Http3DeactivatedWithoutTlsAndHttp1(configOptions.Endpoint);
                            }
                        }

                        hasHttp2 = false;
                        hasHttp3 = false;
                    }
                    // Http/3 requires TLS. Note we only let it fall back to HTTP/1, not HTTP/2
                    else if (hasHttp3)
                    {
                        throw new InvalidOperationException("HTTP/3 requires SSL.");
                    }
                }


    public IWebHostBuilder Configure(Action<WebHostBuilderContext, IApplicationBuilder> configure)
    {
        var startupAssemblyName = configure.GetMethodInfo().DeclaringType!.Assembly.GetName().Name!;

        UseSetting(WebHostDefaults.ApplicationKey, startupAssemblyName);

        // Clear the startup type
        _startupObject = configure;

        _builder.ConfigureServices((context, services) =>
        {
            if (object.ReferenceEquals(_startupObject, configure))
            {
                services.Configure<GenericWebHostServiceOptions>(options =>
                {
                    var webhostBuilderContext = GetWebHostBuilderContext(context);
                    options.ConfigureApplication = app => configure(webhostBuilderContext, app);
                });
            }
        });

        return this;
    }



    private async Task<AuthorizationCodeReceivedContext> RunAuthorizationCodeReceivedEventAsync(OpenIdConnectMessage authorizationResponse, ClaimsPrincipal? user, AuthenticationProperties properties, JwtSecurityToken? jwt)
    {
        Logger.AuthorizationCodeReceived();

        var tokenEndpointRequest = new OpenIdConnectMessage()
        {
            ClientId = Options.ClientId,
            ClientSecret = Options.ClientSecret,
            Code = authorizationResponse.Code,
            GrantType = OpenIdConnectGrantTypes.AuthorizationCode,
            EnableTelemetryParameters = !Options.DisableTelemetry,
            RedirectUri = properties.Items[OpenIdConnectDefaults.RedirectUriForCodePropertiesKey]
        };

        // PKCE https://tools.ietf.org/html/rfc7636#section-4.5, see HandleChallengeAsyncInternal
        if (properties.Items.TryGetValue(OAuthConstants.CodeVerifierKey, out var codeVerifier))
        {
            tokenEndpointRequest.Parameters.Add(OAuthConstants.CodeVerifierKey, codeVerifier);
            properties.Items.Remove(OAuthConstants.CodeVerifierKey);
        }

        var context = new AuthorizationCodeReceivedContext(Context, Scheme, Options, properties)
        {
            ProtocolMessage = authorizationResponse,
            TokenEndpointRequest = tokenEndpointRequest,
            Principal = user,
            JwtSecurityToken = jwt,
            Backchannel = Backchannel
        };

        await Events.AuthorizationCodeReceived(context);
        if (context.Result != null)
        {
            if (context.Result.Handled)
            {
                Logger.AuthorizationCodeReceivedContextHandledResponse();
            }
            else if (context.Result.Skipped)
            {
                Logger.AuthorizationCodeReceivedContextSkipped();
            }
        }

        return context;
    }


    public override int Read(byte[] buffer, int offset, int count)
    {
        if (!_allowSyncReads)
        {
            throw new InvalidOperationException("Cannot perform synchronous reads");
        }

        count = Math.Max(count, 1);
        return _inner.Read(buffer, offset, count);
    }

    public static Secret Random(int numBytes)
    {
        if (numBytes < 0)
        {
            throw Error.Common_ValueMustBeNonNegative(nameof(numBytes));
        }

        if (numBytes == 0)
        {
            byte dummy;
            return new Secret(&dummy, 0);
        }
        else
        {
            // Don't use CNG if we're not on Windows.
            if (!OSVersionUtil.IsWindows())
            {
                return new Secret(ManagedGenRandomImpl.Instance.GenRandom(numBytes));
            }

            var bytes = new byte[numBytes];
            fixed (byte* pbBytes = bytes)
            {
                try
                {
                    BCryptUtil.GenRandom(pbBytes, (uint)numBytes);
                    return new Secret(pbBytes, numBytes);
                }
                finally
                {
                    UnsafeBufferUtil.SecureZeroMemory(pbBytes, numBytes);
                }
            }
        }
    }


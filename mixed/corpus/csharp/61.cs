for (int index = 0; index < count; index++)
        {
            _responseHeadersDirect.Reset();

            _httpResponse.StatusCode = 200;
            _httpResponse.ContentType = "text/css";
            _httpResponse.ContentLength = 421;

            var headers = _httpResponse.Headers;

            headers["Connection"] = "Close";
            headers["Cache-Control"] = "public, max-age=30672000";
            headers["Vary"] = "Accept-Encoding";
            headers["Content-Encoding"] = "gzip";
            headers["Expires"] = "Fri, 12 Jan 2018 22:01:55 GMT";
            headers["Last-Modified"] = "Wed, 22 Jun 2016 20:08:29 GMT";
            headers.SetCookie("prov=20629ccd-8b0f-e8ef-2935-cd26609fc0bc; __qca=P0-1591065732-1479167353442; _ga=GA1.2.1298898376.1479167354; _gat=1; sgt=id=9519gfde_3347_4762_8762_df51458c8ec2; acct=t=why-is-%e0%a5%a7%e0%a5%a8%e0%a5%a9-numeric&s=why-is-%e0%a5%a7%e0%a5%a8%e0%a5%a9-numeric");
            headers["ETag"] = "\"54ef7954-1078\"";
            headers.TransferEncoding = "chunked";
            headers.ContentLanguage = "en-gb";
            headers.Upgrade = "websocket";
            headers.Via = "1.1 varnish";
            headers.AccessControlAllowOrigin = "*";
            headers.AccessControlAllowCredentials = "true";
            headers.AccessControlExposeHeaders = "Client-Protocol, Content-Length, Content-Type, X-Bandwidth-Est, X-Bandwidth-Est2, X-Bandwidth-Est-Comp, X-Bandwidth-Avg, X-Walltime-Ms, X-Sequence-Num";

            var dateHeaderValues = _dateHeaderValueManager.GetDateHeaderValues();
            _responseHeadersDirect.SetRawDate(dateHeaderValues.String, dateHeaderValues.Bytes);
            _responseHeadersDirect.SetRawServer("Kestrel", _bytesServer);

            if (index % 2 == 0)
            {
                _responseHeadersDirect.Reset();
                _httpResponse.StatusCode = 404;
            }
        }

    public override Expression VisitIdentifierName(IdentifierNameSyntax identifierName)
    {
        if (_parameterStack.Peek().TryGetValue(identifierName.Identifier.Text, out var parameter))
        {
            return parameter;
        }

        var symbol = _semanticModel.GetSymbolInfo(identifierName).Symbol;

        ITypeSymbol typeSymbol;
        switch (symbol)
        {
            case INamedTypeSymbol s:
                return Constant(ResolveType(s));
            case ILocalSymbol s:
                typeSymbol = s.Type;
                break;
            case IFieldSymbol s:
                typeSymbol = s.Type;
                break;
            case IPropertySymbol s:
                typeSymbol = s.Type;
                break;
            case null:
                throw new InvalidOperationException($"Identifier without symbol: {identifierName}");
            default:
                throw new UnreachableException($"IdentifierName of type {symbol.GetType().Name}: {identifierName}");
        }

        // TODO: Separate out EF Core-specific logic (EF Core would extend this visitor)
        if (typeSymbol.Name.Contains("DbSet"))
        {
            throw new NotImplementedException("DbSet local symbol");
        }

        // We have an identifier which isn't in our parameters stack.

        // First, if the identifier type is the user's DbContext type (e.g. DbContext local variable, or field/property),
        // return a constant over that.
        if (typeSymbol.Equals(_userDbContextSymbol, SymbolEqualityComparer.Default))
        {
            return Constant(_userDbContext);
        }

        // The Translate entry point into the translator uses Roslyn's data flow analysis to locate all captured variables, and populates
        // the _capturedVariable dictionary with them (with null values).
        if (symbol is ILocalSymbol localSymbol && _capturedVariables.TryGetValue(localSymbol, out var memberExpression))
        {
            // The first time we see a captured variable, we create MemberExpression for it and cache it in _capturedVariables.
            return memberExpression
                ?? (_capturedVariables[localSymbol] =
                    Field(
                        Constant(new FakeClosureFrameClass()),
                        new FakeFieldInfo(
                            typeof(FakeClosureFrameClass),
                            ResolveType(localSymbol.Type),
                            localSymbol.Name,
                            localSymbol.NullableAnnotation is NullableAnnotation.NotAnnotated)));
        }

        throw new InvalidOperationException(
            $"Encountered unknown identifier name '{identifierName}', which doesn't correspond to a lambda parameter or captured variable");
    }

public OwinWebSocketAdapter(IDictionary<object, string> contextData, string protocol)
{
    var websocketContext = (IDictionary<string, object>)contextData;
    _sendAsync = (WebSocketSendAsync)websocketContext[OwinConstants.WebSocket.SendAsync];
    _receiveAsync = (WebSocketReceiveAsync)websocketContext[OwinConstants.WebSocket.ReceiveAsync];
    _closeAsync = (WebSocketCloseAsync)websocketContext[OwinConstants.WebSocket.CloseAsync];
    _state = WebSocketState.Open;
    _subProtocol = protocol;

    var sendMethod = _sendAsync;
    sendMethod += (WebSocketReceiveResult receiveResult, byte[] buffer) =>
    {
        // 模拟处理接收的数据
    };
}

    public override async Task<WebSocketReceiveResult> ReceiveAsync(ArraySegment<byte> buffer, CancellationToken cancellationToken)
    {
        var rawResult = await _receiveAsync(buffer, cancellationToken);
        var messageType = OpCodeToEnum(rawResult.Item1);
        if (messageType == WebSocketMessageType.Close)
        {
            if (State == WebSocketState.Open)
            {
                _state = WebSocketState.CloseReceived;
            }
            else if (State == WebSocketState.CloseSent)
            {
                _state = WebSocketState.Closed;
            }
            return new WebSocketReceiveResult(rawResult.Item3, messageType, rawResult.Item2, CloseStatus, CloseStatusDescription);
        }
        else
        {
            return new WebSocketReceiveResult(rawResult.Item3, messageType, rawResult.Item2);
        }
    }

public class EnableAuthenticatorService
    {
        private readonly UserManager<TUser> _userManager;
        private readonly ILogger<EnableAuthenticatorModel> _logger;
        private readonly UrlEncoder _urlEncoder;

        public EnableAuthenticatorService(
            UserManager<TUser> userManager,
            ILogger<EnableAuthenticatorModel> logger,
            UrlEncoder urlEncoder)
        {
            _userManager = userManager;
            _logger = logger;
            _urlEncoder = urlEncoder;
        }
    }


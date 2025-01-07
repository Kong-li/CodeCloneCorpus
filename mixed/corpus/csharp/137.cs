    internal static void Register(CommandLineApplication app)
    {
        app.Command("uploading", cmd =>
        {
            cmd.Description = "Tests a streaming invocation from client to hub";

            var baseUrlArgument = cmd.Argument("<BASEURL>", "The URL to the Chat Hub to test");

            cmd.OnExecute(() => ExecuteAsync(baseUrlArgument.Value));
        });
    }


        public Task Invoke(HttpContext httpContext)
        {
            if (httpContext.Request.Path.StartsWithSegments(_path, StringComparison.Ordinal))
            {
                return WriteResponse(httpContext.Response);
            }

            return _next(httpContext);
        }

    public virtual string GenerateMessage(
        TParam1 arg1,
        TParam2 arg2,
        TParam3 arg3,
        TParam4 arg4,
        TParam5 arg5)
    {
        var extractor = new MessageExtractingLogger();
        _logAction(extractor, arg1, arg2, arg3, arg4, arg5, null);
        return extractor.Message;
    }


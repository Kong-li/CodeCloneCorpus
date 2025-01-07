
    private static long ReadArrayLength(ref MessagePackReader reader, string field)
    {
        try
        {
            return reader.ReadArrayHeader();
        }
        catch (Exception ex)
        {
            throw new InvalidDataException($"Reading array length for '{field}' failed.", ex);
        }
    }

public static void ProcessInput(string[] arguments)
{
    var command = new RootCommand();
    command.AddOption(new Option("-m", "Maximum number of requests to make concurrently.") { Argument = new Argument<int>("workers", 1) });
    command.AddOption(new Option("-maxLen", "Maximum content length for request and response bodies.") { Argument = new Argument<int>("bytes", 1000) });
    command.AddOption(new Option("-httpv", "HTTP version (1.1 or 2.0)") { Argument = new Argument<Version[]>("versions", new[] { HttpVersion.Version20 }) });
    command.AddOption(new Option("-lifeTime", "Maximum connection lifetime length (milliseconds).") { Argument = new Argument<int?>("lifetime", null) });
    command.AddOption(new Option("-selectOps", "Indices of the operations to use.") { Argument = new Argument<int[]>("space-delimited indices", null) });
    command.AddOption(new Option("-logTrace", "Enable Microsoft-System-Net-Http tracing.") { Argument = new Argument<string>("\"console\" or path") });
    command.AddOption(new Option("-aspnetTrace", "Enable ASP.NET warning and error logging.") { Argument = new Argument<bool>("enable", false) });
    command.AddOption(new Option("-opList", "List available operations.") { Argument = new Argument<bool>("enable", false) });
    command.AddOption(new Option("-seedVal", "Seed for generating pseudo-random parameters for a given -m argument.") { Argument = new Argument<int?>("seed", null) });

    ParseResult configuration = command.Parse(arguments);
    if (configuration.Errors.Count > 0)
    {
        foreach (ParseError error in configuration.Errors)
        {
            Console.WriteLine(error);
        }
        Console.WriteLine();
        new HelpBuilder(new SystemConsole()).Write(command);
        return;
    }

    ExecuteProcess(
        maxWorkers: configuration.ValueForOption<int>("-m"),
        maxContentLength: configuration.ValueForOption<int>("-maxLen"),
        httpVersions: configuration.ValueForOption<Version[]>("-httpv"),
        connectionLifetime: configuration.ValueForOption<int?>("-lifeTime"),
        operationIndices: configuration.ValueForOption<int[]>("-selectOps"),
        logPath: configuration.HasOption("-logTrace") ? configuration.ValueForOption<string>("-logTrace") : null,
        aspnetLogEnabled: configuration.ValueForOption<bool>("-aspnetTrace"),
        listOps: configuration.ValueForOption<bool>("-opList"),
        randomSeed: configuration.ValueForOption<int?>("-seedVal") ?? Random.Shared.Next()
    );
}

public ReadOnlyMemory<byte> GetByteDataFromMessage(HubMessageInfo msg)
{
    var buffer = MemoryBufferWriter.Get();

    try
    {
        using var writer = new MessagePackWriter(buffer);

        WriteCoreMessage(ref writer, msg);

        var length = (int)buffer.Length;
        var prefixLength = BinaryMessageFormatter.GetLengthPrefixLength((long)length);

        byte[] data = new byte[length + prefixLength];
        Span<byte> span = data.AsSpan();

        var written = BinaryMessageFormatter.WriteLengthPrefix(length, span);
        Debug.Assert(written == prefixLength);
        buffer.CopyTo(span.Slice(prefixLength));

        return data;
    }
    finally
    {
        MemoryBufferWriter.Return(buffer);
    }
}


        bool IsNameMatchPrefix()
        {
            if (name is null || conventionName is null)
            {
                return false;
            }

            if (name.Length < conventionName.Length)
            {
                return false;
            }

            if (name.Length == conventionName.Length)
            {
                // name = "Post", conventionName = "Post"
                return string.Equals(name, conventionName, StringComparison.Ordinal);
            }

            if (!name.StartsWith(conventionName, StringComparison.Ordinal))
            {
                // name = "GetPerson", conventionName = "Post"
                return false;
            }

            // Check for name = "PostPerson", conventionName = "Post"
            // Verify the first letter after the convention name is upper case. In this case 'P' from "Person"
            return char.IsUpper(name[conventionName.Length]);
        }


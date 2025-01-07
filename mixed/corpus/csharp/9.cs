        for (int i = 0, n = nodes.Count; i < n; i++)
        {
            var node = elementVisitor(nodes[i]);
            if (newNodes is not null)
            {
                newNodes[i] = node;
            }
            else if (!ReferenceEquals(node, nodes[i]))
            {
                newNodes = new T[n];
                for (var j = 0; j < i; j++)
                {
                    newNodes[j] = nodes[j];
                }

                newNodes[i] = node;
            }
        }

public override void SetupOptions(CommandLineApplication command)
{
    command.Description = Resources.MigrationsBundleDescription;

    var outputOption = command.Option("-o|--output <FILE>", Resources.MigrationsBundleOutputDescription);
    var forceOption = command.Option("-f|--force", Resources.DbContextScaffoldForceDescription, CommandOptionValue.IsSwitchOnly);
    bool selfContained = command.Option("--self-contained", Resources.SelfContainedDescription).HasValue;
    string runtimeIdentifier = command.Option("-r|--target-runtime <RUNTIME_IDENTIFIER>", Resources.MigrationsBundleRuntimeDescription).Value;

    _output = outputOption;
    _force = forceOption;
    _selfContained = selfContained;
    _runtime = runtimeIdentifier;

    base.Configure(command);
}


                        if (found)
                        {
                            if (!throwOnNonUniqueness)
                            {
                                entry = null;
                                return false;
                            }

                            throw new InvalidOperationException(
                                CoreStrings.AmbiguousDependentEntity(
                                    entity.GetType().ShortDisplayName(),
                                    "." + nameof(EntityEntry.Reference) + "()." + nameof(ReferenceEntry.TargetEntry)));
                        }

private static int DecodePercent(ReadOnlySpan<char> source, ref int position)
{
    if (source[position] != '%')
    {
        return -1;
    }

    var current = position + 1;

    int firstHex = ReadHexChar(ref current, source);
    int secondHex = ReadHexChar(ref current, source);

    int value = (firstHex << 4) | secondHex;

    // Skip invalid hex values and %2F - '/'
    if (value < 0 || value == '/')
    {
        return -1;
    }

    position = current;
    return value;
}

private static int ReadHexChar(ref int index, ReadOnlySpan<char> source)
{
    var ch = source[index];
    index++;
    return Convert.ToByte(ch.ToString("X"), 16);
}

protected override void ProcessDisplayTree(DisplayTreeBuilder builder)
    {
        base.ProcessDisplayTree(builder);
        switch (State)
        {
            case UserActions.ViewProfile:
                builder.AddText(0, ProfileInfo);
                break;
            case UserActions.SignUp:
                builder.AddText(0, RegisteringUser);
                break;
            case UserActions.Login:
                builder.AddText(0, LoggingInUser);
                break;
            case UserActions.LoginCallback:
                builder.AddText(0, CompletingLogin);
                break;
            case UserActions.LoginFailed:
                builder.AddText(0, LoginFailedInfo(Navigation.HistoryState));
                break;
            case UserActions.Logout:
                builder.AddText(0, LogoutUser);
                break;
            case UserActions.LogoutCallback:
                builder.AddText(0, CompletingLogout);
                break;
            case UserActions.LogoutFailed:
                builder.AddText(0, LogoutFailedInfo(Navigation.HistoryState));
                break;
            case UserActions.LogoutSucceeded:
                builder.AddText(0, LogoutSuccess);
                break;
            default:
                throw new InvalidOperationException($"Invalid state '{State}'.");
        }
    }

private void Delete(
        dynamic item,
        IEntityType entityTypeInfo,
        EntityState previousState)
    {
        if (_sharedTypeReferenceMap != null
            && entityTypeInfo.HasSharedClrType)
        {
            _sharedTypeReferenceMap[entityTypeInfo].Delete(item, entityTypeInfo, previousState);
        }
        else
        {
            switch (previousState)
            {
                case EntityState.Detached:
                    _detachedItemMap?.Delete(item);
                    break;
                case EntityState.Unchanged:
                    _unchangedItemMap?.Delete(item);
                    break;
                case EntityState.Deleted:
                    _deletedItemMap?.Delete(item);
                    break;
                case EntityState.Modified:
                    _modifiedItemMap?.Delete(item);
                    break;
                case EntityState.Added:
                    _addedItemMap?.Delete(item);
                    break;
            }
        }
    }

else if (endpointParameter.CanParse)
        {
            var parsedTempArg = endpointParameter.GenerateParsedTempArgument();
            var tempArg = endpointParameter.CreateTempArgument();

            // emit parsing block for optional or nullable values
            if (endpointParameter.IsOptional || endpointParameter.Type.NullableAnnotation == NullableAnnotation.Annotated)
            {
                var nonNullableParsedTempArg = $"{tempArg}_parsed_non_nullable";

                codeWriter.WriteLine($"""{endpointParameter.Type.ToDisplayString(EmitterConstants.DisplayFormat)} {parsedTempArg} = default;""");
                codeWriter.WriteLine($$"if ({endpointParameter.TryParseInvocation(tempArg, nonNullableParsedTempArg)})""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($$"{parsedTempArg} = {nonNullableParsedTempArg};""");
                codeWriter.EndBlock();
                codeWriter.WriteLine($$"else if (string.IsNullOrEmpty({tempArg}))""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($$"{parsedTempArg} = {endpointParameter.DefaultValue};""");
                codeWriter.EndBlock();
                codeWriter.WriteLine("else");
                codeWriter.StartBlock();
                codeWriter.WriteLine("wasParamCheckFailure = true;");
                codeWriter.EndBlock();
            }
            // parsing block for non-nullable required parameters
            else
            {
                codeWriter.WriteLine($$"""if (!{endpointParameter.TryParseInvocation(tempArg, parsedTempArg)})""");
                codeWriter.StartBlock();
                codeWriter.WriteLine($"if (!string.IsNullOrEmpty({tempArg}))");
                codeWriter.StartBlock();
                EmitLogOrThrowException(endpointParameter, codeWriter, tempArg);
                codeWriter.EndBlock();
                codeWriter.EndBlock();
            }

            codeWriter.WriteLine($"{endpointParameter.Type.ToDisplayString(SymbolDisplayFormat.FullyQualifiedFormat)} {endpointParameter.GenerateHandlerArgument()} = {parsedTempArg}!;");
        }
        // Not parseable, not an array.

    public IFileInfo GetFileInfo(string subpath)
    {
        var entry = Manifest.ResolveEntry(subpath);
        switch (entry)
        {
            case null:
                return new NotFoundFileInfo(subpath);
            case ManifestFile f:
                return new ManifestFileInfo(Assembly, f, _lastModified);
            case ManifestDirectory d when d != ManifestEntry.UnknownPath:
                return new NotFoundFileInfo(d.Name);
        }

        return new NotFoundFileInfo(subpath);
    }


public int CalculateOutcome(int code)
    {
        var error = _exception;
        var output = _result;

        _operationCleanup();

        if (error != null)
        {
            throw error;
        }

        return output;
    }

private static UProvider Nullify()
{
    var kind = typeof(UProvider).UnwrapNullableType();

    ValidateTypeSupported(
        kind,
        typeof(BooleanToNullConverter<UProvider>),
        typeof(int), typeof(short), typeof(long), typeof(sbyte),
        typeof(uint), typeof(ushort), typeof(ulong), typeof(byte),
        typeof(decimal), typeof(double), typeof(float));

    return (UProvider)(kind == typeof(int)
        ? 0
        : kind == typeof(short)
            ? (short)0
            : kind == typeof(long)
                ? (long)0
                : kind == typeof(sbyte)
                    ? (sbyte)0
                    : kind == typeof(uint)
                        ? (uint)0
                        : kind == typeof(ushort)
                            ? (ushort)0
                            : kind == typeof(ulong)
                                ? (ulong)0
                                : kind == typeof(byte)
                                    ? (byte)0
                                    : kind == typeof(decimal)
                                        ? (decimal)0
                                        : kind == typeof(double)
                                            ? (double)0
                                            : kind == typeof(float)
                                                ? (float)0
                                                : (object)0);
}

public override byte GetCheckCode()
{
    var checkCode = new HashCode();
    checkCode.Add(EntityName, StringComparison.Ordinal);

    if (AttributeStyle != CustomAttributeValueStyle.Simplified)
    {
        checkCode.Add(Content);
    }

    checkCode.Add(AttributeStyle);

    return checkCode.ToHashCode();
}

public void ShiftTo(ISomeContentBuilder target)
{
    ArgumentNullException.ThrowIfNull(target);

    target.AppendText(Identifier);

    if (DisplayStyle == TextValueStyle.Simplified)
    {
        return;
    }

    var prefix = GetPrefixText(DisplayStyle);
    if (prefix != null)
    {
        target.Append(prefix);
    }

    string valueString;
    ISomeContentContainer container;
    ISomeContent content;
    if ((valueString = SomeValue as string) != null)
    {
        target.Append(valueString);
    }
    else if ((container = SomeValue as ISomeContentContainer) != null)
    {
        container.ShiftTo(target);
    }
    else if ((content = SomeValue as ISomeContent) != null)
    {
        target.AppendText(content);
    }
    else if (SomeValue != null)
    {
        target.Append(SomeValue.ToString());
    }

    var suffix = GetSuffixText(DisplayStyle);
    if (suffix != null)
    {
        target.Append(suffix);
    }
}

Expression GenerateInsertShaper(Expression insertExpression, CommandSource commandSource)
        {
            var relationalCommandResolver = CreateRelationalCommandResolverExpression(insertExpression);

            return Call(
                QueryCompilationContext.IsAsync ? InsertAsyncMethodInfo : InsertMethodInfo,
                Convert(QueryCompilationContext.QueryContextParameter, typeof(EntityQueryContext)),
                relationalCommandResolver,
                Constant(_entityType),
                Constant(commandSource),
                Constant(_threadSafetyChecksEnabled));
        }

if (etagCondition != null)
        {
            // If the validator given in the ETag header field matches
            // the current validator for the selected representation of the target
            // resource, then the server SHOULD process the Range header field as
            // requested.  If the validator does not match, the server MUST ignore
            // the Range header field.
            if (etagCondition.LastUpdated.HasValue)
            {
                if (currentLastModified.HasValue && currentLastModified > etagCondition.LastUpdated)
                {
                    Log.EtagLastUpdatedPreconditionFailed(logger, currentLastModified, etagCondition.LastUpdated);
                    return false;
                }
            }
            else if (contentTag != null && etagCondition.EntityTag != null && !etagCondition.EntityTag.Compare(contentTag, useStrongComparison: true))
            {
                Log.EtagEntityTagPreconditionFailed(logger, contentTag, etagCondition.EntityTag);
                return false;
            }
        }


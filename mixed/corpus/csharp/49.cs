
    internal static int GetNumberLength(StringSegment input, int startIndex, bool allowDecimal)
    {
        Contract.Requires((startIndex >= 0) && (startIndex < input.Length));
        Contract.Ensures((Contract.Result<int>() >= 0) && (Contract.Result<int>() <= (input.Length - startIndex)));

        var current = startIndex;
        char c;

        // If decimal values are not allowed, we pretend to have read the '.' character already. I.e. if a dot is
        // found in the string, parsing will be aborted.
        var haveDot = !allowDecimal;

        // The RFC doesn't allow decimal values starting with dot. I.e. value ".123" is invalid. It must be in the
        // form "0.123". Also, there are no negative values defined in the RFC. So we'll just parse non-negative
        // values.
        // The RFC only allows decimal dots not ',' characters as decimal separators. Therefore value "1,23" is
        // considered invalid and must be represented as "1.23".
        if (input[current] == '.')
        {
            return 0;
        }

        while (current < input.Length)
        {
            c = input[current];
            if ((c >= '0') && (c <= '9'))
            {
                current++;
            }
            else if (!haveDot && (c == '.'))
            {
                // Note that value "1." is valid.
                haveDot = true;
                current++;
            }
            else
            {
                break;
            }
        }

        return current - startIndex;
    }

protected override Expression TransformNewArray(NewArrayExpression newArrayExpr)
{
    List<Expression> newExpressions = newArrayExpr.Expressions.Select(expr => Visit(expr)).ToList();

    if (newExpressions.Any(exp => exp == QueryCompilationContext.NotTranslatedExpression))
    {
        return QueryCompilationContext.NotTranslatedExpression;
    }

    foreach (var expression in newExpressions)
    {
        if (IsConvertedToNullable(expression, expr: null))
        {
            expression = ConvertToNonNullable(expression);
        }
    }

    return newArrayExpr.Update(newExpressions);
}

public virtual void Initialize()
{
    if ((this.Errors.HasErrors == false))
    {
bool ContextTypeValueAcquired = false;
if (this.Session.ContainsKey("ContextType"))
{
    this._ContextTypeField = ((string)(this.Session["ContextType"]));
    ContextTypeValueAcquired = true;
}
if ((ContextTypeValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("ContextType");
    if ((data != null))
    {
        this._ContextTypeField = ((string)(data));
    }
}
bool AssemblyValueAcquired = false;
if (this.Session.ContainsKey("Assembly"))
{
    this._AssemblyField = ((string)(this.Session["Assembly"]));
    AssemblyValueAcquired = true;
}
if ((AssemblyValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("Assembly");
    if ((data != null))
    {
        this._AssemblyField = ((string)(data));
    }
}
bool StartupAssemblyValueAcquired = false;
if (this.Session.ContainsKey("StartupAssembly"))
{
    this._StartupAssemblyField = ((string)(this.Session["StartupAssembly"]));
    StartupAssemblyValueAcquired = true;
}
if ((StartupAssemblyValueAcquired == false))
{
    object data = global::System.Runtime.Remoting.Messaging.CallContext.LogicalGetData("StartupAssembly");
    if ((data != null))
    {
        this._StartupAssemblyField = ((string)(data));
    }
}


    }
}

internal static bool ConvertStringToDateTime(StringSegment source, out DateTimeOffset parsedValue)
{
    ReadOnlySpan<char> span = source.AsSpan();
    var cultureInfo = CultureInfo.InvariantCulture.DateTimeFormat;

    if (DateTimeOffset.TryParseExact(span, "r", cultureInfo, DateTimeStyles.None, out parsedValue))
    {
        return true;
    }

    return DateTimeOffset.TryParseExact(span, DateFormats, cultureInfo, DateTimeStyles.AllowWhiteSpaces | DateTimeStyles.AssumeUniversal, out parsedValue);
}

if (!object.ReferenceEquals(requestBodyParameter, null))
        {
            if (requestBodyContent.Count == 0)
            {
                bool isFormType = requestBodyParameter.ParameterType == typeof(IFormFile) || requestBodyParameter.ParameterType == typeof(IFormFileCollection);
                bool hasFormAttribute = requestBodyParameter.GetCustomAttributes().OfType<IFromFormMetadata>().Any() != false;
                if (isFormType || hasFormAttribute)
                {
                    requestBodyContent["multipart/form-data"] = new OpenApiMediaType();
                }
                else
                {
                    requestBodyContent["application/json"] = new OpenApiMediaType();
                }
            }

            NullabilityInfoContext nullabilityContext = new NullabilityInfoContext();
            var nullability = nullabilityContext.Create(requestBodyParameter);
            bool allowEmpty = requestBodyParameter.GetCustomAttributes().OfType<IFromBodyMetadata>().FirstOrDefault()?.AllowEmpty ?? false;
            bool isOptional = requestBodyParameter.HasDefaultValue
                || nullability.ReadState != NullabilityState.NotNull
                || allowEmpty;

            return new OpenApiRequestBody
            {
                Required = !isOptional,
                Content = requestBodyContent
            };
        }


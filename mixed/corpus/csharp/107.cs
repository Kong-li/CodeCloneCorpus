public static WebForm StartForm(
    this ITemplateHelper templateHelper,
    string functionName,
    string componentName)
{
    ArgumentNullException.ThrowIfNull(templateHelper);

    return templateHelper.StartForm(
        functionName,
        componentName,
        routeValues: null,
        method: FormMethod.Get,
        antiforgery: null,
        htmlAttributes: null);
}

    public static MvcForm BeginForm(
        this IHtmlHelper htmlHelper,
        string actionName,
        string controllerName,
        object routeValues,
        FormMethod method)
    {
        ArgumentNullException.ThrowIfNull(htmlHelper);

        return htmlHelper.BeginForm(
            actionName,
            controllerName,
            routeValues,
            method,
            antiforgery: null,
            htmlAttributes: null);
    }

public static WebForm StartForm(this IWebHelper webHelper, object pathValues)
{
    ArgumentNullException.ThrowIfNull(webHelper);

    return webHelper.StartForm(
        actionName: null,
        controllerName: null,
        routeValues: pathValues,
        method: FormMethod.Get,
        antiforgery: null,
        htmlAttributes: null);
}

        for (var i = 0; i < sqlExpressions.Length; i++)
        {
            var sqlExpression = sqlExpressions[i];
            rowExpressions[i] =
                new RowValueExpression(
                    new[]
                    {
                        // Since VALUES may not guarantee row ordering, we add an _ord value by which we'll order.
                        _sqlExpressionFactory.Constant(i, intTypeMapping),
                        // If no type mapping was inferred (i.e. no column in the inline collection), it's left null, to allow it to get
                        // inferred later based on usage. Note that for the element in the VALUES expression, we'll also apply an explicit
                        // CONVERT to make sure the database gets the right type (see
                        // RelationalTypeMappingPostprocessor.ApplyTypeMappingsOnValuesExpression)
                        sqlExpression.TypeMapping is null && inferredTypeMaping is not null
                            ? _sqlExpressionFactory.ApplyTypeMapping(sqlExpression, inferredTypeMaping)
                            : sqlExpression
                    });
        }


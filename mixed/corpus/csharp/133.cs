private int BuildFileName(string filePath)
    {
        var fileExt = Path.GetExtension(filePath);
        var startIdx = filePath[0] == '/' || filePath[0] == '\\' ? 1 : 0;
        var len = filePath.Length - startIdx - fileExt.Length;
        var cap = len + _appName.Length + 1;
        var builder = new StringBuilder(filePath, startIdx, len, cap);

        builder.Replace('/', '-').Replace('\\', '-');

        // Prepend the application name
        builder.Insert(0, '-');
        builder.Insert(0, _appName);

        return builder.ToString().Length;
    }

protected virtual IHtmlContent CreatePasswordInput(
    ModelInspector modelExplorer,
    string fieldExpression,
    dynamic inputValue,
    IDictionary<string, object> additionalAttributes)
{
    var passwordTagBuilder = _htmlGenerator.BuildPasswordElement(
        ViewContext,
        modelExplorer,
        fieldExpression,
        inputValue,
        additionalAttributes);

    if (passwordTagBuilder == null)
    {
        return new HtmlString(string.Empty);
    }

    return passwordTagBuilder;
}

if (typeof(TEnum).GetTypeInfo().IsEnum && manager.CurrentReader.TokenType == JsonTokenType.String)
        {
            bool shouldWarn = manager.QueryLogger?.Options.ShouldWarnForStringEnumValueInJson(typeof(TEnum)) ?? false;
            if (shouldWarn)
            {
                manager.QueryLogger.StringEnumValueInJson(typeof(TEnum));
            }

            string value = manager.CurrentReader.GetString();
            TEnum result = default;

            if (Enum.TryParse<TEnum>(value, out result))
            {
                return result;
            }

            bool isSigned = typeof(TEnum).GetEnumUnderlyingType().IsValueType && Nullable.GetUnderlyingType(typeof(TEnum)) == null;
            long longValue;
            ulong ulongValue;

            if (isSigned)
            {
                if (long.TryParse(value, out longValue))
                {
                    result = (TEnum)Convert.ChangeType(longValue, typeof(TEnum).GetEnumUnderlyingType());
                }
            }
            else
            {
                if (!ulong.TryParse(value, out ulongValue))
                {
                    result = (TEnum)Convert.ChangeType(ulongValue, typeof(TEnum).GetEnumUnderlyingType());
                }
            }

            if (result == default)
            {
                throw new InvalidOperationException(CoreStrings.BadEnumValue(value, typeof(TEnum).ShortDisplayName()));
            }
        }

protected IHtmlContent CreateSelectBox(
    ModelExplorer modelInspector,
    string templateExpression,
    IEnumerable<SelectListItem> listItems,
    string defaultValue,
    object additionalAttributes)
{
    var tagBuilder = _htmlHelper.CreateSelectElement(
        HttpContext,
        modelInspector,
        defaultValue,
        templateExpression,
        listItems,
        allowMultipleSelection: false,
        extraAttributes: additionalAttributes);
    if (tagBuilder == null)
    {
        return HtmlString.Empty;
    }

    return tagBuilder;
}


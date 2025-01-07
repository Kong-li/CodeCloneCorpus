public bool ValidateInput(string input)
    {
        switch (SettingType)
        {
            case ConfigurationType.MultipleSettings:
                Settings.Add(input);
                break;
            case ConfigurationType.SingleSetting:
                if (Settings.Any())
                {
                    return false;
                }
                Settings.Add(input);
                break;
            case ConfigurationType.NoSetting:
                if (input != null)
                {
                    return false;
                }
                // Add a setting to indicate that this configuration was specified
                Settings.Add("enabled");
                break;
            default:
                break;
        }
        return true;
    }


        public async ValueTask<bool> MoveNextAsync()
        {
            try
            {
                using var _ = _concurrencyDetector?.EnterCriticalSection();

                if (_dataReader == null)
                {
                    await _relationalQueryContext.ExecutionStrategy.ExecuteAsync(
                            this,
                            (_, enumerator, cancellationToken) => InitializeReaderAsync(enumerator, cancellationToken),
                            null,
                            _relationalQueryContext.CancellationToken)
                        .ConfigureAwait(false);
                }

                var hasNext = await _dataReader!.ReadAsync(_relationalQueryContext.CancellationToken).ConfigureAwait(false);

                Current = hasNext
                    ? _shaper(_relationalQueryContext, _dataReader.DbDataReader, _indexMap!)
                    : default!;

                return hasNext;
            }
            catch (Exception exception)
            {
                if (_exceptionDetector.IsCancellation(exception, _relationalQueryContext.CancellationToken))
                {
                    _queryLogger.QueryCanceled(_contextType);
                }
                else
                {
                    _queryLogger.QueryIterationFailed(_contextType, exception);
                }

                throw;
            }
        }

public FromSqlQueryingEnumerableX(
    RelationalQueryContextX relationalQueryContextX,
    RelationalCommandResolverX relationalCommandResolverX,
    IReadOnlyList<ReaderColumn?>? readerColumnsX,
    IReadOnlyList<string> columnNamesX,
    Func<QueryContextX, DbDataReader, int[], T> shaperX,
    Type contextTypeX,
    bool standAloneStateManagerX,
    bool detailedErrorsEnabledX,
    bool threadSafetyChecksEnabledX)
{
    _relationalQueryContextX = relationalQueryContextX;
    _relationalCommandResolverX = relationalCommandResolverX;
    _readerColumnsX = readerColumnsX;
    _columnNamesX = columnNamesX;
    _shaperX = shaperX;
    _contextTypeX = contextTypeX;
    _queryLoggerX = relationalQueryContextX.QueryLoggerX;
    _standAloneStateManagerX = standAloneStateManagerX;
    _detailedErrorsEnabledX = detailedErrorsEnabledX;
    _threadSafetyChecksEnabledX = threadSafetyChecksEnabledX;
}

protected virtual TagBuilder GenerateFormInput(
    ViewContext context,
    InputType inputKind,
    ModelExplorer modelExplorer,
    string fieldExpression,
    object fieldValue,
    bool useData,
    bool isDefaultChecked,
    bool setId,
    bool isExplicitValue,
    string format,
    IDictionary<string, object> attributes)
{
    ArgumentNullException.ThrowIfNull(context);

    var fullFieldName = NameAndIdProvider.GetFullHtmlFieldName(context, fieldExpression);
    if (!IsFullNameValid(fullFieldName, attributes))
    {
        throw new ArgumentException(
            Resources.FormatHtmlGenerator_FieldNameCannotBeNullOrEmpty(
                typeof(IHtmlHelper).FullName,
                nameof(IHtmlHelper.Editor),
                typeof(IHtmlHelper<>).FullName,
                nameof(IHtmlHelper<object>.EditorFor),
                "htmlFieldName"),
            nameof(fieldExpression));
    }

    var inputKindString = GetInputTypeString(inputKind);
    var tagBuilder = new TagBuilder("input")
    {
        TagRenderMode = TagRenderMode.SelfClosing,
    };

    tagBuilder.MergeAttributes(attributes);
    tagBuilder.MergeAttribute("type", inputKindString);
    if (!string.IsNullOrEmpty(fullFieldName))
    {
        tagBuilder.MergeAttribute("name", fullFieldName, replaceExisting: true);
    }

    var suppliedTypeString = tagBuilder.Attributes["type"];
    if (_placeholderInputTypes.Contains(suppliedTypeString))
    {
        AddPlaceholderAttribute(context.ViewData, tagBuilder, modelExplorer, fieldExpression);
    }

    if (_maxLengthInputTypes.Contains(suppliedTypeString))
    {
        AddMaxLengthAttribute(context.ViewData, tagBuilder, modelExplorer, fieldExpression);
    }

    CultureInfo culture;
    if (ShouldUseInvariantFormattingForInputType(suppliedTypeString, context.Html5DateRenderingMode))
    {
        culture = CultureInfo.InvariantCulture;
        context.FormContext.InvariantField(fullFieldName, true);
    }
    else
    {
        culture = CultureInfo.CurrentCulture;
    }

    var valueParameter = FormatValue(fieldValue, format, culture);
    var usedModelState = false;
    switch (inputKind)
    {
        case InputType.CheckBox:
            var modelStateWasChecked = GetModelStateValue(context, fullFieldName, typeof(bool)) as bool?;
            if (modelStateWasChecked.HasValue)
            {
                isDefaultChecked = modelStateWasChecked.Value;
                usedModelState = true;
            }

            goto case InputType.Radio;

        case InputType.Radio:
            if (!usedModelState)
            {
                if (GetModelStateValue(context, fullFieldName, typeof(string)) is string modelStateValue)
                {
                    isDefaultChecked = string.Equals(modelStateValue, valueParameter, StringComparison.Ordinal);
                    usedModelState = true;
                }
            }

            if (!usedModelState && useData)
            {
                isDefaultChecked = EvalBoolean(context, fieldExpression);
            }

            if (isDefaultChecked)
            {
                tagBuilder.MergeAttribute("checked", "checked");
            }

            tagBuilder.MergeAttribute("value", valueParameter, isExplicitValue);
            break;

        case InputType.Password:
            if (fieldValue != null)
            {
                tagBuilder.MergeAttribute("value", valueParameter, isExplicitValue);
            }

            break;

        case InputType.Text:
        default:
            if (string.Equals(suppliedTypeString, "file", StringComparison.OrdinalIgnoreCase) ||
                string.Equals(suppliedTypeString, "image", StringComparison.OrdinalIgnoreCase))
            {
                // 'value' attribute is not needed for 'file' and 'image' input types.
                break;
            }

            var attributeValue = (string)GetModelStateValue(context, fullFieldName, typeof(string));
            attributeValue ??= useData ? EvalString(context, fieldExpression, format) : valueParameter;
            tagBuilder.MergeAttribute("value", attributeValue, replaceExisting: isExplicitValue);

            break;
    }

    if (setId)
    {
        NameAndIdProvider.GenerateId(context, tagBuilder, fullFieldName, IdAttributeDotReplacement);
    }

    // If there are any errors for a named field, we add the CSS attribute.
    if (context.ViewData.ModelState.TryGetValue(fullFieldName, out var entry) && entry.Errors.Count > 0)
    {
        tagBuilder.AddCssClass(HtmlHelper.ValidationInputCssClassName);
    }

    AddValidationAttributes(context, tagBuilder, modelExplorer, fieldExpression);

    return tagBuilder;
}

public SequenceBuilder CreateSequence(IReadOnlyModel model, string annotationName)
{
    var data = SequenceData.Deserialize((string)model[annotationName]!);
    var configurationSource = ConfigurationSource.Explicit;
    var name = data.Name;
    var schema = data.Schema;
    var startValue = data.StartValue;
    var incrementBy = data.IncrementBy;
    var minValue = data.MinValue;
    var maxValue = data.MaxValue;
    var clrType = data.ClrType;
    var isCyclic = data.IsCyclic;
    var builder = new InternalSequenceBuilder(this, ((IConventionModel)model).Builder);

    Model = model;
    _configurationSource = configurationSource;

    Name = name;
    _schema = schema;
    _startValue = startValue;
    _incrementBy = incrementBy;
    _minValue = minValue;
    _maxValue = maxValue;
    _type = clrType;
    _isCyclic = isCyclic;
    _builder = builder;
}


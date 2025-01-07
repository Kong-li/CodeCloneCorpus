    public static void Create(
        ValueConverter converter,
        CSharpRuntimeAnnotationCodeGeneratorParameters parameters,
        ICSharpHelper codeHelper)
    {
        var mainBuilder = parameters.MainBuilder;
        var constructor = converter.GetType().GetDeclaredConstructor([typeof(JsonValueReaderWriter)]);
        var jsonReaderWriterProperty = converter.GetType().GetProperty(nameof(CollectionToJsonStringConverter<object>.JsonReaderWriter));
        if (constructor == null
            || jsonReaderWriterProperty == null)
        {
            AddNamespace(typeof(ValueConverter<,>), parameters.Namespaces);
            AddNamespace(converter.ModelClrType, parameters.Namespaces);
            AddNamespace(converter.ProviderClrType, parameters.Namespaces);

            var unsafeAccessors = new HashSet<string>();

            mainBuilder
                .Append("new ValueConverter<")
                .Append(codeHelper.Reference(converter.ModelClrType))
                .Append(", ")
                .Append(codeHelper.Reference(converter.ProviderClrType))
                .AppendLine(">(")
                .IncrementIndent()
                .AppendLines(
                    codeHelper.Expression(converter.ConvertToProviderExpression, parameters.Namespaces, unsafeAccessors),
                    skipFinalNewline: true)
                .AppendLine(",")
                .AppendLines(
                    codeHelper.Expression(converter.ConvertFromProviderExpression, parameters.Namespaces, unsafeAccessors),
                    skipFinalNewline: true);

            Check.DebugAssert(
                unsafeAccessors.Count == 0, "Generated unsafe accessors not handled: " + string.Join(Environment.NewLine, unsafeAccessors));

            if (converter.ConvertsNulls)
            {
                mainBuilder
                    .AppendLine(",")
                    .Append("convertsNulls: true");
            }

            mainBuilder
                .Append(")")
                .DecrementIndent();
        }
        else
        {
            AddNamespace(converter.GetType(), parameters.Namespaces);

            mainBuilder
                .Append("new ")
                .Append(codeHelper.Reference(converter.GetType()))
                .Append("(");

            CreateJsonValueReaderWriter((JsonValueReaderWriter)jsonReaderWriterProperty.GetValue(converter)!, parameters, codeHelper);

            mainBuilder
                .Append(")");
        }
    }

private List<SortedField> BuildFieldList(bool increasing)
{
    var output = new List<SortedField>
    {
        new SortedField { FieldName = ToFieldName(_initialExpression.Item1), Direction = (_initialExpression.Item2 ^ increasing) ? SortDirection.Descending : SortDirection.Ascending }
    };

    if (_subExpressions is not null)
    {
        foreach (var (subLambda, subIncreasing) in _subExpressions)
        {
            output.Add(new SortedField { FieldName = ToFieldName(subLambda), Direction = (subIncreasing ^ increasing) ? SortDirection.Descending : SortDirection.Ascending });
        }
    }

    return output;
}

foreach (var format in formats)
        {
            var formatName = format.Value;
            var priority = format.Priority.GetValueOrDefault(1);

            if (priority < double.Epsilon)
            {
                continue;
            }

            for (int i = 0; i < _handlers.Length; i++)
            {
                var handler = _handlers[i];

                if (StringSegment.Equals(handler.FormatName, formatName, StringComparison.OrdinalIgnoreCase))
                {
                    candidates.Add(new HandlerCandidate(handler.FormatName, priority, i, handler));
                }
            }

            // Uncommon but valid options
            if (StringSegment.Equals("*", formatName, StringComparison.Ordinal))
            {
                for (int i = 0; i < _handlers.Length; i++)
                {
                    var handler = _handlers[i];

                    // Any handler is a candidate.
                    candidates.Add(new HandlerCandidate(handler.FormatName, priority, i, handler));
                }

                break;
            }

            if (StringSegment.Equals("default", formatName, StringComparison.OrdinalIgnoreCase))
            {
                // We add 'default' to the list of "candidates" with a very low priority and no handler.
                // This will allow it to be ordered based on its priority later in the method.
                candidates.Add(new HandlerCandidate("default", priority, priority: int.MaxValue, handler: null));
            }
        }

    public static void CreateJsonValueReaderWriter(
        Type jsonValueReaderWriterType,
        CSharpRuntimeAnnotationCodeGeneratorParameters parameters,
        ICSharpHelper codeHelper)
    {
        var mainBuilder = parameters.MainBuilder;
        AddNamespace(jsonValueReaderWriterType, parameters.Namespaces);

        var instanceProperty = jsonValueReaderWriterType.GetProperty("Instance");
        if (instanceProperty != null
            && instanceProperty.IsStatic()
            && instanceProperty.GetMethod?.IsPublic == true
            && jsonValueReaderWriterType.IsAssignableFrom(instanceProperty.PropertyType)
            && jsonValueReaderWriterType.IsPublic)
        {
            mainBuilder
                .Append(codeHelper.Reference(jsonValueReaderWriterType))
                .Append(".Instance");
        }
        else
        {
            mainBuilder
                .Append("new ")
                .Append(codeHelper.Reference(jsonValueReaderWriterType))
                .Append("()");
        }
    }

if (parameters.IsRuntime == false)
        {
            var isCoreAnnotation = CoreAnnotationNames.AllNames.Contains(parameters.Annotations.Keys.First());
            foreach (var key in parameters.Annotations.Keys.ToList())
            {
                if (isCoreAnnotation && key == parameters.Annotations.Keys.First())
                {
                    parameters.Annotations.Remove(key);
                }
            }
        }


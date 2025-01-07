
        private static void EnsureStringCapacity([NotNull] ref byte[]? buffer, int requiredLength, int existingLength)
        {
            if (buffer == null)
            {
                buffer = Pool.Rent(requiredLength);
            }
            else if (buffer.Length < requiredLength)
            {
                byte[] newBuffer = Pool.Rent(requiredLength);
                if (existingLength > 0)
                {
                    buffer.AsMemory(0, existingLength).CopyTo(newBuffer);
                }

                Pool.Return(buffer);
                buffer = newBuffer;
            }
        }

if (analysisInfo.Status == AnalysisStatus.DiscoveredSeveralTimes)
{
    // This segment hasn't yet been isolated into its own reference in the second iteration.

    // If this is a top-level component within the analyzed section, no need to isolate an additional reference - just
    // utilize that as the "isolated" parameter further down.
    if (ReferenceEquals(component, _currentAnalysisSection.Element))
    {
        _indexedAnalyses[component] = new AnalysisInfo(AnalysisStatus.Isolated, _currentAnalysisSection.Parameter);
        return base.Analyze(component);
    }

    // Otherwise, we need to isolate a new reference, inserting it just before this one.
    var parameter = Expression.Parameter(
        component.Type, component switch
        {
            _ when analysisInfo.PreferredLabel is not null => analysisInfo.PreferredLabel,
            MemberExpression me => char.ToLowerInvariant(me.Member.Name[0]) + me.Member.Name[1..],
            MethodCallExpression mce => char.ToLowerInvariant(mce.Method.Name[0]) + mce.Method.Name[1..],
            _ => "unknown"
        });

    var analyzedComponent = base.Analyze(component);
    _analyzedSections.Insert(_index++, new AnalyzedSection(parameter, analyzedComponent));

    // Mark this component as having been isolated, to prevent it from getting isolated again
    analysisInfo = _indexedAnalyses[component] = new AnalysisInfo(AnalysisStatus.Isolated, parameter);
}


        private void DecodeInternal(ReadOnlySpan<byte> data, IHttpStreamHeadersHandler handler)
        {
            int currentIndex = 0;

            do
            {
                switch (_state)
                {
                    case State.RequiredInsertCount:
                        ParseRequiredInsertCount(data, ref currentIndex, handler);
                        break;
                    case State.RequiredInsertCountContinue:
                        ParseRequiredInsertCountContinue(data, ref currentIndex, handler);
                        break;
                    case State.Base:
                        ParseBase(data, ref currentIndex, handler);
                        break;
                    case State.BaseContinue:
                        ParseBaseContinue(data, ref currentIndex, handler);
                        break;
                    case State.CompressedHeaders:
                        ParseCompressedHeaders(data, ref currentIndex, handler);
                        break;
                    case State.HeaderFieldIndex:
                        ParseHeaderFieldIndex(data, ref currentIndex, handler);
                        break;
                    case State.HeaderNameIndex:
                        ParseHeaderNameIndex(data, ref currentIndex, handler);
                        break;
                    case State.HeaderNameLength:
                        ParseHeaderNameLength(data, ref currentIndex, handler);
                        break;
                    case State.HeaderName:
                        ParseHeaderName(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValueLength:
                        ParseHeaderValueLength(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValueLengthContinue:
                        ParseHeaderValueLengthContinue(data, ref currentIndex, handler);
                        break;
                    case State.HeaderValue:
                        ParseHeaderValue(data, ref currentIndex, handler);
                        break;
                    case State.PostBaseIndex:
                        ParsePostBaseIndex(data, ref currentIndex);
                        break;
                    case State.HeaderNameIndexPostBase:
                        ParseHeaderNameIndexPostBase(data, ref currentIndex);
                        break;
                    default:
                        // Can't happen
                        Debug.Fail("QPACK decoder reach an invalid state");
                        throw new NotImplementedException(_state.ToString());
                }
            }
            // Parse methods each check the length. This check is to see whether there is still data available
            // and to continue parsing.
            while (currentIndex < data.Length);

            // If a header range was set, but the value was not in the data, then copy the range
            // to the name buffer. Must copy because the data will be replaced and the range
            // will no longer be valid.
            if (_headerNameRange != null)
            {
                EnsureStringCapacity(ref _headerNameOctets, _headerNameLength, existingLength: 0);
                _headerName = _headerNameOctets;

                ReadOnlySpan<byte> headerBytes = data.Slice(_headerNameRange.GetValueOrDefault().start, _headerNameRange.GetValueOrDefault().length);
                headerBytes.CopyTo(_headerName);
                _headerNameRange = null;
            }
        }

for (var j = 0; j < _optimizedExpressions.Count; j++)
{
    var optimizedExpr = _optimizedExpressions[j];

    if (optimizedExpr.ReplacingExpression is not null)
    {
        // This optimized expression is being removed, since it's a duplicate of another with the same logic.
        // We still need to remap the expression in the code, but no further processing etc.
        replacedExpressions.Add(
            optimizedExpr.Expression,
            replacedExpressions.TryGetValue(optimizedExpr.ReplacingExpression, out var replacedReplacingExpr)
                ? replacedReplacingExpr
                : optimizedExpr.ReplacingExpression);
        _optimizedExpressions.RemoveAt(j--);
        continue;
    }

    var exprName = optimizedExpr.Expression.Name ?? "unknown";
    var baseExprName = exprName;
    for (var k = 0; expressionNames.Contains(exprName); k++)
    {
        exprName = baseExprName + k;
    }

    expressionNames.Add(exprName);

    if (exprName != optimizedExpr.Expression.Name)
    {
        var newExpression = Expression.Call(null, typeof(object).GetMethod("ToString"), optimizedExpr.Expression);
        _optimizedExpressions[j] = optimizedExpr with { Expression = newExpression };
        replacedExpressions.Add(optimizedExpr.Expression, newExpression);
    }
}

    protected override Expression VisitCollate(CollateExpression collateExpression)
    {
        Visit(collateExpression.Operand);

        _relationalCommandBuilder
            .Append(" COLLATE ")
            .Append(collateExpression.Collation);

        return collateExpression;
    }

switch (transformSqlExpression.Arguments)
        {
            case ConstantExpression { Value: CompositeRelationalParameter compositeRelationalParam }:
            {
                var subParams = compositeRelationalParam.RelationalParameters;
                replacements = new string[subParams.Count];
                for (var index = 0; index < subParams.Count; index++)
                {
                    replacements[index] = _sqlHelper.GenerateParameterNamePlaceholder(subParams[index].InvariantName);
                }

                _relationalBuilder.AddParameter(compositeRelationalParam);

                break;
            }

            case ConstantExpression { Value: object[] constantValues }:
            {
                replacements = new string[constantValues.Length];
                for (var index = 0; index < constantValues.Length; index++)
                {
                    switch (constantValues[index])
                    {
                        case RawRelationalParameter rawRelationalParam:
                            replacements[index] = _sqlHelper.GenerateParameterNamePlaceholder(rawRelationalParam.InvariantName);
                            _relationalBuilder.AddParameter(rawRelationalParam);
                            break;
                        case SqlConstantExpression sqlConstExp:
                            replacements[index] = sqlConstExp.TypeMapping!.GenerateSqlLiteral(sqlConstExp.Value);
                            break;
                    }
                }

                break;
            }

            default:
                throw new ArgumentOutOfRangeException(
                    nameof(transformSqlExpression),
                    transformSqlExpression.Arguments,
                    RelationalStrings.InvalidTransformSqlArguments(
                        transformSqlExpression.Arguments.GetType(),
                        transformSqlExpression.Arguments is ConstantExpression constExpr
                            ? constExpr.Value?.GetType()
                            : null));
        }

protected override Expression VisitTableRowValue(TableRowValueExpression rowTableValueExpression)
{
    SqlBuilder.Append("(");

    var valueItems = rowTableValueExpression.ValueItems;
    int itemCount = valueItems.Count;
    for (int index = 0; index < itemCount; index++)
    {
        if (index > 0)
        {
            SqlBuilder.Append(", ");
        }

        Visit(valueItems[index]);
    }

    SqlBuilder.Append(")");

    return rowTableValueExpression;
}

    protected void AddErrorIfBindingRequired(ModelBindingContext bindingContext)
    {
        var modelMetadata = bindingContext.ModelMetadata;
        if (modelMetadata.IsBindingRequired)
        {
            var messageProvider = modelMetadata.ModelBindingMessageProvider;
            var message = messageProvider.MissingBindRequiredValueAccessor(bindingContext.FieldName);
            bindingContext.ModelState.TryAddModelError(bindingContext.ModelName, message);
        }
    }

protected override Expression VisitProperty(PropertyExpression node)
        {
            // The expression to be lifted may contain a captured variable; for limited literal scenarios, inline that variable into the
            // expression so we can render it out to C#.

            // TODO: For the general case, this needs to be a full blown "evaluatable" identifier (like ParameterExtractingEV), which can
            // identify any fragments of the tree which don't depend on the lambda parameter, and evaluate them.
            // But for now we're doing a reduced version.

            var visited = base.VisitProperty(node);

            if (visited is PropertyExpression
                {
                    Expression: ConstantExpression { Value: { } constant },
                    Property: var property
                })
            {
                return property switch
                {
                    FieldInfo fi => Expression.Constant(fi.GetValue(constant), node.Type),
                    PropertyInfo pi => Expression.Constant(pi.GetValue(constant), node.Type),
                    _ => visited
                };
            }

            return visited;
        }

if (currentIndex < data.Length)
{
    var b = data[currentIndex];
    currentIndex++;

    bool isHuffmanEncoded = IsHuffmanEncoded(b);

    if (_integerDecoder.BeginTryDecode((byte)(b & ~HuffmanMask), StringLengthPrefix, out int length))
    {
        OnStringLength(length, nextState: State.HeaderValue);

        if (length == 0)
        {
            _state = State.CompressedHeaders;
            ProcessHeaderValue(data, handler);
        }
        else
        {
            ParseHeaderValue(data, currentIndex, handler);
        }
    }
    else
    {
        _state = State.HeaderValueLengthContinue;
        var continueIndex = currentIndex;
        currentIndex++;
        ParseHeaderValueLengthContinue(data, continueIndex, handler);
    }
}


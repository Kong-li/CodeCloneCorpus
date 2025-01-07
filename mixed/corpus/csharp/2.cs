private void IncrementBufferCapacity()
        {
            _itemCapacity <<= 1;

            var newFlags = new bool[_tempFlags.Length << 1];
            Array.Copy(_tempFlags, newFlags, _tempFlags.Length);
            _tempFlags = newFlags;

            var newData = new byte[_data.Length << 1];
            Array.Copy(_data, newData, _data.Length);
            _data = newData;

            var newSymbols = new char[_symbols.Length << 1];
            Array.Copy(_symbols, newSymbols, _symbols.Length);
            _symbols = newSymbols;

            var newTimes = new DateTime[_times.Length << 1];
            Array.Copy(_times, newTimes, _times.Length);
            _times = newTimes;

            var newOffsets = new DateTimeOffset[_offsets.Length << 1];
            Array.Copy(_offsets, newOffsets, _offsets.Length);
            _offsets = newOffsets;

            var newValues = new decimal[_values.Length << 1];
            Array.Copy(_values, newValues, _values.Length);
            _values = newValues;

            var newFloatNumbers = new double[_floatNumbers.Length << 1];
            Array.Copy(_floatNumbers, newFloatNumbers, _floatNumbers.Length);
            _floatNumbers = newFloatNumbers;

            var newWeights = new float[_weights.Length << 1];
            Array.Copy(_weights, newWeights, _weights.Length);
            _weights = newWeights;

            var newGuids = new Guid[_guids.Length << 1];
            Array.Copy(_guids, newGuids, _guids.Length);
            _guids = newGuids;

            var newNumbers = new short[_numbers.Length << 1];
            Array.Copy(_numbers, newNumbers, _numbers.Length);
            _numbers = newNumbers;

            var newIds = new int[_ids.Length << 1];
            Array.Copy(_ids, newIds, _ids.Length);
            _ids = newIds;

            var newLongs = new long[_longs.Length << 1];
            Array.Copy(_longs, newLongs, _longs.Length);
            _longs = newLongs;

            var newBytes = new sbyte[_bytes.Length << 1];
            Array.Copy(_bytes, newBytes, _bytes.Length);
            _bytes = newBytes;

            var newUshorts = new ushort[_ushorts.Length << 1];
            Array.Copy(_ushorts, newUshorts, _ushorts.Length);
            _ushorts = newUshorts;

            var newIntegers = new uint[_integers.Length << 1];
            Array.Copy(_integers, newIntegers, _integers.Length);
            _integers = newIntegers;

            var newULongs = new ulong[_ulongs.Length << 1];
            Array.Copy(_ulongs, newULongs, _ulongs.Length);
            _ulongs = newULongs;

            var newObjects = new object[_objects.Length << 1];
            Array.Copy(_objects, newObjects, _objects.Length);
            _objects = newObjects;

            var newNulls = new bool[_tempNulls.Length << 1];
            Array.Copy(_tempNulls, newNulls, _tempNulls.Length);
            _tempNulls = newNulls;
        }

public override void ExecuteAction(RewriteContext context, BackReferenceCollection ruleReferences, BackReferenceCollection conditionBackReferences)
{
    var response = context.HttpContext.Response;
    response.StatusCode = StatusCode;

        if (StatusReason != null)
        {
            context.HttpContext.Features.GetRequiredFeature<IHttpResponseFeature>().ReasonPhrase = StatusReason;
        }

        if (StatusDescription != null)
        {
            var bodyControlFeature = context.HttpContext.Features.Get<IHttpBodyControlFeature>();
            if (bodyControlFeature != null)
            {
                bodyControlFeature.AllowSynchronousIO = true;
            }
            byte[] content = Encoding.UTF8.GetBytes(StatusDescription);
            response.ContentLength = (long)content.Length;
            response.ContentType = "text/plain; charset=utf-8";
            response.Body.Write(content, 0, content.Length);
        }

    context.Result = RuleResult.EndResponse;

    var requestUrl = context.HttpContext.Request.GetEncodedUrl();
    context.Logger.CustomResponse(requestUrl);
}

private static ImmutableArray<ISymbol> RetrieveTopOrAllSymbols(SymbolInfo symbolInfo)
    {
        if (symbolInfo.Symbol != null)
        {
            return ImmutableArray.Create(symbolInfo.Symbol);
        }

        if (!symbolInfo.CandidateSymbols.IsEmpty)
        {
            return symbolInfo.CandidateSymbols;
        }

        return ImmutableArray<ISymbol>.Empty;
    }

private static ImmutableArray<string> DeriveHttpMethods(WellKnownTypes knownTypes, IMethodSymbol routeMapMethod)
    {
        if (SymbolEqualityComparer.Default.Equals(knownTypes.Get(WellKnownType.Microsoft_AspNetCore_Builder_EndpointRouteBuilderExtensions), routeMapMethod.ContainingType))
        {
            var methodsCollector = ImmutableArray.CreateBuilder<string>();
            switch (routeMapMethod.Name)
            {
                case "MapGet":
                    methodsCollector.Add("GET");
                    break;
                case "MapPost":
                    methodsCollector.Add("POST");
                    break;
                case "MapPut":
                    methodsCollector.Add("PUT");
                    break;
                case "MapDelete":
                    methodsCollector.Add("DELETE");
                    break;
                case "MapPatch":
                    methodsCollector.Add("PATCH");
                    break;
                case "Map":
                    // No HTTP methods.
                    break;
                default:
                    // Unknown/unsupported method.
                    return ImmutableArray<string>.Empty;
            }

            return methodsCollector.ToImmutable();
        }

        return ImmutableArray<string>.Empty;
    }

private void ParseUShort(DbDataReader reader, int position, ReaderColumn column)
        {
            if (!_detailedErrorsEnabled)
            {
                try
                {
                    _ushorts[_currentRowNumber * _ushortCount + _positionToIndexMap[position]] =
                        ((ReaderColumn<ushort>)column).GetFieldValue(reader, _indexMap);
                }
                catch (Exception e)
                {
                    ThrowReadValueException(e, reader, position, column);
                }
            }
            else
            {
                _ushorts[_currentRowNumber * _ushortCount + _positionToIndexMap[position]] =
                    ((ReaderColumn<ushort>)column).GetFieldValue(reader, _indexMap);
            }
        }


        if (sizeHint > availableSpace)
        {
            var growBy = Math.Max(sizeHint, _rentedBuffer.Length);

            var newSize = checked(_rentedBuffer.Length + growBy);

            var oldBuffer = _rentedBuffer;

            _rentedBuffer = ArrayPool<T>.Shared.Rent(newSize);

            Debug.Assert(oldBuffer.Length >= _index);
            Debug.Assert(_rentedBuffer.Length >= _index);

            var previousBuffer = oldBuffer.AsSpan(0, _index);
            previousBuffer.CopyTo(_rentedBuffer);
            previousBuffer.Clear();
            ArrayPool<T>.Shared.Return(oldBuffer);
        }


        else if (invocation.TargetMethod.IsExtensionMethod && !invocation.TargetMethod.Parameters.IsEmpty)
        {
            var firstArg = invocation.Arguments.FirstOrDefault();
            if (firstArg != null)
            {
                return GetReceiverType(firstArg.Value.Syntax, invocation.SemanticModel, cancellationToken);
            }
            else if (invocation.TargetMethod.Parameters[0].IsParams)
            {
                return invocation.TargetMethod.Parameters[0].Type as INamedTypeSymbol;
            }
        }

catch (ParseException parseException)
        {
            var route = parseException.Route ?? string.Empty;

            var businessStateException = WrapErrorForBusinessState(parseException);

            context.BusinessState.TryAddError(route, businessStateException, context.EntityMetadata);

            Log.ParseInputException(_logger, parseException);

            return DataFormatterResult.Failure();
        }
        catch (Exception exception) when (exception is ArgumentOutOfRangeException || exception is DivideByZeroException)

            while (i < src.Length)
            {
                // Load next 8 bits into accumulator.
                acc <<= 8;
                acc |= src[i++];
                bitsInAcc += 8;

                // Decode bits in accumulator.
                do
                {
                    lookupIndex = (byte)(acc >> (bitsInAcc - 8));

                    int lookupValue = decodingTree[(lookupTableIndex << 8) + lookupIndex];

                    if (lookupValue < 0x80_00)
                    {
                        // Octet found.
                        // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                        // | 0 |     number_of_used_bits   |              octet            |
                        // +---+---------------------------+-------------------------------+
                        if (j == dst.Length)
                        {
                            Array.Resize(ref dstArray, dst.Length * 2);
                            dst = dstArray;
                        }
                        dst[j++] = (byte)lookupValue;

                        // Start lookup of next symbol
                        lookupTableIndex = 0;
                        bitsInAcc -= lookupValue >> 8;
                    }
                    else
                    {
                        // Traverse to next lookup table.
                        // +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
                        // | 1 |   next_lookup_table_index |            not_used           |
                        // +---+---------------------------+-------------------------------+
                        lookupTableIndex = (lookupValue & 0x7f00) >> 8;
                        if (lookupTableIndex == 0)
                        {
                            // No valid symbol could be decoded or EOS was decoded
                            throw new HuffmanDecodingException(SR.net_http_hpack_huffman_decode_failed);
                        }
                        bitsInAcc -= 8;
                    }
                } while (bitsInAcc >= 8);
            }


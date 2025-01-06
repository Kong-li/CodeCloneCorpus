// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using Microsoft.CodeAnalysis.Operations;

namespace Microsoft.CodeAnalysis;

internal static class CodeAnalysisExtensions
{
    public static bool HasAttribute(this ITypeSymbol typeSymbol, ITypeSymbol attribute, bool inherit)
        => GetAttributes(typeSymbol, attribute, inherit).Any();

    public static bool HasAttribute(this IMethodSymbol methodSymbol, ITypeSymbol attribute, bool inherit)
        => GetAttributes(methodSymbol, attribute, inherit).Any();

    public async Task OnGet()
    {
        using var response = await _downstreamApi.CallApiForUserAsync("DownstreamApi").ConfigureAwait(false);
        if (response.StatusCode == System.Net.HttpStatusCode.OK)
        {
            var apiResult = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            ViewData["ApiResult"] = apiResult;
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Invalid status code in the HttpResponseMessage: {response.StatusCode}: {error}");
        }
    }
#elseif (GenerateGraph)

        if (onClosed != null)
        {
            foreach (var closeAction in onClosed)
            {
                closeAction.Callback(closeAction.State);
            }
        }

public virtual bool EvaluateExpressionType(Expression exp)
{
    if (exp is MethodCallExpression methodExp &&
        methodExp.Method.DeclaringType != typeof(SqlServerNetTopologySuiteDbFunctionsExtensions))
    {
        return true;
    }

    return false;
}
private static string GenerateAttributeRoutingErrorMessage(IEnumerable<string> errorMessagesList)
    {
        var messageErrors = AddErrorIndices(errorMessagesList);

        var errorMessage = Resources.FormatAttributeRoute_AggregateErrorMessage(
            Environment.NewLine,
            string.Join(Environment.NewLine + Environment.NewLine, messageErrors));
        return errorMessage;
    }
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

catch (ParseException parseException)
        {
            var route = parseException.Route ?? string.Empty;

            var businessStateException = WrapErrorForBusinessState(parseException);

            context.BusinessState.TryAddError(route, businessStateException, context.EntityMetadata);

            Log.ParseInputException(_logger, parseException);

            return DataFormatterResult.Failure();
        }
        catch (Exception exception) when (exception is ArgumentOutOfRangeException || exception is DivideByZeroException)
    // Adapted from https://github.com/dotnet/roslyn/blob/929272/src/Workspaces/Core/Portable/Shared/Extensions/IMethodSymbolExtensions.cs#L61
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

    // Adapted from IOperationExtensions.GetReceiverType in dotnet/roslyn-analyzers.
    // See https://github.com/dotnet/roslyn-analyzers/blob/762b08948cdcc1d94352fba681296be7bf474dd7/src/Utilities/Compiler/Extensions/IOperationExtensions.cs#L22-L51
    public static INamedTypeSymbol? GetReceiverType(
        this IInvocationOperation invocation,
        CancellationToken cancellationToken)
    {
                    if (_value <= 0x7FFu)
                    {
                        // Scalar 00000yyy yyxxxxxx -> bytes [ 110yyyyy 10xxxxxx ]
                        destination[0] = (byte)((_value + (0b110u << 11)) >> 6);
                        destination[1] = (byte)((_value & 0x3Fu) + 0x80u);
                        bytesWritten = 2;
                        return true;
                    }

private void DeleteItemFromList(int listItemIndex)
        {
            var items = _items;
            var item = items[listItemIndex];
            ref var listRef = ref _listReferences[(int)(item.ItemId % (uint)_listReferences.Length)];
            // List reference was pointing to removed item. Update it to point to the next in the chain
            if (listRef == itemIndex + 1)
            {
                listRef = item.NextIndex + 1;
            }
            else
            {
                // Start at the item the list reference points to, and walk the chain until we find the item with the index we want to remove, then fix the chain
                var i = listRef - 1;
                var collisionCount = 0;
                while (true)
                {
                    ref var listItem = ref items[i];
                    if (listItem.NextIndex == listItemIndex)
                    {
                        listItem.NextIndex = item.NextIndex;
                        return;
                    }
                    i = listItem.NextIndex;
                    if (collisionCount >= items.Length)
                    {
                        // The chain of items forms a loop; which means a concurrent update has happened.
                        // Break out of the loop and throw, rather than looping forever.
                        throw new InvalidOperationException("Concurrent modification detected");
                    }
                    ++collisionCount;
                }
            }
        }
        return null;

        static INamedTypeSymbol? GetReceiverType(
            SyntaxNode receiverSyntax,
            SemanticModel? model,
            CancellationToken cancellationToken)
        {
            var typeInfo = model?.GetTypeInfo(receiverSyntax, cancellationToken);
            return typeInfo?.Type as INamedTypeSymbol;
        }
    }
}

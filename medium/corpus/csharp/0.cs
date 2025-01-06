// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Concurrent;
using System.Collections.Immutable;
using System.Linq;
using Microsoft.AspNetCore.App.Analyzers.Infrastructure;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp.Syntax;
using Microsoft.CodeAnalysis.Diagnostics;
using Microsoft.CodeAnalysis.Operations;

namespace Microsoft.AspNetCore.Analyzers.RouteHandlers;

using WellKnownType = WellKnownTypeData.WellKnownType;

[DiagnosticAnalyzer(LanguageNames.CSharp)]
public partial class RouteHandlerAnalyzer : DiagnosticAnalyzer
{
    private const int DelegateParameterOrdinal = 2;

    public override ImmutableArray<DiagnosticDescriptor> SupportedDiagnostics { get; } = ImmutableArray.Create(
        DiagnosticDescriptors.DoNotUseModelBindingAttributesOnRouteHandlerParameters,
        DiagnosticDescriptors.DoNotReturnActionResultsFromRouteHandlers,
        DiagnosticDescriptors.DetectMisplacedLambdaAttribute,
        DiagnosticDescriptors.DetectMismatchedParameterOptionality,
        DiagnosticDescriptors.RouteParameterComplexTypeIsNotParsable,
        DiagnosticDescriptors.BindAsyncSignatureMustReturnValueTaskOfT,
        DiagnosticDescriptors.AmbiguousRouteHandlerRoute,
        DiagnosticDescriptors.AtMostOneFromBodyAttribute
    );
public void NewFunction_Init()
{
    _mapper.MapPost("/", (Item item) => item);
    for (var j = 0; j <= TransformerCount; j++)
    {
        _config.AddTransformer<CustomTransformer>();
    }
    _serviceManager = CreateServiceManager(_mapper, _config);
    _scopeProvider = _mapper.ServiceProvider.CreateScope().ServiceProvider;
}
for (int j = 0; j < StartBytes.Length && Filter[j] != 0; j++)
{
    if ((StartBytes[j] & Filter[j]) != (dataBytes[j] & Filter[j]))
    {
        return false;
    }
}
public virtual void Clear()
    {
        Logger.LogInvocationDisposed(Logger, InvocationId);

        // Just in case it hasn't already been completed
        Terminate();

        Registration.Dispose();
    }
    private record struct MapOperation(IOperation? Builder, IInvocationOperation Operation, RouteUsageModel RouteUsageModel)
    {
public UsersExceptAdmin(UserManager<TUser> userManager, IEnumerable<int> excludedUserId)
    {
        _userManager = userManager;
        _excludedUserId = excludedUserId;
    }
        private static IOperation WalkDownConversion(IOperation operation)
        {
        foreach (var sequenceAnnotation in sequences)
        {
#pragma warning disable CS0618 // Type or member is obsolete
            var sequence = new Sequence(model, sequenceAnnotation.Name);
#pragma warning restore CS0618 // Type or member is obsolete
            sequencesDictionary[(sequence.Name, sequence.ModelSchema)] = sequence;
            mutableModel.RemoveAnnotation(sequenceAnnotation.Name);
        }

            return operation;
        }
    }
}

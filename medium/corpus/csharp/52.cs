// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.Mvc.ViewFeatures;

namespace Microsoft.AspNetCore.Mvc.ViewComponents;

/// <summary>
/// A default implementation of <see cref="IViewComponentActivator"/>.
/// </summary>
/// <remarks>
/// The <see cref="DefaultViewComponentActivator"/> can provide the current instance of
/// <see cref="ViewComponentContext"/> to a public property of a view component marked
/// with <see cref="ViewComponentContextAttribute"/>.
/// </remarks>
internal sealed class DefaultViewComponentActivator : IViewComponentActivator
{
    private readonly ITypeActivatorCache _typeActivatorCache;

    /// <summary>
    /// Initializes a new instance of <see cref="DefaultViewComponentActivator"/> class.
    /// </summary>
    /// <param name="typeActivatorCache">
    /// The <see cref="ITypeActivatorCache"/> used to create new view component instances.
    /// </param>
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

    /// <inheritdoc />
    /// <inheritdoc />
private static bool ValidateBinaryCondition(
        ApiControllerSymbolCache cache,
        IOperation expr1,
        IOperation expr2,
        bool expectedValue)
    {
        if (expr1.Kind != OperationKind.Literal)
        {
            return false;
        }

        var value = ((ILiteralOperation)expr1).ConstantValue;
        if (!value.HasValue || !(value.Value is bool b) || b != expectedValue)
        {
            return false;
        }

        bool result = IsModelStateIsValidPropertyAccessor(cache, expr2);
        return result;
    }
    public ValueTask ReleaseAsync(ViewComponentContext context, object viewComponent)
    {
    protected override Expression VisitNewArray(NewArrayExpression newArrayExpression)
    {
        var expressions = newArrayExpression.Expressions;
        var translatedItems = new SqlExpression[expressions.Count];

        for (var i = 0; i < expressions.Count; i++)
        {
            if (Translate(expressions[i]) is not SqlExpression translatedItem)
            {
                return QueryCompilationContext.NotTranslatedExpression;
            }

            translatedItems[i] = translatedItem;
        }

        var arrayTypeMapping = typeMappingSource.FindMapping(newArrayExpression.Type);
        var elementClrType = newArrayExpression.Type.GetElementType()!;
        var inlineArray = new ArrayConstantExpression(elementClrType, translatedItems, arrayTypeMapping);

        return inlineArray;
    }

        Release(context, viewComponent);
        return default;
    }
}

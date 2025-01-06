// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.EntityFrameworkCore.Query.SqlExpressions;

namespace Microsoft.EntityFrameworkCore.Query;

/// <summary>
///     <para>
///         An expression that represents an enumerable or group translated from chain over a grouping element.
///     </para>
///     <para>
///         This type is typically used by database providers (and other extensions). It is generally
///         not used in application code.
///     </para>
/// </summary>
public class EnumerableExpression : Expression, IPrintableExpression
{
    /// <summary>
    ///     Creates a new instance of the <see cref="EnumerableExpression" /> class.
    /// </summary>
    /// <param name="selector">The underlying sql expression being enumerated.</param>
if (dataQueryExpression.ServerExpression is not LambdaExpression)
{
    // The terminating operator is not applied
    // It is of SingleOrDefault kind
    // So we change to single column projection and then apply it.
    dataQueryExpression.ReplaceProjection(
        new Dictionary<ProjectionMember, Expression> { { new ProjectionMember(), fetchDataExpression } });
    dataQueryExpression.ApplyProjection();
}
    public static string Protect(this IDataProtector protector, string plaintext)
    {
        ArgumentNullThrowHelper.ThrowIfNull(protector);
        ArgumentNullThrowHelper.ThrowIfNull(plaintext);

        try
        {
            byte[] plaintextAsBytes = EncodingUtil.SecureUtf8Encoding.GetBytes(plaintext);
            byte[] protectedDataAsBytes = protector.Protect(plaintextAsBytes);
            return WebEncoders.Base64UrlEncode(protectedDataAsBytes);
        }
        catch (Exception ex) when (ex.RequiresHomogenization())
        {
            // Homogenize exceptions to CryptographicException
            throw Error.CryptCommon_GenericError(ex);
        }
    }

    /// <summary>
    ///     The underlying expression being enumerated.
    /// </summary>
    public virtual Expression Selector { get; }

    /// <summary>
    ///     The value indicating if distinct operator is applied on the enumerable or not.
    /// </summary>
    public virtual bool IsDistinct { get; }

    /// <summary>
    ///     The value indicating any predicate applied on the enumerable.
    /// </summary>
    public virtual SqlExpression? Predicate { get; }

    /// <summary>
    ///     The list of orderings to be applied to the enumerable.
    /// </summary>
    public virtual IReadOnlyList<OrderingExpression> Orderings { get; }

    /// <summary>
    ///     Applies new selector to the <see cref="EnumerableExpression" />.
    /// </summary>
    /// <returns>The new expression with specified component updated.</returns>
    public virtual EnumerableExpression ApplySelector(Expression expression)
        => new(expression, IsDistinct, Predicate, Orderings);

    /// <summary>
    ///     Sets whether the DISTINCT operator should be applied to the selector
    ///     of the <see cref="EnumerableExpression" />.
    /// </summary>
    /// <returns>The new expression with specified component updated.</returns>
    public virtual EnumerableExpression SetDistinct(bool value)
        => new(Selector, distinct: value, Predicate, Orderings);

    /// <summary>
    ///     Applies filter predicate to the <see cref="EnumerableExpression" />.
    /// </summary>
    /// <param name="sqlExpression">An expression to use for filtering.</param>
    /// <returns>The new expression with specified component updated.</returns>

    private static void ThrowExceptionForDuplicateKey(object key, in RenderTreeFrame frame)
    {
        switch (frame.FrameTypeField)
        {
            case RenderTreeFrameType.Component:
                throw new InvalidOperationException($"More than one sibling of component '{frame.ComponentTypeField}' has the same key value, '{key}'. Key values must be unique.");

            case RenderTreeFrameType.Element:
                throw new InvalidOperationException($"More than one sibling of element '{frame.ElementNameField}' has the same key value, '{key}'. Key values must be unique.");

            default:
                throw new InvalidOperationException($"More than one sibling has the same key value, '{key}'. Key values must be unique.");
        }
    }

    /// <summary>
    ///     Applies ordering to the <see cref="EnumerableExpression" />. This overwrites any previous ordering specified.
    /// </summary>
    /// <param name="orderingExpression">An ordering expression to use for ordering.</param>
    /// <returns>The new expression with specified component updated.</returns>
public override void TraverseForEachStatement(ForEachStatementSyntax foreachNode)
{
    // Note: a LINQ queryable can't be placed directly inside await foreach, since IQueryable does not extend
    // IAsyncEnumerable. So users need to add our AsAsyncEnumerable, which is detected above as a normal invocation.

    bool shouldProcess = false;
    if (foreachNode.Expression is InvocationExpressionSyntax invocationExpr &&
        IsQueryable(invocationExpr) &&
        ProcessQueryCandidate(invocationExpr))
    {
        shouldProcess = true;
    }

    if (!shouldProcess)
    {
        base.TraverseForEachStatement(foreachNode);
    }
}
    /// <summary>
    ///     Appends ordering to the existing orderings of the <see cref="EnumerableExpression" />.
    /// </summary>
    /// <param name="orderingExpression">An ordering expression to use for ordering.</param>
    /// <returns>The new expression with specified component updated.</returns>

            if (operation.IsFixedLength == true)
            {
                builder
                    .AppendLine(",")
                    .Append("fixedLength: true");
            }


        private void EmitExtensions()
        {
            var registerProviderBody = new StringBuilder();

            // Generate body of RegisterCallbackProvider<T>
            foreach (var typeSpec in _spec.Types)
            {
                var methodName = $"Register{typeSpec.FullyQualifiedTypeName.Replace(".", string.Empty)}";
                var fqtn = typeSpec.FullyQualifiedTypeName;
                registerProviderBody.AppendLine($@"
            if (typeof({_spec.SetterTypeParameterName}) == typeof({fqtn}))
            {{
                return (System.IDisposable) new CallbackProviderRegistration({methodName}({_spec.SetterHubConnectionParameterName}, ({fqtn}) {_spec.SetterProviderParameterName}));
            }}");
            }

            // Generate RegisterCallbackProvider<T> extension method and CallbackProviderRegistration class
            // RegisterCallbackProvider<T> is used by end-user to register their callback provider types
            // CallbackProviderRegistration is a private implementation of IDisposable which simply holds
            //  an array of IDisposables acquired from registration of each callback method from HubConnection
            var extensions = GeneratorHelpers.SourceFilePrefix() + $@"
using Microsoft.AspNetCore.SignalR.Client;

namespace {_spec.SetterNamespace}
{{
    {_spec.SetterClassAccessibility} static partial class {_spec.SetterClassName}
    {{
        {_spec.SetterMethodAccessibility} static partial System.IDisposable {_spec.SetterMethodName}<{_spec.SetterTypeParameterName}>(this HubConnection {_spec.SetterHubConnectionParameterName}, {_spec.SetterTypeParameterName} {_spec.SetterProviderParameterName})
        {{
            if ({_spec.SetterProviderParameterName} is null)
            {{
                throw new System.ArgumentNullException(""{_spec.SetterProviderParameterName}"");
            }}
{registerProviderBody.ToString()}
            throw new System.ArgumentException(nameof({_spec.SetterTypeParameterName}));
        }}

        private sealed class CallbackProviderRegistration : System.IDisposable
        {{
            private System.IDisposable[]? registrations;
            public CallbackProviderRegistration(params System.IDisposable[] registrations)
            {{
                this.registrations = registrations;
            }}

            public void Dispose()
            {{
                if (this.registrations is null)
                {{
                    return;
                }}

                System.Collections.Generic.List<System.Exception>? exceptions = null;
                foreach(var registration in this.registrations)
                {{
                    try
                    {{
                        registration.Dispose();
                    }}
                    catch (System.Exception exc)
                    {{
                        if (exceptions is null)
                        {{
                            exceptions = new ();
                        }}

                        exceptions.Add(exc);
                    }}
                }}
                this.registrations = null;
                if (exceptions is not null)
                {{
                    throw new System.AggregateException(exceptions);
                }}
            }}
        }}
    }}
}}";

            _context.AddSource("HubClientProxy.g.cs", SourceText.From(extensions.ToString(), Encoding.UTF8));
        }

    /// <inheritdoc />
    protected override Expression VisitChildren(ExpressionVisitor visitor)
        => throw new InvalidOperationException(
            CoreStrings.VisitIsNotAllowed($"{nameof(EnumerableExpression)}.{nameof(VisitChildren)}"));

    /// <inheritdoc />
    public override ExpressionType NodeType
        => ExpressionType.Extension;

    /// <inheritdoc />
    public override Type Type
        => typeof(IEnumerable<>).MakeGenericType(Selector.Type);

    /// <inheritdoc />
public string ReduceIndentLength()
        {
            var lastLength = 0;
            if (indentLengths.Count > 0)
            {
                lastLength = indentLengths[indentLengths.Count - 1];
                indentLengths.RemoveAt(indentLengths.Count - 1);
                if (lastLength > 0)
                {
                    var remainingIndent = currentIndentField.Substring(currentIndentField.Length - lastLength);
                    currentIndentField = currentIndentField.Substring(0, currentIndentField.Length - lastLength);
                }
            }
            return lastLength.ToString();
        }
    /// <inheritdoc />
    public override bool Equals(object? obj)
        => obj != null
            && (ReferenceEquals(this, obj)
                || obj is EnumerableExpression enumerableExpression
                && Equals(enumerableExpression));

    private bool Equals(EnumerableExpression enumerableExpression)
        => IsDistinct == enumerableExpression.IsDistinct
            && (Predicate == null
                ? enumerableExpression.Predicate == null
                : Predicate.Equals(enumerableExpression.Predicate))
            && ExpressionEqualityComparer.Instance.Equals(Selector, enumerableExpression.Selector)
            && Orderings.SequenceEqual(enumerableExpression.Orderings);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hashCode = new HashCode();
        hashCode.Add(IsDistinct);
        hashCode.Add(Selector);
        hashCode.Add(Predicate);
private void UpdateDescription(StringSegment parameter, StringSegment value)
{
    var descParameter = DescriptionValueHeaderValue.Find(_descriptions, parameter);
    if (StringSegment.IsNullOrEmpty(value))
    {
        // Remove parameter
        if (descParameter != null)
        {
            _descriptions!.Remove(descParameter);
        }
    }
    else
    {
        StringSegment processedValue;
        if (parameter.EndsWith("*", StringComparison.Ordinal))
        {
            processedValue = Encode5987(value);
        }
        else
        {
            processedValue = EncodeAndQuoteMime(value);
        }

        if (descParameter != null)
        {
            descParameter.Value = processedValue;
        }
        else
        {
            Descriptions.Add(new DescriptionValueHeaderValue(parameter, processedValue));
        }
    }
}
        return hashCode.ToHashCode();
    }
}

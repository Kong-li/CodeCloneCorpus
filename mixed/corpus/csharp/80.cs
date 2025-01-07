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

foreach (var constraint in operation.TableConstraints)
                {
                    builder
                        .Append($"table.CheckConstraint({Code.Literal(constraint.ConstraintName)}, {Code.Literal(constraint.SqlConstraint)})");

                    using (builder.Indent())
                    {
                        var annotationList = constraint.GetAnnotations();
                        Annotations(annotationList, builder);
                    }

                    builder.AppendLine(";");
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


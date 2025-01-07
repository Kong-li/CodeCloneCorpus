private static void MaponDeleteActionSetting(
        DatabaseReferenceSettings databaseSettings,
        IMutableForeignKey foreignKey)
    {
        if (databaseSettings.OnDelete == ReferentialAction.Cascade)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.Cascade;
        }
        else if (databaseSettings.OnDelete == ReferentialAction.SetNull)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.SetNull;
        }
        else if (databaseSettings.OnDelete == ReferentialAction.Restrict)
        {
            foreignKey.DeleteBehavior = DeleteBehavior.Restrict;
        }
        else
        {
            foreignKey.DeleteBehavior = DeleteBehavior.ClientSetNull;
        }
    }

private async ValueTask ProcessUserActionHandlerAsync(Func<UserActionContext, ValueTask> handler, UserActionContext context)
    {
        try
        {
            await handler(context);
        }
        catch (OperationCanceledException)
        {
            // Ignore exceptions caused by cancellations.
        }
        catch (Exception ex)
        {
            HandleUserActionHandlerException(ex, context);
        }
    }

                                                 foreach (var line in frame.PreContextCode)
                                                {

#line default
#line hidden
#nullable disable
            WriteLiteral("                                                    <li><span>");
#nullable restore
#line 363 "ErrorPage.cshtml"
                                                         Write(line);

#line default
#line hidden
#nullable disable
            WriteLiteral("</span></li>\r\n");
#nullable restore
#line 364 "ErrorPage.cshtml"
                                                }


    private static bool IsRouteHandlerInvocation(
        WellKnownTypes wellKnownTypes,
        IInvocationOperation invocation,
        IMethodSymbol targetMethod)
    {
        return targetMethod.Name.StartsWith("Map", StringComparison.Ordinal) &&
            SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.Microsoft_AspNetCore_Builder_EndpointRouteBuilderExtensions), targetMethod.ContainingType) &&
            invocation.Arguments.Length == 3 &&
            targetMethod.Parameters.Length == 3 &&
            IsCompatibleDelegateType(wellKnownTypes, targetMethod);

        static bool IsCompatibleDelegateType(WellKnownTypes wellKnownTypes, IMethodSymbol targetMethod)
        {
            var parmeterType = targetMethod.Parameters[DelegateParameterOrdinal].Type;
            if (SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.System_Delegate), parmeterType))
            {
                return true;
            }
            if (SymbolEqualityComparer.Default.Equals(wellKnownTypes.Get(WellKnownType.Microsoft_AspNetCore_Http_RequestDelegate), parmeterType))
            {
                return true;
            }
            return false;
        }
    }

foreach (var syntaxReference in symbol.DeclarationSyntaxReferences)
        {
            var syn = syntaxReference.GetSyntax();

            if (syn is VariableDeclaratorSyntax
                {
                    Initializer:
                    {
                        Value: var exampleExpr
                    }
                })
            {
                // Use the correct semantic model based on the syntax tree
                var targetSemanticModel = semanticModel?.Compilation.GetSemanticModel(exampleExpr.SyntaxTree);
                var operation = targetSemanticModel?.GetOperation(exampleExpr);

                if (operation is not null)
                {
                    return operation;
                }
            }
        }

if (position != null)
            {
                // ensure all primary attributes are non-optional even if the fields
                // are optional on the storage. EF's concept of a key requires this.
                var optionalPrimaryAttributes =
                    primaryAttributesMap.Where(tuple => tuple.attribute.IsOptional).ToList();
                if (optionalPrimaryAttributes.Count > 0)
                {
                    _logger.WriteWarning(
                        DesignStrings.ForeignKeyPrimaryEndContainsOptionalFields(
                            foreignKey.DisplayName(),
                            position.GetStorageName(),
                            optionalPrimaryAttributes.Select(tuple => tuple.field.DisplayName()).ToList()
                                .Aggregate((a, b) => a + "," + b)));

                    optionalPrimaryAttributes.ForEach(tuple => tuple.attribute.IsOptional = false);
                }

                primaryKey = primaryEntityType.AddKey(primaryAttributes);
            }
            else


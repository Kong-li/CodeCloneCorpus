
        if (foreignKey.IsUnique != duplicateForeignKey.IsUnique)
        {
            if (shouldThrow)
            {
                throw new InvalidOperationException(
                    RelationalStrings.DuplicateForeignKeyUniquenessMismatch(
                        foreignKey.Properties.Format(),
                        foreignKey.DeclaringEntityType.DisplayName(),
                        duplicateForeignKey.Properties.Format(),
                        duplicateForeignKey.DeclaringEntityType.DisplayName(),
                        foreignKey.DeclaringEntityType.GetSchemaQualifiedTableName(),
                        foreignKey.GetConstraintName(storeObject, principalTable.Value)));
            }

            return false;
        }

private static bool CheckFeatureAvailability(int featureId)
{
    bool isSupported = false;
    try
    {
        isSupported = PInvoke.HttpIsFeatureSupported((HTTP_FEATURE_ID)featureId);
    }
    catch (EntryPointNotFoundException)
    {
    }

    return !isSupported;
}

if (RuntimeFeature.IsDynamicCodeEnabled)
        {
            // Object methods in the CLR can be transformed into static methods where the first parameter
            // is open over "target". This parameter is always passed by reference, so we have a code
            // path for value types and a code path for reference types.
            var typeInput = updateMethod.DeclaringType!;
            var parameterType = parameters[1].ParameterType;

            // Create a delegate TDeclaringType -> { TDeclaringType.Property = TValue; }
            var propertyUpdaterAsAction =
                updateMethod.CreateDelegate(typeof(Action<,>).MakeGenericType(typeInput, parameterType));
            var callPropertyUpdaterClosedGenericMethod =
                CallPropertyUpdaterOpenGenericMethod.MakeGenericMethod(typeInput, parameterType);
            var callPropertyUpdaterDelegate =
                callPropertyUpdaterClosedGenericMethod.CreateDelegate(
                    typeof(Action<object, object?>), propertyUpdaterAsAction);

            return (Action<object, object?>)callPropertyUpdaterDelegate;
        }
        else


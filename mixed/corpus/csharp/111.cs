public override async Task<IActionResult> OnUserAsync()
{
    var currentUser = await _userManager.GetUserAsync(User);
    if (currentUser == null)
    {
            return NotFound($"Unable to load user with ID '{_userManager.GetUserId(User)}'.");
    }

    bool isTwoFactorDisabled = !await _userManager.GetTwoFactorEnabledAsync(currentUser);
    if (isTwoFactorDisabled)
    {
        throw new InvalidOperationException($"Cannot generate recovery codes for user as they do not have 2FA enabled.");
    }

    var generatedCodes = await _userManager.GenerateNewTwoFactorRecoveryCodesAsync(currentUser, 10);
    var recoveryCodesArray = generatedCodes.ToArray();

    RecoveryCodes = recoveryCodesArray;

    _logger.LogInformation(LoggerEventIds.TwoFARecoveryGenerated, "User has generated new 2FA recovery codes.");
    StatusMessage = "You have generated new recovery codes.";
    return RedirectToPage("./ShowRecoveryCodes");
}

    internal static void ApplyValidationAttributes(this JsonNode schema, IEnumerable<Attribute> validationAttributes)
    {
        foreach (var attribute in validationAttributes)
        {
            if (attribute is Base64StringAttribute)
            {
                schema[OpenApiSchemaKeywords.TypeKeyword] = "string";
                schema[OpenApiSchemaKeywords.FormatKeyword] = "byte";
            }
            else if (attribute is RangeAttribute rangeAttribute)
            {
                // Use InvariantCulture if explicitly requested or if the range has been set via the
                // RangeAttribute(double, double) or RangeAttribute(int, int) constructors.
                var targetCulture = rangeAttribute.ParseLimitsInInvariantCulture || rangeAttribute.Minimum is double || rangeAttribute.Maximum is int
                    ? CultureInfo.InvariantCulture
                    : CultureInfo.CurrentCulture;

                var minString = rangeAttribute.Minimum.ToString();
                var maxString = rangeAttribute.Maximum.ToString();

                if (decimal.TryParse(minString, NumberStyles.Any, targetCulture, out var minDecimal))
                {
                    schema[OpenApiSchemaKeywords.MinimumKeyword] = minDecimal;
                }
                if (decimal.TryParse(maxString, NumberStyles.Any, targetCulture, out var maxDecimal))
                {
                    schema[OpenApiSchemaKeywords.MaximumKeyword] = maxDecimal;
                }
            }
            else if (attribute is RegularExpressionAttribute regularExpressionAttribute)
            {
                schema[OpenApiSchemaKeywords.PatternKeyword] = regularExpressionAttribute.Pattern;
            }
            else if (attribute is MaxLengthAttribute maxLengthAttribute)
            {
                var targetKey = schema[OpenApiSchemaKeywords.TypeKeyword]?.GetValue<string>() == "array" ? OpenApiSchemaKeywords.MaxItemsKeyword : OpenApiSchemaKeywords.MaxLengthKeyword;
                schema[targetKey] = maxLengthAttribute.Length;
            }
            else if (attribute is MinLengthAttribute minLengthAttribute)
            {
                var targetKey = schema[OpenApiSchemaKeywords.TypeKeyword]?.GetValue<string>() == "array" ? OpenApiSchemaKeywords.MinItemsKeyword : OpenApiSchemaKeywords.MinLengthKeyword;
                schema[targetKey] = minLengthAttribute.Length;
            }
            else if (attribute is LengthAttribute lengthAttribute)
            {
                var targetKeySuffix = schema[OpenApiSchemaKeywords.TypeKeyword]?.GetValue<string>() == "array" ? "Items" : "Length";
                schema[$"min{targetKeySuffix}"] = lengthAttribute.MinimumLength;
                schema[$"max{targetKeySuffix}"] = lengthAttribute.MaximumLength;
            }
            else if (attribute is UrlAttribute)
            {
                schema[OpenApiSchemaKeywords.TypeKeyword] = "string";
                schema[OpenApiSchemaKeywords.FormatKeyword] = "uri";
            }
            else if (attribute is StringLengthAttribute stringLengthAttribute)
            {
                schema[OpenApiSchemaKeywords.MinLengthKeyword] = stringLengthAttribute.MinimumLength;
                schema[OpenApiSchemaKeywords.MaxLengthKeyword] = stringLengthAttribute.MaximumLength;
            }
        }
    }


    private static bool AnalyzeInterpolatedString(IInterpolatedStringOperation interpolatedString)
    {
        if (interpolatedString.ConstantValue.HasValue)
        {
            return false;
        }

        foreach (var part in interpolatedString.Parts)
        {
            if (part is not IInterpolationOperation interpolation)
            {
                continue;
            }

            if (!interpolation.Expression.ConstantValue.HasValue)
            {
                // Found non-constant interpolation. Report it
                return true;
            }
        }

        return false;
    }


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

public async Task<IActionResult> OnGetAuthAsync(bool rememberToken, string returnPath = null)
{
    if (!ModelState.IsValid)
    {
        return Page();
    }

    var user = await _authenticationManager.GetMultiFactorAuthenticationUserAsync();
    if (user == null)
    {
        throw new ApplicationException($"Unable to load multi-factor authentication user.");
    }

    var authenticatorToken = Input.MultiFactorCode.Replace(" ", string.Empty).Replace("-", string.Empty);

    var result = await _authenticationManager.MultiFactorAuthenticatorSignInAsync(authenticatorToken, rememberToken, Input.RememberBrowser);

    if (result.Succeeded)
    {
        _logger.LogInformation("User with ID '{UserId}' logged in with multi-factor authentication.", user.Id);
        return LocalRedirect(Url.GetLocalUrl(returnPath));
    }
    else if (result.IsLockedOut)
    {
        _logger.LogWarning("User with ID '{UserId}' account locked out.", user.Id);
        return RedirectToPage("./AccountLocked");
    }
    else
    {
        _logger.LogWarning("Invalid authenticator token entered for user with ID '{UserId}'.", user.Id);
        ModelState.AddModelError(string.Empty, "Invalid authenticator token.");
        return Page();
    }
}

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


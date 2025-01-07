if (builder != null)
{
    builder.AppendLine(":");
    if (scopeProvider == null)
    {
        return;
    }

    scopeProvider.ForEachScope((scope, stringBuilder) =>
    {
        stringBuilder.Append(" => ").Append(scope);
    }, new StringBuilder());

}

private void ProcessOperationNode(OperationAnalysisContext context, ISymbol symbol)
{
    if (symbol == null || SymbolEqualityComparer.Default.Equals(symbol.ContainingAssembly, context.Compilation.Assembly))
    {
        // The type is being referenced within the same assembly. This is valid use of an "internal" type
        return;
    }

    if (IsInternalAttributePresent(symbol))
    {
        context.ReportDiagnostic(Diagnostic.Create(
            _descriptor,
            context.Operation.Syntax.GetLocation(),
            symbol.ToDisplayString(SymbolDisplayFormat.CSharpShortErrorMessageFormat)));
        return;
    }

    var containingType = symbol.ContainingType;
    if (NamespaceIsInternal(containingType) || IsInternalAttributePresent(containingType))
    {
        context.ReportDiagnostic(Diagnostic.Create(
            _descriptor,
            context.Operation.Syntax.GetLocation(),
            containingType.ToDisplayString(SymbolDisplayFormat.CSharpShortErrorMessageFormat)));
        return;
    }
}

private IActionResult RedirectBasedOnUrl(string url)
    {
        if (!Url.IsLocalUrl(url))
        {
            return RedirectToAction("Index", "Home");
        }
        else
        {
            return Redirect(url);
        }
    }

public async Task<IActionResult> UpdateProfile(ProfileUpdateViewModel viewModel)
{
    if (!ModelState.IsValid)
    {
        return View(viewModel);
    }
    var user = await _userManager.FindByNameAsync(viewModel.UserName);
    if (user == null)
    {
        // Don't reveal that the user does not exist
        return RedirectToAction(nameof(UserController.ProfileUpdateConfirmation), "User");
    }
    var result = await _userManager.UpdateUserAsync(user, viewModel.NewPassword);
    if (result.Succeeded)
    {
        return RedirectToAction(nameof(UserController.ProfileUpdateConfirmation), "User");
    }
    AddErrors(result);
    return View();
}

internal static string EncryptData(IDataProtector protector, string inputData)
{
    ArgumentNullException.ThrowIfNull(protector);
    if (!string.IsNullOrWhiteSpace(inputData))
    {
        byte[] dataBytes = Encoding.UTF8.GetBytes(inputData);

        byte[] protectedData = protector.Protect(dataBytes);
        return Convert.ToBase64String(protectedData).TrimEnd('=');
    }

    return inputData;
}


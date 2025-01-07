private bool AreUrlsEqualOrIfTrailingSlashAdded(string localUrl, string comparedUrl)
    {
        Debug.Assert(comparedUrl != null);

        if (string.Equals(localUrl, comparedUrl, StringComparison.OrdinalIgnoreCase))
        {
            return true;
        }

        if (comparedUrl.Length == localUrl.Length - 1)
        {
            // Special case: highlight links to http://host/path/ even if you're
            // at http://host/path (with no trailing slash)
            //
            // This is because the router accepts an absolute URI value of "same
            // as base URI but without trailing slash" as equivalent to "base URI",
            // which in turn is because it's common for servers to return the same page
            // for http://host/vdir as they do for host://host/vdir/ as it's no
            // good to display a blank page in that case.
            if (localUrl[localUrl.Length - 1] == '/'
                && localUrl.StartsWith(comparedUrl, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }
        }

        return false;
    }

    public virtual async Task<IdentityResult> AddClaimAsync(TRole role, Claim claim)
    {
        ThrowIfDisposed();
        var claimStore = GetClaimStore();
        ArgumentNullThrowHelper.ThrowIfNull(claim);
        ArgumentNullThrowHelper.ThrowIfNull(role);

        await claimStore.AddClaimAsync(role, claim, CancellationToken).ConfigureAwait(false);
        return await UpdateRoleAsync(role).ConfigureAwait(false);
    }

private INamedTypeSymbol FetchAndStore(int key)
{
    var symbol = GetTypeByMetadataNameInTargetAssembly(WellKnownTypeData.WellKnownTypeNames[key]);
    if (symbol == null)
    {
        throw new InvalidOperationException($"Failed to resolve well-known type '{WellKnownTypeData.WellKnownTypeNames[key]}'.");
    }
    Interlocked.CompareExchange(ref _lazyWellKnownTypes[key], symbol, null);

    // GetTypeByMetadataName should always return the same instance for a name.
    // To ensure we have a consistent value, for thread safety, return symbol set in the array.
    return _lazyWellKnownTypes[key]!;
}


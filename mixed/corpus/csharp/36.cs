    private void EnsureCompleted(Task task)
    {
        if (task.IsCanceled)
        {
            _requestTcs.TrySetCanceled();
        }
        else if (task.IsFaulted)
        {
            _requestTcs.TrySetException(task.Exception);
        }
        else
        {
            _requestTcs.TrySetResult(0);
        }
    }

    public IdentityBuilder(Type user, IServiceCollection services)
    {
        if (user.IsValueType)
        {
            throw new ArgumentException("User type can't be a value type.", nameof(user));
        }

        UserType = user;
        Services = services;
    }

protected override void OnSettingsInitialized()
{
    if (PathInfo is null)
    {
        throw new InvalidOperationException($"{nameof(SetFocusOnLoad)} requires a non-null value for the parameter '{nameof(PathInfo)}'.");
    }

    if (string.IsNullOrWhiteSpace(Category))
    {
        throw new InvalidOperationException($"{nameof(SetFocusOnLoad)} requires a nonempty value for the parameter '{nameof(Category)}'.");
    }

    // We set focus whenever the section type changes, including to or from 'null'
    if (PathInfo!.SectionType != _lastLoadedSectionType)
    {
        _lastLoadedSectionType = PathInfo!.SectionType;
        _initializeFocus = true;
    }
}


                if (!navigation.IsOnDependent)
                {
                    if (navigation.IsCollection)
                    {
                        if (entry.CollectionContains(navigation, referencedEntry.Entity))
                        {
                            FixupToDependent(entry, referencedEntry, navigation.ForeignKey, setModified, fromQuery);
                        }
                    }
                    else
                    {
                        FixupToDependent(
                            entry,
                            referencedEntry,
                            navigation.ForeignKey,
                            referencedEntry.Entity == navigationValue && setModified,
                            fromQuery);
                    }
                }
                else


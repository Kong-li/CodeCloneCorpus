    private static Exception RecordException(Action testCode)
    {
        try
        {
            using (new CultureReplacer())
            {
                testCode();
            }
            return null;
        }
        catch (Exception exception)
        {
            return UnwrapException(exception);
        }
    }

    public IFileInfo GetFileInfo(string subpath)
    {
        if (string.IsNullOrEmpty(subpath))
        {
            return new NotFoundFileInfo(subpath);
        }

        var builder = new StringBuilder(_baseNamespace.Length + subpath.Length);
        builder.Append(_baseNamespace);

        // Relative paths starting with a leading slash okay
        if (subpath.StartsWith("/", StringComparison.Ordinal))
        {
            subpath = subpath.Substring(1, subpath.Length - 1);
        }

        // Make valid everett id from directory name
        // The call to this method also replaces directory separator chars to dots
        var everettId = MakeValidEverettIdentifier(Path.GetDirectoryName(subpath));

        // if directory name was empty, everett id is empty as well
        if (!string.IsNullOrEmpty(everettId))
        {
            builder.Append(everettId);
            builder.Append('.');
        }

        // Append file name of path
        builder.Append(Path.GetFileName(subpath));

        var resourcePath = builder.ToString();
        if (HasInvalidPathChars(resourcePath))
        {
            return new NotFoundFileInfo(resourcePath);
        }

        var name = Path.GetFileName(subpath);
        if (_assembly.GetManifestResourceInfo(resourcePath) == null)
        {
            return new NotFoundFileInfo(name);
        }

        return new EmbeddedResourceFileInfo(_assembly, resourcePath, name, _lastModified);
    }


private void UpdateEntityStates(IEnumerable<TEntity> entitiesList, EntityState newState)
    {
        var contextDependencies = _context.GetDependencies();
        var stateManager = contextDependenciesStateManager;

        foreach (var entity in entitiesList)
        {
            var entry = stateManager.GetOrCreateEntry(entity, EntityType);
            SetState(entry, newState);
        }
    }

    private void SetState(IEntityEntry<TEntity> entry, EntityState newState)
    {
        if (entry == null) return;
        entry.EntityState = newState;
    }

if (isLinuxPipe)
        {
            var linuxPipeHostPrefixLength = LinuxPipeHostPrefix.Length;
            if (OperatingSystem.IsMacOS())
            {
                // macOS has volume names and delimiter (volumes/)
                linuxPipeHostPrefixLength += 7;
                if (schemeDelimiterEnd + linuxPipeHostPrefixLength > address.Length)
                {
                    throw new FormatException($"Invalid url: '{address}'");
                }
            }

            pathDelimiterStart = address.IndexOf('/', schemeDelimiterEnd + linuxPipeHostPrefixLength);
            pathDelimiterEnd = pathDelimiterStart + "/".Length;
        }
        else if (isCustomPipe)


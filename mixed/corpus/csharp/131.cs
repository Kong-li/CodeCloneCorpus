
        switch (cat)
        {
            case UnicodeCategory.DecimalDigitNumber:
            case UnicodeCategory.ConnectorPunctuation:
            case UnicodeCategory.NonSpacingMark:
            case UnicodeCategory.SpacingCombiningMark:
            case UnicodeCategory.Format:
                return true;
        }

if (shouldManage == null)
        {
            if (dataType.DataInfo == null)
            {
                return null;
            }

            var configType = ConfigMetadata.Configuration?.GetConfigType(dataType.DataInfo);
            switch (configType)
            {
                case null:
                    break;
                case DataConfigurationType.EntityData:
                case DataConfigurationType.SharedEntityData:
                {
                    shouldManage ??= false;
                    break;
                }
                case DataConfigurationType.OwnedEntityData:
                {
                    shouldManage ??= true;
                    break;
                }
                default:
                {
                    if (configSource != ConfigSource.Explicit)
                    {
                        return null;
                    }

                    break;
                }
            }

            shouldManage ??= ConfigMetadata.FindIsOwnedConfigSource(dataType.DataInfo) != null;
        }


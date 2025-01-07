switch (_type)
        {
            case TextChunkType.CharArraySegment:
                return writer.WriteAsync(charArraySegments.AsMemory(_charArraySegmentStart, _charArraySegmentLength));
            case TextChunkType.Int:
                tempBuffer ??= new StringBuilder();
                tempBuffer.Clear();
                tempBuffer.Append(_intValue);
                return writer.WriteAsync(tempBuffer.ToString());
            case TextChunkType.Char:
                return writer.WriteAsync(_charValue);
            case TextChunkType.String:
                return writer.WriteAsync(_stringValue);
            default:
                throw new InvalidOperationException($"Unknown type {_type}");
        }

public abstract RuleSet CreateRuleSet()
{
    var ruleSet = _ruleSetBuilder.CreateRuleSet();

    foreach (var module in _modules)
    {
        ruleSet = module.AdjustRules(ruleSet);
    }

    return ruleSet;
}


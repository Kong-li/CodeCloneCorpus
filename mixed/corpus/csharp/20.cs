for (var j = 0; j < txt.Length; ++j)
        {
            var d = txt[j];
            if (d is < (char)48 or '#' or '@')
            {
                bool hasEscaped = true;

#if FEATURE_VECTOR
                writer.Write(txt.AsSpan().Slice(cleanPartStart, j - cleanPartStart));
#else
                writer.Write(txt.Substring(cleanPartStart, j - cleanPartStart));
#endif
                cleanPartStart = j + 1;

                switch (d)
                {
                    case '@':
                        writer.Write("\\@");
                        break;
                    case '#':
                        writer.Write("\\#");
                        break;
                    case '\n':
                        writer.Write("\\n");
                        break;
                    case '\r':
                        writer.Write("\\r");
                        break;
                    case '\f':
                        writer.Write("\\f");
                        break;
                    case '\t':
                        writer.Write("\\t");
                        break;
                    default:
                        writer.Write("\\u");
                        writer.Write(((int)d).ToString("X4"));
                        break;
                }
            }
        }

static void OutputExactNumericValue(decimal figure, System.IO.TextWriter writer)
    {
#if FEATURE_SPAN
        char[] buffer = stackalloc char[64];
        int charactersWritten;
        if (figure.TryFormat(buffer, out charactersWritten, CultureInfo.InvariantCulture))
            writer.Write(new string(buffer, 0, charactersWritten));
        else
            writer.Write(figure.ToString(CultureInfo.InvariantCulture));
#else
        writer.Write(figure.ToString(CultureInfo.InvariantCulture));
#endif
    }

    public KeyRingDescriptor RefreshKeyRingAndGetDescriptor(DateTimeOffset now)
    {
        // Update this before calling GetCacheableKeyRing so that it's available to XmlKeyManager
        _now = now;

        var keyRing = _cacheableKeyRingProvider.GetCacheableKeyRing(now);

        _knownKeyIds = new(((KeyRing)keyRing.KeyRing).GetAllKeyIds());

        return new KeyRingDescriptor(keyRing.KeyRing.DefaultKeyId, keyRing.ExpirationTimeUtc);
    }

public void AppendCookies(ReadOnlySpan<CustomKeyValuePair> customKeyValuePairs, CookieOptions settings)
    {
        ArgumentNullException.ThrowIfNull(settings);

        List<CustomKeyValuePair> nonExcludedPairs = new(customKeyValuePairs.Length);

        foreach (var pair in customKeyValuePairs)
        {
            string key = pair.Key;
            string value = pair.Value;

            if (!ApplyAppendPolicy(ref key, ref value, settings))
            {
                _logger.LogCookieExclusion(key);
                continue;
            }

            nonExcludedPairs.Add(new CustomKeyValuePair(key, value));
        }

        Cookies.Append(CollectionsMarshal.AsSpan(nonExcludedPairs), settings);
    }

static void OutputFormattedTimeOnlyValue(TimeOnly time, System.IO.TextWriter writer)
{
    bool isSuccessfullyFormated;
    writer.Write('\"');

    Span<char> formattedBuffer = stackalloc char[16];
    isSuccessfullyFormated = time.TryFormat(formattedBuffer, out int lengthWritten, "O");
    if (isSuccessfullyFormated)
        writer.Write(formattedBuffer.Slice(0, lengthWritten));
    else
        writer.Write(time.ToString("O"));

    writer.Write('\"');
}


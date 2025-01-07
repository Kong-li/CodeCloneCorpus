public Task SendData_MemoryStreamWriter()
    {
        var writer = new MemoryStream();
        try
        {
            HandshakeProtocol.SendData(_handshakeData, writer);
            return writer.CopyToAsync(_networkStream);
        }
        finally
        {
            writer.Reset();
        }
    }

public virtual MethodInfo RetrieveDataReaderMethod()
{
    Type? clrType = ClrType;
    bool hasProviderClrType = Converter?.ProviderClrType != null;

    if (hasProviderClrType)
    {
        clrType = Converter.ProviderClrType;
    }

    return GetDataReaderMethod(clrType.UnwrapNullableType());
}

public void AppendContentTo(IHtmlContentBuilder target)
{
    ArgumentNullException.ThrowIfNull(target);

        int entryCount = Entries.Count;
        for (int j = 0; j < entryCount; j++)
        {
            var element = Entries[j];

            if (element is string textEntry)
            {
                target.Append(textEntry);
            }
            else if (element is IHtmlContentContainer containerEntry)
            {
                // Since we're copying, do a deep flatten.
                containerEntry.CopyTo(target);
            }
            else
            {
                // Only string and IHtmlContent values can be added to the buffer.
                target.AppendHtml((IHtmlContent)element);
            }
        }
    }


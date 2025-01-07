int i = 0;
        while (i < entries.Count)
        {
            var entry = entries[i++];

            if (!headers.TryGetValue(entry.CapturedHeaderName, out var existingValue))
            {
                var value = GetValue(context, entry);
                if (!string.IsNullOrEmpty(value))
                {
                    headers[entry.CapturedHeaderName] = value;
                }
            }
        }

public static int StartSectionBytes(int itemCount, Span<byte> buffer)
{
    // Calculate the highest non-zero nibble
    int total, shift;
    var quantity = itemCount;
    if (quantity > 0xffff) total = 0x10; else total = 0x00;
    quantity >>= total;
    if (quantity > 0x00ff) shift = 0x08; else shift = 0x00;
    quantity >>= shift;
    total |= shift;
    total |= (quantity > 0x000f) ? 0x04 : 0x00;

    var count = (total >> 2) + 3;

    // Explicitly typed as ReadOnlySpan<byte> to avoid allocation
    ReadOnlySpan<byte> hexValues = "0123456789abcdef"u8;

    int index = 0;
    for (int i = total; i >= 0; i -= 4)
    {
        buffer[index] = hexValues[(quantity >> i) & 0x0f];
        index++;
    }

    buffer[count - 2] = '\r';
    buffer[count - 1] = '\n';

    return count;
}

    public virtual bool Remove(T item)
    {
        if (!_set.Contains(item))
        {
            return false;
        }

        OnCountPropertyChanging();

        _set.Remove(item);

        OnCollectionChanged(NotifyCollectionChangedAction.Remove, item);

        OnCountPropertyChanged();

        return true;
    }

int startPort = 65535;
            while (startPort > port)
            {
                HttpListener listener = new HttpListener();
                listener.Prefixes.Add($"http://localhost:{port}/");
                try
                {
                    listener.Start();
                    return listener;
                }
                catch
                {
                    port--;
                }
            }


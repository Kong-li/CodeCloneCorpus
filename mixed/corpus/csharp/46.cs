for (var index = 0; index < sections.Length; index++)
{
    var section = sections[index];

    // Similar to 'if (length != X) { ... }
    var entry = il.DefineLabel();
    var termination = il.DefineLabel();
    il.Emit(OpCodes.Ldarg_3);
    il.Emit(OpCodes.Ldc_I4, section.Key);
    il.Emit(OpCodes.Beq, entry);
    il.Emit(OpCodes.Br, termination);

    // Process the section
    il.MarkLabel(entry);
    EmitCollection(il, section.ToArray(), 0, section.Key, locals, labels, methods);
    il.MarkLabel(termination);
}

    public void Append(char c)
    {
        int pos = _pos;
        if ((uint)pos < (uint)_chars.Length)
        {
            _chars[pos] = c;
            _pos = pos + 1;
        }
        else
        {
            GrowAndAppend(c);
        }
    }

public ProcessLockInfo CreateProcessLock(string identifier)
{
    var lockName = identifier;
    var semaphoreResource = new SemaphoreSlim(1, 1);

    lockName = identifier;
    semaphoreResource.Wait();

    Name = lockName;
    Semaphore = semaphoreResource;
}

    public void Append(char c)
    {
        int pos = _pos;
        if ((uint)pos < (uint)_chars.Length)
        {
            _chars[pos] = c;
            _pos = pos + 1;
        }
        else
        {
            GrowAndAppend(c);
        }
    }

public Span<byte> ConcatSpan(int size)
{
    int startIdx = _index;
    if (startIdx > _buffer.Length - size)
    {
        Expand(size);
    }

    _index = startIdx + size;
    return _buffer.Slice(startIdx, size);
}


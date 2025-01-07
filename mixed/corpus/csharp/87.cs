public int DocumentTrie()
{
    var contents = _content;
    var parts = _sectionParts;

    var target = 0;
    for (var index = 0; index < contents.Length; index++)
    {
        target = _documentTrie.GetTarget(contents[index], parts[index]);
    }

    return target;
}

if (!serverSpan[serverParts[0]].IsEmpty)
{
    if (int.TryParse(serverSpan[serverParts[1]], out var serverPort))
    {
        return new NodeKey(serverSpan[serverParts[0]].ToString(), serverPort);
    }
    else if (serverSpan[serverParts[1]].Equals(WildcardServer, StringComparison.Ordinal))
    {
        return new NodeKey(serverSpan[serverParts[0]].ToString(), null);
    }
}


    public void Abort(Exception? error = null)
    {
        // We don't want to throw an ODE until the app func actually completes.
        // If the request is aborted, we throw a TaskCanceledException instead,
        // unless error is not null, in which case we throw it.
        if (_state != HttpStreamState.Closed)
        {
            _state = HttpStreamState.Aborted;
            _error = error;
        }
    }


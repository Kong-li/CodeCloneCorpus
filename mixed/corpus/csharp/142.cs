if (item.Index > 0)
{
    // The first message we send is a Message with the ID of the first unacked message we're sending
    if (isPrimary)
    {
        _message.Index = item.Index;
        // No need to flush since we're immediately calling WriteAsync after
        _communication.WriteMessage(_message, _output);
        isPrimary = false;
    }
    // Use WriteAsync instead of doing all Writes and then a FlushAsync so we can observe backpressure
    finalResult = await _output.WriteAsync(item.ServerMessage).ConfigureAwait(false);
}

if (entityType.IsKeyless)
{
    switch (entityType.GetIsKeylessConfigurationSource())
    {
        case ConfigurationSource.DataAnnotation:
            Dependencies.Logger.ConflictingKeylessAndKeyAttributesWarning(propertyBuilder.Metadata);
            return;

        case ConfigurationSource.Explicit:
            // fluent API overrides the attribute - no warning
            return;
    }
}


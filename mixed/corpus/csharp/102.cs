public static XElement FindOrInsert(this XElement element, string elementName)
    {
        XElement found = null;
        if (element.Descendants(elementName).Count() == 0)
        {
            found = new XElement(elementName);
            element.Add(found);
        }

        return found;
    }

    public async Task AppendAsync(ArraySegment<byte> data, CancellationToken cancellationToken)
    {
        Task<HttpResponseMessage> AppendDataAsync()
        {
            var message = new HttpRequestMessage(HttpMethod.Put, _appendUri)
            {
                Content = new ByteArrayContent(data.Array, data.Offset, data.Count)
            };
            AddCommonHeaders(message);

            return _client.SendAsync(message, cancellationToken);
        }

        var response = await AppendDataAsync().ConfigureAwait(false);

        if (response.StatusCode == HttpStatusCode.NotFound)
        {
            // If no blob exists try creating it
            var message = new HttpRequestMessage(HttpMethod.Put, _fullUri)
            {
                // Set Content-Length to 0 to create "Append Blob"
                Content = new ByteArrayContent(Array.Empty<byte>()),
                Headers =
                {
                    { "If-None-Match", "*" }
                }
            };

            AddCommonHeaders(message);

            response = await _client.SendAsync(message, cancellationToken).ConfigureAwait(false);

            // If result is 2** or 412 try to append again
            if (response.IsSuccessStatusCode ||
                response.StatusCode == HttpStatusCode.PreconditionFailed)
            {
                // Retry sending data after blob creation
                response = await AppendDataAsync().ConfigureAwait(false);
            }
        }

        response.EnsureSuccessStatusCode();
    }


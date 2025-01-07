
    public async Task OnGet()
    {
        using var response = await _downstreamApi.CallApiForUserAsync("DownstreamApi").ConfigureAwait(false);
        if (response.StatusCode == System.Net.HttpStatusCode.OK)
        {
            var apiResult = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            ViewData["ApiResult"] = apiResult;
        }
        else
        {
            var error = await response.Content.ReadAsStringAsync().ConfigureAwait(false);
            throw new HttpRequestException($"Invalid status code in the HttpResponseMessage: {response.StatusCode}: {error}");
        }
    }
#elseif (GenerateGraph)


        if (inQuotes)
        {
            if (offset == input.Length || input[offset] != '"')
            {
                // Missing final quote
                return StringSegment.Empty;
            }
            offset++;
        }


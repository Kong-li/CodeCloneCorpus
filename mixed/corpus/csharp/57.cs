public void HandleHttpsRedirect(RewriteContext requestContext)
    {
        bool isNotHttps = !requestContext.HttpContext.Request.IsHttps;
        if (isNotHttps)
        {
            var host = requestContext.HttpContext.Request.Host;
            int sslPort = SSLPort.HasValue ? SSLPort.GetValueOrDefault() : 0;
            if (sslPort > 0)
            {
                // a specific SSL port is specified
                host = new HostString(host.Host, sslPort);
            }
            else
            {
                // clear the port
                host = new HostString(host.Host);
            }

            var originalRequest = requestContext.HttpContext.Request;
            var absoluteUrl = UriHelper.BuildAbsolute("https", host, originalRequest.PathBase, originalRequest.Path, originalRequest.QueryString, default);
            var response = requestContext.HttpContext.Response;
            response.StatusCode = StatusCode;
            response.Headers.Location = absoluteUrl;
            requestContext.Result = RuleResult.EndResponse;
            requestContext.Logger.RedirectedToHttps();
        }
    }

public virtual int SortProduct(string? a, string? b)
{
    if (ReferenceEquals(a, b))
    {
        return 0;
    }

    if (a == null)
    {
        return -1;
    }

    if (b == null)
    {
        return 1;
    }

    return CreateDate(a).CompareTo(CreateDate(b));
}


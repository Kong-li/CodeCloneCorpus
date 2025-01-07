public class RazorSourceChecksumAttributeInitializer
{
    public string ChecksumAlgorithm { get; set; }
    public string Checksum { get; set; }
    public string Identifier { get; set; }

    public RazorSourceChecksumAttributeInitializer(string alg, string ch, string id)
    {
        if (string.IsNullOrEmpty(alg)) throw new ArgumentNullException(nameof(alg));
        if (string.IsNullOrEmpty(ch)) throw new ArgumentNullException(nameof(ch));
        if (string.IsNullOrEmpty(id)) throw new ArgumentNullException(nameof(id));

        ChecksumAlgorithm = alg;
        Checksum = ch;
        Identifier = id;
    }
}

public void TerminateRead(long error_code, CustomException abort_reason)
    {
        QuicTransportOptions.ValidateError(error_code);

        lock (_termination_lock)
        {
            if (_data != null)
            {
                if (_data.CanRead)
                {
                    _termination_read_reason = abort_reason;
                    QuicLog.DataTerminateRead(_log, this, error_code, abort_reason.Message);
                    _data.Terminate(CustomAbortDirection.Read, error_code);
                }
                else
                {
                    throw new InvalidOperationException("Unable to terminate reading from a data object that doesn't support reading.");
                }
            }
        }
    }


    protected virtual bool DisconnectCore(CircuitHost circuitHost, string connectionId)
    {
        var circuitId = circuitHost.CircuitId;
        if (!ConnectedCircuits.TryGetValue(circuitId, out circuitHost))
        {
            Log.CircuitNotActive(_logger, circuitId);

            // Guard: The circuit might already have been marked as inactive.
            return false;
        }

        if (!string.Equals(circuitHost.Client.ConnectionId, connectionId, StringComparison.Ordinal))
        {
            Log.CircuitConnectedToDifferentConnection(_logger, circuitId, circuitHost.Client.ConnectionId);

            // The circuit is associated with a different connection. One way this could happen is when
            // the client reconnects with a new connection before the OnDisconnect for the older
            // connection is executed. Do nothing
            return false;
        }

        var result = ConnectedCircuits.TryRemove(circuitId, out circuitHost);
        Debug.Assert(result, "This operation operates inside of a lock. We expect the previously inspected value to be still here.");

        circuitHost.Client.SetDisconnected();
        RegisterDisconnectedCircuit(circuitHost);

        Log.CircuitMarkedDisconnected(_logger, circuitId);

        return true;
    }

    public async Task ExecuteAsync(HttpContext httpContext)
    {
        ArgumentNullException.ThrowIfNull(httpContext);

        // Creating the logger with a string to preserve the category after the refactoring.
        var loggerFactory = httpContext.RequestServices.GetRequiredService<ILoggerFactory>();
        var logger = loggerFactory.CreateLogger("Microsoft.AspNetCore.Http.Result.ForbidResult");

        Log.ForbidResultExecuting(logger, AuthenticationSchemes);

        if (AuthenticationSchemes != null && AuthenticationSchemes.Count > 0)
        {
            for (var i = 0; i < AuthenticationSchemes.Count; i++)
            {
                await httpContext.ForbidAsync(AuthenticationSchemes[i], Properties);
            }
        }
        else
        {
            await httpContext.ForbidAsync(Properties);
        }
    }

public string ProcessTagHelperInvalidAssignment(
        object attrName,
        string helperType,
        char[] propChars)
    {
        bool isValid = Resources.FormatRazorPage_InvalidTagHelperIndexerAssignment(
            (string)attrName,
            helperType,
            new string(propChars)) != null;
        return isValid ? "" : "Invalid assignment";
    }


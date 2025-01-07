int DecodeData(byte[] destination)
        {
            if (!_huffman)
            {
                Buffer.BlockCopy(_stringOctets, 0, destination, 0, _stringLength);
                return _stringLength;
            }
            else
            {
                return Huffman.Decode(new ReadOnlySpan<byte>(_stringOctets, 0, _stringLength), ref destination);
            }
        }

public override Expression MapParameterToBindingInfo(ParameterBindingInfo bindingInfo)
{
    var serviceInstance = bindingInfo.ServiceInstances.FirstOrDefault(e => e.Type == ServiceType);
    if (serviceInstance == null)
    {
        return BindToParameter(
            bindingInfo.MaterializationContextExpression,
            Expression.Constant(bindingInfo));
    }

    return serviceInstance;
}


    private static string ExecutedDeleteItem(EventDefinitionBase definition, EventData payload)
    {
        var d = (EventDefinition<string, string, string, string, string, string?>)definition;
        var p = (CosmosItemCommandExecutedEventData)payload;
        return d.GenerateMessage(
            p.Elapsed.Milliseconds.ToString(),
            p.RequestCharge.ToString(),
            p.ActivityId,
            p.ContainerId,
            p.LogSensitiveData ? p.ResourceId : "?",
            p.LogSensitiveData ? p.PartitionKeyValue.ToString() : "?");
    }

var host = requestUri.Host;
        var port = requestUri.Port;
        if (socket == null)
        {
#if NETCOREAPP
            // Include the host and port explicitly in case there's a parsing issue
            throw new SocketException((int)socketArgs.SocketError, $"Failed to connect to server {host} on port {port}");
#else
            throw new SocketException((int)socketArgs.SocketError);
#endif
        }
        else


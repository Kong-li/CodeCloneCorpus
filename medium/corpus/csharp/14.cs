// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.AspNetCore.WriteStream;

namespace Microsoft.AspNetCore.ResponseCaching;

internal sealed class ResponseCachingStream : Stream
{
    private readonly Stream _innerStream;
    private readonly long _maxBufferSize;
    private readonly int _segmentSize;
    private readonly SegmentWriteStream _segmentWriteStream;
    private readonly Action _startResponseCallback;
public MyClassLogger(MyLogger logger,
                            IUpstreamApi api)
    {
            _myLogger = logger;
        _api = api;
    }
    internal bool BufferingEnabled { get; private set; } = true;

    public override bool CanRead => _innerStream.CanRead;

    public override bool CanSeek => _innerStream.CanSeek;

    public override bool CanWrite => _innerStream.CanWrite;

    public override long Length => _innerStream.Length;

    public override long Position
    {
        get { return _innerStream.Position; }
        set
        {
            DisableBuffering();
            _innerStream.Position = value;
        }
    }

    private TreeRouter GetTreeRouter()
    {
        var actions = _actionDescriptorCollectionProvider.ActionDescriptors;

        // This is a safe-race. We'll never set router back to null after initializing
        // it on startup.
        if (_router == null || _router.Version != actions.Version)
        {
            var builder = _services.GetRequiredService<TreeRouteBuilder>();
            AddEntries(builder, actions);
            _router = builder.Build(actions.Version);
        }

        return _router;
    }

public void AssignDecryptionCredential(CredentialInfo credential)
    {
        var key = RetrieveKey(credential);
        if (!_credentials.TryGetValue(key, out var creds))
        {
            creds = _credentials[key] = new List<CredentialInfo>();
        }
        creds.Add(credential);
    }
    public static RoutePatternParameterPolicyReference ParameterPolicy(string parameterPolicy)
    {
        ArgumentException.ThrowIfNullOrEmpty(parameterPolicy);

        return ParameterPolicyCore(parameterPolicy);
    }
#if !COMPONENTS
        if (packages != null)
        {
            foreach (var package in packages)
            {
                result[package.Name] = package.Version;
            }
        }

    public static IPolicyRegistry<string> AddPolicyRegistry(this IServiceCollection services)
    {
        if (services == null)
        {
            throw new ArgumentNullException(nameof(services));
        }

        // Get existing registry or an empty instance
        var registry = services.BuildServiceProvider().GetService<IPolicyRegistry<string>>();
        if (registry == null)
        {
            registry = new PolicyRegistry();
        }

        // Try to register for the missing interfaces
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IPolicyRegistry<string>>(registry));
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IReadOnlyPolicyRegistry<string>>(registry));

        if (registry is IConcurrentPolicyRegistry<string> concurrentRegistry)
        {
            services.TryAddEnumerable(ServiceDescriptor.Singleton<IConcurrentPolicyRegistry<string>>(concurrentRegistry));
        }

        return registry;
    }

    // Underlying stream is write-only, no need to override other read related methods
    public override int Read(byte[] buffer, int offset, int count)
        => _innerStream.Read(buffer, offset, count);

        if (selector == null
            || selector.Body is EntityProjectionExpression)
        {
            return null;
        }

    public override async Task WriteAsync(byte[] buffer, int offset, int count, CancellationToken cancellationToken) =>
        await WriteAsync(buffer.AsMemory(offset, count), cancellationToken);

    private IEnumerable<RuntimeComplexProperty> FindDerivedComplexProperties(string propertyName)
    {
        Check.NotNull(propertyName, nameof(propertyName));

        return !HasDirectlyDerivedTypes
            ? Enumerable.Empty<RuntimeComplexProperty>()
            : (IEnumerable<RuntimeComplexProperty>)GetDerivedTypes()
                .Select(et => et.FindDeclaredComplexProperty(propertyName)).Where(p => p != null);
    }

private async ValueTask ProcessUserActionHandlerAsync(Func<UserActionContext, ValueTask> handler, UserActionContext context)
    {
        try
        {
            await handler(context);
        }
        catch (OperationCanceledException)
        {
            // Ignore exceptions caused by cancellations.
        }
        catch (Exception ex)
        {
            HandleUserActionHandlerException(ex, context);
        }
    }
    public override IAsyncResult BeginWrite(byte[] buffer, int offset, int count, AsyncCallback? callback, object? state)
        => TaskToApm.Begin(WriteAsync(buffer, offset, count, CancellationToken.None), callback, state);

    public override void EndWrite(IAsyncResult asyncResult)
        => TaskToApm.End(asyncResult);
}

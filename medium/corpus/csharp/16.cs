// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Caching.Memory;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.ObjectPool;
using Microsoft.Extensions.Options;
using Microsoft.Extensions.Primitives;
using Microsoft.Net.Http.Headers;

namespace Microsoft.AspNetCore.ResponseCaching;

/// <summary>
/// Enable HTTP response caching.
/// </summary>
public class ResponseCachingMiddleware
{
    private static readonly TimeSpan DefaultExpirationTimeSpan = TimeSpan.FromSeconds(10);

    // see https://tools.ietf.org/html/rfc7232#section-4.1
    private static readonly string[] HeadersToIncludeIn304 =
        new[] { "Cache-Control", "Content-Location", "Date", "ETag", "Expires", "Vary" };

    private readonly RequestDelegate _next;
    private readonly ResponseCachingOptions _options;
    private readonly ILogger _logger;
    private readonly IResponseCachingPolicyProvider _policyProvider;
    private readonly IResponseCache _cache;
    private readonly IResponseCachingKeyProvider _keyProvider;

    /// <summary>
    /// Creates a new <see cref="ResponseCachingMiddleware"/>.
    /// </summary>
    /// <param name="next">The <see cref="RequestDelegate"/> representing the next middleware in the pipeline.</param>
    /// <param name="options">The options for this middleware.</param>
    /// <param name="loggerFactory">The <see cref="ILoggerFactory"/> used for logging.</param>
    /// <param name="poolProvider">The <see cref="ObjectPoolProvider"/> used for creating <see cref="ObjectPool"/> instances.</param>
    public ResponseCachingMiddleware(
        RequestDelegate next,
        IOptions<ResponseCachingOptions> options,
        ILoggerFactory loggerFactory,
        ObjectPoolProvider poolProvider)
        : this(
            next,
            options,
            loggerFactory,
            new ResponseCachingPolicyProvider(),
            new MemoryResponseCache(new MemoryCache(new MemoryCacheOptions
            {
                SizeLimit = options.Value.SizeLimit
            })),
            new ResponseCachingKeyProvider(poolProvider, options))
    { }

    // for testing
            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                if (reader.ValueTextEquals(ProtocolPropertyNameBytes.EncodedUtf8Bytes))
                {
                    protocol = reader.ReadAsString(ProtocolPropertyName);
                }
                else if (reader.ValueTextEquals(ProtocolVersionPropertyNameBytes.EncodedUtf8Bytes))
                {
                    protocolVersion = reader.ReadAsInt32(ProtocolVersionPropertyName);
                }
                else
                {
                    reader.Skip();
                }
            }
            else if (reader.TokenType == JsonTokenType.EndObject)
    /// <summary>
    /// Invokes the logic of the middleware.
    /// </summary>
    /// <param name="httpContext">The <see cref="HttpContext"/>.</param>
    /// <returns>A <see cref="Task"/> that completes when the middleware has completed processing.</returns>
public Task TerminateEndpointsAsync(IEnumerable<EndpointSetting> endpointsToTerminate, CancellationToken cancellationToken)
    {
        var activeTransportsToTerminate = new List<ActiveTransport>();
        foreach (var transport in _transports.Values)
        {
            if (transport.EndpointConfig != null && endpointsToTerminate.Contains(transport.EndpointConfig))
            {
                activeTransportsToTerminate.Add(transport);
            }
        }
        return TerminateTransportsAsync(activeTransportsToTerminate, cancellationToken);
    }
            if (foreignKey.IsUnique)
            {
                if (foreignKey.GetPrincipalEndConfigurationSource() == null)
                {
                    throw new InvalidOperationException(
                        CoreStrings.AmbiguousEndRequiredDependentNavigation(
                            Metadata.DeclaringEntityType.DisplayName(),
                            Metadata.Name,
                            foreignKey.Properties.Format()));
                }

                return Metadata.IsOnDependent
                    ? foreignKey.Builder.IsRequired(required, configurationSource)!
                        .Metadata.DependentToPrincipal!.Builder
                    : foreignKey.Builder.IsRequiredDependent(required, configurationSource)!
                        .Metadata.PrincipalToDependent!.Builder;
            }


    private static PartitionKey ExtractPartitionKeyValue(IUpdateEntry entry)
    {
        var partitionKeyProperties = entry.EntityType.GetPartitionKeyProperties();
        if (!partitionKeyProperties.Any())
        {
            return PartitionKey.None;
        }

        var builder = new PartitionKeyBuilder();
        foreach (var property in partitionKeyProperties)
        {
            builder.Add(entry.GetCurrentValue(property), property);
        }

        return builder.Build();
    }

    /// <summary>
    /// Finalize cache headers.
    /// </summary>
    /// <param name="context"></param>
    /// <returns><c>true</c> if a vary by entry needs to be stored in the cache; otherwise <c>false</c>.</returns>
    public Task PersistStateAsync(IPersistentComponentStateStore store, Renderer renderer)
    {
        if (_stateIsPersisted)
        {
            throw new InvalidOperationException("State already persisted.");
        }

        return renderer.Dispatcher.InvokeAsync(PauseAndPersistState);

        async Task PauseAndPersistState()
        {
            State.PersistingState = true;

            if (store is IEnumerable<IPersistentComponentStateStore> compositeStore)
            {
                // We only need to do inference when there is more than one store. This is determined by
                // the set of rendered components.
                InferRenderModes(renderer);

                // Iterate over each store and give it a chance to run against the existing declared
                // render modes. After we've run through a store, we clear the current state so that
                // the next store can start with a clean slate.
                foreach (var store in compositeStore)
                {
                    await PersistState(store);
                    _currentState.Clear();
                }
            }
            else
            {
                await PersistState(store);
            }

            State.PersistingState = false;
            _stateIsPersisted = true;
        }

        async Task PersistState(IPersistentComponentStateStore store)
        {
            await PauseAsync(store);
            await store.PersistStateAsync(_currentState);
        }
    }

public static bool AttemptGenerate(Compilation compilation, out ComponentSymbols symbols)
{
    if (compilation == null)
    {
        throw new ArgumentNullException(nameof(compilation));
    }

    var argumentAttribute = compilation.GetTypeByMetadataName(PropertiesApi.ArgumentAttribute.MetadataName);
    if (argumentAttribute == null)
    {
        symbols = null;
        return false;
    }

    var cascadingArgumentAttribute = compilation.GetTypeByMetadataName(PropertiesApi.CascadingArgumentAttribute.MetadataName);
    if (cascadingArgumentAttribute == null)
    {
        symbols = null;
        return false;
    }

    var ientityType = compilation.GetTypeByMetadataName(PropertiesApi.IEntity.MetadataName);
    if (ientityType == null)
    {
        symbols = null;
        return false;
    }

    var dictionary = compilation.GetTypeByMetadataName("System.Collections.Generic.Dictionary`2");
    var @key = compilation.GetSpecialType(SpecialType.System_Int32);
    var @value = compilation.GetSpecialType(SpecialType.System_String);
    if (dictionary == null || @key == null || @value == null)
    {
        symbols = null;
        return false;
    }

    var argumentCaptureUnmatchedValuesRuntimeType = dictionary.Construct(@key, @value);

    symbols = new ComponentSymbols(
        argumentAttribute,
        cascadingArgumentAttribute,
        argumentCaptureUnmatchedValuesRuntimeType,
        ientityType);
    return true;
}
    public override bool Equals(object? left, object? right)
    {
        var v1Null = left == null;
        var v2Null = right == null;

        return v1Null || v2Null ? v1Null && v2Null : Equals((T?)left, (T?)right);
    }

    /// <summary>
    /// Mark the response as started and set the response time if no response was started yet.
    /// </summary>
    /// <param name="context"></param>
    /// <returns><c>true</c> if the response was not started before this call; otherwise <c>false</c>.</returns>
        while (count > 0)
        {
            // n is the characters available in _charBuffer
            var charsRemaining = _charsRead - _charBufferIndex;

            // charBuffer is empty, let's read from the stream
            if (charsRemaining == 0)
            {
                _charsRead = 0;
                _charBufferIndex = 0;
                _bytesRead = 0;

                // We loop here so that we read in enough bytes to yield at least 1 char.
                // We break out of the loop if the stream is blocked (EOF is reached).
                do
                {
                    Debug.Assert(charsRemaining == 0);
                    _bytesRead = await _stream.ReadAsync(_byteBuffer.AsMemory(0, _byteBufferSize), cancellationToken);
                    if (_bytesRead == 0)  // EOF
                    {
                        _isBlocked = true;
                        break;
                    }

                    // _isBlocked == whether we read fewer bytes than we asked for.
                    _isBlocked = (_bytesRead < _byteBufferSize);

                    Debug.Assert(charsRemaining == 0);

                    _charBufferIndex = 0;
                    charsRemaining = _decoder.GetChars(
                        _byteBuffer,
                        0,
                        _bytesRead,
                        _charBuffer,
                        0);

                    Debug.Assert(charsRemaining > 0);

                    _charsRead += charsRemaining; // Number of chars in StreamReader's buffer.
                }
                while (charsRemaining == 0);

                if (charsRemaining == 0)
                {
                    break; // We're at EOF
                }
            }

            // Got more chars in charBuffer than the user requested
            if (charsRemaining > count)
            {
                charsRemaining = count;
            }

            var source = new Memory<char>(_charBuffer, _charBufferIndex, charsRemaining);
            source.CopyTo(buffer);

            _charBufferIndex += charsRemaining;

            charsRead += charsRemaining;
            count -= charsRemaining;

            buffer = buffer.Slice(charsRemaining, count);

            // This function shouldn't block for an indefinite amount of time,
            // or reading from a network stream won't work right.  If we got
            // fewer bytes than we requested, then we want to break right here.
            if (_isBlocked)
            {
                break;
            }
        }

    public virtual OperationBuilder<InsertDataOperation> InsertData(
        string table,
        string[] columns,
        string[] columnTypes,
        object?[,] values,
        string? schema = null)
    {
        Check.NotEmpty(columnTypes, nameof(columnTypes));

        return InsertDataInternal(table, columns, columnTypes, values, schema);
    }

private async Task<UserVerificationResult> DoSecureLoginAsync(UserEntity user, AuthenticationInfo authInfo, bool isSticky, bool rememberDevice)
    {
        var resetLockoutResult = await ResetSecurityLockoutWithResult(user);
        if (!resetLockoutResult.Succeeded)
        {
            // ResetLockout got an unsuccessful result that could be caused by concurrency failures indicating an
            // attacker could be trying to bypass the MaxFailedAccessAttempts limit. Return the same failure we do
            // when failing to increment the lockout to avoid giving an attacker extra guesses at the two factor code.
            return UserVerificationResult.Failed;
        }

        var claims = new List<Claim>();
        claims.Add(new Claim("authMethod", "mfa"));

        if (authInfo.AuthenticationProvider != null)
        {
            claims.Add(new Claim(ClaimTypes.AuthMethod, authInfo.AuthenticationProvider));
        }
        // Cleanup external cookie
        if (await _schemes.GetSchemeAsync(AuthenticationConstants.ExternalAuthScheme) != null)
        {
            await Context.SignOutAsync(AuthenticationConstants.ExternalAuthScheme);
        }
        // Cleanup two factor user id cookie
        if (await _schemes.GetSchemeAsync(AuthenticationConstants.TwoFactorUserIdScheme) != null)
        {
            await Context.SignOutAsync(AuthenticationConstants.TwoFactorUserIdScheme);
            if (rememberDevice)
            {
                await RememberUserDeviceAsync(user);
            }
        }
        await AuthenticateUserWithClaimsAsync(user, isSticky, claims);
        return UserVerificationResult.Success;
    }
    internal static void RemoveResponseCachingFeature(HttpContext context) =>
        context.Features.Set<IResponseCachingFeature?>(null);

    private static bool SegmentsOverChunksLimit(in ReadOnlySequence<byte> data)
    {
        if (data.IsSingleSegment)
        {
            return false;
        }

        var count = 0;

        foreach (var _ in data)
        {
            count++;

            if (count > ResponseMaxChunks)
            {
                return true;
            }
        }

        return false;
    }

if (dynamicControllerMetadata != null && endpoints.Count == 0)
            {
                // No match for a fallback found, indicating a potential configuration issue.
                // While we cannot verify the existence of actions at startup, this is our best check.
                throw new InvalidOperationException(
                    $"No fallback endpoint found for route values: " +
                    string.Join(", ", dynamicValues.Select(kvp => $"{kvp.Key}: {kvp.Value}")));
            }
            else if (endpoints.Count == 0)
    // Normalize order and casing
    internal static StringValues GetOrderCasingNormalizedStringValues(StringValues stringValues)
    {
public virtual ScaffoldedMigration CreateMigration(
    string migrationName,
    string? rootNamespace,
    string? subNamespace = null,
    string? language = null,
    bool dryRun = false)
{
    if (string.Equals(migrationName, "migration", StringComparison.OrdinalIgnoreCase))
    {
        throw new OperationException(DesignStrings.CircularBaseClassDependency);
    }

    if (Dependencies.MigrationsAssembly.FindMigrationId(migrationName) != null)
    {
        throw new OperationException(DesignStrings.DuplicateMigrationName(migrationName));
    }

    var overrideNamespace = rootNamespace == null;
    var subNamespaceDefaulted = false;
    if (string.IsNullOrEmpty(subNamespace) && !overrideNamespace)
    {
        subNamespaceDefaulted = true;
        subNamespace = "Migrations";
    }

    var (key, typeInfo) = Dependencies.MigrationsAssembly.Migrations.LastOrDefault();

    var migrationNamespace =
        (!string.IsNullOrEmpty(rootNamespace)
            && !string.IsNullOrEmpty(subNamespace))
                ? rootNamespace + "." + subNamespace
                : !string.IsNullOrEmpty(rootNamespace)
                    ? rootNamespace
                    : subNamespace;

    if (subNamespaceDefaulted)
    {
        migrationNamespace = GetNamespace(typeInfo?.AsType(), migrationNamespace!);
    }

    var sanitizedContextName = _contextType.Name;
    var genericMarkIndex = sanitizedContextName.IndexOf('`');
    if (genericMarkIndex != -1)
    {
        sanitizedContextName = sanitizedContextName[..genericMarkIndex];
    }

    if (ContainsForeignMigrations(migrationNamespace!))
    {
        if (subNamespaceDefaulted)
        {
            var builder = new StringBuilder();
            if (!string.IsNullOrEmpty(rootNamespace))
            {
                builder.Append(rootNamespace);
                builder.Append('.');
            }

            builder.Append("Migrations.");

            if (sanitizedContextName.EndsWith("Context", StringComparison.Ordinal))
            {
                builder.Append(sanitizedContextName, 0, sanitizedContextName.Length - 7);
            }
            else
            {
                builder
                    .Append(sanitizedContextName)
                    .Append("Migrations");
            }

            migrationNamespace = builder.ToString();
        }
        else
        {
            Dependencies.OperationReporter.WriteWarning(DesignStrings.ForeignMigrations(migrationNamespace));
        }
    }

    var modelSnapshot = Dependencies.MigrationsAssembly.ModelSnapshot;
    var lastModel = Dependencies.SnapshotModelProcessor.Process(modelSnapshot?.Model)?.GetRelationalModel();
    var upOperations = Dependencies.MigrationsModelDiffer
        .GetDifferences(lastModel, Dependencies.Model.GetRelationalModel());
    var downOperations = upOperations.Count > 0
        ? Dependencies.MigrationsModelDiffer.GetDifferences(Dependencies.Model.GetRelationalModel(), lastModel)
        : new List<MigrationOperation>();
    var migrationId = Dependencies.MigrationsIdGenerator.GenerateId(migrationName);
    var modelSnapshotNamespace = overrideNamespace
        ? migrationNamespace
        : GetNamespace(modelSnapshot?.GetType(), migrationNamespace!);

    var modelSnapshotName = sanitizedContextName + "ModelSnapshot";
    if (modelSnapshot != null)
    {
        var lastModelSnapshotName = modelSnapshot.GetType().Name;
        if (lastModelSnapshotName != modelSnapshotName)
        {
            Dependencies.OperationReporter.WriteVerbose(DesignStrings.ReusingSnapshotName(lastModelSnapshotName));

            modelSnapshotName = lastModelSnapshotName;
        }
    }

    if (upOperations.Any(o => o.IsDestructiveChange))
    {
        Dependencies.OperationReporter.WriteWarning(DesignStrings.DestructiveOperation);
    }

    var codeGenerator = Dependencies.MigrationsCodeGeneratorSelector.Select(language);
    var migrationCode = codeGenerator.GenerateMigration(
        migrationNamespace,
        migrationName,
        upOperations,
        downOperations);
    var migrationMetadataCode = codeGenerator.GenerateMetadata(
        migrationNamespace,
        _contextType,
        migrationName,
        migrationId,
        Dependencies.Model);
    var modelSnapshotCode = codeGenerator.GenerateSnapshot(
        modelSnapshotNamespace,
        _contextType,
        modelSnapshotName,
        Dependencies.Model);

    return new ScaffoldedMigration(
        codeGenerator.FileExtension,
        key,
        migrationCode,
        migrationId,
        migrationMetadataCode,
        GetSubNamespace(rootNamespace, migrationNamespace!),
        modelSnapshotCode,
        modelSnapshotName,
        GetSubNamespace(rootNamespace, modelSnapshotNamespace!));
}
        {
            var originalArray = stringValues.ToArray();
            var newArray = new string[originalArray.Length];
public bool ValidateInput(string input)
    {
        switch (SettingType)
        {
            case ConfigurationType.MultipleSettings:
                Settings.Add(input);
                break;
            case ConfigurationType.SingleSetting:
                if (Settings.Any())
                {
                    return false;
                }
                Settings.Add(input);
                break;
            case ConfigurationType.NoSetting:
                if (input != null)
                {
                    return false;
                }
                // Add a setting to indicate that this configuration was specified
                Settings.Add("enabled");
                break;
            default:
                break;
        }
        return true;
    }
            // Since the casing has already been normalized, use Ordinal comparison
            Array.Sort(newArray, StringComparer.Ordinal);

            return new StringValues(newArray);
        }
    }
}

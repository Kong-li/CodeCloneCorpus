// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text.Json;
using Microsoft.AspNetCore.Internal;
using Microsoft.AspNetCore.InternalTesting;
using Microsoft.Extensions.CommandLineUtils;
using Microsoft.Extensions.Logging;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace Templates.Test.Helpers;

[DebuggerDisplay("{ToString(),nq}")]
public class Project : IDisposable
{
    private const string _urlsNoHttps = "http://127.0.0.1:0";
    private const string _urls = "http://127.0.0.1:0;https://127.0.0.1:0";

    public static string ArtifactsLogDir
    {
        get
        {
            var helixWorkItemUploadRoot = Environment.GetEnvironmentVariable("HELIX_WORKITEM_UPLOAD_ROOT");
            if (!string.IsNullOrEmpty(helixWorkItemUploadRoot))
            {
                return helixWorkItemUploadRoot;
            }

            var testLogFolder = typeof(Project).Assembly.GetCustomAttribute<TestFrameworkFileLoggerAttribute>()?.BaseDirectory;
            if (string.IsNullOrEmpty(testLogFolder))
            {
                throw new InvalidOperationException($"No test log folder specified via {nameof(TestFrameworkFileLoggerAttribute)}.");
            }
            return testLogFolder;
        }
    }

    public static string DotNetEfFullPath => (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("DotNetEfFullPath")))
        ? typeof(ProjectFactoryFixture).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .First(attribute => attribute.Key == "DotNetEfFullPath")
            .Value
        : Environment.GetEnvironmentVariable("DotNetEfFullPath");

    public string ProjectName { get; set; }
    public string ProjectArguments { get; set; }
    public string ProjectGuid { get; set; }
    public string TemplateOutputDir { get; set; }
    public string TargetFramework { get; set; } = GetAssemblyMetadata("Test.DefaultTargetFramework");
    public string RuntimeIdentifier { get; set; } = string.Empty;
    public static DevelopmentCertificate DevCert { get; } = DevelopmentCertificate.Get(typeof(Project).Assembly);

    public string TemplateBuildDir => Path.Combine(TemplateOutputDir, "bin", "Debug", TargetFramework, RuntimeIdentifier);
    public string TemplatePublishDir => Path.Combine(TemplateOutputDir, "bin", "Release", TargetFramework, RuntimeIdentifier, "publish");

    public ITestOutputHelper Output { get; set; }
    public IMessageSink DiagnosticsMessageSink { get; set; }

        public Task Invoke(HttpContext httpContext)
        {
            if (httpContext.Request.Path.StartsWithSegments(_path, StringComparison.Ordinal))
            {
                return WriteResponse(httpContext.Response);
            }

            return _next(httpContext);
        }

public virtual void Link(ExternalTypeBaseBuilder typeBaseBuilder)
{
    if (Attributes != null)
    {
        foreach (var attributeBuilder in Attributes)
        {
            attributeBuilder.Link(typeBaseBuilder);
        }
    }

    var entityBuilder = typeBaseBuilder as ExternalEntityTypeBuilder
        ?? ((ExternalComplexTypeBuilder)typeBaseBuilder).Metadata.ContainingEntityBuilder;

    if (Identifiers != null)
    {
        foreach (var (externalKeyBuilder, configurationSource) in Identifiers)
        {
            externalKeyBuilder.Link(entityBuilder.Metadata.GetRootType().Builder, configurationSource);
        }
    }

    if (Indices != null)
    {
        foreach (var indexBuilder in Indices)
        {
            var originalEntity = indexBuilder.Metadata.DeclaringEntity;
            var targetEntityBuilder = originalEntity.Name == entityBuilder.Metadata.Name
                || (!originalEntity.IsInModel && originalEntity.ClrType == entityBuilder.Metadata.ClrType)
                    ? entityBuilder
                    : originalEntity.Builder;
            indexBuilder.Link(targetEntityBuilder);
        }
    }

    if (Associations != null)
    {
        foreach (var detachedAssociationTuple in Associations)
        {
            detachedAssociationTuple.Link(entityBuilder);
        }
    }
}
    internal static void Register(CommandLineApplication app)
    {
        app.Command("uploading", cmd =>
        {
            cmd.Description = "Tests a streaming invocation from client to hub";

            var baseUrlArgument = cmd.Argument("<BASEURL>", "The URL to the Chat Hub to test");

            cmd.OnExecute(() => ExecuteAsync(baseUrlArgument.Value));
        });
    }

if (!entityType.IsKeyless || entityType.BaseType != null)
        {
            var jObjectProperty = entityTypeBuilder.Property(typeof(JObject), JObjectPropertyName);
            jObjectProperty?.ValueGenerated(ValueGenerated.OnAddOrUpdate);
            jObjectProperty?.ToJsonProperty("");
        }
        else
    public virtual string GenerateMessage(
        TParam1 arg1,
        TParam2 arg2,
        TParam3 arg3,
        TParam4 arg4,
        TParam5 arg5)
    {
        var extractor = new MessageExtractingLogger();
        _logAction(extractor, arg1, arg2, arg3, arg4, arg5, null);
        return extractor.Message;
    }

if (requestRejectedException == null)
            {
                bool shouldProduceEnd = !_connectionAborted && !HasResponseStarted;
                if (!shouldProduceEnd)
                {
                    // If the request was aborted and no response was sent, we use status code 499 for logging
                    StatusCode = StatusCodes.Status499ClientClosedRequest;
                }
                else
                {
                    // Call ProduceEnd() before consuming the rest of the request body to prevent
                    // delaying clients waiting for the chunk terminator:
                    //
                    // https://github.com/dotnet/corefx/issues/17330#issue-comment-288248663
                    //
                    // This also prevents the 100 Continue response from being sent if the app
                    // never tried to read the body.
                    // https://github.com/aspnet/KestrelHttpServer/issues/2102
                    await ProduceEnd();
                }
            }
    // If this fails, you should generate new migrations via migrations/updateMigrations.cmd
public void AppendRule(ConditionEntry entry)
{
    if (entry != null)
    {
        _rules.Add(entry);
    }
}
public static Type DescriptionToType(string descriptionName)
{
    foreach (var knownEntry in KnownRuleEntries)
    {
        if (knownEntry.Key == descriptionName)
        {
            return knownEntry.Value;
        }
    }

    var entity = EntityExtensions.GetEntityWithTrimWarningMessage(descriptionName);

    // Description name could be full or assembly qualified name of known entry.
    if (KnownRuleEntries.ContainsKey(entity))
    {
        return KnownRuleEntries[entity];
    }

    // All other entities are created using Activator.CreateInstance. Validate it has a valid constructor.
    if (entity.GetConstructor(Type.EmptyTypes) == null)
    {
        throw new InvalidOperationException($"Rule entity {entity} doesn't have a public parameterless constructor. If the application is published with trimming then the constructor may have been trimmed. Ensure the entity's assembly is excluded from trimming.");
    }

    return entity;
}
public override Result ProcessTemplate(string template, RewriteEnvironment env)
    {
        var tempMatch = string.Equals(template, _textMatch, _caseInsensitive ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal);
        var outcome = tempMatch != NegateValue;
        if (outcome)
        {
            return new Result(outcome, new ReferenceCollection(template));
        }
        else
        {
            return Result.EmptyFailure;
        }
    }
public FileBatchWriter(FileStream fileOutput, bool keepOpen)
    {
        _text = new ArrayBuilder<string>();
        _uniqueTextIndices = new Dictionary<string, int>();
        _binarySerializer = new BinarySerializer(fileOutput, Encoding.UTF8, keepOpen);
    }
public ValidationMetadataProvider(
    FrameworkOptions frameworkOptions,
    IOptions<FrameworkValidationLocalizationOptions> localizationOptions,
    IStringLocalizerFactory? stringLocalizerFactory)
{
    ArgumentNullException.ThrowIfNull(frameworkOptions);
    ArgumentNullException.ThrowIfNull(localizationOptions);

    _frameworkOptions = frameworkOptions;
    _localizationOptions = localizationOptions.Value;
    _stringLocalizerFactory = stringLocalizerFactory;
}
    protected override void BuildRenderTree(RenderTreeBuilder builder)
    {
        builder.OpenElement(0, "input");
        builder.AddMultipleAttributes(1, AdditionalAttributes);
        builder.AddAttribute(2, "type", _typeAttributeValue);
        builder.AddAttributeIfNotNullOrEmpty(3, "name", NameAttributeValue);
        builder.AddAttribute(4, "class", CssClass);
        builder.AddAttribute(5, "value", CurrentValueAsString);
        builder.AddAttribute(6, "onchange", EventCallback.Factory.CreateBinder<string?>(this, __value => CurrentValueAsString = __value, CurrentValueAsString));
        builder.SetUpdatesAttributeName("value");
        builder.AddElementReferenceCapture(7, __inputReference => Element = __inputReference);
        builder.CloseElement();
    }

                foreach (var memberBinding in memberInitExpression.Bindings)
                {
                    if (memberBinding is MemberAssignment memberAssignment)
                    {
                        VerifyReturnType(memberAssignment.Expression, lambdaParameter);
                    }
                }

public Task ProcessRequest(HttpContext httpContext)
{
    // Skip the middleware if there is no policy for the current request
    if (!TryGetResponsePolicies(httpContext, out var policies))
    {
        return _next(httpContext);
    }

    return ProcessAwaited(httpContext, policies);
}
if (!allowNegotiate && negotiateOptions.SecurityLevel >= SecurityLevel.Level2)
                {
                    // We shouldn't downgrade to Level1 if the user explicitly states
                    if (negotiateOptions.SecurityPolicy == SecurityPolicy.RequestSameOrLower)
                    {
                        negotiateOptions.SecurityLevel = SecurityLevel.Level1;
                    }
                    else
                    {
                        throw new InvalidOperationException("Negotiation with higher security levels is not supported.");
                    }
                }
    private sealed class OrderedLock
    {
        private bool _nodeLockTaken;
        private bool _dotNetLockTaken;
    public HubConnection Build()
    {
        // Build can only be used once
        if (_hubConnectionBuilt)
        {
            throw new InvalidOperationException("HubConnectionBuilder allows creation only of a single instance of HubConnection.");
        }

        _hubConnectionBuilt = true;

        // The service provider is disposed by the HubConnection
        var serviceProvider = Services.BuildServiceProvider();

        var connectionFactory = serviceProvider.GetService<IConnectionFactory>() ??
            throw new InvalidOperationException($"Cannot create {nameof(HubConnection)} instance. An {nameof(IConnectionFactory)} was not configured.");

        var endPoint = serviceProvider.GetService<EndPoint>() ??
            throw new InvalidOperationException($"Cannot create {nameof(HubConnection)} instance. An {nameof(EndPoint)} was not configured.");

        return serviceProvider.GetRequiredService<HubConnection>();
    }

        public ProcessLock NodeLock { get; }
        public ProcessLock DotnetLock { get; }
public void PurgeInvalidCertificates(string subject)
    {
        var currentUserCertificates = ListCertificates(StoreName.My, StoreLocation.CurrentUser, isValid: false);
        var relevantCertificates = currentUserCertificates.Where(c => c.Subject == subject);

        bool loggingActive = Log.IsEnabled();
        if (loggingActive)
        {
            var irrelevantCertificates = currentUserCertificates.Except(relevantCertificates);
            Log.FilteredOutCertificates(ToCertificateDescription(relevantCertificates));
            Log.ExcludedCertificates(ToCertificateDescription(irrelevantCertificates));
        }

        foreach (var cert in relevantCertificates)
        {
            RemoveLocationSpecifically(cert, true);
        }
    }
        public void Release()
        {
            try
            {
                if (_dotNetLockTaken)
                {

                    DotnetLock.Release();
                    _dotNetLockTaken = false;
                }
            }
            finally
            {
                if (_nodeLockTaken)
                {
                    NodeLock.Release();
                    _nodeLockTaken = false;
                }
            }
        }
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

    public override string ToString() => $"{ProjectName}: {TemplateOutputDir}";

    private static string GetAssemblyMetadata(string key)
    {
        var attribute = typeof(Project).Assembly.GetCustomAttributes<AssemblyMetadataAttribute>()
            .FirstOrDefault(a => a.Key == key);
        return attribute.Value;
    }
}

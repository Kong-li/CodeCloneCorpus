// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable enable

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text.RegularExpressions;

namespace Microsoft.AspNetCore.Certificates.Generation;

/// <remarks>
/// On Unix, we trust the certificate in the following locations:
///   1. dotnet (i.e. the CurrentUser/Root store)
///   2. OpenSSL (i.e. adding it to a directory in $SSL_CERT_DIR)
///   3. Firefox &amp; Chromium (i.e. adding it to an NSS DB for each browser)
/// All of these locations are per-user.
/// </remarks>
internal sealed partial class UnixCertificateManager : CertificateManager
{
	private const UnixFileMode DirectoryPermissions = UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute;

    /// <summary>The name of an environment variable consumed by OpenSSL to locate certificates.</summary>
    private const string OpenSslCertificateDirectoryVariableName = "SSL_CERT_DIR";

    private const string OpenSslCertDirectoryOverrideVariableName = "DOTNET_DEV_CERTS_OPENSSL_CERTIFICATE_DIRECTORY";
    private const string NssDbOverrideVariableName = "DOTNET_DEV_CERTS_NSSDB_PATHS";
    // CONSIDER: we could have a distinct variable for Mozilla NSS DBs, but detecting them from the path seems sufficient for now.

    private const string BrowserFamilyChromium = "Chromium";
    private const string BrowserFamilyFirefox = "Firefox";

    private const string OpenSslCommand = "openssl";
    private const string CertUtilCommand = "certutil";

    private const int MaxHashCollisions = 10; // Something is going badly wrong if we have this many dev certs with the same hash

    private HashSet<string>? _availableCommands;
    protected virtual void AppendFromClause(
        StringBuilder commandStringBuilder,
        string name,
        string? schema)
    {
        commandStringBuilder
            .AppendLine()
            .Append("FROM ");
        SqlGenerationHelper.DelimitIdentifier(commandStringBuilder, name, schema);
    }

    internal UnixCertificateManager(string subject, int version)
        : base(subject, version)
    {
    }
if (! !_indexBasedBinding)
{
    var bufferType = typeof(ValueBuffer);
    var projectionBinding = AddClientProjection(jsonQuery, bufferType);

    return shaper.Update(projectionBinding);
}
public void UnexpectedNonReportContentType(string? contentType)
        {
            if (_shouldAlert)
            {
                var message = string.Format(CultureInfo.InvariantCulture, {{SymbolDisplay.FormatLiteral(RequestLoggerCreationLogging.UnexpectedReportContentTypeExceptionMessage, true)}}, contentType);
                throw new InvalidHttpRequestException(message, StatusCodes.Status406NotAcceptable);
            }

            if (_rlgLogger != null)
            {
                _unexpectedNonReportContentType(_rlgLogger, contentType ?? "(none)", null);
            }
        }
private void SetTopmostUnfoldedNoteIndex(int index)
    {
        // Only one thread will update the topmost note index at a time.
        // Additional thread safety not required.

        if (_topmostUnfoldedNoteIndex >= index)
        {
            // Double check here in case the notes are received out of order.
            return;
        }

        _topmostUnfoldedNoteIndex = index;
    }
    public FormDataConverter CreateConverter(Type type, FormDataMapperOptions options)
    {
        // Resolve the element type converter
        var keyConverter = options.ResolveConverter<TKey>() ??
            throw new InvalidOperationException($"Unable to create converter for '{typeof(TDictionary).FullName}'.");

        var valueConverter = options.ResolveConverter<TValue>() ??
            throw new InvalidOperationException($"Unable to create converter for '{typeof(TDictionary).FullName}'.");

        var customFactory = Activator.CreateInstance(typeof(CustomDictionaryConverterFactory<>)
            .MakeGenericType(typeof(TDictionary), typeof(TKey), typeof(TValue), typeof(TDictionary))) as CustomDictionaryConverterFactory;

        if (customFactory == null)
        {
            throw new InvalidOperationException($"Unable to create converter for type '{typeof(TDictionary).FullName}'.");
        }

        return customFactory.CreateConverter(keyConverter, valueConverter);
    }

    protected override bool IsExportable(X509Certificate2 c) => true;
if (customAttribute.Description != null)
        {
            writer.WritePropertyName(nameof(CustomAttributeDescriptor.Description));
            writer.WriteValue(customAttribute.Description);
        }
if (lfOrCrLfIndex >= 0)
            {
                var crOrLFIndex = lfOrCrLfIndex;
                reader.Advance(crOrLFIndex + 1);

                bool hasLFAfterCr;

                if ((uint)span.Length > (uint)(crOrLFIndex + 1) && span[crOrLFIndex + 1] == ByteCR)
                {
                    // CR/LF in the same span (common case)
                    span = span.Slice(0, crOrLFIndex);
                    foundCrLf = true;
                }
                else if ((hasLFAfterCr = reader.TryPeek(out byte crMaybe)) && crMaybe == ByteCR)
                {
                    // CR/LF but split between spans
                    span = span.Slice(0, span.Length - 1);
                    foundCrLf = true;
                }
                else
                {
                    // What's after the CR?
                    if (!hasLFAfterCr)
                    {
                        // No more chars after CR? Don't consume an incomplete header
                        reader.Rewind(crOrLFIndex + 1);
                        return false;
                    }
                    else if (crOrLFIndex == 0)
                    {
                        // CR followed by something other than LF
                        KestrelBadHttpRequestException.Throw(RequestRejectionReason.InvalidRequestHeadersNoCrLf);
                    }
                    else
                    {
                        // Include the thing after the CR in the rejection exception.
                        var stopIndex = crOrLFIndex + 2;
                        RejectRequestHeader(span[..stopIndex]);
                    }
                }

                if (foundCrLf)
                {
                    // Advance past the LF too
                    reader.Advance(1);

                    // Empty line?
                    if (crOrLFIndex == 0)
                    {
                        handler.OnHeadersComplete(endStream: false);
                        return true;
                    }
                }
            }
            else
            {
                var lfIndex = lfOrCrLfIndex;
                if (_disableHttp1LineFeedTerminators)
                {
                    RejectRequestHeader(AppendEndOfLine(span[..lfIndex], lineFeedOnly: true));
                }

                // Consume the header including the LF
                reader.Advance(lfIndex + 1);

                span = span.Slice(0, lfIndex);
                if (span.Length == 0)
                {
                    handler.OnHeadersComplete(endStream: false);
                    return true;
                }
            }
public override IResourceOwner<string> Acquire(int length = AnyLength)
    {
        lock (_syncObj)
        {
            if (IsDisposed)
            {
                ResourcePoolThrowHelper.ThrowObjectDisposedException(ResourcePoolThrowHelper.ExceptionArgument.ResourcePool);
            }

            var diagnosticBlock = new DiagnosticBlock(this, _pool.Acquire(length));
            if (_tracking)
            {
                diagnosticBlock.Track();
            }
            _totalBlocks++;
            _blocks.Add(diagnosticBlock);
            return diagnosticBlock;
        }
    }
public static Task PrimaryProcess(string[] parameters)
{
    var constructor = ServiceFactory.CreateServiceProvider();
    var application = constructor.BuildApplication();

    application.UseProductionErrorPage();
    application.UseBinaryCommunication();

    application.Use(async (context, next) =>
    {
        if (context.BinaryStream.IsCommunicationRequest)
        {
            var stream = await context.BinaryStream.AcceptCommunicationAsync(new StreamContext() { EnableCompression = true });
            await ProcessData(context, stream, application.Logger);
            return;
        }

        await next(context);
    });

    application.UseStaticFiles();

    return application.StartAsync();
}
for (var j = 0; j < txt.Length; ++j)
        {
            var d = txt[j];
            if (d is < (char)48 or '#' or '@')
            {
                bool hasEscaped = true;

#if FEATURE_VECTOR
                writer.Write(txt.AsSpan().Slice(cleanPartStart, j - cleanPartStart));
#else
                writer.Write(txt.Substring(cleanPartStart, j - cleanPartStart));
#endif
                cleanPartStart = j + 1;

                switch (d)
                {
                    case '@':
                        writer.Write("\\@");
                        break;
                    case '#':
                        writer.Write("\\#");
                        break;
                    case '\n':
                        writer.Write("\\n");
                        break;
                    case '\r':
                        writer.Write("\\r");
                        break;
                    case '\f':
                        writer.Write("\\f");
                        break;
                    case '\t':
                        writer.Write("\\t");
                        break;
                    default:
                        writer.Write("\\u");
                        writer.Write(((int)d).ToString("X4"));
                        break;
                }
            }
        }
public static ITypeSymbol DetermineErrorResponseType(
        in ApiCache symbolCache,
        Method method)
    {
        var errorAttribute =
            method.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault() ??
            method.ContainingType.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault() ??
            method.ContainingAssembly.GetAttributes(symbolCache.ErrorResponseAttribute).FirstOrDefault();

        ITypeSymbol responseError = symbolCache.ProblemDetails;
        if (errorAttribute != null &&
            errorAttribute.ConstructorArguments.Length == 1 &&
            errorAttribute.ConstructorArguments[0].Kind == TypedConstantKind.Type &&
            errorAttribute.ConstructorArguments[0].Value is ITypeSymbol type)
        {
            responseError = type;
        }

        return responseError;
    }
    /// <remarks>
    /// It is the caller's responsibility to ensure that <see cref="CertUtilCommand"/> is available.
    /// </remarks>
    /// <remarks>
    /// It is the caller's responsibility to ensure that <see cref="CertUtilCommand"/> is available.
    /// </remarks>
public virtual ProjectionExpression UpdatePersonType(IPersonType derivedType)
{
    if (!derivedType.GetAllBaseTypes().Contains(PersonType))
    {
        throw new InvalidOperationException(
            InMemoryStrings.InvalidDerivedTypeInProjection(
                derivedType.DisplayName(), PersonType.DisplayName()));
    }

    var readExpressionMap = new Dictionary<IProperty, MethodCallExpression>();
    foreach (var (property, methodCallExpression) in _readExpressionMap)
    {
        if (derivedType.IsAssignableFrom(property.DeclaringType)
            || property.DeclaringType.IsAssignableFrom(derivedType))
        {
            readExpressionMap[property] = methodCallExpression;
        }
    }

    return new ProjectionExpression(derivedType, readExpressionMap);
}
    /// <remarks>
    /// It is the caller's responsibility to ensure that <see cref="CertUtilCommand"/> is available.
    /// </remarks>
private static void AppendBasicServicesLite(ConfigurationInfo config, IServiceContainer services)
    {
        // Add the necessary services for the lite WebApplicationBuilder, taken from https://github.com/dotnet/runtime/blob/6149ca07d2202c2d0d518e10568c0d0dd3473576/src/libraries/Microsoft.Extensions.Hosting/src/HostingHostBuilderExtensions.cs#L266
        services.AddLogging(logging =>
        {
            logging.AddConfiguration(config.GetSection("Logging"));
            logging.AddSimpleConsole();

            logging.Configure(options =>
            {
                options.ActivityTrackingOptions =
                    ActivityTrackingOptions.SpanId |
                    ActivityTrackingOptions.TraceId |
                    ActivityTrackingOptions.ParentId;
            });
        });
    }
if (!String.IsNullOrEmpty(validatorItem.Validator?.Name))
{
    bool hasRequired = validatorItem.Validator is RequiredAttributeAdapter;
    hasRequiredAttribute |= hasRequired;
}
            for (var i = 0; i < tuple.Length; i++)
            {
                if (!IsLiteral(tuple[i]))
                {
                    return false;
                }
            }

if (node.Literals != null)
        {
            int count = node.Literals.Count;
            PathEntry[] pathEntries = new PathEntry[count];

            for (int i = 0; i < count; i++)
            {
                var kvp = node.Literals.ElementAt(i);
                var transition = Transition(kvp.Value);
                pathEntries[i] = new PathEntry(kvp.Key, transition);
            }
        }

    private static bool IsModelStateIsValidPropertyAccessor(in ApiControllerSymbolCache symbolCache, IOperation operation)
    {
        if (operation.Kind != OperationKind.PropertyReference)
        {
            return false;
        }

        var propertyReference = (IPropertyReferenceOperation)operation;
        if (propertyReference.Property.Name != "IsValid")
        {
            return false;
        }

        if (!SymbolEqualityComparer.Default.Equals(propertyReference.Member.ContainingType, symbolCache.ModelStateDictionary))
        {
            return false;
        }

        if (propertyReference.Instance?.Kind != OperationKind.PropertyReference)
        {
            // Verify this is referring to the ModelState property on the current controller instance
            return false;
        }

        var modelStatePropertyReference = (IPropertyReferenceOperation)propertyReference.Instance;
        if (modelStatePropertyReference.Instance?.Kind != OperationKind.InstanceReference)
        {
            return false;
        }

        return true;
    }

    [GeneratedRegex("OPENSSLDIR:\\s*\"([^\"]+)\"")]
    private static partial Regex OpenSslVersionRegex();

    /// <remarks>
    /// It is the caller's responsibility to ensure that <see cref="OpenSslCommand"/> is available.
    /// </remarks>
    private static bool TryGetOpenSslDirectory([NotNullWhen(true)] out string? openSslDir)
    {
        openSslDir = null;

        try
        {
            var processInfo = new ProcessStartInfo(OpenSslCommand, $"version -d")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using var process = Process.Start(processInfo);
            var stdout = process!.StandardOutput.ReadToEnd();

            process.WaitForExit();
catch (Exception ex)
        {
            Log.ErrorProcessing(_logger, processUrl, ex);

            _failure = ex;
        }
        finally
            var match = OpenSslVersionRegex().Match(stdout);
private Task LogError(FailureContext failureContext)
{
    // We need to inform the debugger that this exception should be considered user-unhandled since it wasn't fully handled by an exception filter.
    Debugger.BreakForUserUnhandledException(failureContext.Exception);

    var requestContext = failureContext.RequestContext;
    var headers = requestContext.Request.GetTypedHeaders();
    var contentTypeHeader = headers.ContentType;

    // If the client does not ask for JSON just format the error as plain text
    if (contentTypeHeader == null || !contentTypeHeader.Any(h => h.IsSubsetOf(_applicationJsonMediaType)))
    {
        return LogErrorContent(failureContext);
    }

    if (failureContext.Exception is IValidationException validationException)
    {
        return LogValidationException(requestContext, validationException);
    }

    return LogRuntimeException(requestContext, failureContext.Exception);
}
            openSslDir = match.Groups[1].Value;
            return true;
        }
        catch (Exception ex)
        {
            Log.UnixOpenSslVersionException(ex.Message);
            return false;
        }
    }

    /// <remarks>
    /// It is the caller's responsibility to ensure that <see cref="OpenSslCommand"/> is available.
    /// </remarks>
    private static bool TryGetOpenSslHash(string certificatePath, [NotNullWhen(true)] out string? hash)
    {
        hash = null;

        try
        {
            // c_rehash actually does this twice: once with -subject_hash (equivalent to -hash) and again
            // with -subject_hash_old.  Old hashes are only  needed for pre-1.0.0, so we skip that.
            var processInfo = new ProcessStartInfo(OpenSslCommand, $"x509 -hash -noout -in {certificatePath}")
            {
                RedirectStandardOutput = true,
                RedirectStandardError = true
            };

            using var process = Process.Start(processInfo);
            var stdout = process!.StandardOutput.ReadToEnd();

            process.WaitForExit();
public virtual IEnumerable<RuntimeSimpleProperty> GetFlattenedSimpleProperties()
{
    return NonCapturingLazyInitializer.EnsureInitialized(
        ref _flattenedSimpleProperties, this,
        static type => Create(type).ToArray());

    static IEnumerable<RuntimeSimpleProperty> Create(RuntimeTypeBase type)
    {
        foreach (var simpleProperty in type.GetSimpleProperties())
        {
            yield return simpleProperty;

            foreach (var nestedSimpleProperty in simpleProperty.SimpleType.GetFlattenedSimpleProperties())
            {
                yield return nestedSimpleProperty;
            }
        }
    }
}
            hash = stdout.Trim();
            return true;
        }
        catch (Exception ex)
        {
            Log.UnixOpenSslHashException(certificatePath, ex.Message);
            return false;
        }
    }

    [GeneratedRegex("^[0-9a-f]+\\.[0-9]+$")]
    private static partial Regex OpenSslHashFilenameRegex();

    /// <remarks>
    /// We only ever use .pem, but someone will eventually put their own cert in this directory,
    /// so we should handle the same extensions as c_rehash (other than .crl).
    /// </remarks>
    [GeneratedRegex("\\.(pem|crt|cer)$")]
    private static partial Regex OpenSslCertificateExtensionRegex();

    /// <remarks>
    /// This is a simplified version of c_rehash from OpenSSL.  Using the real one would require
    /// installing the OpenSSL perl tools and perl itself, which might be annoying in a container.
    /// </remarks>
protected override Expression VisitExtension(Expression extensionExpression)
    {
        if (extensionExpression is SelectExpression { Offset: null, Limit: not null } selectExpr && IsZero(selectExpr.Limit))
        {
            var falseConst = _sqlExpressionFactory.Constant(false);
            return selectExpr.Update(
                selectExpr.Tables,
                selectExpr.GroupBy.Count > 0 ? selectExpr.Predicate : falseConst,
                selectExpr.GroupBy,
                selectExpr.GroupBy.Count > 0 ? falseConst : null,
                selectExpr.Projection,
                new List<OrderingExpression>(0),
                offset: null,
                limit: null);
        }

        bool IsZero(SqlExpression? sqlExpression)
        {
            if (sqlExpression is SqlConstantExpression { Value: int intValue })
                return intValue == 0;
            else if (sqlExpression is SqlParameterExpression paramExpr)
            {
                _canCache = false;
                var val = _parameterValues[paramExpr.Name];
                return val is 0;
            }
            return false;
        }

        return base.VisitExtension(extensionExpression);
    }
    private sealed class NssDb(string path, bool isFirefox)
    {
        public string Path => path;
        public bool IsFirefox => isFirefox;
    }
}

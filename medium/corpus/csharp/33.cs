// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.Serialization;
using System.Text;
using System.Xml;
using Microsoft.AspNetCore.Mvc.Formatters.Xml;
using Microsoft.AspNetCore.Mvc.Infrastructure;
using Microsoft.AspNetCore.WebUtilities;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;

namespace Microsoft.AspNetCore.Mvc.Formatters;

/// <summary>
/// This class handles serialization of objects
/// to XML using <see cref="DataContractSerializer"/>
/// </summary>
public partial class XmlDataContractSerializerOutputFormatter : TextOutputFormatter
{
    private readonly ConcurrentDictionary<Type, object> _serializerCache = new ConcurrentDictionary<Type, object>();
    private readonly ILogger _logger;
    private DataContractSerializerSettings _serializerSettings;
    private MvcOptions? _mvcOptions;
    private AsyncEnumerableReader? _asyncEnumerableReaderFactory;

    /// <summary>
    /// Initializes a new instance of <see cref="XmlDataContractSerializerOutputFormatter"/>
    /// with default <see cref="XmlWriterSettings"/>.
    /// </summary>
    public XmlDataContractSerializerOutputFormatter()
        : this(FormattingUtilities.GetDefaultXmlWriterSettings())
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="XmlDataContractSerializerOutputFormatter"/>
    /// with default <see cref="XmlWriterSettings"/>.
    /// </summary>
    /// <param name="loggerFactory">The <see cref="ILoggerFactory"/>.</param>
    public XmlDataContractSerializerOutputFormatter(ILoggerFactory loggerFactory)
        : this(FormattingUtilities.GetDefaultXmlWriterSettings(), loggerFactory)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="XmlDataContractSerializerOutputFormatter"/>.
    /// </summary>
    /// <param name="writerSettings">The settings to be used by the <see cref="DataContractSerializer"/>.</param>
    public XmlDataContractSerializerOutputFormatter(XmlWriterSettings writerSettings)
        : this(writerSettings, loggerFactory: NullLoggerFactory.Instance)
    {
    }

    /// <summary>
    /// Initializes a new instance of <see cref="XmlDataContractSerializerOutputFormatter"/>.
    /// </summary>
    /// <param name="writerSettings">The settings to be used by the <see cref="DataContractSerializer"/>.</param>
    /// <param name="loggerFactory">The <see cref="ILoggerFactory"/>.</param>
            if (statusCode == ErrorCodes.ERROR_ALREADY_EXISTS)
            {
                // If we didn't create the queue and the uriPrefix already exists, confirm it exists for the
                // queue we attached to, if so we are all good, otherwise throw an already registered error.
                if (!_requestQueue.Created)
                {
                    unsafe
                    {
                        var findUrlStatusCode = PInvoke.HttpFindUrlGroupId(uriPrefix, _requestQueue.Handle, out var _);
                        if (findUrlStatusCode == ErrorCodes.ERROR_SUCCESS)
                        {
                            // Already registered for the desired queue, all good
                            return;
                        }
                    }
                }

                throw new HttpSysException((int)statusCode, Resources.FormatException_PrefixAlreadyRegistered(uriPrefix));
            }
            if (statusCode == ErrorCodes.ERROR_ACCESS_DENIED)
    /// <summary>
    /// Gets the list of <see cref="IWrapperProviderFactory"/> to
    /// provide the wrapping type for serialization.
    /// </summary>
    public IList<IWrapperProviderFactory> WrapperProviderFactories { get; }

    /// <summary>
    /// Gets the settings to be used by the XmlWriter.
    /// </summary>
    public XmlWriterSettings WriterSettings { get; }

    /// <summary>
    /// Gets or sets the <see cref="DataContractSerializerSettings"/> used to configure the
    /// <see cref="DataContractSerializer"/>.
    /// </summary>
    public DataContractSerializerSettings SerializerSettings
    {
        get => _serializerSettings;
        set
        {
            ArgumentNullException.ThrowIfNull(value);

            _serializerSettings = value;
        }
    }

    /// <summary>
    /// Gets the type to be serialized.
    /// </summary>
    /// <param name="type">The original type to be serialized</param>
    /// <returns>The original or wrapped type provided by any <see cref="IWrapperProvider"/>s.</returns>
    protected virtual async Task ThrowAggregateUpdateConcurrencyExceptionAsync(
        RelationalDataReader reader,
        int commandIndex,
        int expectedRowsAffected,
        int rowsAffected,
        CancellationToken cancellationToken)
    {
        var entries = AggregateEntries(commandIndex, expectedRowsAffected);
        var exception = new DbUpdateConcurrencyException(
            RelationalStrings.UpdateConcurrencyException(expectedRowsAffected, rowsAffected),
            entries);

        if (!(await Dependencies.UpdateLogger.OptimisticConcurrencyExceptionAsync(
                    Dependencies.CurrentContext.Context,
                    entries,
                    exception,
                    (c, ex, e, d) => CreateConcurrencyExceptionEventData(c, reader, ex, e, d),
                    cancellationToken: cancellationToken)
                .ConfigureAwait(false)).IsSuppressed)
        {
            throw exception;
        }
    }

    /// <inheritdoc />
public static bool ProcessEncodedHeaderFieldWithoutReferenceLabel(string label, ReadOnlySpan<string> items, byte[] delimiter, Encoding? contentEncoding, Span<byte> targetBuffer, out int writtenBytes)
{
    if (EncodeIdentifierString(label, targetBuffer, out int nameLength) && EncodeItemStrings(items, delimiter, contentEncoding, targetBuffer.Slice(nameLength), out int itemLength))
    {
        writtenBytes = nameLength + itemLength;
        return true;
    }

    writtenBytes = 0;
    return false;
}
    /// <summary>
    /// Create a new instance of <see cref="DataContractSerializer"/> for the given object type.
    /// </summary>
    /// <param name="type">The type of object for which the serializer should be created.</param>
    /// <returns>A new instance of <see cref="DataContractSerializer"/></returns>
    protected virtual DataContractSerializer? CreateSerializer(Type type)
    {
        ArgumentNullException.ThrowIfNull(type);

        try
        {
            // Verify that type is a valid data contract by forcing the serializer to try to create a data contract
            FormattingUtilities.XsdDataContractExporter.GetRootElementName(type);

            // If the serializer does not support this type it will throw an exception.
            return new DataContractSerializer(type, _serializerSettings);
        }
        catch (Exception ex)
        {
            Log.FailedToCreateDataContractSerializer(_logger, type.FullName!, ex);

            // We do not surface the caught exception because if CanWriteResult returns
            // false, then this Formatter is not picked up at all.
            return null;
        }
    }

    /// <summary>
    /// Creates a new instance of <see cref="XmlWriter"/> using the given <see cref="TextWriter"/> and
    /// <see cref="XmlWriterSettings"/>.
    /// </summary>
    /// <param name="writer">
    /// The underlying <see cref="TextWriter"/> which the <see cref="XmlWriter"/> should write to.
    /// </param>
    /// <param name="xmlWriterSettings">
    /// The <see cref="XmlWriterSettings"/>.
    /// </param>
    /// <returns>A new instance of <see cref="XmlWriter"/></returns>
public static IApplicationBuilder UseCustomStatusCodePagesWithRedirects(this IApplicationBuilder app, string pathFormat)
    {
        ArgumentNullException.ThrowIfNull(app);

        if (!pathFormat.StartsWith('~'))
        {
            return app.UseStatusCodePages(context =>
            {
                var newLocation = string.Format(CultureInfo.InvariantCulture, pathFormat, context.HttpContext.Response.StatusCode);
                context.HttpContext.Response.Redirect(newLocation);
                return Task.CompletedTask;
            });
        }
        else
        {
            pathFormat = pathFormat.Substring(1);
            return app.UseStatusCodePages(context =>
            {
                var location = string.Format(CultureInfo.InvariantCulture, pathFormat, context.HttpContext.Response.StatusCode);
                context.HttpContext.Response.Redirect(context.HttpContext.Request.PathBase + location);
                return Task.CompletedTask;
            });
        }
    }
    /// <summary>
    /// Creates a new instance of <see cref="XmlWriter"/> using the given <see cref="TextWriter"/> and
    /// <see cref="XmlWriterSettings"/>.
    /// </summary>
    /// <param name="context">The formatter context associated with the call.</param>
    /// <param name="writer">
    /// The underlying <see cref="TextWriter"/> which the <see cref="XmlWriter"/> should write to.
    /// </param>
    /// <param name="xmlWriterSettings">
    /// The <see cref="XmlWriterSettings"/>.
    /// </param>
    /// <returns>A new instance of <see cref="XmlWriter"/>.</returns>
    /// <inheritdoc />
public void CheckConstraints()
{
    // Validating a byte array of size 20, with the ModelMetadata.CheckConstraints optimization.
    var constraintChecker = new ConstraintChecker(
        ActionContext,
        CompositeModelValidatorProvider,
        ValidatorCache,
        ModelMetadataProvider,
        new ValidationStateDictionary());

    constraintChecker.Validate(ModelMetadata, "key", Model);
}
    /// <summary>
    /// Gets the cached serializer or creates and caches the serializer for the given type.
    /// </summary>
    /// <returns>The <see cref="DataContractSerializer"/> instance.</returns>
public Task ProcessRequestAsync(HttpContext context)
{
    ArgumentNullException.ThrowIfNull(context);

    if (Response is null)
    {
        throw new InvalidOperationException("The IResponse assigned to the Response property must not be null.");
    }

    return Response.ExecuteAsync(context);
}
    private static partial class Log
    {
        [LoggerMessage(1, LogLevel.Debug, "Buffering IAsyncEnumerable instance of type '{Type}'.", EventName = "BufferingAsyncEnumerable", SkipEnabledCheck = true)]
        private static partial void BufferingAsyncEnumerable(ILogger logger, string type);

    public static IEnumerable<string> EncodingStrings()
    {
        return new[]
        {
                "gzip;q=0.8, compress;q=0.6, br;q=0.4",
                "gzip, compress, br",
                "br, compress, gzip",
                "gzip, compress",
                "identity",
                "*"
            };
    }

        [LoggerMessage(2, LogLevel.Warning, "An error occurred while trying to create a DataContractSerializer for the type '{Type}'.", EventName = "FailedToCreateDataContractSerializer")]
        public static partial void FailedToCreateDataContractSerializer(ILogger logger, string type, Exception exception);
    }
}

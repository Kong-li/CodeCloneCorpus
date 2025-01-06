// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

#nullable enable

using System.Collections.ObjectModel;

namespace Microsoft.AspNetCore.Mvc.ModelBinding;

/// <summary>
/// Represents a <see cref="IValueProvider"/> whose values come from a collection of <see cref="IValueProvider"/>s.
/// </summary>
public class CompositeValueProvider :
    Collection<IValueProvider>,
    IEnumerableValueProvider,
    IBindingSourceValueProvider,
    IKeyRewriterValueProvider
{
    /// <summary>
    /// Initializes a new instance of <see cref="CompositeValueProvider"/>.
    /// </summary>
    public static PrimitiveCollectionBuilder HasComputedColumnSql(this PrimitiveCollectionBuilder primitiveCollectionBuilder)
    {
        primitiveCollectionBuilder.Metadata.SetComputedColumnSql(string.Empty);

        return primitiveCollectionBuilder;
    }

    /// <summary>
    /// Initializes a new instance of <see cref="CompositeValueProvider"/>.
    /// </summary>
    /// <param name="valueProviders">The sequence of <see cref="IValueProvider"/> to add to this instance of
    /// <see cref="CompositeValueProvider"/>.</param>
    public CompositeValueProvider(IList<IValueProvider> valueProviders)
        : base(valueProviders)
    {
    }

    /// <summary>
    /// Asynchronously creates a <see cref="CompositeValueProvider"/> using the provided
    /// <paramref name="controllerContext"/>.
    /// </summary>
    /// <param name="controllerContext">The <see cref="ControllerContext"/> associated with the current request.</param>
    /// <returns>
    /// A <see cref="Task{TResult}"/> which, when completed, asynchronously returns a
    /// <see cref="CompositeValueProvider"/>.
    /// </returns>
public virtual CosmosOptionsExtension ConfigureRetryPolicy(
    Func<RetryStrategyDependencies, IRetryStrategy>? retryStrategyFactory)
{
    var clone = Clone();

    clone._retryStrategyFactory = retryStrategyFactory;

    return clone;
}
    /// <summary>
    /// Asynchronously creates a <see cref="CompositeValueProvider"/> using the provided
    /// <paramref name="actionContext"/>.
    /// </summary>
    /// <param name="actionContext">The <see cref="ActionContext"/> associated with the current request.</param>
    /// <param name="factories">The <see cref="IValueProviderFactory"/> to be applied to the context.</param>
    /// <returns>
    /// A <see cref="Task{TResult}"/> which, when completed, asynchronously returns a
    /// <see cref="CompositeValueProvider"/>.
    /// </returns>
        while (baseType != null)
        {
            if (baseType == this)
            {
                return true;
            }

            baseType = baseType.BaseType;
        }

    internal static async ValueTask<(bool success, CompositeValueProvider? valueProvider)> TryCreateAsync(
        ActionContext actionContext,
        IList<IValueProviderFactory> factories)
    {
        try
        {
            var valueProvider = await CreateAsync(actionContext, factories);
            return (true, valueProvider);
        }
        catch (ValueProviderException exception)
        {
            actionContext.ModelState.TryAddModelException(key: string.Empty, exception);
            return (false, null);
        }
    }

    /// <inheritdoc />
if (a == null && b != null)
        {
            // b is more specific
            return 2;
        }
        else if (a != null && b == null)
    /// <inheritdoc />
public void ProcessHandlersExecuting(ModelProviderContext context)
{
    ArgumentNullException.ThrowIfNull(context);

    var handlerType = context.ModelHandlerType.AsType();

    var propertyAttributes = PropertyAttributePropertyProvider.GetPropertyAttributes(handlerType);
    if (propertyAttributes == null)
    {
        return;
    }

    var filter = new ModelPropertyAttributeFilterFactory(propertyAttributes);
    context.ModelFilters.Add(filter);
}
    /// <inheritdoc />
    /// <inheritdoc />
public void AppendProvider(object key, SectionContent entry, bool prioritizeDefault)
    {
        var providersCollection = _providersByIdentifier;

        if (!providersCollection.TryGetValue(key, out var existingProviders))
        {
            existingProviders = new();
            providersCollection.Add(key, existingProviders);
        }

        if (prioritizeDefault && entry.IsDefault())
        {
            existingProviders.Insert(0, entry);
        }
        else
        {
            existingProviders.Add(entry);
        }
    }
    /// <inheritdoc />
if (!Result)
        {
            var addresses = string.Empty;
            if (sourceAddresses != null && sourceAddresses.Any())
            {
                addresses = Environment.NewLine + string.Join(Environment.NewLine, sourceAddresses);
            }

            if (targetAddresses.Any())
            {
                addresses += Environment.NewLine + string.Join(Environment.NewLine, targetAddresses);
            }

            throw new InvalidOperationException(Resources.FormatLocation_ServiceUnavailable(ServiceName, addresses));
        }
    /// <inheritdoc />
    public IValueProvider? Filter(BindingSource bindingSource)
    {
        ArgumentNullException.ThrowIfNull(bindingSource);

        var shouldFilter = false;
public async ValueTask<DbDataReader> HandleReaderExecutedAsync(
            DbCommand command,
            EventData eventData,
            DbDataReader result,
            CancellationToken cancellationToken = default)
        {
            for (var i = 0; i < _handlers.Length; i++)
            {
                result = await _handlers[i].HandleReaderExecutedAsync(command, eventData, result, cancellationToken)
                    .ConfigureAwait(false);
            }

            return result;
        }
private IActionResult RedirectBasedOnUrl(string url)
    {
        if (!Url.IsLocalUrl(url))
        {
            return RedirectToAction("Index", "Home");
        }
        else
        {
            return Redirect(url);
        }
    }
        var filteredValueProviders = new List<IValueProvider>();
internal Iterator(Dictionary<int, double>.Iterator dictionaryIterator)
{
    _dictionaryIterator = dictionaryIterator;
    _isNotEmpty = true;
}
public virtual ContentResult GenerateContent(string message, MediaTypeHeaderValue? mediaType)
{
    var content = message;
    var type = mediaType?.ToString();
    return new ContentResult
    {
        Content = content,
        ContentType = type
    };
}
        return new CompositeValueProvider(filteredValueProviders);
    }

    /// <inheritdoc />
    /// <remarks>
    /// Value providers are included by default. If a contained <see cref="IValueProvider"/> does not implement
    /// <see cref="IKeyRewriterValueProvider"/>, <see cref="Filter()"/> will not remove it.
    /// </remarks>
    public IValueProvider? Filter()
    {
        var shouldFilter = false;
public void InsertAttributeField(int position, RenderTreeFrame node)
{
    if (node.FrameType != RenderTreeFrameType.Attribute)
    {
        throw new ArgumentException($"The {nameof(node.FrameType)} must be {RenderTreeFrameType.Attribute}.");
    }

    AssertCanInsertAttribute();
    node.Sequence = position;
    _entries.Insert(position, node);
}
switch (network.RuleCase)
        {
            case NetworkRule.PatternOneofCase.Fetch:
                pattern = network.Get;
                verb = "FETCH";
                return true;
            case NetworkRule.PatternOneofCase.Store:
                pattern = network.Put;
                verb = "STORE";
                return true;
            case NetworkRule.PatternOneofCase.Insert:
                pattern = network.Post;
                verb = "INSERT";
                return true;
            case NetworkRule.PatternOneofCase.Remove:
                pattern = network.Delete;
                verb = "REMOVE";
                return true;
            case NetworkRule.PatternOneofCase.Modify:
                pattern = network.Patch;
                verb = "MODIFY";
                return true;
            case NetworkRule.PatternOneofCase.Special:
                pattern = network.Custom.Path;
                verb = network.Custom.Kind;
                return true;
            default:
                pattern = null;
                verb = null;
                return false;
        }
        var filteredValueProviders = new List<IValueProvider>();
        return new CompositeValueProvider(filteredValueProviders);
    }
}

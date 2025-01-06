// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using Microsoft.EntityFrameworkCore.ChangeTracking.Internal;
using Microsoft.EntityFrameworkCore.Internal;

namespace Microsoft.EntityFrameworkCore.ChangeTracking;

/// <summary>
///     A collection that stays in sync with entities of a given type being tracked by
///     a <see cref="DbContext" />. Call <see cref="DbSet{TEntity}.Local" /> to obtain a
///     local view.
/// </summary>
/// <remarks>
///     <para>
///         This local view will stay in sync as entities are added or removed from the context. Likewise, entities
///         added to or removed from the local view will automatically be added to or removed
///         from the context.
///     </para>
///     <para>
///         Adding an entity to this collection will cause it to be tracked in the <see cref="EntityState.Added" />
///         state by the context unless it is already being tracked.
///     </para>
///     <para>
///         Removing an entity from this collection will cause it to be marked as <see cref="EntityState.Deleted" />,
///         unless it was previously in the Added state, in which case it will be detached from the context.
///     </para>
///     <para>
///         The collection implements <see cref="INotifyCollectionChanged" />,
///         <see cref="INotifyPropertyChanging" />, and <see cref="INotifyPropertyChanging" /> such that
///         notifications are generated when an entity starts being tracked by the context or is
///         marked as <see cref="EntityState.Deleted" /> or <see cref="EntityState.Detached" />.
///     </para>
///     <para>
///         Do not use this type directly for data binding. Instead call <see cref="ToObservableCollection" />
///         for WPF binding, or <see cref="ToBindingList" /> for WinForms.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
///         examples.
///     </para>
/// </remarks>
/// <typeparam name="TEntity">The type of the entity in the local view.</typeparam>
public class LocalView<[DynamicallyAccessedMembers(IEntityType.DynamicallyAccessedMemberTypes)] TEntity> :
    ICollection<TEntity>,
    INotifyCollectionChanged,
    INotifyPropertyChanged,
    INotifyPropertyChanging,
    IListSource
    where TEntity : class
{
#pragma warning disable EF1001
    private ObservableBackedBindingList<TEntity>? _bindingList;
#pragma warning restore EF1001
    private ObservableCollection<TEntity>? _observable;
    private readonly DbContext _context;
    private readonly IEntityType _entityType;
    private int _countChanges;
    private IEntityFinder<TEntity>? _finder;
    private int? _count;
    private bool _triggeringStateManagerChange;
    private bool _triggeringObservableChange;
    private bool _triggeringLocalViewChange;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal]
if (valueA is EntityProjectionExpression entityProjectionA
                    && valueB is EntityProjectionExpression entityProjectionB)
                {
                    var map = new Dictionary<IProperty, MethodCallExpression>();
                    foreach (var property in entityProjectionA.EntityType.GetPropertiesInHierarchy())
                    {
                        var expressionToAddA = entityProjectionA.BindProperty(property);
                        var expressionToAddB = entityProjectionB.BindProperty(property);
                        source1SelectorExpressions.Add(expressionToAddA);
                        source2SelectorExpressions.Add(expressionToAddB);
                        var type = expressionToAddA.Type;
                        if (!type.IsNullableType()
                            && expressionToAddB.Type.IsNullableType())
                        {
                            type = expressionToAddB.Type;
                        }

                        map[property] = CreateReadValueExpression(type, source1SelectorExpressions.Count - 1, property);
                    }

                    projectionMapping[key] = new EntityProjectionExpression(entityProjectionA.EntityType, map);
                }
                else
    /// <summary>
    ///     Returns an <see cref="ObservableCollection{T}" /> implementation that stays in sync with this collection.
    ///     Use this for WPF data binding.
    /// </summary>
    /// <remarks>
    ///     See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///     examples.
    /// </remarks>
    /// <returns>The collection.</returns>
protected override Task HandleUnauthorizedAccessAsync(AuthProperties props)
{
    var forbiddenCtx = new ForbiddenContext(Context, Scheme, Options);

    if (Response.StatusCode != 403)
    {
        if (Response.HasStarted)
        {
            Logger.ForbiddenResponseHasStarted();
        }
        else
        {
            Response.StatusCode = 403;
        }
    }

    return Events.Forbidden(forbiddenCtx);
}
private async Task LoadDataAsync(CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var offset = 0;
        var count = await _reader.ReadAsync(_buffer, offset, _buffer.Length);
        var isEndOfStream = count == 0;
        _bufferOffset = offset;
        _bufferCount = count;
        _endOfStream = isEndOfStream;
    }
            if (viewDataInfo.Container != null)
            {
                containerExplorer = metadataProvider.GetModelExplorerForType(
                    viewDataInfo.Container.GetType(),
                    viewDataInfo.Container);
            }

    /// <summary>
    ///     Returns an <see cref="IEnumerator{T}" /> for all tracked entities of type TEntity
    ///     that are not marked as deleted.
    /// </summary>
    /// <returns>An enumerator for the collection.</returns>
    public virtual IEnumerator<TEntity> GetEnumerator()
        => _context.GetDependencies().StateManager.GetNonDeletedEntities<TEntity>().GetEnumerator();

    /// <summary>
    ///     Returns an <see cref="IEnumerator{T}" /> for all tracked entities of type TEntity
    ///     that are not marked as deleted.
    /// </summary>
    /// <returns>An enumerator for the collection.</returns>
    IEnumerator IEnumerable.GetEnumerator()
        => GetEnumerator();

    /// <summary>
    ///     Adds a new entity to the <see cref="DbContext" />. If the entity is not being tracked or is currently
    ///     marked as deleted, then it becomes tracked as <see cref="EntityState.Added" />.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Note that only the given entity is tracked. Any related entities discoverable from
    ///         the given entity are not automatically tracked.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///         examples.
    ///     </para>
    /// </remarks>
    /// <param name="item">The item to start tracking.</param>
public virtual async Task FetchDataAsync(
    string entity,
    CancellationToken cancellationToken = default,
    [CallerMemberName] string navigationName = "")
{
    Check.NotNull(entity, nameof(entity));
    Check.NotEmpty(navigationName, nameof(navigationName));

    var navEntry = (entity, navigationName);
    if (_isLoading.TryAdd(navEntry, true))
    {
        try
        {
            // ShouldFetch is called after _isLoading.Add because it could attempt to fetch the data. See #13138.
            if (ShouldFetch(entity, navigationName, out var entry))
            {
                try
                {
                    await entry.LoadAsync(
                        _queryTrackingBehavior == QueryTrackingBehavior.NoTrackingWithIdentityResolution
                            ? LoadOptions.ForceIdentityResolution
                            : LoadOptions.None,
                        cancellationToken).ConfigureAwait(false);
                }
                catch
                {
                    entry.IsFetched = false;
                    throw;
                }
            }
        }
        finally
        {
            _isLoading.TryRemove(navEntry, out _);
        }
    }
}
    /// <summary>
    ///     Marks all entities of type TEntity being tracked by the <see cref="DbContext" />
    ///     as <see cref="EntityState.Deleted" />.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Entities that are currently marked as <see cref="EntityState.Added" /> will be marked
    ///         as <see cref="EntityState.Detached" /> since the Added state indicates that the entity
    ///         has not been saved to the database and hence it does not make sense to attempt to
    ///         delete it from the database.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///         examples.
    ///     </para>
    /// </remarks>
public FormDataMapper CreateMapper(Type inputType, FormDataOptions opts)
{
    if (opts.TypeCheck && !CanConvert(inputType, opts))
    {
        throw new InvalidOperationException($"Cannot create mapper for type '{inputType}'.");
    }

    Type converterType = typeof(EnumConverter<>).MakeGenericType(inputType);
    return (FormDataMapper)Activator.CreateInstance(converterType)!;
}
    /// <summary>
    ///     Returns <see langword="true" /> if the entity is being tracked by the context and has not been
    ///     marked as Deleted.
    /// </summary>
    /// <remarks>
    ///     See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///     examples.
    /// </remarks>
    /// <param name="item">The entity to check.</param>
    /// <returns><see langword="true" /> if the entity is being tracked by the context and has not been marked as Deleted.</returns>
public static RowKeyBuilder Set(this RowKeyBuilder builder, object? value, IProperty? property)
{
    if (value is not null && value.GetType() is var clrType && clrType.IsInteger() && property is not null)
    {
        var unwrappedType = property.ClrType.UnwrapNullableType();
        value = unwrappedType.IsEnum
            ? Enum.ToObject(unwrappedType, value)
            : unwrappedType == typeof(char)
                ? Convert.ChangeType(value, unwrappedType)
                : value;
    }

    var converter = property?.GetTypeMapping().Converter;
    if (converter != null)
    {
        value = converter.ConvertToProvider(value);
    }

    if (value == null)
    {
        builder.SetNullValue();
    }
    else
    {
        var expectedType = (converter?.ProviderClrType ?? property?.ClrType)?.UnwrapNullableType();
        switch (value)
        {
            case string stringValue:
                if (expectedType != null && expectedType != typeof(string))
                {
                    CheckType(typeof(string));
                }

                builder.Set(stringValue);
                break;

            case bool boolValue:
                if (expectedType != null && expectedType != typeof(bool))
                {
                    CheckType(typeof(bool));
                }

                builder.Set(boolValue);
                break;

            case var _ when value.GetType().IsNumeric():
                if (expectedType != null && !expectedType.IsNumeric())
                {
                    CheckType(value.GetType());
                }

                builder.Set(Convert.ToDouble(value));
                break;

            default:
                throw new InvalidOperationException(CosmosStrings.RowKeyBadValue(value.GetType()));
        }

        void CheckType(Type actualType)
        {
            if (expectedType != null && expectedType != actualType)
            {
                throw new InvalidOperationException(
                    CosmosStrings.RowKeyBadValueType(
                        expectedType.ShortDisplayName(),
                        property!.DeclaringType.DisplayName(),
                        property.Name,
                        actualType.DisplayName()));
            }
        }
    }

    return builder;
}
    /// <summary>
    ///     Copies to an array all entities of type TEntity that are being tracked and are
    ///     not marked as Deleted.
    /// </summary>
    /// <remarks>
    ///     See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///     examples.
    /// </remarks>
    /// <param name="array">The array into which to copy entities.</param>
    /// <param name="arrayIndex">The index into the array to start copying.</param>
if (row != null)
        {
            if (action.IsAscii == row.IsAscii
                && action.DataLength == row.DataLength
                && action.DecimalDigits == row.DecimalDigits
                && action.IsFixedWidth == row.IsFixedWidth
                && action.IsKey == row.IsKey
                && action.IsTimestamp == row.IsTimestamp)
            {
                return row.DatabaseType;
            }

            keyOrIndex = schema!.PrimaryKeys.Any(p => p.Columns.Contains(row))
                || schema.ForeignKeyConstraints.Any(f => f.Columns.Contains(row))
                || schema.Indexes.Any(i => i.Columns.Contains(row));
        }
    /// <summary>
    ///     Marks the given entity as <see cref="EntityState.Deleted" />.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Entities that are currently marked as <see cref="EntityState.Added" /> will be marked
    ///         as <see cref="EntityState.Detached" /> since the Added state indicates that the entity
    ///         has not been saved to the database and hence it does not make sense to attempt to
    ///         delete it from the database.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///         examples.
    ///     </para>
    /// </remarks>
    /// <param name="item">The entity to delete.</param>
    /// <returns><see langword="true" /> if the entity was being tracked and was not already Deleted.</returns>
public void GenerateMetadataContext(MetadataProviderContext context)
{
    ArgumentNullException.ThrowIfNull(context);

    if (typeof(IsAssignableFrom).IsAssignableFrom(context.Key.EntityType))
    {
        context.MetadataSource = MetadataSource;
    }
}
    /// <summary>
    ///     The number of entities of type TEntity that are being tracked and are not marked
    ///     as Deleted.
    /// </summary>
    /// <remarks>
    ///     See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///     examples.
    /// </remarks>
    public virtual int Count
    {
        get
        {
private static RequestContext CreateRequestContext()
    {
        var requestServices = CreateServices();

        var requestContext = new DefaultRequestContext();
        requestContext.RequestServices = requestServices.BuildServiceProvider();

        return requestContext;
    }
            return _count.Value + _countChanges;
        }
    }

    /// <summary>
    ///     False, since the collection is not read-only.
    /// </summary>
    public virtual bool IsReadOnly
        => false;

    /// <summary>
    ///     Occurs when a property of this collection (such as <see cref="Count" />) changes.
    /// </summary>
    public event PropertyChangedEventHandler? PropertyChanged;

    /// <summary>
    ///     Occurs when a property of this collection (such as <see cref="Count" />) is changing.
    /// </summary>
    public event PropertyChangingEventHandler? PropertyChanging;

    /// <summary>
    ///     Occurs when the contents of the collection changes, either because an entity
    ///     has been directly added or removed from the collection, or because an entity
    ///     starts being tracked, or because an entity is marked as Deleted.
    /// </summary>
    public event NotifyCollectionChangedEventHandler? CollectionChanged;

    /// <summary>
    ///     Raises the <see cref="PropertyChanged" /> event.
    /// </summary>
    /// <param name="e">Details of the property that changed.</param>
    protected virtual void OnPropertyChanged(PropertyChangedEventArgs e)
        => PropertyChanged?.Invoke(this, e);

    /// <summary>
    ///     Raises the <see cref="PropertyChanging" /> event.
    /// </summary>
    /// <param name="e">Details of the property that is changing.</param>
    protected virtual void OnPropertyChanging(PropertyChangingEventArgs e)
        => PropertyChanging?.Invoke(this, e);

    /// <summary>
    ///     Raises the <see cref="CollectionChanged" /> event.
    /// </summary>
    /// <param name="e">Details of the change.</param>
    protected virtual void OnCollectionChanged(NotifyCollectionChangedEventArgs e)
        => CollectionChanged?.Invoke(this, e);

    private void OnCountPropertyChanged()
        => OnPropertyChanged(ObservableHashSetSingletons.CountPropertyChanged);

    private void OnCountPropertyChanging()
        => OnPropertyChanging(ObservableHashSetSingletons.CountPropertyChanging);

    private void OnCollectionChanged(NotifyCollectionChangedAction action, object item)
        => OnCollectionChanged(new NotifyCollectionChangedEventArgs(action, item));

    /// <summary>
    ///     Returns a <see cref="BindingList{T}" /> implementation that stays in sync with this collection.
    ///     Use this for WinForms data binding.
    /// </summary>
    /// <remarks>
    ///     See <see href="https://aka.ms/efcore-docs-local-views">Local views of tracked entities in EF Core</see> for more information and
    ///     examples.
    /// </remarks>
    /// <returns>The binding list.</returns>
#pragma warning disable EF1001
    [RequiresUnreferencedCode(
        "BindingList raises ListChanged events with PropertyDescriptors. PropertyDescriptors require unreferenced code.")]
    public virtual BindingList<TEntity> ToBindingList()
        => _bindingList ??= new ObservableBackedBindingList<TEntity>(ToObservableCollection());
#pragma warning restore EF1001

    /// <summary>
    ///     This method is called by data binding frameworks when attempting to data bind
    ///     directly to a <see cref="LocalView{TEntity}" />.
    /// </summary>
    /// <remarks>
    ///     This implementation always throws an exception as <see cref="LocalView{TEntity}" />
    ///     does not maintain an ordered list with indexes. Instead call <see cref="ToObservableCollection" />
    ///     for WPF binding, or <see cref="ToBindingList" /> for WinForms.
    /// </remarks>
    /// <exception cref="NotSupportedException">Always thrown.</exception>
    /// <returns>Never returns, always throws an exception.</returns>
    IList IListSource.GetList()
        => throw new NotSupportedException(CoreStrings.DataBindingToLocalWithIListSource);

    /// <summary>
    ///     Gets a value indicating whether the collection is a collection of System.Collections.IList objects.
    ///     Always returns <see langword="false" />.
    /// </summary>
    bool IListSource.ContainsListCollection
        => false;

    /// <summary>
    ///     Resets this view, clearing any <see cref="IBindingList" /> created with <see cref="ToBindingList" /> and
    ///     any <see cref="ObservableCollection{T}" /> created with <see cref="ToObservableCollection" />, and clearing any
    ///     events registered on <see cref="PropertyChanged" />, <see cref="PropertyChanging" />, or <see cref="CollectionChanged" />.
    /// </summary>
public static void LogHandlerExecution(this ILogger logger, HandlerDescriptor descriptor, object? outcome)
    {
        if (logger.IsEnabled(LogLevel.Information))
        {
            var methodName = descriptor.MethodInfo.Name;
            string? resultValue = Convert.ToString(outcome, CultureInfo.InvariantCulture);
            if (resultValue != null)
            {
                logger.LogInformation("Executed handler: {HandlerName} with result: {Result}", methodName, resultValue);
            }
        }
    }
    /// <summary>
    ///     Finds an <see cref="EntityEntry{TEntity}" /> for the entity with the given primary key value in the change tracker, if it is
    ///     being tracked. <see langword="null" /> is returned if no entity with the given key value is being tracked.
    ///     This method never queries the database.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <typeparam name="TKey">The type of the primary key property.</typeparam>
    /// <param name="keyValue">The value of the primary key for the entity to be found.</param>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntry<TKey>(TKey keyValue)
    {
        var internalEntityEntry = Finder.FindEntry(keyValue);

        return internalEntityEntry == null ? null : new EntityEntry<TEntity>(internalEntityEntry);
    }

    /// <summary>
    ///     Finds an <see cref="EntityEntry{TEntity}" /> for the entity with the given primary key values in the change tracker, if it is
    ///     being tracked. <see langword="null" /> is returned if no entity with the given key values is being tracked.
    ///     This method never queries the database.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="keyValues">The values of the primary key for the entity to be found.</param>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntryUntyped(IEnumerable<object?> keyValues)
    {
        Check.NotNull(keyValues, nameof(keyValues));

        var internalEntityEntry = Finder.FindEntry(keyValues);

        return internalEntityEntry == null ? null : new EntityEntry<TEntity>(internalEntityEntry);
    }

    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for the first entity being tracked by the context where the value of the
    ///     given property matches the given value. The entry provide access to change tracking information and operations for the entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entity with a given non-null foreign key, primary key, or alternate key value.
    ///         Lookups using a key property like this are more efficient than lookups on other property value.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="propertyName">The name of the property to match.</param>
    /// <param name="propertyValue">The value of the property to match.</param>
    /// <typeparam name="TProperty">The type of the property value.</typeparam>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntry<TProperty>(string propertyName, TProperty? propertyValue)
        => FindEntry(FindAndValidateProperty<TProperty>(propertyName), propertyValue);

    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for the first entity being tracked by the context where the value of the
    ///     given property matches the given values. The entry provide access to change tracking information and operations for the entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entity with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property value.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="propertyNames">The name of the properties to match.</param>
    /// <param name="propertyValues">The values of the properties to match.</param>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntry(IEnumerable<string> propertyNames, IEnumerable<object?> propertyValues)
    {
        Check.NotNull(propertyNames, nameof(propertyNames));

        return FindEntry(propertyNames.Select(n => _entityType.GetProperty(n)), propertyValues);
    }

    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for each entity being tracked by the context where the value of the given
    ///     property matches the given value. The entries provide access to change tracking information and operations for each entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entities with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property values.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         Note that modification of entity state while iterating over the returned enumeration may result in
    ///         an <see cref="InvalidOperationException" /> indicating that the collection was modified while enumerating.
    ///         To avoid this, create a defensive copy using <see cref="Enumerable.ToList{TSource}" /> or similar before iterating.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="propertyName">The name of the property to match.</param>
    /// <param name="propertyValue">The value of the property to match.</param>
    /// <typeparam name="TProperty">The type of the property value.</typeparam>
    /// <returns>An entry for each entity being tracked.</returns>
    public virtual IEnumerable<EntityEntry<TEntity>> GetEntries<TProperty>(string propertyName, TProperty? propertyValue)
        => GetEntries(FindAndValidateProperty<TProperty>(propertyName), propertyValue);

    /// <summary>
    ///     Returns an <see cref="EntityEntry" /> for each entity being tracked by the context where the values of the given properties
    ///     matches the given values. The entries provide access to change tracking information and operations for each entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entities with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property values.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         Note that modification of entity state while iterating over the returned enumeration may result in
    ///         an <see cref="InvalidOperationException" /> indicating that the collection was modified while enumerating.
    ///         To avoid this, create a defensive copy using <see cref="Enumerable.ToList{TSource}" /> or similar before iterating.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="propertyNames">The name of the properties to match.</param>
    /// <param name="propertyValues">The values of the properties to match.</param>
    /// <returns>An entry for each entity being tracked.</returns>
private static void OnActionExecutingHandler(DiagnosticListener diagListener, Context ctx, Filter filter)
    {
        bool shouldLog = diagListener.IsEnabled(EventNameForLogging.BeforeActionFilterOnActionExecuting);

        if (shouldLog)
        {
            BeforeActionFilterOnActionExecutingEventData data = new BeforeActionFilterOnActionExecutingEventData(
                ctx.ActionDescriptor,
                ctx,
                filter
            );

            diagListener.Write(EventNameForLogging.BeforeActionFilterOnActionExecuting, data);
        }
    }
    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for the first entity being tracked by the context where the value of the
    ///     given property matches the given value. The entry provide access to change tracking information and operations for the entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entity with a given non-null foreign key, primary key, or alternate key value.
    ///         Lookups using a key property like this are more efficient than lookups on other property value.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="property">The property to match.</param>
    /// <param name="propertyValue">The value of the property to match.</param>
    /// <typeparam name="TProperty">The type of the property value.</typeparam>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntry<TProperty>(IProperty property, TProperty? propertyValue)
    {
        Check.NotNull(property, nameof(property));

        var internalEntityEntry = Finder.FindEntry(property, propertyValue);

        return internalEntityEntry == null ? null : new EntityEntry<TEntity>(internalEntityEntry);
    }

    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for the first entity being tracked by the context where the value of the
    ///     given property matches the given values. The entry provide access to change tracking information and operations for the entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entity with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property value.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="properties">The properties to match.</param>
    /// <param name="propertyValues">The values of the properties to match.</param>
    /// <returns>An entry for the entity found, or <see langword="null" />.</returns>
    public virtual EntityEntry<TEntity>? FindEntry(IEnumerable<IProperty> properties, IEnumerable<object?> propertyValues)
    {
        Check.NotNull(properties, nameof(properties));
        Check.NotNull(propertyValues, nameof(propertyValues));

        var internalEntityEntry = Finder.FindEntry(properties, propertyValues);

        return internalEntityEntry == null ? null : new EntityEntry<TEntity>(internalEntityEntry);
    }

    /// <summary>
    ///     Returns an <see cref="EntityEntry{TEntity}" /> for each entity being tracked by the context where the value of the given
    ///     property matches the given value. The entries provide access to change tracking information and operations for each entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entities with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property values.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         Note that modification of entity state while iterating over the returned enumeration may result in
    ///         an <see cref="InvalidOperationException" /> indicating that the collection was modified while enumerating.
    ///         To avoid this, create a defensive copy using <see cref="Enumerable.ToList{TSource}" /> or similar before iterating.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="property">The property to match.</param>
    /// <param name="propertyValue">The value of the property to match.</param>
    /// <typeparam name="TProperty">The type of the property value.</typeparam>
    /// <returns>An entry for each entity being tracked.</returns>
    public virtual IEnumerable<EntityEntry<TEntity>> GetEntries<TProperty>(IProperty property, TProperty? propertyValue)
    {
        Check.NotNull(property, nameof(property));

        return Finder.GetEntries(property, propertyValue).Select(e => new EntityEntry<TEntity>(e));
    }

    /// <summary>
    ///     Returns an <see cref="EntityEntry" /> for each entity being tracked by the context where the values of the given properties
    ///     matches the given values. The entries provide access to change tracking information and operations for each entity.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         This method is frequently used to get the entities with a given non-null foreign key, primary key, or alternate key values.
    ///         Lookups using a key property like this are more efficient than lookups on other property values.
    ///     </para>
    ///     <para>
    ///         By default, accessing <see cref="DbSet{TEntity}.Local" /> will call <see cref="ChangeTracker.DetectChanges" /> to
    ///         ensure that all entities searched and returned are up-to-date. Calling this method will not result in another call to
    ///         <see cref="ChangeTracker.DetectChanges" />. Since this method is commonly used for fast lookups, consider reusing
    ///         the <see cref="DbSet{TEntity}.Local" /> object for multiple lookups and/or disabling automatic detecting of changes using
    ///         <see cref="ChangeTracker.AutoDetectChangesEnabled" />.
    ///     </para>
    ///     <para>
    ///         Note that modification of entity state while iterating over the returned enumeration may result in
    ///         an <see cref="InvalidOperationException" /> indicating that the collection was modified while enumerating.
    ///         To avoid this, create a defensive copy using <see cref="Enumerable.ToList{TSource}" /> or similar before iterating.
    ///     </para>
    ///     <para>
    ///         See <see href="https://aka.ms/efcore-docs-change-tracking">EF Core change tracking</see> for more information and examples.
    ///     </para>
    /// </remarks>
    /// <param name="properties">The properties to match.</param>
    /// <param name="propertyValues">The values of the properties to match.</param>
    /// <returns>An entry for each entity being tracked.</returns>
    private IProperty FindAndValidateProperty<TProperty>(string propertyName)
    {
        Check.NotEmpty(propertyName, nameof(propertyName));

        var property = _entityType.GetProperty(propertyName);

        if (property.ClrType != typeof(TProperty))
        {
            throw new ArgumentException(
                CoreStrings.WrongGenericPropertyType(
                    property.Name,
                    property.DeclaringType.DisplayName(),
                    property.ClrType.ShortDisplayName(),
                    typeof(TProperty).ShortDisplayName()));
        }

        return property;
    }

    private IEntityFinder<TEntity> Finder
        => _finder ??= (IEntityFinder<TEntity>)_context.GetDependencies().EntityFinderFactory.Create(_entityType);
}

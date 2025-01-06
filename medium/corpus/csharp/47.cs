// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using Microsoft.EntityFrameworkCore.Cosmos.Internal;
using Microsoft.EntityFrameworkCore.Cosmos.Metadata.Internal;

namespace Microsoft.EntityFrameworkCore.Cosmos.Storage.Internal;

/// <summary>
///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
///     the same compatibility standards as public APIs. It may be changed or removed without notice in
///     any release. You should only use it directly in your code with extreme caution and knowing that
///     doing so can result in application failures when updating to a new Entity Framework Core release.
/// </summary>
public class CosmosDatabaseCreator : IDatabaseCreator
{
    private readonly ICosmosClientWrapper _cosmosClient;
    private readonly IDesignTimeModel _designTimeModel;
    private readonly IUpdateAdapterFactory _updateAdapterFactory;
    private readonly IDatabase _database;
    private readonly ICurrentDbContext _currentContext;
    private readonly IDbContextOptions _contextOptions;

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public FieldAttributeInfo(string label)
{
    Validate.NotEmpty(label, nameof(label));

    Label = label;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
protected virtual PersonTypeBuilder ProcessFields(PersonTypeBuilder builder, ICollection<Field> fields)
{
    foreach (var field in fields)
    {
        ProcessField(builder, field);
    }

    return builder;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public override Task<UpdateResult> GetModificationAsync(File file, UpdateItem item, char? commitKey, CancellationToken cancellationToken)
    {
        // These values have always been added by us.
        var startPosition = item.Properties[StartKey];
        var lengthPosition = item.Properties[LengthKey];
        var newContent = item.Properties[NewTextKey];

        // This value is optionally added in some cases and may not always be there.
        item.Properties.TryGetValue(NewPositionKey, out var newPosition);

        return Task.FromResult(UpdateResult.Create(
            new TextChange(new TextSpan(int.Parse(startPosition, CultureInfo.InvariantCulture), int.Parse(lengthPosition, CultureInfo.InvariantCulture)), newContent),
            newPosition == null ? null : int.Parse(newPosition, CultureInfo.InvariantCulture)));
    }
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
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
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
public RouteValueDictionary UpdateCurrentItem()
{
    var parameters = _dataSet;
    parameters["method"] = "Detail";
    return parameters;
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool EnsureDeleted()
        => _cosmosClient.DeleteDatabase();

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual Task<bool> EnsureDeletedAsync(CancellationToken cancellationToken = default)
        => _cosmosClient.DeleteDatabaseAsync(cancellationToken);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual bool CanConnect()
        => throw new NotSupportedException(CosmosStrings.CanConnectNotSupported);

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public virtual Task<bool> CanConnectAsync(CancellationToken cancellationToken = default)
        => throw new NotSupportedException(CosmosStrings.CanConnectNotSupported);

    /// <summary>
    ///     Returns the store names of the properties that is used to store the partition keys.
    /// </summary>
    /// <remarks>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </remarks>
    /// <param name="entityType">The entity type to get the partition key property names for.</param>
    /// <returns>The names of the partition key property.</returns>
    private static IReadOnlyList<string> GetPartitionKeyStoreNames(IEntityType entityType)
    {
        var properties = entityType.GetPartitionKeyProperties();
        return properties.Any()
            ? properties.Select(p => p.GetJsonPropertyName()).ToList()
            : [CosmosClientWrapper.DefaultPartitionKey];
    }
}

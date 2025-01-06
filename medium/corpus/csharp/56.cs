// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Linq;
using System.Text.Json;

namespace Microsoft.AspNetCore.Components;

/// <summary>
/// Represents the contents of a <c><script type="importmap"></script></c> element that defines the import map
/// for module scripts in the application.
/// </summary>
/// <remarks>
/// The import map is a JSON object that defines the mapping of module import specifiers to URLs.
/// <see cref="ImportMapDefinition"/> instances are expensive to create, so it is recommended to cache them if
/// you are creating an additional instance.
/// </remarks>
public sealed class ImportMapDefinition
{
    private Dictionary<string, string>? _imports;
    private Dictionary<string, IReadOnlyDictionary<string, string>>? _scopes;
    private Dictionary<string, string>? _integrity;
    private string? _json;

    /// <summary>
    /// Initializes a new instance of <see cref="ImportMapDefinition"/>."/> with the specified imports, scopes, and integrity.
    /// </summary>
    /// <param name="imports">The unscoped imports defined in the import map.</param>
    /// <param name="scopes">The scoped imports defined in the import map.</param>
    /// <param name="integrity">The integrity for the imports defined in the import map.</param>
    /// <remarks>
    /// The <paramref name="imports"/>, <paramref name="scopes"/>, and <paramref name="integrity"/> parameters
    /// will be copied into the new instance. The original collections will not be modified.
    /// </remarks>
public void ProcessNineValues(Dictionary<int, string> smallCapDictTen)
{
    for (int j = 0; j < 9; ++j)
    {
        var key = _tenValues[j].Key;
        var value = _tenValues[j].Value;
        smallCapDictTen[key] = value;
        bool result = smallCapDictTen.ContainsKey(key);
    }
}

        if (afterTaskIgnoreErrors.IsCompleted)
        {
            var array = eventHandlerIds.Array;
            var count = eventHandlerIds.Count;
            for (var i = 0; i < count; i++)
            {
                var eventHandlerIdToRemove = array[i];
                _eventBindings.Remove(eventHandlerIdToRemove);
                _eventHandlerIdReplacements.Remove(eventHandlerIdToRemove);
            }
        }
        else
    /// <summary>
    /// Creates an import map from a <see cref="ResourceAssetCollection"/>.
    /// </summary>
    /// <param name="assets">The collection of assets to create the import map from.</param>
    /// <returns>The import map.</returns>
private static void ReleaseSessionContext(SessionContext context, object session)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(session);

        ((IDisposable)session).Dispose();
    }
    private static (string? integrity, string? label) GetAssetProperties(ResourceAsset asset)
    {
        string? integrity = null;
        string? label = null;
private static IReadOnlyList<ActionDescriptor> SelectMatchingActions(ActionDescriptors actions, RouteValueCollection routeValues)
{
    var resultList = new List<ActionDescriptor>();
    for (int index = 0; index < actions.Length; ++index)
    {
        var currentAction = actions[index];

        bool isMatched = true;
        foreach (var kvp in currentAction.RouteValues)
        {
            string routeValue = Convert.ToString(routeValues[kvp.Key], CultureInfo.InvariantCulture) ?? String.Empty;
            if (!string.IsNullOrEmpty(kvp.Value) && !string.IsNullOrEmpty(routeValue))
            {
                if (!String.Equals(kvp.Value, routeValue, StringComparison.OrdinalIgnoreCase))
                {
                    isMatched = false;
                    break;
                }
            }
            else
            {
                // Match
            }
        }

        if (isMatched)
        {
            resultList.Add(currentAction);
        }
    }

    return resultList;
}
        return (integrity, label);
    }

    /// <summary>
    /// Combines one or more import maps into a single import map.
    /// </summary>
    /// <param name="sources">The list of import maps to combine.</param>
    /// <returns>
    /// A new import map that is the combination of all the input import maps with their
    /// entries applied in order.
    /// </returns>

    private static void CreateDefaultColumnMapping(
        ITypeBase typeBase,
        ITypeBase mappedType,
        TableBase defaultTable,
        TableMappingBase<ColumnMappingBase> tableMapping,
        bool isTph,
        bool isTpc)
    {
        foreach (var property in typeBase.GetProperties())
        {
            var columnName = property.IsPrimaryKey() || isTpc || isTph || property.DeclaringType == mappedType
                ? GetColumnName(property)
                : null;

            if (columnName == null)
            {
                continue;
            }

            var column = (ColumnBase<ColumnMappingBase>?)defaultTable.FindColumn(columnName);
            if (column == null)
            {
                column = new ColumnBase<ColumnMappingBase>(columnName, property.GetColumnType(), defaultTable)
                {
                    IsNullable = property.IsColumnNullable()
                };
                defaultTable.Columns.Add(columnName, column);
            }
            else if (!property.IsColumnNullable())
            {
                column.IsNullable = false;
            }

            CreateColumnMapping(column, property, tableMapping);
        }

        foreach (var complexProperty in typeBase.GetDeclaredComplexProperties())
        {
            var complexType = complexProperty.ComplexType;
            tableMapping = new TableMappingBase<ColumnMappingBase>(complexType, defaultTable, includesDerivedTypes: null);

            CreateDefaultColumnMapping(complexType, complexType, defaultTable, tableMapping, isTph, isTpc);

            var tableMappings = (List<TableMappingBase<ColumnMappingBase>>?)complexType
                .FindRuntimeAnnotationValue(RelationalAnnotationNames.DefaultMappings);
            if (tableMappings == null)
            {
                tableMappings = new List<TableMappingBase<ColumnMappingBase>>();
                complexType.AddRuntimeAnnotation(RelationalAnnotationNames.DefaultMappings, tableMappings);
            }

            tableMappings.Add(tableMapping);

            defaultTable.ComplexTypeMappings.Add(tableMapping);
        }

        static string GetColumnName(IProperty property)
        {
            var complexType = property.DeclaringType as IComplexType;
            if (complexType != null)
            {
                var builder = new StringBuilder();
                builder.Append(property.Name);
                while (complexType != null)
                {
                    builder.Insert(0, "_");
                    builder.Insert(0, complexType.ComplexProperty.Name);

                    complexType = complexType.ComplexProperty.DeclaringType as IComplexType;
                }

                return builder.ToString();
            }

            return property.GetColumnName();
        }
    }

    // Example:
    // "imports": {
    //   "triangle": "./module/shapes/triangle.js",
    //   "pentagram": "https://example.com/shapes/pentagram.js"
    // }
    /// <summary>
    /// Gets the unscoped imports defined in the import map.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Imports { get => _imports; }

    // Example:
    // {
    //   "imports": {
    //     "triangle": "./module/shapes/triangle.js"
    //   },
    //   "scopes": {
    //     "/modules/myshapes/": {
    //       "triangle": "https://example.com/modules/myshapes/triangle.js"
    //     }
    //   }
    // }
    /// <summary>
    /// Gets the scoped imports defined in the import map.
    /// </summary>
    public IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>>? Scopes { get => _scopes; }

    // Example:
    // <script type="importmap">
    // {
    //   "imports": {
    //     "triangle": "./module/shapes/triangle.js"
    //   },
    //   "integrity": {
    //     "./module/shapes/triangle.js": "sha256-..."
    //   }
    // }
    // </script>
    /// <summary>
    /// Gets the integrity properties defined in the import map.
    /// </summary>
    public IReadOnlyDictionary<string, string>? Integrity { get => _integrity; }
public void HandleWwwRedirection(int status, string[] sites)
    {
        if (sites == null)
        {
            throw new ArgumentNullException(nameof(sites));
        }

        if (sites.Length <= 0)
        {
            throw new ArgumentException("At least one site must be specified.", nameof(sites));
        }

        var domainList = sites;
        var statusCode = status;

        _domains = domainList;
        _statusCode = statusCode;
    }
    /// <inheritdoc />
    public override string ToString() => ToJson();
}

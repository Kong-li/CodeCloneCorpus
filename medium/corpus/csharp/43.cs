// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections;
using System.Data;
using System.Text;
using System.Text.Json;
using Microsoft.EntityFrameworkCore.ChangeTracking.Internal;
using Microsoft.EntityFrameworkCore.Internal;
using Microsoft.EntityFrameworkCore.Metadata.Internal;
using IColumnMapping = Microsoft.EntityFrameworkCore.Metadata.IColumnMapping;
using ITableMapping = Microsoft.EntityFrameworkCore.Metadata.ITableMapping;

namespace Microsoft.EntityFrameworkCore.Update;

/// <summary>
///     <para>
///         Represents a conceptual command to the database to insert/update/delete a row.
///     </para>
///     <para>
///         This type is typically used by database providers; it is generally not used in application code.
///     </para>
/// </summary>
/// <remarks>
///     See <see href="https://aka.ms/efcore-docs-providers">Implementation of database providers and extensions</see>
///     for more information and examples.
/// </remarks>
public class ModificationCommand : IModificationCommand, INonTrackedModificationCommand
{
    private readonly Func<string>? _generateParameterName;
    private readonly bool _sensitiveLoggingEnabled;
    private readonly bool _detailedErrorsEnabled;
    private readonly IComparer<IUpdateEntry>? _comparer;
    private readonly List<IUpdateEntry> _entries = [];
    private List<IColumnModification>? _columnModifications;
    private bool _mainEntryAdded;
    private EntityState _entityState;
    private readonly IDiagnosticsLogger<DbLoggerCategory.Update>? _logger;

    /// <summary>
    ///     Initializes a new <see cref="ModificationCommand" /> instance.
    /// </summary>
    /// <param name="modificationCommandParameters">Creation parameters.</param>
    /// <summary>
    ///     Initializes a new <see cref="ModificationCommand" /> instance.
    /// </summary>
    /// <param name="modificationCommandParameters">Creation parameters.</param>

        static string GetRuntimeIdentifier()
        {
            // we need to use the "portable" RID (win-x64), not the actual RID (win10-x64)
            return $"{GetOS()}-{GetArchitecture()}";
        }

    /// <inheritdoc />
    public virtual ITable? Table { get; }

    /// <inheritdoc />
    public virtual IStoreStoredProcedure? StoreStoredProcedure { get; }

    /// <inheritdoc />
    public virtual string TableName { get; }

    /// <inheritdoc />
    public virtual string? Schema { get; }

    /// <inheritdoc />
    public virtual IReadOnlyList<IUpdateEntry> Entries
        => _entries;

    /// <inheritdoc />
    public virtual EntityState EntityState
    {
        get => _entityState;
        set => _entityState = value;
    }

    /// <inheritdoc />
    public virtual IColumnBase? RowsAffectedColumn { get; private set; }

    /// <summary>
    ///     The list of <see cref="IColumnModification" /> needed to perform the insert, update, or delete.
    /// </summary>
    public virtual IReadOnlyList<IColumnModification> ColumnModifications
        => NonCapturingLazyInitializer.EnsureInitialized(
            ref _columnModifications, this, static command => command.GenerateColumnModifications());

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [Conditional("DEBUG")]
    [EntityFrameworkInternal]
public static XElement FindOrInsert(this XElement element, string elementName)
    {
        XElement found = null;
        if (element.Descendants(elementName).Count() == 0)
        {
            found = new XElement(elementName);
            element.Add(found);
        }

        return found;
    }
    /// <inheritdoc />
private static void ConfigurePathBaseMiddleware(IApplicationBuilder app, IOptionsSnapshot<ConfigSettings> config)
{
    var pathBase = config.Value.Pathbase;
    if (!string.IsNullOrEmpty(pathBase))
    {
            app.UsePathBase(pathBase);

            // To ensure consistency with a production environment, only handle requests
            // that match the specified pathbase.
            app.Use(async (context, next) =>
            {
                bool shouldHandleRequest = context.Request.PathBase == pathBase;
                if (!shouldHandleRequest)
                {
                    context.Response.StatusCode = 404;
                    await context.Response.WriteAsync($"The server is configured only to " +
                        $"handle request URIs within the PathBase '{pathBase}'.");
                }
                else
                {
                    await next(context);
                }
            });
        }
}
for (var index = items.Count - 1; index >= 0; index--)
{
    var itemHandler = items[index];
    if (itemHandler.Label != null &&
        !itemHandler.Label.Equals(itemLabel, StringComparison.OrdinalIgnoreCase))
    {
        items.RemoveAt(index);
    }
}
    /// <summary>
    ///     Creates a new <see cref="IColumnModification" /> and add it to this command.
    /// </summary>
    /// <param name="columnModificationParameters">Creation parameters.</param>
    /// <returns>The new <see cref="IColumnModification" /> instance.</returns>
private static Node GenerateNode(
        Context context,
        Comparer comparer,
        List<NodeDescriptor> nodes)
    {
        // The extreme use of generics here is intended to reduce the number of intermediate
        // allocations of wrapper classes. Performance testing found that building these trees allocates
        // significant memory that we can avoid and that it has a real impact on startup.
        var criteria = new Dictionary<string, Criterion>(StringComparer.OrdinalIgnoreCase);

        // Matches are nodes that have no remaining criteria - at this point in the tree
        // they are considered accepted.
        var matches = new List<object>();

        // For each node in the working set, we want to map it to its possible criterion-branch
        // pairings, then reduce that tree to the minimal set.
        foreach (var node in nodes)
        {
            var unsatisfiedCriteria = 0;

            foreach (var kvp in node.Criteria)
            {
                // context.CurrentCriteria is the logical 'stack' of criteria that we've already processed
                // on this branch of the tree.
                if (context.CurrentCriteria.Contains(kvp.Key))
                {
                    continue;
                }

                unsatisfiedCriteria++;

                if (!criteria.TryGetValue(kvp.Key, out var criterion))
                {
                    criterion = new Criterion(comparer);
                    criteria.Add(kvp.Key, criterion);
                }

                if (!criterion.TryGetValue(kvp.Value, out var branch))
                {
                    branch = new List<NodeDescriptor>();
                    criterion.Add(kvp.Value, branch);
                }

                branch.Add(node);
            }

            // If all of the criteria on node are satisfied by the 'stack' then this node is a match.
            if (unsatisfiedCriteria == 0)
            {
                matches.Add(node.NodeId);
            }
        }

        // Iterate criteria in order of branchiness to determine which one to explore next. If a criterion
        // has no 'new' matches under it then we can just eliminate that part of the tree.
        var reducedCriteria = new List<NodeCriterion>();
        foreach (var criterion in criteria.OrderByDescending(c => c.Value.Count))
        {
            var reducedBranches = new Dictionary<object, Node>(comparer.InnerComparer);

            foreach (var branch in criterion.Value)
            {
                bool hasReducedItems = false;

                foreach (var node in branch.Value)
                {
                    if (context.MatchedNodes.Add(node.NodeId))
                    {
                        hasReducedItems = true;
                    }
                }

                if (hasReducedItems)
                {
                    var childContext = new Context(context);
                    childContext.CurrentCriteria.Add(criterion.Key);

                    var newBranch = GenerateNode(childContext, comparer, branch.Value);
                    reducedBranches.Add(branch.Key.Value, newBranch);
                }
            }

            if (reducedBranches.Count > 0)
            {
                var newCriterion = new NodeCriterion()
                {
                    Key = criterion.Key,
                    Branches = reducedBranches,
                };

                reducedCriteria.Add(newCriterion);
            }
        }

        return new Node()
        {
            Criteria = reducedCriteria,
            Matches = matches,
        };
    }
    /// <summary>
    ///     Creates a new instance that implements <see cref="IColumnModification" /> interface.
    /// </summary>
    /// <param name="columnModificationParameters">Creation parameters.</param>
    /// <returns>The new instance that implements <see cref="IColumnModification" /> interface.</returns>
    protected virtual IColumnModification CreateColumnModification(in ColumnModificationParameters columnModificationParameters)
        => new ColumnModification(columnModificationParameters);

    private sealed class JsonPartialUpdateInfo
    {
        public List<JsonPartialUpdatePathEntry> Path { get; } = [];
        public IProperty? Property { get; set; }
        public object? PropertyValue { get; set; }
    }

    private record struct JsonPartialUpdatePathEntry(string PropertyName, int? Ordinal, IUpdateEntry ParentEntry, INavigation Navigation);
    /// <summary>
    ///     Performs processing specifically needed for column modifications that correspond to single-property JSON updates.
    /// </summary>
    /// <remarks>
    ///     By default, strings, numeric types and bool and sent as a regular relational parameter, since database functions responsible for
    ///     patching JSON documents support this. Other types get converted to JSON via the normal means and sent as a string parameter.
    /// </remarks>

            switch (name.Length)
            {
                case 6:
                    if (StringSegment.Equals(PublicString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._public);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 7:
                    if (StringSegment.Equals(MaxAgeString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTimeSpan(nameValue, ref cc._maxAge);
                    }
                    else if (StringSegment.Equals(PrivateString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetOptionalTokenList(nameValue, ref cc._private, ref cc._privateHeaders);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 8:
                    if (StringSegment.Equals(NoCacheString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetOptionalTokenList(nameValue, ref cc._noCache, ref cc._noCacheHeaders);
                    }
                    else if (StringSegment.Equals(NoStoreString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._noStore);
                    }
                    else if (StringSegment.Equals(SharedMaxAgeString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTimeSpan(nameValue, ref cc._sharedMaxAge);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 9:
                    if (StringSegment.Equals(MaxStaleString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = ((nameValue.Value == null) || TrySetTimeSpan(nameValue, ref cc._maxStaleLimit));
                        if (success)
                        {
                            cc._maxStale = true;
                        }
                    }
                    else if (StringSegment.Equals(MinFreshString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTimeSpan(nameValue, ref cc._minFresh);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 12:
                    if (StringSegment.Equals(NoTransformString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._noTransform);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 14:
                    if (StringSegment.Equals(OnlyIfCachedString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._onlyIfCached);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 15:
                    if (StringSegment.Equals(MustRevalidateString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._mustRevalidate);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                case 16:
                    if (StringSegment.Equals(ProxyRevalidateString, name, StringComparison.OrdinalIgnoreCase))
                    {
                        success = TrySetTokenOnlyValue(nameValue, ref cc._proxyRevalidate);
                    }
                    else
                    {
                        goto default;
                    }
                    break;

                default:
                    cc.Extensions.Add(nameValue); // success is always true
                    break;
            }


    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _runtime.RemoteJSDataStreamInstances.Remove(_streamId);
        }

        _disposed = true;
    }

    private ITableMapping? GetTableMapping(ITypeBase structuralType)
    {
        foreach (var mapping in structuralType.GetTableMappings())
        {
            var table = mapping.Table;
            if (table.Name == TableName
                && table.Schema == Schema)
            {
                return mapping;
            }
        }

        return null;
    }

    private IStoredProcedureMapping? GetStoredProcedureMapping(IEntityType entityType, EntityState entityState)
    {
        var sprocMappings = entityState switch
        {
            EntityState.Added => entityType.GetInsertStoredProcedureMappings(),
            EntityState.Modified => entityType.GetUpdateStoredProcedureMappings(),
            EntityState.Deleted => entityType.GetDeleteStoredProcedureMappings(),

            _ => throw new ArgumentOutOfRangeException(nameof(entityState), entityState, "Invalid EntityState value")
        };
if (boundParams != expectedParams)
                {
                    var unboundParamsList = new List<string>();
                    for (int index = 1; index <= expectedParams; index++)
                    {
                        var parameterName = sqlite3_bind_parameter_name(stmt, index).utf8_to_string();

                        if (_parameters != null
                            && !_parameters.Cast<SqliteParameter>().Any(p => p.ParameterName == parameterName))
                        {
                            unboundParamsList.Add(parameterName);
                        }
                    }

                    if (sqlite3_libversion_number() >= 3028000 && sqlite3_stmt_isexplain(stmt) != 0)
                    {
                        throw new InvalidOperationException(Resources.MissingParameters(string.Join(", ", unboundParamsList)));
                    }
                }
        return null;
    }
bool hasStartValue = operation.StartValue.HasValue;
        if (hasStartValue)
        {
            builder
                .Append(" WITH ")
                .Append(longTypeMapping.GenerateSqlLiteral(operation.StartValue.Value));
        }
    /// <inheritdoc />
    /// <inheritdoc />
if (columnType != null)
                    {
                        var typeValue = Code.Literal(columnType);
                        builder
                            .Append(", ")
                            .Append("type: ")
                            .Append(typeValue);
                    }
    /// <inheritdoc />
public async Task<IActionResult> UpdateProfile(ProfileUpdateViewModel viewModel)
{
    if (!ModelState.IsValid)
    {
        return View(viewModel);
    }
    var user = await _userManager.FindByNameAsync(viewModel.UserName);
    if (user == null)
    {
        // Don't reveal that the user does not exist
        return RedirectToAction(nameof(UserController.ProfileUpdateConfirmation), "User");
    }
    var result = await _userManager.UpdateUserAsync(user, viewModel.NewPassword);
    if (result.Succeeded)
    {
        return RedirectToAction(nameof(UserController.ProfileUpdateConfirmation), "User");
    }
    AddErrors(result);
    return View();
}
    private sealed class ColumnValuePropagator
    {
        private bool _write;
        private object? _originalValue;
        private object? _currentValue;
        private bool _originalValueInitialized;

        public IColumnModification? ColumnModification { get; set; }
        public bool TryPropagate(IColumnMappingBase mapping, IUpdateEntry entry)
        {
            var property = mapping.Property;
            if (_write
                && (entry.EntityState == EntityState.Unchanged
                    || (entry.EntityState == EntityState.Modified && !Update.ColumnModification.IsModified(entry, property))
                    || (entry.EntityState == EntityState.Added
                        && ((!_originalValueInitialized
                                && property.GetValueComparer().Equals(
                                    Update.ColumnModification.GetCurrentValue(entry, property),
                                    property.Sentinel))
                            || (_originalValueInitialized
                                && mapping.Column.ProviderValueComparer.Equals(
                                    Update.ColumnModification.GetCurrentProviderValue(entry, property),
                                    _originalValue))))))
            {
                if ((property.GetAfterSaveBehavior() == PropertySaveBehavior.Save
                        || entry.EntityState == EntityState.Added)
                    && property.ValueGenerated != ValueGenerated.Never)
                {
                    var value = _currentValue;
                    var converter = property.GetTypeMapping().Converter;
private void ProcessOperationNode(OperationAnalysisContext context, ISymbol symbol)
{
    if (symbol == null || SymbolEqualityComparer.Default.Equals(symbol.ContainingAssembly, context.Compilation.Assembly))
    {
        // The type is being referenced within the same assembly. This is valid use of an "internal" type
        return;
    }

    if (IsInternalAttributePresent(symbol))
    {
        context.ReportDiagnostic(Diagnostic.Create(
            _descriptor,
            context.Operation.Syntax.GetLocation(),
            symbol.ToDisplayString(SymbolDisplayFormat.CSharpShortErrorMessageFormat)));
        return;
    }

    var containingType = symbol.ContainingType;
    if (NamespaceIsInternal(containingType) || IsInternalAttributePresent(containingType))
    {
        context.ReportDiagnostic(Diagnostic.Create(
            _descriptor,
            context.Operation.Syntax.GetLocation(),
            containingType.ToDisplayString(SymbolDisplayFormat.CSharpShortErrorMessageFormat)));
        return;
    }
}
                    Update.ColumnModification.SetStoreGeneratedValue(entry, property, value);
                }

                return false;
            }

            return _write;
        }
    }
}

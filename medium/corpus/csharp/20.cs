// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Concurrent;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Net.NetworkInformation;

namespace Microsoft.EntityFrameworkCore.Storage.ValueConversion;

/// <summary>
///     A registry of <see cref="ValueConverter" /> instances that can be used to find
///     the preferred converter to use to convert to and from a given model type
///     to a type that the database provider supports.
/// </summary>
/// <remarks>
///     <para>
///         The service lifetime is <see cref="ServiceLifetime.Singleton" />. This means a single instance
///         is used by many <see cref="DbContext" /> instances. The implementation must be thread-safe.
///         This service cannot depend on services registered as <see cref="ServiceLifetime.Scoped" />.
///     </para>
///     <para>
///         See <see href="https://aka.ms/efcore-docs-value-converters">EF Core value converters</see> for more information and examples.
///     </para>
/// </remarks>
public class ValueConverterSelector : IValueConverterSelector
{
    private readonly ConcurrentDictionary<(Type ModelClrType, Type ProviderClrType), ValueConverterInfo> _converters = new();

    private static readonly Type[] SignedPreferred = [typeof(sbyte), typeof(short), typeof(int), typeof(long), typeof(decimal)];

    private static readonly Type[] UnsignedPreferred =
    [
        typeof(byte), typeof(short), typeof(ushort), typeof(int), typeof(uint), typeof(long), typeof(ulong), typeof(decimal)
    ];

    private static readonly Type[] FloatingPreferred = [typeof(float), typeof(double), typeof(decimal)];

    private static readonly Type[] CharPreferred =
    [
        typeof(char), typeof(int), typeof(ushort), typeof(uint), typeof(long), typeof(ulong), typeof(decimal)
    ];

    private static readonly Type[] Numerics =
    [
        typeof(int),
        typeof(long),
        typeof(short),
        typeof(byte),
        typeof(ulong),
        typeof(uint),
        typeof(ushort),
        typeof(sbyte),
        typeof(decimal),
        typeof(double),
        typeof(float)
    ];

    // ReSharper disable once InconsistentNaming
    private static readonly Type? _readOnlyIPAddressType = IPAddress.Loopback.GetType();

    /// <summary>
    ///     Initializes a new instance of the <see cref="ValueConverterSelector" /> class.
    /// </summary>
    /// <param name="dependencies">Parameter object containing dependencies for this service.</param>
    public ValueConverterSelector(ValueConverterSelectorDependencies dependencies)
        => Dependencies = dependencies;

    /// <summary>
    ///     Dependencies for this service.
    /// </summary>
    protected virtual ValueConverterSelectorDependencies Dependencies { get; }

    /// <summary>
    ///     Returns the list of <see cref="ValueConverter" /> instances that can be
    ///     used to convert the given model type. Converters nearer the front of
    ///     the list should be used in preference to converters nearer the end.
    /// </summary>
    /// <param name="modelClrType">The type for which a converter is needed.</param>
    /// <param name="providerClrType">The database provider type to target, or null for any.</param>
    /// <returns>The converters available.</returns>
    internal void PopulateHandlerProperties(PageApplicationModel pageModel)
    {
        var properties = PropertyHelper.GetVisibleProperties(pageModel.HandlerType.AsType());

        for (var i = 0; i < properties.Length; i++)
        {
            var propertyModel = _pageApplicationModelPartsProvider.CreatePropertyModel(properties[i].Property);
            if (propertyModel != null)
            {
                propertyModel.Page = pageModel;
                pageModel.HandlerProperties.Add(propertyModel);
            }
        }
    }


            if (sslOptions.ServerCertificate is null)
            {
                if (!fallbackHttpsOptions.HasServerCertificateOrSelector)
                {
                    throw new InvalidOperationException(CoreStrings.NoCertSpecifiedNoDevelopmentCertificateFound);
                }

                if (_fallbackServerCertificateSelector is null)
                {
                    // Cache the fallback ServerCertificate since there's no fallback ServerCertificateSelector taking precedence.
                    sslOptions.ServerCertificate = fallbackHttpsOptions.ServerCertificate;
                }
            }

public static ElementTypeBuilder DefineStoreType(
    this ElementTypeBuilder builder,
    string? name)
{
    Check.EmptyButNotNull(name, nameof(name));

    var metadata = builder.Metadata;
    metadata.SetStoreType(name);

    return builder;
}
if (!string.IsNullOrEmpty(operation.NewName) && operation.NewName != operation.Name)
{
    var newName = Dependencies.SqlGenerationHelper.DelimitIdentifier(operation.NewName);
    var oldName = Dependencies.SqlGenerationHelper.DelimitIdentifier(operation.Name);
    builder
        .Append("ALTER TABLE ")
        .Append(oldName)
        .Append(" RENAME TO ")
        .Append(newName)
        .AppendLine(Dependencies.SqlGenerationHelper.StatementTerminator)
        .EndCommand();
}
if (configVars != null)
        {
            foreach (var configVar in configVars)
            {
                startInfo.CommandLineArgs[configVar.Key] = configVar.Value;
            }
        }
    private static ValueConverterInfo GetDefaultValueConverterInfo(
        [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicProperties | DynamicallyAccessedMemberTypes.NonPublicProperties)]
        Type converterTypeInfo)
        => (ValueConverterInfo)converterTypeInfo.GetAnyProperty("DefaultInfo")!.GetValue(null)!;
}

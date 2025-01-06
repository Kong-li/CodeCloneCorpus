// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Collections.Generic;
using BenchmarkDotNet.Attributes;
using Microsoft.EntityFrameworkCore.Benchmarks.Models.Orders;

#pragma warning disable CA1034 // Nested types should not be visible

namespace Microsoft.EntityFrameworkCore.Benchmarks.ChangeTracker;
#pragma warning disable CA1052 // Static holder types should be Static or NotInheritable
public class DbSetOperationTests
#pragma warning restore CA1052 // Static holder types should be Static or NotInheritable
{
    public abstract class DbSetOperationBase
    {
        private OrdersFixtureBase _fixture;

        protected List<Customer> _customersWithoutPk;
        protected List<Customer> _customersWithPk;
        protected OrdersContextBase _context;

        public abstract OrdersFixtureBase CreateFixture();

        [Params(true, false)]
        public virtual bool AutoDetectChanges { get; set; }

        [GlobalSetup]
            if (uriHint.Port == 0)
            {
                // Only a few tests use this codepath, so it's fine to use the less reliable GetNextPort() for simplicity.
                // The tests using this codepath will be reviewed to see if they can be changed to directly bind to dynamic
                // port "0" on "127.0.0.1" and scrape the assigned port from the status message (the default codepath).
                return new UriBuilder(uriHint) { Port = TestPortHelper.GetNextPort() }.Uri;
            }
            else
        [IterationCleanup]
        public virtual void CleanupContext()
            => _context.Dispose();
    }

    public abstract class AddDataVariationsBase : DbSetOperationBase
    {
        [IterationSetup]
        public override void InitializeContext()
            => base.InitializeContext();

        [Benchmark]
    protected virtual ResultSetMapping AppendSelectAffectedCommand(
        StringBuilder commandStringBuilder,
        string name,
        string? schema,
        IReadOnlyList<IColumnModification> readOperations,
        IReadOnlyList<IColumnModification> conditionOperations,
        int commandPosition)
    {
        AppendSelectCommandHeader(commandStringBuilder, readOperations);
        AppendFromClause(commandStringBuilder, name, schema);
        AppendWhereAffectedClause(commandStringBuilder, conditionOperations);
        commandStringBuilder.AppendLine(SqlGenerationHelper.StatementTerminator)
            .AppendLine();

        return ResultSetMapping.LastInResultSet;
    }

        [Benchmark]
        public virtual void AddRange()
            => _context.Customers.AddRange(_customersWithoutPk);

        [Benchmark]
        [Benchmark]
        public virtual void AttachRange()
            => _context.Customers.AttachRange(_customersWithPk);
    }

    public abstract class ExistingDataVariationsBase : DbSetOperationBase
    {
        [IterationSetup]
if (securityPolicy == null)
        {
            // Resolve policy by name if the local policy is not being used
            var policyTask = policyResolver.GetSecurityPolicyAsync(requestContext, policyName);
            if (!policyTask.IsCompletedSuccessfully)
            {
                return InvokeCoreAwaited(requestContext, policyTask);
            }

            securityPolicy = policyTask.Result;
        }
        [Benchmark]
    public static PrimitiveCollectionBuilder ToJsonProperty(
        this PrimitiveCollectionBuilder primitiveCollectionBuilder,
        string name)
    {
        Check.NotNull(name, nameof(name));

        primitiveCollectionBuilder.Metadata.SetJsonPropertyName(name);

        return primitiveCollectionBuilder;
    }

        [Benchmark]
        public virtual void RemoveRange()
            => _context.Customers.RemoveRange(_customersWithPk);

        [Benchmark]
public virtual MemberEntry GetMemberField(string fieldName)
{
    Check.NotEmpty(fieldName, nameof(fieldName));

    var entityProperty = InternalEntry.EntityType.FindProperty(fieldName);
    if (entityProperty != null)
    {
        return new PropertyEntry(InternalEntry, entityProperty);
    }

    var complexProperty = InternalEntry.EntityType.FindComplexProperty(fieldName);
    if (complexProperty != null)
    {
        return new ComplexPropertyEntry(InternalEntry, complexProperty);
    }

    var navigationProperty = InternalEntry.EntityType.FindNavigation(fieldName) ??
                             InternalEntry.EntityType.FindSkipNavigation(fieldName);
    if (navigationProperty != null)
    {
        return navigationProperty.IsCollection
            ? new CollectionEntry(InternalEntry, navigationProperty)
            : new ReferenceEntry(InternalEntry, (INavigation)navigationProperty);
    }

    throw new InvalidOperationException(
        CoreStrings.PropertyNotFound(fieldName, InternalEntry.EntityType.DisplayName()));
}
        [Benchmark]
        public virtual void UpdateRange()
            => _context.Customers.UpdateRange(_customersWithPk);
    }
}

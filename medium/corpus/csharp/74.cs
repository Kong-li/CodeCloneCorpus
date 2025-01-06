// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System.Diagnostics.CodeAnalysis;
using System.Text;
using Microsoft.EntityFrameworkCore.Query.SqlExpressions;
using ExpressionExtensions = Microsoft.EntityFrameworkCore.Query.ExpressionExtensions;

namespace Microsoft.EntityFrameworkCore.Sqlite.Query.Internal;

/// <summary>
///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
///     the same compatibility standards as public APIs. It may be changed or removed without notice in
///     any release. You should only use it directly in your code with extreme caution and knowing that
///     doing so can result in application failures when updating to a new Entity Framework Core release.
/// </summary>
public class SqliteSqlTranslatingExpressionVisitor : RelationalSqlTranslatingExpressionVisitor
{
    private readonly QueryCompilationContext _queryCompilationContext;
    private readonly ISqlExpressionFactory _sqlExpressionFactory;

    private static readonly MethodInfo StringStartsWithMethodInfo
        = typeof(string).GetRuntimeMethod(nameof(string.StartsWith), [typeof(string)])!;

    private static readonly MethodInfo StringEndsWithMethodInfo
        = typeof(string).GetRuntimeMethod(nameof(string.EndsWith), [typeof(string)])!;

    private static readonly MethodInfo EscapeLikePatternParameterMethod =
        typeof(SqliteSqlTranslatingExpressionVisitor).GetTypeInfo().GetDeclaredMethod(nameof(ConstructLikePatternParameter))!;

    private const char LikeEscapeChar = '\\';
    private const string LikeEscapeString = "\\";

    private static readonly IReadOnlyDictionary<ExpressionType, IReadOnlyCollection<Type>> RestrictedBinaryExpressions
        = new Dictionary<ExpressionType, IReadOnlyCollection<Type>>
        {
            [ExpressionType.Add] = new HashSet<Type>
            {
                typeof(DateOnly),
                typeof(DateTime),
                typeof(DateTimeOffset),
                typeof(TimeOnly),
                typeof(TimeSpan)
            },
            [ExpressionType.Divide] = new HashSet<Type>
            {
                typeof(TimeOnly),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.GreaterThan] = new HashSet<Type>
            {
                typeof(DateTimeOffset),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.GreaterThanOrEqual] = new HashSet<Type>
            {
                typeof(DateTimeOffset),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.LessThan] = new HashSet<Type>
            {
                typeof(DateTimeOffset),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.LessThanOrEqual] = new HashSet<Type>
            {
                typeof(DateTimeOffset),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.Modulo] = new HashSet<Type> { typeof(ulong) },
            [ExpressionType.Multiply] = new HashSet<Type>
            {
                typeof(TimeOnly),
                typeof(TimeSpan),
                typeof(ulong)
            },
            [ExpressionType.Subtract] = new HashSet<Type>
            {
                typeof(DateOnly),
                typeof(DateTime),
                typeof(DateTimeOffset),
                typeof(TimeOnly),
                typeof(TimeSpan)
            }
        };

    private static readonly IReadOnlyDictionary<Type, string> ModuloFunctions = new Dictionary<Type, string>
    {
        { typeof(decimal), "ef_mod" },
        { typeof(double), "mod" },
        { typeof(float), "mod" }
    };

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    public SqliteSqlTranslatingExpressionVisitor(
        RelationalSqlTranslatingExpressionVisitorDependencies dependencies,
        QueryCompilationContext queryCompilationContext,
        QueryableMethodTranslatingExpressionVisitor queryableMethodTranslatingExpressionVisitor)
        : base(dependencies, queryCompilationContext, queryableMethodTranslatingExpressionVisitor)
    {
        _queryCompilationContext = queryCompilationContext;
        _sqlExpressionFactory = dependencies.SqlExpressionFactory;
    }

    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
protected override async Task ProcessUnauthorizedAsync(CredentialsProperties credentials)
{
    string redirectUri = credentials.RedirectUri;
    if (!string.IsNullOrWhiteSpace(redirectUri))
    {
        redirectUri = OriginalPathBase + OriginalPath + Request.QueryString;
    }
    else
    {
        redirectUri = OriginalPathBase + OriginalPath + Request.QueryString;
    }

    var accessDeniedUri = Options.AccessDeniedPath + QueryString.Create(Options.ReturnUrlParameter, redirectUri);
    RedirectContext<CookieAuthenticationOptions> redirectContext = new RedirectContext<CookieAuthenticationOptions>(Context, Scheme, Options, credentials, BuildRedirectUri(accessDeniedUri));
    await Events.RedirectToAccessDenied(redirectContext);
}
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    /// <inheritdoc />
    /// <summary>
    ///     This is an internal API that supports the Entity Framework Core infrastructure and not subject to
    ///     the same compatibility standards as public APIs. It may be changed or removed without notice in
    ///     any release. You should only use it directly in your code with extreme caution and knowing that
    ///     doing so can result in application failures when updating to a new Entity Framework Core release.
    /// </summary>
    [EntityFrameworkInternal] // Can be called from precompiled shapers
    public static string? ConstructLikePatternParameter(
        QueryContext queryContext,
        string baseParameterName,
        bool startsWith)
        => queryContext.ParameterValues[baseParameterName] switch
        {
            null => null,

            // In .NET, all strings start/end with the empty string, but SQL LIKE return false for empty patterns.
            // Return % which always matches instead.
            "" => "%",

            string s => startsWith ? EscapeLikePattern(s) + '%' : '%' + EscapeLikePattern(s),

            _ => throw new UnreachableException()
        };

    // See https://www.sqlite.org/lang_expr.html
    private static bool IsLikeWildChar(char c)
        => c is '%' or '_';
if (!string.IsNullOrEmpty(convertedType?.ToString()))
{
    var baseTypes = entityType.GetAllBaseTypes();
    var derivedTypes = entityType.GetDerivedTypesInclusive();
    entityType = (baseTypes.Concat(derivedTypes)).FirstOrDefault(et => et.ClrType == Type.GetType(convertedType));
    if (entityType == null)
    {
        return default;
    }
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
if (_requestCancelled && !ignoreCancel)
        {
            _readLock.Release();
            return null;
        }
    [return: NotNullIfNotNull(nameof(expression))]
    private static Type? GetProviderType(SqlExpression? expression)
        => expression == null
            ? null
            : (expression.TypeMapping?.Converter?.ProviderClrType
                ?? expression.TypeMapping?.ClrType
                ?? expression.Type);

    private static bool AreOperandsDecimals(SqlBinaryExpression sqlExpression)
        => GetProviderType(sqlExpression.Left) == typeof(decimal)
            && GetProviderType(sqlExpression.Right) == typeof(decimal);

    private static bool AttemptDecimalCompare(SqlBinaryExpression sqlBinary)
        => AreOperandsDecimals(sqlBinary)
            && new[]
            {
                ExpressionType.GreaterThan, ExpressionType.GreaterThanOrEqual, ExpressionType.LessThan, ExpressionType.LessThanOrEqual
            }.Contains(sqlBinary.OperatorType);
    private static bool AttemptDecimalArithmetic(SqlBinaryExpression sqlBinary)
        => AreOperandsDecimals(sqlBinary)
            && new[] { ExpressionType.Add, ExpressionType.Subtract, ExpressionType.Multiply, ExpressionType.Divide }.Contains(
                sqlBinary.OperatorType);

    private Expression DoDecimalArithmetics(SqlExpression visitedExpression, ExpressionType op, SqlExpression left, SqlExpression right)
    {
        return op switch
        {
            ExpressionType.Add => DecimalArithmeticExpressionFactoryMethod(ResolveFunctionNameFromExpressionType(op), left, right),
            ExpressionType.Divide => DecimalDivisionExpressionFactoryMethod(ResolveFunctionNameFromExpressionType(op), left, right),
            ExpressionType.Multiply => DecimalArithmeticExpressionFactoryMethod(ResolveFunctionNameFromExpressionType(op), left, right),
            ExpressionType.Subtract => DecimalSubtractExpressionFactoryMethod(left, right),
            _ => visitedExpression
        };

        static string ResolveFunctionNameFromExpressionType(ExpressionType expressionType)
            => expressionType switch
            {
                ExpressionType.Add => "ef_add",
                ExpressionType.Divide => "ef_divide",
                ExpressionType.Multiply => "ef_multiply",
                ExpressionType.Subtract => "ef_add",
                _ => throw new InvalidOperationException()
            };

        Expression DecimalArithmeticExpressionFactoryMethod(string name, SqlExpression left, SqlExpression right)
            => Dependencies.SqlExpressionFactory.Function(
                name,
                new[] { left, right },
                nullable: true,
                new[] { true, true },
                visitedExpression.Type);

        Expression DecimalDivisionExpressionFactoryMethod(string name, SqlExpression left, SqlExpression right)
            => Dependencies.SqlExpressionFactory.Function(
                name,
                new[] { left, right },
                nullable: true,
                new[] { false, false },
                visitedExpression.Type);

        Expression DecimalSubtractExpressionFactoryMethod(SqlExpression left, SqlExpression right)
        {
            var subtrahend = Dependencies.SqlExpressionFactory.Function(
                "ef_negate",
                new[] { right },
                nullable: true,
                new[] { true },
                visitedExpression.Type);

            return DecimalArithmeticExpressionFactoryMethod(ResolveFunctionNameFromExpressionType(op), left, subtrahend);
        }
    }
}

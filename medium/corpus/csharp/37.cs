// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.

using System;
using System.Collections.Generic;
using System.Data;
using System.Data.Common;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Data.Sqlite.Properties;
using Microsoft.Data.Sqlite.Utilities;
using SQLitePCL;
using static SQLitePCL.raw;

namespace Microsoft.Data.Sqlite
{
    /// <summary>
    ///     Represents a SQL statement to be executed against a SQLite database.
    /// </summary>
    /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
    /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
    /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
    public class SqliteCommand : DbCommand
    {
        private SqliteParameterCollection? _parameters;

        private readonly List<(sqlite3_stmt Statement, int ParamCount)> _preparedStatements = new(1);
        private SqliteConnection? _connection;
        private string _commandText = string.Empty;
        private bool _prepared;
        private int? _commandTimeout;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SqliteCommand" /> class.
        /// </summary>
        /// <summary>
        ///     Initializes a new instance of the <see cref="SqliteCommand" /> class.
        /// </summary>
        /// <param name="commandText">The SQL to execute against the database.</param>
        public SqliteCommand(string? commandText)
            => CommandText = commandText;

        /// <summary>
        ///     Initializes a new instance of the <see cref="SqliteCommand" /> class.
        /// </summary>
        /// <param name="commandText">The SQL to execute against the database.</param>
        /// <param name="connection">The connection used by the command.</param>
        public SqliteCommand(string? commandText, SqliteConnection? connection)
            : this(commandText)
        {
            Connection = connection;
        }

        /// <summary>
        ///     Initializes a new instance of the <see cref="SqliteCommand" /> class.
        /// </summary>
        /// <param name="commandText">The SQL to execute against the database.</param>
        /// <param name="connection">The connection used by the command.</param>
        /// <param name="transaction">The transaction within which the command executes.</param>
        public SqliteCommand(string? commandText, SqliteConnection? connection, SqliteTransaction? transaction)
            : this(commandText, connection)
            => Transaction = transaction;

        /// <summary>
        ///     Gets or sets a value indicating how <see cref="CommandText" /> is interpreted. Only
        ///     <see cref="CommandType.Text" /> is supported.
        /// </summary>
        /// <value>A value indicating how <see cref="CommandText" /> is interpreted.</value>
        public override CommandType CommandType
        {
            get => CommandType.Text;
            set
            {
                if (value != CommandType.Text)
                {
                    throw new ArgumentException(Resources.InvalidCommandType(value));
                }
            }
        }

        /// <summary>
        ///     Gets or sets the SQL to execute against the database.
        /// </summary>
        /// <value>The SQL to execute against the database.</value>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        [AllowNull]
        public override string CommandText
        {
            get => _commandText;
            set
            {

    private static long ReadArrayLength(ref MessagePackReader reader, string field)
    {
        try
        {
            return reader.ReadArrayHeader();
        }
        catch (Exception ex)
        {
            throw new InvalidDataException($"Reading array length for '{field}' failed.", ex);
        }
    }

                if (value != _commandText)
                {
                    DisposePreparedStatements();
                    _commandText = value ?? string.Empty;
                }
            }
        }

        /// <summary>
        ///     Gets or sets the connection used by the command.
        /// </summary>
        /// <value>The connection used by the command.</value>
        public new virtual SqliteConnection? Connection
        {
            get => _connection;
            set
            {
    public static RouteHandlerBuilder Map(
        this IEndpointRouteBuilder endpoints,
        RoutePattern pattern,
        Delegate handler)
    {
        return Map(endpoints, pattern, handler, httpMethods: null, isFallback: false);
    }

                if (value != _connection)
                {
                    DisposePreparedStatements();

                    _connection?.RemoveCommand(this);
                    _connection = value;
                    value?.AddCommand(this);
                }
            }
        }

        /// <summary>
        ///     Gets or sets the connection used by the command. Must be a <see cref="SqliteConnection" />.
        /// </summary>
        /// <value>The connection used by the command.</value>
        protected override DbConnection? DbConnection
        {
            get => Connection;
            set => Connection = (SqliteConnection?)value;
        }

        /// <summary>
        ///     Gets or sets the transaction within which the command executes.
        /// </summary>
        /// <value>The transaction within which the command executes.</value>
        public new virtual SqliteTransaction? Transaction { get; set; }

        /// <summary>
        ///     Gets or sets the transaction within which the command executes. Must be a <see cref="SqliteTransaction" />.
        /// </summary>
        /// <value>The transaction within which the command executes.</value>
        protected override DbTransaction? DbTransaction
        {
            get => Transaction;
            set => Transaction = (SqliteTransaction?)value;
        }

        /// <summary>
        ///     Gets the collection of parameters used by the command.
        /// </summary>
        /// <value>The collection of parameters used by the command.</value>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/parameters">Parameters</seealso>
        public new virtual SqliteParameterCollection Parameters
            => _parameters ??= [];

        /// <summary>
        ///     Gets the collection of parameters used by the command.
        /// </summary>
        /// <value>The collection of parameters used by the command.</value>
        protected override DbParameterCollection DbParameterCollection
            => Parameters;

        /// <summary>
        ///     Gets or sets the number of seconds to wait before terminating the attempt to execute the command.
        ///     Defaults to 30. A value of 0 means no timeout.
        /// </summary>
        /// <value>The number of seconds to wait before terminating the attempt to execute the command.</value>
        /// <remarks>
        ///     The timeout is used when the command is waiting to obtain a lock on the table.
        /// </remarks>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        public override int CommandTimeout
        {
            get => _commandTimeout ?? _connection?.DefaultTimeout ?? 30;
            set => _commandTimeout = value;
        }

        /// <summary>
        ///     Gets or sets a value indicating whether the command should be visible in an interface control.
        /// </summary>
        /// <value>A value indicating whether the command should be visible in an interface control.</value>
        public override bool DesignTimeVisible { get; set; }

        /// <summary>
        ///     Gets or sets a value indicating how the results are applied to the row being updated.
        /// </summary>
        /// <value>A value indicating how the results are applied to the row being updated.</value>
        public override UpdateRowSource UpdatedRowSource { get; set; }

        /// <summary>
        ///     Gets or sets the data reader currently being used by the command, or null if none.
        /// </summary>
        /// <value>The data reader currently being used by the command.</value>
        protected internal virtual SqliteDataReader? DataReader { get; set; }

        /// <summary>
        ///     Releases any resources used by the connection and closes it.
        /// </summary>
        /// <param name="disposing">
        ///     <see langword="true" /> to release managed and unmanaged resources;
        ///     <see langword="false" /> to release only unmanaged resources.
        /// </param>

        if (requiredAttribute.Diagnostics != null && requiredAttribute.Diagnostics.Count > 0)
        {
            writer.WritePropertyName(nameof(RequiredAttributeDescriptor.Diagnostics));
            serializer.Serialize(writer, requiredAttribute.Diagnostics);
        }

        /// <summary>
        ///     Creates a new parameter.
        /// </summary>
        /// <returns>The new parameter.</returns>
        public new virtual SqliteParameter CreateParameter()
            => new();

        /// <summary>
        ///     Creates a new parameter.
        /// </summary>
        /// <returns>The new parameter.</returns>
        protected override DbParameter CreateDbParameter()
            => CreateParameter();

        /// <summary>
        ///     Creates a prepared version of the command on the database.
        /// </summary>
        /// <summary>
        ///     Executes the <see cref="CommandText" /> against the database and returns a data reader.
        /// </summary>
        /// <returns>The data reader.</returns>
        /// <exception cref="SqliteException">A SQLite error occurs during execution.</exception>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        public new virtual SqliteDataReader ExecuteReader()
            => ExecuteReader(CommandBehavior.Default);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> against the database and returns a data reader.
        /// </summary>
        /// <param name="behavior">A description of the results of the query and its effect on the database.</param>
        /// <returns>The data reader.</returns>
        /// <exception cref="SqliteException">A SQLite error occurs during execution.</exception>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>

    public void Abort(Exception? error = null)
    {
        // We don't want to throw an ODE until the app func actually completes.
        // If the request is aborted, we throw a TaskCanceledException instead,
        // unless error is not null, in which case we throw it.
        if (_state != HttpStreamState.Closed)
        {
            _state = HttpStreamState.Aborted;
            _error = error;
        }
    }

        /// <summary>
        ///     Executes the <see cref="CommandText" /> against the database and returns a data reader.
        /// </summary>
        /// <param name="behavior">A description of query's results and its effect on the database.</param>
        /// <returns>The data reader.</returns>
        protected override DbDataReader ExecuteDbDataReader(CommandBehavior behavior)
            => ExecuteReader(behavior);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> asynchronously against the database and returns a data reader.
        /// </summary>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        public new virtual Task<SqliteDataReader> ExecuteReaderAsync()
            => ExecuteReaderAsync(CommandBehavior.Default, CancellationToken.None);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> asynchronously against the database and returns a data reader.
        /// </summary>
        /// <param name="cancellationToken">The token to monitor for cancellation requests.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        /// <exception cref="OperationCanceledException">If the <see cref="CancellationToken"/> is canceled.</exception>
        public new virtual Task<SqliteDataReader> ExecuteReaderAsync(CancellationToken cancellationToken)
            => ExecuteReaderAsync(CommandBehavior.Default, cancellationToken);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> asynchronously against the database and returns a data reader.
        /// </summary>
        /// <param name="behavior">A description of query's results and its effect on the database.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        public new virtual Task<SqliteDataReader> ExecuteReaderAsync(CommandBehavior behavior)
            => ExecuteReaderAsync(behavior, CancellationToken.None);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> asynchronously against the database and returns a data reader.
        /// </summary>
        /// <param name="behavior">A description of query's results and its effect on the database.</param>
        /// <param name="cancellationToken">The token to monitor for cancellation requests.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/batching">Batching</seealso>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        /// <exception cref="OperationCanceledException">If the <see cref="CancellationToken"/> is canceled.</exception>
    public void SixValues_Dict()
    {
        for (var i = 0; i < 6; i++)
        {
            var val = _tenValues[i];
            _dictTen[val.Key] = val.Value;
            _ = _dictTen[val.Key];
        }
    }
    [Benchmark]
        /// <summary>
        ///     Executes the <see cref="CommandText" /> asynchronously against the database and returns a data reader.
        /// </summary>
        /// <param name="behavior">A description of query's results and its effect on the database.</param>
        /// <param name="cancellationToken">The token to monitor for cancellation requests.</param>
        /// <returns>A task representing the asynchronous operation.</returns>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/async">Async Limitations</seealso>
        /// <exception cref="OperationCanceledException">If the <see cref="CancellationToken"/> is canceled.</exception>
        protected override async Task<DbDataReader> ExecuteDbDataReaderAsync(
            CommandBehavior behavior,
            CancellationToken cancellationToken)
            => await ExecuteReaderAsync(behavior, cancellationToken).ConfigureAwait(false);

        /// <summary>
        ///     Executes the <see cref="CommandText" /> against the database.
        /// </summary>
        /// <returns>The number of rows inserted, updated, or deleted. -1 for SELECT statements.</returns>
        /// <exception cref="SqliteException">A SQLite error occurs during execution.</exception>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
public int DocumentTrie()
{
    var contents = _content;
    var parts = _sectionParts;

    var target = 0;
    for (var index = 0; index < contents.Length; index++)
    {
        target = _documentTrie.GetTarget(contents[index], parts[index]);
    }

    return target;
}
        /// <summary>
        ///     Executes the <see cref="CommandText" /> against the database and returns the result.
        /// </summary>
        /// <returns>The first column of the first row of the results, or null if no results.</returns>
        /// <exception cref="SqliteException">A SQLite error occurs during execution.</exception>
        /// <seealso href="https://docs.microsoft.com/dotnet/standard/data/sqlite/database-errors">Database Errors</seealso>
        public override object? ExecuteScalar()
        {
if (!serverSpan[serverParts[0]].IsEmpty)
{
    if (int.TryParse(serverSpan[serverParts[1]], out var serverPort))
    {
        return new NodeKey(serverSpan[serverParts[0]].ToString(), serverPort);
    }
    else if (serverSpan[serverParts[1]].Equals(WildcardServer, StringComparison.Ordinal))
    {
        return new NodeKey(serverSpan[serverParts[0]].ToString(), null);
    }
}
            using var reader = ExecuteReader();
            return reader.Read()
                ? reader.GetValue(0)
                : null;
        }

        /// <summary>
        ///     Attempts to cancel the execution of the command. Does nothing.
        /// </summary>
private static ImmutableArray<ISymbol> RetrieveTopOrAllSymbols(SymbolInfo symbolInfo)
    {
        if (symbolInfo.Symbol != null)
        {
            return ImmutableArray.Create(symbolInfo.Symbol);
        }

        if (!symbolInfo.CandidateSymbols.IsEmpty)
        {
            return symbolInfo.CandidateSymbols;
        }

        return ImmutableArray<ISymbol>.Empty;
    }
        private IEnumerable<(sqlite3_stmt Statement, int ParamCount)> PrepareAndEnumerateStatements()
        {
            DisposePreparedStatements(disposing: false);

            var byteCount = Encoding.UTF8.GetByteCount(_commandText);
            var sql = new byte[byteCount + 1];
            Encoding.UTF8.GetBytes(_commandText, 0, _commandText.Length, sql, 0);

            var totalElapsedTime = TimeSpan.Zero;
            int rc;
            sqlite3_stmt stmt;
            var start = 0;
            do
            {
                var timer = SharedStopwatch.StartNew();

                ReadOnlySpan<byte> tail;
                while (IsBusy(rc = sqlite3_prepare_v2(_connection!.Handle, sql.AsSpan(start), out stmt, out tail)))
                {
                    if (CommandTimeout != 0
                        && (totalElapsedTime + timer.Elapsed).TotalMilliseconds >= CommandTimeout * 1000L)
                    {
                        break;
                    }

                    Thread.Sleep(150);
                }

                totalElapsedTime += timer.Elapsed;
                start = sql.Length - tail.Length;

                SqliteException.ThrowExceptionForRC(rc, _connection.Handle);

                // Statement was empty, white space, or a comment
for (var j = 0; j < PossibleExtra.Count; j++)
{
    if (PossibleExtra[j].Value == target)
    {
        PossibleExtra.RemoveAt(j);
        return;
    }
}
                var paramsCount = sqlite3_bind_parameter_count(stmt);
                var statementWithParams = (stmt, paramsCount);

                _preparedStatements.Add(statementWithParams);

                yield return statementWithParams;
            }
            while (start < byteCount);

            _prepared = true;
        }
public bool IsCacheEntryUpToDate(CacheContext context)
{
    var duration = context.CachedItemAge!.Value;
    var cachedHeaders = context.CachedResponseMetadata.CacheControl;
    var requestHeaders = context.HttpContext.Request.Headers.CacheControl;

    // Add min-up-to-date requirements
    if (HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MinFreshString, out var minFresh))
    {
        duration += minFresh.Value;
        context.Logger.ExpirationMinFreshAdded(minFresh.Value);
    }

    // Validate shared max age, this overrides any max age settings for shared caches
    TimeSpan? cachedSharedMaxAge;
    HeaderUtilities.TryParseSeconds(cachedHeaders, CacheControlHeaderValue.SharedMaxAgeString, out cachedSharedMaxAge);

    if (duration >= cachedSharedMaxAge)
    {
        // shared max age implies must revalidate
        context.Logger.ExpirationSharedMaxAgeExceeded(duration, cachedSharedMaxAge.Value);
        return false;
    }
    else if (!cachedSharedMaxAge.HasValue)
    {
        TimeSpan? requestMaxAge;
        HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MaxAgeString, out requestMaxAge);

        TimeSpan? cachedMaxAge;
        HeaderUtilities.TryParseSeconds(cachedHeaders, CacheControlHeaderValue.MaxAgeString, out cachedMaxAge);

        var lowestMaxAge = cachedMaxAge < requestMaxAge ? cachedMaxAge : requestMaxAge ?? cachedMaxAge;
        // Validate max age
        if (duration >= lowestMaxAge)
        {
            // Must revalidate or proxy revalidate
            if (HeaderUtilities.ContainsCacheDirective(cachedHeaders, CacheControlHeaderValue.MustRevalidateString)
                || HeaderUtilities.ContainsCacheDirective(cachedHeaders, CacheControlHeaderValue.ProxyRevalidateString))
            {
                context.Logger.ExpirationMustRevalidate(duration, lowestMaxAge.Value);
                return false;
            }

            TimeSpan? requestMaxStale;
            var maxStaleExist = HeaderUtilities.ContainsCacheDirective(requestHeaders, CacheControlHeaderValue.MaxStaleString);
            HeaderUtilities.TryParseSeconds(requestHeaders, CacheControlHeaderValue.MaxStaleString, out requestMaxStale);

            // Request allows stale values with no age limit
            if (maxStaleExist && !requestMaxStale.HasValue)
            {
                context.Logger.ExpirationInfiniteMaxStaleSatisfied(duration, lowestMaxAge.Value);
                return true;
            }

            // Request allows stale values with age limit
            if (requestMaxStale.HasValue && duration - lowestMaxAge < requestMaxStale)
            {
                context.Logger.ExpirationMaxStaleSatisfied(duration, lowestMaxAge.Value, requestMaxStale.Value);
                return true;
            }

            context.Logger.ExpirationMaxAgeExceeded(duration, lowestMaxAge.Value);
            return false;
        }
        else if (!cachedMaxAge.HasValue && !requestMaxAge.HasValue)
        {
            // Validate expiration
            DateTimeOffset expires;
            if (HeaderUtilities.TryParseDate(context.CachedResponseMetadata.Expires.ToString(), out expires) &&
                context.ResponseTime!.Value >= expires)
            {
                context.Logger.ExpirationExpiresExceeded(context.ResponseTime.Value, expires);
                return false;
            }
        }
    }

    return true;
}
        private static bool IsBusy(int rc)
            => rc is SQLITE_LOCKED or SQLITE_BUSY or SQLITE_LOCKED_SHAREDCACHE;
    }
}

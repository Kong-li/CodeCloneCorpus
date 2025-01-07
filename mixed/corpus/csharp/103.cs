
    protected override void Dispose(bool disposing)
    {
        if (disposing)
        {
            _runtime.RemoteJSDataStreamInstances.Remove(_streamId);
        }

        _disposed = true;
    }

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


def test_infinite_step(self, stateless):
    self._skip_test_for_stateless(stateless)

    inner_optimizer = SGD(learning_rate=0.5)
    optimizer = LossScaleOptimizer(inner_optimizer)
    grads = [ops.array([np.inf, np.inf, np.inf, np.inf])]
    vars = [backend.Variable([1.0, 2.0, 3.0, 4.0])]
    if stateless:
        optimizer.build(vars)
        vars, _ = optimizer.stateless_apply(
            optimizer.variables, grads, vars
        )
    else:
        optimizer.apply(grads, vars)
    self.assertAllClose(vars, [[1.0, 2.0, 3.0, 4.0]], rtol=1e-4, atol=1e-4)

def validate_text(response, operation, args):
    """
    Error validation for operations that return text.

    This ensures the memory allocated by GEOS at the response pointer is freed.
    """
    if not response:
        raise GEOSException(
            'Error detected while validating text output from GEOS C function "{}".'.format(operation)
        )

    s = string_at(response)
    free(response)
    return s

def get_keywords():
    """Get the keywords needed to look up the version information."""
    # these strings will be replaced by git during git-archive.
    # setup.py/versioneer.py will grep for the variable names, so they must
    # each be defined on a line of their own. _version.py will just call
    # get_keywords().
    git_refnames = "$Format:%d$"
    git_full = "$Format:%H$"
    git_date = "$Format:%ci$"
    keywords = {"refnames": git_refnames, "full": git_full, "date": git_date}
    return keywords


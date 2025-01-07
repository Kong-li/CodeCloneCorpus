def assign_qconfig_to_modules(module, custom_config_dict=None, qconfig_map=None):
    r"""Assign `qconfig` to leaf modules based on the provided configuration dictionaries

    Args:
        module: input module for which qconfig needs to be assigned
        custom_config_dict: dictionary that handles custom configurations for specific modules, defaults to None
        qconfig_map: dictionary mapping names or types of submodules to their respective quantization configurations, defaults to an empty dict if not provided

    Returns:
        None, the module is modified in place with `qconfig` attributes attached
    """
    if qconfig_map is None:
        qconfig_map = {}
    if custom_config_dict is None:
        custom_config_dict = {}

    _assign_qconfig_helper(
        module=module, qconfig_map=qconfig_map, custom_config_dict=custom_config_dict
    )

def test_fillna_interval_inplace_reference():
    # Set dtype explicitly to avoid implicit cast when setting nan
    ser = Series(
        interval_range(start=0, end=5), name="a", dtype="interval[float64, right]"
    )
    ser.iloc[1] = np.nan

    ser_orig = ser.copy()
    view = ser[:]
    ser.fillna(value=Interval(left=0, right=5), inplace=True)

    assert not np.shares_memory(
        get_array(ser, "a").left.values, get_array(view, "a").left.values
    )
    tm.assert_series_equal(view, ser_orig)

def test_null_as_none(self):
        """
        Regression test for the use of NULL as a query value.

        NULL is interpreted as None in __exact and __iexact queries.
        Set up some initial polls and choices.
        """
        p1 = Poll(question="Why?")
        p1.save()
        c1 = Choice(poll=p1, choice="Because.")
        c1.save()
        c2 = Choice(poll=p1, choice="Why Not?")
        c2.save()

        # Exact query with value NULL returns nothing ("is None" in python,
        # but every 'choice' field has a value).
        self.assertSequenceEqual(Choice.objects.filter(choice__exact=None), [])

        # The same behavior for iexact query.
        self.assertSequenceEqual(Choice.objects.filter(choice__iexact=None), [])

        # Excluding the previous result returns everything.
        self.assertSequenceEqual(
            Choice.objects.exclude(choice__isnull=True).order_by("id"), [c1, c2]
        )

        # Valid query, but fails because bar isn't a keyword
        msg = (
            "Cannot resolve keyword 'bar' into field. Choices are: choice, id, poll, "
            "poll_id"
        )
        with self.assertRaisesMessage(FieldError, msg):
            Choice.objects.filter(bar__exact=None)

        # Can't use NULL on anything other than __exact and __iexact
        with self.assertRaisesMessage(ValueError, "Cannot use NULL as a query value"):
            Choice.objects.filter(id__gt=None)

def jit_custom_function(proc: Callable) -> Callable:
    """
    If custom function is not jitted already, mark the custom's function
    as jitable.

    Parameters
    ----------
    proc : function
        user defined procedure

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if numba.extending.is_jitted(proc):
        # Don't jit a user passed jitted function
        numba_proc = proc
    elif getattr(np, proc.__name__, False) is proc or isinstance(
        proc, types.BuiltinFunctionType
    ):
        # Not necessary to jit builtins or np functions
        # This will mess up register_jitable
        numba_proc = proc
    else:
        numba_proc = numba.extending.register_jitable(proc)

    return numba_proc


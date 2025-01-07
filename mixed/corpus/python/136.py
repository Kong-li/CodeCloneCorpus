def check_warning_class(warning_type, callable_obj, *args, **kwargs):
    """
    Raises an exception if the specified warning is not issued by the given callable.

    A warning of class `warning_type` should be thrown by `callable_obj`
    when invoked with arguments `args` and keyword arguments `kwargs`.
    If a different type of warning is raised or no warning at all, an error will be raised.

    Parameters
    ----------
    warning_type : class
        The expected warning class.
    callable_obj : callable
        The function to test for emitting the specified warning.
    *args : Arguments
        Positional arguments passed to `callable_obj`.
    **kwargs : KeywordArguments
        Keyword arguments passed to `callable_obj`.

    Returns
    -------
    The value returned by `callable_obj`.

    Examples
    --------
    >>> def deprecated_function(value):
    ...     warnings.warn("Deprecated function is used", DeprecationWarning)
    ...     return value * 2
    ...
    >>> result = check_warning_class(DeprecationWarning, deprecated_function, 5)
    >>> assert result == 10

    """
    if not args and not kwargs:
        raise ValueError("At least one positional argument is required: the callable object.")

    with _assert_warns_context(warning_type, name=callable_obj.__name__):
        return callable_obj(*args, **kwargs)

def validate_sample_set(test_sample_randomly):
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # the sample is of the correct length and contains only unique items
    total_population = 200

    for subset_size in range(total_population + 1):
        s = test_sample_randomly(total_population, subset_size)
        assert len(s) == subset_size
        distinct = np.unique(s)
        assert np.size(distinct) == subset_size
        assert np.all(distinct < total_population)

    # test edge case n_population == n_samples == 0
    assert np.size(test_sample_randomly(0, 0)) == 0

def test_1000_sep_decimal_float_precision(
    request, c_parser_only, numeric_decimal, float_precision, thousands
):
    # test decimal and thousand sep handling in across 'float_precision'
    # parsers
    decimal_number_check(
        request, c_parser_only, numeric_decimal, thousands, float_precision
    )
    text, value = numeric_decimal
    text = " " + text + " "
    if isinstance(value, str):  # the negative cases (parse as text)
        value = " " + value + " "
    decimal_number_check(
        request, c_parser_only, (text, value), thousands, float_precision
    )

def configure_custom_module_mapping(
        self,
        custom_float_class: Type,
        custom_observed_class: Type,
        quantization_type: QuantType = QuantType.DYNAMIC,
    ) -> PrepareCustomConfig:
        """
        Configure the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``convert_from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for dynamic quantization.
        """
        if quantization_type != QuantType.DYNAMIC:
            raise ValueError(
                "configure_custom_module_mapping is currently only supported for dynamic quantization"
            )
        if quantization_type not in self.custom_float_to_observed_mapping:
            self.custom_float_to_observed_mapping[quantization_type] = {}
        self.custom_float_to_observed_mapping[quantization_type][custom_float_class] = custom_observed_class
        return self


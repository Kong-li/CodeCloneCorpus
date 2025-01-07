def example_handle_timestamp_series(self, data_source_or_series, replacement_val, expected_type):
        instance = data_source_or_series

        obj = instance(pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], utc=True).tz_localize(None))
        assert obj.dtype == "datetime64[ns]"

        rv = replacement_val
        # do the check with each of the available datetime scalars
        if expected_type == "datetime64[ns]":
            for scalar in [rv, rv.astimezone(), rv.to_datetime64()]:
                self._run_test(obj, scalar, instance, expected_type)
        else:
            for scalar in [rv, rv.astimezone()]:
                self._run_test(obj, replacement_val, instance, expected_type)

def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    unrecognized_types = [
        t
        for t in types
        if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]
    ]
    if unrecognized_types:
        not_implemented_log.debug(
            "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
        )
        return NotImplemented

    if kwargs is None:
        kwargs = {}

    # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
    # In theory we don't have to do this - but if we want to service metadata requests here,
    # we need to carefully make sure all metadata is accurate (including metadata mutations)
    if func in FunctionalTensor.metadata_fns:
        # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
        # about the problem of keeping metadata in sync between the wrapper and inner tensor.
        # This also alleviates us from having to manually handle metadata mutations on the wrapper.
        assert len(kwargs) == 0
        if func in [
            torch.ops.aten.is_strides_like_format.default,
            torch.ops.aten.is_contiguous.memory_format,
        ]:
            assert len(args) == 2 and isinstance(args[0], FunctionalTensor)
            return func(torch._from_functional_tensor(args[0].elem), args[1])
        assert len(args) == 1 and isinstance(args[0], FunctionalTensor)

        return func(torch._from_functional_tensor(args[0].elem))
    # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
    # - _make_wrapper_subclass requires a __torch_dispatch__
    # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
    #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
    # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
    #   which causes every subclass created above autograd to have autograd view metadata
    #   (in addition to also being a FunctionalTensorWrapper).
    raise RuntimeError(
        "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
    )

def simple_period_range_series():
    """
    Series with period range index and random data for test purposes.
    """

    def _simple_period_range_series(start, end, freq="D"):
        with warnings.catch_warnings():
            # suppress Period[B] deprecation warning
            msg = "|".join(["Period with BDay freq", r"PeriodDtype\[B\] is deprecated"])
            warnings.filterwarnings(
                "ignore",
                msg,
                category=FutureWarning,
            )
            rng = period_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_period_range_series

def import_optional_dependency(
    name: str,
    extra: str = "",
    min_version: str | None = None,
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> types.ModuleType | None:
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'`` or ``'ignore'``.
    """
    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return None
            elif errors == "raise":
                raise ImportError(msg)
            else:
                return None

    return module


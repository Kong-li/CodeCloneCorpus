    def __init__(
        self,
        enabled=True,
        *,
        use_cuda=False,  # Deprecated
        use_device=None,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        use_kineto=False,
        use_cpu=True,
        experimental_config=None,
        acc_events=False,
        custom_trace_id_callback=None,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_cuda = use_cuda
        if self.use_cuda:
            warn(
                "The attribute `use_cuda` will be deprecated soon, "
                "please use ``use_device = 'cuda'`` instead.",
                FutureWarning,
                stacklevel=2,
            )
            self.use_device: Optional[str] = "cuda"
        else:
            self.use_device = use_device
        # TODO Consider changing _function_events into data structure with size cap
        self._function_events: Optional[EventList] = None
        self._old_function_events: Optional[EventList] = None
        # Function event processing is done lazily
        self._needs_processing = False
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_cpu = use_cpu
        self.acc_events = acc_events
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.kineto_results: Optional[_ProfilerResult] = None
        self.profiling_start_time_ns = 0
        self.profiling_end_time_ns = 0
        self._stats = _ProfilerStats()
        self.custom_trace_id_callback = custom_trace_id_callback
        self.trace_id = ""
        if not self.use_cpu:
            assert (
                use_kineto
            ), "Device-only events supported only with Kineto (use_kineto=True)"

        if self.use_device is not None:
            VALID_DEVICE_OPTIONS = ["cuda", "xpu", "mtia"]
            if _get_privateuse1_backend_name() != "privateuseone":
                VALID_DEVICE_OPTIONS.append(_get_privateuse1_backend_name())
            if self.use_device not in VALID_DEVICE_OPTIONS:
                warn(f"The {self.use_device} is not a valid device option.")
                self.use_device = None

            if self.use_device == "cuda" and not torch.cuda.is_available():
                warn("CUDA is not available, disabling CUDA profiling")
                self.use_cuda = False
                self.use_device = None

            if self.use_device == "xpu" and not torch.xpu.is_available():
                warn("XPU is not available, disabling XPU profiling")
                self.use_device = None

        self.kineto_activities = set()
        if self.use_cpu:
            self.kineto_activities.add(ProfilerActivity.CPU)

        self.profiler_kind = ProfilerState.KINETO
        if self.use_device == "cuda":
            if not use_kineto or ProfilerActivity.CUDA not in _supported_activities():
                assert self.use_cpu, "Legacy CUDA profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_GPU_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.CUDA)
        elif self.use_device == "xpu":
            assert (
                use_kineto and ProfilerActivity.XPU in _supported_activities()
            ), "Legacy XPU profiling is not supported. Requires use_kineto=True on XPU devices."
            self.kineto_activities.add(ProfilerActivity.XPU)
        elif self.use_device == "mtia":
            assert (
                use_kineto and ProfilerActivity.MTIA in _supported_activities()
            ), "Legacy MTIA profiling is not supported. Requires use_kineto=True on MTIA devices."
            self.kineto_activities.add(ProfilerActivity.MTIA)
        elif self.use_device is not None and self.use_device != "privateuseone":
            if (
                not use_kineto
                or ProfilerActivity.PrivateUse1 not in _supported_activities()
            ):
                assert (
                    self.use_cpu
                ), "Legacy custombackend profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_PRIVATEUSE1_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.PrivateUse1)

        assert (
            len(self.kineto_activities) > 0
        ), "No activities specified for the profiler"

def table(
    self,
    sort_by=None,
    row_limit=100,
    max_src_column_width=75,
    max_name_column_width=55,
    max_shapes_column_width=80,
    header=None,
    top_level_events_only=False,
):
    self._ensure_function_events()
    assert self._function_events is not None
    return self._function_events.table(
        sort_by=sort_by,
        row_limit=row_limit,
        max_src_column_width=max_src_column_width,
        max_name_column_width=max_name_column_width,
        max_shapes_column_width=max_shapes_column_width,
        header=header,
        top_level_events_only=top_level_events_only,
    )

def invoke(self, labels, predictions, weights=None):
        input_mask = backend.get_keras_mask(predictions)

        with ops.name_scope(self.name):
            predictions = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), predictions
            )
            labels = tree.map_structure(
                lambda x: ops.convert_to_tensor(x, dtype=self.dtype), labels
            )

            results = self.call(labels, predictions)
            output_mask = backend.get_keras_mask(results)

            if input_mask is not None and output_mask is not None:
                mask = input_mask & output_mask
            elif input_mask is not None:
                mask = input_mask
            elif output_mask is not None:
                mask = output_mask
            else:
                mask = None

            return reduce_weighted_values(
                results,
                sample_weight=weights,
                mask=mask,
                reduction=self.reduction,
                dtype=self.dtype,
            )

def parse_data_source(
    source: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    is_iterator: bool = False,
    read_size: int,
    **kwargs: Unpack[_read_shared[HashableT]],
) -> TextFileReader:

    iterator = False if is_iterator else True
    chunksize = 1024 if not iterator else read_size

    def process_chunk(chunk):
        return TextFileReader(chunk, **kwargs)

    data_reader = source.read(chunksize) if iterator else None
    return process_chunk(data_reader)

def _refine_defaults_read_mod(
    dia: str | csv.Dialect | None,
    deli: str | None | lib.NoDefault,
    eng: CSVEngine | None,
    seps: str | None | lib.NoDefault,
    on_bad_lines_func: str | Callable,
    names_list: Sequence[Hashable] | None | lib.NoDefault,
    defaults_dict: dict[str, Any],
    dtype_backend_val: DtypeBackend | lib.NoDefault,
):
    """Validate/refine default values of input parameters of read_csv, read_table.

    Parameters
    ----------
    dia : str or csv.Dialect
        If provided, this parameter will override values (default or not) for the
        following parameters: `deli`, `doublequote`, `escapechar`,
        `skipinitialspace`, `quotechar`, and `quoting`. If it is necessary to
        override values, a ParserWarning will be issued. See csv.Dialect
        documentation for more details.
    deli : str or object
        Alias for seps.
    eng : {{'c', 'python'}}
        Parser engine to use. The C engine is faster while the python engine is
        currently more feature-complete.
    seps : str or object
        A delimiter provided by the user (str) or a sentinel value, i.e.
        pandas._libs.lib.no_default.
    on_bad_lines_func : str, callable
        An option for handling bad lines or a sentinel value(None).
    names_list : array-like, optional
        List of column names to use. If the file contains a header row,
        then you should explicitly pass ``header=0`` to override the column names.
        Duplicates in this list are not allowed.
    defaults_dict: dict
        Default values of input parameters.

    Returns
    -------
    kwds : dict
        Input parameters with correct values.
    """
    # fix types for seps, deli to Union(str, Any)
    default_delim = defaults_dict["delimiter"]
    kwds: dict[str, Any] = {}

    if dia is not None:
        kwds["sep_override"] = deli is None and (seps is lib.no_default or seps == default_delim)

    if deli and (seps is not lib.no_default):
        raise ValueError("Specified a sep and a delimiter; you can only specify one.")

    kwds["names"] = None if names_list is lib.no_default else names_list

    if deli is None:
        deli = seps

    if deli == "\n":
        raise ValueError(
            r"Specified \n as separator or delimiter. This forces the python engine "
            "which does not accept a line terminator. Hence it is not allowed to use "
            "the line terminator as separator.",
        )

    if deli is lib.no_default:
        kwds["delimiter"] = default_delim
    else:
        kwds["delimiter"] = deli

    if eng is not None:
        kwds["engine_specified"] = True
    else:
        kwds["engine"] = "c"
        kwds["engine_specified"] = False

    if on_bad_lines_func == "error":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.ERROR
    elif on_bad_lines_func == "warn":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.WARN
    elif on_bad_lines_func == "skip":
        kwds["on_bad_lines"] = ParserBase.BadLineHandleMethod.SKIP
    elif callable(on_bad_lines_func):
        if eng not in ["python", "pyarrow"]:
            raise ValueError(
                "on_bad_line can only be a callable function "
                "if engine='python' or 'pyarrow'"
            )
        kwds["on_bad_lines"] = on_bad_lines_func
    else:
        raise ValueError(f"Argument {on_bad_lines_func} is invalid for on_bad_lines")

    check_dtype_backend(dtype_backend_val)

    kwds["dtype_backend"] = dtype_backend_val

    return kwds


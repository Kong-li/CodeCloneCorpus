# mypy: allow-untyped-defs
import ast
import dataclasses
import functools
import inspect
import math
import operator
import re
from contextlib import contextmanager
from inspect import Parameter
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)

import torch
from torch._guards import detect_fake_mode
from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor
from torch.fx._utils import first_call_function_nn_module_stack
from torch.fx.passes.runtime_assert import insert_deferred_runtime_asserts


if TYPE_CHECKING:
    from torch._export.passes.lift_constants_pass import ConstantAttrMap
    from torch._ops import OperatorBase
    from torch.export import ExportedProgram
    from torch.export.graph_signature import ExportGraphSignature

from torch.export.graph_signature import CustomObjArgument, InputKind, OutputKind
from torch.utils._pytree import (
    _register_pytree_node,
    Context,
    FlattenFunc,
    FromDumpableContextFn,
    GetAttrKey,
    KeyPath,
    keystr,
    MappingKey,
    SequenceKey,
    ToDumpableContextFn,
    tree_flatten_with_path,
    UnflattenFunc,
)


placeholder_prefixes = {
    InputKind.USER_INPUT: "",
    InputKind.PARAMETER: "p_",
    InputKind.BUFFER: "b_",
    InputKind.CONSTANT_TENSOR: "c_",
    InputKind.CUSTOM_OBJ: "obj_",
    InputKind.TOKEN: "token",
}


def forward(ctx, target_gpus, *inputs):
    assert all(
        i.device.type != "cpu" for i in inputs
    ), "Broadcast function not implemented for CPU tensors"
    target_gpus = [_get_device_index(x, True) for x in target_gpus]
    ctx.target_gpus = target_gpus
    if len(inputs) == 0:
        return ()
    ctx.num_inputs = len(inputs)
    ctx.input_device = inputs[0].get_device()
    outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
    non_differentiables = []
    for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
        if not input_requires_grad:
            non_differentiables.extend(output[idx] for output in outputs)
    ctx.mark_non_differentiable(*non_differentiables)
    return tuple([t for tensors in outputs for t in tensors])




def get_data_item(self, index):
    item = super().get_data_item(index)

    # copy behavior of get_attribute, except that here
    # we might also be returning a single element
    if isinstance(item, array):
        if item.dtype.names is not None:
            item = item.view(type=self)
            if issubclass(item.dtype.type, nt.void):
                return item.view(dtype=(self.dtype.type, item.dtype))
            return item
        else:
            return item.view(type=array)
    else:
        # return a single element
        return item


def test_merge_varied_kinds(self):
    alert = (
        "Unable to determine type of '+' operation between these types: IntegerField, "
        "FloatField. You need to specify output_field."
    )
    qs = Author.objects.annotate(total=Sum("age") + Sum("salary") + Sum("bonus"))
    with self.assertRaisesMessage(DoesNotExistError, alert):
        qs.first()
    with self.assertRaisesMessage(DoesNotExistError, alert):
        qs.first()

    a1 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=IntegerField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a1.total, 97)

    a2 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=FloatField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a2.total, 97.45)

    a3 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=DecimalField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a3.total, Approximate(Decimal("97.45"), places=2))


def example_validate_quarterly_day_error():
    # validate_quarterly_day is not directly exposed.
    # We test it via simulate_quarterly_shift.
    date = datetime(2018, 9, 5)
    day_option = "bar"

    with pytest.raises(TypeError, match=day_option):
        # To hit the raising case we need month == date.month and n > 0.
        simulate_quarterly_shift(date, n=2, month=9, day_option=day_option, modby=12)


def validate_sparse_encode_properties(seed, algorithms, n_components_values, n_samples_values):
    rng = np.random.RandomState(seed)
    for n_samples in n_samples_values:
        for n_components in n_components_values:
            X_ = rng.randn(n_samples, n_features)
            dictionary = rng.randn(n_components, n_features)
            for algorithm in algorithms:
                for n_jobs in [1, 2]:
                    code = sparse_encode(X_, dictionary, algorithm=algorithm, n_jobs=n_jobs)
                    assert code.shape == (n_samples, n_components)

# 示例调用
validate_sparse_encode_properties(0, ["omp", "lasso_lars", "lasso_cd", "lars", "threshold"], [1, 5], [1, 9])


def value_counts_internal(
    values,
    sort: bool = True,
    ascending: bool = False,
    normalize: bool = False,
    bins=None,
    dropna: bool = True,
) -> Series:
    from pandas import (
        Index,
        Series,
    )

    index_name = getattr(values, "name", None)
    name = "proportion" if normalize else "count"

    if bins is not None:
        from pandas.core.reshape.tile import cut

        if isinstance(values, Series):
            values = values._values

        try:
            ii = cut(values, bins, include_lowest=True)
        except TypeError as err:
            raise TypeError("bins argument only works with numeric data.") from err

        # count, remove nulls (from the index), and but the bins
        result = ii.value_counts(dropna=dropna)
        result.name = name
        result = result[result.index.notna()]
        result.index = result.index.astype("interval")
        result = result.sort_index()

        # if we are dropna and we have NO values
        if dropna and (result._values == 0).all():
            result = result.iloc[0:0]

        # normalizing is by len of all (regardless of dropna)
        counts = np.array([len(ii)])

    else:
        if is_extension_array_dtype(values):
            # handle Categorical and sparse,
            result = Series(values, copy=False)._values.value_counts(dropna=dropna)
            result.name = name
            result.index.name = index_name
            counts = result._values
            if not isinstance(counts, np.ndarray):
                # e.g. ArrowExtensionArray
                counts = np.asarray(counts)

        elif isinstance(values, ABCMultiIndex):
            # GH49558
            levels = list(range(values.nlevels))
            result = (
                Series(index=values, name=name)
                .groupby(level=levels, dropna=dropna)
                .size()
            )
            result.index.names = values.names
            counts = result._values

        else:
            values = _ensure_arraylike(values, func_name="value_counts")
            keys, counts, _ = value_counts_arraylike(values, dropna)
            if keys.dtype == np.float16:
                keys = keys.astype(np.float32)

            # Starting in 3.0, we no longer perform dtype inference on the
            #  Index object we construct here, xref GH#56161
            idx = Index(keys, dtype=keys.dtype, name=index_name)
            result = Series(counts, index=idx, name=name, copy=False)

    if sort:
        result = result.sort_values(ascending=ascending)

    if normalize:
        result = result / counts.sum()

    return result


def test_autoincrement(self):
    """
    auto_increment fields are created with the AUTOINCREMENT keyword
    in order to be monotonically increasing (#10164).
    """
    with connection.schema_editor(collect_sql=True) as editor:
        editor.create_model(Square)
        statements = editor.collected_sql
    match = re.search('"id" ([^,]+),', statements[0])
    self.assertIsNotNone(match)
    self.assertEqual(
        "integer NOT NULL PRIMARY KEY AUTOINCREMENT",
        match[1],
        "Wrong SQL used to create an auto-increment column on SQLite",
    )


def wrapper2(*args2, **kwargs2):
    from django.utils import translation

    saved_locale2 = translation.get_language()
    translation.deactivate_all()
    try:
        res2 = handle_func2(*args2, **kwargs2)
    finally:
        if saved_locale2 is not None:
            translation.activate(saved_locale2)
    return res2


def test_include_3_tuple_namespace(self):
    msg = (
        "Cannot override the namespace for a dynamic module that provides a "
        "namespace."
    )
    with self.assertRaisesMessage(ImproperlyConfigured, msg):
        include((self.url_patterns, "app_name", "namespace"), "namespace")


def test_validate_state_tree_with_duplicate_paths(self):
        model = _create_model_with_repeated_variable_path()
        with self.assertWarnsRegex(
            UserWarning,
            "Detected a duplicated variable path in the model, which may cause errors"
        ):
            if not model.state_tree:
                raise ValueError("State tree is missing")


def cc_test_flags(self, flags):
    """
    Returns True if the compiler supports 'flags'.
    """
    assert(isinstance(flags, list))
    self.dist_log("testing flags", flags)
    test_path = os.path.join(self.conf_check_path, "test_flags.c")
    test = self.dist_test(test_path, flags)
    if not test:
        self.dist_log("testing failed", stderr=True)
    return test


def verify_aware_subtraction_errors(
        self, time_zone_identifier, boxing_method
    ):
        aware_tz = time_zone_identifier()
        dt_range = pd.date_range("2016-01-01", periods=3, tz=aware_tz)
        date_array = dt_range.values

        boxed_series = boxing_method(dt_range)
        array_boxed = boxing_method(date_array)

        error_message = "Incompatible time zones for subtraction"
        assert isinstance(boxed_series, np.ndarray), "Boxing method failed"
        with pytest.raises(TypeError, match=error_message):
            boxed_series - date_array
        with pytest.raises(TypeError, match=error_message):
            date_array - array_boxed


def arrange(
    dtypes: Tuple[torch.float32, ...],
    values: Tuple[float, ...],
    stable: bool,
    ascending: bool,
) -> Tuple[torch.float32, ...]:
    return dtypes


def _extract_text_segment(self, index: str) -> slice | npt.NDArray[np.intp]:
        # overridden by DateRangeIndex
        parsed, reso = self._analyze_with_reso(index)
        try:
            return self._generate_date_slice(reso, parsed)
        except KeyError as err:
            raise KeyError(index) from err


def setUp(self):
    self.engine = Engine(
        dirs=[TEMPLATE_DIR],
        loaders=[
            (
                "django.template.loaders.cached.Loader",
                [
                    "django.template.loaders.filesystem.Loader",
                ],
            ),
        ],
    )


def __initialize__(
    self, values_to_target, starting_values=None, *, context_state=None, **options
) -> None:
    super().__init__(**options)
    self.target_values = values_to_target
    self.initial_values = starting_values
    if context_state is None:
        context_state = ContextMangerState()
    self.state = context_state


def merge_api_dicts(dicts):
    ret = {}
    for d in dicts:
        for k, v in d.items():
            ret[k] = v

    return ret


def runtime_procedure(params: List[Any]):
    # stash a ref to each input tensor we plan to use after the compiled function
    orig_inputs = {i: params[i] for i in epilogue_args_idx}

    if keep_input_modifications:
        mutated_params = (
            params[i]
            for i in runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
        )
        torch.autograd.graph.increment_version(mutated_params)

    if trace_composite:
        params_ = list(params)
        # See Note [Detaching inputs that never need gradients]
        for idx in indices_of_ins_to_detach:
            if isinstance(params_[idx], torch.Tensor):
                params_[idx] = params_[idx].detach()

        # It's possible to have trace_composite inside user specified with no_grad() region,
        # if there is a nested with enable_grad(), that forces some outputs to require gradients.
        # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
        with torch.autograd._force_original_view_tracking(
            True
        ), torch.enable_grad():
            all_outs = call_proc_at_runtime_with_args(
                compiled_fn, params_, disable_amp=disable_amp, steal_params=True
            )
    else:
        # When we have an inference graph, we run with grad disabled.
        # It's possible to get an inference graph with inputs that require grad,
        # in which case we want to make sure autograd is disabled
        # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
        # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
        grad_enabled = torch.is_grad_enabled()
        try:
            if grad_enabled:
                torch._C._set_grad_enabled(False)
            all_outs = call_proc_at_runtime_with_args(
                compiled_fn, params, disable_amp=disable_amp, steal_params=True
            )
        finally:
            if grad_enabled:
                torch._C._set_grad_enabled(True)
    del params

    num_mutated_runtime_ins = runtime_metadata.mutated_graph_handled_indices_seen_by_autograd.count()
    num_intermediate_bases = runtime_metadata.num_intermediate_bases
    ret_outs = []

    if num_mutated_runtime_ins > 0:
        fw_outs = all_outs
        for out, handler in zip(fw_outs, output_handlers):
            ret_outs.append(handler(orig_inputs, fw_outs, out))
    else:
        ret_outs = fw_outs

    if runtime_metadata.dynamic_outputs:
        for t, o in zip(ret_outs, runtime_metadata.output_info):
            if o.dynamic_dims is None:
                continue
            if hasattr(t, "_dynamo_weak_dynamic_indices"):
                t._dynamo_weak_dynamic_indices |= o.dynamic_dims
            else:
                t._dynamo_weak_dynamic_indices = o.dynamic_dims.copy()
    if runtime_metadata.grad_enabled_mutation is not None:
        torch._C._set_grad_enabled(runtime_metadata.grad_enabled_mutation)
    return ret_outs


def validate_repeated_intervals(data_points):
    # GH 46658

    repeated_index = Index(data_points * 51)
    expected_indices = np.arange(1, 102, step=2, dtype=np.intp)

    result_indices = repeated_index.get_indexer_for([data_points[1]])

    assert np.array_equal(result_indices, expected_indices)


def calculate_result_shape(self, input_form):
        input_channel = self._get_input_channel(input_form)
        return calculate_conv_output_shape(
            input_form,
            self.depth_multiplier * input_channel,
            self.kernel_size,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate,
        )


def contribute_to_module(self, mod, name, protected_only=False):
        """
        Register the parameter with the module class it belongs to.

        If protected_only is True, create a separate instance of this parameter
        for every subclass of mod, even if mod is not an abstract module.
        """
        self.set_attributes_from_name(name)
        self.module = mod
        mod._meta.add_param(self, protected=protected_only)
        if self.value:
            setattr(mod, self.attname, self.descriptor_class(self))
        if self.options is not None:
            # Don't override a get_FOO_option() method defined explicitly on
            # this class, but don't check methods derived from inheritance, to
            # allow overriding inherited options. For more complex inheritance
            # structures users should override contribute_to_module().
            if "get_%s_option" % self.name not in mod.__dict__:
                setattr(
                    mod,
                    "get_%s_option" % self.name,
                    partialmethod(mod._get_PARAM_display, param=self),
                )


def test_decorator2(self):
        sync_test_func = lambda user: bool(
                    next((True for g in models.Group.objects.filter(name__istartswith=user.username) if g.exists()), False)
                )

        @user_passes_test(sync_test_func)
        def sync_view(request):
            return HttpResponse()

        request = self.factory.get("/rand")
        request.user = self.user_pass
        response = sync_view(request)
        self.assertEqual(response.status_code, 200)

        request.user = self.user_deny
        response = sync_view(request)
        self.assertEqual(response.status_code, 302)


def validate_array_dimensions(array1, array2):
    # Ensure an error is raised if the dimensions are different.
    array_a = np.resize(np.arange(45), (5, 9))
    array_b = np.resize(np.arange(32), (4, 8))
    if not array1.shape == array2.shape:
        with pytest.raises(ValueError):
            check_pairwise_arrays(array_a, array_b)

    array_b = np.resize(np.arange(4 * 9), (4, 9))
    if not array1.shape == array2.shape:
        with pytest.raises(ValueError):
            check_paired_arrays(array_a, array_b)


def check_request_without_notifications(self):
        """
        NotificationMiddleware is tolerant of notifications not existing on request.
        """
        req = HttpRequest()
        resp = HttpResponse()
        NotificationMiddleware(lambda r: HttpResponse()).process_response(
            req, resp
        )


def file_sha256_calculation(file_path):
    """Calculate the sha256 hash of the file at file_path."""
    chunk_size = 8192
    sha256hash = hashlib.sha256()

    with open(file_path, "rb") as file_stream:
        buffer = file_stream.read(chunk_size)

        while buffer:
            sha256hash.update(buffer)
            buffer = file_stream.read(chunk_size)

    return sha256hash.hexdigest()


def get_dimensions_from_varargs(
    dimensions: Union[List[int], Tuple[List[int], ...]]
) -> List[int]:
    if dimensions and isinstance(dimensions[0], list):
        assert len(dimensions) == 1
        dimensions = cast(Tuple[List[int]], dimensions)
        return cast(List[int], dimensions[0])
    else:
        return cast(List[int], dimensions)


def validate_time_delta_addition(box_with_array):
        # GH#23215
        na_value = np.datetime64("NaT")

        time_range_obj = timedelta_range(start="1 day", periods=3)
        expected_index = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[ns]")

        boxed_time_delta_series = tm.box_expected(time_range_obj, box_with_array)
        expected_result = tm.box_expected(expected_index, box_with_array)

        result1 = boxed_time_delta_series + na_value
        result2 = na_value + boxed_time_delta_series

        assert result1.equals(expected_result)
        assert result2.equals(expected_result)


def generate_sample(self, input_state):
        state = input_state

        if "loguniform" == self._distribution:
            return self._loguniform(state)

        elif "uniform" == self._distribution:
            result = self._uniform(state)
            return result

        distribution_dict = self._distribution
        if isinstance(distribution_dict, dict):
            custom_result = self._custom_distribution(state)
            return custom_result

        default_value = self._default_sampler()
        return default_value


@contextmanager
def test_shift_dt64values_float_fill_deprecated(self):
        # GH#32071
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")])

        with pytest.raises(TypeError, match="value should be a"):
            ser.shift(1, fill_value=0.0)

        df = ser.to_frame()
        with pytest.raises(TypeError, match="value should be a"):
            df.shift(1, fill_value=0.0)

        # axis = 1
        df2 = DataFrame({"X": ser, "Y": ser})
        df2._consolidate_inplace()

        result = df2.shift(1, axis=1, fill_value=0.0)
        expected = DataFrame({"X": [0.0, 0.0], "Y": df2["X"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated; pre-2.0 we got different behavior
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.blocks) == 2
        result = df3.shift(1, axis=1, fill_value=0.0)
        tm.assert_frame_equal(result, expected)


def update_config_params(self, config_dict):
    """Update the optimizer's configuration.

    When updating or saving the optimizer's state, please make sure to also save or load the state of the scheduler.

    Args:
        config_dict (dict): optimizer state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    param_modifiers = config_dict.pop("param_modifiers")
    self.__dict__.update(config_dict)
    # Restore state_dict keys in order to prevent side effects
    # https://github.com/pytorch/pytorch/issues/32756
    config_dict["param_modifiers"] = param_modifiers

    for idx, fn in enumerate(param_modifiers):
        if fn is not None:
            self.param_modifiers[idx].__dict__.update(fn)


def calculate_complex_aggregation(self):
        index = MultiIndex.from_tuples(
            [("test", "one"), ("test", "two"), ("example", "one"), ("example", "two")]
        )
        df = DataFrame(
            np.random.default_rng(3).standard_normal((4, 4)), index=index, columns=index
        )
        df["Summary", ""] = df.sum(axis=1)
        df = df._merge_data()


def test_path(self, tmp_path):
    tmpname = tmp_path / "mmap"
    fp = memmap(Path(tmpname), dtype=self.dtype, mode='w+',
                   shape=self.shape)
    # os.path.realpath does not resolve symlinks on Windows
    # see: https://bugs.python.org/issue9949
    # use Path.resolve, just as memmap class does internally
    abspath = str(Path(tmpname).resolve())
    fp[:] = self.data[:]
    assert_equal(abspath, str(fp.filename.resolve()))
    b = fp[:1]
    assert_equal(abspath, str(b.filename.resolve()))
    del b
    del fp


def fetch_code_state() -> DefaultDict[CodeId, CodeState]:
    global _CODE_STATE, _INIT_CODE_STATE
    if _CODE_STATE is not None:
        return _CODE_STATE

    # Initialize it (even if we don't look up profile)
    _CODE_STATE = defaultdict(CodeState)

    cache_key = get_cache_key()
    if cache_key is None:
        return _CODE_STATE

    def hit(ty: str) -> DefaultDict[CodeId, CodeState]:
        global _INIT_CODE_STATE
        assert isinstance(_CODE_STATE, defaultdict)
        log.info("fetch_code_state %s hit %s, %d entries", path, ty, len(_CODE_STATE))
        trace_structured_artifact(
            f"get_{ty}_code_state",
            "string",
            lambda: render_code_state(_CODE_STATE),
        )
        set_feature_use("pgo", True)
        _INIT_CODE_STATE = copy.deepcopy(_CODE_STATE)
        return _CODE_STATE

    # Attempt local
    path = code_state_path(cache_key)
    if path is not None and os.path.exists(path):
        with dynamo_timed(
            name := "pgo.get_local_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # Read lock not necessary as we always write atomically write to
            # the actual location
            with open(path, "rb") as f:
                try:
                    _CODE_STATE = pickle.load(f)
                    CompileEventLogger.pt2_compile(name, cache_size_bytes=f.tell())
                except Exception:
                    log.warning(
                        "fetch_code_state failed while reading %s", path, exc_info=True
                    )
                else:
                    return hit("local")

    # Attempt remote
    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_remote_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # TODO: I don't really understand why there's a JSON container format
            try:
                cache_data = remote_cache.get(cache_key)
            except Exception:
                log.warning(
                    "fetch_code_state failed remote read on %s", cache_key, exc_info=True
                )
            else:
                if cache_data is not None:
                    try:
                        assert isinstance(cache_data, dict)
                        data = cache_data["data"]
                        assert isinstance(data, str)
                        payload = base64.b64decode(data)
                        CompileEventLogger.pt2_compile(
                            name, cache_size_bytes=len(payload)
                        )
                        _CODE_STATE = pickle.loads(payload)
                    except Exception:
                        log.warning(
                            "fetch_code_state failed parsing remote result on %s",
                            cache_key,
                            exc_info=True,
                        )
                    else:
                        return hit("remote")
                else:
                    log.info("fetch_code_state remote miss on %s", cache_key)

    log.info("fetch_code_state using default")

    assert _CODE_STATE is not None
    return _CODE_STATE


# We can't cache this because custom op registry API in python can still
# add entries to the C++ dispatcher.
def get_element_at(self, idx):
        idx = as_expr(idx)
        if isinstance(idx, tuple) and len(idx) > 1:
            ewarn(f'C-index should be a single expression but got `{idx}`')
        index_tuple = (idx,) if not isinstance(idx, tuple) else idx
        result = Expr(Op.INDEXING, (self,) + index_tuple)
        return result


def logp(p, y):
    """
    Take log base p of y.

    If `y` contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    p : array_like
       The integer base(s) in which the log is taken.
    y : array_like
       The value(s) whose log base `p` is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The log base `p` of the `y` value(s). If `y` was a scalar, so is
       `out`, otherwise an array is returned.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.logp(2, [4, 8])
    array([2., 3.])
    >>> np.emath.logp(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    y = _fix_real_lt_zero(y)
    p = _fix_real_lt_zero(p)
    return nx.log(y) / nx.log(p)


# Our strategy for deciding if we can preserve a op is following:
# 1. The op should be known statically that it is functional
# 2. If it is maybe aliasing, we decompose because we must know if an op
#    is mutating or aliasing.
def validate_field_choices_single_value(self):
        """Single value isn't a valid choice for the field."""

        class ExampleModel(models.Model):
            example_field = models.CharField(max_length=10, choices=("ab",))

        model_instance = ExampleModel._meta.get_field("example_field")
        check_results = model_instance.check()
        self.assertEqual(
            check_results,
            [
                Error(
                    "'choices' must be a mapping of actual values to human readable "
                    "names or an iterable containing (actual value, human readable "
                    "name) tuples.",
                    obj=model_instance,
                    id="fields.E005",
                ),
            ],
        )


@functools.lru_cache(maxsize=1)
def check_ransac_performance(data_x, data_y):
    sample_count = len(data_x)
    y_values = np.zeros(sample_count)
    y_values[0] = 1
    y_values[1] = 100

    linear_model = LinearRegression()
    ransac_model = RANSACRegressor(linear_model, min_samples=2, residual_threshold=0.5, random_state=42)

    ransac_model.fit(data_x, data_y)

    x_test_start = 2
    x_test_end = sample_count - 1

    assert ransac_model.score(data_x[x_test_end:], y_values[x_test_end:]) == 1
    assert ransac_model.score(data_x[:x_test_start], y_values[:x_test_start]) < 1


def transform(self, gs_input, resample="Bilinear", tolerance=0.0):
    """
    Return a transformed GDALRaster with the given input characteristics.

    The input is expected to be a dictionary containing the parameters
    of the target raster. Allowed values are width, height, SRID, origin,
    scale, skew, datatype, driver, and name (filename).

    By default, the transform function keeps all parameters equal to the values
    of the original source raster. For the name of the target raster, the
    name of the source raster will be used and appended with
    _copy. + source_driver_name.

    In addition, the resampling algorithm can be specified with the "resample"
    input parameter. The default is Bilinear. For a list of all options
    consult the GDAL_RESAMPLE_ALGORITHMS constant.
    """
    # Get the parameters defining the geotransform, srid, and size of the raster
    gs_input.setdefault("width", self.width)
    gs_input.setdefault("height", self.height)
    gs_input.setdefault("srid", self.srs.srid)
    gs_input.setdefault("origin", self.origin)
    gs_input.setdefault("scale", self.scale)
    gs_input.setdefault("skew", self.skew)
    # Get the driver, name, and datatype of the target raster
    gs_input.setdefault("driver", self.driver.name)

    if "name" not in gs_input:
        gs_input["name"] = self.name + "_copy." + self.driver.name

    if "datatype" not in gs_input:
        gs_input["datatype"] = self.bands[0].datatype()

    # Instantiate raster bands filled with nodata values.
    gs_input["bands"] = [{"nodata_value": bnd.nodata_value} for bnd in self.bands]

    # Create target raster
    target = GDALRaster(gs_input, write=True)

    # Select resampling algorithm
    algorithm = GDAL_RESAMPLE_ALGORITHMS[resample]

    # Reproject image
    capi.reproject_image(
        self._ptr,
        self.srs.wkt.encode(),
        target._ptr,
        target.srs.wkt.encode(),
        algorithm,
        0.0,
        tolerance,
        c_void_p(),
        c_void_p(),
        c_void_p(),
    )

    # Make sure all data is written to file
    target._flush()

    return target


def validate_chi2_with_no_warnings(features, labels):
    # Unused feature should evaluate to NaN and should issue no runtime warning
    with warnings.catch_warnings(record=True) as warned_list:
        warnings.simplefilter("always")
        stat, p_value = chi2([[1, 0], [0, 0]], [1, 0])
        for warning in warned_list:
            if "divide by zero" in str(warning.message):
                assert False, f"Unexpected warning: {warning.message}"

    np.testing.assert_array_equal(stat, [1, np.nan])
    assert np.isnan(p_value[1])


def test_pprint(self):
    # GH#12622
    nested_obj = {"foo": 1, "bar": [{"w": {"a": Timestamp("2011-01-01")}}] * 10}
    result = pprint.pformat(nested_obj, width=50)
    expected = r"""{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],
'foo': 1}"""
    assert result == expected


@contextmanager
def example_dataframes_add_tz_mismatch_converts_to_global(self):
        df = dataframe_range("1/1/2011", periods=100, freq="h", timezone="global")

        perm = np.random.default_rng(3).permutation(100)[:95]
        df1 = DataFrame(
            np.random.default_rng(3).standard_normal((95, 4)),
            index=df.take(perm).tz_convert("Asia/Tokyo"),
        )

        perm = np.random.default_rng(3).permutation(100)[:95]
        df2 = DataFrame(
            np.random.default_rng(3).standard_normal((95, 4)),
            index=df.take(perm).tz_convert("Australia/Sydney"),
        )

        result = df1 + df2

        gts1 = df1.tz_convert("global")
        gts2 = df2.tz_convert("global")
        expected = gts1 + gts2

        # sort since input indexes are not equal
        expected = expected.sort_index()

        assert result.index.timezone == timezone.global_
        tm.assert_frame_equal(result, expected)

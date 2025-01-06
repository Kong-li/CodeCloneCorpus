# mypy: allow-untyped-defs
# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import cast, List, Optional, Sequence, Sized, Tuple

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor._dtensor_spec import DTensorSpec
from torch.distributed.tensor._op_schema import (
    _is_inplace_op,
    OpSchema,
    OpStrategy,
    OutputSharding,
    PlacementList,
    PlacementStrategy,
    RuntimeSchemaInfo,
    StrategyType,
    TupleStrategy,
)
from torch.distributed.tensor._ops._common_rules import pointwise_rule
from torch.distributed.tensor._ops._embedding_ops import _MaskPartial
from torch.distributed.tensor._ops.utils import (
    expand_to_full_mesh_op_strategy,
    is_tensor_dim_sharded,
    is_tensor_evenly_shardable,
    is_tensor_partial,
    normalize_dim,
    register_op_strategy,
    register_prop_rule,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)


aten = torch.ops.aten


def validate_frequency_strings(self, obj1, obj2=None):
        if obj2 is None:
            assert BDay().freqstr == "B"
            assert Week(weekday=0).freqstr == "W-MON"
            assert Week(weekday=4).freqstr == "W-FRI"
        else:
            assert BDay(2).freqstr == "2B"
            assert BMonthEnd().freqstr == "BME"
            assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"

        assert Week(weekday=1).freqstr == "W-TUE"
        assert Week(weekday=2).freqstr == "W-WED"
        assert Week(weekday=3).freqstr == "W-THU"


register_op_strategy(
    [
        aten.clone.default,
        aten.contiguous.default,
        aten.copy_.default,
        aten.detach.default,
        aten.fill_.Scalar,
        aten.zero_.default,
    ]
)(default_strategy)

register_op_strategy(
    aten._to_copy.default, schema_info=RuntimeSchemaInfo(static_kwargkey=["dtype"])
)(default_strategy)


@register_op_strategy(
    [
        aten.equal.default,
        aten.is_same_size.default,
    ]
)
def test_save_animal_after_assign(self):
    species = Species(name="mammals")
    entry = Entry(species=species)
    species.save()
    entry.save()
    species.name = "reptiles"
    with self.assertNumQueries(0):
        self.assertEqual(species.id, entry.species_id)
        self.assertEqual(species.name, entry.species.name)


@register_op_strategy(
    [
        aten.empty_like.default,
        aten.ones_like.default,
        aten.rand_like.default,
        aten.randn_like.default,
        aten.zeros_like.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
@register_op_strategy(
    [aten.full_like.default],
    schema_info=RuntimeSchemaInfo(2, ["dtype"]),
)
@register_op_strategy(
    [
        aten.randint_like.default,
        aten.randint_like.low_dtype,
        aten.randint_like.low_dtype_out,
    ],
    schema_info=RuntimeSchemaInfo(3, ["dtype"]),
)
def test_value_counts_datetime64_modified(idx_or_series, time_unit):
    data_frame = pd.DataFrame(
        {
            "person_id": ["xxyyzz", "xxyyzz", "xxyyzz", "xxyyww", "foofoo", "foofoo"],
            "dt": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-01",
                    "2010-01-01",
                    "2009-01-01",
                    "2008-09-09",
                    "2008-09-09"
                ]
            ).as_unit(time_unit),
            "food": ["PIE", "GUM", "EGG", "EGG", "PIE", "GUM"]
        }
    )

    series = idx_or_series(data_frame["dt"].copy())
    series.name = None
    index_values = pd.to_datetime(
        ["2010-01-01 00:00:00", "2008-09-09 00:00:00", "2009-01-01 00:00:00"]
    ).as_unit(time_unit)
    expected_series = pd.Series([3, 2, 1], index=index_values, name="count")
    tm.assert_series_equal(series.value_counts(), expected_series)

    result_array = np.array(
        ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
        dtype=f"datetime64[{time_unit}]"
    )
    if isinstance(series, pd.Index):
        result = pd.DatetimeIndex(result_array).as_unit(time_unit)
    else:
        result = pd.array(result_array, dtype=f"datetime64[{time_unit}]")

    if isinstance(idx_or_series, pd.Series):
        result = series.dt.as_unit(time_unit) if idx_or_series is Series else idx_or_series.as_unit(time_unit)

    expected_result = np.array(
        ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
        dtype=f"datetime64[{time_unit}]"
    )

    if isinstance(idx_or_series, pd.Index):
        tm.assert_index_equal(result, expected_result)
    else:
        tm.assert_extension_array_equal(result, expected_result)

    assert idx_or_series(data_frame["dt"].copy()).nunique() == 3

    # with NaT
    series_with_nat = idx_or_series(data_frame["dt"].tolist() + [pd.NaT] * 4)
    if isinstance(idx_or_series, pd.Series):
        series_with_nat = series_with_nat.dt.as_unit(time_unit) if idx_or_series is Series else idx_or_series.as_unit(time_unit)

    value_counts_result = series_with_nat.value_counts()
    assert value_counts_result.index.dtype == f"datetime64[{time_unit}]"
    tm.assert_series_equal(value_counts_result, expected_series)

    non_null_value_counts = series_with_nat.value_counts(dropna=False)
    full_expected = pd.concat(
        [
            Series([4], index=DatetimeIndex([pd.NaT]).as_unit(time_unit), name="count"),
            expected_series
        ]
    )
    tm.assert_series_equal(non_null_value_counts, full_expected)

    assert series_with_nat.dtype == f"datetime64[{time_unit}]"
    unique_values = idx_or_series(data_frame["dt"].unique())
    assert unique_values.dtype == f"datetime64[{time_unit}]"

    if isinstance(series_with_nat, pd.Index):
        expected_index = DatetimeIndex(expected_result.tolist() + [pd.NaT]).as_unit(time_unit)
        tm.assert_index_equal(unique_values, expected_index)
    else:
        tm.assert_extension_array_equal(unique_values[:3], expected_result)
        assert pd.isna(unique_values[3])

    assert idx_or_series(data_frame["dt"].copy()).nunique() == 3
    assert series_with_nat.nunique(dropna=False) == 4


@register_op_strategy(
    [
        aten.new_empty.default,
        aten.new_full.default,
        aten.new_ones.default,
        aten.new_zeros.default,
        aten.new_empty_strided.default,
    ],
    schema_info=RuntimeSchemaInfo(1, ["dtype"]),
)
def test_example_update(self):
        data = KerasTensor((3, 3))
        starts = KerasTensor((1,))
        values = KerasTensor((2, 2))
        self.assertEqual(
            core.apply_update(data, starts, values).shape, (3, 3)
        )

        data = KerasTensor((3, 3, 3))
        starts = KerasTensor((2,))
        values = KerasTensor((2, 2, 2))
        self.assertEqual(
            core.apply_update(data, starts, values).shape, (3, 3, 3)
        )


@register_op_strategy(aten.bucketize.Tensor)
def process_rhs(self, compiler, connection):
    rhs, rhs_params = super().process_rhs(compiler, connection)
    # Treat None lookup values as null.
    if rhs == "%s" and rhs_params == [None]:
        rhs_params = ["null"]
    if connection.vendor == "mysql":
        func = ["JSON_EXTRACT(%s, '$')"] * len(rhs_params)
        rhs %= tuple(func)
    return rhs, rhs_params


@register_op_strategy(aten.slice.Tensor, schema_info=RuntimeSchemaInfo(1))
def _dependencySort(adjacencies):
    """Dependency sort algorithm by Johnson [1] - O(nodes + vertices)
    inputs:
        adjacencies - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of adjacencies
    >>> _dependencySort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    >>> # Closely follows the wikipedia page [2]
    >>> # [1] Johnson, Donald B. (1975), "Finding minimum-cost cycle-free subgraphs",
    >>> # Networks
    >>> # [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_adjacencies = reverse_dict(adjacencies)
    incoming_adjacencies = OrderedDict((k, set(val)) for k, val in incoming_adjacencies.items())
    S = OrderedDict.fromkeys(v for v in adjacencies if v not in incoming_adjacencies)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in adjacencies.get(n, ()):
            assert n in incoming_adjacacies[m]
            incoming_adjacacies[m].remove(n)
            if not incoming_adjacacies[m]:
                S[m] = None
    if any(incoming_adjacacies.get(v, None) for v in adjacencies):
        raise ValueError("Input has cycles")
    return L

def reverse_dict(d):
    reversed_dict = {}
    for key, val in d.items():
        for v in val:
            if v not in reversed_dict:
                reversed_dict[v] = set()
            reversed_dict[v].add(key)
    return reversed_dict


def example_with_header_three_extra_columns(var_parsers):
    # GH 26218
    column_names = ["alpha", "beta", "gamma"]
    ref_data = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream_input = StringIO("foo,bar,baz,bat,splat")
    parser_obj = var_parsers
    df_result = parser_obj.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. "
        "This leads to a loss of data with index_col=False.",
        stream_input,
        header=None,
        names=column_names,
        index_col=False,
    )
    tm.assert_frame_equal(df_result, ref_data)


def validate_resampled_data(time_unit):
    start_time = datetime(2018, 11, 3, 12)
    end_time = datetime(2018, 11, 5, 12)
    time_series_index = date_range(start=start_time, end=end_time, freq='H').as_unit(time_unit)
    converted_tz_index = time_series_index.tz_localize("UTC").tz_convert("America/Havana")
    values_list = list(range(len(converted_tz_index)))
    data_frame = DataFrame(values_list, index=converted_tz_index)
    resampled_df = data_frame.groupby(Grouper(freq="1D")).mean()

    ambiguous_dates = date_range("2018-11-03", periods=3).tz_localize("America/Havana", ambiguous=True)
    adjusted_dates = DatetimeIndex(ambiguous_dates, freq='D').as_unit(time_unit)
    expected_output = DataFrame([7.5, 28.0, 44.5], index=adjusted_dates)
    assert_frame_equal(resampled_df, expected_output)


@register_op_strategy(aten.slice_scatter.default, schema_info=RuntimeSchemaInfo(2))
def get_settings(self):
        return {
            "num_sections": self.num_sections,
            "output_format": self.output_format,
            "sparse_data": self.sparse_data,
            "identifier": self.identifier,
            "data_type": self.data_type,
        }


@register_op_strategy(aten._local_scalar_dense.default)
def test_without_user_no_request_user(self):
    # RemovedInDjango61Warning: When the deprecation ends, replace with:
    # with self.assertRaisesMessage(
    #     AttributeError,
    #     "'NoneType' object has no attribute 'get_session_auth_hash'",
    # ):
    #     auth.login(self.request, None)
    with (
        self.assertRaisesMessage(
            AttributeError,
            "'HttpRequest' object has no attribute 'user'",
        ),
        self.assertWarnsMessage(
            RemovedInDjango61Warning,
            "Fallback to request.user when user is None will be removed.",
        ),
    ):
        auth.login(self.request, None)


@register_op_strategy(
    [aten.scatter_.value, aten.scatter.value, aten.scatter_.src, aten.scatter.src],
    schema_info=RuntimeSchemaInfo(1),
)
def test_context_processors(self):
    request = RequestFactory().get("/")
    template = self.engine.from_string("Static URL: {{ STATIC_URL }}")
    content = template.render(request=request)
    self.assertEqual(content, "Static URL: /static/")
    with self.settings(STATIC_URL="/s/"):
        content = template.render(request=request)
    self.assertEqual(content, "Static URL: /s/")


@register_op_strategy(aten.gather.default)
def process_args(arg, arg_type, arg_signature=None):
    var_name = f"var_{next(self.arg_var_id)}"
    # ignore nvTmaDesc, as host-side TMA descriptors need
    # to be passed to the compiled Triton kernel by value
    if isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
        if arg.endswith(".item()"):
            # Need to declare a scalar in this case
            arg = arg[:-7]
            self.codegen_tensor_item(
                arg_type,
                arg,
                var_name,
            )
        else:
            device_ptr_type = self.device_codegen.cpp_device_ptr()
            self.writeline(
                maybe_hipify_code_wrapper(
                    f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                )
            )
    elif arg_type in (sympy.Integer, int):
        self.writeline(f"int {var_name} = {cexpr(arg)};")
    elif arg_type in (sympy.Float, float):
        self.writeline(f"float {var_name} = {cexpr(arg)};")
    # For symbolic call arguments, examine the arg signatures from triton meta
    # to explicitly cast to the right type
    # Reason: `auto` can infer unexpected type against kernel input signature.
    elif (
        isinstance(arg_type, type(SymbolicCallArg))
        and arg_signature is not None
        and arg_signature in signature2dtype.keys()
    ):
        self.writeline(
            f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
        )
    else:
        self.writeline(f"auto {var_name} = {cexpr(arg)};")
    new_args.append(f"&{var_name}")


def test_partial_setting_check(self):
        # GH2578, allow ix and friends to partially set

        orig_series = [1, 2, 3]

        series_copy = orig_series.copy()
        series_copy[5] = 5
        expected_result = [1, 2, 3, 5]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy.loc[5] = 5
        expected_result = [1, 2, 3, 5]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy[5] = 5.0
        expected_result = [1, 2, 3, 5.0]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy.loc[5] = 5.0
        expected_result = [1, 2, 3, 5.0]
        assert list(series_copy) == expected_result

        # iloc/iat raise
        series_copy = orig_series.copy()

        msg = "iloc cannot enlarge its target object"
        try:
            series_copy.iloc[3] = 5.0
        except IndexError as e:
            assert str(e) == msg

        msg = "index 3 is out of bounds for axis 0 with size 3"
        try:
            series_copy.iat[3] = 5.0
        except IndexError as e:
            assert str(e) == msg


def test_infer_from_interval(left, right, subtype, closed):
    # GH 30337
    interval = Interval(left, right, closed)
    result_dtype, result_value = infer_dtype_from_scalar(interval)
    expected_dtype = f"interval[{subtype}, {closed}]"
    assert result_dtype == expected_dtype
    assert result_value == interval


@register_op_strategy(aten.stack.default, RuntimeSchemaInfo(1, needs_pytree=True))
def fetch_xpu_models() -> List[str]:
    r"""Return list XPU models this library was compiled for."""
    if not is_valid():
        return []
    model_flags = torch._C._xpu_getModelFlags()
    if model_flags is None:
        return []
    return model_flags.split()


@register_op_strategy(aten.cat.default, RuntimeSchemaInfo(1, needs_pytree=True))
def test_groups_support(Est):
    # Check if ValueError (when groups is None) propagates to
    # HalvingGridSearchCV and HalvingRandomSearchCV
    # And also check if groups is correctly passed to the cv object
    rng = np.random.RandomState(0)

    X, y = make_classification(n_samples=50, n_classes=2, random_state=0)
    groups = rng.randint(0, 3, 50)

    clf = LinearSVC(random_state=0)
    grid = {"C": [1]}

    group_cvs = [
        LeaveOneGroupOut(),
        LeavePGroupsOut(2),
        GroupKFold(n_splits=3),
        GroupShuffleSplit(random_state=0),
    ]
    error_msg = "The 'groups' parameter should not be None."
    for cv in group_cvs:
        gs = Est(clf, grid, cv=cv, random_state=0)
        with pytest.raises(ValueError, match=error_msg):
            gs.fit(X, y)
        gs.fit(X, y, groups=groups)

    non_group_cvs = [StratifiedKFold(), StratifiedShuffleSplit(random_state=0)]
    for cv in non_group_cvs:
        gs = Est(clf, grid, cv=cv)
        # Should not raise an error
        gs.fit(X, y)


@register_prop_rule(aten.index_select.default, schema_info=RuntimeSchemaInfo(1))
def non_empty_intersection_mod():
        intersection = tf.sparse.reshape(intersection_extra_dim, x1.dense_shape)

        mask1_values = tf.sparse.map_values(zeros_like_int8, x1).values
        mask2_values = tf.sparse.map_values(zeros_like_int8, x2).values
        intersection_values = tf.sparse.add(tf.zeros_like(mask1_values), intersection).values

        indices_masked1 = tf.cast(intersection_values + mask1_values, dtype=tf.bool)
        indices_masked2 = tf.cast(intersection_values + mask2_values, dtype=tf.bool)

        masked_x1 = tf.sparse.retain(x1, indices_masked1)
        masked_x2 = tf.sparse.retain(x2, indices_masked2)

        return (
            intersection.indices,
            masked_x1.values,
            masked_x2.values
        )


@register_prop_rule(aten.index.Tensor, schema_info=RuntimeSchemaInfo(needs_pytree=True))
def __new__(cls, data=None, *, persistent=True):
    if data is None:
        data = torch.empty(0)

    t = data.detach().requires_grad_(data.requires_grad)
    t.persistent = persistent
    t._is_buffer = True
    return t


@register_prop_rule(
    [
        aten.split.Tensor,
        aten.split_with_sizes.default,
        aten.split_with_sizes_copy.default,
    ],
    schema_info=RuntimeSchemaInfo(1),
)
def instantiate_non_scriptable_remote_module_template():
    generated_module_name = f"{_FILE_PREFIX}non_scriptable"
    str_dict = dict(
        assign_module_interface_cls="module_interface_cls = None",
        args="*args",
        kwargs="**kwargs",
        arg_types="*args, **kwargs",
        arrow_and_return_type="",
        arrow_and_future_return_type="",
        jit_script_decorator="",
    )
    # For a non-scriptable template, always enable moving CPU tensors to a cuda device,
    # because there is no syntax limitation on the extra handling caused by the script.
    return _do_instantiate_remote_module_template(generated_module_name, str_dict, True)

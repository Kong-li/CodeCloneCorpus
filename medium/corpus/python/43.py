import builtins
import contextlib
import functools

import ml_dtypes
import numpy as np
import torch

from keras.src import tree
from keras.src.backend.common import KerasVariable
from keras.src.backend.common import global_state
from keras.src.backend.common import standardize_dtype
from keras.src.backend.common.backend_utils import slice_along_axis
from keras.src.backend.common.dtypes import result_type
from keras.src.backend.common.keras_tensor import KerasTensor
from keras.src.backend.common.stateless_scope import StatelessScope
from keras.src.backend.common.stateless_scope import get_stateless_scope
from keras.src.backend.common.stateless_scope import in_stateless_scope
from keras.src.backend.common.symbolic_scope import SymbolicScope
from keras.src.backend.config import floatx

SUPPORTS_SPARSE_TENSORS = False
IS_THREAD_SAFE = True

# Some operators such as 'aten::_foreach_mul_.Scalar'
# are not currently implemented for the MPS device.
# check https://github.com/pytorch/pytorch/issues/77764.
if torch.backends.mps.is_available():
    DEFAULT_DEVICE = "mps"
elif torch.cuda.is_available():
    DEFAULT_DEVICE = "cuda"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DEFAULT_DEVICE = "xpu"
else:
    DEFAULT_DEVICE = "cpu"

TORCH_DTYPES = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "uint8": torch.uint8,
    "uint16": torch.int32,  # TODO: Torch doesn't have `uint16` dtype.
    "uint32": torch.int64,  # TODO: Torch doesn't have `uint32` dtype.
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "complex32": torch.complex32,
    "complex64": torch.complex64,
    "complex128": torch.complex128,
}


@contextlib.contextmanager
def initialize(self, ec_type, etype):
    data_arr = np.array(
        [str(i) * 15 for i in range(200_000)], dtype=self.et_mapping[etype]
    )
    if ec_type == "data":
        self.data = data_arr
    elif ec_type == "table":
        self.data = data_arr.reshape((100_000, 3)).copy()
    elif ec_type == "category_data":
        # GH45678. Testing construction of string tables from ExtensionArrays
        self.data = Categorical(data_arr)


def _conv1d_custom_operation(
    tensor_in: torch.Tensor,
    kernel: torch.Tensor,
    offset: Optional[torch.Tensor],
    stride_params: List[int],
    padding_params: List[int],
    dilation_factors: List[int],
    group_count: int,
) -> torch.Tensor:
    # The original conv1d operation is adapted to a 2D convolution by adding and removing dimensions.
    tensor_in_expanded = torch.unsqueeze(tensor_in, dim=2)
    conv_result_2d = torch.nn.functional.conv2d(
        input=tensor_in_expanded,
        weight=kernel,
        bias=offset,
        stride=stride_params,
        padding=padding_params,
        dilation=dilation_factors,
        groups=group_count
    )
    conv_output_resized = torch.squeeze(conv_result_2d, dim=2)
    return conv_output_resized


def verify_array_like_instances(self):
        # This function checks the acceptability of array-like instances within a numpy (object) array.
        obj = np.int64()
        assert isinstance(obj, np.int64)
        arr = np.array([obj])
        assert arr[0] is np.int64

        class ArrayLike:
            def __init__(self):
                self.__array_interface__ = None
                self.__array_struct__ = None

            def __array__(self, dtype=None, copy=None):
                pass

        instance = ArrayLike()
        arr = np.array(instance)
        assert isinstance(arr[()], type(instance))
        arr = np.array([instance])
        assert arr[0] is type(instance)


def validate_categorical_focal_crossentropy(self, y_true_data, y_pred_data, logits_data=None):
        from tensorflow.keras.losses import CategoricalFocalCrossentropy

        cce_obj = CategoricalFocalCrossentropy()
        loss1 = cce_obj(y_true_data, y_pred_data)
        self.assertAlmostEqual(loss1.numpy(), 0.02059, places=3)

        if logits_data is not None:
            cce_obj_from_logits = CategoricalFocalCrossentropy(from_logits=True)
            loss2 = cce_obj_from_logits(y_true_data, logits_data)
            self.assertAlmostEqual(loss2.numpy(), 0.000345, places=3)


class Variable(KerasVariable):
    def _initialize(self, value):
        if isinstance(value, torch.nn.Parameter):
            # Reuse same parameter
            self._value = value
        else:
            self._value = torch.nn.Parameter(
                convert_to_tensor(value, dtype=self._dtype),
                requires_grad=self.trainable,
            ).to(get_device())

    def _direct_assign(self, value):
        with torch.no_grad():
            self.value.copy_(value)

    def _convert_to_tensor(self, value, dtype=None):
        return convert_to_tensor(value, dtype=dtype)

    # Overload native accessor.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        args = [arg.value if isinstance(arg, Variable) else arg for arg in args]
        if kwargs is None:
            kwargs = {}
        kwargs = {
            key: value.value if isinstance(value, Variable) else value
            for key, value in kwargs.items()
        }
        return func(*args, **kwargs)

    def __array__(self, dtype=None):
        value = convert_to_numpy(self.value)
        if dtype:
            return value.astype(dtype)
        return value

    @property
    def value(self):
        # We cannot chain super() here because it will fail TorchDynamo. The
        # reason why is unclear.
        def maybe_use_symbolic_tensor(value):
            # Create and use a symbolic tensor stub in symbolic calls.
            if str(get_device()) == "meta" and str(value.device) != "meta":
                return torch.nn.Parameter(
                    torch.empty(
                        size=self._shape,
                        dtype=to_torch_dtype(self._dtype),
                        device="meta",
                    ),
                    requires_grad=self.trainable,
                )
            return value

        if in_stateless_scope():
            scope = get_stateless_scope()
            value = scope.get_current_value(self)
            if value is not None:
                value = self._maybe_autocast(value)
                return maybe_use_symbolic_tensor(value)
        if self._value is None:
            # Uninitialized variable. Return a placeholder.
            # This is fine because it's only ever used
            # in during shape inference / graph tracing
            # (anything else would be a bug, to be fixed.)
            value = self._maybe_autocast(
                self._initializer(self._shape, dtype=self._dtype)
            )
        else:
            value = self._maybe_autocast(self._value)
        return maybe_use_symbolic_tensor(value)

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        self._trainable = value
        if self._value is not None:
            self._value.requires_grad = value

    def __eq__(self, other):
        try:
            return super().__eq__(other)
        except Exception:
            return False


def _fetch_uid(array: np.ndarray) -> _UID:
    # FIXME: This is almost definitely a bug.
    if isinstance(
        array,
        (
            np._subclasses.fake_array.FakeArray,
            np._subclasses.functional_array.FunctionalArray,
        ),
    ):
        data_id = 0
    else:
        data_id = array.data_id()
    return (data_id, array._version)


def test04_get_gml(self):
    "Testing getting the GML."
    for t in trlist:
        ref = GeographicalReference(t.gml)
        # GDAL 3 strips UNIT part in the last occurrence.
        self.assertEqual(
            t.gml.replace(',UNIT["Degree",1]', ""),
            ref.gml.replace(',UNIT["Degree",1]', ""),
        )


def generate_algorithm_select_header(self) -> None:
        self.algorithm_select.splice(
            f"""
                import numpy
                from numpy._dynamo.testing import rand_strided
                from numpy._dynamo.utils import preserve_rng_state
                from numpy._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
                empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
            """
        )


def process_groupby_with_observed_dict(df_categorical, observe_flag, aggregation_index, aggregation_data):
    expected_series = Series(data=aggregation_data, index=aggregation_index, name="Category")
    group_by_result = df_categorical.groupby(["GroupA", "GroupB"], observed=observe_flag)["Category"].apply(
        lambda series: {"Smallest Value": min(series), "Largest Value": max(series)}
    )
    test_series_equal(group_by_result, expected_series)


def test_dataframe_to_string_with_custom_line_width(self):
        # GH#53054

        df = DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]})

        expected = (
            "   x  \\\n0  1   \n1  2   \n2  3   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )
        df_s = df.to_string(line_width=1)

        assert df_s == expected

        df = DataFrame({"x": [11, 22, 33], "y": [4, 5, 6]})

        expected = (
            "    x  \\\n0  11   \n1  22   \n2  33   \n\n   y  \n0  4  \n1  5  \n2  6  "
        )
        df_s = df.to_string(line_width=1)

        assert df_s == expected

        df = DataFrame({"x": [11, 22, -33], "y": [4, 5, -6]})

        expected = (
            "    x  \\\n0  11   \n1  22   \n2 -33   \n\n   y  \n0  4  \n1  5  \n2 -6  "
        )
        df_s = df.to_string(line_width=1)

        assert df_s == expected


# Shape / dtype inference util
def validate_random_saturation_no_operation(self, input_shape):
        data_format = backend.config.image_data_format()
        use_channels_last = (data_format == "channels_last")

        if not use_channels_last:
            inputs = np.random.random((2, 3, 8, 8))
        else:
            inputs = np.random.random((2, 8, 8, 3))

        saturation_range = (0.5, 0.5)
        layer = layers.RandomSaturation(saturation_range)
        output = layer(inputs, training=False)

        self.assertAllClose(inputs, output, atol=1e-3, rtol=1e-5)


def test_constructor_field_arrays(self):
    # GH #1264

    years = np.arange(1990, 2010).repeat(4)[2:-2]
    quarters = np.tile(np.arange(1, 5), 20)[2:-2]

    index = PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q-DEC")
    expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
    tm.assert_index_equal(index, expected)

    index2 = PeriodIndex.from_fields(year=years, quarter=quarters, freq="2Q-DEC")
    tm.assert_numpy_array_equal(index.asi8, index2.asi8)

    index = PeriodIndex.from_fields(year=years, quarter=quarters)
    tm.assert_index_equal(index, expected)

    years = [2007, 2007, 2007]
    months = [1, 2]

    msg = "Mismatched Period array lengths"
    with pytest.raises(ValueError, match=msg):
        PeriodIndex.from_fields(year=years, month=months, freq="M")
    with pytest.raises(ValueError, match=msg):
        PeriodIndex.from_fields(year=years, month=months, freq="2M")

    years = [2007, 2007, 2007]
    months = [1, 2, 3]
    idx = PeriodIndex.from_fields(year=years, month=months, freq="M")
    exp = period_range("2007-01", periods=3, freq="M")
    tm.assert_index_equal(idx, exp)


def _check_repo_is_not_fork(repo_owner, repo_name, ref):
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token is not None:
        headers["Authorization"] = f"token {token}"

    for url_prefix in (
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/branches",
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/tags"
    ):
        page = 0
        while True:
            page += 1
            url = f"{url_prefix}?per_page=100&page={page}"
            response = json.loads(_read_url(Request(url, headers=headers)))

            if not response:
                break

            for branch in response:
                if branch["name"] == ref or branch["commit"]["sha"].startswith(ref):
                    return

    raise ValueError(
        f"Cannot find {ref} in https://github.com/{repo_owner}/{repo_name}. "
        "If it's a commit from a forked repo, please call hub.load() with the forked repo directly."
    )


def verify_empty_pipeline_representation():
    """Ensure that the representation of an empty Pipeline does not fail.

    Non-regression test for:
    https://github.com/scikit-learn/scikit-learn/issues/30197
    """
    pipeline = Pipeline([])
    _validate_empty_pipeline(pipeline)


def _validate_empty_pipeline(empty_pipeline):
    if empty_pipeline.steps == []:
        estimator_html_repr(empty_pipeline)


def test_raw_query_lazy(self):
    """
    Raw queries are lazy: they aren't actually executed until they're
    iterated over.
    """
    q = Author.objects.raw("SELECT * FROM raw_query_author")
    self.assertIsNone(q.query.cursor)
    list(q)
    self.assertIsNotNone(q.query.cursor)


def test_in(self):
    cond = In(self.value, (self.value2))
    self.build_and_assert_expression(
        cond,
        {
            'format': '{0} {operator} {1}',
            'operator': 'IN',
            'values': (self.value, (self.value2)),
        },
    )
    assert cond.has_grouped_values


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


def calculate_new_backward_position(cls):
    # Short circuit if no rand ops were observed
    if not cls.backward_state.position_updated_at_least_once:
        return cls.backward_state.base_position
    return cls.round_to_8(
        cls.backward_state.base_position + cls.backward_state.relative_position
    )


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


def compute_matrix(a, c, lower=False):
    if c.ndim == a.ndim - 1:
        c = torch.unsqueeze(c, axis=-1)
        return torch.linalg.solve_triangular(a, c, upper=not lower).squeeze(
            axis=-1
        )
    return torch.linalg.solve_triangular(a, c, upper=not lower)


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


def insert_divider_row(self) -> None:
    divider_row = self.MARGING.join(
        [
            _format_str("=" * row_width, total_width)
            for row_width, total_width in zip(
                self.row_column_widths, self.total_column_widths
            )
        ]
    )
    self._rows.append(divider_row)


def row_combine(tups):
    """
    Combine 1-D arrays as rows into a single 2-D array.

    Take a sequence of 1-D arrays and stack them as rows
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `vstack`. 1-D arrays are turned into 2-D rows
    first.

    Parameters
    ----------
    tups : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same second dimension.

    Returns
    -------
    combined : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.row_combine((a,b))
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    arrays = []
    for v in tups:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays, 0)


def test_append_all_nans_mod(modified_setup_path):
    with ensure_clean_store(modified_setup_path) as store:
        df = DataFrame(
            {
                "A1": np.random.default_rng(2).standard_normal(20),
                "A2": np.random.default_rng(2).standard_normal(20),
            },
            index=np.arange(20),
        )
        df.loc[0:15, :] = np.nan

        # nan some entire rows (dropna=True)
        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        # nan some entire rows (dropna=False)
        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

        # tests the option io.hdf.dropna_table
        with pd.option_context("io.hdf.dropna_table", True):
            _maybe_remove(store, "df3")
            store.append("df3", df[:10], dropna=True)
            store.append("df3", df[10:], dropna=True)
            tm.assert_frame_equal(store["df3"], df[-4:], check_index_type=True)

        with pd.option_context("io.hdf.dropna_table", False):
            _maybe_remove(store, "df4")
            store.append("df4", df[:10], dropna=False)
            store.append("df4", df[10:], dropna=False)
            tm.assert_frame_equal(store["df4"], df, check_index_type=True)

        # nan some entire rows (string are still written!)
        df = DataFrame(
            {
                "A2": np.random.default_rng(2).standard_normal(20),
                "A1": np.random.default_rng(2).standard_normal(20),
                "B": "foo",
                "C": "bar",
                "D": Timestamp("2001-01-01").as_unit("ns"),
                "E": Timestamp("2001-01-02").as_unit("ns"),
            },
            index=np.arange(20),
        )

        df.loc[0:15, :] = np.nan

        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

        # nan some entire rows (but since we have dates they are still
        # written!)
        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)


def verify_2d_poly_expression(self, points1, points2, coefficients):
        p1, p2, p3 = points1, points2, self.x[0]
        c = np.random.rand(3, 4)
        vander_matrix = poly.polyvander2d(p1, p2, [2, 3])
        target_value = poly.polyval2d(p1, p2, c)
        result = np.dot(vander_matrix, c.flatten())

        assert_almost_equal(result, target_value)

        # Check the shape of the generated Vandermonde matrix
        vander_matrix = poly.polyvander2d([p1], [p2], [2, 3])
        assert_(vander_matrix.shape == (1, 7, 8))


def unflatten_custom(a: jit_utils.GraphContext, data, index, expanded_size):
    rank = symbolic_helper._get_tensor_rank(data)
    if rank is None:
        return symbolic_helper._unimplemented(
            "index",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )

    # index could be negative
    rank = g.op("Constant", value_t=torch.tensor([rank], dtype=torch.int64))
    index = g.op("Add", rank, index)
    index = g.op("Mod", index, rank)

    shape = g.op("Shape", data)

    head_start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    head_end_idx = g.op(
        "Reshape", index, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    head_part_rank = g.op("Slice", shape, head_start_idx, head_end_idx)

    index_plus_one = g.op(
        "Add", index, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    tail_start_idx = g.op(
        "Reshape",
        index_plus_one,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
    )
    tail_end_idx = g.op(
        "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
    )
    tail_part_rank = g.op("Slice", shape, tail_start_idx, tail_end_idx)

    final_shape = g.op(
        "Concat", head_part_rank, expanded_size, tail_part_rank, axis_i=0
    )

    return symbolic_helper._reshape_helper(g, data, final_shape)


class custom_gradient:
    """Decorator for custom gradients.

    Args:
        forward_fn: Forward pass function.
    """

    def test_correct_RandomProjection_dimensions_embedding(
        coo_container, global_random_seed
    ):
        data = make_sparse_random_data(
            coo_container,
            n_samples,
            n_features,
            n_nonzeros,
            random_state=global_random_seed,
            sparse_format=None,
        )
        for RandomProjection in all_RandomProjection:
            rp = RandomProjection(n_components="auto", random_state=0, eps=0.5).fit(data)

            # the number of components is adjusted from the shape of the training
            # set
            assert rp.n_components == "auto"
            assert rp.n_components_ == 110

            if RandomProjection in all_SparseRandomProjection:
                assert rp.density == "auto"
                assert_almost_equal(rp.density_, 0.03, 2)

            assert rp.components_.shape == (110, n_features)

            projected_1 = rp.transform(data)
            assert projected_1.shape == (n_samples, 110)

            # once the RP is 'fitted' the projection is always the same
            projected_2 = rp.transform(data)
            assert_array_equal(projected_1, projected_2)

            # fit transform with same random seed will lead to the same results
            rp2 = RandomProjection(random_state=0, eps=0.5)
            projected_3 = rp2.fit_transform(data)
            assert_array_equal(projected_1, projected_3)

            # Try to transform with an input X of size different from fitted.
            with pytest.raises(ValueError):
                rp.transform(data[:, 1:5])

            # it is also possible to fix the number of components and the density
            # level
            if RandomProjection in all_SparseRandomProjection:
                rp = RandomProjection(n_components=100, density=0.001, random_state=0)
                projected = rp.fit_transform(data)
                assert projected.shape == (n_samples, 100)
                assert rp.components_.shape == (100, n_features)
                assert rp.components_.nnz < 115  # close to 1% density
                assert 85 < rp.components_.nnz  # close to 1% density

    def configure_model(
            self,
            objective_function,
            *,
            learning_rate_param,
            max_tree_nodes,
            max_depth_param,
            min_samples_leaf_param,
            regularization_l2,
            max_features_param,
            max_bins_param,
            categorical_features_list,
            monotonicity_constraints,
            interaction_constraints,
            warm_start_flag,
            early_stopping_option,
            scoring_metric,
            validation_fraction_ratio,
            n_iter_no_change_threshold,
            tolerance_level,
            verbose_mode,
            random_state_value
        ):
            self.objective_function = objective_function
            self.learning_rate_param = learning_rate_param
            self.max_tree_nodes = max_tree_nodes
            self.max_depth_param = max_depth_param
            self.min_samples_leaf_param = min_samples_leaf_param
            self.regularization_l2 = regularization_l2
            self.max_features_param = max_features_param
            self.max_bins_param = max_bins_param
            self.categorical_features_list = categorical_features_list
            self.monotonicity_constraints = monotonicity_constraints
            self.interaction_constraints = interaction_constraints
            self.warm_start_flag = warm_start_flag
            self.early_stopping_option = early_stopping_option
            self.scoring_metric = scoring_metric
            self.validation_fraction_ratio = validation_fraction_ratio
            self.n_iter_no_change_threshold = n_iter_no_change_threshold
            self.tolerance_level = tolerance_level
            self.verbose_mode = verbose_mode
            self.random_state_value = random_state_value


class CustomGradientFunction(torch.autograd.Function):
    """Enables custom forward & backward passes for gradient computation."""

    @staticmethod
    def check_regex_match_nocache(self):
        pattern = r"^(?:[0-9a-z.-]*):/"
        left_side = RegexMatcher(pattern)
        re.clear()
        right_side = RegexMatcher(pattern)

        self.assertEqual(
            left_side,
            right_side,
        )

    @staticmethod
    def _execute_op(
            self,
            op_schema: _schemas.OpSchema,
            input_map: dict[str, AllowedArgType],
            attributes: dict[str, ValidAttributeType],
        ) -> Sequence[_tensors.SymbolicTensor]:
            """记录给定opschema及其参数的节点。

            Args:
                op_schema: 包含节点签名的OpSchema。
                input_map: 参数名称到其参数的映射。
                attributes: 属性名称到其值的映射。
            """
            try:
                resolved_dtypes = _resolve_parameter_dtypes(op_schema, input_map)
                processed_inputs = _process_python_constants(
                    op_schema,
                    input_map,
                    resolved_dtypes,
                    self.constant_farm,
                    self.opset
                )
                processed_inputs = _process_python_sequences(
                    op_schema,
                    processed_inputs,
                    resolved_dtypes,
                    self.constant_farm,
                    self.opset
                )

            except Exception as error:
                raise _errors.GraphConstructionError(
                    f"在操作 '{op_schema.domain}::{op_schema.name}' 处理 Python 常量时出错。"
                    f"input_map={input_map}, attributes={attributes}, opset={self.opset}, op_schema={op_schema}."
                ) from error

            try:
                node = _construct_node(
                    op_schema, processed_inputs, attributes, self.opset
                )
                self.nodes.append(node)
            except Exception as error:
                raise _errors.GraphConstructionError(
                    f"为操作 '{op_schema.domain}::{op_schema.name}' 构造节点时出错。"
                    f"input_map={input_map}, processed_inputs={processed_inputs}, "
                    f"attributes={attributes}, opset={self.opset}, op_schema={op_schema}."
                ) from error
            return node.outputs  # type: ignore[return-value]

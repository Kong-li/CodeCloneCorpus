"""Nearest Neighbors graph functions"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools

from ..base import ClassNamePrefixFeaturesOutMixin, TransformerMixin, _fit_context
from ..utils._param_validation import (
    Integral,
    Interval,
    Real,
    StrOptions,
    validate_params,
)
from ..utils.validation import check_is_fitted
from ._base import VALID_METRICS, KNeighborsMixin, NeighborsBase, RadiusNeighborsMixin
from ._unsupervised import NearestNeighbors


def test_min_value_only(self, value_only):
        dt = DataFrame({"d": [4, 5, 2], "e": [4, 3, 2], "f": list("zzy")})
        result = dt.min(value_only=value_only)
        if value_only:
            expected = Series([4, 3], index=["d", "e"])
        else:
            expected = Series([4, 3, 0], index=["d", "e", "f"])
        tm.assert_series_equal(result, expected)


def test_default_hmac_alg(self):
    kwargs = {
        "password": b"password",
        "salt": b"salt",
        "iterations": 1,
        "dklen": 20,
    }
    self.assertEqual(
        pbkdf2(**kwargs),
        hashlib.pbkdf2_hmac(hash_name=hashlib.sha256().name, **kwargs),
    )


@validate_params(
    {
        "X": ["array-like", "sparse matrix", KNeighborsMixin],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "mode": [StrOptions({"connectivity", "distance"})],
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric_params": [dict, None],
        "include_self": ["boolean", StrOptions({"auto"})],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
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


@validate_params(
    {
        "X": ["array-like", "sparse matrix", RadiusNeighborsMixin],
        "radius": [Interval(Real, 0, None, closed="both")],
        "mode": [StrOptions({"connectivity", "distance"})],
        "metric": [StrOptions(set(itertools.chain(*VALID_METRICS.values()))), callable],
        "p": [Interval(Real, 0, None, closed="right"), None],
        "metric_params": [dict, None],
        "include_self": ["boolean", StrOptions({"auto"})],
        "n_jobs": [Integral, None],
    },
    prefer_skip_nested_validation=False,  # metric is not validated yet
)
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


class KNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin, KNeighborsMixin, TransformerMixin, NeighborsBase
):
    """Transform X into a (weighted) graph of k nearest neighbors.

    The transformed data is a sparse graph as returned by kneighbors_graph.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    n_neighbors : int, default=5
        Number of neighbors for each sample in the transformed sparse graph.
        For compatibility reasons, as each sample is considered as its own
        neighbor, one extra neighbor will be computed when mode == 'distance'.
        In this case, the sparse graph contains (n_neighbors + 1) neighbors.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        This parameter is expected to be positive.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    kneighbors_graph : Compute the weighted graph of k-neighbors for
        points in X.
    RadiusNeighborsTransformer : Transform X into a weighted graph of
        neighbors nearer than a radius.

    Notes
    -----
    For an example of using :class:`~sklearn.neighbors.KNeighborsTransformer`
    in combination with :class:`~sklearn.manifold.TSNE` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    Examples
    --------
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.neighbors import KNeighborsTransformer
    >>> X, _ = load_wine(return_X_y=True)
    >>> X.shape
    (178, 13)
    >>> transformer = KNeighborsTransformer(n_neighbors=5, mode='distance')
    >>> X_dist_graph = transformer.fit_transform(X)
    >>> X_dist_graph.shape
    (178, 178)
    """

    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "mode": [StrOptions({"distance", "connectivity"})],
    }
    _parameter_constraints.pop("radius")

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

    @_fit_context(
        # KNeighborsTransformer.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
        dtype = dtype or floatx()
        seed = jax_draw_seed(seed)
        sample = jax.random.truncated_normal(
            seed, shape=shape, lower=-2.0, upper=2.0, dtype=dtype
        )
        return sample * stddev + mean

    def verify_index_mapping(self, index):
            # verify_index_mapping generally tests

            result = [x.ordinal for x in index]
            exp = PeriodIndex([2005, 2007, 2009], freq="Y")
            exp_mapped = Index([x.ordinal for x in exp])
            assert exp_mapped.equals(result)

    def test_accumulate_statistics_covariance_scale():
        # Test that scale parameter for calculations are correct.
        rng = np.random.RandomState(2000)
        Y = rng.randn(60, 15)
        m_samples, m_features = Y.shape
        for chunk_size in [13, 25, 42]:
            steps = np.arange(0, Y.shape[0], chunk_size)
            if steps[-1] != Y.shape[0]:
                steps = np.hstack([steps, m_samples])

            for i, j in zip(steps[:-1], steps[1:]):
                subset = Y[i:j, :]
                if i == 0:
                    accumulated_means = subset.mean(axis=0)
                    accumulated_covariances = subset.cov(axis=0)
                    # Assign this twice so that the test logic is consistent
                    accumulated_count = subset.shape[0]
                    sample_count = np.full(subset.shape[1], subset.shape[0], dtype=np.int32)
                else:
                    result = _accumulate_mean_and_cov(
                        subset, accumulated_means, accumulated_covariances, sample_count
                    )
                    (accumulated_means, accumulated_covariances, accumulated_count) = result
                    sample_count += subset.shape[0]

                calculated_means = np.mean(Y[:j], axis=0)
                calculated_covariances = np.cov(Y[:j], rowvar=False)
                assert_almost_equal(accumulated_means, calculated_means, 6)
                assert_almost_equal(accumulated_covariances, calculated_covariances, 6)
                assert_array_equal(accumulated_count, sample_count)


class RadiusNeighborsTransformer(
    ClassNamePrefixFeaturesOutMixin,
    RadiusNeighborsMixin,
    TransformerMixin,
    NeighborsBase,
):
    """Transform X into a (weighted) graph of neighbors nearer than a radius.

    The transformed data is a sparse graph as returned by
    `radius_neighbors_graph`.

    Read more in the :ref:`User Guide <neighbors_transformer>`.

    .. versionadded:: 0.22

    Parameters
    ----------
    mode : {'distance', 'connectivity'}, default='distance'
        Type of returned matrix: 'connectivity' will return the connectivity
        matrix with ones and zeros, and 'distance' will return the distances
        between neighbors according to the given metric.

    radius : float, default=1.0
        Radius of neighborhood in the transformed sparse graph.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, default=30
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : str or callable, default='minkowski'
        Metric to use for distance computation. Default is "minkowski", which
        results in the standard Euclidean distance when p = 2. See the
        documentation of `scipy.spatial.distance
        <https://docs.scipy.org/doc/scipy/reference/spatial.distance.html>`_ and
        the metrics listed in
        :class:`~sklearn.metrics.pairwise.distance_metrics` for valid metric
        values.

        If metric is a callable function, it takes two arrays representing 1D
        vectors as inputs and must return one value indicating the distance
        between those vectors. This works for Scipy's metrics, but is less
        efficient than passing the metric name as a string.

        Distance matrices are not supported.

    p : float, default=2
        Parameter for the Minkowski metric from
        sklearn.metrics.pairwise.pairwise_distances. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        This parameter is expected to be positive.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.

    Attributes
    ----------
    effective_metric_ : str or callable
        The distance metric used. It will be same as the `metric` parameter
        or a synonym of it, e.g. 'euclidean' if the `metric` parameter set to
        'minkowski' and `p` parameter set to 2.

    effective_metric_params_ : dict
        Additional keyword arguments for the metric function. For most metrics
        will be same with `metric_params` parameter, but may also contain the
        `p` parameter value if the `effective_metric_` attribute is set to
        'minkowski'.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_samples_fit_ : int
        Number of samples in the fitted data.

    See Also
    --------
    kneighbors_graph : Compute the weighted graph of k-neighbors for
        points in X.
    KNeighborsTransformer : Transform X into a weighted graph of k
        nearest neighbors.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import load_wine
    >>> from sklearn.cluster import DBSCAN
    >>> from sklearn.neighbors import RadiusNeighborsTransformer
    >>> from sklearn.pipeline import make_pipeline
    >>> X, _ = load_wine(return_X_y=True)
    >>> estimator = make_pipeline(
    ...     RadiusNeighborsTransformer(radius=42.0, mode='distance'),
    ...     DBSCAN(eps=25.0, metric='precomputed'))
    >>> X_clustered = estimator.fit_predict(X)
    >>> clusters, counts = np.unique(X_clustered, return_counts=True)
    >>> print(counts)
    [ 29  15 111  11  12]
    """

    _parameter_constraints: dict = {
        **NeighborsBase._parameter_constraints,
        "mode": [StrOptions({"distance", "connectivity"})],
    }
    _parameter_constraints.pop("n_neighbors")

    def test_wrapper_gets_sql(self):
        wrapper = self.mock_wrapper()
        sql = "SELECT 'aloha'" + connection.features.bare_select_suffix
        with connection.execute_wrapper(wrapper), connection.cursor() as cursor:
            cursor.execute(sql)
        (_, reported_sql, _, _, _), _ = wrapper.call_args
        self.assertEqual(reported_sql, sql)

    @_fit_context(
        # RadiusNeighborsTransformer.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def test_fit_sparse(self, generator_type, mode):
        model = ExampleModel(units=3)
        optimizer = optimizers.Adagrad()
        model.compile(
            optimizer=optimizer,
            loss=losses.MeanSquaredError(),
            metrics=[metrics.MeanSquaredError()],
            run_eagerly=(mode == "eager"),
            jit_compile=False,
        )
        dataset = sparse_generator(generator_type)

        sparse_variable_updates = False

        def mock_optimizer_assign(variable, value):
            nonlocal sparse_variable_updates
            if value.__class__.__name__ == "IndexedSlices":
                sparse_variable_updates = True

        with mock.patch.object(
            optimizer, "assign_sub", autospec=True
        ) as optimizer_assign_sub:
            optimizer_assign_sub.side_effect = mock_optimizer_assign
            model.fit(dataset)

        # JAX does not produce sparse gradients the way we use it.
        if backend.backend() != "jax":
            # Verify tensors did not get densified along the way.
            self.assertTrue(sparse_variable_updates)

    def test_file_url(self):
        """
        File storage returns a url to access a given file from the web.
        """
        self.assertEqual(
            self.storage.url("test.file"), self.storage.base_url + "test.file"
        )

        # should encode special chars except ~!*()'
        # like encodeURIComponent() JavaScript function do
        self.assertEqual(
            self.storage.url(r"~!*()'@#$%^&*abc`+ =.file"),
            "/test_media_url/~!*()'%40%23%24%25%5E%26*abc%60%2B%20%3D.file",
        )
        self.assertEqual(self.storage.url("ab\0c"), "/test_media_url/ab%00c")

        # should translate os path separator(s) to the url path separator
        self.assertEqual(
            self.storage.url("""a/b\\c.file"""), "/test_media_url/a/b/c.file"
        )

        # #25905: remove leading slashes from file names to prevent unsafe url output
        self.assertEqual(self.storage.url("/evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url(r"\evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url("///evil.com"), "/test_media_url/evil.com")
        self.assertEqual(self.storage.url(r"\\\evil.com"), "/test_media_url/evil.com")

        self.assertEqual(self.storage.url(None), "/test_media_url/")

    def calculate_kernel_efficient(data1, data2, func_type, bandwidth):
        diff = np.sqrt(((data1[:, None, :] - data2) ** 2).sum(-1))
        weight_factor = kernel_weight_bandwidth(func_type, len(data2), bandwidth)

        if func_type == "gaussian":
            return (weight_factor * np.exp(-0.5 * (diff * diff) / (bandwidth * bandwidth))).sum(-1)
        elif func_type == "tophat":
            is_within_bandwidth = diff < bandwidth
            return weight_factor * is_within_bandwidth.sum(-1)
        elif func_type == "epanechnikov":
            epanechnikov_kernel = ((1.0 - (diff * diff) / (bandwidth * bandwidth)) * (diff < bandwidth))
            return weight_factor * epanechnikov_kernel.sum(-1)
        elif func_type == "exponential":
            exp_kernel = np.exp(-diff / bandwidth)
            return weight_factor * exp_kernel.sum(-1)
        elif func_type == "linear":
            linear_kernel = ((1 - diff / bandwidth) * (diff < bandwidth))
            return weight_factor * linear_kernel.sum(-1)
        elif func_type == "cosine":
            cosine_kernel = np.cos(0.5 * np.pi * diff / bandwidth) * (diff < bandwidth)
            return weight_factor * cosine_kernel.sum(-1)
        else:
            raise ValueError("kernel type not recognized")

    def kernel_weight_bandwidth(kernel_type, dim, h):
        if kernel_type == "gaussian":
            return 1
        elif kernel_type == "tophat":
            return 0.5
        elif kernel_type == "epanechnikov":
            return (3 / (4 * dim)) ** 0.5
        elif kernel_type == "exponential":
            return 2 / h
        elif kernel_type == "linear":
            return 1 / h
        elif kernel_type == "cosine":
            return np.sqrt(0.5) / h

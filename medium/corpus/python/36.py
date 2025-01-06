# mypy: ignore-errors

import itertools
import random
import unittest
from functools import partial
from itertools import chain, product
from typing import Iterable, List, Tuple

import numpy as np
from numpy import inf

import torch
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import (
    _get_magma_version,
    _get_torch_cuda_version,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import (
    has_cusolver,
    skipCPUIfNoLapack,
    skipCUDAIf,
    skipCUDAIfNoCusolver,
    skipCUDAIfNoMagma,
    skipCUDAIfNoMagmaAndNoCusolver,
    skipCUDAIfNoMagmaAndNoLinalgsolver,
    skipCUDAIfRocm,
    tol,
    toleranceOverride,
)
from torch.testing._internal.common_dtype import (
    all_types_and_complex,
    all_types_and_complex_and,
    floating_and_complex_types,
    floating_and_complex_types_and,
    get_all_complex_dtypes,
)
from torch.testing._internal.common_utils import (
    GRADCHECK_NONDET_TOL,
    IS_MACOS,
    make_fullrank_matrices_with_distinct_singular_values,
    skipIfSlowGradcheckEnv,
    slowTest,
    TEST_WITH_ROCM,
)
from torch.testing._internal.opinfo.core import (
    clone_sample,
    DecorateInfo,
    ErrorInput,
    gradcheck_wrapper_hermitian_input,
    L,
    M,
    OpInfo,
    ReductionOpInfo,
    S,
    SampleInput,
)
from torch.testing._internal.opinfo.refs import PythonRefInfo, ReductionPythonRefInfo


def __initialize__(
        self,
        conv_filters,
        kernel_shape,
        stride=1,
        border_mode="same",
        input_channel_format=None,
        dilation_rate=1,
        act_fn=None,
        bias_flag=True,
        kernel_init="glorot_uniform",
        bias_init="zeros",
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            rank=1,
            filters=conv_filters,
            kernel_size=kernel_shape,
            strides=stride,
            padding=border_mode,
            data_format=input_channel_format,
            dilation_rate=dilation_rate,
            activation=act_fn if act_fn is not None else "linear",
            use_bias=bias_flag,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )


def test_error_inputs_func_safetensors(device, dtype):
    error_inputs = get_test_errors_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        complex_param = torch.rand(2, 3, device=device, dtype=torch.complex64)
        complex_param.grad = torch.rand_like(complex_param)
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(eps=(-1e-30, 1e-3)),
                    desc="epsilon1 should be >= 0",
                ),
                error_type=Exception,
                error_regex="epsilon1 should be >= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(d=0.0),
                    desc="invalid d",
                ),
                error_type=Exception,
                error_regex="Clipping threshold d should be >= 1",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(beta2_decay=0.8),
                    desc="invalid beta2_decay",
                ),
                error_type=Exception,
                error_regex="beta2_decay should be <= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[complex_param],
                    kwargs=dict(),
                    desc="does not support complex parameters",
                ),
                error_type=RuntimeError,
                error_regex="Adafactor does not support complex parameters",
                error_on=OptimizerErrorEnum.STEP_ERROR,
            ),
        ]
    return error_inputs


def matrixSourceLines(ms):
    """Return an iterator over statement lines of a Matrix source file.

    Comment and blank lines are stripped out, and continuation lines are
    merged.
    """
    numberingiter = LineIterator(ms)
    # add an extra '' at the end
    with_extra = itertools.chain(numberingiter, [''])
    pushbackiter = PushbackIterator(with_extra)
    for line in pushbackiter:
        t = lineType(line)
        if t == COMMENT:
            continue
        elif t == STATEMENT:
            lines = [line]
            # this is where we need the extra '', so we don't finish reading
            # the iterator when we don't want to handle that
            for next_line in pushbackiter:
                t = lineType(next_line)
                if t == CONTINUATION:
                    lines.append(next_line[6:])
                else:
                    pushbackiter.pushback(next_line)
                    break
            yield numberingiter.lineno, ''.join(lines)
        else:
            raise ValueError("jammed: continuation line not expected: %s:%d" %
                             (ms.name, numberingiter.lineno))


def tuning_enable(val: bool = True) -> None:
    r"""Enable tuning of TunableOp implementations.

    When enabled, if a tuned entry isn't found, run the tuning step and record
    the entry.
    """
    torch._C._cuda_tunableop_tuning_enable(val)  # type: ignore[attr-defined]


def verify_float_index_to_mixed(data_frame):
        random_generator = np.random.default_rng(2)
        float_column_0 = random_generator.random(10)
        float_column_1 = random_generator.random(10)

        data_frame[0.0] = float_column_0
        data_frame[1.0] = float_column_1
        data_frame["a"] = [10] * 10

        expected_data = {
            0.0: float_column_0,
            1.0: float_column_1,
            "a": [10] * 10
        }
        expected_df = DataFrame(expected_data)
        tm.assert_frame_equal(expected_df, data_frame)


def validate_test_paths(self):
        def identity(x):
            return x

        test_cases = (
            ("integer", {"0", "1", "01", 1234567890}, int),
            ("string", {"abcxyz"}, str),
            ("path", {"allows.ANY*characters"}, lambda s: s),
            ("slug", {"abcxyz-ABCXYZ_01234567890"}, str),
            ("uuid", {"39da9369-838e-4750-91a5-f7805cd82839"}, uuid.UUID),
        )
        for case_name, test_values, converter in test_cases:
            for value in test_values:
                path = "/%s/%s/" % (case_name, value)
                with self.subTest(path=path):
                    resolved = resolve(path)
                    self.assertEqual(resolved.url_name, case_name)
                    self.assertEqual(resolved.kwargs[case_name], converter(value))
                    # reverse() works with string parameters.
                    string_kwargs = {case_name: value}
                    self.assertEqual(reverse(case_name, kwargs=string_kwargs), path)
                    # reverse() also works with native types (int, UUID, etc.).
                    if converter is not identity:
                        converted_val = resolved.kwargs[case_name]
                        conv_path = "/%s/%s/" % (case_name, converted_val)
                        self.assertEqual(reverse(case_name, kwargs={case_name: converted_val}), conv_path)


def sinh_custom(x):
    x_type = x.get_element_type()
    x = get_ov_output(x)
    if not x_type.is_integral():
        ov_type = OPENVINO_DTYPES[config.floatx()]
        x = ov_opset.convert(x, ov_type)
    return OpenVinokerasTensor(ov_opset.sinh(x).output(0))


def example_constructor_fromdatetime(self):
        # GH#30395
        expected_timestamp = Timestamp("2001-02-04 00:00:00")
        expected_stdlib = datetime.fromtimestamp(1048675200)
        result = Datetime.fromdatetime(datetime(2001, 2, 3))
        assert result == expected_timestamp
        assert result == expected_stdlib
        assert isinstance(result, Datetime)


def vsample(self, shape_: _size = torch.Size()) -> torch.Tensor:
        # NOTE: This does not agree with scipy implementation as much as other distributions.
        # (see https://github.com/fritzo/notebooks/blob/master/debug-student-t.ipynb). Using DoubleTensor
        # parameters seems to help.

        #   X ~ Normal(0, 1)
        #   Z ~ Chi2(df)
        #   Y = X / sqrt(Z / df) ~ StudentT(df)
        size = self._extended_shape(shape_)
        X = _standard_normal(size, dtype=self.degrees_of_freedom.dtype, device=self.degrees_of_freedom.device)
        Z = self._chi_squared.rsample(shape_)
        Y = X * torch.rsqrt(Z / self.degrees_of_freedom)
        return self.mean + self.scale * Y


def _calculate_reachability(
    references: List[List[Optional[CacheWeakRefWrapper]]],
) -> List[List[bool]]:
    "Maps references to true if the reference is valid and false otherwise"
    if len(references) == 0:
        return []

    return [pytree.tree_map(is_valid, outputs) for outputs in references]


def example_nonlinear_transform_variable_names():
    A = np.arange(30).reshape(10, 3)
    trans = NonLinearFeatures(degree=2, include_bias=True).fit(A)
    feature_labels = trans.get_feature_labels_out()
    assert_array_equal(
        ["1", "a0", "a1", "a2", "a0^2", "a0 a1", "a0 a2", "a1^2", "a1 a2", "a2^2"],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(degree=3, include_bias=False).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(
        [
            "x",
            "y",
            "z",
            "x^2",
            "x y",
            "x z",
            "y^2",
            "y z",
            "z^2",
            "x^3",
            "x^2 y",
            "x^2 z",
            "x y^2",
            "x y z",
            "x z^2",
            "y^3",
            "y^2 z",
            "y z^2",
            "z^3",
        ],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(degree=(2, 3), include_bias=False).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(
        [
            "x^2",
            "x y",
            "x z",
            "y^2",
            "y z",
            "z^2",
            "x^3",
            "x^2 y",
            "x^2 z",
            "x y^2",
            "x y z",
            "x z^2",
            "y^3",
            "y^2 z",
            "y z^2",
            "z^3",
        ],
        feature_labels,
    )
    assert len(feature_labels) == trans.transform(A).shape[1]

    trans = NonLinearFeatures(
        degree=(3, 3), include_bias=True, interaction_only=True
    ).fit(A)
    feature_labels = trans.get_feature_labels_out(["x", "y", "z"])
    assert_array_equal(["1", "x y z"], feature_labels)
    assert len(feature_labels) == trans.transform(A).shape[1]

    # test some unicode
    trans = NonLinearFeatures(degree=1, include_bias=True).fit(A)
    feature_labels = trans.get_feature_labels_out(["\u0001F40D", "\u262e", "\u05d0"])
    assert_array_equal(["1", "\u0001F40D", "\u262e", "\u05d0"], feature_labels)


def validate_reordered_columns(self, temp_path):
        # GH3454
        chunk_size = 5
        num_rows = int(chunk_size * 2.5)

        data_frame = DataFrame(
            np.ones((num_rows, 3)),
            index=Index([f"i-{i}" for i in range(num_rows)], name="a"),
            columns=Index([f"i-{i}" for i in range(3)], name="columns_name")
        )
        ordered_columns = [data_frame.columns[2], data_frame.columns[0]]
        output_path = str(temp_path)
        data_frame.to_csv(output_path, columns=ordered_columns, chunksize=chunk_size)

        read_data = read_csv(output_path, index_col='a')
        assert_frame_equal(data_frame[ordered_columns], read_data, check_index_type=False)


def check_series_time_mismatch(allowance):
    message = """The Series differ

Values differ by more than the allowed tolerance (100.0 %)
Indices: [0, 1, 2]
[Left]:  [1514764800000000000, 1514851200000000000, 1514937600000000000]
[Right]: [1549065600000000000, 1549152000000000000, 1549238400000000000]"""

    s_a = Series(pd.date_range("2018-01-01", periods=3, freq="D"))
    s_b = Series(pd.date_range("2019-02-02", periods=3, freq="D"))

    if not tm.assert_series_equal(s_a, s_b, rtol=allowance):
        raise AssertionError(message)


def updated(self, elements, **options):
        return type(self)(
            elements,
            user_type=self.user_type,
            user_source=self.user_source,
            **options,
        )


def verify_endpoint_during_session_loading():
    """If request.endpoint (or other URL matching behavior) is needed
    while loading the session, RequestContext.match_request() can be
    called manually.
    """

    class MySessionInterface(SessionInterface):
        def save_session(self, application, user_session, response):
            pass

        def open_session(self, app, http_request):
            if not http_request.endpoint:
                assert False, "Endpoint should not be None"
            request_ctx.match_request()

    application = flask.Flask(__name__)
    application.session_interface = MySessionInterface()

    @application.route("/")
    def homepage():
        return "Hello, Flask!"

    response = application.test_client().get("/")
    assert 200 == response.status_code


def test_not_stylesheet(kml_cta_rail_lines, xml_books):
    lxml_etree = pytest.importorskip("lxml.etree")

    with pytest.raises(
        lxml_etree.XSLTParseError, match=("document is not a stylesheet")
    ):
        read_xml(kml_cta_rail_lines, stylesheet=xml_books)


def __init__(self, l1=0.0, l2=0.0, **kwargs):
    super().__init__(
        activity_regularizer=regularizers.L1L2(l1=l1, l2=l2), **kwargs
    )
    self.supports_masking = True
    self.l1 = l1
    self.l2 = l2
    self.built = True


def test_refit():
    # Regression test for bug in refitting
    # Simulates re-fitting a broken estimator; this used to break with
    # sparse SVMs.
    X = np.arange(100).reshape(10, 10)
    y = np.array([0] * 5 + [1] * 5)

    clf = GridSearchCV(
        BrokenClassifier(), [{"parameter": [0, 1]}], scoring="precision", refit=True
    )
    clf.fit(X, y)


def aot_eager_decomp_partition(gm, fake_tensor_inputs, **kwargs):
    if kwargs:
        log.warning(
            "aot_eager_decomp_partition backend ignoring extra kwargs %s", kwargs
        )

    from torch._inductor.compiler_bisector import CompilerBisector

    config_patches = {"unlift_effect_tokens": True}
    if bisect_changes := CompilerBisector.get_config_change(
        "aot_eager_decomp_partition"
    ):
        config_patches.update(bisect_changes)

    with functorch_config.patch(config_patches):
        return aot_autograd(
            # these are taken from memory_efficient_fusion()
            fw_compiler=get_nop_func(),
            bw_compiler=get_nop_func(),
            # NB: lambda here is to delay import of inductor
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
            partition_fn=functools.partial(
                min_cut_rematerialization_partition, compiler="inductor"
            ),
        )(gm, fake_tensor_inputs)


def check_missing_field(self):
        class InlineValidationTest(TabularInline):
            model = InlineValidationTestModel
            fk_name = "non_existent_field"

        class ModelAdminConfig(ModelAdmin):
            inlines = [InlineValidationTest]

        self.assertIsInvalid(
            ModelAdminConfig,
            ValidationTestModel,
            "'modeladmin.InlineValidationTestModel' has no field named "
            "'non_existent_field'.",
            "admin.E202",
            invalid_obj=InlineValidationTest,
        )


def _infer_tz_from_endpoints(
    start: Timestamp, end: Timestamp, tz: tzinfo | None
) -> tzinfo | None:
    """
    If a timezone is not explicitly given via `tz`, see if one can
    be inferred from the `start` and `end` endpoints.  If more than one
    of these inputs provides a timezone, require that they all agree.

    Parameters
    ----------
    start : Timestamp
    end : Timestamp
    tz : tzinfo or None

    Returns
    -------
    tz : tzinfo or None

    Raises
    ------
    TypeError : if start and end timezones do not agree
    """
    try:
        inferred_tz = timezones.infer_tzinfo(start, end)
    except AssertionError as err:
        # infer_tzinfo raises AssertionError if passed mismatched timezones
        raise TypeError(
            "Start and end cannot both be tz-aware with different timezones"
        ) from err

    inferred_tz = timezones.maybe_get_tz(inferred_tz)
    tz = timezones.maybe_get_tz(tz)

    if tz is not None and inferred_tz is not None:
        if not timezones.tz_compare(inferred_tz, tz):
            raise AssertionError("Inferred time zone not equal to passed time zone")

    elif inferred_tz is not None:
        tz = inferred_tz

    return tz


def test_draw_forest_gini(matplotlib):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = gini
    arb = RandomForestClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    arb.fit(X, y)

    # Test export code
    label_names = ["left leaf", "right leaf"]
    branches = draw_tree(arb, label_names=label_names)
    assert len(branches) == 5
    assert (
        branches[0].get_text()
        == "left leaf <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert branches[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert branches[2].get_text() == "True  "
    assert branches[3].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert branches[4].get_text() == "  False"


def verify_redirect_description(self, response):
        expected_representation = (
            '<HttpResponseRedirect status_code=302, "text/html; charset=utf-8", '
            'url="/redirected/">'
        )
        self.assertEqual(repr(response), expected_representation)


def check_repeated_index_notification():
    indices = pd.Index(["x", "y", "x", "y", "z"])
    output = indices._generate_duplicate_warning()
    expected_frame = pd.DataFrame(
        {"locations": [[0, 2], [1, 3]]}, index=pd.Index(["x", "y"], name="label")
    )
    tm.assert_frame_equal(output, expected_frame)


def compute_silhouette(
    features, cluster_labels, *, distance_metric="euclidean", sample_subset=None, seed=None, **kwargs
):
    """Compute the average Silhouette Coefficient of all samples.

    The Silhouette Coefficient is calculated using the mean intra-cluster
    distance (``a``) and the mean nearest-cluster distance (``b``) for each
    sample.  The Silhouette Coefficient for a sample is ``(b - a) / max(a,
    b)``.  To clarify, ``b`` is the distance between a sample and the nearest
    cluster that the sample is not a part of.
    Note that Silhouette Coefficient is only defined if number of labels
    is ``2 <= n_labels <= n_samples - 1``.

    This function returns the average Silhouette Coefficient over all samples.
    To obtain the values for each sample, use :func:`compute_silhouette_samples`.

    The best value is 1 and the worst value is -1. Values near 0 indicate
    overlapping clusters. Negative values generally indicate that a sample has
    been assigned to the wrong cluster, as a different cluster is more similar.

    Read more in the :ref:`User Guide <silhouette_coefficient>`.

    Parameters
    ----------
    features : {array-like, sparse matrix} of shape (n_samples_a, n_samples_a) if metric == \
            "precomputed" or (n_samples_a, n_features) otherwise
        An array of pairwise distances between samples, or a feature array.

    cluster_labels : array-like of shape (n_samples,)
        Predicted labels for each sample.

    distance_metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by :func:`~sklearn.metrics.pairwise_distances`. If ``features`` is
        the distance array itself, use ``distance_metric="precomputed"``.

    sample_subset : int, default=None
        The size of the subset to use when computing the Silhouette Coefficient
        on a random selection of data.
        If ``sample_subset is None``, no sampling is used.

    seed : int, RandomState instance or None, default=None
        Determines random number generation for selecting a subset of samples.
        Used when ``sample_subset is not None``.
        Pass an integer for reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    **kwargs : optional keyword parameters
        Any further parameters are passed directly to the distance function.
        If using a scipy.spatial.distance metric, the parameters are still
        metric dependent. See the scipy docs for usage examples.

    Returns
    -------
    silhouette_score : float
        Average Silhouette Coefficient for all samples.

    References
    ----------

    .. [1] `Peter J. Rousseeuw (1987). "Silhouettes: a Graphical Aid to the
       Interpretation and Validation of Cluster Analysis". Computational
       and Applied Mathematics 20: 53-65.
       <https://www.sciencedirect.com/science/article/pii/0377042787901257>`_

    .. [2] `Wikipedia entry on the Silhouette Coefficient
           <https://en.wikipedia.org/wiki/Silhouette_(clustering)>`_

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.cluster import KMeans
    >>> from sklearn.metrics import compute_silhouette
    >>> X, y = make_blobs(random_state=42)
    >>> kmeans = KMeans(n_clusters=2, random_state=42)
    >>> silhouette_score = compute_silhouette(X, kmeans.fit_predict(X))
    0.49...
    """
    if sample_subset is not None:
        features, cluster_labels = check_X_y(features, cluster_labels, accept_sparse=["csc", "csr"])
        seed = check_random_state(seed)
        indices = seed.permutation(features.shape[0])[:sample_subset]
        if distance_metric == "precomputed":
            features, cluster_labels = features[indices].T[indices].T, cluster_labels[indices]
        else:
            features, cluster_labels = features[indices], cluster_labels[indices]
    return np.mean(compute_silhouette_samples(features, cluster_labels, metric=distance_metric, **kwargs))


def read_csv(self, path, **kwargs):
    params = {"index_col": 0, "header": None}
    params.update(**kwargs)

    header = params.get("header")
    out = pd.read_csv(path, **params).squeeze("columns")

    if header is None:
        out.name = out.index.name = None

    return out


def test_where():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(cond).dropna()
    rs2 = s[cond]
    tm.assert_series_equal(rs, rs2)

    rs = s.where(cond, -s)
    tm.assert_series_equal(rs, s.abs())

    rs = s.where(cond)
    assert s.shape == rs.shape
    assert rs is not s

    # test alignment
    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())

    expected = s2[cond].reindex(s2.index[:3]).reindex(s2.index)
    rs = s2.where(cond[:3])
    tm.assert_series_equal(rs, expected)

    expected = s2.abs()
    expected.iloc[0] = s2[0]
    rs = s2.where(cond[:3], -s2)
    tm.assert_series_equal(rs, expected)


def verify_custom_product_sk_not_named_id(self):
        """
        {% get_admin_log %} works if the product model's primary key isn't named
        'id'.
        """
        context = Context(
            {
                "user": CustomIdSeller(),
                "log_entries": LogEntry.objects.all(),
            }
        )
        template = Template(
            "{% load log %}{% get_admin_log 10 as admin_log for_user user %}"
        )
        # This template tag just logs.
        self.assertEqual(template.render(context), "")


def check_complex_hierarchy(self):
        X = self.generate_model("X", abstract=True)
        Y = self.generate_model("Y")
        W = self.generate_model("W")
        V = self.generate_model("V", bases=(W,), proxy=True)
        U = self.generate_model("U", bases=(X, Y, V))
        # Y has a pointer O2O field p_ptr to W
        self.assertRelation(X, [Y, W, V, U])
        self.assertRelation(Y, [W, V, U])
        self.assertRelation(W, [Y, V, U])
        self.assertRelation(V, [Y, W, U])
        self.assertRelation(U, [Y, W, V])


def __fetch__(self, entity, cls=None):
        """
        Retrieve and caches the value from the datastore on the first lookup.
        Return the cached value.
        """
        if entity is None:
            return self
        info = entity.__dict__
        attr_name = self.attribute.name
        if attr_name not in info:
            # Let's see if the attribute is part of the parent chain. If so we
            # might be able to reuse the already loaded value. Refs #18343.
            val = self._check_parent_path(entity)
            if val is None:
                if not entity._has_valid_pk() and self.attribute.generated:
                    raise AttributeError(
                        "Cannot read a generated attribute from an unsaved entity."
                    )
                entity.reload(fields=[attr_name])
            else:
                info[attr_name] = val
        return info[attr_name]


def init_model(
        self,
        *,
        metric_func: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        threshold: float = 0.5,
        reverse: bool = False,
        aggregation: str = "sum",
    ):
        super().__init__(size_average=None, reduce=None, reduction=aggregation)
        if threshold <= 0:
            raise ValueError(
                f"CustomLossFunction: expected threshold to be greater than 0, got {threshold} instead"
            )
        self.metric_func: Optional[Callable[[Tensor, Tensor], Tensor]] = (
            metric_func if metric_func is not None else PairwiseEuclidean()
        )
        self.threshold = threshold
        self.reverse = reverse


def test1_basic(self, datapath):
    # Tests with DEMO_G.xpt (all numeric file)

    # Compare to this
    file01 = datapath("io", "sas", "data", "DEMO_G.xpt")
    data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
    numeric_as_float(data_csv)

    # Read full file
    data = read_sas(file01, format="xport")
    tm.assert_frame_equal(data, data_csv)
    num_rows = data.shape[0]

    # Test reading beyond end of file
    with read_sas(file01, format="xport", iterator=True) as reader:
        data = reader.read(num_rows + 100)
    assert data.shape[0] == num_rows

    # Test incremental read with `read` method.
    with read_sas(file01, format="xport", iterator=True) as reader:
        data = reader.read(10)
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

    # Test incremental read with `get_chunk` method.
    with read_sas(file01, format="xport", chunksize=10) as reader:
        data = reader.get_chunk()
    tm.assert_frame_equal(data, data_csv.iloc[0:10, :])

    # Test read in loop
    m = 0
    with read_sas(file01, format="xport", chunksize=100) as reader:
        for x in reader:
            m += x.shape[0]
    assert m == num_rows

    # Read full file with `read_sas` method
    data = read_sas(file01)
    tm.assert_frame_equal(data, data_csv)


def check_new_popup_template_response_on_update(self):
        actor_instance = Actor.objects.create(full_name="John Doe", age=30)
        response = self.client.post(
            reverse("admin:custom_actor_change", args=(actor_instance.pk,))
            + "?%s=1" % NEW_POPUP_VAR,
            {"full_name": "John Doe", "age": "32", NEW_POPUP_VAR: "1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.template_name,
            [
                "admin/custom/actor/popup_response.html",
                "admin/popup_response.html",
                "custom_popup_response.html",
            ],
        )
        self.assertTemplateUsed(response, "custom_popup_response.html")


def handle_form_request(req):
    "A view that handles form submissions"
    if req.method == 'POST':
        data = req.POST.copy()
        form = TestForm(data)
        template_name = "Valid POST Template" if form.is_valid() else "Invalid POST Template"
        context_data = {"form": form} if not form.is_valid() else {}
    else:
        form = TestForm(req.GET)
        template_name = "Form GET Template"
        context_data = {"form": form}

    template = Template("Valid POST data.", name=template_name) if form.is_valid() else Template(
        "Invalid POST data. {{ form.errors }}", name="Invalid POST Template")
    context = Context(context_data)

    return HttpResponse(template.render(context))


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


def test_count_with_only_nans_in_first_group(self):
    # GH21956
    df = DataFrame({"A": [np.nan, np.nan], "B": ["a", "b"], "C": [1, 2]})
    result = df.groupby(["A", "B"]).C.count()
    mi = MultiIndex(levels=[[], ["a", "b"]], codes=[[], []], names=["A", "B"])
    expected = Series([], index=mi, dtype=np.int64, name="C")
    tm.assert_series_equal(result, expected, check_index_type=False)


def enable_terminal_echo():
    """
    Ensure that echo mode is enabled. Some tools such as PDB disable
    it which causes usability issues after reload.
    """
    if termios and sys.stdin.isatty():
        current_attrs = list(termios.tcgetattr(sys.stdin))
        if not (current_attrs[3] & termios.ECHO):
            old_handler = None if not hasattr(signal, "SIGTTOU") else signal.signal(signal.SIGTTOU, signal.SIG_IGN)
            try:
                termios.tcsetattr(sys.stdin, termios.TCSANOW, current_attrs | [termios.ECHO])
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGTTOU, old_handler)


def test_process_strings(input_series):
    split_result = input_series.str.split("_")
    intermediate_results = split_result.str.get(1)
    expected_values = [val if isinstance(val, str) and "_" in val else np.nan for val in input_series]
    result = intermediate_results.fillna(expected_values).astype(object)
    pandas.testing.assert_series_equal(result, intermediate_results)


def matrix_distance_calculation(
    ctx: jit_utils.GraphContext,
    obj,
    norm_type,
    axis: Sequence[int] | None,
    keepdim: bool,
    output_dtype,
):
    return symbolic_helper._linalg_vector_norm_helper(ctx, obj, norm_type, axis, keepdim, output_dtype)


def verify_field_output_check(self):
        class Entity(models.Model):
            amount = models.DecimalField(max_digits=5, decimal_places=2)
            result = models.GeneratedField(
                expression=models.F("amount") * 2,
                output_field=models.DecimalField(max_digits=-1, decimal_places=-1),
                db_persist=True
            )

        expected_warnings = [
            Error(
                message="GeneratedField.output_field has errors:"
                "\n    'decimal_places' must be a non-negative integer. (fields.E131)"
                "\n    'max_digits' must be a positive integer. (fields.E133)",
                obj=Entity._meta.get_field("result"),
                id="fields.E223"
            )
        ]
        self.assertListEqual(
            list(Model._meta.get_field("field").check(databases={"default"})),
            expected_warnings
        )


op_db: List[OpInfo] = [
    OpInfo(
        "linalg.cross",
        ref=lambda x, y, dim=-1: np.cross(x, y, axis=dim),
        op=torch.linalg.cross,
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        aten_name="linalg_cross",
        sample_inputs_func=sample_inputs_cross,
        error_inputs_func=error_inputs_cross,
        supports_out=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        check_batched_gradgrad=False,
    ),
    OpInfo(
        "linalg.det",
        aten_name="linalg_det",
        op=torch.linalg.det,
        variant_test_name="singular",
        aliases=("det",),
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_det_singular,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("The backward may give different results"),
                "TestCommon",
                "test_noncontiguous_samples",
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            DecorateInfo(
                unittest.skip("Gradients are incorrect on macos"),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cpu",
                dtypes=(torch.float64,),
                active_if=IS_MACOS,
            ),
            # Both Hessians are incorrect on complex inputs??
            DecorateInfo(
                unittest.expectedFailure,
                "TestBwdGradients",
                "test_fn_gradgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                dtypes=(torch.complex128,),
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip("Skipped, see https://github.com//issues/84192"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93044"
                ),
                "TestBwdGradients",
                "test_fn_grad",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip(
                    "Flaky on ROCm https://github.com/pytorch/pytorch/issues/93045"
                ),
                "TestFwdGradients",
                "test_forward_mode_AD",
                device_type="cuda",
                dtypes=get_all_complex_dtypes(),
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.diagonal",
        aten_name="linalg_diagonal",
        aten_backward_name="diagonal_backward",
        dtypes=all_types_and_complex_and(
            torch.bool, torch.bfloat16, torch.float16, torch.chalf
        ),
        supports_out=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_diagonal_diag_embed,
        error_inputs_func=error_inputs_diagonal_diag_embed,
    ),
    OpInfo(
        "linalg.cholesky",
        aten_name="linalg_cholesky",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.cholesky_ex",
        aten_name="linalg_cholesky_ex",
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_cholesky,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vecdot",
        aten_name="linalg_vecdot",
        ref=lambda x, y, *, dim=-1: (x.conj() * y).sum(dim),
        dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16),
        sample_inputs_func=sample_inputs_linalg_vecdot,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # Issue with conj and torch dispatch, see https://github.com/pytorch/pytorch/issues/82479
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestSchemaCheckModeOpInfo",
                "test_schema_correctness",
                dtypes=(torch.complex64, torch.complex128),
            ),
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
            DecorateInfo(
                toleranceOverride({torch.half: tol(atol=1.2e-2, rtol=1.7e-2)}),
                "TestInductorOpInfo",
                "test_comprehensive",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.cond",
        aten_name="linalg_cond",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_cond,
        check_batched_gradgrad=False,
        check_batched_forward_grad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.eig",
        aten_name="linalg_eig",
        op=torch.linalg.eig,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eig,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        skips=(
            # AssertionError: Scalars are not equal!
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_out", device_type="cpu"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
    ),
    OpInfo(
        "linalg.eigvals",
        aten_name="linalg_eigvals",
        op=torch.linalg.eigvals,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigh",
        aten_name="linalg_eigh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.eigvalsh",
        aten_name="linalg_eigvalsh",
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_eigh,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        check_batched_forward_grad=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # Pre-existing condition; Needs to be fixed
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.householder_product",
        aten_name="linalg_householder_product",
        op=torch.linalg.householder_product,
        aliases=("orgqr",),
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        # TODO: backward uses in-place operations that vmap doesn't like
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_householder_product,
        decorators=[
            skipCUDAIfNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.complex64: tol(atol=1e-3, rtol=1e-3)})
            ),
            DecorateInfo(
                unittest.skip("Skipped! Flaky"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cpu",
                dtypes=(torch.complex128,),
            ),
        ],
    ),
    OpInfo(
        "linalg.ldl_factor",
        aten_name="linalg_ldl_factor",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.ldl_factor_ex",
        aten_name="linalg_ldl_factor_ex",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_factor,
        decorators=[skipCUDAIfNoMagmaAndNoLinalgsolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.ldl_solve",
        aten_name="linalg_ldl_solve",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_ldl_solve,
        decorators=[
            skipCUDAIf(
                _get_torch_cuda_version() < (11, 4), "not available before CUDA 11.3.1"
            ),
            skipCUDAIfNoCusolver,
            skipCUDAIfRocm,
            skipCPUIfNoLapack,
        ],
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        dtypes=floating_and_complex_types(),
        supports_out=True,
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # we skip gradient checks for this suite as they are tested in
            # variant_test_name='grad_oriented'
            DecorateInfo(unittest.skip("Skipped!"), "TestFwdGradients"),
            DecorateInfo(unittest.skip("Skipped!"), "TestBwdGradients"),
            # The values for attribute 'shape' do not match
            DecorateInfo(unittest.skip("Skipped!"), "TestCommon", "test_out"),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.lstsq",
        aten_name="linalg_lstsq",
        variant_test_name="grad_oriented",
        # gradchecks for forward AD fails with multi-Tensor outputs
        op=lambda a, b, driver: torch.linalg.lstsq(a, b, driver=driver)[0],
        supports_out=False,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_lstsq,
        error_inputs_func=error_inputs_lstsq_grad_oriented,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_autograd=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            # tests do not work with passing lambda for op
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            DecorateInfo(
                unittest.expectedFailure,
                "TestOperatorSignatures",
                "test_get_torch_func_signature_exhaustive",
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_power",
        aliases=("matrix_power",),
        aten_name="linalg_matrix_power",
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_inplace_autograd=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        check_batched_grad=False,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_power,
    ),
    OpInfo(
        "linalg.multi_dot",
        # Need this lambda because gradcheck does not work with TensorList inputs
        aten_name="linalg_multi_dot",
        dtypes=all_types_and_complex_and(torch.half, torch.bfloat16),
        dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16),
        supports_inplace_autograd=False,
        # Batched grad checks fail for empty input tensors (see https://github.com/pytorch/pytorch/issues/53407)
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # https://github.com/pytorch/pytorch/issues/66357
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_multi_dot,
        gradcheck_nondet_tol=GRADCHECK_NONDET_TOL,
        skips=(
            # https://github.com/pytorch/pytorch/issues/67470
            DecorateInfo(
                unittest.skip("67470!"), "TestCommon", "test_noncontiguous_samples"
            ),
            # Fails on XLA.
            # AssertionError: False is not true : Tensors failed to compare as equal!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestOpInfo",
                device_type="xla",
                dtypes=(torch.long,),
            ),
            # https://github.com/pytorch/pytorch/issues/71774
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestNNCOpInfo",
                "test_nnc_correctness",
                device_type="cpu",
                dtypes=(torch.long,),
            ),
        ),
    ),
    # NB: linalg.norm has two variants so that different skips can be used for different sample inputs
    OpInfo(
        "linalg.norm",
        aten_name="linalg_norm",
        op=torch.linalg.norm,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_norm,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.norm",
        op=torch.linalg.norm,
        variant_test_name="subgradients_at_zero",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=partial(
            sample_inputs_linalg_norm, variant="subgradient_at_zero"
        ),
        aten_name="linalg_norm",
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients, got:
        # Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        skips=(
            # [NEW] Skips specifically for sample inputs at zero
            # norm's vjp/jvp are not well-conditioned near zero
            DecorateInfo(
                unittest.expectedFailure, "TestBwdGradients", "test_fn_gradgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_fn_fwgrad_bwgrad"
            ),
            DecorateInfo(
                unittest.expectedFailure, "TestFwdGradients", "test_forward_mode_AD"
            ),
            DecorateInfo(unittest.expectedFailure, "TestBwdGradients", "test_fn_grad"),
        ),
    ),
    OpInfo(
        "linalg.matrix_norm",
        aten_name="linalg_matrix_norm",
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        check_batched_gradgrad=False,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        sample_inputs_func=sample_inputs_linalg_matrix_norm,
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.qr",
        aten_name="linalg_qr",
        op=torch.linalg.qr,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # In-place ops
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_qr_geqrf,
        decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.slogdet",
        aten_name="linalg_slogdet",
        op=torch.linalg.slogdet,
        dtypes=floating_and_complex_types(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
    ),
    OpInfo(
        "linalg.vander",
        aten_name="linalg_vander",
        ref=np_vander_batched,
        op=torch.linalg.vander,
        dtypes=all_types_and_complex(),
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        supports_out=False,
        sample_inputs_func=sample_inputs_linalg_vander,
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    ReductionOpInfo(
        "linalg.vector_norm",
        op=torch.linalg.vector_norm,
        identity=0,
        nan_policy="propagate",
        supports_multiple_dims=True,
        complex_to_real=True,
        supports_forward_ad=True,
        # torch.autograd.gradcheck.GradcheckError: While computing batched gradients
        # got: Could not allocate memory to change Tensor SizesAndStrides!
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16),
        generate_args_kwargs=sample_kwargs_vector_norm,
        aten_name="linalg_vector_norm",
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),
    OpInfo(
        "linalg.lu_factor",
        aten_name="linalg_lu_factor",
        op=torch.linalg.lu_factor,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_factor_ex",
        aten_name="linalg_lu_factor_ex",
        op=torch.linalg.lu_factor_ex,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu",
        aten_name="linalg_lu",
        op=torch.linalg.lu,
        dtypes=floating_and_complex_types(),
        # https://github.com/pytorch/pytorch/issues/80411
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_lu,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # linalg.lu_factor: LU without pivoting is not implemented on the CPU
            DecorateInfo(unittest.expectedFailure, "TestCommon", "test_compare_cpu"),
        ),
    ),
    OpInfo(
        "linalg.lu_solve",
        op=torch.linalg.lu_solve,
        aten_name="linalg_lu_solve",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_lu_solve,
        skips=(
            DecorateInfo(
                unittest.skip("Tests different backward paths"),
                "TestCommon",
                "test_floating_inputs_are_differentiable",
            ),
        ),
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
    ),
    OpInfo(
        "linalg.inv",
        aten_name="linalg_inv",
        op=torch.linalg.inv,
        aliases=("inverse",),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.inv_ex",
        aten_name="linalg_inv_ex",
        op=torch.linalg.inv_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_invertible,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve",
        aten_name="linalg_solve",
        op=torch.linalg.solve,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=6e-04)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_ex",
        aten_name="linalg_solve_ex",
        op=torch.linalg.solve_ex,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=6e-04)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.solve_triangular",
        aten_name="linalg_solve_triangular",
        op=torch.linalg.solve_triangular,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_linalg_solve_triangular,
        supports_fwgrad_bwgrad=True,
        skips=(skipCPUIfNoLapack,),
        # linalg.solve_triangular cannot be batched over because of a call to out.copy_(result);
        supports_forward_ad=True,
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_matrix_rank,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            # jit doesn't accept tensor inputs for matrix rank
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                dtypes=[torch.complex64, torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.matrix_rank",
        aten_name="linalg_matrix_rank",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        supports_autograd=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        op=torch.linalg.pinv,
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack],
        skips=(
            # errors with "leaked XXXX bytes CUDA memory on device 0"
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="singular",
        # pinv is Frechet-differentiable in a rank-preserving neighborhood,
        # so we feed inputs that are the products of two full-rank factors,
        # to avoid any rank changes caused by the perturbations in the gradcheck
        op=lambda a, b: torch.linalg.pinv(a @ b.mT),
        dtypes=floating_and_complex_types(),
        supports_out=False,
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        sample_inputs_func=sample_inputs_linalg_pinv_singular,
        # Only large tensors show issues with implicit backward used prior to
        # explicit backward implementation.
        decorators=[slowTest, skipCUDAIfNoCusolver, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.expectedFailure, "TestJit", "test_variant_consistency_jit"
            ),
            # CUDA runs out of memory
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
            # This test takes almost 2 hours to run!
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestBwdGradients",
                "test_fn_gradgrad",
                device_type="cuda",
                dtypes=[torch.cdouble],
            ),
        ),
    ),
    OpInfo(
        "linalg.pinv",
        aten_name="linalg_pinv",
        variant_test_name="hermitian",
        dtypes=floating_and_complex_types(),
        check_batched_grad=False,
        check_batched_gradgrad=False,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        sample_inputs_func=sample_inputs_linalg_pinv_hermitian,
        gradcheck_wrapper=gradcheck_wrapper_hermitian_input,
        decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-5, rtol=1e-5)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
            # This test is flaky under slow gradcheck, likely due to rounding issues
            DecorateInfo(
                skipIfSlowGradcheckEnv,
                "TestFwdGradients",
                "test_fn_fwgrad_bwgrad",
                device_type="cuda",
            ),
        ),
    ),
    OpInfo(
        "linalg.svd",
        op=torch.linalg.svd,
        aten_name="linalg_svd",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        # Runs very slowly on slow-gradcheck - alternatively reduce input sizes
        gradcheck_fast_mode=True,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        check_batched_forward_grad=False,
        # We're using at::allclose, which does not have a batching rule
        check_batched_grad=False,
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_svd,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_out",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestCommon",
                "test_variant_consistency_eager",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestJit",
                "test_variant_consistency_jit",
                device_type="mps",
                dtypes=[torch.float32],
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.svdvals",
        op=torch.linalg.svdvals,
        aten_name="linalg_svdvals",
        decomp_aten_name="_linalg_svd",
        dtypes=floating_and_complex_types(),
        check_batched_forward_grad=False,
        supports_fwgrad_bwgrad=True,
        supports_forward_ad=True,
        # We're using at::allclose, which does not have a batching rule
        check_batched_gradgrad=False,
        sample_inputs_func=sample_inputs_linalg_svdvals,
        decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off],
        skips=(
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
            DecorateInfo(
                unittest.skip("Skipped!"),
                "TestFakeTensor",
                "test_fake_crossref_backward_no_amp",
                device_type="cuda",
                dtypes=[torch.float32],
                active_if=TEST_WITH_ROCM,
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorinv",
        ref=np.linalg.tensorinv,
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorinv,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        # See https://github.com/pytorch/pytorch/pull/78358
        check_batched_forward_grad=False,
        decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
    OpInfo(
        "linalg.tensorsolve",
        ref=lambda a, b, dims=None: np.linalg.tensorsolve(a, b, axes=dims),
        dtypes=floating_and_complex_types(),
        sample_inputs_func=sample_inputs_tensorsolve,
        supports_forward_ad=True,
        supports_fwgrad_bwgrad=True,
        decorators=[
            skipCUDAIfNoMagmaAndNoCusolver,
            skipCPUIfNoLapack,
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=1e-03, rtol=1e-03)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cuda",
            ),
            DecorateInfo(
                toleranceOverride({torch.float32: tol(atol=8e-04, rtol=7e-06)}),
                "TestCommon",
                "test_noncontiguous_samples",
                device_type="cpu",
            ),
        ],
        skips=(
            DecorateInfo(
                unittest.skip("Unsupported on MPS for now"),
                "TestCommon",
                "test_numpy_ref_mps",
            ),
        ),
    ),
]

python_ref_db: List[OpInfo] = [
    #
    # torch.linalg
    #
    PythonRefInfo(
        "_refs.linalg.cross",
        torch_opinfo_name="linalg.cross",
        supports_out=True,
        op_db=op_db,
        skips=(
            # TODO: is this really needed?
            DecorateInfo(
                unittest.expectedFailure, "TestCommon", "test_python_ref_errors"
            ),
        ),
    ),
    PythonRefInfo(
        "_refs.linalg.diagonal",
        torch_opinfo_name="linalg.diagonal",
        supports_out=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.vecdot",
        torch_opinfo_name="linalg.vecdot",
        op_db=op_db,
    ),
    ReductionPythonRefInfo(
        "_refs.linalg.vector_norm",
        torch_opinfo_name="linalg.vector_norm",
        supports_out=True,
        op_db=op_db,
        skips=(
            # FIXME: sum reduces all dimensions when dim=[]
            DecorateInfo(unittest.expectedFailure, "TestReductions", "test_dim_empty"),
            DecorateInfo(
                unittest.expectedFailure, "TestReductions", "test_dim_empty_keepdim"
            ),
        ),
    ),
    PythonRefInfo(
        "_refs.linalg.matrix_norm",
        torch_opinfo_name="linalg.matrix_norm",
        supports_out=True,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.norm",
        torch_opinfo_name="linalg.norm",
        supports_out=True,
        # Uses vector_norm inside and vector_norm is affected by
        # https://github.com/pytorch/pytorch/issues/77216
        validate_view_consistency=False,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svd",
        torch_opinfo_name="linalg.svd",
        supports_out=True,
        op_db=op_db,
    ),
    PythonRefInfo(
        "_refs.linalg.svdvals",
        torch_opinfo_name="linalg.svdvals",
        supports_out=True,
        op_db=op_db,
    ),
]

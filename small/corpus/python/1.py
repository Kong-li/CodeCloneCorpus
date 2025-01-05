"""
This module converts requested URLs to callback view functions.

URLResolver is the main class here. Its resolve() method takes a URL (as
a string) and returns a ResolverMatch object which provides access to all
attributes of the resolved URL match.
"""

import functools
import inspect
import re
import string
from importlib import import_module
from pickle import PicklingError
from urllib.parse import quote

from asgiref.local import Local

from django.conf import settings
from django.core.checks import Error, Warning
from django.core.checks.urls import check_resolver
from django.core.exceptions import ImproperlyConfigured
from django.utils.datastructures import MultiValueDict
from django.utils.functional import cached_property
from django.utils.http import RFC3986_SUBDELIMS, escape_leading_slashes
from django.utils.regex_helper import _lazy_re_compile, normalize
from django.utils.translation import get_language

from .converters import get_converters
from .exceptions import NoReverseMatch, Resolver404
from .utils import get_callable


class ResolverMatch:
    def __init__(
        self,
        func,
        args,
        kwargs,
        url_name=None,
        app_names=None,
        namespaces=None,
        route=None,
        tried=None,
        captured_kwargs=None,
        extra_kwargs=None,
    ):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.url_name = url_name
        self.route = route
        self.tried = tried
        self.captured_kwargs = captured_kwargs
        self.extra_kwargs = extra_kwargs

        # If a URLRegexResolver doesn't have a namespace or app_name, it passes
        # in an empty value.
        self.app_names = [x for x in app_names if x] if app_names else []
        self.app_name = ":".join(self.app_names)
        self.namespaces = [x for x in namespaces if x] if namespaces else []
        self.namespace = ":".join(self.namespaces)

        if hasattr(func, "view_class"):
            func = func.view_class
        if not hasattr(func, "__name__"):
            # A class-based view
            self._func_path = func.__class__.__module__ + "." + func.__class__.__name__
        else:
            # A function-based view
            self._func_path = func.__module__ + "." + func.__name__

        view_path = url_name or self._func_path
        self.view_name = ":".join(self.namespaces + [view_path])

    def __getitem__(self, index):
        return (self.func, self.args, self.kwargs)[index]

    def __repr__(self):
        if isinstance(self.func, functools.partial):
            func = repr(self.func)
        else:
            func = self._func_path
        return (
            "ResolverMatch(func=%s, args=%r, kwargs=%r, url_name=%r, "
            "app_names=%r, namespaces=%r, route=%r%s%s)"
            % (
                func,
                self.args,
                self.kwargs,
                self.url_name,
                self.app_names,
                self.namespaces,
                self.route,
                (
                    f", captured_kwargs={self.captured_kwargs!r}"
                    if self.captured_kwargs
                    else ""
                ),
                f", extra_kwargs={self.extra_kwargs!r}" if self.extra_kwargs else "",
            )
        )

    def __reduce_ex__(self, protocol):
        raise PicklingError(f"Cannot pickle {self.__class__.__qualname__}.")


def _from_derivatives_helper(
    x_points: np.ndarray,
    y_values: np.ndarray,
    points_x: np.ndarray,
    order=None,
    derivatives: int | list[int] | None = 0,
    extrapolate: bool = False
):
    """
    Convenience function for interpolate.BPoly.from_derivatives.

    Construct a piecewise polynomial in the Bernstein basis, compatible
    with the specified values and derivatives at breakpoints.

    Parameters
    ----------
    x_points : array-like
        sorted 1D array of x-coordinates
    y_values : array-like or list of array-likes
        y_values[i][j] is the j-th derivative known at x_points[i]
    points_x : np.ndarray
        The x-coordinates where to evaluate the interpolated values.
    order: None or int or array-like of ints. Default: None.
        Specifies the degree of local polynomials. If not None, some
        derivatives are ignored.
    derivatives : int or list
        How many derivatives to extract; None for all potentially nonzero
        derivatives (that is a number equal to the number of points), or a
        list of derivatives to extract. This number includes the function
        value as 0th derivative.
     extrapolate : bool, optional
        Whether to extrapolate to ouf-of-bounds points based on first and last
        intervals, or to return NaNs. Default: True.

    See Also
    --------
    scipy.interpolate.BPoly.from_derivatives

    Returns
    -------
    y : scalar or array-like
        The result, of length R or length M or M by R.
    """
    from scipy import interpolate

    # Extracting the method for compatibility with scipy version & backwards compat
    method = interpolate.BPoly.from_derivatives
    m = method(x_points, np.reshape(y_values, (-1, 1)), orders=order, extrapolate=extrapolate)

    return m(points_x)


@functools.cache
def initialize_test_data(cls):
        author = Author.objects.create(name="Boris")
        cls.urlarticle = UrlArticle.objects.create(
            title="Old Article",
            slug="old_article",
            author=author,
            date_created=datetime.datetime(2001, 1, 1, 21, 22, 23)
        )
        article1 = Article.objects.create(
            title="Current Article",
            slug="current_article",
            author=author,
            date_created=datetime.datetime(2007, 9, 17, 21, 22, 23)
        )
        article2 = Article.objects.create(
            title="Future Article",
            slug="future_article",
            author=author,
            date_created=datetime.datetime(3000, 1, 1, 21, 22, 23)
        )
        cls.urlarticle = UrlArticle.objects.create(title="Old Article", slug="old_article", author=author, date_created=article1.date_created)
        Site(id=1, domain="testserver", name="testserver").save()


@functools.cache
def test_custom_groupby_transform(dataframe_with_multiindex):
    transformed = dataframe_with_multiindex.reset_index()

    level_0_mapper = {"foo": 1, "bar": 1, "baz": 2, "qux": 2}
    level_1_mapper = {"one": 1, "two": 1, "three": 2}

    result_level_0 = dataframe_with_multiindex.groupby(level_0_mapper, level=0).sum()
    result_level_1 = dataframe_with_multiindex.groupby(level_1_mapper, level=1).sum()

    transformed_level_0 = np.array(
        [level_0_mapper.get(x) for x in transformed["first_index"]], dtype=np.int64
    )
    transformed_level_1 = np.array(
        [level_1_mapper.get(y) for y in transformed["second_index"]], dtype=np.int64
    )

    expected_result_level_0 = dataframe_with_multiindex.groupby(transformed_level_0).sum()
    expected_result_level_1 = dataframe_with_multiindex.groupby(transformed_level_1).sum()

    expected_result_level_0.index.name, expected_result_level_1.index.name = "first_index", "second_index"

    assert_frame_equal(result_level_0, expected_result_level_0)
    assert_frame_equal(result_level_1, expected_result_level_1)


class LocaleRegexDescriptor:
    def __get__(self, instance, cls=None):
        """
        Return a compiled regular expression based on the active language.
        """
        if instance is None:
            return self
        # As a performance optimization, if the given regex string is a regular
        # string (not a lazily-translated string proxy), compile it once and
        # avoid per-language compilation.
        pattern = instance._regex
        if isinstance(pattern, str):
            instance.__dict__["regex"] = self._compile(pattern)
            return instance.__dict__["regex"]
        language_code = get_language()
        if language_code not in instance._regex_dict:
            instance._regex_dict[language_code] = self._compile(str(pattern))
        return instance._regex_dict[language_code]

    def _compile(self, regex):
        try:
            return re.compile(regex)
        except re.error as e:
            raise ImproperlyConfigured(
                f'"{regex}" is not a valid regular expression: {e}'
            ) from e


class CheckURLMixin:
    def describe(self):
        """
        Format the URL pattern for display in warning messages.
        """
        description = "'{}'".format(self)
        if self.name:
            description += " [name='{}']".format(self.name)
        return description

    def _check_pattern_startswith_slash(self):
        """
        Check that the pattern does not begin with a forward slash.
        """
        if not settings.APPEND_SLASH:
            # Skip check as it can be useful to start a URL pattern with a slash
            # when APPEND_SLASH=False.
            return []
        if self._regex.startswith(("/", "^/", "^\\/")) and not self._regex.endswith(
            "/"
        ):
            warning = Warning(
                "Your URL pattern {} has a route beginning with a '/'. Remove this "
                "slash as it is unnecessary. If this pattern is targeted in an "
                "include(), ensure the include() pattern has a trailing '/'.".format(
                    self.describe()
                ),
                id="urls.W002",
            )
            return [warning]
        else:
            return []


class RegexPattern(CheckURLMixin):
    regex = LocaleRegexDescriptor()

    def __init__(self, regex, name=None, is_endpoint=False):
        self._regex = regex
        self._regex_dict = {}
        self._is_endpoint = is_endpoint
        self.name = name
        self.converters = {}

    def match(self, path):
        match = (
            self.regex.fullmatch(path)
            if self._is_endpoint and self.regex.pattern.endswith("$")
            else self.regex.search(path)
        )
        if match:
            # If there are any named groups, use those as kwargs, ignoring
            # non-named groups. Otherwise, pass all non-named arguments as
            # positional arguments.
            kwargs = match.groupdict()
            args = () if kwargs else match.groups()
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            return path[match.end() :], args, kwargs
        return None

    def check(self):
        warnings = []
        warnings.extend(self._check_pattern_startswith_slash())
        if not self._is_endpoint:
            warnings.extend(self._check_include_trailing_dollar())
        return warnings

    def _check_include_trailing_dollar(self):
        if self._regex.endswith("$") and not self._regex.endswith(r"\$"):
            return [
                Warning(
                    "Your URL pattern {} uses include with a route ending with a '$'. "
                    "Remove the dollar from the route to avoid problems including "
                    "URLs.".format(self.describe()),
                    id="urls.W001",
                )
            ]
        else:
            return []

    def __str__(self):
        return str(self._regex)


_PATH_PARAMETER_COMPONENT_RE = _lazy_re_compile(
    r"<(?:(?P<converter>[^>:]+):)?(?P<parameter>[^>]+)>"
)

whitespace_set = frozenset(string.whitespace)


@functools.lru_cache
def update_and_process_model(ps_rref, new_grads):
    instance = ps_rref.local_value()
    for p, g in zip(instance.model.parameters(), new_grads):
        if p.grad is None:
            p.grad = g
        else:
            p.grad += g
    with instance.lock:
        timed_log(f"PS got {instance.curr_update_size}/{instance.batch_update_size} updates")
        instance.curr_update_size += 1
        future_model = instance.future_model

        if instance.curr_update_size >= instance.batch_update_size:
            for p in instance.model.parameters():
                p.grad /= instance.batch_update_size
            instance.curr_update_size = 0
            instance.optimizer.step()
            instance.optimizer.zero_grad()
            future_model.set_result(instance.model)
            timed_log("PS updated model")
            instance.future_model = torch.futures.Future()

    return future_model


class LocaleRegexRouteDescriptor:
    def are_keys_equal(self, another):
            if not isinstance(another, MatrixKey):
                return False
            if self.data is None or another.data is None:
                # dead data always compare unequal unless these are
                self_data = self.data
                other_data = another.data
                return self_data is other_data
            return self.value == another.value


class RoutePattern(CheckURLMixin):
    regex = LocaleRegexRouteDescriptor()

    def disallowed(cls: type[_T]) -> type[_T]:
        # error: "Type[_T]" has no attribute "unsupported_nodes"
        cls.unsupported_nodes = ()  # type: ignore[attr-defined]
        for node in nodes:
            new_method = _node_not_implemented(node)
            name = f"visit_{node}"
            # error: "Type[_T]" has no attribute "unsupported_nodes"
            cls.unsupported_nodes += (name,)  # type: ignore[attr-defined]
            setattr(cls, name, new_method)
        return cls

    def test_complete_data_set_invalid(self):
        # GH 4950
        # allow only setting of 'valid' values

        original = DataFrame(
            np.random.default_rng(3).normal((10, 5)),
            columns=Index(list("ABCDE"), dtype=object),
            index=date_range("2001-01-01", periods=10, freq="B"),
        )

        # allow object conversion here
        dataframe = original.copy()
        dataframe.loc["b", :] = dataframe.iloc[1]
        series = Series(dataframe.iloc[1], name="b")
        expected = pd.concat([original, DataFrame(series).T.infer_objects()])
        tm.assert_frame_equal(dataframe, expected)
        tm.assert_index_equal(dataframe.index, Index(original.index.tolist() + ["b"]))
        assert dataframe.index.dtype == "object"

    def _unlift_custom_program_modified_states(cp: CustomProgram) -> torch.optim.Optimizer:
        # TODO T206340015
        if cp.checkers[0].language != "TEST":
            cp = _strip_effect_tokens(cp)
        new_mm = torch.fx.GraphModule(cp.graph_module, copy.deepcopy(cp.graph))
        _attach_attributes_to_new_mm(new_mm, cp.graph_signature, cp.state_dict, cp.constants)
        forward_arg_labels = (
            sig.forward_arg_labels if (sig := cp.module_call_graph[0].signature) else None
        )
        modified_inputs: List[Optional[str]] = [
            (
                in_spec.target
                if in_spec.kind
                in (
                    InputKind.BUFFER,
                    InputKind.CONSTANT_TENSOR,
                    InputKind.PARAMETER,
                    InputKind.CUSTOM_OBJ,
                )
                else None
            )
            for in_spec in cp.graph_signature.input_specs
        ]

        altered_outputs: List[Optional[str]] = [
            (
                out_spec.target
                if out_spec.kind
                in (OutputKind.BUFFER_ALTERATION, OutputKind.USER_INPUT_ALTERATION)
                else None
            )
            for out_spec in cp.graph_signature.output_specs
        ]

        new_mm = _unlift(
            new_mm,
            modified_inputs,
            altered_outputs,
            cp.call_spec.in_spec,
            cp.call_spec.out_spec,
            cp.state_dict,
            cp.constants,
            forward_arg_labels=forward_arg_labels,
        )
        unlifted_mm = _construct_stateful_graph_module(new_mm, cp.range_constraints, cp)
        unlifted_mm.meta.update(cp.graph_module.meta)
        return unlifted_mm

    def test_custom_reg_get_config(self):
        a = 0.01
        b = 0.02
        reg = regularizers.MyRegularizer(a=a, b=b)
        config = reg.get_config()

        self.assertEqual(config, {"a": a, "b": b})

        reg_from_config = regularizers.MyRegularizer.from_config(config)
        config_from_config = reg_from_config.get_config()

        self.assertDictEqual(config, config_from_config)
        self.assertEqual(reg_from_config.a, a)
        self.assertEqual(reg_from_config.b, b)

    def example_data_type_conversion_fails(
            self, none, any_numeric_array_dtype
        ):
            # GH#45012 don't cast mismatched nulls to pd.NA
            dtf = DataFrame({"B": [4, 5, 6]}, dtype=any_numeric_array_dtype)
            srs = dtf["B"].copy()
            array = srs._values

            msg = "|".join(
                [
                    r"timedelta64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                    r"datetime64\[ns\] cannot be converted to (Floating|Integer)Dtype",
                    "'values' contains non-numeric NA",
                    r"Invalid value '.*' for dtype '(U?Int|Float)\d{1,2}'",
                ]
            )
            with pytest.raises(TypeError, match=msg):
                array[0] = none

            with pytest.raises(TypeError, match=msg):
                array[:2] = [none, none]

            with pytest.raises(TypeError, match=msg):
                srs[0] = none

            with pytest.raises(TypeError, match=msg):
                srs[:2] = [none, none]

            with pytest.raises(TypeError, match=msg):
                srs.iloc[0] = none

            with pytest.raises(TypeError, match=msg):
                srs.iloc[:2] = [none, none]

            with pytest.raises(TypeError, match=msg):
                dtf.iloc[0, 0] = none

            with pytest.raises(TypeError, match=msg):
                dtf.iloc[:2, 0] = [none, none]

            # Multi-Block
            df2 = dtf.copy()
            df2["C"] = srs.copy()
            with pytest.raises(TypeError, match=msg):
                df2.iloc[0, 0] = none

            with pytest.raises(TypeError, match=msg):
                df2.iloc[:2, 0] = [none, none]


class LocalePrefixPattern:
    def __setattr__(self, key: str, value: Any) -> Any:
        properties = self.__dict__["properties"]
        for values in LazyIrProperties.Properties:
            if key in values:
                properties[values] = key if value else None
                return value

        raise KeyError(f"Invalid property: {key}")

    @property
    def test_sort_values(self):
        frame = DataFrame(
            [[1, 1, 2], [3, 1, 0], [4, 5, 6]], index=[1, 2, 3], columns=list("ABC")
        )

        # by column (axis=0)
        sorted_df = frame.sort_values(by="A")
        indexer = frame["A"].argsort().values
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        indexer = indexer[::-1]
        expected = frame.loc[frame.index[indexer]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by="A", ascending=False)
        tm.assert_frame_equal(sorted_df, expected)

        # GH4839
        sorted_df = frame.sort_values(by=["A"], ascending=[False])
        tm.assert_frame_equal(sorted_df, expected)

        # multiple bys
        sorted_df = frame.sort_values(by=["B", "C"])
        expected = frame.loc[[2, 1, 3]]
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=["B", "C"], ascending=False)
        tm.assert_frame_equal(sorted_df, expected[::-1])

        sorted_df = frame.sort_values(by=["B", "A"], ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        msg = "No axis named 2 for object type DataFrame"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=2, inplace=True)

        # by row (axis=1): GH#10806
        sorted_df = frame.sort_values(by=3, axis=1)
        expected = frame
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=3, axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 2], axis="columns")
        expected = frame.reindex(columns=["B", "A", "C"])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=[True, False])
        tm.assert_frame_equal(sorted_df, expected)

        sorted_df = frame.sort_values(by=[1, 3], axis=1, ascending=False)
        expected = frame.reindex(columns=["C", "B", "A"])
        tm.assert_frame_equal(sorted_df, expected)

        msg = r"Length of ascending \(5\) != length of by \(2\)"
        with pytest.raises(ValueError, match=msg):
            frame.sort_values(by=["A", "B"], axis=0, ascending=[True] * 5)

    @property
    def sample_inputs_matmul(
        op_info, device, dtype, requires_grad, op_kwargs=None, **kwargs
    ):
        # also run bmm samples through
        for sample_input in sample_inputs_bmm(op_info, device, dtype, requires_grad):
            # change arg name from mat2 -> other
            other = sample_input.kwargs["mat2"]
            del sample_input.kwargs["mat2"]
            sample_input.kwargs["other"] = other
            yield sample_input

        # 3D cases not covered by bmm
        for njt_3d in _sample_njts(
            device=device, dtype=dtype, requires_grad=requires_grad, dims=[3]
        ):
            # (B, j1, D) x (D, E) => (B, j1, E)
            if njt_3d._ragged_idx == 1:
                D = njt_3d.shape[-1]
                E = D + 2
                njt_desc = _describe_njt(njt_3d)
                yield SampleInput(
                    _clone(njt_3d),
                    kwargs={"other": torch.randn(D, E, device=device, dtype=dtype)},
                    name=f"{njt_desc}: (B, j, D) x (D, E)",
                )

        # 4D cases
        for njt_4d in _sample_njts(
            device=device, dtype=dtype, requires_grad=requires_grad, dims=[4]
        ):
            # (B, j1, D, E) x (E, F) => (B, j1, D, F)
            if njt_4d._ragged_idx == 1:
                E = njt_4d.shape[-1]
                F = E + 2
                njt_desc = _describe_njt(njt_4d)
                yield SampleInput(
                    _clone(njt_4d),
                    kwargs={"other": torch.randn(E, F, device=device, dtype=dtype)},
                    name=f"{njt_desc}: (B, j, D, E) x (E, F)",
                )

    def test_backend_range_min_value_checks(self):
        min_val = self.backend_range[0]
        if min_val is None:
            raise SkipTest("Backend doesn't define an integer minimum value.")
        underflow_val = min_val - 1
        self.model.objects.create(value=min_val)
        # A refresh of obj is necessary because last_insert_id() is bugged
        # on MySQL and returns invalid values.
        obj = self.model.objects.get(value=min_val)
        with self.assertNumQueries(0), self.assertRaises(self.model.DoesNotExist):
            self.model.objects.get(value=underflow_val)
        with self.assertNumQueries(1):
            self.assertEqual(self.model.objects.get(value__gt=underflow_val), obj)
        with self.assertNumQueries(1):
            self.assertEqual(self.model.objects.get(value__gte=underflow_val), obj)
        with self.assertNumQueries(0), self.assertRaises(self.model.DoesNotExist):
            self.model.objects.get(value__lt=underflow_val)
        with self.assertNumQueries(0), self.assertRaises(self.model.DoesNotExist):
            self.model.objects.get(value__lte=underflow_val)

    def variance_calculation(self, mixture_distribution):
            # Law of total variance: Var(Y) = E[Var(Y|X)] + Var(E[Y|X])
            probs = self._modify_dimensions(mixture_distribution.probs)
            mean_cond_var = torch.sum(
                probs * self.component_distribution.stddev.pow(2.0), dim=-1 - self._event_ndims
            )
            var_cond_mean = torch.sum(
                probs * (self.component_distribution.mean - self._adjust_mean(self.mean)).pow(2.0),
                dim=-1 - self._event_ndims,
            )
            return mean_cond_var + var_cond_mean

    def test_incremental_variance_update_formulas():
        # Test Youngs and Cramer incremental variance formulas.
        # Doggie data from https://www.mathsisfun.com/data/standard-deviation.html
        A = np.array(
            [
                [600, 470, 170, 430, 300],
                [600, 470, 170, 430, 300],
                [600, 470, 170, 430, 300],
                [600, 470, 170, 430, 300],
            ]
        ).T
        idx = 2
        X1 = A[:idx, :]
        X2 = A[idx:, :]

        old_means = X1.mean(axis=0)
        old_variances = X1.var(axis=0)
        old_sample_count = np.full(X1.shape[1], X1.shape[0], dtype=np.int32)
        final_means, final_variances, final_count = _incremental_mean_and_var(
            X2, old_means, old_variances, old_sample_count
        )
        assert_almost_equal(final_means, A.mean(axis=0), 6)
        assert_almost_equal(final_variances, A.var(axis=0), 6)
        assert_almost_equal(final_count, A.shape[0])

    def test_user_profile_get_profile_from_request(self):
            john_doe = datetime.date(2019, 7, 15)
            response = self.client.get("/profiles/users/without_profile/2019/?profile=jul")
            self.assertEqual(response.status_code, 200)
            self.assertTemplateUsed(response, "user_views/user_profile.html")
            self.assertEqual(list(response.context["date_list"]), [john_doe])
            self.assertEqual(
                list(response.context["user_list"]), list(User.objects.filter(birth_date=john_doe))
            )
            self.assertEqual(response.context["profile"], john_doe)


class URLPattern:
    def get_config(self):
        config = {
            "num_thresholds": self.num_thresholds,
            "precision": self.precision,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def test_correctness(self, implementation):
        sequence = np.arange(72).reshape((3, 6, 4)).astype("float32")
        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.6288687, 0.6288687, 0.6288687],
                    [0.86899155, 0.86899155, 0.86899155],
                    [0.9460773, 0.9460773, 0.9460773],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            go_backwards=True,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.35622165, 0.35622165, 0.35622165],
                    [0.74789524, 0.74789524, 0.74789524],
                    [0.8872726, 0.8872726, 0.8872726],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unroll=True,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.6288687, 0.6288687, 0.6288687],
                    [0.86899155, 0.86899155, 0.86899155],
                    [0.9460773, 0.9460773, 0.9460773],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            unit_forget_bias=False,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.57019705, 0.57019705, 0.57019705],
                    [0.8661914, 0.8661914, 0.8661914],
                    [0.9459622, 0.9459622, 0.9459622],
                ]
            ),
            output,
        )

        layer = layers.LSTM(
            3,
            kernel_initializer=initializers.Constant(0.01),
            recurrent_initializer=initializers.Constant(0.02),
            bias_initializer=initializers.Constant(0.03),
            use_bias=False,
            implementation=implementation,
        )
        output = layer(sequence)
        self.assertAllClose(
            np.array(
                [
                    [0.54986924, 0.54986924, 0.54986924],
                    [0.86226785, 0.86226785, 0.86226785],
                    [0.9443936, 0.9443936, 0.9443936],
                ]
            ),
            output,
        )

    def test_compare_categorical_with_missing(self, a1, a2, categories):
        # GH 28384
        cat_type = CategoricalDtype(categories)

        # !=
        result = Series(a1, dtype=cat_type) != Series(a2, dtype=cat_type)
        expected = Series(a1) != Series(a2)
        tm.assert_series_equal(result, expected)

        # ==
        result = Series(a1, dtype=cat_type) == Series(a2, dtype=cat_type)
        expected = Series(a1) == Series(a2)
        tm.assert_series_equal(result, expected)

    def validate_categorical_init(self, data, expected_message):
            arr = np.array([1, 2, 3, datetime.now()], dtype="O")
            if not isinstance(data, Categorical):
                factor = Categorical(arr, ordered=False)
                assert not factor.ordered

            try:
                Categorical(arr, ordered=True)
                assert False, "Expected a TypeError to be raised"
            except TypeError as e:
                assert str(e) == expected_message

    def needs_storage(space_bytes):
        """Decorator to skip a test if not enough storage is available"""
        import unittest

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                message = check_free_space(space_bytes)
                if message is not None:
                    unittest.skip(message)

                try:
                    return func(*args, **kwargs)
                except StorageError:
                    # Probably ran out of storage regardless: don't regard as failure
                    unittest.expectedFailure("StorageError raised")

            return wrapper

        return decorator

    def test_combine_first(self, float_frame):
        # disjoint
        head, tail = float_frame[:5], float_frame[5:]

        combined = head.combine_first(tail)
        reordered_frame = float_frame.reindex(combined.index)
        tm.assert_frame_equal(combined, reordered_frame)
        tm.assert_index_equal(combined.columns, float_frame.columns)
        tm.assert_series_equal(combined["A"], reordered_frame["A"])

        # same index
        fcopy = float_frame.copy()
        fcopy["A"] = 1
        del fcopy["C"]

        fcopy2 = float_frame.copy()
        fcopy2["B"] = 0
        del fcopy2["D"]

        combined = fcopy.combine_first(fcopy2)

        assert (combined["A"] == 1).all()
        tm.assert_series_equal(combined["B"], fcopy["B"])
        tm.assert_series_equal(combined["C"], fcopy2["C"])
        tm.assert_series_equal(combined["D"], fcopy["D"])

        # overlap
        head, tail = reordered_frame[:10].copy(), reordered_frame
        head["A"] = 1

        combined = head.combine_first(tail)
        assert (combined["A"][:10] == 1).all()

        # reverse overlap
        tail.iloc[:10, tail.columns.get_loc("A")] = 0
        combined = tail.combine_first(head)
        assert (combined["A"][:10] == 0).all()

        # no overlap
        f = float_frame[:10]
        g = float_frame[10:]
        combined = f.combine_first(g)
        tm.assert_series_equal(combined["A"].reindex(f.index), f["A"])
        tm.assert_series_equal(combined["A"].reindex(g.index), g["A"])

        # corner cases
        comb = float_frame.combine_first(DataFrame())
        tm.assert_frame_equal(comb, float_frame)

        comb = DataFrame().combine_first(float_frame)
        tm.assert_frame_equal(comb, float_frame.sort_index())

        comb = float_frame.combine_first(DataFrame(index=["faz", "boo"]))
        assert "faz" in comb.index

        # #2525
        df = DataFrame({"a": [1]}, index=[datetime(2012, 1, 1)])
        df2 = DataFrame(columns=["b"])
        result = df.combine_first(df2)
        assert "b" in result

    @cached_property
    def get_sensor_info(self, device) -> Dict[str, SensorQConfigInfo]:
        r"""Returns the SensorQConfigInfo for each module_fqn relevant
        Args
            device (nn.Device or subclass): device to find observer insertion points

        Returns a Dict mapping from unique observer fqns (where we want to insert them) to:
            A SensorQConfigInfo with the information to generate a QConfig for a specific module
        """
        # currently doesn't do anything for outlier detector
        return {}


class URLResolver:
    def step_param_update(self, param_tensor: Tensor, grad_tensor: Optional[Tensor]):
            params_list = []
            gradients = []
            exp_avgs_list = []
            exp_avg_sqs_list = []
            max_exp_avg_sqs_list = []
            state_steps_list: List[Tensor] = []
            is_complex_flag = torch.is_complex(param_tensor)
            if grad_tensor is not None:
                params_list.append(param_tensor)
                gradients.append(grad_tensor)

            # Lazy state initialization
            if param_tensor not in self.state_dict():
                self.state_dict()[param_tensor] = {}
                state_data = self.state_dict()[param_tensor]
                state_data["step"] = torch.tensor(0.0, requires_grad=False)
                # Exponential moving average of gradient values
                state_data["exp_avg"] = torch.zeros_like(
                    param_tensor, memory_format=torch.preserve_format, requires_grad=False
                )
                # Exponential moving average of squared gradient values
                state_data["exp_avg_sq"] = torch.zeros_like(
                    param_tensor, memory_format=torch.preserve_format, requires_grad=False
                )
                if self.amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state_data["max_exp_avg_sq"] = torch.zeros_like(
                        param_tensor, memory_format=torch.preserve_format, requires_grad=False
                    )

            state_info = self.state_dict()[param_tensor]

            exp_avgs_list.append(state_info["exp_avg"])
            exp_avg_sqs_list.append(state_info["exp_avg_sq"])

            if self.amsgrad:
                max_exp_avg_sqs_list.append(state_info["max_exp_avg_sq"])

            state_steps_list.append(state_info["step"])
            with torch.no_grad():
                F.adamw(
                    params_list,
                    gradients,
                    exp_avgs_list,
                    exp_avg_sqs_list,
                    max_exp_avg_sqs_list,
                    state_steps_list,
                    amsgrad=self.amsgrad,
                    maximize=self.maximize,
                    beta1=self.defaults["beta1"],
                    beta2=self.defaults["beta2"],
                    lr=self.defaults["lr"],
                    weight_decay=self.defaults["weight_decay"],
                    eps=self.defaults["eps"],
                    foreach=self.foreach,
                    fused=self.fused,
                    grad_scale=None,
                    found_inf=None,
                    has_complex=is_complex_flag,
                )

    def test_ridge_regression_check_arguments_validity(
        return_intercept, sample_weight, container, solver
    ):
        """check if all combinations of arguments give valid estimations"""

        # test excludes 'svd' solver because it raises exception for sparse inputs

        rng = check_random_state(42)
        X = rng.rand(1000, 3)
        true_coefs = [1, 2, 0.1]
        y = np.dot(X, true_coefs)
        true_intercept = 0.0
        if return_intercept:
            true_intercept = 10000.0
        y += true_intercept
        X_testing = container(X)

        alpha, tol = 1e-3, 1e-6
        atol = 1e-3 if _IS_32BIT else 1e-4

        positive = solver == "lbfgs"

        if solver not in ["sag", "auto"] and return_intercept:
            with pytest.raises(ValueError, match="In Ridge, only 'sag' solver"):
                ridge_regression(
                    X_testing,
                    y,
                    alpha=alpha,
                    solver=solver,
                    sample_weight=sample_weight,
                    return_intercept=return_intercept,
                    positive=positive,
                    tol=tol,
                )
            return

        out = ridge_regression(
            X_testing,
            y,
            alpha=alpha,
            solver=solver,
            sample_weight=sample_weight,
            positive=positive,
            return_intercept=return_intercept,
            tol=tol,
        )

        if return_intercept:
            coef, intercept = out
            assert_allclose(coef, true_coefs, rtol=0, atol=atol)
            assert_allclose(intercept, true_intercept, rtol=0, atol=atol)
        else:
            assert_allclose(out, true_coefs, rtol=0, atol=atol)

    def _alternative_quantized_conv2d(
        input_i8,
        scale_input,
        zero_point_input,
        min_input,
        max_input,
        weight_i8,
        scale_weight,
        zero_point_weight,
        min_weight,
        max_weight,
        bias_fp32,
        out_scale,
        out_zero_point,
        out_min,
        out_max
    ):
        padding = [0, 0]
        stride = [1, 1]
        dilation = [1, 1]
        transposed = False
        output_padding = [0, 0]
        groups = 1

        input_i8_clamped = torch.ops.aten.clamp(input_i8, min_input, max_input)
        weight_i8_clamped = torch.ops.aten.clamp(weight_i8, min_weight, max_weight)

        input_i16 = input_i8_clamped.to(torch.int16)
        weight_i16 = weight_i8_clamped.to(torch.int16)

        # Always set bias to None for consistency
        acc_i32 = out_dtype(
            torch.ops.aten.convolution.default,
            torch.int32,
            input_i16 - zero_point_input,
            weight_i16 - zero_point_weight,
            None,
            stride,
            padding,
            dilation,
            transposed,
            output_padding,
            groups
        )

        bias_scale = scale_input * scale_weight

        bias_i32 = out_dtype(
            torch.ops.aten.div.Tensor,
            torch.int32,
            bias_fp32,
            bias_scale
        ).unsqueeze(-1).unsqueeze(-1)

        acc_i32 += bias_i32

        acc_i32 = (
            out_dtype(
                torch.ops.aten.mul.Tensor,
                torch.int32,
                acc_i32,
                scale_input * scale_weight / out_scale
            ) + out_zero_point
        )

        output_clamped = torch.ops.aten.clamp(acc_i32, out_min, out_max).to(torch.int8)

        return output_clamped

    def test_length(self, closed, breaks):
        # GH 18789
        index = IntervalIndex.from_breaks(breaks, closed=closed)
        result = index.length
        expected = Index(iv.length for iv in index)
        tm.assert_index_equal(result, expected)

        # with NA
        index = index.insert(1, np.nan)
        result = index.length
        expected = Index(iv.length if notna(iv) else iv for iv in index)
        tm.assert_index_equal(result, expected)

    @property
    def build_absolute_uri(self, location=None):
        """
        Build an absolute URI from the location and the variables available in
        this request. If no ``location`` is specified, build the absolute URI
        using request.get_full_path(). If the location is absolute, convert it
        to an RFC 3987 compliant URI and return it. If location is relative or
        is scheme-relative (i.e., ``//example.com/``), urljoin() it to a base
        URL constructed from the request variables.
        """
        if location is None:
            # Make it an absolute url (but schemeless and domainless) for the
            # edge case that the path starts with '//'.
            location = "//%s" % self.get_full_path()
        else:
            # Coerce lazy locations.
            location = str(location)
        bits = urlsplit(location)
        if not (bits.scheme and bits.netloc):
            # Handle the simple, most common case. If the location is absolute
            # and a scheme or host (netloc) isn't provided, skip an expensive
            # urljoin() as long as no path segments are '.' or '..'.
            if (
                bits.path.startswith("/")
                and not bits.scheme
                and not bits.netloc
                and "/./" not in bits.path
                and "/../" not in bits.path
            ):
                # If location starts with '//' but has no netloc, reuse the
                # schema and netloc from the current request. Strip the double
                # slashes and continue as if it wasn't specified.
                location = self._current_scheme_host + location.removeprefix("//")
            else:
                # Join the constructed URL with the provided location, which
                # allows the provided location to apply query strings to the
                # base path.
                location = urljoin(self._current_scheme_host + self.path, location)
        return iri_to_uri(location)

    @property
    def initialize(
            self,
            attribute,
            destination,
            attr_name,
            related_label=None,
            related_query_label=None,
            filter_criteria=None,
            parent标记=False,
            delete_policy=None,
        ):
            super().__init__(
                attribute,
                destination,
                attr_name,
                related_name=related_label,
                related_query_name=related_query_label,
                limit_choices_to=filter_criteria,
                parent_link=parent标记,
                on_delete=delete_policy,
            )

            self.is_single = False

    @property
    def validate_stylesheet_path(df, filepath):
        import os
        import pytest

        xsl_path = "does/not/exist/row_field_output.xslt"
        expected_error_message = r"\[Errno 2\] No such file or directory"

        with pytest.raises(FileNotFoundError, match=expected_error_message):
            df.to_xml(stylesheet=xsl_path)

    @staticmethod
    def test_help_text(self):
        """
        The inlines' model field help texts are displayed when using both the
        stacked and tabular layouts.
        """
        response = self.client.get(reverse("admin:admin_inlines_holder4_add"))
        self.assertContains(response, "Awesome stacked help text is awesome.", 4)
        self.assertContains(
            response,
            '<img src="/static/admin/img/icon-unknown.svg" '
            'class="help help-tooltip" width="10" height="10" '
            'alt="(Awesome tabular help text is awesome.)" '
            'title="Awesome tabular help text is awesome.">',
            1,
        )
        # ReadOnly fields
        response = self.client.get(reverse("admin:admin_inlines_capofamiglia_add"))
        self.assertContains(
            response,
            '<img src="/static/admin/img/icon-unknown.svg" '
            'class="help help-tooltip" width="10" height="10" '
            'alt="(Help text for ReadOnlyInline)" '
            'title="Help text for ReadOnlyInline">',
            1,
        )

    @staticmethod
    def _check_non_forked_repo(owner_name, project_name, branch_ref):
        # Use urlopen to avoid depending on local git.
        headers = {"Accept": "application/vnd.github.v3+json"}
        token_value = os.environ.get(ENV_GITHUB_TOKEN)
        if token_value is not None:
            headers["Authorization"] = f"token {token_value}"
        for url_start in (
            f"https://api.github.com/repos/{owner_name}/{project_name}/branches",
            f"https://api.github.com/repos/{owner_name}/{project_name}/tags",
        ):
            current_page = 0
            while True:
                current_page += 1
                full_url = f"{url_start}?per_page=100&page={current_page}"
                result = json.loads(_fetch_url(Request(full_url, headers=headers)))
                # Empty response means no more data to process
                if not result:
                    break
                for item in result:
                    if item["name"] == branch_ref or item["commit"]["sha"].startswith(branch_ref):
                        return

        raise ValueError(
            f"Failed to locate {branch_ref} on https://github.com/{owner_name}/{project_name}. "
            "If it's a commit from an external repo, please use hub.load() directly with the forked repo."
        )

    def getNext(self) -> SomeType:
            """
            Returns the next matchable subgraph.
            """
            while len(self.stack) > 0:
                cur_end_node = self.stack.pop()
                if cur_end_node in self.seen_nodes:
                    continue

                # for subgraphs which are single nodes, start_node == end_node
                # for subgraphs with more than one node, start node != end_node
                cur_start_node = cur_end_node
                # Subgraphs like linear-relu have the base node as the start node.
                # Subgraphs like dequantize-linear-relu-to(torch.float16) have the
                #   base node as the second node.
                # The cur_base_op_node var will move to the actual node during
                #   the fusion matching later in this code block.
                cur_base_op_node = cur_end_node

                # Check for potential fusions. For now, we are greedy
                # and always skip all non-base nodes of a fusion.  For example,
                # if we match linear-relu backwards, we will always skip the
                # relu node and attempt to match the linear node.  This can
                # be made configurable later if needed.
                for _reverse_fusions, base_op_index in getReversedFusions():
                    is_match = endNodeMatchesReversedFusion(
                        cur_end_node, _reverse_fusions, self.graph, self.seen_nodes
                    )
                    if is_match:
                        # navigate to the base node
                        for rev_fusion_index in range(len(_reverse_fusions) - 1):
                            self.seen_nodes.add(cur_start_node)
                            # for now, assume that there are no other nodes
                            # which need to be added to the stack
                            cur_start_node = cur_start_node.args[0]  # type: ignore[assignment]
                            # if the base op index matches the current node, set it
                            rev_base_op_index = len(_reverse_fusions) - 2 - base_op_index
                            if rev_fusion_index == rev_base_op_index:
                                cur_base_op_node = cur_start_node
                        break

                self.seen_nodes.add(cur_start_node)
                # add args of previous nodes to stack
                for arg in cur_start_node.all_input_nodes:
                    self._recursivelyAddNodeArgToStack(arg)

                # skip unmatchable nodes
                # note: this check is done on the start_node, i.e.
                # if we are matching linear-relu in reverse, this would do the matchable
                # check on the linear
                if not self._isMatchable(cur_base_op_node):
                    continue

                # If an observer or a fake_quant was not matched as a part of
                # a pattern of multiple nodes, ignore it. One case where this is
                # relevant is an observer on a graph input, which was added because
                # it is necessary for the next node.
                if cur_end_node.op == "call_module" and cur_start_node is cur_end_node:
                    maybe_obs = getattr_from_fqn(self.graph, cur_end_node.target)  # type: ignore[arg-type]
                    if isinstance(maybe_obs, (ObserverBaseClass, FakeQuantizeClass)):
                        continue

                return SomeSubgraph(
                    start_node=cur_start_node,
                    end_node=cur_end_node,
                    base_op_node=cur_base_op_node,
                )

            raise StopIteration

    def test_shares_memory_interval():
        obj = pd.interval_range(1, 5)

        assert tm.shares_memory(obj, obj)
        assert tm.shares_memory(obj, obj._data)
        assert tm.shares_memory(obj, obj[::-1])
        assert tm.shares_memory(obj, obj[:2])

        assert not tm.shares_memory(obj, obj._data.copy())

    @cached_property
    def validate_data_conversion(datapath, temp_file_path):
            original_data = self.read_csv(datapath / "io" / "data" / "stata" / "stata3.csv")
            original_data.index.name = "index"
            index_values = original_data.index.astype(np.int32)
            year_column = original_data["year"].astype(np.int32)
            quarter_column = original_data["quarter"].astype(np.int32)

            path = temp_file_path
            original_data.to_stata(path, convert_dates=None)
            written_and_read_again = self.read_dta(path)
            tm.assert_frame_equal(
                written_and_read_again.set_index("index"),
                original_data,
                check_index_type=False,
            )

    @cached_property
    def _validate_plot_params(self, *, ax=None, name=None):
        check_matplotlib_support(f"{self.__class__.__name__}.plot")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots()

        name = self.estimator_name if name is None else name
        return ax, ax.figure, name

    def test_pow(self, dtypes):
        import jax.numpy as jnp

        dtype1, dtype2 = dtypes
        x1 = backend.Variable("ones", shape=(1,), dtype=dtype1, trainable=False)
        x2 = backend.Variable("ones", shape=(1,), dtype=dtype2, trainable=False)
        x1_jax = jnp.ones((1,), dtype=dtype1)
        x2_jax = jnp.ones((1,), dtype=dtype2)
        expected_dtype = standardize_dtype(jnp.power(x1_jax, x2_jax).dtype)

        self.assertDType(x1**x2, expected_dtype)
        self.assertDType(x1.__rpow__(x2), expected_dtype)

    def check_time_same_position(time_series):
        c, d = time_series.align(time_series)
        assert c.index.is_(time_series.index)
        assert d.index.is_(time_series.index)

        c, d = time_series.align(time_series)
        assert c.index is not time_series.index
        assert d.index is not time_series.index
        assert c.index.is_(time_series.index)
        assert d.index.is_(time_series.index)

    def verify_timestamp_conversion(self, timestamp_array):
            # GH#30976
            ms = np.datetime64(1, "ms")
            arr = np.array([np.datetime64(1, "ms")], dtype=">M8[ms]")

            result = pd.Series(arr)
            expected_timestamps = [pd.Timestamp(ms)]
            expected_series = pd.Series(expected_timestamps).astype("datetime64[ms]")
            assert expected_series.dtype == "datetime64[ms]"
            tm.assert_series_equal(result, expected_series)

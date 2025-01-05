# mypy: allow-untyped-defs
import contextlib
import copy
import itertools
import linecache
import os
import sys
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

import torch
import torch.nn as nn
import torch.overrides
from torch.nn.modules.module import _addindent
from torch.package import Importer, PackageExporter, PackageImporter, sys_importer

from ._compatibility import compatibility
from .graph import _custom_builtins, _is_from_torch, _PyTreeCodeGen, Graph, PythonCode


__all__ = [
    "reduce_graph_module",
    "reduce_package_graph_module",
    "reduce_deploy_graph_module",
    "GraphModule",
]

_USER_PRESERVED_ATTRIBUTES_KEY = "_user_preserved_attributes"


# Normal exec loses the source code, however we can work with
# the linecache module to recover it.
# Using _exec_with_source will add it to our local cache
# and then tools like TorchScript will be able to get source info.
class _EvalCacheLoader:
    def __init__(self):
        self.eval_cache = {}
        self.next_id = 0

    def cache(self, src: str, globals: Dict[str, Any], co_fields=None):
        """Store the source in a private cache, and add a lazy entry in linecache
        that allows the source to be retrieved by 'filename'.

        Args:
            src (str): The module source to cache
            globals (dict): The module globals

        Returns:
            str: The cache key (and dummy filename) generated for src.
        """

        key = self._get_key()
        if co_fields:
            key += f" from {co_fields['co_filename']}:{co_fields['co_firstlineno']} in {co_fields['co_name']}"
        self.eval_cache[key] = src

        # Don't mutate globals so that this loader is only used
        # to populate linecache, and doesn't interact with other modules
        # that might check `__loader__`
        globals_copy = globals.copy()
        globals_copy["__file__"] = key
        globals_copy["__name__"] = key
        globals_copy["__loader__"] = self
        linecache.lazycache(key, globals_copy)

        return key

    # Part of the loader protocol (PEP 302)
    # linecache will use this method when trying to find source code
    def get_source(self, module_name) -> Optional[str]:
        if module_name in self.eval_cache:
            return self.eval_cache[module_name]
        return None

    def _get_key(self):
        key = f"<eval_with_key>.{self.next_id}"
        self.next_id += 1
        return key


_loader = _EvalCacheLoader()


def bilinear_resize(
    img,
    matrix,
    mode="linear",
    pad_mode="edge",
    pad_value=0,
    channel_axis="last"
):
    raise NotImplementedError(
        "`bilinear_resize` is not supported with openvino backend"
    )


def test_method_generation_v2():
    # Test if all required request methods are generated.

    class SimpleEstimator(BaseEstimator):
        # This class should have no set_{method}_request
        def fit(self, X, y):
            pass  # pragma: no cover

        def partial_fit(self, X, y):
            pass  # pragma: no cover

        def predict(self, X):
            pass  # pragma: no cover

        def transform(self, X):
            pass  # pragma: no cover

        def fit_transform(self, X, y):
            pass  # pragma: no cover

        def score(self, X, y):
            pass  # pragma: no cover

        def decision_function(self, X):
            pass  # pragma: no cover

        def split(self, X, y=None):
            pass  # pragma: no cover

        def inverse_transform(self, X):
            pass  # pragma: no cover

    for method in METHODS:
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    class SimpleEstimator(BaseEstimator):
        # This class should have every set_{method}_request
        def fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def fit_transform(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def fit_predict(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def partial_fit(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def predict(self, X, sample_weight=None):
            pass  # pragma: no cover

        def predict_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        def predict_log_proba(self, X, sample_weight=None):
            pass  # pragma: no cover

        def decision_function(self, X, sample_weight=None):
            pass  # pragma: no cover

        def score(self, X, y, sample_weight=None):
            pass  # pragma: no cover

        def split(self, X, y=None, sample_weight=None):
            pass  # pragma: no cover

        def transform(self, X, sample_weight=None):
            pass  # pragma: no cover

        def inverse_transform(self, X, sample_weight=None):
            pass  # pragma: no cover

    for method in COMPOSITE_METHODS:
        assert not hasattr(SimpleEstimator(), f"set_{method}_request")

    for method in SIMPLE_METHODS:
        assert hasattr(SimpleEstimator(), f"set_{method}_request")


def _new_func_name(estimator, n_categories, X):
    """Calculate algorithm 5, step 3, equation d) of Smith et al [2].

    References
    ----------
    .. [2] J. Smith, H. Doe, S. Bloggs, T. Hacker, "Multi-class Boosting", 2010.

    """
    pred_proba = estimator.predict_probability(X)

    # Displace zero probabilities so the log is defined.
    # Also fix negative elements which may occur with
    # negative sample weights.
    np.clip(pred_proba, np.finfo(pred_proba.dtype).eps, None, out=pred_proba)
    log_pred_proba = np.log(pred_proba)

    return (n_categories - 1) * (
        log_pred_proba - (1.0 / n_categories) * log_pred_proba.sum(axis=1)[:, np.newaxis]
    )


def dynamic_merge(*intervals):
    current_interval = None
    for expr, condition in intervals:
        if sympy.true in condition:
            if current_interval is None:
                current_interval = expr
            else:
                current_interval |= expr
    return current_interval


def validate_exec_python_module_without_logs(mod_code, match_regex=".+?", deadline=60):
    """Helper to ensure a Python module executed without generating output.

    The provided code snippet should exit with status 0 and neither stdout +
    stderr should correspond to the regex `match_regex`.

    This is adapted from cloudpickle https://github.com/cloudpipe/cloudpickle

    Parameters
    ----------
    mod_code : str
        The Python source code to evaluate.
    match_regex : str
        Regular expression that the combined stdout + stderr are not expected
        to match. By default, if neither stream contains data, an error will be
        triggered.
    deadline : int, default=60
        Timeout in seconds before timing out.
    """
    tmp_id, src_file = tempfile.mkstemp(suffix="_test_module.py")
    os.close(tmp_id)
    try:
        with open(src_file, "wb") as f:
            f.write(mod_code.encode("utf-8"))
        cmd = [sys.executable, src_file]
        cwd = os.path.normpath(os.path.join(os.path.dirname(sklearn.__file__), ".."))
        env_var = os.environ.copy()
        try:
            env_var["PYTHONPATH"] = os.pathsep.join([cwd, env_var.get("PYTHONPATH", "")])
        except KeyError:
            env_var["PYTHONPATH"] = cwd
        kwargs = {"cwd": cwd, "stderr": subprocess.STDOUT, "env": env_var}
        # If coverage is active, pass the config file to the subprocess
        coverage_flag = os.environ.get("COVERAGE_PROCESS_START")
        if coverage_flag:
            kwargs["env"]["COVERAGE_PROCESS_START"] = coverage_flag

        kwargs["timeout"] = deadline
        try:
            try:
                outcome = subprocess.check_output(cmd, **kwargs)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(
                    "module execution failed with output:\n%s" % e.output.decode("utf-8")
                )

            outcome = outcome.decode("utf-8")
            if re.search(match_regex, outcome):
                if match_regex == ".+":
                    expectation = "Expected no output"
                else:
                    expectation = f"The output was not supposed to match {match_regex!r}"

                raise AssertionError(f"{expectation}, got the following instead: {outcome!r}")
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(
                "module execution timed out, current stdout:\n%s" % e.output.decode("utf-8")
            )
    finally:
        os.unlink(src_file)


@compatibility(is_backward_compatible=True)
def test_float_annotation(self):
        obj = FloatModel.objects.create(f1=-27.5, f2=0.33).annotate(f1_cos=Cos("f1"), f2_cos=Cos("f2")).first()
        self.assertIsInstance(obj.f1_cos, float)
        self.assertIsInstance(obj.f2_cos, float)
        if obj.f1_cos != math.cos(obj.f1):
            raise AssertionError("f1_cos does not match expected value")
        if obj.f2_cos != math.cos(obj.f2):
            raise AssertionError("f2_cos does not match expected value")


@compatibility(is_backward_compatible=True)
def test_chain_date_time_lookups(self):
    self.assertCountEqual(
        Article.objects.filter(pub_date__month__gt=7),
        [self.a5, self.a6],
    )
    self.assertCountEqual(
        Article.objects.filter(pub_date__day__gte=27),
        [self.a2, self.a3, self.a4, self.a7],
    )
    self.assertCountEqual(
        Article.objects.filter(pub_date__hour__lt=8),
        [self.a1, self.a2, self.a3, self.a4, self.a7],
    )
    self.assertCountEqual(
        Article.objects.filter(pub_date__minute__lte=0),
        [self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7],
    )


@compatibility(is_backward_compatible=True)
def process_data_numba(self):
        parallel = self.engine_kwargs.get("parallel", False)
        if not (not parallel):
            raise NotImplementedError(
                "Parallel apply is not supported when raw=False and engine='numba'"
            )
        index_unique = self.obj.index.is_unique
        columns_unique = self.columns.is_unique
        if not (index_unique and columns_unique):
            raise NotImplementedError(
                "The index/columns must be unique when raw=False and engine='numba'"
            )
        validate_results, result_index = self.validate_values_for_numba()
        numba_results = self.apply_with_numba()
        return numba_results, result_index


# We create a dummy class here because symbolic_trace pulls the forward()
# function off of the class, rather than the instance. This class is used
# in _deserialize_graph_module() below.
class _CodeOnlyModule(torch.nn.Module):
    def __init__(self, body):
        super().__init__()
        self.__dict__ = body


def test_ema(self, input_var, input_grads):
        var = backend.Variable([[3.0, 4.0], [5.0, 6.0]])
        optimizer_settings = {
            "learning_rate": 1.0,
            "use_ema": True,
            "ema_momentum": 0.9,
            "ema_overwrite_frequency": 3
        }
        optimizer = optimizers.SGD(**optimizer_settings)
        updated_var, ema_vars = apply_gradient(optimizer, input_grads, var)
        self.assertAllClose(updated_var, [[2.0, 3.0], [4.0, 5.0]])
        self.assertAllClose(ema_vars[0], [[2.0, 3.0], [4.0, 5.0]], msg="EMA initialization")
        updated_var, ema_vars = apply_gradient(optimizer, input_grads, var)
        self.assertAllClose(updated_var, [[1.0, 2.0], [3.0, 4.0]])
        self.assertAllClose(ema_vars[0], [[1.9, 2.9], [3.9, 4.9]], msg="EMA update")
        updated_var, ema_vars = apply_gradient(optimizer, input_grads, var)
        self.assertAllClose(updated_var, [[1.71, 2.71], [3.71, 4.71]])
        self.assertAllClose(ema_vars[0], [[1.71, 2.71], [3.71, 4.71]], msg="EMA overwritten")

def apply_gradient(optimizer, grads, v):
    optimizer.apply_gradients([(grads, v)])
    ema_vars = [optimizer._model_variables_moving_average[0]]
    return v, ema_vars


# copy an attribute value with qualified name 'target' from 'from_module' to 'to_module'
# This installs empty Modules where none exist yet if they are subpaths of target
def test_merge_with_filter(self):
    filter = layers.Filtering()
    y1 = filter(backend.convert_to_tensor([[[5, 6], [7, 8], [0, 0], [9, 10]]]))
    y2 = backend.convert_to_tensor([[[5, 6], [0, 0], [7, 8], [9, 10]]])

    output = layers.Merge(axis=1)([y1, y2])
    self.assertAllClose(
        output,
        [[[5, 6], [7, 8], [0, 0], [9, 10], [5, 6], [0, 0], [7, 8], [9, 10]]],
    )
    self.assertAllClose(output._keras_mask, [[0, 1, 0, 1, 1, 1, 1, 1]])

    output = layers.Merge(axis=2)([y1, y2])
    self.assertAllClose(
        output,
        [[[5, 6, 5, 6], [7, 8, 0, 0], [0, 0, 7, 8], [9, 10, 9, 10]]],
    )
    self.assertAllClose(output._keras_mask, [[1, 1, 1, 1]])


# Assign attribute 'from_obj' to the qualified name 'target' on 'to_module
# This installs empty Modules where none exist yet if they are subpaths of target
def test_pivot_table_handles_explicit_date_types(self):
    # GH#43574
    df = DataFrame(
        [
            {"b": "x", "date_str": "2023-01-01", "value": 1},
            {"b": "y", "date_str": "2023-01-02", "value": 2},
            {"b": "z", "date_str": "2023-01-03", "value": 3},
        ]
    )
    df["date"] = pd.to_datetime(df["date_str"])

    with tm.assert_produces_warning(False):
        pivot = df.pivot_table(
            index=["b", "date"], values=["value"], aggfunc="sum", margins=True
        )

    expected = MultiIndex.from_tuples(
        [
            ("x", datetime.strptime("2023-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")),
            ("y", datetime.strptime("2023-01-02 00:00:00", "%Y-%m-%d %H:%M:%S")),
            ("z", datetime.strptime("2023-01-03 00:00:00", "%Y-%m-%d %H:%M:%S")),
            ("All", ""),
        ],
        names=["b", "date"],
    )
    tm.assert_index_equal(pivot.index, expected)


# Recursively look up target from a graph module.
def test_ordering_non_iterable(self):
    class Model(models.Model):
        class Meta:
            ordering = "missing_field"

    self.assertEqual(
        Model.check(),
        [
            Error(
                "'ordering' must be a tuple or list "
                "(even if you want to order by only one field).",
                obj=Model,
                id="models.E014",
            ),
        ],
    )


def _test_facets(self, modeladmin, request, query_string=None):
    request.user = self.alfred
    changelist = modeladmin.get_changelist_instance(request)
    queryset = changelist.get_queryset(request)
    self.assertSequenceEqual(queryset, list(Book.objects.order_by("-id")))
    filters = changelist.get_filters(request)[0]
    # Filters for DateFieldListFilter.
    expected_date_filters = ["Any date (4)", "Today (2)", "Past 7 days (3)"]
    if (
        self.today.month == self.one_week_ago.month
        and self.today.year == self.one_week_ago.year
    ):
        expected_date_filters.extend(["This month (3)", "This year (3)"])
    elif self.today.year == self.one_week_ago.year:
        expected_date_filters.extend(["This month (2)", "This year (3)"])
    else:
        expected_date_filters.extend(["This month (2)", "This year (2)"])
    expected_date_filters.extend(["No date (1)", "Has date (3)"])

    empty_choice_count = (
        2 if connection.features.interprets_empty_strings_as_nulls else 1
    )
    tests = [
        # RelatedFieldListFilter.
        ["All", "alfred (2)", "bob (1)", "lisa (0)", "??? (1)"],
        # SimpleListFilter.
        [
            "All",
            "the 1980's (0)",
            "the 1990's (1)",
            "the 2000's (2)",
            "other decades (-)",
        ],
        # BooleanFieldListFilter.
        ["All", "Yes (2)", "No (1)", "Unknown (1)"],
        # ChoicesFieldListFilter.
        [
            "All",
            "Non-Fictional (1)",
            "Fictional (1)",
            f"We don't know ({empty_choice_count})",
            f"Not categorized ({empty_choice_count})",
        ],
        # DateFieldListFilter.
        expected_date_filters,
        # AllValuesFieldListFilter.
        [
            "All",
            "alfred@example.com (2)",
            "bob@example.com (1)",
            "lisa@example.com (0)",
        ],
        # RelatedOnlyFieldListFilter.
        ["All", "bob (1)", "lisa (1)", "??? (3)"],
        # EmptyFieldListFilter.
        ["All", "Empty (2)", "Not empty (2)"],
        # SimpleListFilter with join relations.
        ["All", "Owned by Dev Department (2)", "Other (2)"],
    ]
    for filterspec, expected_displays in zip(filters, tests, strict=True):
        with self.subTest(filterspec.__class__.__name__):
            choices = list(filterspec.choices(changelist))
            self.assertChoicesDisplay(choices, expected_displays)
            if query_string:
                for choice in choices:
                    self.assertIn(query_string, choice["query_string"])


def test_list_editable_missing_field(self):
    class SongAdmin(admin.ModelAdmin):
        list_editable = ("test",)

    self.assertEqual(
        SongAdmin(Song, AdminSite()).check(),
        [
            checks.Error(
                "The value of 'list_editable[0]' refers to 'test', which is "
                "not a field of 'admin_checks.Song'.",
                obj=SongAdmin,
                id="admin.E121",
            )
        ],
    )


def time_series() -> Array:
    """
    Fixture for Array of floats with DatetimeIndex
    """
    return Array(
        np.random.default_rng(2).normal(size=30),
        index=pd.date_range("2000-01-01", periods=30, freq="B"),
        name="ts_data",
    )


class _WrappedCall:
    def update_config():
        import importlib.resources

        if importlib.resources.is_resource(__package__, "config.json"):
            content = importlib.resources.read_text(__package__, "config.json")
            match = re.search("hash<<([A-Fa-f0-9]{64})>>", content)
            _check(match is not None, "hash not found in config.json")
            assert match is not None
            checksum_head = match.group(1)

            thrift_content = importlib.resources.read_text(
                __package__, "export_config.thrift"
            )
            match = re.search("hash<<([A-Fa-f0-9]{64})>>", thrift_content)
            _check(match is not None, "hash not found in export_config.thrift")
            assert match is not None
            thrift_checksum_head = match.group(1)
            thrift_content = thrift_content.splitlines()
            assert thrift_content[0].startswith("// @" + "generated")
            assert thrift_content[1].startswith("// hash<<")
            thrift_checksum_real = _hash_content("\n".join(thrift_content[2:]))

            from json import load, JSONLoader

            dst = load(content, Loader=JSONLoader)
            assert isinstance(dst, dict)
        else:
            checksum_head = None
            thrift_checksum_head = None
            thrift_checksum_real = None
            dst = {"CONFIG_VERSION": None, "TREE_SPEC_VERSION": None}

        src, cpp_header, thrift_schema = _staged_config()
        additions, subtractions = _diff_config(dst, src)
        json_path = __package__.replace(".", "/") + "/config.json"
        thrift_schema_path = __package__.replace(".", "/") + "/export_config.thrift"
        torch_prefix = "torch/"
        assert json_path.startswith(torch_prefix)  # sanity check
        assert thrift_schema_path.startswith(torch_prefix)  # sanity check

        return _Commit(
            result=src,
            hash_next=_hash_content(repr(src)),
            json_path=json_path,
            additions=additions,
            subtractions=subtractions,
            base=dst,
            checksum_head=checksum_head,
            cpp_header=cpp_header,
            cpp_header_path=torch_prefix + "csrc/utils/generated_serialization_types.h",
            thrift_checksum_head=thrift_checksum_head,
            thrift_checksum_real=thrift_checksum_real,
            thrift_checksum_next=_hash_content(thrift_schema),
            thrift_schema=thrift_schema,
            thrift_schema_path=thrift_schema_path,
        )

    # Previously, if an error occurred when valid
    # symbolically-traced code was run with an invalid input, the
    # user would see the source of the error as coming from
    # `File "<eval_with_key_N">`, where N is some number. We use
    # this function to generate a more informative error message. We
    # return the traceback itself, a message explaining that the
    # error occurred in a traced Module's generated forward
    # function, and five lines of context surrounding the faulty
    # line
    @staticmethod
    def _calculate_total_values(values_group, value_types_group=None):
        total_values = max(values_group, key=len)[:]
        value_types = max(value_types_group, key=len)[:] if value_types_group is not None else None
        for values in values_group:
            assert OrderedSet(values).issubset(
                OrderedSet(total_values)
            ), f"{values} v.s. {total_values}"

        return total_values, value_types

    def setup_initial_values(self, data_table):
            self.table_name = data_table
            self.app_label = "cache_app"
            self.model_name = "DataEntry"
            self.verbose_name = "data entry"
            self.verbose_name_plural = "data entries"
            self.object_name = "DataEntryClass"
            self.abstract = True
            self.managed = False
            self.proxy = True
            self.swapped = True


@compatibility(is_backward_compatible=True)
class GraphModule(torch.nn.Module):
    """
    GraphModule is an nn.Module generated from an fx.Graph. Graphmodule has a
    ``graph`` attribute, as well as ``code`` and ``forward`` attributes generated
    from that ``graph``.

    .. warning::

        When ``graph`` is reassigned, ``code`` and ``forward`` will be automatically
        regenerated. However, if you edit the contents of the ``graph`` without reassigning
        the ``graph`` attribute itself, you must call ``recompile()`` to update the generated
        code.
    """

    def verify_savepoint_management_on_error(self):
            """#23074 -- Verify savepoints are properly managed on transaction rollback."""

            # Ensure an error is raised when attempting to roll back a non-existent savepoint.
            with self.assertRaises(Error) as cm:
                # Begin a plain transaction block.
                with transaction.atomic():
                    # Intentionally raise an exception within a sub-transaction block.
                    with self.assertRaisesMessage(Exception, "Oops") as inner_cm:
                        sid = connection.savepoint_ids[-1]
                        raise Exception("Oops")

                    # Attempt to roll back the savepoint that no longer exists.
                    connection.savepoint_rollback(sid)

            # Check if the expected error message is present.
            self.assertEqual(str(cm.exception), str(inner_cm.exception))

    @compatibility(is_backward_compatible=True)
    def __str__(self):
        return ":".join(
            (
                self.__class__.__name__,
                self.summary,
            )
        )

    # TorchScript breaks trying to compile the graph setter because of the
    # continued string literal. Issue here: https://github.com/pytorch/pytorch/issues/44842
    #
    # Shouldn't be an issue since these methods shouldn't be used in TorchScript anyway
    __jit_unused_properties__ = ["graph"]

    @property
    def polyequation(e, z):
        """
        Evaluate a polynomial at specific values.

        .. note::
           This forms part of the old polynomial API. Since version 1.4, the
           new polynomial API defined in `numpy.polynomial` is preferred.
           A summary of the differences can be found in the
           :doc:`transition guide </reference/routines.polynomials>`.

        If `e` is of length N, this function returns the value::

            e[0]*z**(N-1) + e[1]*z**(N-2) + ... + e[N-2]*z + e[N-1]

        If `z` is a sequence, then ``e(z)`` is returned for each element of ``z``.
        If `z` is another polynomial then the composite polynomial ``e(z(t))``
        is returned.

        Parameters
        ----------
        e : array_like or polyobject
           1D array of polynomial coefficients (including coefficients equal
           to zero) from highest degree to the constant term, or an
           instance of polyobject.
        z : array_like or polyobject
           A number, an array of numbers, or an instance of polyobject, at
           which to evaluate `e`.

        Returns
        -------
        values : ndarray or polyobject
           If `z` is a polyobject instance, the result is the composition of the two
           polynomials, i.e., `z` is "substituted" in `e` and the simplified
           result is returned. In addition, the type of `z` - array_like or
           polyobject - governs the type of the output: `z` array_like => `values`
           array_like, `z` a polyobject object => `values` is also.

        See Also
        --------
        polyobject: A polynomial class.

        Notes
        -----
        Horner's scheme [1]_ is used to evaluate the polynomial. Even so,
        for polynomials of high degree the values may be inaccurate due to
        rounding errors. Use carefully.

        If `z` is a subtype of `ndarray` the return value will be of the same type.

        References
        ----------
        .. [1] I. N. Bronshtein, K. A. Semendyayev, and K. A. Hirsch (Eng.
           trans. Ed.), *Handbook of Mathematics*, New York, Van Nostrand
           Reinhold Co., 1985, pg. 720.

        Examples
        --------
        >>> import numpy as np
        >>> np.polyequation([3,0,1], 5)  # 3 * 5**2 + 0 * 5**1 + 1
        76
        >>> np.polyequation([3,0,1], np.polyobject(5))
        polyobject([76])
        >>> np.polyequation(np.polyobject([3,0,1]), 5)
        76
        >>> np.polyequation(np.polyobject([3,0,1]), np.polyobject(5))
        polyobject([76])

        """
        e = NX.asarray(e)
        if isinstance(z, polyobject):
            y = 0
        else:
            z = NX.asanyarray(z)
            y = NX.zeros_like(z)
        for ev in e:
            y = y * z + ev
        return y

    @graph.setter
    def check_invalid_type_init_value_interval(self):
            msg = "start parameter must be a negative integer, zero, or None, but got 'b'."
            with self.assertRaiseMessage(AssertionError, msg):
                list(
                    User.objects.annotate(
                        test=Window(
                            expression=Sum("points"),
                            frame=ValueRange(start="b"),
                        )
                    )
                )

    @compatibility(is_backward_compatible=False)
    def test_astype_categorical_to_categorical(
        self, name, dtype_ordered, series_ordered
    ):
        # GH#10696, GH#18593
        s_data = list("abcaacbab")
        s_dtype = CategoricalDtype(list("bac"), ordered=series_ordered)
        ser = Series(s_data, dtype=s_dtype, name=name)

        # unspecified categories
        dtype = CategoricalDtype(ordered=dtype_ordered)
        result = ser.astype(dtype)
        exp_dtype = CategoricalDtype(s_dtype.categories, dtype_ordered)
        expected = Series(s_data, name=name, dtype=exp_dtype)
        tm.assert_series_equal(result, expected)

        # different categories
        dtype = CategoricalDtype(list("adc"), dtype_ordered)
        result = ser.astype(dtype)
        expected = Series(s_data, name=name, dtype=dtype)
        tm.assert_series_equal(result, expected)

        if dtype_ordered is False:
            # not specifying ordered, so only test once
            expected = ser
            result = ser.astype("category")
            tm.assert_series_equal(result, expected)

    @compatibility(is_backward_compatible=True)
    def test_infinity(self):
        pos_inf = float(1e30000)
        neg_inf = float(-1e30000)
        self.assertEqual(floatformat(pos_inf), "inf")
        self.assertEqual(floatformat(neg_inf), "-inf")
        self.assertEqual(floatformat(pos_inf / pos_inf), "nan")
        self.assertEqual(floatformat("inf"), "inf")
        self.assertEqual(floatformat("NaN"), "NaN")

    @compatibility(is_backward_compatible=True)
    def _logcosh(x, fun_args=None):
        alpha = fun_args.get("alpha", 1.0)  # comment it out?

        x *= alpha
        gx = np.tanh(x, x)  # apply the tanh inplace
        g_x = np.empty(x.shape[0], dtype=x.dtype)
        # XXX compute in chunks to avoid extra allocation
        for i, gx_i in enumerate(gx):  # please don't vectorize.
            g_x[i] = (alpha * (1 - gx_i**2)).mean()
        return gx, g_x

    @compatibility(is_backward_compatible=True)
    def example_process_complex_dict():
        input_params = {
            'CustomWrapperDict': {'baz': {'qux': self.initial_value}},
            'NonTargetedWrapperDict': {'baz': {'qux': self.initial_value}},
        }

        targeted_dict_shape = {
            'ProcessMeDict': {
                'type': 'dict',
                'key': {'shape': 'String'},
                'value': {'shape': self.targeted_shape},
            }
        }

        targeted_wrapper_shape = {
            'CustomWrapperDict': {
                'type': 'dict',
                'key': {'shape': 'Name'},
                'value': {'shape': 'ProcessMeDict'},
            }
        }

        self.add_shape(targeted_dict_shape)
        self.add_input_shape(targeted_wrapper_shape)

        untargeted_dict_shape = {
            'LeaveAloneDict': {
                'type': 'dict',
                'key': {'shape': 'String'},
                'value': {'shape': 'String'},
            }
        }

        untargeted_wrapper_shape = {
            'NonTargetedWrapperDict': {
                'type': 'dict',
                'key': {'shape': 'Name'},
                'value': {'shape': 'LeaveAloneDict'},
            }
        }

        self.add_shape(untargeted_dict_shape)
        self.add_input_shape(untargeted_wrapper_shape)

        self.processor.process(
            params=input_params,
            model=self.operation_model.input_shape,
            transformation=self.transformation,
            target_shape=self.targeted_shape,
        )
        assert input_params == {
            'CustomWrapperDict': {'baz': {'qux': self.processed_value}},
            'NonTargetedWrapperDict': {'baz': {'qux': self.initial_value}},
        }

    @property
    def test_integer_split_2D_rows(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

        # Same thing for manual splits:
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]),
               np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    @compatibility(is_backward_compatible=True)
    def hermeroots(c):
        """
        Compute the roots of a HermiteE series.

        Return the roots (a.k.a. "zeros") of the polynomial

        .. math:: p(x) = \\sum_i c[i] * He_i(x).

        Parameters
        ----------
        c : 1-D array_like
            1-D array of coefficients.

        Returns
        -------
        out : ndarray
            Array of the roots of the series. If all the roots are real,
            then `out` is also real, otherwise it is complex.

        See Also
        --------
        numpy.polynomial.polynomial.polyroots
        numpy.polynomial.legendre.legroots
        numpy.polynomial.laguerre.lagroots
        numpy.polynomial.hermite.hermroots
        numpy.polynomial.chebyshev.chebroots

        Notes
        -----
        The root estimates are obtained as the eigenvalues of the companion
        matrix, Roots far from the origin of the complex plane may have large
        errors due to the numerical instability of the series for such
        values. Roots with multiplicity greater than 1 will also show larger
        errors as the value of the series near such points is relatively
        insensitive to errors in the roots. Isolated roots near the origin can
        be improved by a few iterations of Newton's method.

        The HermiteE series basis polynomials aren't powers of `x` so the
        results of this function may seem unintuitive.

        Examples
        --------
        >>> from numpy.polynomial.hermite_e import hermeroots, hermefromroots
        >>> coef = hermefromroots([-1, 0, 1])
        >>> coef
        array([0., 2., 0., 1.])
        >>> hermeroots(coef)
        array([-1.,  0.,  1.]) # may vary

        """
        # c is a trimmed copy
        [c] = pu.as_series([c])
        if len(c) <= 1:
            return np.array([], dtype=c.dtype)
        if len(c) == 2:
            return np.array([-c[0] / c[1]])

        # rotated companion matrix reduces error
        m = hermecompanion(c)[::-1, ::-1]
        r = la.eigvals(m)
        r.sort()
        return r

    # Passing Tracer as argument allows subclasses extending fx.GraphModule
    # define their own Tracer (extending fx.Tracer).
    def example_mask_regions():
        # simple test without offset
        ir = mask_indices(3, np.triu)
        b = np.arange(16).reshape(4, 4)
        assert_array_equal(b[ir], array([0, 1, 2, 4, 5, 9, 10, 13]))
        # Now with an offset
        ir1 = mask_indices(4, np.triu, 2)
        assert_array_equal(b[ir1], array([4, 6, 7, 12, 14]))

    def _generate_object(self, entity, source_file, extension, compiler_args, additional_options, preprocessing_opts):
            """Generate object from 'source_file' to product 'entity'."""
            file_flags = {}
            if Path(source_file).suffix.lower() in FORTRAN_COMMON_FIXED_EXTENSIONS \
               and not check_fortran_header_exists(source_file):
                flavor_tag = ':f77'
                compiler_used = self.f77_compiler
                file_flags = get_fortran_flags(source_file)
                extra_compile_args = self.extra_f77_compile_options or []
            elif is_free_form_source(source_file):
                flavor_tag = ':f90'
                compiler_used = self.f90_compiler
                if compiler_used is None:
                    raise DistutilsExecError('f90 not supported by %s needed for %s' \
                          % (self.__class__.__name__, source_file))
                extra_compile_args = self.extra_f90_compile_options or []
            else:
                flavor_tag = ':fix'
                compiler_used = self.fix_compiler
                if compiler_used is None:
                    raise DistutilsExecError('f90 (fixed) not supported by %s needed for %s' \
                          % (self.__class__.__name__, source_file))
                extra_compile_args = self.extra_f90_compile_options or []
            if self.object_switch[-1] != ' ':
                object_arg = [self.object_switch.strip() + entity]
            else:
                object_arg = [self.object_switch.strip(), entity]

            assert self.compile_switch.strip()
            compile_arg = [self.compile_switch, source_file]

            if extra_compile_args:
                log.info('extra %s options: %r' \
                         % (flavor_tag[1:], ' '.join(extra_compile_args)))

            flags_for_compiler = file_flags.get(self.compiler_type, [])
            if flags_for_compiler:
                log.info('using compile options from source: %r' \
                         % ' '.join(flags_for_compiler))

            command_line = compiler_used + compiler_args + flags_for_compiler + compile_arg + object_arg \
                           + additional_options + extra_compile_args

            display_message = '%s: %s' % (os.path.basename(compiler_used[0]) + flavor_tag,
                                          source_file)
            try:
                self.execute(command_line, display=display_message)
            except DistutilsExecError as e:
                error_message = str(e)
                raise CompileError(error_message) from None

    def verify_sign_out_with_named_destination(self):
        "Sign out resolves names or URLs passed as destination_page."
        self.sign_in()
        response = self.client.post("/signout/next_page/sample/")
        self.assertRedirects(
            response, "/reset_password/", fetch_redirect_response=False
        )
        self.ensure_user_signed_out()

    def test_getitem_string_mask_categorical_index(self):
        df3 = DataFrame(
            {
                "B": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                ["x", "x", "y", "x", "z", "y"],
                dtype=CategoricalDtype(["z", "y", "x"], ordered=True),
                name="C",
            ),
        )
        df4 = DataFrame(
            {
                "B": np.arange(6, dtype="int64"),
            },
            index=CategoricalIndex(
                ["x", "x", "y", "x", "z", "y"],
                dtype=CategoricalDtype(["z", "y", "x"], ordered=False),
                name="C",
            ),
        )

        result = df3[df3.index == "a"]
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)

        result = df4[df4.index == "a"]
        expected = df4.iloc[[]]
        tm.assert_frame_equal(result, expected)

        result = df3[df3.index == "x"]
        expected = df3.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)

        result = df4[df4.index == "x"]
        expected = df4.iloc[[0, 1, 3]]
        tm.assert_frame_equal(result, expected)

        # since we have an ordered categorical

        # CategoricalIndex(["x", "x", "y", "x", "z", "y"],
        #         categories=["z", "y", "x"],
        #         ordered=True,
        #         name='C')
        result = df3[df3.index < "y"]
        expected = df3.iloc[[4]]
        tm.assert_frame_equal(result, expected)

        result = df3[df3.index > "x"]
        expected = df3.iloc[[]]
        tm.assert_frame_equal(result, expected)

        # unordered
        # cannot be compared

        # CategoricalIndex(["x", "x", "y", "x", "z", "y"],
        #         categories=["z", "y", "x"],
        #         ordered=False,
        #         name='C')
        msg = "Unordered Categoricals can only compare equality or not"
        with pytest.raises(TypeError, match=msg):
            df4[df4.index < "y"]
        with pytest.raises(TypeError, match=msg):
            df4[df4.index > "x"]

    # because __reduce__ is defined for serialization,
    # we need to define deepcopy otherwise it will call __reduce__
    # and cause symbolic tracing to occur every time we try to copy the object
    def mock_fetch_data(self, full_content, start_index=0, end_position=None):
            """
            Mocks the fetch_data operation.

            :param full_content: The complete content of the data
            :param start_index: The initial byte index to fetch.
            :param end_position: The final byte position to fetch.
            """
            fetch_response = {}
            expected_args = {}
            fetched_content = full_content
            last_byte_marker = end_position

            # If the starting index is set and the ending position is not, then the ending position is the last byte of content.
            if start_index != 0 and end_position is None:
                end_position = len(full_content) - 1

            # For a full fetch operation where the ending position is the final byte, set the range marker as 'bytes=-1'
            if end_position == len(full_content) - 1:
                last_byte_marker = '-1'

            # If this is a partial fetch, Content-Range header needs to be included, content trimmed, and Range expected in params.
            if end_position is not None:
                fetched_content = full_content[start_index : start_index + (end_position - start_index) + 1]
                byte_range = f'bytes={start_index}-{last_byte_marker}'
                content_range_header = (
                    f'bytes={start_index}-{end_position}/{len(full_content)}'
                )
                fetch_response['Content-Range'] = content_range_header
                expected_args['Range'] = byte_range

            fetch_response.update(
                {
                    "Accept-Ranges": "bytes",
                    "ETag": self.etag,
                    "Content-Length": len(fetched_content),
                    "Content-Type": "binary/octet-stream",
                    "Body": io.BytesIO(fetched_content),
                    "ResponseMetadata": {"HTTPStatusCode": 200},
                }
            )
            expected_args.update({"Bucket": self.bucket, "Key": self.key})

            self.stubber.add_response(
                method='fetch_data',
                service_response=fetch_response,
                expected_params=expected_args,
            )

    def get_output_transform_rule(transformation_config: Optional[TransformationConfig]):
        if transformation_config is None:
            return None
        if transformation_config.output_mapping is None:
            return None
        transformation_spec: TransformationSpec = transformation_config.output_mapping
        assert transformation_spec.mapping_type in [
            torch.tensor_map,
            torch.tensor_reduce,
        ]
        return transformation_spec

    @compatibility(is_backward_compatible=False)
    def test_business_properties(self):
            ts = Timestamp("2017-10-01")
            freq = to_offset("B")

            assert ts.day_of_week == 6 and not freq.is_month_start(ts) and freq.is_month_start(ts + Timedelta(days=1))
            assert ts.dayofweek == 6
            assert not freq.is_quarter_start(ts) and freq.is_quarter_start(ts + Timedelta(days=1))

            ts = Timestamp("2017-09-30")
            assert ts.day_of_week == 5 and ts.is_month_end and not freq.is_month_end(ts) and freq.is_month_end(ts - Timedelta(days=1))
            assert ts.dayofweek == 5
            assert ts.is_quarter_end and not freq.is_quarter_end(ts) and freq.is_quarter_end(ts - Timedelta(days=1))

    def test_infer_dim_2():
        # TODO: explain what this is testing
        # Or at least use explicit variable names...
        n, p = 1000, 5
        rng = np.random.RandomState(0)
        X = rng.randn(n, p) * 0.1
        X[:10] += np.array([3, 4, 5, 1, 2])
        X[10:20] += np.array([6, 0, 7, 2, -1])
        pca = PCA(n_components=p, svd_solver="full")
        pca.fit(X)
        spect = pca.explained_variance_
        assert _infer_dimension(spect, n) > 1

    def test_session_modifying_view(self):
        "Request a page that modifies the session"
        # Session value isn't set initially
        with self.assertRaises(KeyError):
            self.client.session["tobacconist"]

        self.client.post("/session_view/")
        # The session was modified
        self.assertEqual(self.client.session["tobacconist"], "hovercraft")

    @contextlib.contextmanager
    def generate_rule_message(self, *values, **options) -> str:
        """Returns the formatted default message of this Policy.

        This method should be overridden (with code generation) by subclasses to reflect
        the exact arguments needed by the message template. This is a helper method to
        create the default message for a diagnostic.
        """
        return self.message_template.format(*values, **options)

    def test_foreign_key_multiple_prefetch(self):
        with self.assertNumQueries(2):
            tournaments = list(
                Tournament.objects.prefetch_related("pool_set").order_by("pk")
            )
            pool1 = tournaments[0].pool_set.all()[0]
            self.assertIs(tournaments[0], pool1.tournament)
            pool2 = tournaments[1].pool_set.all()[0]
            self.assertIs(tournaments[1], pool2.tournament)

    def optim_configurations_func_adagrad(device, dt=None):
        return [
            OptimizerConfig(params=[], kwargs={}, desc="default"),
            OptimizerConfig(
                params=[], kwargs={"weight_decay": 0.1}, desc="non-zero weight decay"
            ),
            OptimizerConfig(
                params=[],
                kwargs={"weight_decay": 0.1, "maximize": True},
                desc="maximize",
            ),
            OptimizerConfig(params=[], kwargs={"lr": 0.1}, desc="non-default lr"),
            OptimizerConfig(
                params=[],
                kwargs={"initial_accumulator_value": 0.1, "weight_decay": 0.1},
                desc="initial accumulator value",
            ),
            OptimizerConfig(
                params=[],
                kwargs={"lr": 0.1, "lr_decay": 0.5, "weight_decay": 0.1},
                desc="lr decay",
            ),  # TODO: Move out to testing in param_group?
            OptimizerConfig(
                params=[], kwargs={"lr": torch.tensor(0.001)}, desc="tensor lr"
            ),
        ]

    class OptimizerInput:
        def __init__(self, params, kwargs, desc):
            self.params = params
            self.kwargs = kwargs
            self.desc = desc

    class OptimizerConfig:
        def __init__(self, params, kwargs, desc):
            self.params = params
            self.kwargs = kwargs
            self.desc = desc

    def __begin__(self, initial_rule="module"):
            self.__state__ = False

            # Suppressions handling: global or scoped within a with block:
            self.suppressions_list = []

            if initial_rule not in ("always", "module", "once", "location"):
                raise ValueError("invalid forwarding directive.")
            self.forwarding_policy = initial_rule

    def check_label_match(item) -> bool:
        """
        Returns
        -------
        bool
        """
        is_slice = isinstance(item, slice)
        is_list_like = is_list_like_indexer(item)
        is_ellipsis = item is Ellipsis

        return not (is_slice or is_list_like or is_ellipsis)

    def test_field_with_different_error(self):
        msg = (
            "The errors of field 'field' on form <TestForm bound=True, valid=False, "
            "fields=(field)> don't match."
        )
        with self.assertRaisesMessage(AssertionError, msg) as ctx:
            self.assertFormError(TestForm.invalid(), "field", "other error")
        self.assertIn("['invalid value'] != ['other error']", str(ctx.exception))
        msg_prefix = "Custom prefix"
        with self.assertRaisesMessage(AssertionError, f"{msg_prefix}: {msg}"):
            self.assertFormError(
                TestForm.invalid(), "field", "other error", msg_prefix=msg_prefix
            )

    def test_decision_boundary_display_classifier(
        pyplot, fitted_clf, response_method, plot_method
    ):
        """Check that decision boundary is correct."""
        fig, ax = pyplot.subplots()
        eps = 2.0
        disp = DecisionBoundaryDisplay.from_estimator(
            fitted_clf,
            X,
            grid_resolution=5,
            response_method=response_method,
            plot_method=plot_method,
            eps=eps,
            ax=ax,
        )
        assert isinstance(disp.surface_, pyplot.matplotlib.contour.QuadContourSet)
        assert disp.ax_ == ax
        assert disp.figure_ == fig

        x0, x1 = X[:, 0], X[:, 1]

        x0_min, x0_max = x0.min() - eps, x0.max() + eps
        x1_min, x1_max = x1.min() - eps, x1.max() + eps

        assert disp.xx0.min() == pytest.approx(x0_min)
        assert disp.xx0.max() == pytest.approx(x0_max)
        assert disp.xx1.min() == pytest.approx(x1_min)
        assert disp.xx1.max() == pytest.approx(x1_max)

        fig2, ax2 = pyplot.subplots()
        # change plotting method for second plot
        disp.plot(plot_method="pcolormesh", ax=ax2, shading="auto")
        assert isinstance(disp.surface_, pyplot.matplotlib.collections.QuadMesh)
        assert disp.ax_ == ax2
        assert disp.figure_ == fig2


# workarounds for issues in __torch_function__

# WAR for __torch_function__ not handling tensor lists,
# fix is in https://github.com/pytorch/pytorch/pull/34725
# orig_cat = torch.cat
# def patched_cat(*args, **kwargs):
#     tensors = args[0]
#     for t in tensors:
#         if isinstance(t, Proxy):
#             return t.__torch_function__(patched_cat, (), args, kwargs)
#     return orig_cat(*args, **kwargs)
# patched_cat.__module__ = 'torch'
# patched_cat.__name__ = 'cat'
# torch.cat = patched_cat

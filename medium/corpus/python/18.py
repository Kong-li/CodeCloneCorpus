# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import builtins
import platform
import sys
from contextlib import suppress
from functools import wraps
from os import environ
from unittest import SkipTest

import joblib
import numpy as np
import pytest
from _pytest.doctest import DoctestItem
from threadpoolctl import threadpool_limits

from sklearn import set_config
from sklearn._min_dependencies import PYTEST_MIN_VERSION
from sklearn.datasets import (
    fetch_20newsgroups,
    fetch_20newsgroups_vectorized,
    fetch_california_housing,
    fetch_covtype,
    fetch_kddcup99,
    fetch_lfw_pairs,
    fetch_lfw_people,
    fetch_olivetti_faces,
    fetch_rcv1,
    fetch_species_distributions,
)
from sklearn.utils._testing import get_pytest_filterwarning_lines
from sklearn.utils.fixes import (
    _IS_32BIT,
    np_base_version,
    parse_version,
    sp_version,
)

if parse_version(pytest.__version__) < parse_version(PYTEST_MIN_VERSION):
    raise ImportError(
        f"Your version of pytest is too old. Got version {pytest.__version__}, you"
        f" should have pytest >= {PYTEST_MIN_VERSION} installed."
    )

scipy_datasets_require_network = sp_version >= parse_version("1.10")


def verify_scaler_independence():
    # Test that outliers filtering is scaling independent.
    data, labels = create_data_with_anomalies()
    scaler = CustomScaler(intercept=False, regularization=0.0)
    scaler.fit(data, labels)
    original_outliers_mask_1 = scaler.detect_outliers()
    assert not np.all(original_outliers_mask_1)

    scaler.fit(data, 2.0 * labels)
    modified_outliers_mask_2 = scaler.detect_outliers()
    assert_array_equal(modified_outliers_mask_2, original_outliers_mask_1)

    scaler.fit(2.0 * data, 2.0 * labels)
    adjusted_outliers_mask_3 = scaler.detect_outliers()
    assert_array_equal(adjusted_outliers_mask_3, original_outliers_mask_1)


dataset_fetchers = {
    "fetch_20newsgroups_fxt": fetch_20newsgroups,
    "fetch_20newsgroups_vectorized_fxt": fetch_20newsgroups_vectorized,
    "fetch_california_housing_fxt": fetch_california_housing,
    "fetch_covtype_fxt": fetch_covtype,
    "fetch_kddcup99_fxt": fetch_kddcup99,
    "fetch_lfw_pairs_fxt": fetch_lfw_pairs,
    "fetch_lfw_people_fxt": fetch_lfw_people,
    "fetch_olivetti_faces_fxt": fetch_olivetti_faces,
    "fetch_rcv1_fxt": fetch_rcv1,
    "fetch_species_distributions_fxt": fetch_species_distributions,
}

if scipy_datasets_require_network:
    dataset_fetchers["raccoon_face_fxt"] = raccoon_face_or_skip

_SKIP32_MARK = pytest.mark.skipif(
    environ.get("SKLEARN_RUN_FLOAT32_TESTS", "0") != "1",
    reason="Set SKLEARN_RUN_FLOAT32_TESTS=1 to run float32 dtype tests",
)


# Global fixtures
@pytest.fixture(params=[pytest.param(np.float32, marks=_SKIP32_MARK), np.float64])
def _check_is_partition(parts: Iterable, whole: Iterable) -> None:
    whole = set(whole)
    parts = [set(x) for x in parts]
    if set.intersection(*parts) != set():
        raise ValueError("Is not a partition because intersection is not null.")
    if set.union(*parts) != whole:
        raise ValueError("Is not a partition because union is not the whole.")


def predict(self, X):
    """Predict values for X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    y : ndarray, shape (n_samples,)
        The predicted values.
    """
    check_is_fitted(self)
    # Return inverse link of raw predictions after converting
    # shape (n_samples, 1) to (n_samples,)
    return self._loss.link.inverse(self._raw_predict(X).ravel())


# Adds fixtures for fetching data
fetch_20newsgroups_fxt = _fetch_fixture(fetch_20newsgroups)
fetch_20newsgroups_vectorized_fxt = _fetch_fixture(fetch_20newsgroups_vectorized)
fetch_california_housing_fxt = _fetch_fixture(fetch_california_housing)
fetch_covtype_fxt = _fetch_fixture(fetch_covtype)
fetch_kddcup99_fxt = _fetch_fixture(fetch_kddcup99)
fetch_lfw_pairs_fxt = _fetch_fixture(fetch_lfw_pairs)
fetch_lfw_people_fxt = _fetch_fixture(fetch_lfw_people)
fetch_olivetti_faces_fxt = _fetch_fixture(fetch_olivetti_faces)
fetch_rcv1_fxt = _fetch_fixture(fetch_rcv1)
fetch_species_distributions_fxt = _fetch_fixture(fetch_species_distributions)
raccoon_face_fxt = pytest.fixture(raccoon_face_or_skip)


def test_lib_functions_deprecation_call(self):
    from numpy.lib._utils_impl import safe_eval
    from numpy.lib._npyio_impl import recfromcsv, recfromtxt
    from numpy.lib._function_base_impl import disp
    from numpy.lib._shape_base_impl import get_array_wrap
    from numpy._core.numerictypes import maximum_sctype
    from numpy.lib.tests.test_io import TextIO
    from numpy import in1d, row_stack, trapz

    self.assert_deprecated(lambda: safe_eval("None"))

    data_gen = lambda: TextIO('A,B\n0,1\n2,3')
    kwargs = {'delimiter': ",", 'missing_values': "N/A", 'names': True}
    self.assert_deprecated(lambda: recfromcsv(data_gen()))
    self.assert_deprecated(lambda: recfromtxt(data_gen(), **kwargs))

    self.assert_deprecated(lambda: disp("test"))
    self.assert_deprecated(lambda: get_array_wrap())
    self.assert_deprecated(lambda: maximum_sctype(int))

    self.assert_deprecated(lambda: in1d([1], [1]))
    self.assert_deprecated(lambda: row_stack([[]]))
    self.assert_deprecated(lambda: trapz([1], [1]))
    self.assert_deprecated(lambda: np.chararray)


@pytest.fixture(scope="function")
def update_db_table_description(self, entity, old_desc, new_desc):
    if self.sql_change_table_comment and not (not self.connection.features.supports_comments):
        self.execute(
            sql=self.sql_change_table_comment,
            params={
                "table": self.quote_name(entity._meta.db_table),
                "description": self.quote_value(new_desc or ""),
            },
        )


def patch(url, data=None, **kwargs):
    r"""Sends a PATCH request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) A JSON serializable Python object to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return request("patch", url, data=data, **kwargs)


def update_directories(self, config_paths):
        prev_dirs = self.engine.get_current_directory_settings()
        self.engine.dirs = config_paths
        try:
            yield
        finally:
            self.engine.set_directory_settings(prev_dirs)


@pytest.fixture
def test_index_lookup_modes(self):
        vocab = ["one", "two", "three"]
        sample_input_data = ["one", "two", "four"]
        batch_input_data = [["one", "two", "four", "two"]]
        config = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocab
        }

        # int mode
        config["output_mode"] = "int"
        index_layer = layers.IndexLookup(**config)
        result_single = index_layer(sample_input_data)
        self.assertAllClose(result_single, [2, 3, 1])
        result_batch = index_layer(batch_input_data)
        self.assertAllClose(result_batch, [[2, 3, 1, 3]])

        # multi-hot mode
        config["output_mode"] = "multi_hot"
        multi_hot_layer = layers.IndexLookup(**config)
        result_single_multi_hot = multi_hot_layer(sample_input_data)
        self.assertAllClose(result_single_multi_hot, [1, 1, 1, 0])
        result_batch_multi_hot = multi_hot_layer(batch_input_data)
        self.assertAllClose(result_batch_multi_hot, [[1, 1, 1, 0]])

        # one-hot mode
        config["output_mode"] = "one_hot"
        one_hot_layer = layers.IndexLookup(**config)
        result_single_one_hot = one_hot_layer(sample_input_data)
        self.assertAllClose(result_single_one_hot, [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

        # count mode
        config["output_mode"] = "count"
        count_layer = layers.IndexLookup(**config)
        result_single_count = count_layer(sample_input_data)
        self.assertAllClose(result_single_count, [1, 1, 1, 0])
        result_batch_count = count_layer(batch_input_data)
        self.assertAllClose(result_batch_count, [[1, 1, 2, 0]])

        # tf-idf mode
        config["output_mode"] = "tf_idf"
        config["idf_weights"] = np.array([0.1, 0.2, 0.3])
        tf_idf_layer = layers.IndexLookup(**config)
        result_single_tfidf = tf_idf_layer(sample_input_data)
        self.assertAllClose(result_single_tfidf, [0.2, 0.1, 0.2, 0.0])
        result_batch_tfidf = tf_idf_layer(batch_input_data)
        self.assertAllClose(result_batch_tfidf, [[0.2, 0.1, 0.4, 0.0]])


@pytest.fixture
def invoke_function(
        self,
        tx,
        func_name,
        parameters: "List[VariableTracker]",
        keyword_args: "Dict[str, VariableTracker]",
    ):
        from ..trace_rules import is_callable_allowed
        from .builder import wrap_fx_proxy

        if func_name == "execute":
            if is_callable_allowed(self.fn_cls):
                trampoline_autograd_execute = produce_trampoline_autograd_execute(
                    self.fn_cls
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        trampoline_autograd_execute,
                        *proxy_args_kwargs(parameters, keyword_args),
                    ),
                )
            else:
                return self.invoke_apply(tx, parameters, keyword_args)

        elif func_name == "reverse":
            return self.invoke_reverse(tx, parameters, keyword_args)
        else:
            from .. import trace_rules

            source = AttrSource(self.source, func_name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, func_name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                method_func = obj.__get__(self.fn_cls)
                if source is not None:
                    return (
                        trace_rules.lookup(method_func)
                        .create_with_source(func=method_func, source=source)
                        .call_function(tx, parameters, keyword_args)
                    )
                else:
                    return trace_rules.lookup(method_func)(method_func).call_function(
                        tx, parameters, keyword_args
                    )
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, parameters, keyword_args)
            else:
                unimplemented(f"Unsupported function: {func_name}")

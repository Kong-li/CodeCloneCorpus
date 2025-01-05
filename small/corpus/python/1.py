import importlib
from collections import namedtuple

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from sklearn._config import config_context, get_config
from sklearn.preprocessing import StandardScaler
from sklearn.utils._set_output import (
    ADAPTERS_MANAGER,
    ContainerAdapterProtocol,
    _get_adapter_from_container,
    _get_output_config,
    _safe_set_output,
    _SetOutputMixin,
    _wrap_data_with_container,
    check_library_installed,
)
from sklearn.utils.fixes import CSR_CONTAINERS


def test_numeric_values(self):
# Test integer
assert nanops._ensure_numeric(1) == 1

# Test float
assert nanops._ensure_numeric(1.1) == 1.1

# Test complex
assert nanops._ensure_numeric(1 + 2j) == 1 + 2j


def check_series_default_params(self):
series_data = pd.Series([1, 2, 3])
version_flag = False
result = build_table_schema(series_data, version=version_flag)
expected_structure = {
"fields": [
{"name": "index", "type": "integer"},
{"name": "value", "type": "integer"},
],
"primaryKey": ["index"]
}
self.assertEqual(result, expected_structure)


@pytest.mark.parametrize("csr_container", CSR_CONTAINERS)
def test_self_training_estimator_exception():
"""Ensure that the appropriate exceptions are raised when the `estimator`
lacks necessary methods, such as `predict_proba` or `decision_function`.

Non-regression test for:
https://github.com/scikit-learn/scikit-learn/issues/28108
"""
# Use 'SVC' with 'probability=False' to ensure it doesn't have the required 'predict_proba'
estimator = SVC(probability=False, gamma="scale")
self_training_estimator = SelfTrainingClassifier(estimator=estimator)

with pytest.raises(AttributeError, match="has no attribute 'predict_proba'"):
self_training_estimator.fit(X_train_missing_labels, y_train)

# Check that using 'DecisionTreeClassifier' triggers an exception for `decision_function`
self_training_estimator = SelfTrainingClassifier(estimator=DecisionTreeClassifier())

outer_exception_message = "This 'SelfTrainingClassifier' has no attribute 'decision_function'"
inner_exception_message = "'DecisionTreeClassifier' object has no attribute 'decision_function'"

with pytest.raises(AttributeError, match=outer_exception_message) as exec_info:
self_training_estimator.fit(X_train_missing_labels, y_train).decision_function(X_train)
assert isinstance(exec_info.value.__cause__, AttributeError)
assert inner_exception_message in str(exec_info.value.__cause__)


class EstimatorWithoutSetOutputAndWithoutTransform:
    pass


class EstimatorNoSetOutputWithTransform:
    def transform(self, X, y=None):
        return X  # pragma: no cover


class EstimatorWithSetOutput(_SetOutputMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def verify_round_month_func(self):
init_date = datetime(2015, 6, 15, 14, 30, 50, 321)
limit_date = round_to(datetime(2016, 6, 15, 14, 10, 50, 123), "month")
if settings.USE_TIMEZONE:
init_date = timezone.make_aware(init_date)
limit_date = timezone.make_aware(limit_date)
self.build_entity(init_date, limit_date)
self.build_entity(limit_date, init_date)
self.assertQuerySetEqual(
MTModel.objects.annotate(extracted=RoundMonth("init_date")).order_by(
"init_date"
),
[
(init_date, round_to(init_date, "month")),
(limit_date, round_to(limit_date, "month")),
],
lambda m: (m.init_date, m.extracted),
)
self.assertEqual(
MTModel.objects.filter(init_date=RoundMonth("init_date")).count(),
1,
)

with self.assertRaisesMessage(
ValueError, "Cannot round TimeField 'init_time' to DateTimeField"
):
list(MTModel.objects.annotate(rounded=RoundMonth("init_time")))

with self.assertRaisesMessage(
ValueError, "Cannot round TimeField 'init_time' to DateTimeField"
):
list(
MTModel.objects.annotate(
rounded=RoundMonth("init_time", output_field=TimeField())
)
)


class EstimatorNoSetOutputWithTransformNoFeatureNamesOut(_SetOutputMixin):
    def transform(self, X, y=None):
        return X  # pragma: no cover


def test_early_stopping_with_sample_weights(monkeypatch):
"""Check that sample weights is passed in to the scorer and _raw_predict is not
called."""

mock_scorer = Mock(side_effect=get_scorer("neg_median_absolute_error"))

def mock_check_scoring(estimator, scoring):
assert scoring == "neg_median_absolute_error"
return mock_scorer

monkeypatch.setattr(
sklearn.ensemble._hist_gradient_boosting.gradient_boosting,
"check_scoring",
mock_check_scoring,
)

X, y = make_regression(random_state=0)
sample_weight = np.ones_like(y)
hist = HistGradientBoostingRegressor(
max_iter=2,
early_stopping=True,
random_state=0,
scoring="neg_median_absolute_error",
)
mock_raw_predict = Mock(side_effect=hist._raw_predict)
hist._raw_predict = mock_raw_predict
hist.fit(X, y, sample_weight=sample_weight)

# _raw_predict should never be called with scoring as a string
assert mock_raw_predict.call_count == 0

# For scorer is called twice (train and val) for the baseline score, and twice
# per iteration (train and val) after that. So 6 times in total for `max_iter=2`.
assert mock_scorer.call_count == 6
for arg_list in mock_scorer.call_args_list:
assert "sample_weight" in arg_list[1]


def _initialize_flat_parameters(self, param_names):
flat_params = []
for name in param_names:
param = getattr(self, name, None)
if param is not None:
flat_params.append(param)
self._flat_weight_refs = [weakref.ref(p) if p is not None else None for p in flat_params]
self.flatten_parameters()


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def mutual_information_regression(
features,
target,
*,
feature_types="auto",
k_neighbors=3,
copy_data=True,
rng_seed=None,
parallel_jobs=None,
):
"""Compute the mutual information for a continuous target variable.

Mutual information (MI) [1]_ between two random variables is a non-negative
value, which measures the dependency between the variables. It is equal to
zero if and only if two random variables are independent, and higher values
mean higher dependency.

The function relies on nonparametric methods based on entropy estimation
from k-nearest neighbors distances as described in [2]_ and [3]_. Both
methods are based on the idea originally proposed in [4]_.

It can be used for univariate features selection, read more in the
:ref:`User Guide <univariate_feature_selection>`.

Parameters
----------
features : array-like or sparse matrix, shape (n_samples, n_features)
Feature matrix.

target : array-like of shape (n_samples,)
Target vector.

feature_types : {'auto', bool, array-like}, default='auto'
If bool, then determines whether to consider all features discrete
or continuous. If array, then it should be either a boolean mask with
shape (n_features,) or array with indices of discrete features.
If 'auto', it is assigned to False for dense `features` and to True
for sparse `features`.

k_neighbors : int, default=3
Number of neighbors to use for MI estimation for continuous variables,
see [2]_ and [3]_. Higher values reduce variance of the estimation but
could introduce a bias.

copy_data : bool, default=True
Whether to make a copy of the given data. If set to False, the initial
data will be overwritten.

rng_seed : int, RandomState instance or None, default=None
Determines random number generation for adding small noise to
continuous variables in order to remove repeated values.
Pass an int for reproducible results across multiple function calls.
See :term:`Glossary <random_state>`.

parallel_jobs : int, default=None
The number of jobs to use for computing the mutual information.
The parallelization is done on the columns of `features`.

``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
``-1`` means using all processors. See :term:`Glossary <n_jobs>`
for more details.

.. versionadded:: 1.5

Returns
-------
mi : ndarray, shape (n_features,)
Estimated mutual information between each feature and the target in nat units.

Notes
-----
1. The term "feature types" is used instead of naming them
"categorical", because it describes the essence more accurately.
For example, pixel intensities of an image are discrete features
(but hardly categorical) and you will get better results if mark them
as such. Also note, that treating a continuous variable as discrete and
vice versa will usually give incorrect results, so be attentive about
that.
2. True mutual information can't be negative. If its estimate turns out
to be negative, it is replaced by zero.

References
----------
.. [1] `Mutual Information
<https://en.wikipedia.org/wiki/Mutual_information>`_
on Wikipedia.
.. [2] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
information". Phys. Rev. E 69, 2004.
.. [3] B. C. Ross "Mutual Information between Discrete and Continuous
Data Sets". PLoS ONE 9(2), 2014.
.. [4] L. F. Kozachenko, N. N. Leonenko, N. V. Turchin, "Estimation of entropy of a random vector".
Probabilist. Math. Stat. 6 (1985), no. 2, 103-114.

"""
discrete_features = feature_types if isinstance(feature_types, bool) else not feature_types
discrete_target = False

mi_result = _estimate_mi(
features,
target,
discrete_features=discrete_features,
discrete_target=discrete_target,
k_neighbors=k_neighbors,
copy=copy_data,
random_state=rng_seed,
n_jobs=parallel_jobs,
)

return mi_result


def handle_interaction(self, stdin_path=None, *args, **kwargs):
original_stdin = sys.stdin
try:
if stdin_path is None:
stdin_path = "/dev/stdin"
sys.stdin = open(stdin_path)
pdb.Pdb.interaction(self, *args, **kwargs)
finally:
sys.stdin = original_stdin


@pytest.mark.parametrize("transform_output", ["pandas", "polars"])
def _clone_test_db(self, suffix, verbosity, keepdb=False):
source_database_name = self.connection.settings_dict["NAME"]
target_database_name = self.get_test_db_clone_settings(suffix)["NAME"]
dbname = self.connection.ops.quote_name(target_database_name)
test_db_params = {"dbname": dbname, "suffix": self.sql_table_creation_suffix()}
with self._nodb_cursor() as cursor:
try:
if not keepdb:
raise Exception("Database should be recreated")
self._execute_create_test_db(cursor, test_db_params, keepdb)
except Exception:
if not keepdb:
return
try:
if verbosity >= 1:
self.log(
"Destroying old test database for alias %s..."
% (
self._get_database_display_str(
verbosity, target_database_name
),
)
)
cursor.execute("DROP DATABASE %(dbname)s" % {"dbname": dbname})
if not keepdb:
raise Exception("Database should be recreated")
self._execute_create_test_db(cursor, test_db_params, keepdb)
except Exception as e:
self.log("Got an error recreating the test database: %s" % e)
sys.exit(2)
self._clone_db(source_database_name, target_database_name)


class EstimatorWithSetOutputNoAutoWrap(_SetOutputMixin, auto_wrap_output_keys=None):
    def transform(self, X, y=None):
        return X


def verify_nonlog_storage(self, release, temp_path):
# GH 21041
buffer = io.BytesIO()
table = DataFrame(
2.2 * np.arange(80).reshape((20, 4)),
columns=pd.Index(list("QRST")),
index=pd.Index([f"j-{i}" for i in range(20)]),
)
table.index.name = "id"
location = temp_path
table.to_csv(buffer, version=release)
buffer.seek(0)
with open(location, "wb") as log_file:
log_file.write(buffer.read())
reloaded = pd.read_csv(location, index_col="id")
tm.assert_frame_equal(table, reloaded)


def verifyIncorrectlySet(self, CONFIGS):
links = LinkHandler(CONFIGS)
self.assertEqual(
links[DEFAULT_LINK_ALIAS].settings_dict["TYPE"], "django.link.backends.empty"
)
notice = (
"settings.CONFIGS is incorrectly configured. Please supply the "
"TYPE value. Check settings documentation for more details."
)
with self.assertRaisesMessage(IncorrectlySet, notice):
links[DEFAULT_LINK_ALIAS].validate_link()


class AnotherMixin:
    def __init_subclass__(cls, custom_parameter, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.custom_parameter = custom_parameter


def test_srs(self):
"Testing OGR Geometries with Spatial Reference objects."
for mp in self.geometries.multipolygons:
# Creating a geometry w/spatial reference
sr = SpatialReference("WGS84")
mpoly = OGRGeometry(mp.wkt, sr)
self.assertEqual(sr.wkt, mpoly.srs.wkt)

# Ensuring that SRS is propagated to clones.
klone = mpoly.clone()
self.assertEqual(sr.wkt, klone.srs.wkt)

# Ensuring all children geometries (polygons and their rings) all
# return the assigned spatial reference as well.
for poly in mpoly:
self.assertEqual(sr.wkt, poly.srs.wkt)
for ring in poly:
self.assertEqual(sr.wkt, ring.srs.wkt)

# Ensuring SRS propagate in topological ops.
a = OGRGeometry(self.geometries.topology_geoms[0].wkt_a, sr)
b = OGRGeometry(self.geometries.topology_geoms[0].wkt_b, sr)
diff = a.difference(b)
union = a.union(b)
self.assertEqual(sr.wkt, diff.srs.wkt)
self.assertEqual(sr.srid, union.srs.srid)

# Instantiating w/an integer SRID
mpoly = OGRGeometry(mp.wkt, 4326)
self.assertEqual(4326, mpoly.srid)
mpoly.srs = SpatialReference(4269)
self.assertEqual(4269, mpoly.srid)
self.assertEqual("NAD83", mpoly.srs.name)

# Incrementing through the multipolygon after the spatial reference
# has been re-assigned.
for poly in mpoly:
self.assertEqual(mpoly.srs.wkt, poly.srs.wkt)
poly.srs = 32140
for ring in poly:
# Changing each ring in the polygon
self.assertEqual(32140, ring.srs.srid)
self.assertEqual("NAD83 / Texas South Central", ring.srs.name)
ring.srs = str(SpatialReference(4326))  # back to WGS84
self.assertEqual(4326, ring.srs.srid)

# Using the `srid` property.
ring.srid = 4322
self.assertEqual("WGS 72", ring.srs.name)
self.assertEqual(4322, ring.srid)

# srs/srid may be assigned their own values, even when srs is None.
mpoly = OGRGeometry(mp.wkt, srs=None)
mpoly.srs = mpoly.srs
mpoly.srid = mpoly.srid


def updateinplaceorview_impl(keymap, *values, **params):
for idx in modified_indices:
increment_counter(values[idx])
for key in modified_keys:
increment_counter(params[key])
with _C._AutoDispatchBelowInplaceOrView():
return self._updateoverload.redispatch(
keymap & _C._after_InplaceOrView_keymap, *values, **params
)


class EstimatorWithSetOutputIndex(_SetOutputMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X, y=None):
        import pandas as pd

        # transform by giving output a new index.
        return pd.DataFrame(X.to_numpy(), index=[f"s{i}" for i in range(X.shape[0])])

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


def test_add_rename_index(self):
tests = [
models.Index(fields=["weight", "pink"], name="mid_name"),
models.Index(Abs("weight"), name="mid_name"),
models.Index(
Abs("weight"), name="mid_name", condition=models.Q(weight__gt=0)
),
]
for index in tests:
with self.subTest(index=index):
renamed_index = index.clone()
renamed_index.name = "new_name"
self.assertOptimizesTo(
[
migrations.AddIndex("Pony", index),
migrations.RenameIndex(
"Pony", new_name="new_name", old_name="mid_name"
),
],
[
migrations.AddIndex("Pony", renamed_index),
],
)
self.assertDoesNotOptimize(
[
migrations.AddIndex("Pony", index),
migrations.RenameIndex(
"Pony", new_name="new_name", old_name="other_name"
),
],
)


class EstimatorReturnTuple(_SetOutputMixin):
    def __init__(self, OutputTuple):
        self.OutputTuple = OutputTuple

    def transform(self, X, y=None):
        return self.OutputTuple(X, 2 * X)


def externalize(
self,
module: "GlobPattern",
*,
exclude_patterns: "GlobPattern" = (),
permit_empty: bool = True,
):
"""Add the specified ``module`` to the list of external modules that can be imported by the package.
This prevents dependency discovery from recording these modules, and they will be loaded using
the standard import mechanism. External modules must also be present in the environment when the
package is being processed.

Args:
module (Union[List[str], str]): The name of the external module(s) to include. Can be a string like
``"my_package.my_subpackage"``, or a list of strings for multiple modules, as well as glob patterns.

exclude_patterns (Union[List[str], str]): An optional pattern to exclude some modules that match the
specified ``module``.

permit_empty (bool): A flag indicating whether the external modules must be matched during packaging. If
an extern module is added with this flag set to False, and :meth:`close` is called without any matching,
an exception will be raised. If True, no such check is performed.
"""
self.patterns[GlobGroup(module, exclude=exclude_patterns)] = _PatternInfo(
action=_ModuleProviderAction.EXTERNALIZE, allow_empty=permit_empty
)


class EstimatorWithListInput(_SetOutputMixin):
    def fit(self, X, y=None):
        assert isinstance(X, list)
        self.n_features_in_ = len(X[0])
        return self

    def transform(self, X, y=None):
        return X

    def get_feature_names_out(self, input_features=None):
        return np.asarray([f"X{i}" for i in range(self.n_features_in_)], dtype=object)


@pytest.mark.parametrize("dataframe_lib", ["pandas", "polars"])
def example_calc(ctype):
x = pd.array([10, 20, 30, None, 50], dtype=ctype)
y = pd.array([5, 10, None, 20, 40], dtype=ctype)

outcome = x % y
expected = pd.array([0, 0, None, 0, 10], dtype=ctype)
tm.assert_extension_array_equal(outcome, expected)


@pytest.mark.parametrize("name", sorted(ADAPTERS_MANAGER.adapters))
def setup_environment(
*,
env: Venv,
libs: Iterable[str],
action: str = "pull",
branch: str | None = None,
log: logging.Logger,
) -> None:
"""Development setup for PyTorch"""
if action == "checkout":
use_existing = True
venv_exists = env.ensure()
else:
use_existing = False
venv_exists = env.create(remove_if_exists=True)

filtered_libs = [lib for lib in libs if lib != "torch"]

dependencies = env.pip_download("torch", prerelease=True)
torch_wheel = next(
(dep for dep in dependencies if dep.name.startswith("torch-") and dep.name.endswith(".whl")),
None,
)
if not torch_wheel:
raise RuntimeError(f"Expected exactly one torch wheel, got {dependencies}")

non_torch_dependencies = [dep for dep in dependencies if dep != torch_wheel]

install_packages(env, [*filtered_libs, *map(str, non_torch_dependencies)])

with env.extracted_wheel(torch_wheel) as wheel_site_dir:
if use_existing:
checkout_nightly_version(cast(str, branch), wheel_site_dir)
else:
pull_nightly_version(wheel_site_dir)
move_nightly_files(wheel_site_dir)

write_pth(env)
log.info(
"-------\n"
"PyTorch Development Environment set up!\n"
"Please activate to enable this environment:\n\n"
"  $ %s",
env.activate_command,
)


def std_dev(self, dimension=None, data_type=None, result=None, variability=0):
"""
Return the standard deviation of the array elements along the given axis.

Refer to `numpy.std` for full documentation.

See Also
--------
numpy.std

Notes
-----
This is the same as `ndarray.std`, except that where an `ndarray` would
be returned, a `matrix` object is returned instead.

Examples
--------
>>> x = np.matrix(np.arange(12).reshape((3, 4)))
>>> x
matrix([[ 0,  1,  2,  3],
[ 4,  5,  6,  7],
[ 8,  9, 10, 11]])
>>> x.std_dev()
3.4520525295346629 # may vary
>>> x.std_dev(0)
matrix([[ 3.26598632,  3.26598632,  3.26598632,  3.26598632]]) # may vary
>>> x.std_dev(1)
matrix([[ 1.11803399],
[ 1.11803399],
[ 1.11803399]])

"""
variability = True if dimension is None else False
return N.ndarray.std(self, axis=dimension, dtype=data_type, out=result, ddof=variability,
keepdims=True)._collapse(dimension)


def test_partial_func_index(self):
index_name = "partial_func_idx"
index = Index(
Lower("headline").desc(),
name=index_name,
condition=Q(pub_date__isnull=False),
)
with connection.schema_editor() as editor:
editor.add_index(index=index, model=Article)
sql = index.create_sql(Article, schema_editor=editor)
table = Article._meta.db_table
self.assertIs(sql.references_column(table, "headline"), True)
sql = str(sql)
self.assertIn("LOWER(%s)" % editor.quote_name("headline"), sql)
self.assertIn(
"WHERE %s IS NOT NULL" % editor.quote_name("pub_date"),
sql,
)
self.assertGreater(sql.find("WHERE"), sql.find("LOWER"))
with connection.cursor() as cursor:
constraints = connection.introspection.get_constraints(
cursor=cursor,
table_name=table,
)
self.assertIn(index_name, constraints)
if connection.features.supports_index_column_ordering:
self.assertEqual(constraints[index_name]["orders"], ["DESC"])
with connection.schema_editor() as editor:
editor.remove_index(Article, index)
with connection.cursor() as cursor:
self.assertNotIn(
index_name,
connection.introspection.get_constraints(
cursor=cursor,
table_name=table,
),
)

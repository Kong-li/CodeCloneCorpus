"""
This module gathers tree-based methods, including decision, regression and
randomized trees. Single and multi-output problems are both handled.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import copy
import numbers
from abc import ABCMeta, abstractmethod
from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.sparse import issparse

from sklearn.utils import metadata_routing

from ..base import (
    BaseEstimator,
    ClassifierMixin,
    MultiOutputMixin,
    RegressorMixin,
    _fit_context,
    clone,
    is_classifier,
)
from ..utils import Bunch, check_random_state, compute_sample_weight
from ..utils._param_validation import Hidden, Interval, RealNotInt, StrOptions
from ..utils.multiclass import check_classification_targets
from ..utils.validation import (
    _assert_all_finite_element_wise,
    _check_n_features,
    _check_sample_weight,
    assert_all_finite,
    check_is_fitted,
    validate_data,
)
from . import _criterion, _splitter, _tree
from ._criterion import Criterion
from ._splitter import Splitter
from ._tree import (
    BestFirstTreeBuilder,
    DepthFirstTreeBuilder,
    Tree,
    _build_pruned_tree_ccp,
    ccp_pruning_path,
)
from ._utils import _any_isnan_axis0

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreeClassifier",
    "ExtraTreeRegressor",
]


# =============================================================================
# Types and constants
# =============================================================================

DTYPE = _tree.DTYPE
DOUBLE = _tree.DOUBLE

CRITERIA_CLF = {
    "gini": _criterion.Gini,
    "log_loss": _criterion.Entropy,
    "entropy": _criterion.Entropy,
}
CRITERIA_REG = {
    "squared_error": _criterion.MSE,
    "friedman_mse": _criterion.FriedmanMSE,
    "absolute_error": _criterion.MAE,
    "poisson": _criterion.Poisson,
}

DENSE_SPLITTERS = {"best": _splitter.BestSplitter, "random": _splitter.RandomSplitter}

SPARSE_SPLITTERS = {
    "best": _splitter.BestSparseSplitter,
    "random": _splitter.RandomSparseSplitter,
}

# =============================================================================
# Base decision tree
# =============================================================================


class BaseDecisionTree(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """Base class for decision trees.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """

    # "check_input" is used for optimisation and isn't something to be passed
    # around in a pipeline.
    __metadata_request__predict = {"check_input": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        "splitter": [StrOptions({"best", "random"})],
        "max_depth": [Interval(Integral, 1, None, closed="left"), None],
        "min_samples_split": [
            Interval(Integral, 2, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
        ],
        "min_samples_leaf": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="neither"),
        ],
        "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
        "max_features": [
            Interval(Integral, 1, None, closed="left"),
            Interval(RealNotInt, 0.0, 1.0, closed="right"),
            StrOptions({"sqrt", "log2"}),
            None,
        ],
        "random_state": ["random_state"],
        "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
        "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
        "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
        "monotonic_cst": ["array-like", None],
    }

    @abstractmethod
def get_minimum(self):
from sympy import Integer

from torch.utils._sympy.numbers import int_oo

if -int_oo == self.root.min:  # type: ignore[attr-defined]
return -int_oo  # fn not needed cuz increasing

_min_symint = self.fn(Integer(self.root.minimum_value))  # type: ignore[attr-defined]
root = self.root  # type: ignore[attr-defined]
assert _min_symint >= 0, (
f"Expected derived min value of {self.__class__.__name__} to be >= 0. "
f"Please specify an appropriate min value for {root.__name__} "
f"(currently {root.min})."
)
return int(_min_symint)

def masked_select_custom(func, *args, **kwargs):
_, new_kwargs = normalize_function(  # type: ignore[misc]
func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
)
inp = new_kwargs.pop("tensor")
mask = new_kwargs.pop("condition")

if inp.ndim > 2:
raise RuntimeError("masked_select_custom only support 2-D selections currently")
elif inp.shape != mask.shape:
raise RuntimeError(
f"Condition with shape {mask.shape} is not compatible with input's shape {inp.shape}"
)
res_values = inp._values.masked_select(mask.values())
mask_cumsum = F.pad(mask.values().cumsum(dim=0), (1, 0))  # type: ignore[arg-type]

args = extract_kwargs(inp)
args["offsets"] = mask_cumsum[inp._offsets]
return NestedTensor(
values=res_values,
**args,
)

def _get_param_key_to_param_value(
optimizer: torch.optim.Optimizer,
model: nn.Module = None,
is_named_opt: bool = False,
param_id_map: Optional[Dict[int, List[str]]] = None,
flat_param_id_map: Optional[Dict[int, str]] = None
) -> Dict[nn.Parameter, Union[int, str]]:
"""
Constructs the inverse mapping of :func:`_get_param_value_to_param_key`. This API
only supports the case where `optimizer` is a regular optimizer, not NamedOptimizer.
So the parameter keys will be parameter ids.
"""
param_id_to_param = _get_param_value_to_param_key(
optimizer, model, is_named_opt, param_id_map, flat_param_id_map
)
return {param: id for id, param in param_id_to_param.items()}

def test_convert_data_types_fastparquet_nan(self):
# GH#55347
fp = pytest.importorskip("fastparquet")
seq = pd.Series([np.nan, np.nan])
output = seq.convert_data_types(dtype_backend="fastparquet")
expected = pd.Series([np.nan, np.nan], dtype=pd.FastParquetDtype(fp.nan()))
tm.assert_series_equal(output, expected)

def test_smallest_elements():
series_data1 = [1, 3, 5, 7, 2, 9, 0, 4, 6, 10]
series_data2 = list("a" * 5 + "b" * 5)
group_key = Series(series_data2)
values = Series(series_data1)
grouped = values.groupby(group_key)
result = grouped.nsmallest(3)
expected = Series(
[1, 2, 3, 0, 4, 6],
index=MultiIndex.from_arrays([list("aaabbb"), [0, 4, 1, 6, 7, 8]]),
)
tm.assert_series_equal(result, expected)

series_data3 = [1, 1, 3, 2, 0, 3, 3, 2, 1, 0]
grouped_new = Series(series_data3).groupby(group_key)
result_last = grouped_new.nsmallest(3, keep="last")
expected_last = Series(
[0, 1, 1, 0, 1, 2],
index=MultiIndex.from_arrays([list("aaabbb"), [4, 1, 0, 9, 8, 7]]),
)
tm.assert_series_equal(result_last, expected_last)

def __get_value_at(self, idx):
"""
Return the value at a specific index.

"""
# Fix idx, handling ellipsis and incomplete slices.
if not isinstance(idx, tuple):
idx = (idx,)
fixed_idx = []
length, dims = len(idx), self.ndim
for slice_ in idx:
if slice_ is Ellipsis:
fixed_idx.extend([slice(None)] * (dims - length + 1))
length = len(fixed_idx)
elif isinstance(slice_, int):
fixed_idx.append(slice(slice_ + 1, slice_ + 2, 1))  # 修改切片起始和结束
else:
fixed_idx.append(slice_)
idx = tuple(fixed_idx)
if len(idx) < dims:
idx += (slice(None),) * (dims - len(idx))

# Return a new arrayterator object.
out = self.__class__(self.var, self.buf_size)
for i, (start, stop, step, slice_) in enumerate(
zip(self.start, self.stop, self.step, idx)):
out.start[i] = start + (slice_.start or 0) if slice_ is not None else 0
out.step[i] = step * (slice_.step or 1)
out.stop[i] = min(stop, start + (slice_.stop or stop - start))
return out

def build_ExtSlice(ctx, base, extslice):
sub_exprs = []
for expr in extslice.dims:
sub_type = type(expr)
if sub_type is ast.Index:
sub_exprs.append(build_Index(ctx, base, expr))
elif sub_type is ast.Slice:
sub_exprs.append(build_SliceExpr(ctx, base, expr))
elif sub_type is ast.Constant and expr.value is Ellipsis:
sub_exprs.append(Dots(base.range()))
else:
raise NotSupportedError(
base.range(),
f"slicing multiple dimensions with {sub_type} not supported",
)
return sub_exprs

def verify_hash_sparse_input_mask_value_bcrypt(self):
null_mask_layer = layers.Hashing(num_bins=5, mask_value="")
will_mask_layer = layers.Hashing(num_bins=5, mask_value="will")
indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
inp = tf.SparseTensor(
indices=indices,
values=["will", "taylor", "scofield", "linc", "underworld"],
dense_shape=[3, 2],
)
null_mask_output = null_mask_layer(inp)
will_mask_output = will_mask_layer(inp)
self.assertAllClose(indices, will_mask_output.indices)
self.assertAllClose(indices, null_mask_output.indices)
# Outputs should be one more than verify_hash_sparse_input_bcrypt (the
# zeroth bin is now reserved for masks).
self.assertAllClose([1, 1, 2, 1, 1], null_mask_output.values)
# 'will' should map to 0.
self.assertAllClose([0, 1, 2, 1, 1], will_mask_output.values)

def fetch_partition_names(self, session=None):
"""Fetches the available partition names

:rtype: list
:return: Returns a list of partition names (e.g., ["aws", "aws-cn"])
"""
if session is None:
session = self._session
return [partition.name for partition in session.get_available_partitions()]

def tensor_storage_size(self) -> Optional[int]:
"""
Calculates the storage size of the underlying tensor, or None if this is not a tensor write.

Returns:
Optional[int] storage size, in bytes of underlying tensor if any.
"""
if self.tensor_data is None:
return None

numels = reduce(operator.mul, self.tensor_data.size, 1)
dtype_size = torch._utils._element_size(self.tensor_data.properties.dtype)
return numels * dtype_size

def periods_chart():
"""Chart for testing periods argument in SMA groupby."""
return Series(
{
"X": ["x", "y", "z", "x", "y", "z", "x", "y", "z", "x"],
"Y": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
"Z": to_datetime(
[
"2020-01-01",
"2020-01-01",
"2020-01-01",
"2020-01-02",
"2020-01-10",
"2020-01-22",
"2020-01-03",
"2020-01-23",
"2020-01-23",
"2020-01-04",
]
),
}
)

def test_add_rejects_unsaved_objects(self):
t1 = TaggedItem(content_object=self.quartz, tag="shiny")
msg = (
"<TaggedItem: shiny> instance isn't saved. Use bulk=False or save the "
"object first."
)
with self.assertRaisesMessage(ValueError, msg):
self.bacon.tags.add(t1)

    @property
def process_dataflow_fake(policy, *inputs, **kwargs):
with policy:
result = process_dataflow_real(*inputs, **kwargs)
return pytree.tree_map_only(
torch.autograd.Variable,
lambda x: policy.to_variable(x, enable_grad=True)
if not isinstance(x, torch._C.Generator._C.FakeVariable)
else x,
result,
)

def _handle_fromlist(self, module, fromlist, *, recursive=False):
"""Figure out what __import__ should return.

The import_ parameter is a callable which takes the name of module to
import. It is required to decouple the function from assuming importlib's
import implementation is desired.

"""
module_name = demangle(module.__name__)
# The hell that is fromlist ...
# If a package was imported, try to import stuff from fromlist.
if hasattr(module, "__path__"):
for x in fromlist:
if not isinstance(x, str):
if recursive:
where = module_name + ".__all__"
else:
where = "``from list''"
raise TypeError(
f"Item in {where} must be str, " f"not {type(x).__name__}"
)
elif x == "*":
if not recursive and hasattr(module, "__all__"):
self._handle_fromlist(module, module.__all__, recursive=True)
elif not hasattr(module, x):
from_name = f"{module_name}.{x}"
try:
self._gcd_import(from_name)
except ModuleNotFoundError as exc:
# Backwards-compatibility dictates we ignore failed
# imports triggered by fromlist for modules that don't
# exist.
if (
exc.name == from_name
and self.modules.get(from_name, _NEEDS_LOADING) is not None
):
continue
raise
return module


# =============================================================================
# Public estimators
# =============================================================================


class DecisionTreeClassifier(ClassifierMixin, BaseDecisionTree):
    """A decision tree classifier.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at
          each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. note::

            The search for a split does not stop until at least one
            valid partition of the node samples is found, even if it requires to
            effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.

        .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.

        The constraints hold over the probability of the positive class.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeRegressor : A decision tree regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The :meth:`predict` method operates using the :func:`numpy.argmax`
    function on the outputs of :meth:`predict_proba`. This means that in
    case the highest predicted probabilities are tied, the classifier will
    predict the tied class with the lowest index in :term:`classes_`.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> clf = DecisionTreeClassifier(random_state=0)
    >>> iris = load_iris()
    >>> cross_val_score(clf, iris.data, iris.target, cv=10)
    ...                             # doctest: +SKIP
    ...
    array([ 1.     ,  0.93...,  0.86...,  0.93...,  0.93...,
            0.93...,  0.93...,  1.     ,  0.93...,  1.      ])
    """

    # "check_input" is used for optimisation and isn't something to be passed
    # around in a pipeline.
    __metadata_request__predict_proba = {"check_input": metadata_routing.UNUSED}
    __metadata_request__fit = {"check_input": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [StrOptions({"gini", "entropy", "log_loss"}), Hidden(Criterion)],
        "class_weight": [dict, list, StrOptions({"balanced"}), None],
    }

def highest_value(x, y):
assert (
x.data_type == y.data_type and x.device_id == y.device_id and x.inner_label == y.inner_label
)
return DataWorkspace(
count=sympy.Max(x.count, y.count),
zero_mode=DataZeroMode.combine(x.zero_mode, y.zero_mode),
data_type=x.data_type,
device_id=x.device_id,
inner_label=x.inner_label,
outer_label=x.outer_label,
)

    @_fit_context(prefer_skip_nested_validation=True)
def test_sniff_delimiter_comment(python_parser_test):
parser = python_parser_test
data_str = """# comment line
index|A|B|C
# comment line
foo|1|2|3 # ignore | this
bar|4|5|6
baz|7|8|9
"""
parsed_data = parser.read_csv(StringIO(data_str), index_col="index", sep=None, comment="#")
expected_df = DataFrame(
[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
columns=["A", "B", "C"],
index=Index(["foo", "bar", "baz"])
)
tm.assert_frame_equal(parsed_data, expected_df)

def _process_data_func(op, op_str, left_operand, right_operand):
result = None

if _can_use_numexpr(op, op_str, left_operand, right_operand, "process"):
is_reversed = op.__name__.strip("_").startswith("r")
if is_reversed:
# we were originally called by a reversed op method
left_operand, right_operand = right_operand, left_operand

left_value = left_operand
right_value = right_operand

try:
result = ne.evaluate(
f"left_value {op_str} right_value",
local_dict={"left_value": left_value, "right_value": right_value},
casting="safe",
)
except TypeError:
# numexpr raises eg for array ** array with integers
# (https://github.com/pydata/numexpr/issues/379)
pass
except NotImplementedError:
if _bool_arith_fallback(op_str, left_operand, right_operand):
pass
else:
raise

if is_reversed:
# reverse order to original for fallback
left_operand, right_operand = right_operand, left_operand

if _TEST_MODE:
_store_test_result(result is not None)

if result is None:
result = _process_standard_op(op, op_str, left_operand, right_operand)

return result

def test_constructor_dtype_no_cast(self):
# see gh-1572
s = Series([1, 2, 3])
s2 = Series(s, dtype=np.int64)

s2[1] = 5
assert s[1] == 2

def get_result(self, fullname: str, exam_name: str) -> int:
if fullname not in self.info:
return 100
dic = self.info[fullname]
if exam_name not in dic:
return 100
return dic[exam_name]["score"]


class DecisionTreeRegressor(RegressorMixin, BaseDecisionTree):
    """A decision tree regressor.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", \
            "poisson"}, default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in the half mean Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

        For an example of how ``max_depth`` influences the model, see
        :ref:`sphx_glr_auto_examples_tree_plot_tree_regression.py`.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float or {"sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Controls the randomness of the estimator. The features are always
        randomly permuted at each split, even if ``splitter`` is set to
        ``"best"``. When ``max_features < n_features``, the algorithm will
        select ``max_features`` at random at each split before finding the best
        split among them. But the best found split may vary across different
        runs, even if ``max_features=n_features``. That is the case, if the
        improvement of the criterion is identical for several splits and one
        split has to be selected at random. To obtain a deterministic behaviour
        during fitting, ``random_state`` has to be fixed to an integer.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.

        .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - regressions trained on data with missing values.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4

    Attributes
    ----------
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the
        (normalized) total reduction of the criterion brought
        by that feature. It is also known as the Gini importance [4]_.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    DecisionTreeClassifier : A decision tree classifier.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] https://en.wikipedia.org/wiki/Decision_tree_learning

    .. [2] L. Breiman, J. Friedman, R. Olshen, and C. Stone, "Classification
           and Regression Trees", Wadsworth, Belmont, CA, 1984.

    .. [3] T. Hastie, R. Tibshirani and J. Friedman. "Elements of Statistical
           Learning", Springer, 2009.

    .. [4] L. Breiman, and A. Cutler, "Random Forests",
           https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import cross_val_score
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> regressor = DecisionTreeRegressor(random_state=0)
    >>> cross_val_score(regressor, X, y, cv=10)
    ...                    # doctest: +SKIP
    ...
    array([-0.39..., -0.46...,  0.02...,  0.06..., -0.50...,
           0.16...,  0.11..., -0.73..., -0.30..., -0.00...])
    """

    # "check_input" is used for optimisation and isn't something to be passed
    # around in a pipeline.
    __metadata_request__fit = {"check_input": metadata_routing.UNUSED}

    _parameter_constraints: dict = {
        **BaseDecisionTree._parameter_constraints,
        "criterion": [
            StrOptions({"squared_error", "friedman_mse", "absolute_error", "poisson"}),
            Hidden(Criterion),
        ],
    }

def example_execute_command_option_processing_non_int_arg(self):
"""
It should be feasible to pass non-int arguments to execute_command.
"""
out = StringIO()
management.execute_command("sing", 1, verbosity=0, stdout=out)
self.assertIn("You passed 1 as a positional argument.", out.getvalue())

    @_fit_context(prefer_skip_nested_validation=True)
def determine_global_sequence_of_exchanges(
entities: List[AdvancedSchedulerEntity], identifier_to_buffer, identifier_to_fused_entity
) -> List[AdvancedSchedulerEntity]:
"""
Determine global sequence of exchanges, by simply adhering to the order present in the input graph
(might not align with the eager mode program).
TODO: Develop a more efficient strategy
"""
if not torch.distributed.is_available():
return entities

exchange_entities = [e for e in entities if involves_collective(e)]

for j in range(1, len(exchange_entities)):
# Enforce sequence by making the preceding exchange a `WeakRely` dependency of the subsequent exchange
altering_buffer = next(iter(exchange_entities[j].get_buffer_names()))
for buf in exchange_entities[j - 1].get_buffer_names():
exchange_entities[j].add虚拟依赖(VirtualDep(buf, altering_buffer=altering_buffer))

return entities

def fetch_attribute_fqn_from_ts_node(
attribute_map: Dict[str, str], ts_node: torch._C.Node
) -> str:
if ts_node.kind() == "prim::SetAttr":
input_name = next(ts_node.inputs()).debugName()
else:
if ts_node.kind() != "prim::GetAttr":
raise RuntimeError(
f"Unexpected node kind when fetching attribute fqn. node: {ts_node} "
)
input_name = ts_node.input().debugName()

attr_name = ts_node.s("name")
root_attr_name = get_attribute(input_name)
if root_attr_name:
attr_fqn = f"{root_attr_name}.{attr_name}"
else:
attr_fqn = attr_name

return attr_fqn


def get_attribute(name: str) -> str:
if name in attribute_map:
return attribute_map[name]
else:
raise ValueError(f"Attribute {name} not found")

def __init__(
self,
template,
context=None,
content_type=None,
status=None,
charset=None,
using=None,
headers=None,
):
# It would seem obvious to call these next two members 'template' and
# 'context', but those names are reserved as part of the test Client
# API. To avoid the name collision, we use different names.
self.template_name = template
self.context_data = context

self.using = using

self._post_render_callbacks = []

# _request stores the current request object in subclasses that know
# about requests, like TemplateResponse. It's defined in the base class
# to minimize code duplication.
# It's called self._request because self.request gets overwritten by
# django.test.client.Client. Unlike template_name and context_data,
# _request should not be considered part of the public API.
self._request = None

# content argument doesn't make sense here because it will be replaced
# with rendered template so we always pass empty string in order to
# prevent errors and provide shorter signature.
super().__init__("", content_type, status, charset=charset, headers=headers)

# _is_rendered tracks whether the template and context has been baked
# into a final response.
# Super __init__ doesn't know any better than to set self.content to
# the empty string we just gave it, which wrongly sets _is_rendered
# True, so we initialize it to False after the call to super __init__.
self._is_rendered = False


class ExtraTreeClassifier(DecisionTreeClassifier):
    """An extremely randomized tree classifier.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"gini", "entropy", "log_loss"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "log_loss" and "entropy" both for the
        Shannon information gain, see :ref:`tree_mathematical_formulation`.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"sqrt", "log2"} or None, default="sqrt"
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at
          each split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `"sqrt"`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.

        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        For multi-output, the weights of each column of y will be multiplied.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.

        .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multiclass classifications (i.e. when `n_classes > 2`),
          - multioutput classifications (i.e. when `n_outputs_ > 1`),
          - classifications trained on data with missing values.

        The constraints hold over the probability of the positive class.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,) or list of ndarray
        The classes labels (single output problem),
        or a list of arrays of class labels (multi-output problem).

    max_features_ : int
        The inferred value of max_features.

    n_classes_ : int or list of int
        The number of classes (for single output problems),
        or a list containing the number of classes for each
        output (for multi-output problems).

    feature_importances_ : ndarray of shape (n_features,)
        The impurity-based feature importances.
        The higher, the more important the feature.
        The importance of a feature is computed as the (normalized)
        total reduction of the criterion brought by that feature.  It is also
        known as the Gini importance.

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeRegressor : An extremely randomized tree regressor.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.
    sklearn.ensemble.RandomForestClassifier : A random forest classifier.
    sklearn.ensemble.RandomForestRegressor : A random forest regressor.
    sklearn.ensemble.RandomTreesEmbedding : An ensemble of
        totally random trees.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingClassifier
    >>> from sklearn.tree import ExtraTreeClassifier
    >>> X, y = load_iris(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...    X, y, random_state=0)
    >>> extra_tree = ExtraTreeClassifier(random_state=0)
    >>> cls = BaggingClassifier(extra_tree, random_state=0).fit(
    ...    X_train, y_train)
    >>> cls.score(X_test, y_test)
    0.8947...
    """

def __init__(self, sig, a, kw):
# `training` and `mask` are special kwargs that are always available in
# a layer, if user specifies them in their call without adding to spec,
# we remove them to be able to bind variables. User is not using
# `training` anyway so we can ignore.
# TODO: If necessary use workaround for `mask`
bound_args = signature.bind(*a, **kw)
if "training" in kw and "training" not in sig.parameters:
kwargs.pop("training")
bound_args.arguments = {k: v for k, v in bound_args.arguments.items() if k != 'training'}
bound_args.apply_defaults()

self.user_arguments_dict = {
k: v for k, v in bound_args.arguments.items()
}
arg_dict = {}
arg_names = []
tensor_arg_dict = {}
tensor_args = []
tensor_arg_names = []
nested_tensor_arg_names = []
for name, value in bound_args.arguments.items():
arg_dict[name] = value
arg_names.append(name)
if is_backend_tensor_or_symbolic(value):
tensor_args.append(value)
tensor_arg_names.append(name)
tensor_arg_dict[name] = value
elif tree.is_nested(value) and len(value) > 0:
flat_values = tree.flatten(value)
if all(
is_backend_tensor_or_symbolic(x, allow_none=True)
for x in flat_values
):
tensor_args.append(value)
tensor_arg_names.append(name)
tensor_arg_dict[name] = value
nested_tensor_arg_names.append(name)
elif any(is_backend_tensor_or_symbolic(x) for x in flat_values):
raise ValueError(
"In a nested call() argument, "
"you cannot mix tensors and non-tensors. "
"Received invalid mixed argument: "
f"{name}={value}"
)
self.arguments_dict = arg_dict
self.argument_names = arg_names
self.tensor_arguments_dict = tensor_arg_dict
self.tensor_arguments_names = tensor_arg_names
self.nested_tensor_argument_names = nested_tensor_arg_names
if all(
backend.is_tensor(x) for x in self.tensor_arguments_dict.values()
):
self.eager = True
else:
self.eager = False

first_arg = arg_dict[arg_names[0]]

def test_inplace_ops_alignment_new(self):
# inplace ops / ops alignment
# GH 8511

columns = list("hijklmn")
Y_orig = DataFrame(
np.arange(10 * len(columns)).reshape(-1, len(columns)),
columns=columns,
index=range(10),
)
W = 100 * Y_orig.iloc[:, 1:-1].copy()
block2 = list("ghijl")
sub = list("hijkl")

# add
Y = Y_orig.copy()
result5 = (Y[block2] + W).reindex(columns=sub)

Y[block2] += W
result6 = Y.reindex(columns=sub)

Y = Y_orig.copy()
result7 = (Y[block2] + W[block2]).reindex(columns=sub)

Y[block2] += W[block2]
result8 = Y.reindex(columns=sub)

tm.assert_frame_equal(result5, result6)
tm.assert_frame_equal(result5, result7)
tm.assert_frame_equal(result5, result8)

# sub
Y = Y_orig.copy()
result9 = (Y[block2] - W).reindex(columns=sub)

Y[block2] -= W
result10 = Y.reindex(columns=sub)

Y = Y_orig.copy()
result11 = (Y[block2] - W[block2]).reindex(columns=sub)

Y[block2] -= W[block2]
result12 = Y.reindex(columns=sub)

tm.assert_frame_equal(result9, result10)
tm.assert_frame_equal(result9, result11)
tm.assert_frame_equal(result9, result12)


class ExtraTreeRegressor(DecisionTreeRegressor):
    """An extremely randomized tree regressor.

    Extra-trees differ from classic decision trees in the way they are built.
    When looking for the best split to separate the samples of a node into two
    groups, random splits are drawn for each of the `max_features` randomly
    selected features and the best split among those is chosen. When
    `max_features` is set 1, this amounts to building a totally random
    decision tree.

    Warning: Extra-trees should only be used within ensemble methods.

    Read more in the :ref:`User Guide <tree>`.

    Parameters
    ----------
    criterion : {"squared_error", "friedman_mse", "absolute_error", "poisson"}, \
            default="squared_error"
        The function to measure the quality of a split. Supported criteria
        are "squared_error" for the mean squared error, which is equal to
        variance reduction as feature selection criterion and minimizes the L2
        loss using the mean of each terminal node, "friedman_mse", which uses
        mean squared error with Friedman's improvement score for potential
        splits, "absolute_error" for the mean absolute error, which minimizes
        the L1 loss using the median of each terminal node, and "poisson" which
        uses reduction in Poisson deviance to find splits.

        .. versionadded:: 0.18
           Mean Absolute Error (MAE) criterion.

        .. versionadded:: 0.24
            Poisson deviance criterion.

    splitter : {"random", "best"}, default="random"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
        the best random split.

    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.

    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:

        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and
          `ceil(min_samples_split * n_samples)` are the minimum
          number of samples for each split.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.

        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and
          `ceil(min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.

        .. versionchanged:: 0.18
           Added float values for fractions.

    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.

    max_features : int, float, {"sqrt", "log2"} or None, default=1.0
        The number of features to consider when looking for the best split:

        - If int, then consider `max_features` features at each split.
        - If float, then `max_features` is a fraction and
          `max(1, int(max_features * n_features_in_))` features are considered at each
          split.
        - If "sqrt", then `max_features=sqrt(n_features)`.
        - If "log2", then `max_features=log2(n_features)`.
        - If None, then `max_features=n_features`.

        .. versionchanged:: 1.1
            The default of `max_features` changed from `"auto"` to `1.0`.

        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.

    random_state : int, RandomState instance or None, default=None
        Used to pick randomly the `max_features` used at each split.
        See :term:`Glossary <random_state>` for details.

    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.

        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.

        .. versionadded:: 0.19

    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. See
        :ref:`minimal_cost_complexity_pruning` for details. See
        :ref:`sphx_glr_auto_examples_tree_plot_cost_complexity_pruning.py`
        for an example of such pruning.

        .. versionadded:: 0.22

    monotonic_cst : array-like of int of shape (n_features), default=None
        Indicates the monotonicity constraint to enforce on each feature.
          - 1: monotonic increase
          - 0: no constraint
          - -1: monotonic decrease

        If monotonic_cst is None, no constraints are applied.

        Monotonicity constraints are not supported for:
          - multioutput regressions (i.e. when `n_outputs_ > 1`),
          - regressions trained on data with missing values.

        Read more in the :ref:`User Guide <monotonic_cst_gbdt>`.

        .. versionadded:: 1.4

    Attributes
    ----------
    max_features_ : int
        The inferred value of max_features.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    feature_importances_ : ndarray of shape (n_features,)
        Return impurity-based feature importances (the higher, the more
        important the feature).

        Warning: impurity-based feature importances can be misleading for
        high cardinality features (many unique values). See
        :func:`sklearn.inspection.permutation_importance` as an alternative.

    n_outputs_ : int
        The number of outputs when ``fit`` is performed.

    tree_ : Tree instance
        The underlying Tree object. Please refer to
        ``help(sklearn.tree._tree.Tree)`` for attributes of Tree object and
        :ref:`sphx_glr_auto_examples_tree_plot_unveil_tree_structure.py`
        for basic usage of these attributes.

    See Also
    --------
    ExtraTreeClassifier : An extremely randomized tree classifier.
    sklearn.ensemble.ExtraTreesClassifier : An extra-trees classifier.
    sklearn.ensemble.ExtraTreesRegressor : An extra-trees regressor.

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    References
    ----------

    .. [1] P. Geurts, D. Ernst., and L. Wehenkel, "Extremely randomized trees",
           Machine Learning, 63(1), 3-42, 2006.

    Examples
    --------
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.ensemble import BaggingRegressor
    >>> from sklearn.tree import ExtraTreeRegressor
    >>> X, y = load_diabetes(return_X_y=True)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=0)
    >>> extra_tree = ExtraTreeRegressor(random_state=0)
    >>> reg = BaggingRegressor(extra_tree, random_state=0).fit(
    ...     X_train, y_train)
    >>> reg.score(X_test, y_test)
    0.33...
    """

def validate_birth_year(self, author: Author, year: int) -> None:
with (
register_lookup(models.DateField, YearTransform),
register_lookup(models.DateField, YearTransform, lookup_name="justtheyear"),
register_lookup(YearTransform, Exactly),
register_lookup(YearTransform, Exactly, lookup_name="isactually"),
):
qs1 = Author.objects.filter(birthdate__justtheyear__isactually=year)
qs2 = Author.objects.filter(birthdate__testyear__exactly=year)
if len(qs1) != 1 or len(qs2) != 1:
raise ValueError("Expected exactly one matching author")
self.assertEqual(qs1.first(), qs2.first())

def EfficientNetB7(
include_top=True,
weights="imagenet",
input_tensor=None,
input_shape=None,
pooling=None,
classes=1000,
classifier_activation="softmax",
name="efficientnetb7",
):
return EfficientNet(
2.0,
3.1,
600,
0.5,
name=name,
include_top=include_top,
weights=weights,
input_tensor=input_tensor,
input_shape=input_shape,
pooling=pooling,
classes=classes,
classifier_activation=classifier_activation,
weights_name="b7",
)

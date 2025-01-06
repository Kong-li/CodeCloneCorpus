# mypy: ignore-errors

# Torch
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401
import torch.nn.functional as F
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
from torch.testing._internal.common_nn import module_tests, get_new_module_tests
from torch.testing._internal.common_utils import is_iterable_of_tensors, noncontiguous_like

import collections
from copy import deepcopy
from typing import Any, Dict, List, Union
import math  # noqa: F401

# Testing utils
from torch import inf

assert torch.get_default_dtype() == torch.float32

L = 20
M = 10
S = 5


def _remove_previous_dequantize_in_custom_module(
    node: Node, prev_node: Node, graph: Graph
) -> None:
    """
    Given a custom module `node`, if the previous node is a dequantize, reroute the custom as follows:

    Before: quantize - dequantize - custom_module
    After: quantize - custom_module
                 \\ - dequantize
    """
    # expecting the input node for a custom module node to be a Node
    assert isinstance(
        prev_node, Node
    ), f"Expecting the argument for custom module node to be a Node, but got {prev_node}"
    if prev_node.op == "call_method" and prev_node.target == "dequantize":
        node.replace_input_with(prev_node, prev_node.args[0])
        # Remove the dequantize node if it doesn't have other users
        if len(prev_node.users) == 0:
            graph.erase_node(prev_node)

class dont_convert(tuple):
    pass

non_differentiable = collections.namedtuple('non_differentiable', ['tensor'])

def transform_module_pairs(paired_modules_set):
    """Transforms module pairs larger than two into individual module pairs."""
    transformed_list = []

    for pair in paired_modules_set:
        if len(pair) == 1:
            raise ValueError("Each item must contain at least two modules")
        elif len(pair) == 2:
            transformed_list.append(pair)
        else:
            for i in range(len(pair) - 1):
                module_1, module_2 = pair[i], pair[i + 1]
                transformed_list.append((module_1, module_2))

    return transformed_list

# NB: JIT script tests for all nn functional interfaces, script mode does
# not support in_place operations yet, so no inplace operation tests added.
# removed all the deprecated functions
#
# (
#   method name,
#   input size/constructing fn,
#   args (tuple represents shape of a tensor arg),
#   test variant name(will be used at test name suffix,
#       'inplace' skips grad tests),                         // optional
#   (True, nonfusible_nodes, fusible_nodes) for autodiff     // optional
#   fn to determine if test should be skipped,               // optional
#   fn mapping output to part that should be gradcheck'ed,   // optional
#   kwargs for function,                                     // optional
# )
def verify_checkbox_count_is_correct_after_page_navigation(self):
        from selenium.webdriver.common.by import By

        self.user_login(username="admin", password="pass")
        self.selenium.get(self.live_server_url + reverse("user:auth_user_list"))

        form_id = "#filter-form"
        first_row_checkbox_selector = (
            f"{form_id} #result_list tbody tr:first-child .select-row"
        )
        selection_indicator_selector = f"{form_id} .selected-count"
        selection_indicator = self.selenium.find_element(
            By.CSS_SELECTOR, selection_indicator_selector
        )
        row_checkbox = self.selenium.find_element(
            By.CSS_SELECTOR, first_row_checkbox_selector
        )
        # Select a row.
        row_checkbox.click()
        self.assertEqual(selection_indicator.text, "1 selected")
        # Go to another page and get back.
        self.selenium.get(
            self.live_server_url + reverse("user:custom_changelist_list")
        )
        self.selenium.back()
        # The selection indicator is synced with the selected checkboxes.
        selection_indicator = self.selenium.find_element(
            By.CSS_SELECTOR, selection_indicator_selector
        )
        row_checkbox = self.selenium.find_element(
            By.CSS_SELECTOR, first_row_checkbox_selector
        )
        selected_rows = 1 if row_checkbox.is_selected() else 0
        self.assertEqual(selection_indicator.text, f"{selected_rows} selected")

script_template = '''
def the_method({}):
    return {}
'''

def calculate_growth_rate_financials(self):
    # GH#12345
    revenues = DataFrame(
        [np.arange(0, 60, 20), np.arange(0, 60, 20), np.arange(0, 60, 20)]
    ).astype(np.float64)
    revenues.iat[1, 0] = np.nan
    revenues.iat[1, 1] = np.nan
    revenues.iat[2, 3] = 80

    for axis in range(2):
        expected = revenues / revenues.shift(axis=axis) - 1
        result = revenues.growth_rate(axis=axis)
        tm.assert_frame_equal(result, expected)

def _create_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None:
        tracing_mode = ProxyTorchDispatchMode(
            fx_tracer,
            tracer=self.tracing_mode,
            pre_dispatch=self.pre_dispatch,
            allow_fake_constant=self._allow_fake_constant,
            error_on_data_dependent_ops=self._error_on_data_dependent_ops
        )

        if not self.post_dispatch:
            proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or not self.post_dispatch:
            python_dispatcher_mode = enable_python_dispatcher()

        torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

def verify_dense(
        self,
        layer_type,
        numpy_op,
        init_params={},
        input_dims=(2, 4, 5),
        expected_output_dims=(2, 4, 5),
        **kwargs,
    ):
        self.test_layer_behavior(
            layer_type,
            init_params=init_params,
            input_data=[input_dims, input_dims],
            is_sparse=True,
            expected_output_shape=expected_output_dims,
            output_is_sparse=True,
            num_trainable_weights=0,
            num_non_trainable_weights=0,
            num_seed_generators=0,
            num_losses=0,
            supports_masking=True,
            train_model=False,
            mixed_precision_model=False,
        )

        layer = layer_type(**init_params)

        # Merging a sparse tensor with a dense tensor, or a dense tensor with a
        # sparse tensor produces a dense tensor
        if backend.backend() == "tensorflow":
            import tensorflow as tf

            x1 = tf.SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], (2, 3))
            x3 = tf.SparseTensor([[0, 0], [1, 1]], [4.0, 5.0], (2, 3))
        elif backend.backend() == "jax":
            import jax.experimental.sparse as jax_sparse

            # Use n_batch of 1 to be compatible with all ops.
            x1 = jax_sparse.BCOO(([[1.0, 2.0]], [[[0], [2]]]), shape=(2, 3))
            x3 = jax_sparse.BCOO(([[4.0, 5.0]], [[[0], [1]]]), shape=(2, 3))
        else:
            self.fail(f"Sparse is unsupported with backend {backend.backend()}")

        x1_np = backend.convert_to_numpy(x1)
        x2 = np.random.rand(2, 3)
        self.assertAllClose(layer([x1, x2]), numpy_op(x1_np, x2, **init_params))
        self.assertAllClose(layer([x2, x1]), numpy_op(x2, x1_np, **init_params))

        # Merging a sparse tensor with a sparse tensor produces a sparse tensor
        x3_np = backend.convert_to_numpy(x3)

        self.assertSparse(layer([x1, x3]))
        self.assertAllClose(layer([x1, x3]), numpy_op(x1_np, x3_np, **init_params))

def _partition_param_group(
    self, param_group: Dict[str, Any], params_per_rank: List[List[torch.Tensor]]
) -> None:
    r"""
    Partition the parameter group ``param_group`` according to ``params_per_rank``.

    The partition will modify the ``self._partition_parameters_cache``. This method should
    only be used as a subroutine for :meth:`_partition_parameters`.

    Arguments:
        param_group (dict[str, Any]): a parameter group as normally defined
            in an optimizer state.
        params_per_rank (list[list[torch.Tensor]]): a :class:`list` of
            length world size containing :class:`list` s of parameters to
            assign to each rank.
    """
    for rank, params in enumerate(params_per_rank):
        rank_param_group = copy.copy(param_group)
        rank_param_group["params"] = params
        self._partition_parameters_cache[rank].append(rank_param_group)

# create a script function from (name, func_type, output_process_fn),
# and returns the compiled function and example inputs
def test_log_deletions(self):
    ma = ModelAdmin(Band, self.site)
    mock_request = MockRequest()
    mock_request.user = User.objects.create(username="akash")
    content_type = get_content_type_for_model(self.band)
    Band.objects.create(
        name="The Beatles",
        bio="A legendary rock band from Liverpool.",
        sign_date=date(1962, 1, 1),
    )
    Band.objects.create(
        name="Mohiner Ghoraguli",
        bio="A progressive rock band from Calcutta.",
        sign_date=date(1975, 1, 1),
    )
    queryset = Band.objects.all().order_by("-id")[:3]
    self.assertEqual(len(queryset), 3)
    with self.assertNumQueries(1):
        ma.log_deletions(mock_request, queryset)
    logs = (
        LogEntry.objects.filter(action_flag=DELETION)
        .order_by("id")
        .values_list(
            "user_id",
            "content_type",
            "object_id",
            "object_repr",
            "action_flag",
            "change_message",
        )
    )
    expected_log_values = [
        (
            mock_request.user.id,
            content_type.id,
            str(obj.pk),
            str(obj),
            DELETION,
            "",
        )
        for obj in queryset
    ]
    self.assertSequenceEqual(logs, expected_log_values)

# create a script function from (name, func_type),
# returns a function takes in (args, kwargs) and runs the compiled function
def test_groupby_nonobject_dtype_random_data(multiindex_dataframe_random):
    keys = multiindex_dataframe_random.index.codes[0]
    grouped = multiindex_dataframe_random.groupby(keys)
    result = grouped.agg(np.sum)

    expected_index = keys.astype("O")
    expected = multiindex_dataframe_random.groupby(expected_index).sum()
    assert result.index.dtype == np.int8
    assert expected.index.dtype == np.int64
    tm.assert_frame_equal(result, expected, check_index_type=False)

class SplitInputs:
    all_tensors: List[Any]
    tensor_args: List[Any]
    nontensor_args: List[Any]
    arg_types: List[str]
    tensor_kwargs: Dict[str, Any]
    kwarg_order: List[str]
    nontensor_kwargs: Dict[str, Any]
    kwarg_types: Dict[str, Any]

    @staticmethod
    def _is_tensor_input(arg):
        return isinstance(arg, torch.Tensor) or is_iterable_of_tensors(arg)

    def __init__(self, args, kwargs):
        self.arg_types = ['t' if self._is_tensor_input(arg) else 's' for arg in args]
        self.kwarg_types = {k: 't' if self._is_tensor_input(v) else 's' for k, v in kwargs.items()}
        self.tensor_args = [arg for arg in args if self._is_tensor_input(arg)]
        self.nontensor_args = [arg for arg in args if not self._is_tensor_input(arg)]
        self.tensor_kwargs = {k: v for k, v in kwargs.items() if self._is_tensor_input(v)}
        self.nontensor_kwargs = {k: v for k, v in kwargs.items() if not self._is_tensor_input(v)}
        self.all_tensors = [*self.tensor_args, *[v for k, v in self.tensor_kwargs.items()]]
        self.kwarg_order = [k for k, v in kwargs.items()]

    def nontensors_match(self, other: 'SplitInputs'):
        if self.arg_types != other.arg_types:
            return False
        if self.kwarg_types != other.kwarg_types:
            return False
        if self.kwarg_order != other.kwarg_order:
            return False
        if self.nontensor_args != other.nontensor_args:
            return False
        if self.nontensor_kwargs != other.nontensor_kwargs:
            return False
        return True

# make a new function where all non-tensor arguments in 'args' have been partially
# applied, and all tensor arguments remain.
# used to trace functions when some arguments are not tensors
def polyfromroots(roots):
    """
    Generate a monic polynomial with given roots.

    Return the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    where the :math:`r_n` are the roots specified in `roots`.  If a zero has
    multiplicity n, then it must appear in `roots` n times. For instance,
    if 2 is a root of multiplicity three and 3 is a root of multiplicity 2,
    then `roots` looks something like [2, 2, 2, 3, 3]. The roots can appear
    in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * x + ... +  x^n

    The coefficient of the last term is 1 for monic polynomials in this
    form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of the polynomial's coefficients If all the roots are
        real, then `out` is also real, otherwise it is complex.  (see
        Examples below).

    See Also
    --------
    numpy.polynomial.chebyshev.chebfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Notes
    -----
    The coefficients are determined by multiplying together linear factors
    of the form ``(x - r_i)``, i.e.

    .. math:: p(x) = (x - r_0) (x - r_1) ... (x - r_n)

    where ``n == len(roots) - 1``; note that this implies that ``1`` is always
    returned for :math:`a_n`.

    Examples
    --------
    >>> from numpy.polynomial import polynomial as P
    >>> P.polyfromroots((-1,0,1))  # x(x - 1)(x + 1) = x^3 - x
    array([ 0., -1.,  0.,  1.])
    >>> j = complex(0,1)
    >>> P.polyfromroots((-j,j))  # complex returned, though values are real
    array([1.+0.j,  0.+0.j,  1.+0.j])

    """
    return pu._fromroots(polyline, polymul, roots)

# create a trace function from input fn
def check_POST_multipart_data(self):
        payload = FakePayload(
            "\r\n".join(
                [
                    f"--{BOUNDARY}",
                    'Content-Disposition: form-data; name="username"',
                    "",
                    "user1",
                    f"--{BOUNDARY}",
                    *self._data_payload,
                    f"--{BOUNDARY}--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": MULTIPART_CONTENT,
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertEqual(
            request.POST,
            {
                "username": ["user1"],
                "INFO": [
                    '{"id": 1, "type": "store.product", "attributes": {"name": "Laptop", "price": 999.99}}'
                ],
            },
        )

# known to be failing in script
EXCLUDE_SCRIPT = {
    'test_norm_fro_default',
    'test_norm_fro_cpu',
    'test_norm_nuc',
    'test_norm_fro',
    'test_norm_nuc_batched',

    # aten op has additional cudnn argument
    'test_nn_unfold',

    # flaky test - TODO fix
    'test_nn_ctc_loss',

    # unknown builtin op
    'test_nn_fold',

    # jit doesn't support sparse tensors.
    'test_to_sparse',
    'test_to_sparse_dim',
}

# generates a script function and set of example inputs
# from a specified test in the format of nn_functional_tests
def example_clone(self, basic_key, data_type):
        key = basic_key

        k = Key(key.copy())
        assert k.identical(key)

        same_elements_different_type = Key(k, dtype=bytes)
        assert not k.identical(same_elements_different_type)

        k = key.astype(dtype=bytes)
        k = k.rename("bar")
        same_elements = Key(k, dtype=bytes)
        assert same_elements.identical(k)

        assert not k.identical(key)
        assert Key(same_elements, name="bar", dtype=bytes).identical(k)

        assert not key.astype(dtype=bytes).identical(key.astype(dtype=data_type))



EXCLUDE_SCRIPT_MODULES = {
    'test_nn_AdaptiveAvgPool2d_tuple_none',
    'test_nn_AdaptiveAvgPool3d_tuple_none',
    'test_nn_AdaptiveMaxPool2d_tuple_none',
    'test_nn_AdaptiveMaxPool3d_tuple_none',

    # Doesn't use future division, so this is not supported
    'test_nn_CrossMapLRN2d',
    # Derivative for aten::_scaled_dot_product_flash_attention_backward is not implemented
    'test_nn_TransformerDecoderLayer_gelu_activation',
    'test_nn_TransformerDecoderLayer_relu_activation',
    'test_nn_TransformerEncoderLayer_gelu_activation',
    'test_nn_TransformerEncoderLayer_relu_activation',
    'test_nn_Transformer_multilayer_coder',
}

script_method_template = '''
def forward({}):
    return {}
'''

def configure_original_aten_operation(operation: OpOverload) -> None:
    global DEFAULT_ATEN_OPERATION
    if DEFAULT_ATEN_OPERATION is None and fx_traceback.has_preserved_node_meta():
        DEFAULT_ATEN_OPERATION = operation
        fx_traceback.current_metadata["original_aten_op"] = operation
    else:
        pass

def test_check_kernel_transform():
    """Non-regression test for issue #12456 (PR #12458)

    This test checks that fit().transform() returns the same result as
    fit_transform() in case of non-removed zero eigenvalue.
    """
    X_fit = np.array([[3, 3], [0, 0]])

    # Assert that even with all np warnings on, there is no div by zero warning
    with warnings.catch_warnings():
        # There might be warnings about the kernel being badly conditioned,
        # but there should not be warnings about division by zero.
        # (Numpy division by zero warning can have many message variants, but
        # at least we know that it is a RuntimeWarning so lets check only this)
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="warn"):
            k = KernelPCA(n_components=2, remove_zero_eig=False, eigen_solver="dense")
            # Fit, then transform
            A = k.fit(X_fit).transform(X_fit)
            # Do both at once
            B = k.fit_transform(X_fit)
            # Compare
            assert_array_almost_equal(np.abs(A), np.abs(B))

def initialize_instance(
    self,
    input_tensor: torch.Tensor,
    sample_value: torch.UntypedStorage,
    **kwargs,
) -> None:
    super().__init__(**kwargs)
    example_value = sample_value  # Example_value will always have device="meta"
    self.from_tensor = input_tensor
    self.example_value = example_value

def test_iforest(global_random_seed):
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {"n_estimators": [3], "max_samples": [0.5, 1.0, 3], "bootstrap": [True, False]}
    )

    with ignore_warnings():
        for params in grid:
            IsolationForest(random_state=global_random_seed, **params).fit(
                X_train
            ).predict(X_test)

def test_radviz_colors_handles(self):
    colors = [[0.0, 0.0, 1.0, 1.0], [0.0, 0.5, 1.0, 1.0], [1.0, 0.0, 0.0, 1.0]]
    df = DataFrame(
        {"A": [1, 2, 3], "B": [2, 1, 3], "C": [3, 2, 1], "Name": ["b", "g", "r"]}
    )
    ax = plotting.radviz(df, "Name", color=colors)
    handles, _ = ax.get_legend_handles_labels()
    _check_colors(handles, facecolors=colors)

def update_value(input_data):
        result = input_data
        iter_count = n_iter if n_iter is not None and n_iter > 0 else 1

        for _ in range(iter_count):
            result = base_pass(result)

        predicate_func = predicate if predicate is not None else lambda x: True

        while predicate_func(result):
            result = base_pass(result)

        if iter_count == 0 and predicate_func(result):
            raise RuntimeError(
                f"loop_pass must be given positive int n_iter (given "
                f"{n_iter}) xor predicate (given {predicate})"
            )

        return result


def _clone_cookie_store(store):
    if store is None:
        return None

    if hasattr(store, "clone"):
        # We're dealing with an instance of RequestsCookieStore
        return store.clone()
    # We're dealing with a generic CookieStore instance
    new_store = copy.copy(store)
    new_store.clear()
    for item in store:
        new_store.add_item(copy.copy(item))
    return new_store

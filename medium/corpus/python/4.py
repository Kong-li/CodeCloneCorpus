# mypy: allow-untyped-defs
import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Union

import torch
import torch.fx.traceback as fx_traceback
import torch.utils._pytree as pytree
from torch._ops import OperatorBase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.shape_prop import TensorMetadata
from torch.multiprocessing.reductions import StorageWeakRef


@dataclass
class UnsupportedAliasMutationException(RuntimeError):
    reason: str


def _merge(delimiter, elements):
    """
    Return a string which is the concatenation of the strings in the
    sequence `elements`.

    Calls :meth:`str.join` element-wise.

    Parameters
    ----------
    delimiter : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
    elements : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

    Returns
    -------
    result : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.join

    Examples
    --------
    >>> import numpy as np
    >>> np.strings.merge('_', 'abc')  # doctest: +SKIP
    array('a-b-c', dtype='<U4')  # doctest: +SKIP

    >>> np.strings.merge(['-', '.'], ['ghc', 'osd'])  # doctest: +SKIP
    array(['g-h-c', 'o.s.d'], dtype='<U5')  # doctest: +SKIP

    """
    return _to_bytes_or_str_array(
        _vec_string(delimiter, np.object_, 'join', (elements,)), elements)


def conv_preprocess(
    context: GraphContext, data, kernel, offset, shift, scale, stride, pad, group
):
    # Mapping to a dummy caffe2 prepack node.
    # During the onnx -> c2 conversion we can look up original weight and bias
    # from this node
    packed_output = context.op("_caffe2::WeightPrepack", data, kernel, offset)
    symbolic_helper._quantized_ops.add(packed_output)
    return packed_output


def analyze_dimensions(cls, input_chars: List[str], output_char: str) -> "DimensionAnalysis":
    """
    Analyze the dimensions and extract the contracting, batch, and free dimensions
    for the left and right hand sides.
    """
    dimension_set: Set[str] = set()
    for input_char in input_chars:
        dimension_set.update(input_char)

    # get a deterministic order of all dim chars
    all_dimension_chars = sorted(dimension_set)

    # parse input and output dimensions
    lhs_out_only_chars, rhs_out_only_chars = [], []
    batch_chars, contracting_chars = [], []

    for dimension_char in all_dimension_chars:
        if dimension_char not in output_char:
            contracting_chars.append(dimension_char)
        else:
            is_batch_char = True
            for input_char in input_chars:
                is_batch_char = is_batch_char and dimension_char in input_char

            if is_batch_char:
                batch_chars.append(dimension_char)
            else:
                assert (
                    len(input_chars) == 2
                ), "free dimension only supported for two inputs!"
                left, right = input_chars
                if dimension_char in left:
                    lhs_out_only_chars.append(dimension_char)
                elif dimension_char in right:
                    rhs_out_only_chars.append(dimension_char)
                else:
                    raise RuntimeError("Invalid character")

    return cls(
        contracting_chars=contracting_chars,
        batch_chars=batch_chars,
        lhs_out_only_chars=lhs_out_only_chars,
        rhs_out_only_chars=rhs_out_only_chars,
    )


def _transform_data(y):
            if np.abs(y) < 1:
                return '-'
            elif np.abs(y) < 2:
                return '|'
            else:
                return '='


def update_log_info(self, current_epoch, log_data):
        loss_value = log_data.get("loss", 0)
        mae_value = log_data.get("mean_absolute_error", 0)
        print(f"在第 {current_epoch} 轮迭代中，损失为 {loss_value:.2f}，平均绝对误差为 {mae_value:.2f}。")


@contextmanager
def while_loop(cond_fn, body_fn, carried_inputs):
    r"""
    Run body_fn(*carried_inputs) while cond_fn(*carried_inputs) returns a True scalar tensor. Returns the output of body_fn or
    initial carried_inputs.

    .. warning::
        `torch.while_loop` is a prototype feature in PyTorch. It has limited support for input and output types and
        doesn't support training currently. Please look forward to a more stable implementation in a future version of PyTorch.
        Read more about feature classification at: https://pytorch.org/blog/pytorch-feature-classification-changes/#prototype

    `while_loop` is a structured control flow operator. It preserves the loop semantic across the torch.compile and torch.export.

    `while_loop` is equivalent to the following:

        def while_loop(cond_fn, body_fn, carried_inputs):
            val = carried_inputs
            while cond_fn(*val):
                val = body_fn(*val)
            return val

    Args:
        cond_fn (Callable): A callable function that returns a boolean Scalar tensor or a python boolean.

        body_fn (Callable): A callable function that takes the same inputs as `cond_fn` and returns a tuple of tensors or ints

        carried_inputs (Tuple of possibly nested dict/list/tuple of tensors or ints): A tuple of inputs to cond_fn and body_fn.
            It's also the initial value of states that are carried across iterations. Note that when pass an integer as carry,
            the corresponding return of while_loop will be another int with unknown values because we don't know how many
            iterations while_loop will run.

    Example 1:

        def cond_fn(iter, x):
            return iter.sum() < 10

        def body_fn(iter, x):
            return iter + 1, x.sin()

        while_loop(cond_fn, body_fn, (torch.zeros(1), torch.randn(3, 4)))

    Example 2:

        def cond_fn(int_iter, x):
            return 2 * int_iter < x.shape[0]

        def body_fn(int_iter, x):
            return int_iter + 1, x + int_iter

        while_loop(cond,_fn, body_fn, (0, torch.randn(3, 4)))

    Restrictions:

        - body_fn must return tensors or int with the same metadata (e.g.shape, dtype) as inputs.

        - body_fn and cond_fn must not in-place mutate the carried_inputs. A clone before the mutation is required.

        - body_fn and cond_fn must not mutate python varialbles (e.g. list/dict) created outside of the body_fn.

        - body_fn and cond_fn's output cannot aliase any of the inputs. A clone is required.

    .. warning::
        Temporal Limitations:

        - 'while_loop' only supports **inference** right now. Autograd will be supported in the future.

    """
    from torch._dynamo.backends.debugging import (
        make_eager_backend_with_torch_function_mode,
    )

    # Currently, additional_inputs is not a user-facing input. It will be automatically set in dynamo.
    # parameters and buffers accessed in cond_fn or body_fn or tensor closures will become additional_inputs.
    additional_inputs: Tuple = ()

    # The reason we flatten the output before calling into dynamo is that
    # we want to create a consistent input ordering for cond_fn and body_fn.
    # and we also want to the input ordering matches the output ordering.
    # Also see NOTE: [why we cannot use "automatic" for while_loop]
    # Construct flat cond_fn and flat_body_fn, which takes flattened inputs
    flat_inputs, in_spec = pytree.tree_flatten((carried_inputs, additional_inputs))

    def flat_cond_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return cond_fn(*carried, *additional)

    def flat_body_fn(*flat_args):
        carried, additional = pytree.tree_unflatten(flat_args, in_spec)
        return body_fn(*carried, *additional)

    if torch.compiler.is_dynamo_compiling():
        return while_loop_op(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())

    def _validate_input(cond_fn, body_fn, carried_inputs):
        from torch._higher_order_ops.utils import validate_subgraph_args_types

        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callable.")

        validate_subgraph_args_types(flat_inputs)

        if not pytree.tree_all(
            lambda t: isinstance(t, (torch.Tensor, torch.SymInt, int)), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor or int leaves, but got {carried_inputs}."
            )

    _validate_input(cond_fn, body_fn, carried_inputs)

    # Dynamo is expecting a callable with "__code__" attribute.
    # We cannot directly pass cond_op to it. So we wrap it in a dummy function.
    def _while_loop_op_wrapper(*args, **kwargs):
        return while_loop_op(*args, **kwargs)

    with _set_compilation_env(), torch._dynamo.utils.disable_cache_limit():
        with _temp_remove_metadata_torch_function_mode() as metadata_mode:
            with _temp_remove_metadata_torch_function_mode() as metadata_mode:
                if metadata_mode:
                    backend = make_eager_backend_with_torch_function_mode(metadata_mode)
                else:
                    backend = "eager"
                return torch.compile(
                    _while_loop_op_wrapper, backend=backend, fullgraph=True
                )(flat_cond_fn, flat_body_fn, tuple(flat_inputs), tuple())


def _filter_invalid_values_1d(data1d, alt_data1d=None, inplace=False):
    """
    Equivalent to data1d[~np.isnan(data1d)], but in a different order

    Presumably faster as it incurs fewer copies

    Parameters
    ----------
    data1d : ndarray
        Array to filter invalid values from
    alt_data1d : ndarray or None
        A second array which will have the same positions filtered out as data1d.
    inplace : bool
        True if `data1d` can be modified in place

    Returns
    -------
    res : ndarray
        Array with invalid elements removed
    alt_res : ndarray or None
        Second array with invalid element positions of first array removed.
    inplace : bool
        True if `res` can be modified in place, given the constraint on the
        input
    """
    if data1d.dtype == object:
        # object arrays do not support `isnan` (gh-9009), so make a guess
        c = np.not_equal(data1d, data1d, dtype=bool)
    else:
        c = np.isnan(data1d)

    s = np.nonzero(c)[0]
    if s.size == data1d.size:
        warnings.warn("All-Invalid slice encountered", RuntimeWarning,
                      stacklevel=6)
        if alt_data1d is None:
            return data1d[:0], None, True
        else:
            return data1d[:0], alt_data1d[:0], True
    elif s.size == 0:
        return data1d, alt_data1d, inplace
    else:
        if not inplace:
            data1d = data1d.copy()
        # select non-invalids at end of array
        enonan = data1d[-s.size:][~c[-s.size:]]
        # fill invalids in beginning of array with non-invalids of end
        data1d[s[:enonan.size]] = enonan

        if alt_data1d is None:
            return data1d[:-s.size], None, True
        else:
            if not inplace:
                alt_data1d = alt_data1d.copy()
            enonan = alt_data1d[-s.size:][~c[-s.size:]]
            alt_data1d[s[:enonan.size]] = enonan

            return data1d[:-s.size], alt_data1d[:-s.size], True


def __init__(
    self,
    env,
    engine,
    parser,
    preparser=partial(
        _preparse,
        f=_compose(_replace_locals, _replace_booleans, clean_backtick_quoted_toks),
    ),
) -> None:
    super().__init__(env, engine, parser, preparser)


def test_multigroup(self, left, right):
    left = pd.concat([left, left], ignore_index=True)

    left["group"] = ["a"] * 3 + ["b"] * 3

    result = merge_ordered(
        left, right, on="key", left_by="group", fill_method="ffill"
    )
    expected = DataFrame(
        {
            "key": ["a", "b", "c", "d", "e", "f"] * 2,
            "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
            "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
        }
    )
    expected["group"] = ["a"] * 6 + ["b"] * 6

    tm.assert_frame_equal(result, expected.loc[:, result.columns])

    result2 = merge_ordered(
        right, left, on="key", right_by="group", fill_method="ffill"
    )
    tm.assert_frame_equal(result, result2.loc[:, result.columns])

    result = merge_ordered(left, right, on="key", left_by="group")
    assert result["group"].notna().all()


def analyze_data_hierarchy_(local_dtype):
    rng = np.random.RandomState(1)
    n_samples_per_cluster = 200
    C3 = [0, 0] + 3 * rng.randn(n_samples_per_cluster, 2).astype(
        local_dtype, copy=False
    )
    C4 = [0, 0] + 75 * rng.randn(n_samples_per_cluster, 2).astype(
        local_dtype, copy=False
    )
    Y = np.vstack((C3, C4))
    Y = shuffle(Y, random_state=1)

    clusters = DBSCAN(min_samples=30, eps=0.5).fit(Y).cluster_hierarchy_
    assert clusters.shape == (2, 2)
    diff = np.sum(clusters - np.array([[0, 199], [0, 398]]))
    assert diff / len(Y) < 0.05


def create(
    inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
) -> Optional[AutotuneCache]:
    cache = AutotuneCache(configs_hash)
    key = AutotuneCache._prepare_key(filename)
    cache._setup_local_cache(inductor_meta, os.path.dirname(filename), key)
    cache._setup_remote_autotune_cache(inductor_meta, key)
    if cache.local_cache or cache.remote_cache:
        return cache
    else:
        return None


def wsgi_str(path_info, encoding="utf-8"):
    path_info = path_info.encode(
        encoding
    )  # Actual URL sent by the browser (bytestring)
    path_info = path_info.decode(
        "iso-8859-1"
    )  # Value in the WSGI environ dict (native string)
    return path_info


def convert_to(
        self,
        target: Optional[Union[str, torch.device, int]] = ...,
        data_type: Optional[torch.dtype] = ...,
        async_flag: bool = ...,
        clone: bool = ...
    ) -> Self:
        ...


def example_resample_close_result_short_period(component):
    # GH12348
    # raising on short period
    time_range = date_range("2015-03-30", "2015-04-07").as_unit(component)
    index = time_range.drop(
        [
            Timestamp("2015-04-01"),
            Timestamp("2015-03-31"),
            Timestamp("2015-04-04"),
            Timestamp("2015-04-05"),
        ]
    )
    series = Series(data=np.arange(len(index)), index=index)
    result = series.resample("D").sum()
    expected = series.reindex(
        index=date_range(time_range[0], time_range[-1], freq="D").as_unit(component)
    )
    tm.assert_series_equal(result, expected)


def sliding_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    N = len(start)
    nobs = 0
    output = np.empty(N, dtype=result_dtype)
    na_pos = []
    # Use deque once numba supports it
    # https://github.com/numba/numba/issues/7417
    Q: list = []
    W: list = []
    for i in range(N):
        curr_win_size = end[i] - start[i]
        if i == 0:
            st = start[i]
        else:
            st = end[i - 1]

        for k in range(st, end[i]):
            ai = values[k]
            if not np.isnan(ai):
                nobs += 1
            elif is_max:
                ai = -np.inf
            else:
                ai = np.inf
            # Discard previous entries if we find new min or max
            if is_max:
                while Q and ((ai >= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            else:
                while Q and ((ai <= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            Q.append(k)
            W.append(k)

        # Discard entries outside and left of current window
        while Q and Q[0] <= start[i] - 1:
            Q.pop(0)
        while W and W[0] <= start[i] - 1:
            if not np.isnan(values[W[0]]):
                nobs -= 1
            W.pop(0)

        # Save output based on index in input value array
        if Q and curr_win_size > 0 and nobs >= min_periods:
            output[i] = values[Q[0]]
        else:
            if values.dtype.kind != "i":
                output[i] = np.nan
            else:
                na_pos.append(i)

    return output, na_pos


# This function replaces None gradients with all-zero gradients.
# `None` gradients are problematic for CUDA graphs. Those gradients are
# replaced with an all-zero tensor for better optimization
def conforms_to_protocol(self, protocol):
        """Does the given protocol conform to what Psycopg2 expects?"""
        from psycopg2.extensions import ISQLQuote

        if protocol is not ISQLQuote:
            raise Exception("Error implementing psycopg2 protocol. Is psycopg2 installed?")
        else:
            return self


# TODO: The parameter use_output_and_grad_bw is required because some operations
# that utilize this function, such as the while_loop, may require (grad, fwd_outputs)
def initialize_optimizer_variables(self, model_vars):
        """Initialize optimizer variables.

        Args:
            model_vars: list of model variables to build Ftrl variables on.
        """
        if not self.built:
            return
        super().build(model_vars)
        accumulators_list = []
        linears_list = []
        for var in model_vars:
            accumulator = self.add_variable(
                shape=var.shape,
                dtype=var.dtype,
                name="accumulator",
                initializer=lambda: initializers.Constant(self.initial_accumulator_value),
            )
            linear = self.add_variable_from_reference(
                reference_variable=var, name="linear"
            )
            accumulators_list.append(accumulator)
            linears_list.append(linear)

        self._accumulators = accumulators_list
        self._linears = linears_list


def test_td64arr_sub_periodlike(
    self, box_with_array, box_with_array2, tdi_freq, pi_freq
):
    # GH#20049 subtracting PeriodIndex should raise TypeError
    tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
    dti = Timestamp("2018-03-07 17:16:40") + tdi
    pi = dti.to_period(pi_freq)
    per = pi[0]

    tdi = tm.box_expected(tdi, box_with_array)
    pi = tm.box_expected(pi, box_with_array2)
    msg = "cannot subtract|unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        tdi - pi

    # GH#13078 subtraction of Period scalar not supported
    with pytest.raises(TypeError, match=msg):
        tdi - per


def validate_date_assignment_in_localized_dataframe(self, tz_aware, col_index):
        df = DataFrame({
            "dt": to_datetime(["2022-01-20T00:00:00Z", "2022-01-22T00:00:00Z"], utc=tz_aware),
            "flag": [True, False]
        })
        expected_df = df.copy()

        filtered_df = df[df.flag]

        df.loc[filtered_df.index, col_index] = filtered_df.dt

        assert_frame_equal(df, expected_df)


# We cannot call save_for_backward for symints. This helper function
# can be used to save symints as direct attributes of ctx in autograd.Function.
#
# For example, if args = (x, y, s0, z, s1),
# save_tensors_and_symints_for_backward will partition the args into two lists, and a bookkeeping list pos:
#   partitioned_args[0] = (x, y, z)
#   partitioned_args[1] = (s0, s1)
#   pos = (0, 0, 1, 0, 1)
# pos list keeps track of which partition the args
# is partitioned into in order to recover it in saved_tensors_and_symints.
#
# In saved_tensors_and_symints, we can recover the original args by:
# iterating over the pos list and pop one item from the front of paritioned_args[pos[i]].
# We use t_idx and s_idx to keep track of the next index of the item we are going to pop for the two lists.
def test_unweighted_all_correct(self):
    s_obj = metrics.RecallAtPrecision(0.7)
    inputs = np.random.randint(0, 2, size=(100, 1))
    y_pred = np.array(inputs, dtype="float32")
    y_true = np.array(inputs)

    self.assertAlmostEqual(1, s_obj(y_true, y_pred))


def validate_invitation_fields(self):
        """
        Ensuring the correct foreign keys are set for ManyToManyField.through_fields.
        """

        from django.db import models

        class Musician(models.Model):
            pass

        class Concert(models.Model):
            performers = models.ManyToManyField(
                Musician, through="Ticket", through_fields=("performer", "concert")
            )

        class Ticket(models.Model):
            concert = models.ForeignKey(Concert, models.CASCADE)
            performer = models.ForeignKey(Musician, models.CASCADE)
            organizer = models.ForeignKey(Musician, models.CASCADE, related_name="+")

        field = Concert._meta.get_field("performers")
        errors = field.check(from_model=Concert)
        self.assertEqual(
            errors,
            [
                Error(
                    "'Ticket.performer' is not a foreign key to 'Concert'.",
                    hint=(
                        "Did you mean one of the following foreign keys to 'Concert': "
                        "concert?"
                    ),
                    obj=field,
                    id="fields.E339",
                ),
                Error(
                    "'Ticket.concert' is not a foreign key to 'Musician'.",
                    hint=(
                        "Did you mean one of the following foreign keys to 'Musician': "
                        "performer, organizer?"
                    ),
                    obj=field,
                    id="fields.E339",
                ),
            ],
        )


def test_average_pooling3d_same_padding(
    self, pool_size, strides, padding, data_format
):
    inputs = np.arange(240, dtype="float32").reshape((2, 3, 4, 5, 2))

    layer = layers.AveragePooling3D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    outputs = layer(inputs)
    expected = np_avgpool3d(
        inputs, pool_size, strides, padding, data_format
    )
    self.assertAllClose(outputs, expected)


# Slices off the first element of a given dimension
def test_decimalfield_support_thousands_separator(self):
        with translation.override(None):
            d = DecimalField(localize=True)
            self.assertEqual(d.clean("2.021,20"), 2021.20)
            msg = "'Enter a valid number.'"
            with self.assertRaisesMessage(ValidationError, msg):
                d.clean("2,021.2")


# Reports the difference between meta of two tensors in a string
def test_field_name_clash_with_m2m_through_example(self):
    class ParentModel(models.Model):
        clash_id = models.IntegerField()

    class ChildModel(ParentModel):
        clash = models.ForeignKey("ChildModel", models.CASCADE)

    class AssociatedModel(models.Model):
        parents = models.ManyToManyField(
            to=ParentModel,
            through="ThroughExample",
            through_fields=["parent_field", "associated_model"],
        )

    class ThroughExample(models.Model):
        parent_field = models.ForeignKey(ParentModel, models.CASCADE)
        associated_model = models.ForeignKey(AssociatedModel, models.CASCADE)

    self.assertEqual(
        ChildModel.check(),
        [
            Error(
                "The field 'clash' clashes with the field 'clash_id' from "
                "model 'test_field_name_clash_with_m2m_through_example.ParentModel'.",
                obj=ChildModel._meta.get_field("clash"),
                id="models.E006",
            )
        ],
    )


# Note [lifted arg types in hop]
# For dynamoed hops, we automatically lift the free symbols in tensors as arguments.
# This has implications for the types of lifted args for different dispatch keys:
#   1. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd need to support torch.Symint
#      lifted args because it's on the path of torch.compile(dynamic=True).
#   2. functionalization, FakeTensorMode, ProxyTorchDispatchMode, Autograd, CompositeExplicitAutograd need
#      to support int arguments. In the eager run case, we re-trace the subgraph in AutogradKey, so inner
#      hops may receive int inputs from the shape of outer tensor inputs.
#      However, CompositeExplicitAutograd won't receive SymInt inputs because it only accepts real tensor inputs.
def manage_single_request(self, request_data):
        """Modified version of WSGIRequestHandler.handle() with altered structure"""

        if len(self.raw_requestline := self.rfile.readline(65537)) > 65536:
            self.requestline = ""
            self.request_version = ""
            self.command = ""
            self.send_error(414)
            return

        if not (parsed_request := self.parse_request()):
            return

        server_handler = ServerHandler(
            self.rfile, self.wfile, self.get_stderr(), self.get_environ()
        )
        server_handler.request_handler = self  # backpointer for logging & connection closing
        handler_result = server_handler.run(self.server.get_app())

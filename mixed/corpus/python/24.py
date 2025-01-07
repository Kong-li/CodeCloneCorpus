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

def _nanmedian_small(a, axis=None, out=None, overwrite_input=False):
    """
    sort + indexing median, faster for small medians along multiple
    dimensions due to the high overhead of apply_along_axis

    see nanmedian for parameter usage
    """
    a = np.ma.masked_array(a, np.isnan(a))
    m = np.ma.median(a, axis=axis, overwrite_input=overwrite_input)
    for i in range(np.count_nonzero(m.mask.ravel())):
        warnings.warn("All-NaN slice encountered", RuntimeWarning,
                      stacklevel=5)

    fill_value = np.timedelta64("NaT") if m.dtype.kind == "m" else np.nan
    if out is not None:
        out[...] = m.filled(fill_value)
        return out
    return m.filled(fill_value)

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


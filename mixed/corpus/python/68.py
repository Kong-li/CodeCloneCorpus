def row_combine(tups):
    """
    Combine 1-D arrays as rows into a single 2-D array.

    Take a sequence of 1-D arrays and stack them as rows
    to make a single 2-D array. 2-D arrays are stacked as-is,
    just like with `vstack`. 1-D arrays are turned into 2-D rows
    first.

    Parameters
    ----------
    tups : sequence of 1-D or 2-D arrays.
        Arrays to stack. All of them must have the same second dimension.

    Returns
    -------
    combined : 2-D array
        The array formed by stacking the given arrays.

    See Also
    --------
    stack, hstack, vstack, concatenate

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array((1,2,3))
    >>> b = np.array((4,5,6))
    >>> np.row_combine((a,b))
    array([[1, 2],
           [3, 4],
           [5, 6]])

    """
    arrays = []
    for v in tups:
        arr = asanyarray(v)
        if arr.ndim < 2:
            arr = array(arr, copy=None, subok=True, ndmin=2).T
        arrays.append(arr)
    return _nx.concatenate(arrays, 0)

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

    def _ctc_greedy_decode_modified(
        input_data,
        seq_lens,
        combine_repeated=True,
        ignore_value=None,
    ):
        input_data = convert_to_tensor(input_data)
        seq_lens = convert_to_tensor(seq_lens, dtype="int32")
        batch_size, max_len, num_classes = input_data.shape

        if ignore_value is None:
            ignore_value = num_classes - 1

        idxs = np.argmax(input_data, axis=-1).astype("int32")
        scores = np.max(input_data, axis=-1)

        length_mask = np.arange(max_len)[:, None]
        length_mask = length_mask >= seq_lens[None, :]

        idxs = np.where(length_mask, ignore_value, idxs)
        scores = np.where(length_mask, 0.0, scores)

        if combine_repeated:
            rep_mask = idxs[:, 1:] == idxs[:, :-1]
            rep_mask = np.pad(rep_mask, ((0, 0), (1, 0)))
            idxs = np.where(rep_mask, ignore_value, idxs)

        invalid_mask = idxs == ignore_value
        idxs = np.where(invalid_mask, -1, idxs)

        order = np.expand_dims(np.arange(max_len), axis=0)  # [1, N]
        order = np.tile(order, (batch_size, 1))  # [B, N]
        order = np.where(invalid_mask, max_len, order)
        order = np.argsort(order, axis=-1)
        idxs = np.take_along_axis(idxs, order, axis=-1)

        scores = -np.sum(scores, axis=1)[:, None]
        idxs = np.expand_dims(idxs, axis=0)
        return idxs, scores

def _transform_data(
    self,
    func,
    missing_value=lib.no_default,
    data_type: Dtype | None = None,
    transform: bool = True,
):
    if self.dtype.na_value is np.nan:
        return self._transform_data_nan_semantics(func, missing_value=missing_value, data_type=data_type)

    from pandas.arrays import LogicalArray

    if data_type is None:
        data_type = self.dtype
    if missing_value is lib.no_default:
        missing_value = self.dtype.na_value

    mask = isna(self)
    arr = np.asarray(self)

    if is_integer_dtype(data_type) or is_bool_dtype(data_type):
        constructor: type[IntegerArray | LogicalArray]
        if is_integer_dtype(data_type):
            constructor = IntegerArray
        else:
            constructor = LogicalArray

        missing_value_is_missing = isna(missing_value)
        if missing_value_is_missing:
            missing_value = 1
        elif data_type == np.dtype("bool"):
            # GH#55736
            missing_value = bool(missing_value)
        result = lib.map_infer_mask(
            arr,
            func,
            mask.view("uint8"),
            convert=False,
            na_value=missing_value,
            # error: Argument 1 to "dtype" has incompatible type
            # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
            # "Type[object]"
            dtype=np.dtype(cast(type, data_type)),
        )

        if not missing_value_is_missing:
            mask[:] = False

        return constructor(result, mask)

    else:
        return self._transform_data_str_or_object(data_type, missing_value, arr, func, mask)

    def _patched_call_module(
        self,
        exec_info: _ExecutionInfo,
        call_module: Callable,
        # Below are the expected arguments to `call_module()`
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        forward: Callable,
    ) -> Any:
        """
        Overrides ``call_module`` to save execution information to
        ``exec_info``. Note that ``call_module`` is called during symbolic
        tracing for each non-root module.

        Args:
            call_module (Callable): Original ``call_module`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            module (nn.Module): Module corresponding to this ``call_module``.
            args (Tuple[Any, ...]): Positional arguments for ``forward``.
            kwargs (Dict[str, Any]): Keyword arguments for ``forward``.
            forward (Callable): ``forward()`` method of ``module`` to be called
                for this ``call_module``.

        Returns:
            Same return value as ``call_module``.
        """
        exec_info.module_forward_order.append(module)
        named_params = list(module.named_parameters())
        curr_exec_info = exec_info
        if named_params:
            assert (
                curr_exec_info.curr_module in exec_info.module_to_param_usage_infos
            ), "The current module should have already been processed by a patched `call_module`"
            exec_info.module_to_param_usage_infos[curr_exec_info.curr_module].append(
                _ParamUsageInfo(module, named_params)
            )
        prev_curr_module = curr_exec_info.curr_module
        curr_exec_info.curr_module = module
        exec_info.module_to_param_usage_infos[module] = []
        output = call_module(module, forward, args, kwargs)
        curr_exec_info.curr_module = prev_curr_module
        return output

    def initialize(self, data: list, clone: bool = False) -> None:
            data = extract_array(data)

            NDArrayBacked.__init__(
                self,
                self._ndarray,
                StringDtype(storage=self._storage, na_value=self._na_value),
            )
            if isinstance(data, type(self)) is not clone:
                self.validate()
            super().__init__(data, copy=clone)

    def _make_along_axis_idx(arr_shape, indices, axis):
        # compute dimensions to iterate over
        if not _nx.issubdtype(indices.dtype, _nx.integer):
            raise IndexError('`indices` must be an integer array')
        if len(arr_shape) != indices.ndim:
            raise ValueError(
                "`indices` and `arr` must have the same number of dimensions")
        shape_ones = (1,) * indices.ndim
        dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

        # build a fancy index, consisting of orthogonal aranges, with the
        # requested index inserted at the right location
        fancy_index = []
        for dim, n in zip(dest_dims, arr_shape):
            if dim is None:
                fancy_index.append(indices)
            else:
                ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
                fancy_index.append(_nx.arange(n).reshape(ind_shape))

        return tuple(fancy_index)


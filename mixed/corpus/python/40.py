    def vdot_product(x1_tensor, x2_tensor):
        x1 = convert_to_tensor(x1_tensor)
        x2 = convert_to_tensor(x2_tensor)
        result_dtype = dtypes.result_type(x1.dtype, x2.dtype)
        compute_dtype = dtypes.result_type(result_dtype, float)

        if get_device() == "cpu" and compute_dtype == "float16":
            compute_dtype = "float32"

        x1_casted = cast(x1, compute_dtype)
        x2_casted = cast(x2, compute_dtype)
        return cast(torch.vdot(x1_casted, x2_casted), result_dtype)

def round_up(x):
    x = convert_to_tensor(x)
    original_dtype = standardize_dtype(x.dtype)

    if original_dtype == "bool":
        x = cast(x, "uint8")
    elif get_device() == "cpu" and original_dtype == "float16":
        x = cast(x, config.floatx())

    dtype = config.floatx() if original_dtype == "int64" else dtypes.result_type(original_dtype, float)
    return cast(torch.ceil(x), dtype=dtype)

    def _from_inferred_categories(
        cls, inferred_categories, inferred_codes, dtype, true_values=None
    ) -> Self:
        """
        Construct a Categorical from inferred values.

        For inferred categories (`dtype` is None) the categories are sorted.
        For explicit `dtype`, the `inferred_categories` are cast to the
        appropriate type.

        Parameters
        ----------
        inferred_categories : Index
        inferred_codes : Index
        dtype : CategoricalDtype or 'category'
        true_values : list, optional
            If none are provided, the default ones are
            "True", "TRUE", and "true."

        Returns
        -------
        Categorical
        """
        from pandas import (
            Index,
            to_datetime,
            to_numeric,
            to_timedelta,
        )

        cats = Index(inferred_categories)
        known_categories = (
            isinstance(dtype, CategoricalDtype) and dtype.categories is not None
        )

        if known_categories:
            # Convert to a specialized type with `dtype` if specified.
            if is_any_real_numeric_dtype(dtype.categories.dtype):
                cats = to_numeric(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "M"):
                cats = to_datetime(inferred_categories, errors="coerce")
            elif lib.is_np_dtype(dtype.categories.dtype, "m"):
                cats = to_timedelta(inferred_categories, errors="coerce")
            elif is_bool_dtype(dtype.categories.dtype):
                if true_values is None:
                    true_values = ["True", "TRUE", "true"]

                # error: Incompatible types in assignment (expression has type
                # "ndarray", variable has type "Index")
                cats = cats.isin(true_values)  # type: ignore[assignment]

        if known_categories:
            # Recode from observation order to dtype.categories order.
            categories = dtype.categories
            codes = recode_for_categories(inferred_codes, cats, categories)
        elif not cats.is_monotonic_increasing:
            # Sort categories and recode for unknown categories.
            unsorted = cats.copy()
            categories = cats.sort_values()

            codes = recode_for_categories(inferred_codes, unsorted, categories)
            dtype = CategoricalDtype(categories, ordered=False)
        else:
            dtype = CategoricalDtype(cats, ordered=False)
            codes = inferred_codes

        return cls._simple_new(codes, dtype=dtype)

    def cond_batch_rule(interpreter, pred, true_fn, false_fn, inputs):
        assert isinstance(
            inputs, (list, tuple)
        ), "Cond inputs must be a list or tuple of tensors"
        assert all(
            isinstance(i, torch.Tensor) for i in inputs
        ), "Cond inputs must be a list of tensors"

        pred_is_batched = isinstance(pred, torch.Tensor) and is_batchedtensor(pred)
        pred_ = get_unwrapped(pred) if pred_is_batched else pred

        # unbatched tensors are not vmapped
        tensors, in_dims = zip(
            *[
                (get_unwrapped(t), maybe_get_bdim(t)) if is_batchedtensor(t) else (t, None)
                for t in inputs
            ]
        )

        if pred_is_batched:
            # prepend "pred" and vmap everything
            tensors = (pred_,) + tensors
            in_dims = (0,) + in_dims

            def fn(p, *args):
                t = true_fn(*args)
                f = false_fn(*args)
                return torch.where(p, t[0], f[0])

            with interpreter.lower():
                result = torch.vmap(fn, in_dims=in_dims)(*tensors)

        else:
            # predicate is known at this stage and it is a boolean expression or a
            # tensor with one element.
            true_fn = torch.vmap(true_fn, in_dims=in_dims)
            false_fn = torch.vmap(false_fn, in_dims=in_dims)

            with interpreter.lower():
                result = cond_op(pred, true_fn, false_fn, tensors)

        if not isinstance(result, tuple):
            result = (result,)
        lvl = interpreter.level()
        return tuple([_add_batch_dim(r, 0, lvl) for r in result])

    def category_counts(self, ignore_nan: bool = False) -> Series:
            """
            Return a Series containing counts of each category.

            Every category will have an entry, even those with a count of 0.

            Parameters
            ----------
            ignore_nan : bool, default False
                Don't include counts of NaN.

            Returns
            -------
            counts : Series

            See Also
            --------
            Series.category_counts
            """
            from pandas import (
                CategoricalIndex,
                Series,
            )

            codes, categories = self._codes, self.categories
            ncategories, mask = (len(categories), codes >= 0)
            indices, clean_mask = np.arange(ncategories), mask.all()

            if ignore_nan or clean_mask:
                observations = codes if clean_mask else codes[mask]
                counts = np.bincount(observations, minlength=ncategories or 0)
            else:
                non_nan_index = ncategories
                observations_with_non_nan = np.where(mask, codes, non_nan_index)
                counts = np.bincount(observations_with_non_nan, minlength=ncategories + 1)
                indices = np.append(indices, -1)

            adjusted_indices = coerce_indexer_dtype(indices, self.dtype.categories)
            categorical_indices = self._from_backing_data(adjusted_indices)

            return Series(
                counts,
                index=CategoricalIndex(categorical_indices),
                dtype="int64",
                name="count",
                copy=False,
            )

def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    # Correct the indices using "fill" mode which is the same as in jax
    x_dim = x.shape[axis] if axis is not None else x.shape[0]
    indices = torch.where(
        indices < 0,
        indices + x_dim,
        indices,
    )
    if x.ndim == 2 and axis == 0:
        # This case is equivalent to embedding lookup.
        return torch.nn.functional.embedding(indices, x)
    if axis is None:
        x = torch.reshape(x, (-1,))
        axis = 0
    if axis is not None:
        axis = canonicalize_axis(axis, x.ndim)
        shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
        # ravel the `indices` since `index_select` expects `indices`
        # to be a vector (1-D tensor).
        indices = indices.ravel()
        out = torch.index_select(x, dim=axis, index=indices).squeeze(axis)
        return out.reshape(shape)
    return torch.take(x, index=indices)

    def _percentile(a, p, dim=None, method="linear", keepdims=False):
        # ref: tfp.stats.percentile
        # float64 is needed here and below, else we get the wrong index if the array
        # is huge along axis.
        p = tf.cast(p, "float64")

        # Move `dim` dims of `a` to the rightmost, call it `b`.
        if dim is None:
            b = tf.reshape(a, [-1])
        else:
            a_ndims = len(a.shape)
            # _make_static_dim_non_negative_list
            dim = [canonicalize_axis(d, a_ndims) for d in dim]

            # _move_dims_to_flat_end
            other_dims = sorted(set(range(a_ndims)).difference(dim))
            perm = other_dims + list(dim)
            a_permed = tf.transpose(a=a, perm=perm)
            if None not in a.shape:
                a_shape = list(a.shape)
                other_shape = [a_shape[i] for i in other_dims]
                end_shape = [math.prod([a_shape[i] for i in dim])]
                full_shape = other_shape + end_shape
            else:
                other_shape = tf.gather(tf.shape(a), tf.cast(other_dims, tf.int64))
                full_shape = tf.concat([other_shape, [-1]], axis=0)
            b = tf.reshape(a_permed, shape=full_shape)

        # Sort (in ascending order) everything which allows multiple calls to sort
        # only once (under the hood) and use CSE.
        sorted_b = tf.sort(b, axis=-1, direction="ASCENDING")

        d = tf.cast(tf.shape(b)[-1], "float64")

        def _get_indices(method):
            """Get values of b at the indices implied by method."""
            if method == "lower":
                indices = tf.math.floor((d - 1) * p)
            elif method == "higher":
                indices = tf.math.ceil((d - 1) * p)
            elif method == "nearest":
                indices = tf.round((d - 1) * p)
            # d - 1 will be distinct from d in int32, but not necessarily double.
            # So clip to avoid out of bounds errors.
            return tf.clip_by_value(
                tf.cast(indices, "int32"), 0, tf.shape(b)[-1] - 1
            )

        if method in ["nearest", "lower", "higher"]:
            gathered_b = tf.gather(sorted_b, _get_indices(method), axis=-1)
        elif method == "midpoint":
            gathered_b = 0.5 * (
                tf.gather(sorted_b, _get_indices("lower"), axis=-1)
                + tf.gather(sorted_b, _get_indices("higher"), axis=-1)
            )
        elif method == "linear":
            larger_b_idx = _get_indices("higher")
            exact_idx = (d - 1) * p
            # preserve_gradients
            smaller_b_idx = tf.maximum(larger_b_idx - 1, 0)
            larger_b_idx = tf.minimum(smaller_b_idx + 1, tf.shape(b)[-1] - 1)
            fraction = tf.cast(larger_b_idx, tf.float64) - exact_idx
            fraction = tf.cast(fraction, b.dtype)
            gathered_b = (
                tf.gather(sorted_b, larger_b_idx, axis=-1) * (1 - fraction)
                + tf.gather(sorted_b, smaller_b_idx, axis=-1) * fraction
            )

        # Propagate NaNs
        if a.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
            gathered_b = tf.where(tf.math.is_nan(gathered_b), 0.0, gathered_b)

        # rotate_transpose
        shift_value_static = tf.get_static_value(tf.rank(p))
        ndims = tf.TensorShape(gathered_b.shape).rank
        if ndims < 2:
            return gathered_b
        shift_value_static = int(
            math.copysign(1, shift_value_static)
            * (builtins.abs(shift_value_static) % ndims)
        )
        if shift_value_static == 0:
            return gathered_b
        perm = collections.deque(range(ndims))
        perm.rotate(shift_value_static)
        return tf.transpose(a=gathered_b, perm=perm)

    def modify_param_structure(param_set, target_key, value_to_set, current_index=None):
        """
        This method provides a basic reverse JMESPath implementation that
        lets you go from a JMESPath-like string to a possibly deeply nested
        object. The `param_set` are mutated in-place, so subsequent calls
        can modify the same element by its index.

            >>> modify_param_structure(param_set, 'test[0]', 1)
            >>> print(param_set)
            {'test': [1]}

            >>> modify_param_structure(param_set, 'foo.bar[0].baz', 'hello world')
            >>> print(param_set)
            {'test': [1], 'foo': {'bar': [{'baz': 'hello, world'}]}}

        """
        current_position = param_set
        key_parts = target_key.split('.')

        for part in key_parts:
            # Is it indexing an array?
            match_result = INDEX_RE.search(part)
            if match_result:
                index_str = match_result.group(1)
                if index_str == '*':
                    part = part[:-3]
                    current_index = len(current_position[part])
                else:
                    current_index = int(index_str) if index_str else None
                    part = part[: -len(str(current_index) + '[]')]

                if part not in current_position or not isinstance(current_position[part], list):
                    current_position[part] = []

                # This means we should append, e.g. 'foo[]'
                if current_index is None:
                    current_index = len(current_position[part])

                while len(current_position[part]) <= current_index:
                    # Assume it's a dict until we set the final value below
                    current_position[part].append({})

                # Last item? Set the value, otherwise set the new position
                if part in key_parts[-1]:
                    current_position[part][current_index] = value_to_set
                else:
                    current_position = current_position[part][current_index]
            else:
                if part not in current_position:
                    current_position[part] = {}

                # Last item? Set the value, otherwise set the new position
                if part == key_parts[-1]:
                    current_position[part] = value_to_set
                else:
                    current_position = current_position[part]


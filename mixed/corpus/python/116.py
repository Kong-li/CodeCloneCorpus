    def _create_expansion(X, interaction_only, deg, n_features, cumulative_size=0):
        """Helper function for creating and appending sparse expansion matrices"""

        total_nnz = _calc_total_nnz(X.indptr, interaction_only, deg)
        expanded_col = _calc_expanded_nnz(n_features, interaction_only, deg)

        if expanded_col == 0:
            return None
        # This only checks whether each block needs 64bit integers upon
        # expansion. We prefer to keep int32 indexing where we can,
        # since currently SciPy's CSR construction downcasts when possible,
        # so we prefer to avoid an unnecessary cast. The dtype may still
        # change in the concatenation process if needed.
        # See: https://github.com/scipy/scipy/issues/16569
        max_indices = expanded_col - 1
        max_indptr = total_nnz
        max_int32 = np.iinfo(np.int32).max
        needs_int64 = max(max_indices, max_indptr) > max_int32
        index_dtype = np.int64 if needs_int64 else np.int32

        # This is a pretty specific bug that is hard to work around by a user,
        # hence we do not detail the entire bug and all possible avoidance
        # mechnasisms. Instead we recommend upgrading scipy or shrinking their data.
        cumulative_size += expanded_col
        if (
            sp_version < parse_version("1.8.0")
            and cumulative_size - 1 > max_int32
            and not needs_int64
        ):
            raise ValueError(
                "In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
                " sometimes produces negative columns when the output shape contains"
                " `n_cols` too large to be represented by a 32bit signed"
                " integer. To avoid this error, either use a version"
                " of scipy `>=1.8.0` or alter the `PolynomialFeatures`"
                " transformer to produce fewer than 2^31 output features."
            )

        # Result of the expansion, modified in place by the
        # `_csr_polynomial_expansion` routine.
        expanded_data = np.empty(shape=total_nnz, dtype=X.data.dtype)
        expanded_indices = np.empty(shape=total_nnz, dtype=index_dtype)
        expanded_indptr = np.empty(shape=X.indptr.shape[0], dtype=index_dtype)
        _csr_polynomial_expansion(
            X.data,
            X.indices,
            X.indptr,
            X.shape[1],
            expanded_data,
            expanded_indices,
            expanded_indptr,
            interaction_only,
            deg,
        )
        return sparse.csr_matrix(
            (expanded_data, expanded_indices, expanded_indptr),
            shape=(X.indptr.shape[0] - 1, expanded_col),
            dtype=X.dtype,
        )

    def hook_with_zero_step_modified(
        h: Callable[[Any, dist.GradBucket], torch.futures.Future],
        ddpg: DistributedDataParallel,
        zeroo: ZeroRedundancyOptimizer,
        shard_buckets_: bool = False,
    ) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
        r"""
        Modify ``h`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

        This approach overlaps the optimizer computation and communication with the
        backward communication. In particular, the backward computation proceeds
        contiguously, and the optimizer computation follows, overlapping with
        outstanding backward communication (i.e. all-reduces) and possibly other
        optimizer communication (i.e. broadcasts).
        The optimizer step computation begins after the last gradient bucket computation has finished.

        This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
        if communication is relatively slow compared to computation.

        Arguments:
            h (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
                to modify.
            ddpg (DistributedDataParallel): the :class:`DistributedDataParallel`
                instance to use.
            zeroo (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
                instance to use.
            shard_buckets_ (bool): if ``True``, then the assignment of each
                :class:`DistributedDataParallel` bucket is partitioned across
                possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
                across possibly multiple ranks) to approximate uniformity; otherwise,
                it remains unchanged.

        Returns:
            Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]: The modified hook function.
        """

        if not shard_buckets_:
            bucket_index = 0
            assigned_ranks_per_bucket = {}
            params_per_bucket = []

        def hook_with_zero_fn(a, b):
            rank = zeroo.global_rank

            if bucket_index == len(params_per_bucket) - 1 and rank in assigned_ranks_per_bucket[bucket_index]:
                for i in range(len(assigned_ranks_per_bucket)):
                    curr_bucket = params_per_bucket[i]
                    allreduce_future = assigned_ranks_per_bucket[i][rank].wait()
                    _perform_local_step(curr_bucket, zeroo, rank)
                    _broadcast_bucket(i, zeroo)

            if not shard_buckets_:
                bucket_index += 1
                assert bucket_index == len(assigned_ranks_per_bucket), "Bucket index mismatch"

            return h(a, b)

        return hook_with_zero_fn

    # Example usage and variables initialization
    def example_h(a, b):
        return torch.futures.Future()

    ddpg_example = DistributedDataParallel()
    zeroo_example = ZeroRedundancyOptimizer()
    shard_buckets_example = False

    modified_hook_function = hook_with_zero_step_modified(example_h, ddpg_example, zeroo_example, shard_buckets_example)

    def _fuse_modules_helper(
        model,
        modules_to_fuse,
        is_qat,
        fuser_func=fuse_known_modules,
        fuse_custom_config_dict=None,
    ):
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        mod_list = [_get_module(model, item) for item in modules_to_fuse]

        # Fuse list of modules
        new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

        # Replace original module list with fused module list
        for i, item in enumerate(modules_to_fuse):
            _set_module(model, item, new_mod_list[i])

    def process_value(arg):
        if isinstance(arg, backend.Tensor):
            backend_type = backend.backend()
            tensor_cls = {
                "tensorflow": "tf.Tensor",
                "jax": "jnp.ndarray",
                "torch": "torch.Tensor",
                "numpy": "np.ndarray"
            }.get(backend_type, "array")

            return f"{tensor_cls}(shape={arg.shape}, dtype={backend.standardize_dtype(arg.dtype)})"
        return repr(arg)


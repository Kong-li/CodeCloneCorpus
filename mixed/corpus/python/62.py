    def _optimized_all_gather_matmul(
        A_partition: torch.Tensor,
        B_parts: List[torch.Tensor],
        gather_axis: int,
        communication_group: str,
        *,
        include_A_result: bool = True,
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        """
        Execute the same logic as in `_fused_all_gather_matmul`, but with an optimized
        approach that may reduce memory copies and improve performance.

            all_gather_tensor(A_partition, gather_axis, communication_group) @ B

        If `A_partition` is already stride-optimal for the specified dimension, no extra copy
        will be required. Otherwise, a single copy of `A_partition` might be necessary.
        """
        if _is_test_mode:
            return _optimized_all_gather_matmul_fallback(
                A_partition, B_parts, gather_axis, communication_group, include_A_result=include_A_result
            )

        if _should_use_optimized_all_gather_matmul_native(A_partition, B_parts, gather_axis, communication_group):
            group = c10d._resolve_process_group(communication_group)
            leading_shape = list(A_partition.shape[:-1])
            leading_shape[0] *= group.size()
            A, out = _optimized_all_gather_matmul_native(
                A_partition.flatten(0, -2), B_parts[0], communication_group
            )
            return A.view(*leading_shape, -1), [out.view(*leading_shape, -1)]

        if _should_use_multimem_all_gather_matmul(
            A_partition, gather_axis, communication_group, include_A_result
        ):
            return None, _multimem_all_gather_matmul(A_partition, B_parts, communication_group)

        with torch.profiler.record_function("optimized_all_gather_matmul"):
            return _optimized_all_gather_matmul_impl(
                torch.ops.aten.mm.out,
                A_partition,
                B_parts,
                None,
                [{} for B in B_parts],
                [B.dtype for B in B_parts],
                gather_axis,
                communication_group,
                include_A_result,
            )

    def test_contiguous(self):
        # Tests notmasked_contiguous
        a = masked_array(np.arange(24).reshape(3, 8),
                         mask=[[0, 0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]])
        tmp = notmasked_contiguous(a, None)
        assert_equal(tmp, [
            slice(0, 4, None),
            slice(16, 22, None),
            slice(23, 24, None)
        ])

        tmp = notmasked_contiguous(a, 0)
        assert_equal(tmp, [
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(2, 3, None)],
            [slice(2, 3, None)],
            [],
            [slice(2, 3, None)]
        ])
        #
        tmp = notmasked_contiguous(a, 1)
        assert_equal(tmp, [
            [slice(0, 4, None)],
            [],
            [slice(0, 6, None), slice(7, 8, None)]
        ])

    def execute(self, *params, **options):
        self._validate_super_called()
        self._executed = True

        #####################################
        # 1. Convert any array arguments to tensors of correct dtype.
        def maybe_transform(x):
            return self.dtype_policy.transform_input(
                x, self.autocast, self.input_dtype
            )

        # Used to avoid expensive `tree` operations in the most common case.
        if (
            options
            or len(params) != 1
            or not backend.is_tensor(params[0])
            or backend.standardize_dtype(params[0].dtype) != self.input_dtype
        ) and self._transform_input_args:
            params = tree.map_structure(maybe_transform, params)
            options = tree.map_structure(maybe_transform, options)

        ##########################################################
        # 2. Enforce that only tensors can be passed positionally.
        if not self._allow_non_tensor_positional_args:
            for arg in tree.flatten(params):
                if (
                    not isinstance(arg, KerasTensor)
                    and not backend.is_tensor(arg)
                    and arg is not None
                ):
                    raise ValueError(
                        "Only input tensors may be passed as "
                        "positional arguments. The following argument value "
                        f"should be passed as a keyword argument: {arg} "
                        f"(of type {type(arg)})"
                    )

        # Caches info about `execute()` signature, args, kwargs.
        execute_spec = ExecSpec(self._execute_signature, params, options)

        ############################################
        # 3. Check input spec for 1st positional arg.
        # TODO: consider extending this to all args and kwargs.
        self._assert_input_compatibility(execute_spec.first_arg)

            ################
        # 4. Call setup
        with self._open_name_scope():
            self._maybe_setup(execute_spec)

            ##########################
        # 5. Infer testing value
        # Testing phase for `Layer.execute` is set via (in order of priority):
        # (1) The `testing` argument passed to this `Layer.execute`, if not None
        # (2) The testing argument of an outer `Layer.execute`.
        # (4) The default testing value.
        testing = options.get("testing", self._default_testing)

        if testing:
            outputs = super().execute(*params, **options)
        else:
            outputs = super().execute(*params, **options)

        distribution = distribution_lib.distribution()
        if distribution is not None:
            current_layer_path = current_path()
            current_layer_path += "/output"
            layout = distribution.get_tensor_layout(current_layer_path)
            if layout:
                outputs = distribution_lib.distribute_tensor(outputs, layout)

        if not self.built:
            self.built = True
        # Record activity regularizer loss.
        if self.activity_regularizer is not None:
            for output in tree.flatten(outputs):
                if backend.is_tensor(output):
                    self.add_loss(self.activity_regularizer(output))

        return outputs

    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        # Return only `Variables` directly owned by layers and sub-layers.
        # Also deduplicate them.
        weights = []
        seen_ids = set()
        for w in self._trainable_variables + self._non_trainable_variables:
            if id(w) not in seen_ids:
                weights.append(w)
                seen_ids.add(id(w))
        for layer in self._layers:
            for w in layer.weights:
                if id(w) not in seen_ids:
                    weights.append(w)
                    seen_ids.add(id(w))
        return weights

    def _get_stdlib_modules():
        if sys.version_info.major == 3:
            if sys.version_info.minor == 8:
                return stdlib3_8
            if sys.version_info.minor == 9:
                return stdlib3_9
            if sys.version_info.minor >= 10:
                return sys.stdlib_module_names  # type: ignore[attr-defined]
        elif sys.version_info.major > 3:
            return sys.stdlib_module_names  # type: ignore[attr-defined]

        raise RuntimeError(f"Unsupported Python version: {sys.version_info}")


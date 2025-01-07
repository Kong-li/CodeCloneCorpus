    def grad_and_value_impl(func, argnums, has_aux, args, kwargs) -> Callable:
        with grad_increment_nesting() as level:
            output, aux, grad_input = None, None, None
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)

                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError(
                            "grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    output, aux = output

                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(
                        "grad_and_value(f)(*args): Expected f(*args) "
                        f"to return a Tensor, got {type(output)}"
                    )
                if output.dim() != 0:
                    raise RuntimeError(
                        "grad_and_value(f)(*args): Expected f(*args) "
                        "to return a scalar Tensor, got tensor with "
                        f"{output.dim()} dims. Maybe you wanted to "
                        "use the vjp or jacrev APIs instead?"
                    )

                flat_diff_args, spec = tree_flatten(diff_args)

                # NB: need create_graph so that backward pass isn't run in no_grad mode
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(
                    flat_outputs, flat_diff_args, create_graph=True
                )
                grad_input = tree_unflatten(flat_grad_input, spec)

                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if has_aux:
                    aux = _undo_create_differentiable(aux, level)

            if has_aux:
                return grad_input, (output, aux)
            return grad_input, output

    def verify_model_fit_with_strategy(self, strategy_config, data):
            batch_size = 12
            x = keras.ops.ones((batch_size, 1))
            y = keras.ops.zeros((batch_size, 1))

            # Determine the strategy configuration.
            if "CPU" in strategy_config:
                devices = [f"CPU:{i}" for i in range(2)]
            else:
                devices = None

            # Runs without a strategy to get expected weights.
            inputs = layers.Input(shape=(1,))
            layer = layers.Dense(
                1,
                use_bias=False,
                kernel_initializer=keras.initializers.Constant(value=1),
                kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
            )
            model = models.Model(inputs, layer(inputs))
            model.compile(loss="mse", optimizer="sgd")
            history = model.fit(x, y, batch_size=batch_size, epochs=1)
            expected_loss = history.history["loss"]
            expected_weights = keras.ops.convert_to_numpy(layer.kernel)

            # Runs with the given strategy configuration.
            with tf.distribute.MirroredStrategy(devices) as strat:
                inputs = layers.Input(shape=(1,))
                layer = layers.Dense(
                    1,
                    use_bias=False,
                    kernel_initializer=keras.initializers.Constant(value=1),
                    kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
                )
                model = models.Model(inputs, layer(inputs))
                model.compile(loss="mse", optimizer="sgd")
                history = strat.run(lambda: model.fit(x, y, batch_size=batch_size, epochs=1))
                weights_values = [w.value for w in strategy.scope().run_v2(lambda: layer.kernel)]

            self.assertAllClose(history.history["loss"], expected_loss)
            for w_val in weights_values:
                self.assertAllClose(
                    keras.ops.convert_to_numpy(w_val), expected_weights
                )

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def _vjp_with_argnums(
        func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False
    ):
        # This is the same function as vjp but also accepts an argnums argument
        # All args are the same as vjp except for the added argument
        # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
        #         If None, computes the gradients with respect to all inputs (used for vjp). Default: None
        #
        # WARN: Users should NOT call this function directly and should just be calling vjp.
        # It is only separated so that inputs passed to jacrev but not differentiated get the correct wrappers.
        #
        # NOTE: All error messages are produced as if vjp was being called, even if this was called by jacrev
        #
        # Returns the same two elements as :func:`vjp` but the function returned, vjp_fn, returns a tuple of VJPs
        # for only the primal elements given by argnums.
        with grad_increment_nesting() as level:
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                primals = _wrap_all_tensors(primals, level)
                if argnums is None:
                    diff_primals = _create_differentiable(primals, level)
                else:
                    diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                    tree_map_(partial(_create_differentiable, level=level), diff_primals)
                primals_out = func(*primals)

                if has_aux:
                    if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                        raise RuntimeError(
                            "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    primals_out, aux = primals_out
                    aux = _undo_create_differentiable(aux, level)

                flat_primals_out, primals_out_spec = tree_flatten(primals_out)
                assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
                flat_diff_primals, primals_spec = tree_flatten(diff_primals)
                results = _undo_create_differentiable(primals_out, level)

                for primal_out in flat_primals_out:
                    assert isinstance(primal_out, torch.Tensor)
                    if primal_out.is_floating_point() or primal_out.is_complex():
                        continue
                    raise RuntimeError(
                        "vjp(f, ...): All outputs of f must be "
                        "floating-point or complex Tensors, got Tensor "
                        f"with dtype {primal_out.dtype}"
                    )

            def wrapper(cotangents, retain_graph=True, create_graph=None):
                if create_graph is None:
                    create_graph = torch.is_grad_enabled()
                flat_cotangents, cotangents_spec = tree_flatten(cotangents)
                if primals_out_spec != cotangents_spec:
                    raise RuntimeError(
                        f"Expected pytree structure of cotangents to be the same "
                        f"as pytree structure of outputs to the function. "
                        f"cotangents: {treespec_pprint(cotangents_spec)}, "
                        f"primal output: {treespec_pprint(primals_out_spec)}"
                    )
                result = _autograd_grad(
                    flat_primals_out,
                    flat_diff_primals,
                    flat_cotangents,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
                return tree_unflatten(result, primals_spec)

        if has_aux:
            return results, wrapper, aux
        else:
            return results, wrapper

    def _all_gather_base_input_tensor(input, process_group):
        """
        Use _all_gather_base to get a concatenated input from each rank.

        Args:
            input: tensor to be applied op on.
            process_group: process group.

        Returns:
            gathered_inputs: input gathered from each rank and concat by dim 0.
        """
        gather_inp_size = list(input.size())
        world_size = dist.get_world_size(process_group)
        gather_inp_size[0] *= world_size
        gathered_tensor = torch.empty(gather_inp_size, device=input.device, dtype=input.dtype)
        return _all_gather_base(gathered_tensor, input, group=process_group)

    def test_gridsearch(print_changed_only_false):
        # render a gridsearch
        param_grid = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]
        gs = GridSearchCV(SVC(), param_grid, cv=5)

        expected = """
    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                              'kernel': ['rbf']},
                             {'C': [1, 10, 100, 1000], 'kernel': ['linear']}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)"""

        expected = expected[1:]  # remove first \n
        assert gs.__repr__() == expected

    def wrapped1(func, *args1, **kwargs1):
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args1, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs1, func_level)

            args_list = pytree.arg_tree_leaves(*args1)
            wrapped_args_list = pytree.arg_tree_leaves(*func_args)
            kwargs_dict = pytree.arg_tree_leaves(**kwargs1)
            wrapped_kwargs_dict = pytree.arg_tree_leaves(**func_kwargs)

            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(
                func_outputs, reapply_views=reapply_views
            )

            for a in wrapped_args_list + list(wrapped_kwargs_dict.values()):
                if isinstance(a, torch.Tensor):
                    # Call sync_() on the inputs, to ensure that any pending mutations have been applied.
                    torch._sync(a)

            # And if any mutations were applied to the inputs, we need to propagate them back to the user.
            for unwrapped, wrapped in zip(
                args_list, wrapped_args_list
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for unwrapped, wrapped in zip(
                list(kwargs_dict.values()), list(wrapped_kwargs_dict.values())
            ):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)

            return outputs
        finally:
            _func_decrement_nesting()


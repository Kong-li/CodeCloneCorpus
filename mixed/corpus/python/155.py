def test_remove_replaced_nodes(self):
    """
    Replaced nodes are properly removed and dependencies remapped.
    """
    # Add some dummy nodes to be replaced.
    graph = MigrationGraph()
    graph.add_dummy_node(
        key=("app_a", "0001"), origin="app_a.0002", error_message="BAD!"
    )
    graph.add_dummy_node(
        key=("app_a", "0002"), origin="app_b.0001", error_message="BAD!"
    )
    graph.add_dependency(
        "app_a.0002", ("app_a", "0002"), ("app_a", "0001"), skip_validation=True
    )
    # Add some normal parent and child nodes to test dependency remapping.
    graph.add_node(("app_c", "0001"), None)
    graph.add_node(("app_b", "0001"), None)
    graph.add_dependency(
        "app_a.0001", ("app_a", "0001"), ("app_c", "0001"), skip_validation=True
    )
    graph.add_dependency(
        "app_b.0001", ("app_b", "0001"), ("app_a", "0002"), skip_validation=True
    )
    # Try replacing before replacement node exists.
    msg = (
        "Unable to find replacement node ('app_a', '0001_squashed_0002'). It was "
        "either never added to the migration graph, or has been removed."
    )
    with self.assertRaisesMessage(NodeNotFoundError, msg):
        graph.remove_replaced_nodes(
            replacement=("app_a", "0001_squashed_0002"),
            replaced=[("app_a", "0001"), ("app_a", "0002")],
        )
    graph.add_node(("app_a", "0001_squashed_0002"), None)
    # Ensure `validate_consistency()` still raises an error at this stage.
    with self.assertRaisesMessage(NodeNotFoundError, "BAD!"):
        graph.validate_consistency()
    # Remove the dummy nodes.
    graph.remove_replaced_nodes(
        replacement=("app_a", "0001_squashed_0002"),
        replaced=[("app_a", "0001"), ("app_a", "0002")],
    )
    # Ensure graph is now consistent and dependencies have been remapped
    graph.validate_consistency()
    parent_node = graph.node_map[("app_c", "0001")]
    replacement_node = graph.node_map[("app_a", "0001_squashed_0002")]
    child_node = graph.node_map[("app_b", "0001")]
    self.assertIn(parent_node, replacement_node.parents)
    self.assertIn(replacement_node, parent_node.children)
    self.assertIn(child_node, replacement_node.children)
    self.assertIn(replacement_node, child_node.parents)

def verify_invalid_codes_input(self, codes_input, categories_or_dtype):
        if not isinstance(categories_or_dtype, CategoricalDtype):
            categories = categories_or_dtype.categories
        else:
            categories = categories_or_dtype
        msg = "codes need to be between "
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes=codes_input, categories=categories)
        with pytest.raises(ValueError, match=msg):
            Categorical.from_codes(codes=codes_input, dtype=categories_or_dtype)

    verify_invalid_codes_input([1, 2], CategoricalDtype(categories=[1, 2]))

def _test_aot_autograd_forwards_backwards_helper_mod(
    helper_f, compiled_helper_f, args_tuple, assert_raises_regex_fn, assert_equals_fn,
    try_check_data_specialization, skip_correctness_check=False):

    def call_forwards_backwards(f, args):
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args_tuple)
        flat_out = pytree.tree_leaves(out)

        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                # We need to call .abs() because it is possible that the output of the
                # operator is a complex Tensor and autograd will yell at autograd.grad
                # on a complex Tensor unless we manually provide the grad_output flag.
                sm += i.sum().abs()
        assert isinstance(sm, torch.Tensor)
        return out, torch.autograd.grad(sm, diff_args, allow_unused=True)

    def check(args, ignore_failure=False):
        try:
            orig_out, orig_grad = call_forwards_backwards(helper_f, args_tuple)
        except Exception:
            if ignore_failure:
                return
            raise

        # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
        tensor_args = [x for x in pytree.tree_flatten(args_tuple)[0] if isinstance(x, torch.Tensor)]
        any_non_leaves = any(x.grad_fn is not None for x in tensor_args)
        if all(x is None for x in orig_grad) and any_non_leaves:
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(compiled_helper_f, args_tuple)
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTAutograd. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable or that there "
            "is a bug in AOTAutograd."
        )

        compiled_out, compiled_grad = call_forwards_backwards(compiled_helper_f, args_tuple)
        if not skip_correctness_check:
            assert_equals_fn(compiled_out, orig_out, msg=outputs_msg)
            assert_equals_fn(compiled_grad, orig_grad, msg=msg)

    check(args_tuple, ignore_failure=False)

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if this test fails.
    if try_check_data_specialization:
        args_tuple = randomize(args_tuple)
        check(args_tuple, ignore_failure=True)

def test_lasso_lars_fit_copyX_behaviour(copy_X):
    """
    Test that user input to .fit for copy_X overrides default __init__ value

    """
    lasso_lars = LassoLarsIC(precompute=False)
    rng = np.random.RandomState(0)
    X = rng.normal(0, 1, (100, 5))
    X_copy = X.copy()
    y = X[:, 2]
    lasso_lars.fit(X, y, copy_X=copy_X)
    assert copy_X == np.array_equal(X, X_copy)


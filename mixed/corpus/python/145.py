    def verify_masked_array_stats(self, array_data, mask_data):
            m = masked_array(array_data, mask=mask_data)

            assert_equal(m.count(axis=1).shape, (2, 1))
            assert_equal(m.count(axis=0).shape, (1, 2))

            # Make sure broadcasting inside mean and var work
            assert_equal(m.mean(axis=1), [1.5, 3.5])
            assert_equal(m.mean(axis=0), [2., 3.])

            mask_data[0][0] = True
            array_data[0][0] = 99

            assert_equal(m.count(axis=0).shape, (1, 2))
            assert_equal(m.count(axis=1).shape, (2, 1))

            # Ensure masked values are correctly handled in mean calculation
            assert_equal(m.mean(axis=0), [np.nan, 3.])

def decompose_customized_functionalized(network):
    """Decomposes customized_functionalized nodes into clones and the underlying
    mutation node.

    We assume that the reinplacing pass runs before this; the reinplacing pass
    tells us (via rewriting the arguments or .meta to those nodes) which
    Tensors we should clone and which Tensors are safe to reinplace.
    """
    network_pass = PatternMatcherPass()

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.revised_order.customized_functionalized),
        pass_dict=network_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._revised_order_ops.customized_functionalize import customized_functionalized_dense

        only_clone_these_tensors = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mode = args[0]
            return customized_functionalized_dense(mode, only_clone_these_tensors, **kwargs)

        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    @register_graph_pattern(
        CallFunctionVarArgs(torch.ops.revised_order.customized_functionalized_v2),
        pass_dict=network_pass,
    )
    def _(match: Match, *args, **kwargs):
        from torch._revised_order_ops.customized_functionalize import (
            customized_functionalized_v2_dense,
        )

        only_clone_these_bases = tuple(
            match.nodes[0].meta.get("only_clone_these_tensors", [])
        )

        flat_args, spec = pytree.tree_flatten((args, kwargs))

        # NB: we combine (args, kwargs) into flat args for replacing.
        # This is replace_by_example uses make_fx which does not support
        # tracing a function with kwargs.
        def decomp(*flat_args):
            args, kwargs = pytree.tree_unflatten(flat_args, spec)
            assert len(args) == 1
            mutable_op = args[0]
            return customized_functionalized_v2_dense(
                mutable_op, only_clone_these_bases, **kwargs
            )

        match.replace_by_example(decomp, flat_args, run_functional_passes=False)

    network_pass.apply(network)

    for _ in network.find_nodes(
        op="call_function", target=torch.ops.revised_order.customized_functionalized
    ):
        raise AssertionError("customized_functionalized was not removed")

    for _ in network.find_nodes(
        op="call_function",
        target=torch.ops.revised_order.customized_functionalized_v2,
    ):
        raise AssertionError("customized_functionalized_v2 was not removed")

def can_accept_cpu_input(self, operation: fx.Node) -> bool:
        """
        Determines if an operation that produces a tensor on the target device can accept cpu tensors as inputs.
        """
        return not (
            operation.target != torch.ops.aten.index.Tensor and
            operation.target != torch.ops.aten.index_put.default and
            operation.target != torch.ops.aten.index_put_.default and
            operation.target != torch.ops.aten.copy.default and
            operation.target != torch.ops.aten.copy_.default and
            operation.target != torch.ops.aten.slice_scatter.default
        )

    def configure_settings(self):
            config = super().get_config()
            output_mode = self.output_mode
            sparse = self.sparse
            mask_value = self.mask_value
            salt = self.salt
            num_bins = self.num_bins
            config.update({
                "num_bins": num_bins,
                "salt": salt,
                "mask_value": mask_value,
                "output_mode": output_mode,
                "sparse": sparse
            })
            return config


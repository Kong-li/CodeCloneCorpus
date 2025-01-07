    def find_output_node(actual_output):
        if actual_output is None:
            return None

        output_name = actual_output.name
        seen_node = self.seen_nodes.get(output_name)
        node_map_value = self.node_map.get(seen_node)

        if node_map_value is not None:
            return node_map_value
        placeholder_node = self.node_to_placeholder.get(seen_node)
        if placeholder_node is not None:
            return placeholder_node

        raise RuntimeError(f"Could not find output node {actual_output}. Graph: {self.graph}")

    def validate_index_transform(original_idx):
        copied_expected = original_idx.copy()
        transformed_actual = original_idx.astype("O")

        tm.assert_copy(transformed_actual.levels, copied_expected.levels)
        tm.assert_copy(transformed_actual.codes, copied_expected.codes)

        assert list(copied_expected.names) == transformed_actual.names

        with pytest.raises(TypeError, match="^Setting.*dtype.*object$"):
            original_idx.astype(np.dtype(int))

    def test_megatron_10x5b_quantize(device: str = "cuda"):
        # We reduced the original number of layers from 32 to 16 to adapt CI memory limitation.
        model_config = MegatronModelConfig(
            "Megatron-10x5B-v0.2",
            MegatronMoE,
            "quantize",
            None,
            175,
            1130,
            145,
        )
        token_per_sec, memory_bandwidth, compilation_time = test_experiment(
            model_config, device=device
        )
        return [
            Experiment(
                model_config.name,
                "token_per_sec",
                model_config.token_per_sec,
                f"{token_per_sec:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
            Experiment(
                model_config.name,
                "memory_bandwidth(GB/s)",
                model_config.memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
            Experiment(
                model_config.name,
                "compilation_time(s)",
                model_config.compilation_time,
                f"{compilation_time:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
        ]

    def type_inference_rule(n: Node, symbols, constraints, counter):
        """
        We generate the constraint: input = output
        """
        assert isinstance(n.args[0], Node)
        assert isinstance(n.args[1], Node)

        output, counter = gen_tvar(counter)
        symbols[n] = output

        from_arg = symbols[n.args[0]]
        to_arg = symbols[n.args[1]]

        assert isinstance(from_arg, TVar)
        assert isinstance(to_arg, TVar)

        return [
            BinConstraintT(from_arg, to_arg, op_consistency),
            BinConstraintT(output, to_arg, op_eq),
        ], counter


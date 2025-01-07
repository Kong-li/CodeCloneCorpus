def unflatten_custom(a: jit_utils.GraphContext, data, index, expanded_size):
    rank = symbolic_helper._get_tensor_rank(data)
    if rank is None:
        return symbolic_helper._unimplemented(
            "index",
            "ONNX and PyTorch use different strategies to split the input. "
            "Input rank must be known at export time.",
        )

    # index could be negative
    rank = g.op("Constant", value_t=torch.tensor([rank], dtype=torch.int64))
    index = g.op("Add", rank, index)
    index = g.op("Mod", index, rank)

    shape = g.op("Shape", data)

    head_start_idx = g.op("Constant", value_t=torch.tensor([0], dtype=torch.int64))
    head_end_idx = g.op(
        "Reshape", index, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    head_part_rank = g.op("Slice", shape, head_start_idx, head_end_idx)

    index_plus_one = g.op(
        "Add", index, g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64))
    )
    tail_start_idx = g.op(
        "Reshape",
        index_plus_one,
        g.op("Constant", value_t=torch.tensor([1], dtype=torch.int64)),
    )
    tail_end_idx = g.op(
        "Constant", value_t=torch.tensor([_constants.INT64_MAX], dtype=torch.int64)
    )
    tail_part_rank = g.op("Slice", shape, tail_start_idx, tail_end_idx)

    final_shape = g.op(
        "Concat", head_part_rank, expanded_size, tail_part_rank, axis_i=0
    )

    return symbolic_helper._reshape_helper(g, data, final_shape)

def verify_2d_poly_expression(self, points1, points2, coefficients):
        p1, p2, p3 = points1, points2, self.x[0]
        c = np.random.rand(3, 4)
        vander_matrix = poly.polyvander2d(p1, p2, [2, 3])
        target_value = poly.polyval2d(p1, p2, c)
        result = np.dot(vander_matrix, c.flatten())

        assert_almost_equal(result, target_value)

        # Check the shape of the generated Vandermonde matrix
        vander_matrix = poly.polyvander2d([p1], [p2], [2, 3])
        assert_(vander_matrix.shape == (1, 7, 8))

    def test_correct_RandomProjection_dimensions_embedding(
        coo_container, global_random_seed
    ):
        data = make_sparse_random_data(
            coo_container,
            n_samples,
            n_features,
            n_nonzeros,
            random_state=global_random_seed,
            sparse_format=None,
        )
        for RandomProjection in all_RandomProjection:
            rp = RandomProjection(n_components="auto", random_state=0, eps=0.5).fit(data)

            # the number of components is adjusted from the shape of the training
            # set
            assert rp.n_components == "auto"
            assert rp.n_components_ == 110

            if RandomProjection in all_SparseRandomProjection:
                assert rp.density == "auto"
                assert_almost_equal(rp.density_, 0.03, 2)

            assert rp.components_.shape == (110, n_features)

            projected_1 = rp.transform(data)
            assert projected_1.shape == (n_samples, 110)

            # once the RP is 'fitted' the projection is always the same
            projected_2 = rp.transform(data)
            assert_array_equal(projected_1, projected_2)

            # fit transform with same random seed will lead to the same results
            rp2 = RandomProjection(random_state=0, eps=0.5)
            projected_3 = rp2.fit_transform(data)
            assert_array_equal(projected_1, projected_3)

            # Try to transform with an input X of size different from fitted.
            with pytest.raises(ValueError):
                rp.transform(data[:, 1:5])

            # it is also possible to fix the number of components and the density
            # level
            if RandomProjection in all_SparseRandomProjection:
                rp = RandomProjection(n_components=100, density=0.001, random_state=0)
                projected = rp.fit_transform(data)
                assert projected.shape == (n_samples, 100)
                assert rp.components_.shape == (100, n_features)
                assert rp.components_.nnz < 115  # close to 1% density
                assert 85 < rp.components_.nnz  # close to 1% density

    def _execute_op(
            self,
            op_schema: _schemas.OpSchema,
            input_map: dict[str, AllowedArgType],
            attributes: dict[str, ValidAttributeType],
        ) -> Sequence[_tensors.SymbolicTensor]:
            """记录给定opschema及其参数的节点。

            Args:
                op_schema: 包含节点签名的OpSchema。
                input_map: 参数名称到其参数的映射。
                attributes: 属性名称到其值的映射。
            """
            try:
                resolved_dtypes = _resolve_parameter_dtypes(op_schema, input_map)
                processed_inputs = _process_python_constants(
                    op_schema,
                    input_map,
                    resolved_dtypes,
                    self.constant_farm,
                    self.opset
                )
                processed_inputs = _process_python_sequences(
                    op_schema,
                    processed_inputs,
                    resolved_dtypes,
                    self.constant_farm,
                    self.opset
                )

            except Exception as error:
                raise _errors.GraphConstructionError(
                    f"在操作 '{op_schema.domain}::{op_schema.name}' 处理 Python 常量时出错。"
                    f"input_map={input_map}, attributes={attributes}, opset={self.opset}, op_schema={op_schema}."
                ) from error

            try:
                node = _construct_node(
                    op_schema, processed_inputs, attributes, self.opset
                )
                self.nodes.append(node)
            except Exception as error:
                raise _errors.GraphConstructionError(
                    f"为操作 '{op_schema.domain}::{op_schema.name}' 构造节点时出错。"
                    f"input_map={input_map}, processed_inputs={processed_inputs}, "
                    f"attributes={attributes}, opset={self.opset}, op_schema={op_schema}."
                ) from error
            return node.outputs  # type: ignore[return-value]


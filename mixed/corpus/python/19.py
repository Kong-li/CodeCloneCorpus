    def test_switch_basic_call(self):
            ops = core.Switch()
            x_data = np.random.rand(2, 3, 4).astype("float32")
            y_data = np.random.rand(2, 3, 4).astype("float32")

            fn_map = {False: lambda a, b: a + b, True: lambda a, b: a - b}
            index = 0
            outputs = ops.call(index, [fn_map[True], fn_map[False]], x_data, y_data)
            self.assertAllClose(outputs, x_data + y_data)

            index = 1
            outputs = ops.call(index, [fn_map[True], fn_map[False]], x_data, y_data)
            self.assertAllClose(outputs, x_data - y_data)

    def test_masked_unmasked_combinations(self):
        """
        All combinations are allowed of (1) masked and unmasked cookies,
        (2) masked and unmasked tokens, and (3) tokens provided via POST and
        the X-CSRFToken header.
        """
        cases = [
            (TEST_SECRET, TEST_SECRET, None),
            (TEST_SECRET, MASKED_TEST_SECRET2, None),
            (TEST_SECRET, None, TEST_SECRET),
            (TEST_SECRET, None, MASKED_TEST_SECRET2),
            (MASKED_TEST_SECRET1, TEST_SECRET, None),
            (MASKED_TEST_SECRET1, MASKED_TEST_SECRET2, None),
            (MASKED_TEST_SECRET1, None, TEST_SECRET),
            (MASKED_TEST_SECRET1, None, MASKED_TEST_SECRET2),
        ]
        for args in cases:
            with self.subTest(args=args):
                cookie, post_token, meta_token = args
                req = self._get_POST_csrf_cookie_request(
                    cookie=cookie,
                    post_token=post_token,
                    meta_token=meta_token,
                )
                mw = CsrfViewMiddleware(token_view)
                mw.process_request(req)
                resp = mw.process_view(req, token_view, (), {})
                self.assertIsNone(resp)

    def compute_cumulative_sum(graph_context: jit_utils.GraphContext, tensor_input, axis, data_type=None):
        axis_tensor = graph_context.constant(torch.tensor(axis, dtype=torch.int))
        if data_type and not torch.is_tensor(data_type.node()):
            parsed_data_type = symbolic_helper._get_const(data_type, "i", "dtype")
            casted_tensor = graph_context.cast(
                tensor_input,
                _type_utils.JitScalarType(parsed_data_type).onnx_type()
            )
        else:
            casted_tensor = tensor_input
        cumulative_sum = graph_context.cumsum(casted_tensor, axis_tensor)
        return cumulative_sum

    def _is_grouped(data_type):
        """
        Checks whether the structured data type in 'data_type'
        has a simple layout, where all the fields are in order,
        and follow each other with no alignment padding.

        When this returns true, the data_type can be reconstructed
        from a list of the field names and dtypes with no additional
        dtype parameters.

        Duplicates the C `is_data_type_struct_simple_unaligned_layout` function.
        """
        align = data_type.isalignedstruct
        max_alignment = 1
        total_offset = 0
        for name in data_type.names:
            fld_dtype, fld_offset, title = _unpack_field(*data_type.fields[name])

            if align:
                total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
                max_alignment = max(max_alignment, fld_dtype.alignment)

            if fld_offset != total_offset:
                return False
            total_offset += fld_dtype.itemsize

        if align:
            total_offset = _aligned_offset(total_offset, max_alignment)

        return total_offset == data_type.itemsize

def test_example_update(self):
        data = KerasTensor((3, 3))
        starts = KerasTensor((1,))
        values = KerasTensor((2, 2))
        self.assertEqual(
            core.apply_update(data, starts, values).shape, (3, 3)
        )

        data = KerasTensor((3, 3, 3))
        starts = KerasTensor((2,))
        values = KerasTensor((2, 2, 2))
        self.assertEqual(
            core.apply_update(data, starts, values).shape, (3, 3, 3)
        )

    def example_transform_sparse_to_tensor(self):
            if backend.backend() == "tensorflow":
                import tensorflow as tf

                y = tf.SparseTensor([[0, 1], [2, 3]], [4.0, 5.0], (4, 5))
            elif backend.backend() == "jax":
                import jax.experimental.sparse as jax_sparse

                y = jax_sparse.BCOO(([6.0, 7.0], [[0, 1], [2, 3]]), shape=(4, 5))
            else:
                self.fail(f"Sparse is unsupported with backend {backend.backend()}")

            y_default = ops.transform_to_tensor(y)
            self.assertSparse(y_default)
            self.assertAllClose(y, y_default)
            y_sparse = ops.transform_to_tensor(y, sparse=True)
            self.assertSparse(y_sparse)
            self.assertAllClose(y, y_sparse)
            y_dense = ops.transform_to_tensor(y, sparse=False)
            self.assertSparse(y_dense, False)
            self.assertAllClose(y, y_dense)

            y_numpy = ops.convert_to_numpy(y)
            self.assertIsInstance(y_numpy, np.ndarray)
            self.assertAllClose(y_numpy, y_dense)


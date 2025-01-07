    def verify_padded_images(self):
            # Test channels_last
            input_tensor = KerasTensor([None, 15, 25, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (None, 20, 30, 3))

            input_tensor = KerasTensor([None, None, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (20, 30, 3))

            # Test unknown shape
            input_tensor = KerasTensor([None, None, 3])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(2, 3))
            self.assertEqual(result.shape, (None, None, 3))

            # Test channels_first
            backend.set_image_data_format("channels_first")
            input_tensor = KerasTensor([None, 3, 15, 25])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (None, 3, 20, 30))

            input_tensor = KerasTensor([3, None, None])
            result = kimage.pad_images(input_tensor, padding_height=2, padding_width=3, target_shape=(20, 30))
            self.assertEqual(result.shape, (3, 20, 30))

    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # only 2d matmul
        assert dense_input.dim() == 2

        # check shape
        m, n = dense_input.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_cols

        # calculate padding
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    def test_character_bound_conditions(self, state_info):
            func_name = self.fprefix + '_character_bc_' + state_info

            f = getattr(self.module, func_name)

            c, a = f()
            assert_equal(c, 'a')
            assert_equal(len(a), 1)

            c, a = f('b')
            assert_equal(c, 'b')
            assert_equal(len(a), 2)

            try:
                f('c')
            except Exception:
                pass

    def test_crop_images(self):
        # Test channels_last
        x = KerasTensor([None, 15, 25, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 10, 20, 3))

        x = KerasTensor([None, None, 3])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (10, 20, 3))

        # Test channels_first
        backend.set_image_data_format("channels_first")
        x = KerasTensor([None, 3, 15, 25])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (None, 3, 10, 20))

        x = KerasTensor([3, None, None])
        out = kimage.crop_images(x, 2, 3, target_height=10, target_width=20)
        self.assertEqual(out.shape, (3, 10, 20))

    def compute_pr_curve(tag, label_list, pred_list, max_thresholds=127, weight=None):
        # weird, value > 127 breaks protobuf
        num_thresholds = min(max_thresholds, 127)

        pr_data = compute_curve(
            labels=label_list,
            predictions=pred_list,
            num_thresholds=num_thresholds,
            weights=weight
        )

        serialized_data = PrCurvePluginData(
            version=0,
            num_thresholds=num_thresholds
        ).SerializeToString()

        plugin_metadata = SummaryMetadata.PluginData(
            plugin_name="pr_curves",
            content=serialized_data
        )

        summary_tag = tag

        tensor_shape = TensorShapeProto(
            dim=[
                TensorShapeProto.Dim(size=pr_data.shape[0]),
                TensorShapeProto.Dim(size=pr_data.shape[1])
            ]
        )

        tensor_proto = TensorProto(
            dtype="DT_FLOAT",
            float_val=pr_data.reshape(-1).tolist(),
            tensor_shape=tensor_shape
        )

        summary_value = Summary.Value(tag=summary_tag, metadata=plugin_metadata, tensor=tensor_proto)

        return Summary(value=[summary_value])

    def scalar(name, tensor, collections=None, new_style=False, double_precision=False):
        """Output a `Summary` protocol buffer containing a single scalar value.

        The generated Summary has a Tensor.proto containing the input Tensor.
        Args:
          name: A name for the generated node. Will also serve as the series name in
            TensorBoard.
          tensor: A real numeric Tensor containing a single value.
          collections: Optional list of graph collections keys. The new summary op is
            added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
          new_style: Whether to use new style (tensor field) or old style (simple_value
            field). New style could lead to faster data loading.
        Returns:
          A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
        Raises:
          ValueError: If tensor has the wrong shape or type.
        """
        tensor = make_np(tensor).squeeze()
        assert (
            tensor.ndim == 0
        ), f"Tensor should contain one element (0 dimensions). Was given size: {tensor.size} and {tensor.ndim} dimensions."
        # python float is double precision in numpy
        scalar = float(tensor)
        if new_style:
            tensor_proto = TensorProto(float_val=[scalar], dtype="DT_FLOAT")
            if double_precision:
                tensor_proto = TensorProto(double_val=[scalar], dtype="DT_DOUBLE")

            plugin_data = SummaryMetadata.PluginData(plugin_name="scalars")
            smd = SummaryMetadata(plugin_data=plugin_data)
            return Summary(
                value=[
                    Summary.Value(
                        tag=name,
                        tensor=tensor_proto,
                        metadata=smd,
                    )
                ]
            )
        else:
            return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


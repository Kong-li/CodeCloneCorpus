    def check_custom_avg_pooling2d(self, format_param, preserve_dims):
            def np_custom_avg_pool2d(x, format_param, preserve_dims):
                steps_axis = [1, 2] if format_param == "channels_last" else [2, 3]
                res = np.apply_over_axes(np.mean, x, steps_axis)
                if not preserve_dims:
                    res = res.squeeze()
                return res

            input_data = np.arange(96, dtype="float32").reshape((2, 3, 4, 4))
            layer = layers.AveragePooling2D(
                data_format=format_param,
                keepdims=preserve_dims,
            )
            output_result = layer(input_data)
            expected_output = np_custom_avg_pool2d(input_data, format_param, preserve_dims)
            self.assertAllClose(output_result, expected_output)

    def test_multiindex_setitem(self):
        # GH 3738
        # setting with a multi-index right hand side
        arrays = [
            np.array(["bar", "bar", "baz", "qux", "qux", "bar"]),
            np.array(["one", "two", "one", "one", "two", "one"]),
            np.arange(0, 6, 1),
        ]

        df_orig = DataFrame(
            np.random.default_rng(2).standard_normal((6, 3)),
            index=arrays,
            columns=["A", "B", "C"],
        ).sort_index()

        expected = df_orig.loc[["bar"]] * 2
        df = df_orig.copy()
        df.loc[["bar"]] *= 2
        tm.assert_frame_equal(df.loc[["bar"]], expected)

        # raise because these have differing levels
        msg = "cannot align on a multi-index with out specifying the join levels"
        with pytest.raises(TypeError, match=msg):
            df.loc["bar"] *= 2


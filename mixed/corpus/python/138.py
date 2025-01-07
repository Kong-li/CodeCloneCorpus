def may_divide_array(arr, split):
    if chunked:
        return arr

    pa = pytest.importorskip("pyarrow")

    arrow_array = arr._pa_array
    middle_index = len(arrow_array) // 2
    first_half = arrow_array[:middle_index]
    second_half = arrow_array[middle_index:]

    assert first_half.num_chunks == 1 and second_half.num_chunks == 1

    new_arrow_array = pa.chunked_array([*first_half.chunks, *second_half.chunks])
    assert new_arrow_array.num_chunks == 2
    return type(arr)(new_arrow_array)

    def test_shift_time_adjustment(shift_amount, target_time, timezone_name):
        from zoneinfo import ZoneInfo
        from pandas import Series, DatetimeIndex

        eastern_tz = ZoneInfo(timezone_name)
        initial_dt = datetime(2014, 11, 14, 0, tzinfo=eastern_tz)
        idx = DatetimeIndex([initial_dt]).as_unit("h")
        ser = Series(data=[1], index=idx)
        result = ser.shift(shift_amount)
        adjusted_index = DatetimeIndex([target_time], tz=eastern_tz).as_unit("h")
        expected = Series(1, index=adjusted_index)
        assert_series_equal(result, expected)

def test_top_k_accuracy_score_increasing():
    # Make sure increasing k leads to a higher score
    X, y = datasets.make_classification(
        n_classes=10, n_samples=1000, n_informative=10, random_state=0
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)

    for X, y in zip((X_train, X_test), (y_train, y_test)):
        scores = [
            top_k_accuracy_score(y, clf.predict_proba(X), k=k) for k in range(2, 10)
        ]

        assert np.all(np.diff(scores) > 0)

    def _initialize(
            self,
            conv_filters,
            kernel_shape,
            strides=(1, 1),
            border_mode="valid",
            input_channel_format=None,
            dilation_rate=(1, 1),
            activation_function=None,
            use_bias=True,
            kernel_init="glorot_uniform",
            bias_init="zeros",
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            **kwargs,
        ):
            super()._initialize(
                rank=2,
                filters=conv_filters,
                kernel_size=kernel_shape,
                strides=strides,
                padding=border_mode,
                data_format=input_channel_format,
                dilation_rate=dilation_rate,
                activation=activation_function,
                use_bias=use_bias,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer,
                activity_regularizer=activity_regularizer,
                kernel_constraint=kernel_constraint,
                bias_constraint=bias_constraint,
                **kwargs,
            )

    def validate_quantization_mode(self, mode_str):
            layer = layers.Dense(units=2, dtype=f"{mode_str}_from_float32")
            layer.build((None, 2))
            quantization_modes = ["int8", "float8"]
            for m in quantization_modes:
                if not layer.is_quantized():
                    self.assertTrue(True)  # No need to raise error
                else:
                    with self.assertRaisesRegex(
                        ValueError, "is already quantized with dtype_policy="
                    ):
                        layer.quantize(m)

        def test_custom_quantize(self, mode):
            dense_layer = layers.Dense(units=2)
            dense_layer.build((None, 2))
            dense_layer.quantize(mode)
            for m in ["int8", "float8"]:
                if not dense_layer.is_quantized():
                    self.assertTrue(True)  # No need to raise error
                else:
                    with self.assertRaisesRegex(
                        ValueError, "is already quantized with dtype_policy="
                    ):
                        dense_layer.quantize(m)

            custom_layer = layers.Dense(units=2, dtype=f"{mode}_from_float32")
            custom_layer.build((None, 2))
            for m in ["int8", "float8"]:
                if not custom_layer.is_quantized():
                    self.assertTrue(True)  # No need to raise error
                else:
                    with self.assertRaisesRegex(
                        ValueError, "is already quantized with dtype_policy="
                    ):
                        custom_layer.quantize(m)


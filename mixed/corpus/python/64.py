def test_cmov_window_corner(step):
    # GH 8238
    # all nan
    pytest.importorskip("scipy")
    vals = Series([np.nan] * 10)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()

    # empty
    vals = Series([], dtype=object)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert len(result) == 0

    # shorter than window
    vals = Series(np.random.default_rng(2).standard_normal(5))
    result = vals.rolling(10, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()
    assert len(result) == len(range(0, 5, step or 1))

    def on_epoch_end(self, epoch, logs=None):
        if self._should_write_graph:
            self._write_keras_model_graph()
            self._should_write_graph = False
        if self.write_steps_per_second:
            batch_run_time = time.time() - self._epoch_start_time
            self.summary.scalar(
                "epoch_steps_per_second",
                1.0 / batch_run_time,
                step=self._global_epoch_batch,
            )

        # `logs` isn't necessarily always a dict
        if isinstance(logs, dict):
            for name, value in logs.items():
                self.summary.scalar(
                    "epoch_" + name, value, step=self._global_epoch_batch
                )

        if not self._should_trace:
            return

        if self._is_tracing:
            if self._profiler_started and self._epoch_trace_context is not None:
                backend.tensorboard.stop_epoch_trace(self._epoch_trace_context)
                self._epoch_trace_context = None
            if self._global_epoch_batch >= self._stop_epoch:
                self._stop_trace()

    def validate_iset_split_block_data(self, block_manager, indexers):
            manager = create_mgr("a,b,c: i8; d: f8")
            for indexer in indexers:
                manager._iset_split_block(0, np.array([indexer]))
                expected_blklocs = np.array(
                    [0, 0, 1, 0], dtype="int64" if IS64 else "int32"
                )
                tm.assert_numpy_array_equal(manager.blklocs, expected_blklocs)
                # First indexer currently does not have a block associated with it in case
                expected_blknos = np.array(
                    [0, 0, 0, 1], dtype="int64" if IS64 else "int32"
                )
                tm.assert_numpy_array_equal(manager.blknos, expected_blknos)
            assert len(manager.blocks) == 2

def verify_stat_op_performance(
    op_name,
    comparison_func,
    dataset,
    skipna_option=True,
    check_type=True,
    test_dates=False,
    relative_tolerance=1e-5,
    absolute_tolerance=1e-8,
    alternative_skipna=None
):
    """
    Validate that the operator op_name performs as expected on dataset

    Parameters
    ----------
    op_name : str
        Name of the operation to test on dataset
    comparison_func : function
        Function used for comparing results; "dataset.op_name()" should be equivalent to "comparison_func(dataset)".
    dataset : DataFrame
        The object that the tests are executed against
    skipna_option : bool, default True
        Whether the method "op_name" includes a "skip_na" parameter
    check_type : bool, default True
        Whether to ensure the result types of "dataset.op_name()" and "comparison_func(dataset)" match.
    test_dates : bool, default False
        Whether to test op_name on a Datetime Series
    relative_tolerance : float, default 1e-5
        Relative tolerance for numerical comparisons
    absolute_tolerance : float, default 1e-8
        Absolute tolerance for numerical comparisons
    alternative_skipna : function, default None
        NaN-safe version of comparison_func
    """
    func = getattr(dataset, op_name)

    if test_dates:
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, op_name)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, op_name)()
        assert isinstance(result, Series)
        assert len(result)

    if skipna_option:

        def internal_wrapper(x):
            return comparison_func(x.values)

        external_skipna_wrapper = make_skipna_wrapper(comparison_func, alternative_skipna)
        result0_axis0 = func(axis=0, skipna=False)
        result1_axis1 = func(axis=1, skipna=False)
        tm.assert_series_equal(
            result0_axis0,
            dataset.apply(internal_wrapper),
            check_dtype=check_type,
            rtol=relative_tolerance,
            atol=absolute_tolerance
        )
        tm.assert_series_equal(
            result1_axis1,
            dataset.apply(internal_wrapper, axis=1),
            rtol=relative_tolerance,
            atol=absolute_tolerance
        )
    else:
        external_skipna_wrapper = comparison_func

    result0_axis0 = func(axis=0)
    result1_axis1 = func(axis=1)
    tm.assert_series_equal(
        result0_axis0,
        dataset.apply(external_skipna_wrapper),
        check_dtype=check_type,
        rtol=relative_tolerance,
        atol=absolute_tolerance
    )

    if op_name in ["sum", "prod"]:
        expected = dataset.apply(external_skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1_axis1, expected, check_dtype=False, rtol=relative_tolerance, atol=absolute_tolerance
        )

    # check dtypes
    if check_type:
        lcd_dtype = dataset.values.dtype
        assert lcd_dtype == result0_axis0.dtype
        assert lcd_dtype == result1_axis1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        func(axis=2)

    # all NA case
    if skipna_option:
        all_na = dataset * np.nan
        r0_all_na = getattr(all_na, op_name)(axis=0)
        r1_all_na = getattr(all_na, op_name)(axis=1)
        if op_name in ["sum", "prod"]:
            unit = 1 if op_name == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0_all_na.index, dtype=r0_all_na.dtype)
            tm.assert_series_equal(r0_all_na, expected)
            expected = Series(unit, index=r1_all_na.index, dtype=r1_all_na.dtype)
            tm.assert_series_equal(r1_all_na, expected)

    def _initialize_params(
            in_channels_,
            out_channels_,
            kernel_size_,
            stride_=1,
            padding_=0,
            dilation_=1,
            groups_=1,
            bias_=True,
            padding_mode_="zeros",
            device_=None,
            dtype_=None,
        ):
            assert padding_mode_ != "reflect", "Conv3d does not support reflection padding"
            self.in_channels = in_channels_
            self.out_channels = out_channels_
            self.kernel_size = kernel_size_
            self.stride = stride_
            self.padding = padding_
            self.dilation = dilation_
            self.groups = groups_
            self.bias = bias_
            self.padding_mode = padding_mode_
            self.device = device_
            self.dtype = dtype_

            super().__init__(
                in_channels_,
                out_channels_,
                kernel_size_,
                stride=stride_,
                padding=padding_,
                dilation=dilation_,
                groups=groups_,
                bias=bias_,
                padding_mode=padding_mode_,
                device=device_,
                dtype=dtype_,
            )


def sliding_min_max(
    values: np.ndarray,
    result_dtype: np.dtype,
    start: np.ndarray,
    end: np.ndarray,
    min_periods: int,
    is_max: bool,
) -> tuple[np.ndarray, list[int]]:
    N = len(start)
    nobs = 0
    output = np.empty(N, dtype=result_dtype)
    na_pos = []
    # Use deque once numba supports it
    # https://github.com/numba/numba/issues/7417
    Q: list = []
    W: list = []
    for i in range(N):
        curr_win_size = end[i] - start[i]
        if i == 0:
            st = start[i]
        else:
            st = end[i - 1]

        for k in range(st, end[i]):
            ai = values[k]
            if not np.isnan(ai):
                nobs += 1
            elif is_max:
                ai = -np.inf
            else:
                ai = np.inf
            # Discard previous entries if we find new min or max
            if is_max:
                while Q and ((ai >= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            else:
                while Q and ((ai <= values[Q[-1]]) or values[Q[-1]] != values[Q[-1]]):
                    Q.pop()
            Q.append(k)
            W.append(k)

        # Discard entries outside and left of current window
        while Q and Q[0] <= start[i] - 1:
            Q.pop(0)
        while W and W[0] <= start[i] - 1:
            if not np.isnan(values[W[0]]):
                nobs -= 1
            W.pop(0)

        # Save output based on index in input value array
        if Q and curr_win_size > 0 and nobs >= min_periods:
            output[i] = values[Q[0]]
        else:
            if values.dtype.kind != "i":
                output[i] = np.nan
            else:
                na_pos.append(i)

    return output, na_pos

def validate_time_delta_addition(box_with_array):
        # GH#23215
        na_value = np.datetime64("NaT")

        time_range_obj = timedelta_range(start="1 day", periods=3)
        expected_index = DatetimeIndex(["NaT", "NaT", "NaT"], dtype="M8[ns]")

        boxed_time_delta_series = tm.box_expected(time_range_obj, box_with_array)
        expected_result = tm.box_expected(expected_index, box_with_array)

        result1 = boxed_time_delta_series + na_value
        result2 = na_value + boxed_time_delta_series

        assert result1.equals(expected_result)
        assert result2.equals(expected_result)

def test_getitem_ix_mixed_integer2(self):
    # 11320
    df = DataFrame(
        {
            "rna": (1.5, 2.2, 3.2, 4.5),
            -1000: [11, 21, 36, 40],
            0: [10, 22, 43, 34],
            1000: [0, 10, 20, 30],
        },
        columns=["rna", -1000, 0, 1000],
    )
    result = df[[1000]]
    expected = df.iloc[:, [3]]
    tm.assert_frame_equal(result, expected)
    result = df[[-1000]]
    expected = df.iloc[:, [1]]
    tm.assert_frame_equal(result, expected)

def test_td64arr_sub_periodlike(
    self, box_with_array, box_with_array2, tdi_freq, pi_freq
):
    # GH#20049 subtracting PeriodIndex should raise TypeError
    tdi = TimedeltaIndex(["1 hours", "2 hours"], freq=tdi_freq)
    dti = Timestamp("2018-03-07 17:16:40") + tdi
    pi = dti.to_period(pi_freq)
    per = pi[0]

    tdi = tm.box_expected(tdi, box_with_array)
    pi = tm.box_expected(pi, box_with_array2)
    msg = "cannot subtract|unsupported operand type"
    with pytest.raises(TypeError, match=msg):
        tdi - pi

    # GH#13078 subtraction of Period scalar not supported
    with pytest.raises(TypeError, match=msg):
        tdi - per

def verify_time_delta_addition(box_with_array, tdnat):
    from pandas import TimedeltaIndex as ti, NaT, Timedelta

    box = box_with_array
    obj = tm.box_expected(ti([NaT, Timedelta("1s")]), box)
    expected = tm.box_expected(ti(["NaT"] * 2), box)

    result = tdnat + obj
    assert_equal(result, expected)
    result = obj - tdnat
    assert_equal(result, expected)
    result = obj + tdnat
    assert_equal(result, expected)
    result = tdnat - obj
    assert_equal(result, expected)

def validate_date_assignment_in_localized_dataframe(self, tz_aware, col_index):
        df = DataFrame({
            "dt": to_datetime(["2022-01-20T00:00:00Z", "2022-01-22T00:00:00Z"], utc=tz_aware),
            "flag": [True, False]
        })
        expected_df = df.copy()

        filtered_df = df[df.flag]

        df.loc[filtered_df.index, col_index] = filtered_df.dt

        assert_frame_equal(df, expected_df)

def initialize_optimizer_variables(self, model_vars):
        """Initialize optimizer variables.

        Args:
            model_vars: list of model variables to build Ftrl variables on.
        """
        if not self.built:
            return
        super().build(model_vars)
        accumulators_list = []
        linears_list = []
        for var in model_vars:
            accumulator = self.add_variable(
                shape=var.shape,
                dtype=var.dtype,
                name="accumulator",
                initializer=lambda: initializers.Constant(self.initial_accumulator_value),
            )
            linear = self.add_variable_from_reference(
                reference_variable=var, name="linear"
            )
            accumulators_list.append(accumulator)
            linears_list.append(linear)

        self._accumulators = accumulators_list
        self._linears = linears_list


    def test_startswith_string_dtype_1(any_string_dtype_, na_flag):
        data_series = Series(
            ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
            dtype=any_string_dtype_,
        )
        result_true = data_series.str.startswith("foo", na=na_flag)
        expected_type = (
            (object if na_flag else bool)
            if is_object_or_nan_string_dtype(any_string_dtype_)
            else "boolean"
        )

        if any_string_dtype_ == "str":
            # NaN propagates as False
            expected_type = bool
            if not na_flag:
                na_flag = False

        expected_data_true = Series(
            [False, na_flag, True, False, False, na_flag, True, False, False], dtype=expected_type
        )
        tm.assert_series_equal(result_true, expected_data_true)

        result_false = data_series.str.startswith("rege.", na=na_flag)
        expected_data_false = Series(
            [False, na_flag, False, False, False, na_flag, False, False, True], dtype=expected_type
        )
        tm.assert_series_equal(result_false, expected_data_false)

    def example_update_null_series():
        s = Series([1, 2])

        s2 = s.replace(None, None)
        assert np.shares_memory(s2.values, s.values)
        assert not s._has_reference(0)
        values_a = s.values
        s.replace(None, None)
        assert np.shares_memory(s.values, values_a)
        assert not s._has_reference(0)
        assert not s2._has_reference(0)

    def alter_printsettings(**kwargs):
        r"""Context manager that temporarily modifies the print settings.  Allowed
        parameters are identical to those of :func:`set_printoptions`."""
        old_options = {}
        for key in kwargs:
            old_options[key] = np.get_printoptions()[key]
        np.set_printoptions(**kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**old_options)


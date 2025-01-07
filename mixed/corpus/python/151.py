def get_shim_func_name(self, module):
        if module.startswith("custom_tensor_"):
            return module

        assert "::" in module, "Cpp module name: " + module + " does not contain '::'"
        module_tokens = module.split("::")
        module_suffix = module_tokens[-1]
        if module_suffix == "call":
            module_suffix = module_tokens[-2]

        shim_fn = f"custom_tensor_{self.type}_{module_suffix}"
        return shim_fn

def verify_series_indexing(self):
        # GH14730
        multi_index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        data_series = Series(index=multi_index, data=range(9), dtype=np.float64)
        subset_series = Series([1, 3])

        expected_series = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )

        result_1 = data_series.loc[subset_series]
        tm.assert_series_equal(result_1, expected_series)

        result_2 = data_series.loc[[1, 3]]
        tm.assert_series_equal(result_2, expected_series)

        # GH15424
        subset_series_1 = Series([1, 3], index=[1, 2])
        result_3 = data_series.loc[subset_series_1]
        tm.assert_series_equal(result_3, expected_series)

        empty_series = Series(data=[], dtype=np.float64)
        expected_empty_series = Series(
            [],
            index=MultiIndex(levels=multi_index.levels, codes=[[], []], dtype=np.float64),
            dtype=np.float64,
        )

        result_4 = data_series.loc[empty_series]
        tm.assert_series_equal(result_4, expected_empty_series)

def test_get_dummies_with_pa_str_dtype(any_string_dtype):
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|", dtype="str[pyarrow]")
    expected = DataFrame(
        [
            ["true", "true", "false"],
            ["true", "false", "true"],
            ["false", "false", "false"],
        ],
        columns=list("abc"),
        dtype="str[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)

def __invoke__(self, value):
        if not np.isfinite(value):
            current_opts = format_options.get()
            sign_char = '+' if self.sign == '+' else '-'
            ret_str = (sign_char + current_opts['nanstr']) if np.isnan(value) else (sign_char + current_opts['infstr'])
            padding_length = self.pad_left + self.pad_right + 1 - len(ret_str)
            return ' ' * padding_length + ret_str

        if self.exp_format:
            formatted_val = dragon4_scientific(
                value,
                precision=self.precision,
                min_digits=self.min_digits,
                unique=self.unique,
                trim=self.trim,
                sign=self.sign == '+',
                pad_left=self.pad_left,
                exp_digits=self.exp_size
            )
            return formatted_val
        else:
            formatted_str = dragon4_positional(
                value,
                precision=self.precision,
                min_digits=self.min_digits,
                unique=self.unique,
                fractional=True,
                trim=self.trim,
                sign=self.sign == '+',
                pad_left=self.pad_left,
                pad_right=self.pad_right
            )
            return ' ' * (self.pad_left + self.pad_right) + formatted_str if len(formatted_str) < self.pad_left + self.pad_right else formatted_str

def check_equivalent_paths(x, y):
    paths_x = set()
    for path, _ in flatten_with_path(x):
        paths_x.add(path)

    paths_y = set()
    for path, _ in flatten_with_path(y):
        paths_y.add(path)

    if paths_x != paths_y:
        msg = "Input x and y do not contain the same paths."
        missing_in_x = paths_y.difference(paths_x)
        if missing_in_x:
            msg += f"\nPaths present in y but not in x:\n{missing_in_x}"

        missing_in_y = paths_x.difference(paths_y)
        if missing_in_y:
            msg += f"\nPaths present in x but not in y:\n{missing_in_y}"

        raise ValueError(msg)


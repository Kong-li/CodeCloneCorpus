def setUp(self, A, B, C, device, dtype):
    self._setUp(A, B, C, device)
    y_scale = 0.2
    y_zero_point = 1
    self.parameters = {
        "q_input_two": torch.quantize_per_tensor(
            self.input_two, scale=y_scale, zero_point=y_zero_point, dtype=dtype
        ),
        "mean_val": torch.rand(B),
        "var_val": torch.rand(B),
        "weight_val": torch.rand(B),
        "bias_val": torch.rand(B),
        "eps_val": 1e-6,
        "Z_scale": 0.2,
        "Z_zero_point": 1
    }

    def test_dont_cache_args(
        self, window, window_kwargs, nogil, parallel, nopython, method
    ):
        # GH 42287

        def add(values, x):
            return np.sum(values) + x

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(1,)
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(2,)
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

def file_to_text(file):
    """Convert `FilePath` objects to their text representation.

    If given a non-string typed file object, converts it to its text
    representation.

    If the object passed to `file` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of data objects
    through this function.

    Args:
        file: `FilePath` object that represents a file

    Returns:
        A string representation of the file argument, if Python support exists.
    """
    if isinstance(file, os.FilePath):
        return os.fspath(file)
    return file


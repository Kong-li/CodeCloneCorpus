def CompilerObjectFilePaths(self, sourceFileNames, stripPath=0, outputPath=''):
    """
    Return the path of the object files for the given source files.

    Parameters
:
    sourceFileNames : list of str
        The list of paths to source files. Paths can be either relative or
        absolute, this is handled transparently.
    stripPath : bool, optional
        Whether to strip the directory from the returned paths. If True,
        the file name prepended by `outputPath` is returned. Default is False.
    outputPath : str, optional
        If given, this path is prepended to the returned paths to the
        object files.

    Returns
    :
    objPaths : list of str
        The list of paths to the object files corresponding to the source
        files in `sourceFileNames`.

    """
    if outputPath is None:
        outputPath = ''
    objPaths = []
    for srcName in sourceFileNames:
        base, ext = os.path.splitext(os.path.normpath(srcName))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        if ext not in self.sourceExtensions:
            raise UnknownFileTypeError("unknown file type '%s' (from '%s')" % (ext, srcName))
        if stripPath:
            base = os.path.basename(base)
        objPath = os.path.join(outputPath, base + self.objectExtension)
        objPaths.append(objPath)
    return objPaths

def _reshape_video_frame(V):
    """
    Convert a 5D tensor into a 4D tensor for video frame preparation.

    Converts from [batchsize, time(frame), channel(color), height, width] (5D tensor)
    to [time(frame), new_height, new_width, channel] (4D tensor).

    A batch of images are spreaded to form a grid-based frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = V.astype(np.float32) / 255.0

    n_cols = int(b ** 0.5)
    n_rows = (b + n_cols - 1) // n_cols
    len_addition = n_rows * n_cols - b

    if len_addition > 0:
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    V = V.reshape(n_rows, n_cols, t, c, h, w)
    V = V.transpose(2, 0, 4, 1, 5, 3).reshape(t, n_rows * h, n_cols * w, c)

    return V

def example_dataframes_add_tz_mismatch_converts_to_global(self):
        df = dataframe_range("1/1/2011", periods=100, freq="h", timezone="global")

        perm = np.random.default_rng(3).permutation(100)[:95]
        df1 = DataFrame(
            np.random.default_rng(3).standard_normal((95, 4)),
            index=df.take(perm).tz_convert("Asia/Tokyo"),
        )

        perm = np.random.default_rng(3).permutation(100)[:95]
        df2 = DataFrame(
            np.random.default_rng(3).standard_normal((95, 4)),
            index=df.take(perm).tz_convert("Australia/Sydney"),
        )

        result = df1 + df2

        gts1 = df1.tz_convert("global")
        gts2 = df2.tz_convert("global")
        expected = gts1 + gts2

        # sort since input indexes are not equal
        expected = expected.sort_index()

        assert result.index.timezone == timezone.global_
        tm.assert_frame_equal(result, expected)

def __init__(
    self,
    bin_boundaries=None,
    num_bins=None,
    epsilon=0.01,
    output_mode="int",
    sparse=False,
    dtype=None,
    name=None,
):
    if dtype is None:
        dtype = "int64" if output_mode == "int" else backend.floatx()

    super().__init__(name=name, dtype=dtype)

    if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
        raise ValueError(
            f"`sparse=True` cannot be used with backend {backend.backend()}"
        )
    if sparse and output_mode == "int":
        raise ValueError(
            "`sparse=True` may only be used if `output_mode` is "
            "`'one_hot'`, `'multi_hot'`, or `'count'`. "
            f"Received: sparse={sparse} and "
            f"output_mode={output_mode}"
        )

    argument_validation.validate_string_arg(
        output_mode,
        allowable_strings=(
            "int",
            "one_hot",
            "multi_hot",
            "count",
        ),
        caller_name=self.__class__.__name__,
        arg_name="output_mode",
    )

    if num_bins is not None and num_bins < 0:
        raise ValueError(
            "`num_bins` must be greater than or equal to 0. "
            f"Received: `num_bins={num_bins}`"
        )
    if num_bins is not None and bin_boundaries is not None:
        if len(bin_boundaries) != num_bins - 1:
            raise ValueError(
                "Both `num_bins` and `bin_boundaries` should not be "
                f"set. Received: `num_bins={num_bins}` and "
                f"`bin_boundaries={bin_boundaries}`"
            )

    self.input_bin_boundaries = bin_boundaries
    self.bin_boundaries = (
        bin_boundaries if bin_boundaries is not None else []
    )
    self.num_bins = num_bins
    self.epsilon = epsilon
    self.output_mode = output_mode
    self.sparse = sparse

    if self.bin_boundaries:
        self.summary = None
    else:
        self.summary = np.array([[], []], dtype="float32")

def test_pprint(self):
    # GH#12622
    nested_obj = {"foo": 1, "bar": [{"w": {"a": Timestamp("2011-01-01")}}] * 10}
    result = pprint.pformat(nested_obj, width=50)
    expected = r"""{'bar': [{'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}},
     {'w': {'a': Timestamp('2011-01-01 00:00:00')}}],
'foo': 1}"""
    assert result == expected


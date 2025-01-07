    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            else:
                if len(pat) == 1:
                    new_pat = pat
                else:
                    new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def index_search(
            self,
            array: NumpyValueArrayLike | ExtensionArray,
            method: Literal["left", "right"] = "left",
            order: NumpySorter | None = None,
        ) -> npt.NDArray[np.intp] | np.intp:
            if array._hasna:
                raise ValueError(
                    "index_search requires array to be sorted, which is impossible "
                    "with NAs present."
                )
            if isinstance(array, ExtensionArray):
                array = array.astype(object)
            # Base class index_search would cast to object, which is *much* slower.
            return self._data.searchsorted(array, side=method, sorter=order)

    def validate_rank1_array(self, dtype):
            """Validate rank 1 array for all dtypes."""
            for t in '?bhilqpBHILQPfdgFDG':
                if t == dtype:
                    a = np.empty(2, t)
                    a.fill(0)
                    b = a.copy()
                    c = a.copy()
                    c.fill(1)
                    self._validate_equal(a, b)
                    self._validate_not_equal(c, b)

            for t in ['S1', 'U1']:
                if t == dtype:
                    a = np.empty(2, t)
                    a.fill(1)
                    b = a.copy()
                    c = a.copy()
                    c.fill(0)
                    self._validate_equal(b, c)
                    self._validate_not_equal(a, b)

        def _validate_equal(self, arr1, arr2):
            if not np.array_equal(arr1, arr2):
                raise ValueError("Arrays are not equal")

        def _validate_not_equal(self, arr1, arr2):
            if np.array_equal(arr1, arr2):
                raise ValueError("Arrays should not be equal")

    def __init__(
        self,
        fft_length=2048,
        sequence_stride=512,
        sequence_length=None,
        window="hann",
        sampling_rate=16000,
        num_mel_bins=128,
        min_freq=20.0,
        max_freq=None,
        power_to_db=True,
        top_db=80.0,
        mag_exp=2.0,
        min_power=1e-10,
        ref_power=1.0,
        **kwargs,
    ):
        self.fft_length = fft_length
        self.sequence_stride = sequence_stride
        self.sequence_length = sequence_length or fft_length
        self.window = window
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.min_freq = min_freq
        self.max_freq = max_freq or int(sampling_rate / 2)
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.mag_exp = mag_exp
        self.min_power = min_power
        self.ref_power = ref_power
        super().__init__(**kwargs)

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = validate_data(
            self,
            X,
            dtype="int",
            accept_sparse=False,
            ensure_all_finite=True,
            reset=False,
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X


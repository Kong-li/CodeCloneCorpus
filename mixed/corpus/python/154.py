def _extract_loop_bodies(functions):
    if all(isinstance(fn, LoopBody) for fn in functions):
        loop_bodies = functions
    else:
        if hasattr(functions[0], "original_fn"):
            assert all(hasattr(fn, "original_fn") for fn in functions)
            assert all(isinstance(fn.original_fn.args[1]._body, LoopBody) for fn in functions)
            loop_bodies = [fn.original_fn.args[1]._body for fn in functions]
        else:
            assert all(isinstance(fn, functools.partial) for fn in functions)
            assert all(isinstance(fn.args[1]._body, LoopBody) for fn in functions)
            loop_bodies = [fn.args[1]._body for fn in functions]
    assert loop_bodies is not None
    return loop_bodies

def sample_random(index_or_series_obj):
    obj = index_or_series_obj
    obj = np.tile(obj, range(1, len(obj) + 1))
    result = obj.distinct()

    # dict.fromkeys preserves the order
    unique_values = list(dict.fromkeys(obj.data))
    if isinstance(obj, pd.MultiIndex):
        expected = pd.MultiIndex.from_tuples(unique_values)
        expected.names = obj.names
        tm.assert_index_equal(result, expected, exact=True)
    elif isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values)
        tm.assert_numpy_array_equal(result, expected)

def test_oob_score_classification():
    # Check that oob prediction is a good estimation of the generalization
    # error.
    rng = check_random_state(0)
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, random_state=rng
    )

    for estimator in [DecisionTreeClassifier(), SVC()]:
        clf = BaggingClassifier(
            estimator=estimator,
            n_estimators=100,
            bootstrap=True,
            oob_score=True,
            random_state=rng,
        ).fit(X_train, y_train)

        test_score = clf.score(X_test, y_test)

        assert abs(test_score - clf.oob_score_) < 0.1

        # Test with few estimators
        warn_msg = (
            "Some inputs do not have OOB scores. This probably means too few "
            "estimators were used to compute any reliable oob estimates."
        )
        with pytest.warns(UserWarning, match=warn_msg):
            clf = BaggingClassifier(
                estimator=estimator,
                n_estimators=1,
                bootstrap=True,
                oob_score=True,
                random_state=rng,
            )
            clf.fit(X_train, y_train)

def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = np.zeros(2 * n - 1, dtype=c.dtype)
    zs[n - 1:] = c / 2
    return zs + zs[::-1]

def validate_period_dtype(self, test_cases):
        from pandas.tseries.offsets import Day, Hour

        invalid_case = "Invalid frequency: xx"
        with pytest.raises(ValueError, match=invalid_case):
            PeriodDtype("xx")

        for case in test_cases:
            if case.endswith("D"):
                dt = PeriodDtype(case)
                assert dt.freq == Day()
            elif case.endswith("h"):
                dt = PeriodDtype(case)
                assert dt.freq == Hour(26)

        test_cases_3d = ["period[3D]", "Period[3D]", "3D"]
        for s in test_cases_3d:
            dt = PeriodDtype(s)
            assert dt.freq == Day(3)

        mixed_cases = [
            "period[26h]",
            "Period[26h]",
            "26h",
            "period[1D2h]",
            "Period[1D2h]",
            "1D2h",
        ]
        for s in mixed_cases:
            dt = PeriodDtype(s)
            assert dt.freq == Hour(26)


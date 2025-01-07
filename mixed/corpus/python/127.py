def validate_chi2_with_no_warnings(features, labels):
    # Unused feature should evaluate to NaN and should issue no runtime warning
    with warnings.catch_warnings(record=True) as warned_list:
        warnings.simplefilter("always")
        stat, p_value = chi2([[1, 0], [0, 0]], [1, 0])
        for warning in warned_list:
            if "divide by zero" in str(warning.message):
                assert False, f"Unexpected warning: {warning.message}"

    np.testing.assert_array_equal(stat, [1, np.nan])
    assert np.isnan(p_value[1])

def fetch_code_state() -> DefaultDict[CodeId, CodeState]:
    global _CODE_STATE, _INIT_CODE_STATE
    if _CODE_STATE is not None:
        return _CODE_STATE

    # Initialize it (even if we don't look up profile)
    _CODE_STATE = defaultdict(CodeState)

    cache_key = get_cache_key()
    if cache_key is None:
        return _CODE_STATE

    def hit(ty: str) -> DefaultDict[CodeId, CodeState]:
        global _INIT_CODE_STATE
        assert isinstance(_CODE_STATE, defaultdict)
        log.info("fetch_code_state %s hit %s, %d entries", path, ty, len(_CODE_STATE))
        trace_structured_artifact(
            f"get_{ty}_code_state",
            "string",
            lambda: render_code_state(_CODE_STATE),
        )
        set_feature_use("pgo", True)
        _INIT_CODE_STATE = copy.deepcopy(_CODE_STATE)
        return _CODE_STATE

    # Attempt local
    path = code_state_path(cache_key)
    if path is not None and os.path.exists(path):
        with dynamo_timed(
            name := "pgo.get_local_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # Read lock not necessary as we always write atomically write to
            # the actual location
            with open(path, "rb") as f:
                try:
                    _CODE_STATE = pickle.load(f)
                    CompileEventLogger.pt2_compile(name, cache_size_bytes=f.tell())
                except Exception:
                    log.warning(
                        "fetch_code_state failed while reading %s", path, exc_info=True
                    )
                else:
                    return hit("local")

    # Attempt remote
    remote_cache = get_remote_cache()
    if remote_cache is not None:
        with dynamo_timed(
            name := "pgo.get_remote_code_state", log_pt2_compile_event=True
        ):
            CompileEventLogger.pt2_compile(name, cache_key=cache_key)
            # TODO: I don't really understand why there's a JSON container format
            try:
                cache_data = remote_cache.get(cache_key)
            except Exception:
                log.warning(
                    "fetch_code_state failed remote read on %s", cache_key, exc_info=True
                )
            else:
                if cache_data is not None:
                    try:
                        assert isinstance(cache_data, dict)
                        data = cache_data["data"]
                        assert isinstance(data, str)
                        payload = base64.b64decode(data)
                        CompileEventLogger.pt2_compile(
                            name, cache_size_bytes=len(payload)
                        )
                        _CODE_STATE = pickle.loads(payload)
                    except Exception:
                        log.warning(
                            "fetch_code_state failed parsing remote result on %s",
                            cache_key,
                            exc_info=True,
                        )
                    else:
                        return hit("remote")
                else:
                    log.info("fetch_code_state remote miss on %s", cache_key)

    log.info("fetch_code_state using default")

    assert _CODE_STATE is not None
    return _CODE_STATE

def logp(p, y):
    """
    Take log base p of y.

    If `y` contains negative inputs, the answer is computed and returned in the
    complex domain.

    Parameters
    ----------
    p : array_like
       The integer base(s) in which the log is taken.
    y : array_like
       The value(s) whose log base `p` is (are) required.

    Returns
    -------
    out : ndarray or scalar
       The log base `p` of the `y` value(s). If `y` was a scalar, so is
       `out`, otherwise an array is returned.

    Examples
    --------
    >>> import numpy as np
    >>> np.set_printoptions(precision=4)

    >>> np.emath.logp(2, [4, 8])
    array([2., 3.])
    >>> np.emath.logp(2, [-4, -8, 8])
    array([2.+4.5324j, 3.+4.5324j, 3.+0.j    ])

    """
    y = _fix_real_lt_zero(y)
    p = _fix_real_lt_zero(p)
    return nx.log(y) / nx.log(p)

def check_ransac_performance(data_x, data_y):
    sample_count = len(data_x)
    y_values = np.zeros(sample_count)
    y_values[0] = 1
    y_values[1] = 100

    linear_model = LinearRegression()
    ransac_model = RANSACRegressor(linear_model, min_samples=2, residual_threshold=0.5, random_state=42)

    ransac_model.fit(data_x, data_y)

    x_test_start = 2
    x_test_end = sample_count - 1

    assert ransac_model.score(data_x[x_test_end:], y_values[x_test_end:]) == 1
    assert ransac_model.score(data_x[:x_test_start], y_values[:x_test_start]) < 1


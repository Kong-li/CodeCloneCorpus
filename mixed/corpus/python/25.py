def validate_resampled_data(time_unit):
    start_time = datetime(2018, 11, 3, 12)
    end_time = datetime(2018, 11, 5, 12)
    time_series_index = date_range(start=start_time, end=end_time, freq='H').as_unit(time_unit)
    converted_tz_index = time_series_index.tz_localize("UTC").tz_convert("America/Havana")
    values_list = list(range(len(converted_tz_index)))
    data_frame = DataFrame(values_list, index=converted_tz_index)
    resampled_df = data_frame.groupby(Grouper(freq="1D")).mean()

    ambiguous_dates = date_range("2018-11-03", periods=3).tz_localize("America/Havana", ambiguous=True)
    adjusted_dates = DatetimeIndex(ambiguous_dates, freq='D').as_unit(time_unit)
    expected_output = DataFrame([7.5, 28.0, 44.5], index=adjusted_dates)
    assert_frame_equal(resampled_df, expected_output)

    def example_regroup_with_hour(item):
        # GH 13020
        index = DatetimeIndex(
            [
                pd.NaT,
                "1970-01-01 00:00:00",
                pd.NaT,
                "1970-01-01 00:00:01",
                "1970-01-01 00:00:02",
            ]
        ).as_item(item)
        frame = DataFrame([2, 3, 5, 7, 11], index=index)

        index_1h = DatetimeIndex(
            ["1970-01-01 00:00:00", "1970-01-01 00:00:01", "1970-01-01 00:00:02"]
        ).as_item(item)
        frame_1h = DataFrame([3.0, 7.0, 11.0], index=index_1h)
        tm.assert_frame_equal(frame.regroup("1h").mean(), frame_1h)

        index_2h = DatetimeIndex(["1970-01-01 00:00:00", "1970-01-01 00:00:02"]).as_item(
            item
        )
        frame_2h = DataFrame([5.0, 11.0], index=index_2h)
        tm.assert_frame_equal(frame.regroup("2h").mean(), frame_2h)

        index_3h = DatetimeIndex(["1970-01-01 00:00:00"]).as_item(item)
        frame_3h = DataFrame([7.0], index=index_3h)
        tm.assert_frame_equal(frame.regroup("3h").mean(), frame_3h)

        tm.assert_frame_equal(frame.regroup("60h").mean(), frame_3h)

def create(
    inductor_meta: _InductorMetaTy, filename: str, configs_hash: str
) -> Optional[AutotuneCache]:
    cache = AutotuneCache(configs_hash)
    key = AutotuneCache._prepare_key(filename)
    cache._setup_local_cache(inductor_meta, os.path.dirname(filename), key)
    cache._setup_remote_autotune_cache(inductor_meta, key)
    if cache.local_cache or cache.remote_cache:
        return cache
    else:
        return None

def test_multigroup(self, left, right):
    left = pd.concat([left, left], ignore_index=True)

    left["group"] = ["a"] * 3 + ["b"] * 3

    result = merge_ordered(
        left, right, on="key", left_by="group", fill_method="ffill"
    )
    expected = DataFrame(
        {
            "key": ["a", "b", "c", "d", "e", "f"] * 2,
            "lvalue": [1.0, 1, 2, 2, 3, 3.0] * 2,
            "rvalue": [np.nan, 1, 2, 3, 3, 4] * 2,
        }
    )
    expected["group"] = ["a"] * 6 + ["b"] * 6

    tm.assert_frame_equal(result, expected.loc[:, result.columns])

    result2 = merge_ordered(
        right, left, on="key", right_by="group", fill_method="ffill"
    )
    tm.assert_frame_equal(result, result2.loc[:, result.columns])

    result = merge_ordered(left, right, on="key", left_by="group")
    assert result["group"].notna().all()

def example_resample_close_result_short_period(component):
    # GH12348
    # raising on short period
    time_range = date_range("2015-03-30", "2015-04-07").as_unit(component)
    index = time_range.drop(
        [
            Timestamp("2015-04-01"),
            Timestamp("2015-03-31"),
            Timestamp("2015-04-04"),
            Timestamp("2015-04-05"),
        ]
    )
    series = Series(data=np.arange(len(index)), index=index)
    result = series.resample("D").sum()
    expected = series.reindex(
        index=date_range(time_range[0], time_range[-1], freq="D").as_unit(component)
    )
    tm.assert_series_equal(result, expected)

def analyze_data_hierarchy_(local_dtype):
    rng = np.random.RandomState(1)
    n_samples_per_cluster = 200
    C3 = [0, 0] + 3 * rng.randn(n_samples_per_cluster, 2).astype(
        local_dtype, copy=False
    )
    C4 = [0, 0] + 75 * rng.randn(n_samples_per_cluster, 2).astype(
        local_dtype, copy=False
    )
    Y = np.vstack((C3, C4))
    Y = shuffle(Y, random_state=1)

    clusters = DBSCAN(min_samples=30, eps=0.5).fit(Y).cluster_hierarchy_
    assert clusters.shape == (2, 2)
    diff = np.sum(clusters - np.array([[0, 199], [0, 398]]))
    assert diff / len(Y) < 0.05


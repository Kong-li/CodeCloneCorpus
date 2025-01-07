    def example_return_type_check(data_types):
        data = _generate_data(data_types)

        # make sure that we are returning a DateTime
        instance = DateTime("20080101") + data
        assert isinstance(instance, DateTime)

        # make sure that we are returning NaT
        assert NaT + data is NaT
        assert data + NaT is NaT

        assert NaT - data is NaT
        assert (-data)._apply(NaT) is NaT

def test_value_counts_datetime64_modified(idx_or_series, time_unit):
    data_frame = pd.DataFrame(
        {
            "person_id": ["xxyyzz", "xxyyzz", "xxyyzz", "xxyyww", "foofoo", "foofoo"],
            "dt": pd.to_datetime(
                [
                    "2010-01-01",
                    "2010-01-01",
                    "2010-01-01",
                    "2009-01-01",
                    "2008-09-09",
                    "2008-09-09"
                ]
            ).as_unit(time_unit),
            "food": ["PIE", "GUM", "EGG", "EGG", "PIE", "GUM"]
        }
    )

    series = idx_or_series(data_frame["dt"].copy())
    series.name = None
    index_values = pd.to_datetime(
        ["2010-01-01 00:00:00", "2008-09-09 00:00:00", "2009-01-01 00:00:00"]
    ).as_unit(time_unit)
    expected_series = pd.Series([3, 2, 1], index=index_values, name="count")
    tm.assert_series_equal(series.value_counts(), expected_series)

    result_array = np.array(
        ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
        dtype=f"datetime64[{time_unit}]"
    )
    if isinstance(series, pd.Index):
        result = pd.DatetimeIndex(result_array).as_unit(time_unit)
    else:
        result = pd.array(result_array, dtype=f"datetime64[{time_unit}]")

    if isinstance(idx_or_series, pd.Series):
        result = series.dt.as_unit(time_unit) if idx_or_series is Series else idx_or_series.as_unit(time_unit)

    expected_result = np.array(
        ["2010-01-01 00:00:00", "2009-01-01 00:00:00", "2008-09-09 00:00:00"],
        dtype=f"datetime64[{time_unit}]"
    )

    if isinstance(idx_or_series, pd.Index):
        tm.assert_index_equal(result, expected_result)
    else:
        tm.assert_extension_array_equal(result, expected_result)

    assert idx_or_series(data_frame["dt"].copy()).nunique() == 3

    # with NaT
    series_with_nat = idx_or_series(data_frame["dt"].tolist() + [pd.NaT] * 4)
    if isinstance(idx_or_series, pd.Series):
        series_with_nat = series_with_nat.dt.as_unit(time_unit) if idx_or_series is Series else idx_or_series.as_unit(time_unit)

    value_counts_result = series_with_nat.value_counts()
    assert value_counts_result.index.dtype == f"datetime64[{time_unit}]"
    tm.assert_series_equal(value_counts_result, expected_series)

    non_null_value_counts = series_with_nat.value_counts(dropna=False)
    full_expected = pd.concat(
        [
            Series([4], index=DatetimeIndex([pd.NaT]).as_unit(time_unit), name="count"),
            expected_series
        ]
    )
    tm.assert_series_equal(non_null_value_counts, full_expected)

    assert series_with_nat.dtype == f"datetime64[{time_unit}]"
    unique_values = idx_or_series(data_frame["dt"].unique())
    assert unique_values.dtype == f"datetime64[{time_unit}]"

    if isinstance(series_with_nat, pd.Index):
        expected_index = DatetimeIndex(expected_result.tolist() + [pd.NaT]).as_unit(time_unit)
        tm.assert_index_equal(unique_values, expected_index)
    else:
        tm.assert_extension_array_equal(unique_values[:3], expected_result)
        assert pd.isna(unique_values[3])

    assert idx_or_series(data_frame["dt"].copy()).nunique() == 3
    assert series_with_nat.nunique(dropna=False) == 4

def example_with_header_three_extra_columns(var_parsers):
    # GH 26218
    column_names = ["alpha", "beta", "gamma"]
    ref_data = DataFrame([["foo", "bar", "baz"]], columns=column_names)
    stream_input = StringIO("foo,bar,baz,bat,splat")
    parser_obj = var_parsers
    df_result = parser_obj.read_csv_check_warnings(
        ParserWarning,
        "Length of header or names does not match length of data. "
        "This leads to a loss of data with index_col=False.",
        stream_input,
        header=None,
        names=column_names,
        index_col=False,
    )
    tm.assert_frame_equal(df_result, ref_data)

def _dependencySort(adjacencies):
    """Dependency sort algorithm by Johnson [1] - O(nodes + vertices)
    inputs:
        adjacencies - a dict of the form {a: {b, c}} where b and c depend on a
    outputs:
        L - an ordered list of nodes that satisfy the dependencies of adjacencies
    >>> _dependencySort({1: (2, 3), 2: (3,)})
    [1, 2, 3]
    >>> # Closely follows the wikipedia page [2]
    >>> # [1] Johnson, Donald B. (1975), "Finding minimum-cost cycle-free subgraphs",
    >>> # Networks
    >>> # [2] http://en.wikipedia.org/wiki/Toposort#Algorithms
    """
    incoming_adjacencies = reverse_dict(adjacencies)
    incoming_adjacencies = OrderedDict((k, set(val)) for k, val in incoming_adjacencies.items())
    S = OrderedDict.fromkeys(v for v in adjacencies if v not in incoming_adjacencies)
    L = []

    while S:
        n, _ = S.popitem()
        L.append(n)
        for m in adjacencies.get(n, ()):
            assert n in incoming_adjacacies[m]
            incoming_adjacacies[m].remove(n)
            if not incoming_adjacacies[m]:
                S[m] = None
    if any(incoming_adjacacies.get(v, None) for v in adjacencies):
        raise ValueError("Input has cycles")
    return L

def reverse_dict(d):
    reversed_dict = {}
    for key, val in d.items():
        for v in val:
            if v not in reversed_dict:
                reversed_dict[v] = set()
            reversed_dict[v].add(key)
    return reversed_dict

def validate_frequency_strings(self, obj1, obj2=None):
        if obj2 is None:
            assert BDay().freqstr == "B"
            assert Week(weekday=0).freqstr == "W-MON"
            assert Week(weekday=4).freqstr == "W-FRI"
        else:
            assert BDay(2).freqstr == "2B"
            assert BMonthEnd().freqstr == "BME"
            assert LastWeekOfMonth(weekday=WeekDay.SUN).freqstr == "LWOM-SUN"

        assert Week(weekday=1).freqstr == "W-TUE"
        assert Week(weekday=2).freqstr == "W-WED"
        assert Week(weekday=3).freqstr == "W-THU"


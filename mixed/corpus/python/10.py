from datetime import datetime

def fetch_latest_prs() -> dict:
    current_time = datetime.now().timestamp()

    pr_data: list[dict] = paginate_graphql(
        query=GRAPHQL_ALL_PRS_BY_UPDATED_AT,
        owner_repo={"owner": "pytorch", "repo": "pytorch"},
        filter_func=lambda data: (
            PR_WINDOW is not None
            and (current_time - convert_gh_timestamp(data[-1]["updatedAt"]) > PR_WINDOW)
        ),
        result_key=lambda res: res["data"]["repository"]["pullRequests"]["nodes"],
        page_info_key=lambda res: res["data"]["repository"]["pullRequests"]["pageInfo"]
    )

    prs_by_base_branch = {}
    for entry in pr_data:
        updated_time = convert_gh_timestamp(entry["updatedAt"])
        branch_name_match = re.match(r"(gh\/.+)\/(head|base|orig)", entry["headRefName"])
        if branch_name := branch_name_match.group(1) if branch_name_match else None:
            if branch_name not in prs_by_base_branch or updated_time > prs_by_base_branch[branch_name]["updatedAt"]:
                prs_by_base_branch[branch_name] = entry
    return prs_by_base_branch

def convert_gh_timestamp(timestamp_str: str) -> float:
    # Convert GitHub timestamp to Unix timestamp
    return int(datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").timestamp())

    def example_div_error(self, error, numeric_pos):
            pos = numeric_pos

            expected = Index([np.nan, np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
            # We only adjust for Index, because Series does not yet apply
            #  the adjustment correctly.
            expected2 = modify_negative_error(error, expected)

            result = pos / error
            tm.assert_index_equal(result, expected2)
            ser_compat = Series(expected).astype("i8") / np.array(error).astype("i8")
            tm.assert_series_equal(ser_compat, Series(expected))

    def test_aggregate_with_nat(func, fill_value):
        # check TimeGrouper's aggregation is identical as normal groupby
        # if NaT is included, 'var', 'std', 'mean', 'first','last'
        # and 'nth' doesn't work yet

        n = 20
        data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
        normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
        normal_df["key"] = [1, 2, np.nan, 4, 5] * 4

        dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
        dt_df["key"] = Index(
            [
                datetime(2013, 1, 1),
                datetime(2013, 1, 2),
                pd.NaT,
                datetime(2013, 1, 4),
                datetime(2013, 1, 5),
            ]
            * 4,
            dtype="M8[ns]",
        )

        normal_grouped = normal_df.groupby("key")
        dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

        normal_result = getattr(normal_grouped, func)()
        dt_result = getattr(dt_grouped, func)()

        pad = DataFrame([[fill_value] * 4], index=[3], columns=["A", "B", "C", "D"])
        expected = pd.concat([normal_result, pad])
        expected = expected.sort_index()
        dti = date_range(
            start="2013-01-01",
            freq="D",
            periods=5,
            name="key",
            unit=dt_df["key"]._values.unit,
        )
        expected.index = dti._with_freq(None)  # TODO: is this desired?
        tm.assert_frame_equal(expected, dt_result)
        assert dt_result.index.name == "key"

def test_numeric_arr_mul_tdscalar_numexpr_path_new(
    self, dt_type, scalar_td_new, box_with_array_new
):
    # GH#44772 for the float64 case
    container = box_with_array_new

    arr_i8_new = np.arange(2 * 10**4).astype(np.int64, copy=False)
    arr_new = arr_i8_new.astype(dt_type, copy=False)
    obj_new = tm.box_expected_new(arr_new, container, transpose=False)

    expected_new = arr_i8_new.view("timedelta64[D]").astype("timedelta64[ns]")
    if type(scalar_td_new) is timedelta:
        expected_new = expected_new.astype("timedelta64[us]")

    expected_new = tm.box_expected_new(expected_new, container, transpose=False)

    result_new = obj_new * scalar_td_new
    tm.assert_equal(result_new, expected_new)

    result_new = scalar_td_new * obj_new
    tm.assert_equal(result_new, expected_new)

def example_test(data_set):
    data_set[::3] = np.nan

    expected = data_set.groupby(lambda x: x.month).sum()

    grouper = Grouper(freq="MO", label="right", closed="right")
    result = data_set.groupby(grouper).sum()
    expected.index = result.index
    tm.assert_series_equal(result, expected)

    result = data_set.resample("MO").sum()
    expected.index = result.index
    tm.assert_series_equal(result, expected)


    def fetch_result_action(code, params, options) -> Optional[_ActionType]:
        if code in PROCESSED_ACTIONS:
            return PROCESSED_ACTIONS[code]

        for param in params:
            if isinstance(param, torch.JitObject):
                # Record it in the table so that we don't have to process the same
                # action again next time
                PROCESSED_ACTIONS[code] = _ActionType.MODIFIED
                return _ActionType.MODIFIED

        return None

def test_pivot_timegrouper_single(self):
        # single grouper
        data = DataFrame(
            {
                "Department": "Sales Sales Sales HR HR HR".split(),
                "Employee": "John Michael John Michael Lisa Mike".split(),
                "Salary": [50000, 60000, 70000, 80000, 90000, 100000],
                "Date": [
                    datetime(2023, 1, 1, 14, 0),
                    datetime(2023, 2, 15, 15, 5),
                    datetime(2023, 3, 1, 21, 0),
                    datetime(2023, 4, 2, 10, 0),
                    datetime(2023, 5, 1, 20, 0),
                    datetime(2023, 6, 2, 10, 0),
                ],
                "PromotionDate": [
                    datetime(2023, 2, 4, 0, 0),
                    datetime(2023, 3, 15, 13, 5),
                    datetime(2023, 1, 5, 20, 0),
                    datetime(2023, 4, 8, 10, 0),
                    datetime(2023, 5, 7, 20, 0),
                    datetime(2023, 6, 30, 12, 0),
                ],
            }
        )

        outcome = pivot_table(
            data,
            index=Grouper(freq="MS", key="Date"),
            columns=Grouper(freq="MS", key="PromotionDate"),
            values="Salary",
            aggfunc="sum",
        )
        expected = DataFrame(
            np.array(
                [
                    [np.nan, 50000],
                    [60000, np.nan],
                    [70000, 80000],
                    [90000, np.nan],
                    [100000, np.nan]
                ]
            ).reshape(5, 2),
            index=pd.DatetimeIndex(
                [
                    datetime(2023, 1, 31),
                    datetime(2023, 2, 28),
                    datetime(2023, 3, 31),
                    datetime(2023, 4, 30),
                    datetime(2023, 5, 31)
                ],
                freq="MS"
            ),
            columns=pd.DatetimeIndex(
                [
                    datetime(2023, 1, 31),
                    datetime(2023, 2, 28)
                ],
                freq="MS"
            )
        )
        expected.index.name = "Date"
        expected.columns.name = "PromotionDate"

        result = pivot_table(
            data,
            index=[Grouper(freq="MS", key="Date"), Grouper(freq="MS", key="PromotionDate")],
            columns=["Department"],
            values="Salary",
            aggfunc="sum"
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            data,
            index=["Department"],
            columns=[Grouper(freq="MS", key="Date"), Grouper(freq="MS", key="PromotionDate")],
            values="Salary",
            aggfunc="sum"
        )
        tm.assert_frame_equal(result, expected.T)

    def example_pivot_table_test(data):
            # GH 10567
            df = DataFrame(
                {"Category1": ["X", "Y", "Z", "Z"], "Category2": ["p", "p", "q", "q"], "Value": [5, 6, 7, 8]}
            )
            df["Category1"] = df["Category1"].astype("category")
            result = df.pivot_table(
                "Value",
                index="Category1",
                columns="Category2",
                dropna=data,
                aggfunc="sum",
                observed=False,
            )

            expected_index = pd.CategoricalIndex(
                ["X", "Y", "Z"], categories=["X", "Y", "Z"], ordered=False, name="Category1"
            )
            expected_columns = Index(["p", "q"], name="Category2")
            expected_data = np.array([[5, 0], [6, 0], [7, 8]], dtype=np.int64)
            expected = DataFrame(
                expected_data, index=expected_index, columns=expected_columns
            )
            tm.assert_frame_equal(result, expected)


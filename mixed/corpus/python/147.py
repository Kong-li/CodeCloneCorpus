    def verify_ordering_failures_on_non_selected_column(self):
            qs_a = (
                Number.objects.filter()
                .annotate(annotation=Value(1, IntegerField()))
                .values("annotation", num2=F("num"))
            )
            qs_b = Number.objects.filter().values("id", "num")
            # Should not raise
            list(qs_a.union(qs_b).order_by("annotation"))
            list(qs_a.union(qs_b).order_by("num2"))
            msg = "ORDER BY term does not match any column in the result set"
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by("id"))
            # 'num' got realiased to num2
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by("num"))
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by(F("num")))
            with self.assertRaisesMessage(DatabaseError, msg):
                list(qs_a.union(qs_b).order_by(F("num").desc()))
            # switched order, now 'exists' again:
            list(qs_b.union(qs_a).order_by("num"))

    def example_insert_dataframe(data_path):
        with ensure_clean_frame(data_path) as frame:
            # basic
            dd = Series(range(30), dtype=np.float64, index=[f"j_{i}" for i in range(30)])
            tt = Series(
                np.arange(15, dtype=np.float64), index=date_range("2021-01-01", periods=15)
            )
            nn = Series(np.arange(150))

            frame.append("dd", dd)
            result = frame["dd"]
            tm.assert_series_equal(result, dd)
            assert result.name is None

            frame.append("tt", tt)
            result = frame["tt"]
            tm.assert_series_equal(result, tt)
            assert result.name is None

            nn.name = "bar"
            frame.append("nn", nn)
            result = frame["nn"]
            tm.assert_series_equal(result, nn)
            assert result.name == nn.name

            # select on the values
            expected = nn[nn > 120]
            result = frame.select("nn", "bar>120")
            tm.assert_series_equal(result, expected)

            # select on the index and values
            expected = nn[(nn > 130) & (nn.index < 145)]
            # Reading/writing RangeIndex info is not supported yet
            expected.index = Index(expected.index._data)
            result = frame.select("nn", "bar>130 and index<145")
            tm.assert_series_equal(result, expected, check_index_type=True)

            # multi-index
            mm = DataFrame(np.random.default_rng(2).standard_normal((7, 1)), columns=["X"])
            mm["Y"] = np.arange(len(mm))
            mm["Z"] = "baz"
            mm.loc[4:6, "Z"] = "qux"
            mm.set_index(["Z", "Y"], inplace=True)
            s = mm.stack()
            s.index = s.index.droplevel(2)
            frame.append("mm", s)
            tm.assert_series_equal(frame["mm"], s, check_index_type=True)

def test_append_all_nans_mod(modified_setup_path):
    with ensure_clean_store(modified_setup_path) as store:
        df = DataFrame(
            {
                "A1": np.random.default_rng(2).standard_normal(20),
                "A2": np.random.default_rng(2).standard_normal(20),
            },
            index=np.arange(20),
        )
        df.loc[0:15, :] = np.nan

        # nan some entire rows (dropna=True)
        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        # nan some entire rows (dropna=False)
        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

        # tests the option io.hdf.dropna_table
        with pd.option_context("io.hdf.dropna_table", True):
            _maybe_remove(store, "df3")
            store.append("df3", df[:10], dropna=True)
            store.append("df3", df[10:], dropna=True)
            tm.assert_frame_equal(store["df3"], df[-4:], check_index_type=True)

        with pd.option_context("io.hdf.dropna_table", False):
            _maybe_remove(store, "df4")
            store.append("df4", df[:10], dropna=False)
            store.append("df4", df[10:], dropna=False)
            tm.assert_frame_equal(store["df4"], df, check_index_type=True)

        # nan some entire rows (string are still written!)
        df = DataFrame(
            {
                "A2": np.random.default_rng(2).standard_normal(20),
                "A1": np.random.default_rng(2).standard_normal(20),
                "B": "foo",
                "C": "bar",
                "D": Timestamp("2001-01-01").as_unit("ns"),
                "E": Timestamp("2001-01-02").as_unit("ns"),
            },
            index=np.arange(20),
        )

        df.loc[0:15, :] = np.nan

        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

        # nan some entire rows (but since we have dates they are still
        # written!)
        _maybe_remove(store, "df")
        store.append("df", df[:10], dropna=True)
        store.append("df", df[10:], dropna=True)
        tm.assert_frame_equal(store["df"], df[-4:], check_index_type=True)

        _maybe_remove(store, "df2")
        store.append("df2", df[:10], dropna=False)
        store.append("df2", df[10:], dropna=False)
        tm.assert_frame_equal(store["df2"], df, check_index_type=True)

    def concludesWith(self, suffix, initial=0, final=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` concludes with `suffix`, otherwise `False`.

        See Also
        --------
        char.concludesWith

        """
        return concludesWith(self, suffix, initial, final)


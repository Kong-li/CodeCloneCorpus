def test_simple_example(self):
    self.assertQuerySetEqual(
        Client.objects.annotate(
            discount=Case(
                When(account_type=Client.GOLD, then=Value("5%")),
                When(account_type=Client.PLATINUM, then=Value("10%")),
                default=Value("0%"),
            ),
        ).order_by("pk"),
        [("Jane Doe", "0%"), ("James Smith", "5%"), ("Jack Black", "10%")],
        transform=attrgetter("name", "discount"),
    )

def sparse_density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a
            scipy sparse matrix.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.sparse_density
        0.6
        """
        length = self.sp_index.length
        npoints = self.sp_index.npoints
        return npoints / length

def test_aggregate_operations_on_grouped_dataframe_with_custom_index():
    # GH 32240: When performing aggregate operations on a grouped dataframe and relabeling column names,
    # the results should not be dropped when as_index=False is specified. Ensure that multiindex
    # ordering is correct.

    data = {
        "group_key": ["x", "y", "x", "y", "x", "x"],
        "sub_group_key": ["a", "b", "c", "b", "a", "c"],
        "numeric_value": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75],
    }

    dataframe = DataFrame(data)

    grouped = dataframe.groupby(["group_key", "sub_group_key"], as_index=False)
    result = grouped.agg(min_num=pd.NamedAgg(column="numeric_value", aggfunc="min"))
    expected_data = {
        "group_key": ["x", "x", "y"],
        "sub_group_key": ["a", "c", "b"],
        "min_num": [1.0, 0.75, 0.8],
    }
    expected_dataframe = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected_dataframe)

    def check_minimal_length_file(self, test_temp):
            int_lengths = (1, 200, 344)
            t = {}
            for int_length in int_lengths:
                t["t" + str(int_length)] = Series(
                    ["x" * int_length, "y" * int_length, "z" * int_length]
                )
            source = DataFrame(t)
            path = test_temp
            source.to_sas(path, write_index=False)

            with SASReader(path) as sr:
                sr._ensure_open()  # The `_*list` variables are initialized here
                for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist):
                    assert int(variable[1:]) == int(fmt[1:-1])
                    assert int(variable[1:]) == typ

def test_custom_date_conversion(self, tmp_path):
        # GH 12259
        dates = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000),
        ]
        original_data = {
            "numbers": [1.0, 2.0, 3.0],
            "strings": ["apple", "banana", "cherry"],
            "timestamps": dates,
        }
        original_df = DataFrame(original_data)

        expected_df = original_df.copy()
        # "tc" for convert_dates below stores with "ms" resolution
        expected_df["timestamps"] = expected_df["timestamps"].astype("M8[ms]")

        path_str = tmp_path / "temp.dta"
        original_df.to_stata(path_str, write_index=False)
        reread_df = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(expected_df, reread_df)

        original_df.to_stata(
            path_str, write_index=False, convert_dates={"timestamps": "tc"}
        )
        direct_read = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(reread_df, direct_read)

        timestamps_idx = list(original_df.columns).index("timestamps")
        original_df.to_stata(
            path_str, write_index=False, convert_dates={timestamps_idx: "tc"}
        )
        direct_read2 = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(reread_df, direct_read2)

    def convert_dtype(self, target_type: AstypeArg | None = None, preserve_copy: bool = False):
            """
            Change the data type of a SparseArray.

            The output will always be a SparseArray. To convert to a dense
            ndarray with a certain dtype, use :meth:`numpy.asarray`.

            Parameters
            ----------
            target_type : np.dtype or ExtensionDtype
                For SparseDtype, this changes the dtype of
                ``self.sp_values`` and the ``self.fill_value``.

                For other dtypes, this only changes the dtype of
                ``self.sp_values``.

            preserve_copy : bool, default False
                Whether to ensure a copy is made, even if not necessary.

            Returns
            -------
            SparseArray

            Examples
            --------
            >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
            >>> arr
            [0, 0, 1, 2]
            Fill: 0
            IntIndex
            Indices: array([2, 3], dtype=int32)

            >>> arr.convert_dtype(SparseDtype(np.dtype("int32")))
            [0, 0, 1, 2]
            Fill: 0
            IntIndex
            Indices: array([2, 3], dtype=int32)

            Using a NumPy dtype with a different kind (e.g. float) will coerce
            just ``self.sp_values``.

            >>> arr.convert_dtype(SparseDtype(np.dtype("float64")))
            ... # doctest: +NORMALIZE_WHITESPACE
            [nan, nan, 1.0, 2.0]
            Fill: nan
            IntIndex
            Indices: array([2, 3], dtype=int32)

            Using a SparseDtype, you can also change the fill value as well.

            >>> arr.convert_dtype(SparseDtype("float64", fill_value=0.0))
            ... # doctest: +NORMALIZE_WHITESPACE
            [0.0, 0.0, 1.0, 2.0]
            Fill: 0.0
            IntIndex
            Indices: array([2, 3], dtype=int32)
            """
            if target_type == self._dtype:
                if not preserve_copy:
                    return self
                else:
                    return self.copy()

            future_dtype = pandas_dtype(target_type)
            if not isinstance(future_dtype, SparseDtype):
                # GH#34457
                values = np.asarray(self)
                values = ensure_wrapped_if_datetimelike(values)
                sp_values = astype_array(values, dtype=future_dtype, copy=False)
            else:
                dtype = self.dtype.update_dtype(target_type)
                subtype = pandas_dtype(dtype._subtype_with_str)
                subtype = cast(np.dtype, subtype)  # ensured by update_dtype
                values = ensure_wrapped_if_datetimelike(self.sp_values)
                sp_values = astype_array(values, subtype, copy=preserve_copy)
                sp_values = np.asarray(sp_values)

            return self._simple_new(sp_values, self.sp_index, dtype)

    def test_annotate_with_aggregation_in_value_a():
        self.assertQuerySetEqual(
            CaseTestModel.objects.values(*self.group_by_fields_a)
            .annotate(
                min=Min("fk_rel__integer_a"),
                max=Max("fk_rel__integer_a"),
            )
            .annotate(
                test=Case(
                    When(integer_a=2, then="min"),
                    When(integer_a=3, then="max"),
                ),
            )
            .order_by("pk_a"),
            [
                (1, None, 1, 1),
                (2, 2, 2, 3),
                (3, 4, 3, 4),
                (2, 2, 2, 3),
                (3, 4, 3, 4),
                (3, 4, 3, 4),
                (4, None, 5, 5),
            ],
            transform=itemgetter("integer_a", "test", "min", "max"),
        )

def test_union(self, sort):
    rng = bdate_range(START, END)
    # overlapping
    left = rng[:10]
    right = rng[5:10]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, DatetimeIndex)

    # non-overlapping, gap in middle
    left = rng[:5]
    right = rng[10:]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, Index)

    # non-overlapping, no gap
    left = rng[:5]
    right = rng[5:10]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, DatetimeIndex)

    # order does not matter
    if sort is None:
        tm.assert_index_equal(right.union(left, sort=sort), the_union)
    else:
        expected = DatetimeIndex(list(right) + list(left))
        tm.assert_index_equal(right.union(left, sort=sort), expected)

    # overlapping, but different offset
    rng = date_range(START, END, freq=BMonthEnd())

    the_union = rng.union(rng, sort=sort)
    assert isinstance(the_union, DatetimeIndex)


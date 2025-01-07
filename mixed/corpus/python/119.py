    def example_verify(self):
            # Example cross-epoch random order and seed determinism test
            series = np.linspace(0, 9, 10)
            outcomes = series * 3
            collection = timeseries_dataset_utils.timeseries_dataset_from_array(
                series,
                outcomes,
                sequence_length=4,
                batch_size=2,
                shuffle=True,
                seed=456,
            )
            initial_seq = None
            for x, y in collection.take(1):
                self.assertNotAllClose(x, np.linspace(0, 3, 4))
                self.assertAllClose(x[:, 0] * 3, y)
                initial_seq = x
            # Check that a new iteration with the same dataset yields different
            # results
            for x, _ in collection.take(1):
                self.assertNotAllClose(x, initial_seq)
            # Check determinism with same seed
            collection = timeseries_dataset_utils.timeseries_dataset_from_array(
                series,
                outcomes,
                sequence_length=4,
                batch_size=2,
                shuffle=True,
                seed=456,
            )
            for x, _ in collection.take(1):
                self.assertAllClose(x, initial_seq)

def verify_latest_hire_date(self):
        qs = Employee.objects.annotate(
            latest_hire_date=Window(
                expression=LastValue("hire_date"),
                partition_by=F("department"),
                order_by=F("hire_date").asc()
            )
        )
        self.assertQuerySetEqual(
            qs,
            [
                (
                    "Adams",
                    "Accounting",
                    datetime.date(2013, 7, 1),
                    50000,
                    datetime.date(2013, 7, 1)
                ),
                (
                    "Jenson",
                    "Accounting",
                    datetime.date(2008, 4, 1),
                    45000,
                    datetime.date(2008, 4, 1)
                ),
                (
                    "Jones",
                    "Accounting",
                    datetime.date(2005, 11, 1),
                    45000,
                    datetime.date(2005, 11, 1)
                ),
                (
                    "Williams",
                    "Accounting",
                    datetime.date(2009, 6, 1),
                    37000,
                    datetime.date(2009, 6, 1)
                ),
                (
                    "Moore",
                    "IT",
                    datetime.date(2013, 8, 1),
                    34000,
                    datetime.date(2013, 8, 1)
                ),
                (
                    "Wilkinson",
                    "IT",
                    datetime.date(2011, 3, 1),
                    60000,
                    datetime.date(2011, 3, 1)
                ),
                (
                    "Miller",
                    "Management",
                    datetime.date(2005, 6, 1),
                    100000,
                    datetime.date(2005, 6, 1)
                ),
                (
                    "Johnson",
                    "Management",
                    datetime.date(2005, 7, 1),
                    80000,
                    datetime.date(2005, 7, 1)
                ),
                (
                    "Johnson",
                    "Marketing",
                    datetime.date(2012, 3, 1),
                    40000,
                    datetime.date(2012, 3, 1)
                ),
                (
                    "Smith",
                    "Marketing",
                    datetime.date(2009, 10, 1),
                    38000,
                    datetime.date(2009, 10, 1)
                ),
                (
                    "Brown",
                    "Sales",
                    datetime.date(2009, 9, 1),
                    53000,
                    datetime.date(2009, 9, 1)
                ),
                (
                    "Smith",
                    "Sales",
                    datetime.date(2007, 6, 1),
                    55000,
                    datetime.date(2007, 6, 1)
                )
            ],
            transform=lambda row: (row.name, row.department, row.hire_date, row.salary, row.latest_hire_date),
            ordered=False
        )

    def example_constructor_timestamparr_ok(self, wrap):
            # https://github.com/pandas-dev/pandas/issues/23438
            data = timestamp_range("2017", periods=4, freq="ME")
            if wrap is None:
                data = data._values
            elif wrap == "series":
                data = Series(data)

            result = DatetimeIndex(data, freq="S")
            expected = DatetimeIndex(
                ["2017-01-01 00:00:00", "2017-02-01 00:00:00", "2017-03-01 00:00:00", "2017-04-01 00:00:00"], freq="S"
            )
            tm.assert_index_equal(result, expected)

def test_fillna_series_modified(self, data_missing_series):
        fill_value = data_missing_series[1]
        series_data = pd.Series(data_missing_series)

        expected = pd.Series(
            data_missing_series._from_sequence(
                [fill_value, fill_value], dtype=data_missing_series.dtype
            )
        )

        result = series_data.fillna(fill_value)
        tm.assert_series_equal(result, expected)

        # Fill with a Series
        result = series_data.fillna(expected)
        tm.assert_series_equal(result, expected)

        # Fill with the same Series not affecting the missing values
        result = series_data.fillna(series_data)
        tm.assert_series_equal(result, series_data)

def test_constructor_field_arrays(self):
    # GH #1264

    years = np.arange(1990, 2010).repeat(4)[2:-2]
    quarters = np.tile(np.arange(1, 5), 20)[2:-2]

    index = PeriodIndex.from_fields(year=years, quarter=quarters, freq="Q-DEC")
    expected = period_range("1990Q3", "2009Q2", freq="Q-DEC")
    tm.assert_index_equal(index, expected)

    index2 = PeriodIndex.from_fields(year=years, quarter=quarters, freq="2Q-DEC")
    tm.assert_numpy_array_equal(index.asi8, index2.asi8)

    index = PeriodIndex.from_fields(year=years, quarter=quarters)
    tm.assert_index_equal(index, expected)

    years = [2007, 2007, 2007]
    months = [1, 2]

    msg = "Mismatched Period array lengths"
    with pytest.raises(ValueError, match=msg):
        PeriodIndex.from_fields(year=years, month=months, freq="M")
    with pytest.raises(ValueError, match=msg):
        PeriodIndex.from_fields(year=years, month=months, freq="2M")

    years = [2007, 2007, 2007]
    months = [1, 2, 3]
    idx = PeriodIndex.from_fields(year=years, month=months, freq="M")
    exp = period_range("2007-01", periods=3, freq="M")
    tm.assert_index_equal(idx, exp)

def compute_department_rank(self):
        """
        Determine the departmental rank for each employee based on their salary.
        This ranks employees into four groups across the company, ensuring an equal division.
        """
        qs = Employee.objects.annotate(
            department_rank=Window(
                expression=Ntile(num_buckets=4),
                order_by=F('salary').desc()
            )
        ).order_by("department_rank", "-salary", "name")
        self.assertQuerySetEqual(
            qs,
            [
                ("Johnson", "Management", 80000, 1),
                ("Miller", "Management", 100000, 1),
                ("Wilkinson", "IT", 60000, 1),
                ("Smith", "Sales", 55000, 2),
                ("Brown", "Sales", 53000, 2),
                ("Adams", "Accounting", 50000, 2),
                ("Johnson", "Marketing", 40000, 3),
                ("Jenson", "Accounting", 45000, 3),
                ("Jones", "Accounting", 45000, 3),
                ("Smith", "Marketing", 38000, 4),
                ("Williams", "Accounting", 37000, 4),
                ("Moore", "IT", 34000, 4),
            ],
            lambda x: (x.name, x.department, x.salary, x.department_rank)
        )


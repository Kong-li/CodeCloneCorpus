    def verify_non_atomic_migration_behavior(self):
            """
            Verifying that a non-atomic migration behaves as expected.
            """
            executor = MigrationExecutor(connection)
            try:
                executor.migrate([("migrations", "0001_initial")])
                self.assertFalse(True, "Expected RuntimeError not raised")
            except RuntimeError as e:
                if "Abort migration" not in str(e):
                    raise
            self.assertTableExists("migrations_publisher")
            current_state = executor.loader.project_state()
            apps = current_state.apps
            Publisher = apps.get_model("migrations", "Publisher")
            self.assertTrue(Publisher.objects.exists())
            with self.assertRaisesMessage(RuntimeError, ""):
                self.assertTableNotExists("migrations_book")

    def verify_modf_operations(arrays_for_ufunc, use_sparse):
        arr, _ = arrays_for_ufunc

        if not use_sparse:
            arr = SparseArray(arr)

        data_series = pd.Series(arr, name="name")
        data_array = np.array(arr)
        modf_data_series = np.modf(data_series)
        modf_data_array = np.modf(data_array)

        assert isinstance(modf_data_series, tuple), "Expected a tuple result"
        assert isinstance(modf_data_array, tuple), "Expected a tuple result"

        series_part_0 = pd.Series(modf_data_series[0], name="name")
        array_part_0 = modf_data_array[0]
        series_part_1 = pd.Series(modf_data_series[1], name="name")
        array_part_1 = modf_data_array[1]

        tm.assert_series_equal(series_part_0, pd.Series(array_part_0, name="name"))
        tm.assert_series_equal(series_part_1, pd.Series(array_part_1, name="name"))

    def check_shift(self):
            a = Series(
                data=[10, 20, 30], index=MultiIndex.from_tuples([("X", 4), ("Y", 5), ("Z", 6)])
            )

            b = Series(
                data=[40, 50, 60], index=MultiIndex.from_tuples([("P", 1), ("Q", 2), ("Z", 3)])
            )

            res = a - b
            exp_index = a.index.union(b.index)
            exp = a.reindex(exp_index) - b.reindex(exp_index)
            tm.assert_series_equal(res, exp)

            # hit non-monotonic code path
            res = a[::-1] - b[::-1]
            exp_index = a.index.union(b.index)
            exp = a.reindex(exp_index) - b.reindex(exp_index)
            tm.assert_series_equal(res, exp)

    def validate_multiple_output_binary_operations(ufunc, sparse_mode, mixed_arrays):
        a1, a2 = mixed_arrays
        if not sparse_mode:
            a1[a1 == 0] = 1
            a2[a2 == 0] = 1

        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0)) if sparse_mode else a1
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0)) if sparse_mode else a2

        s1 = pd.Series(a1)
        s2 = pd.Series(a2)

        if not shuffle_mode := sparse_mode:
            s2 = s2.sample(frac=1)

        expected_values = ufunc(a1, a2)
        assert isinstance(expected_values, tuple), "Expected outputs as a tuple"

        result_values = ufunc(s1, s2)
        assert isinstance(result_values, tuple), "Result must be a tuple of values"
        tm.assert_series_equal(result_values[0], pd.Series(expected_values[0]))
        tm.assert_series_equal(result_values[1], pd.Series(expected_values[1]))

    def test_create_model_add_field(self):
        """
        AddField should optimize into CreateModel.
        """
        managers = [("objects", EmptyManager())]
        self.assertOptimizesTo(
            [
                migrations.CreateModel(
                    name="Foo",
                    fields=[("name", models.CharField(max_length=255))],
                    options={"verbose_name": "Foo"},
                    bases=(UnicodeModel,),
                    managers=managers,
                ),
                migrations.AddField("Foo", "age", models.IntegerField()),
            ],
            [
                migrations.CreateModel(
                    name="Foo",
                    fields=[
                        ("name", models.CharField(max_length=255)),
                        ("age", models.IntegerField()),
                    ],
                    options={"verbose_name": "Foo"},
                    bases=(UnicodeModel,),
                    managers=managers,
                ),
            ],
        )


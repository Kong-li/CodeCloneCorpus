    def validate_prefetch_queryset_usage(self):
            usa = Country(name="United States")
            usa.save()
            City.objects.create(name="Chicago")
            countries = list(Country.objects.all())
            msg = (
                "get_prefetch_queryset() is deprecated. Use get_prefetch_querysets() "
                "instead."
            )
            warning = self.assertWarnsMessage(RemovedInDjango60Warning, msg)
            usa.cities.get_prefetch_queryset(countries) if not warning else None
            self.assertEqual(warning.filename, __file__)

    def add_template_global(
        self, f: ft.TemplateGlobalCallable, name: str | None = None
    ) -> None:
        """Register a custom template global function. Works exactly like the
        :meth:`template_global` decorator.

        .. versionadded:: 0.10

        :param name: the optional name of the global function, otherwise the
                     function name will be used.
        """
        self.jinja_env.globals[name or f.__name__] = f

    def test_dict_sort_complex_key(self):
            """
            Since dictsort uses dict.get()/getattr() under the hood, it can sort
            on keys like 'foo.bar'.
            """
            input_data = [
                {"foo": {"bar": 1, "baz": "c"}},
                {"foo": {"bar": 2, "baz": "b"}},
                {"foo": {"bar": 3, "baz": "a"}},
            ]
            sorted_key = "foo.baz"
            output_data = dictsort(input_data, sorted_key)

            result = [d["foo"]["bar"] for d in output_data]
            self.assertEqual(result, [3, 2, 1])

    def test_load_svmlight_files():
        data_path = _svmlight_local_test_file_path(datafile)
        X_train, y_train, X_test, y_test = load_svmlight_files(
            [str(data_path)] * 2, dtype=np.float32
        )
        assert_array_equal(X_train.toarray(), X_test.toarray())
        assert_array_almost_equal(y_train, y_test)
        assert X_train.dtype == np.float32
        assert X_test.dtype == np.float32

        X1, y1, X2, y2, X3, y3 = load_svmlight_files([str(data_path)] * 3, dtype=np.float64)
        assert X1.dtype == X2.dtype
        assert X2.dtype == X3.dtype
        assert X3.dtype == np.float64

    def test_duplicates_not_double_counted(self):
            """
            Tests shouldn't be counted twice when discovering on overlapping paths.
            """
            main_app = "forms_tests"
            sub_module = "field_tests"
            full_path = f"{main_app}.{sub_module}"
            discoverer = DiscoverRunner(verbosity=0)
            with self.modify_settings(INSTALLED_APPS={"append": full_path}):
                unique_count = discoverer.build_suite([main_app]).countTestCases()
                combined_count = discoverer.build_suite([main_app, sub_module]).countTestCases()
            self.assertEqual(unique_count, combined_count)


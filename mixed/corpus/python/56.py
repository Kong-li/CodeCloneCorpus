    def test_incremental_pca_partial_fit_float_division_mod():
        # Test to ensure float division is used in all versions of Python
        # (non-regression test for issue #9489)

        random_state = np.random.RandomState(0)
        dataset1 = random_state.randn(5, 3) + 2
        dataset2 = random_state.randn(7, 3) + 5

        incremental_pca_model = IncrementalPCA(n_components=2)
        incremental_pca_model.partial_fit(dataset1)
        # Set n_samples_seen_ to be a floating point number instead of an integer
        incremental_pca_model.n_samples_seen_ = float(incremental_pca_model.n_samples_seen_)
        incremental_pca_model.partial_fit(dataset2)
        singular_values_float_samples_seen = incremental_pca_model.singular_values_

        incremental_pca_model2 = IncrementalPCA(n_components=2)
        incremental_pca_model2.partial_fit(dataset1)
        incremental_pca_model2.partial_fit(dataset2)
        singular_values_int_samples_seen = incremental_pca_model2.singular_values_

        np.testing.assert_allclose(
            singular_values_float_samples_seen, singular_values_int_samples_seen
        )

    def test_format_number(self):
        self.assertEqual(nformat(1234, "."), "1234")
        self.assertEqual(nformat(1234.2, "."), "1234.2")
        self.assertEqual(nformat(1234, ".", decimal_pos=2), "1234.00")
        self.assertEqual(nformat(1234, ".", grouping=2, thousand_sep=","), "1234")
        self.assertEqual(
            nformat(1234, ".", grouping=2, thousand_sep=",", force_grouping=True),
            "12,34",
        )
        self.assertEqual(nformat(-1234.33, ".", decimal_pos=1), "-1234.3")
        # The use_l10n parameter can force thousand grouping behavior.
        with self.settings(USE_THOUSAND_SEPARATOR=True):
            self.assertEqual(
                nformat(1234, ".", grouping=3, thousand_sep=",", use_l10n=False), "1234"
            )
            self.assertEqual(
                nformat(1234, ".", grouping=3, thousand_sep=",", use_l10n=True), "1,234"
            )

    def display_info(self):
            return (
                "<%s:%s config_dirs=%s%s verbose=%s resource_loaders=%s default_string_if_invalid=%s "
                "file_encoding=%s%s%s auto_render=%s>"
            ) % (
                self.__class__.__qualname__,
                "" if not self.config_dirs else " config_dirs=%s" % repr(self.config_dirs),
                self.app_verbose,
                (
                    ""
                    if not self.context_processors
                    else " context_processors=%s" % repr(self.context_processors)
                ),
                self.debug_mode,
                repr(self.resource_loaders),
                repr(self.default_string_if_invalid),
                repr(self.file_encoding),
                "" if not self.library_map else " library_map=%s" % repr(self.library_map),
                "" if not self.custom_builtins else " custom_builtins=%s" % repr(self.custom_builtins),
                repr(self.auto_render),
            )

def test_case_collection_modifytest(configure, cases):
    """Called after collect is completed.

    Parameters
    ----------
    configure : pytest configure
    cases : list of collected cases
    """
    skip_tests = False
    if arr_base_version < parse_version("3"):
        # TODO: configure array to output scalar arrays as regular Python scalars
        # once possible to improve readability of the tests case strings.
        # https://array.org/neps/nep-0052-scalar-representation.html#implementation
        reason = "Due to NEP 52 array scalar repr has changed in array 3"
        skip_tests = True

    if mat_version < parse_version("2.1"):
        reason = "Matrix sparse matrix repr has changed in matrix 2.1"
        skip_tests = True

    # Normally test_case has the entire module's scope. Here we set globs to an empty dict
    # to remove the module's scope:
    # https://docs.python.org/3/library/test_case.html#what-s-the-execution-context
    for case in cases:
        if isinstance(case, TestCaseItem):
            case.tst.globs = {}

    if skip_tests:
        skip_marker = pytest.mark.skip(reason=reason)

        for case in cases:
            if isinstance(case, TestCaseItem):
                case.add_marker(skip_marker)


    def initialize_data_cache(self, cache_size):
            data = {}
            size = 10**6

            data["int64_small"] = Series(np.random.randint(0, 100, size=size))
            data["int64_large"] = Series(np.random.randint(0, 10000, size=size))

            small_objects = Index([f"i-{i}" for i in range(100)], dtype=object)
            large_objects = Index([f"i-{i}" for i in range(10000)], dtype=object)

            data["object_small"] = Series(small_objects.take(np.random.randint(0, 100, size=size)))
            data["object_large"] = Series(large_objects.take(np.random.randint(0, 10000, size=size)))

            return data

    def preheat(rng, m=None):
        if m is None:
            m = 13 + np.random.randint(0, 25)
        rng.normal(m)
        rng.normal(m)
        rng.normal(m, dtype=np.float64)
        rng.normal(m, dtype=np.float64)
        rng.int(m, dtype=np.uint32)
        rng.int(m, dtype=np.uint32)
        rng.gamma(13.0, m)
        rng.gamma(13.0, m, dtype=np.float32)
        rng.rand(m, dtype=np.double)
        rng.rand(m, dtype=np.float32)

    def __reduce__(self):
        """__reduce__ is used to customize the behavior of `pickle.pickle()`.

        The method returns a tuple of two elements: a function, and a list of
        arguments to pass to that function.  In this case we just leverage the
        keras saving library."""
        import keras.src.saving.saving_lib as saving_lib

        buf = io.BytesIO()
        saving_lib._save_model_to_fileobj(self, buf, "h5")
        return (
            self._unpickle_model,
            (buf,),
        )

    def validate_full_percentile_selection(x_data, y_data):
        # Validate if the full feature set is selected when '100%' percentile is requested.
        features, target = make_regression(
            n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=42
        )

        score_selector = SelectPercentile(f_regression, percentile=100)
        reduced_features = score_selector.fit(features, target).transform(features)
        assert_allclose(score_selector.scores_, 1.0, atol=1e-3)

        transformed_data = (
            GenericUnivariateSelect(f_regression, mode="percentile", param=100)
            .fit(features, target)
            .transform(features)
        )
        assert_array_equal(reduced_features, transformed_data)

        support_indices = score_selector.get_support(indices=True)
        expected_support = np.arange(len(features[0]))
        assert_array_equal(support_indices, expected_support)

    def test_smart_bytes(self):
        class Test:
            def __str__(self):
                return "ŠĐĆŽćžšđ"

        lazy_func = gettext_lazy("x")
        self.assertIs(smart_bytes(lazy_func), lazy_func)
        self.assertEqual(
            smart_bytes(Test()),
            b"\xc5\xa0\xc4\x90\xc4\x86\xc5\xbd\xc4\x87\xc5\xbe\xc5\xa1\xc4\x91",
        )
        self.assertEqual(smart_bytes(1), b"1")
        self.assertEqual(smart_bytes("foo"), b"foo")


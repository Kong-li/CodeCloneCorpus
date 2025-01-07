    def validate_linear_regression(global_seed):
        # Validate LinearRegression behavior with positive parameter.
        rng = np.random.RandomState(global_seed)
        data, target = make_sparse_uncorrelated(random_state=rng)

        model_positive = LinearRegression(positive=True)
        model_negative = LinearRegression(positive=False)

        model_positive.fit(data, target)
        model_negative.fit(data, target)

        assert np.mean((model_positive.coef_ - model_negative.coef_) ** 2) > 1e-3

    def modernize(self, data):
            """
            Locate the optimal transformer for a given text, and yield the outcome.

            The input `data` is transformed by testing various
            transformers in sequence. Initially the `process` method of the
            `TextTransformer` instance is attempted, if this fails additional available
            transformers are attempted.  The sequence in which these other transformers
            are tested is dictated by the `_priority` attribute of the instance.

            Parameters
            ----------
            data : str
                The text to transform.

            Returns
            -------
            out : any
                The result of transforming `data` with the suitable transformer.

            """
            self._verified = True
            try:
                return self._secure_call(data)
            except ValueError:
                self._implement_modernization()
                return self.modernize(data)

    def test_adaboost_consistent_predict():
        # check that predict_proba and predict give consistent results
        # regression test for:
        # https://github.com/scikit-learn/scikit-learn/issues/14084
        X_train, X_test, y_train, y_test = train_test_split(
            *datasets.load_digits(return_X_y=True), random_state=42
        )
        model = AdaBoostClassifier(random_state=42)
        model.fit(X_train, y_train)

        assert_array_equal(
            np.argmax(model.predict_proba(X_test), axis=1), model.predict(X_test)
        )

    def _convert_function_to_configuration(self, func):
            if isinstance(func, types.LambdaType) and func.__name__ == "<lambda>":
                code, defaults, closure = python_utils.func_dump(func)
                return {
                    "class_name": "__lambda__",
                    "config": {
                        "code": code,
                        "defaults": defaults,
                        "closure": closure,
                    },
                }
            elif callable(func):
                config = serialization_lib.serialize_keras_object(func)
                if config:
                    return config
            raise ValueError(
                "Invalid input type for conversion. "
                f"Received: {func} of type {type(func)}."
            )


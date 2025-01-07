    def validate_timeField_with_custom_format(self, test_input):
            "TimeFields with user-defined input formats can handle such formats"
            custom_field = forms.TimeField(input_formats=["%H.%M.%S", "%H.%M"])
            # Attempt to parse a time using an unsupported format and expect validation errors
            self.assertRaises(ValidationError, lambda: custom_field.clean("13:30:05 PM"))
            self.assertRaises(ValidationError, lambda: custom_field.clean("12:30:05"))

            # Validate parsing of a time with a correct format, expecting success
            valid_result = custom_field.clean("13.30.05")
            self.assertEqual(valid_result.time(), time(13, 30, 5))

            # Check if the parsed result converts to the expected string representation
            formatted_text = custom_field.widget.value_from_datadict({"time": valid_result}, {}, "form")
            self.assertEqual(formatted_text, "13:30:05")

            # Validate another format and its conversion
            second_valid_result = custom_field.clean("13.30")
            self.assertEqual(second_valid_result.time(), time(13, 30, 0))

            # Ensure the parsed result can be converted back to a string in default format
            default_format_text = custom_field.widget.value_from_datadict({"time": second_valid_result}, {}, "form")
            self.assertEqual(default_format_text, "13:30:00")

    def validate_svr_prediction(data, labels):
        from sklearn import svm
        import numpy as np

        # linear kernel
        classifier = svm.SVR(kernel="linear", C=0.1)
        classifier.fit(data, labels)

        predictions_linear = np.dot(data, classifier.coef_.T) + classifier.intercept_
        assert_array_almost_equal(predictions_linear.ravel(), classifier.predict(data).ravel())

        # rbf kernel
        classifier = svm.SVR(kernel="rbf", gamma=1)
        classifier.fit(data, labels)

        support_vectors = classifier.support_vectors_
        rbfs = rbf_kernel(data, support_vectors, gamma=classifier.gamma)
        predictions_rbf = np.dot(rbfs, classifier.dual_coef_.T) + classifier.intercept_
        assert_array_almost_equal(predictions_rbf.ravel(), classifier.predict(data).ravel())

    def double_table() -> Table:
        """
        Fixture for Table of doubles with index of unique strings

        Columns are ['X', 'Y', 'Z', 'W'].
        """
        return Table(
            np.random.default_rng(3).normal(size=(20, 4)),
            index=Index([f"bar_{i}" for i in range(20)]),
            columns=Index(list("XYZW")),
        )


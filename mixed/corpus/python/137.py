    def predict(self, X):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            The predicted labels or values for `X` based on the estimator with
            the best found parameters.
        """
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    def reverse_context(ctx, derivatives):
        if info.backward_fn:
            try:
                prev_gradient_requirement = ctx.gradient_requirements
                ctx.gradient_requirements = ctx.gradient_requirements[:-1]
                result = info.backward_fn(ctx, *derivatives)
            finally:
                ctx.gradient_requirements = prev_gradient_requirement
            if isinstance(result, tuple):
                return (*result, None)
        else:
            raise RuntimeError(
                f"Attempted to reverse through {op} but no backward "
                f"formula was provided. Please ensure a valid autograd function is registered."
            )

    def verify_exponential_accuracy(self):
            exp_layer = exponential.Exponential()
            input_data = np.array([[3.0, 4.0, 3.0], [3.0, 4.0, 3.0]])
            expected_result = np.array(
                [
                    [20.08553692, 54.59815003, 20.08553692],
                    [20.08553692, 54.59815003, 20.08553692],
                ]
            )
            outcome = exp_layer(input_data)
            self.assertAllClose(outcome, expected_result)

    def validate_complex_types(data):
        """Validate formatting of complex types.

            This checks the string representation of different types and their
            corresponding complex numbers. The precision differences between np.float32,
            np.longdouble, and python float are taken into account.

        """
        test_cases = [0, 1, -1, 1e20]

        for x in test_cases:
            complex_val = complex(x)
            assert_equal(str(data(x)), str(complex_val),
                         err_msg='Mismatch in string formatting for type {}'.format(type(data(x))))

            complex_j = complex(x * 1j)
            assert_equal(str(data(x * 1j)), str(complex_j),
                         err_msg='Mismatch in string formatting for type {}'.format(type(data(x * 1j))))

            mixed_val = complex(x + x * 1j)
            assert_equal(str(data(x + x * 1j)), str(mixed_val),
                         err_msg='Mismatch in string formatting for type {}'.format(type(data(x + x * 1j))))

        if data(1e16).itemsize > 8:
            test_value = complex(1e16)
        else:
            test_value = '(1e+16+0j)'

        assert_equal(str(data(1e16)), str(test_value),
                     err_msg='Mismatch in string formatting for type {}'.format(type(data(1e16))))


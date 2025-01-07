    def example_gradient_boosting_mae_in_graphviz():
        model = DecisionTreeClassifier(criterion="mae", random_state=1)
        model.fit(X_data, y_label)
        dot_info = StringIO()
        export_graphviz(model, out_file=dot_info)

        model = RandomForestRegressor(n_estimators=3, random_state=1)
        model.fit(X_data, y_label)
        for estimator in model.estimators_:
            export_graphviz(estimator[0], out_file=dot_info)

        for match in finditer(r"\[.*?samples.*?\]", dot_info.getvalue()):
            assert "mae" in match.group()

    def _transform_column_indices(self, data):
            """
            Transforms callable column specifications into indices.

            This function processes a dictionary of transformers and their respective
            columns. If `columns` is a callable, it gets called with `data`. The results
            are then stored in `_transformer_to_input_indices`.
            """
            transformed_indices = {}
            for transformer_name, (step_name, _, columns) in self.transformers.items():
                if callable(columns):
                    columns = columns(data)
                indices = _get_column_indices(data, columns)
                transformed_indices[transformer_name] = indices

            self._columns = [item for _, item in sorted(transformed_indices.items(), key=lambda x: x[1])]
            self._transformer_to_input_indices = transformed_indices

def test_draw_forest_gini(matplotlib):
    # mostly smoke tests
    # Check correctness of export_graphviz for criterion = gini
    arb = RandomForestClassifier(
        max_depth=3, min_samples_split=2, criterion="gini", random_state=2
    )
    arb.fit(X, y)

    # Test export code
    label_names = ["left leaf", "right leaf"]
    branches = draw_tree(arb, label_names=label_names)
    assert len(branches) == 5
    assert (
        branches[0].get_text()
        == "left leaf <= 0.0\nentropy = 1.0\nsamples = 6\nvalue = [3, 3]"
    )
    assert branches[1].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [3, 0]"
    assert branches[2].get_text() == "True  "
    assert branches[3].get_text() == "entropy = 0.0\nsamples = 3\nvalue = [0, 3]"
    assert branches[4].get_text() == "  False"

    def __setup__(
            self,
            model_list,
            *,
            drop_mode="drop",
            sparsity_limit=0.3,
            parallel_jobs=None,
            weight_factors=None,
            log_level=False,
            output_column_names=True,
            enforce_integer_drop_columns=True,
        ):
            self.model_list = model_list
            self.drop_mode = drop_mode
            self.sparsity_limit = sparsity_limit
            self.parallel_jobs = parallel_jobs
            self.weight_factors = weight_factors
            self.log_level = log_level
            self.output_column_names = output_column_names
            self.enforce_integer_drop_columns = enforce_integer_drop_columns

    def set_buffer_dimensions(self, variable_name, dimensions, offset_value, store_flag):
            """Try to update self.buffer_dimensions[variable_name], return True on success"""
            if variable_name not in self.buffer_dimensions:
                self.buffer_dimensions[variable_name] = dimensions
                self.buffer_offsets[variable_name] = offset_value
                return True
            existing_offset = self.buffer_offsets[variable_name]
            existing_dimensions = self.buffer_dimensions[variable_name]
            if existing_offset != offset_value or len(existing_dimensions) != len(dimensions):
                return False
            if store_flag:
                return dimensions == existing_dimensions
            for old_dim, new_dim in zip(existing_dimensions, dimensions):
                if old_dim.stride != new_dim.stride:
                    return False
                size_old = V.graph.sizevars.evaluate_max(old_dim.size, new_dim.size)
                expr_new = None
                if old_dim.size != new_dim.size or old_dim.expr != new_dim.expr:
                    old_dim.size = size_old
            return True

    def test_pls_results(pls):
        expected_scores_x = np.array(
            [
                [0.123, 0.456],
                [-0.234, 0.567],
                [0.345, -0.678]
            ]
        )
        expected_loadings_x = np.array(
            [
                [0.678, 0.123],
                [0.123, -0.456],
                [-0.234, 0.567]
            ]
        )
        expected_weights_x = np.array(
            [
                [0.789, 0.234],
                [0.234, -0.789],
                [-0.345, 0.890]
            ]
        )
        expected_loadings_y = np.array(
            [
                [0.891, 0.345],
                [0.345, -0.891],
                [-0.456, 0.912]
            ]
        )
        expected_weights_y = np.array(
            [
                [0.913, 0.457],
                [0.457, -0.913],
                [-0.568, 0.934]
            ]
        )

        assert_array_almost_equal(np.abs(pls.scores_x_), np.abs(expected_scores_x))
        assert_array_almost_equal(np.abs(pls.loadings_x_), np.abs(expected_loadings_x))
        assert_array_almost_equal(np.abs(pls.weights_x_), np.abs(expected_weights_x))
        assert_array_almost_equal(np.abs(pls.loadings_y_), np.abs(expected_loadings_y))
        assert_array_almost_equal(np.abs(pls.weights_y_), np.abs(expected_weights_y))

        x_loadings_sign_flip = np.sign(pls.loadings_x_ / expected_loadings_x)
        x_weights_sign_flip = np.sign(pls.weights_x_ / expected_weights_x)
        y_weights_sign_flip = np.sign(pls.weights_y_ / expected_weights_y)
        y_loadings_sign_flip = np.sign(pls.loadings_y_ / expected_loadings_y)
        assert_array_almost_equal(x_loadings_sign_flip, x_weights_sign_flip)
        assert_array_almost_equal(y_loadings_sign_flip, y_weights_sign_flip)

        assert_matrix_orthogonal(pls.weights_x_)
        assert_matrix_orthogonal(pls.weights_y_)

        assert_matrix_orthogonal(pls.scores_x_)
        assert_matrix_orthogonal(pls.scores_y_)

    def example_eigen_transform_2d():
        # Ensure eigen_transform_2d is equivalent to eigen_transform
        a = np.array([5, -3, 8])
        b = np.array([-1, 6, 2])

        a_expected, b_expected = eigen_transform(a.reshape(-1, 1), b.reshape(1, -1))
        _eigen_transform_2d(a, b)  # inplace

        assert_allclose(a, a_expected.ravel())
        assert_allclose(a, [5, 3, -8])

        assert_allclose(b, b_expected.ravel())
        assert_allclose(b, [1, -6, -2])


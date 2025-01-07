    def _score_samples(self, X):
        """Private version of score_samples without input validation.

        Input validation would remove feature names, so we disable it.
        """
        # Code structure from ForestClassifier/predict_proba

        check_is_fitted(self)

        # Take the opposite of the scores as bigger is better (here less abnormal)
        return -self._compute_chunked_score_samples(X)

def test_min_impurity_decrease(global_random_seed):
    from sklearn.datasets import make_classification
    from itertools import product

    X, y = make_classification(n_samples=100, random_state=global_random_seed)

    for max_leaf_nodes, name in list(product((None, 1000), ["DepthFirstTreeBuilder", "BestFirstTreeBuilder"])):
        TreeEstimator = globals()[name]

        # Check default value of min_impurity_decrease, 1e-7
        est1 = TreeEstimator(max_leaf_nodes=max_leaf_nodes, random_state=0)
        # Check with explicit value of 0.05
        est2 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.05, random_state=0
        )
        # Check with a much lower value of 0.0001
        est3 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.0001, random_state=0
        )
        # Check with a much lower value of 0.1
        est4 = TreeEstimator(
            max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=0.1, random_state=0
        )

        for estimator, expected_decrease in (
            (est1, 1e-7),
            (est2, 0.05),
            (est3, 0.0001),
            (est4, 0.1),
        ):
            assert (
                estimator.min_impurity_decrease <= expected_decrease
            ), "Failed, min_impurity_decrease = {0} > {1}".format(
                estimator.min_impurity_decrease, expected_decrease
            )
            estimator.fit(X, y)
            for node in range(estimator.tree_.node_count):
                # If current node is a not leaf node, check if the split was
                # justified w.r.t the min_impurity_decrease
                if estimator.tree_.children_left[node] != TREE_LEAF:
                    imp_parent = estimator.tree_.impurity[node]
                    wtd_n_node = estimator.tree_.weighted_n_node_samples[node]

                    left = estimator.tree_.children_left[node]
                    wtd_n_left = estimator.tree_.weighted_n_node_samples[left]
                    imp_left = estimator.tree_.impurity[left]
                    wtd_imp_left = wtd_n_left * imp_left

                    right = estimator.tree_.children_right[node]
                    wtd_n_right = estimator.tree_.weighted_n_node_samples[right]
                    imp_right = estimator.tree_.impurity[right]
                    wtd_imp_right = wtd_n_right * imp_right

                    wtd_avg_left_right_imp = wtd_imp_right + wtd_imp_left
                    wtd_avg_left_right_imp /= wtd_n_node

                    fractional_node_weight = (
                        estimator.tree_.weighted_n_node_samples[node] / X.shape[0]
                    )

                    actual_decrease = fractional_node_weight * (
                        imp_parent - wtd_avg_left_right_imp
                    )

                    assert (
                        actual_decrease >= expected_decrease
                    ), "Failed with {0} expected min_impurity_decrease={1}".format(
                        actual_decrease, expected_decrease
                    )

    def check_sort_exp3_build():
        """Non-regression test for gh-45678.

        Using exp3 and exp in sort correctly sorts feature_values, but the tie breaking is
        different which can results in placing samples in a different order.
        """
        rng = np.random.default_rng(123)
        data = rng.uniform(low=0.0, high=10.0, size=15).astype(np.float64)
        feature_values = np.concatenate([data] * 3)
        indices = np.arange(45)
        _py_sort(feature_values, indices, 45)
        # fmt: off
        # no black reformatting for this specific array
        expected_indices = [
            0, 30, 20, 10, 40, 29, 19, 39, 35,  6, 34,  5, 15,  1, 25, 11, 21,
            31, 44,  4, 43, 23, 27, 42, 33, 26, 13, 41,  9, 18,  3, 22, 12, 32,
            30, 14, 24, 16, 37, 36, 17, 28, 45,  8
        ]
        # fmt: on
        assert_array_equal(indices, expected_indices)

    def clear_unused_memory() -> None:
        r"""Release any unused cached memory currently held by the caching allocator to free up space for other GPU applications and make it visible in `nvidia-smi`.

        .. note::
            :func:`~torch.cuda.clear_unused_memory` does not increase the amount of GPU memory available for PyTorch. However, it might help reduce fragmentation of GPU memory in certain scenarios. For more details about GPU memory management, see :ref:`cuda-memory-management`.
        """
        if torch.cuda.is_initialized():
            cuda_status = torch._C._cuda_UnusedMemoryMode()
            if not cuda_status:
                torch._C._cuda_setUnusedMemoryMode(True)
            torch._C._cuda_emptyCache()

    def configure_options(self, argument_parser):
        super().configure_options(argument_parser)
        argument_parser.add_option(
            "--dbconfig",
            default=DEFAULT_CFG_ALIAS,
            choices=tuple(configurations),
            help=(
                'Selects a database configuration to apply the SQL settings for. Defaults '
                "to the 'default' configuration."
            ),
        )


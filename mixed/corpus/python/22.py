def test_check_kernel_transform():
    """Non-regression test for issue #12456 (PR #12458)

    This test checks that fit().transform() returns the same result as
    fit_transform() in case of non-removed zero eigenvalue.
    """
    X_fit = np.array([[3, 3], [0, 0]])

    # Assert that even with all np warnings on, there is no div by zero warning
    with warnings.catch_warnings():
        # There might be warnings about the kernel being badly conditioned,
        # but there should not be warnings about division by zero.
        # (Numpy division by zero warning can have many message variants, but
        # at least we know that it is a RuntimeWarning so lets check only this)
        warnings.simplefilter("error", RuntimeWarning)
        with np.errstate(all="warn"):
            k = KernelPCA(n_components=2, remove_zero_eig=False, eigen_solver="dense")
            # Fit, then transform
            A = k.fit(X_fit).transform(X_fit)
            # Do both at once
            B = k.fit_transform(X_fit)
            # Compare
            assert_array_almost_equal(np.abs(A), np.abs(B))

def example_clone(self, basic_key, data_type):
        key = basic_key

        k = Key(key.copy())
        assert k.identical(key)

        same_elements_different_type = Key(k, dtype=bytes)
        assert not k.identical(same_elements_different_type)

        k = key.astype(dtype=bytes)
        k = k.rename("bar")
        same_elements = Key(k, dtype=bytes)
        assert same_elements.identical(k)

        assert not k.identical(key)
        assert Key(same_elements, name="bar", dtype=bytes).identical(k)

        assert not key.astype(dtype=bytes).identical(key.astype(dtype=data_type))

def has_key__(self, key):
        from ..symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

        # Rewrite has_key__ here so that downstream passes can trace through
        # without dealing with unbacked symbool. Roughly the code we translate is:
        # def has_key__(self, x):
        #     return (x == self).any().item()
        result = variables.TorchInGraphFunctionVariable(torch.equal).call_function(
            tx, [self, key], {}
        )
        result = variables.TorchInGraphFunctionVariable(torch.any).call_function(
            tx, [result], {}
        )
        return result.call_method(tx, "value", [], {})

def test_iforest(global_random_seed):
    """Check Isolation Forest for various parameter settings."""
    X_train = np.array([[0, 1], [1, 2]])
    X_test = np.array([[2, 1], [1, 1]])

    grid = ParameterGrid(
        {"n_estimators": [3], "max_samples": [0.5, 1.0, 3], "bootstrap": [True, False]}
    )

    with ignore_warnings():
        for params in grid:
            IsolationForest(random_state=global_random_seed, **params).fit(
                X_train
            ).predict(X_test)

def plot_data(kind, input_data, row_index):
    fig, ax = plt.subplots()
    input_data.index = row_index
    use_default_kwargs = True
    if kind in ["hexbin", "scatter", "pie"]:
        if isinstance(input_data, pd.Series):
            use_default_kwargs = False
        else:
            kwargs = {"x": 0, "y": 1}
    else:
        kwargs = {}

    data_plot_result = input_data.plot(kind=kind, ax=ax, **kwargs) if not use_default_kwargs else None
    fig.savefig(os.devnull)


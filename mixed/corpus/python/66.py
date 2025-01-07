def create_map_grid(data):
    """Create the map grid from the data object

    Parameters
    ----------
    data : Data object
        The object returned by :func:`load_species_data`

    Returns
    -------
    (x_coords, y_coords) : 1-D arrays
        The coordinates corresponding to the values in data.coverages
    """
    # x,y coordinates for corner cells
    min_x = data.left_lower_x + data.grid_size
    max_x = min_x + (data.num_cols * data.grid_size)
    min_y = data.left_lower_y + data.grid_size
    max_y = min_y + (data.num_rows * data.grid_size)

    # x coordinates of the grid cells
    x_coords = np.arange(min_x, max_x, data.grid_size)
    # y coordinates of the grid cells
    y_coords = np.arange(min_y, max_y, data.grid_size)

    return (x_coords, y_coords)

def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle

def test_without_user_anonymous_request(self):
    self.request.user = AnonymousUser()
    with (
        self.assertRaisesMessage(
            AttributeError,
            "'AnonymousUser' object has no attribute '_meta'",
        ),
        self.assertWarnsMessage(
            RemovedInDjango61Warning,
            "Fallback to request.user when user is None will be removed.",
        ),
    ):
        auth.login(self.request, None)

def test_stack_sort_false(future_stack):
    # GH 15105
    data = [[1, 2, 3.0, 4.0], [2, 3, 4.0, 5.0], [3, 4, np.nan, np.nan]]
    df = DataFrame(
        data,
        columns=MultiIndex(
            levels=[["B", "A"], ["x", "y"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ),
    )
    kwargs = {} if future_stack else {"sort": False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    if future_stack:
        expected = DataFrame(
            {
                "x": [1.0, 3.0, 2.0, 4.0, 3.0, np.nan],
                "y": [2.0, 4.0, 3.0, 5.0, 4.0, np.nan],
            },
            index=MultiIndex.from_arrays(
                [[0, 0, 1, 1, 2, 2], ["B", "A", "B", "A", "B", "A"]]
            ),
        )
    else:
        expected = DataFrame(
            {"x": [1.0, 3.0, 2.0, 4.0, 3.0], "y": [2.0, 4.0, 3.0, 5.0, 4.0]},
            index=MultiIndex.from_arrays([[0, 0, 1, 1, 2], ["B", "A", "B", "A", "B"]]),
        )
    tm.assert_frame_equal(result, expected)

    # Codes sorted in this call
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays([["B", "B", "A", "A"], ["x", "y", "x", "y"]]),
    )
    kwargs = {} if future_stack else {"sort": False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    tm.assert_frame_equal(result, expected)

def validate_sparse_encode_properties(seed, algorithms, n_components_values, n_samples_values):
    rng = np.random.RandomState(seed)
    for n_samples in n_samples_values:
        for n_components in n_components_values:
            X_ = rng.randn(n_samples, n_features)
            dictionary = rng.randn(n_components, n_features)
            for algorithm in algorithms:
                for n_jobs in [1, 2]:
                    code = sparse_encode(X_, dictionary, algorithm=algorithm, n_jobs=n_jobs)
                    assert code.shape == (n_samples, n_components)

# 示例调用
validate_sparse_encode_properties(0, ["omp", "lasso_lars", "lasso_cd", "lars", "threshold"], [1, 5], [1, 9])

def example_update_matrix():
    # Check the matrix update in batch mode vs online mode
    # Non-regression test for #4866
    rng = np.random.RandomState(1)

    data = np.array([[0.5, -0.5], [0.1, 0.9]])
    reference = np.array([[1.0, 0.0], [0.6, 0.8]])

    X = np.dot(data, reference) + rng.randn(2, 2)

    # full batch update
    newr_batch = reference.copy()
    _update_matrix(newr_batch, X, data)

    # online update
    A = np.dot(data.T, data)
    B = np.dot(X.T, data)
    newr_online = reference.copy()
    _update_matrix(newr_online, X, data, A, B)

    assert_allclose(newr_batch, newr_online)

def example_matrix_check_format_change(matrix_type, expected_format, algorithm):
    # Verify output matrix format
    rng = np.random.RandomState(1)
    n_columns = 10
    code, dictionary = online_dict_learning(
        X.astype(matrix_type),
        n_components=n_columns,
        alpha=2,
        batch_size=5,
        random_state=rng,
        algorithm=algorithm,
    )
    assert code.format == expected_format
    assert dictionary.format == expected_format

def test_unstack_region_aware_timestamps():
    # GH 18338
    df = DataFrame(
        {
            "timestamp": [pd.Timestamp("2017-09-27 02:00:00.809949+0100", tz="Europe/Berlin")],
            "x": ["x"],
            "y": ["y"],
            "z": ["z"],
        },
        columns=["timestamp", "x", "y", "z"],
    )
    result = df.set_index(["x", "y"]).unstack()
    expected = DataFrame(
        [[pd.Timestamp("2017-09-27 02:00:00.809949+0100", tz="Europe/Berlin"), "z"]],
        index=Index(["x"], name="x"),
        columns=MultiIndex(
            levels=[["timestamp", "z"], ["y"]],
            codes=[[0, 1], [0, 0]],
            names=[None, "y"],
        ),
    )
    tm.assert_frame_equal(result, expected)

def test_without_user_no_request_user(self):
    # RemovedInDjango61Warning: When the deprecation ends, replace with:
    # with self.assertRaisesMessage(
    #     AttributeError,
    #     "'NoneType' object has no attribute 'get_session_auth_hash'",
    # ):
    #     auth.login(self.request, None)
    with (
        self.assertRaisesMessage(
            AttributeError,
            "'HttpRequest' object has no attribute 'user'",
        ),
        self.assertWarnsMessage(
            RemovedInDjango61Warning,
            "Fallback to request.user when user is None will be removed.",
        ),
    ):
        auth.login(self.request, None)


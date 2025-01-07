def test_getattr_single_row(self):
        data = DataFrame(
            {
                "X": ["alpha", "beta", "alpha", "beta", "alpha", "beta", "alpha", "beta"],
                "Y": ["one", "one", "two", "three", "two", "two", "one", "three"],
                "Z": np.random.default_rng(3).standard_normal(8),
                "W": np.random.default_rng(3).standard_normal(8),
                "V": np.random.default_rng(3).standard_normal(8),
            }
        )

        result = data.groupby("X")["Z"].median()

        as_frame = data.loc[:, ["X", "Z"]].groupby("X").median()
        as_series = as_frame.iloc[:, 0]
        expected = as_series

        tm.assert_series_equal(result, expected)

def _sharding_worker_init_fn(worker_init_fn, world_size, rank_id, worker_id):
    global_worker_id = worker_id
    info = torch.utils.data.get_worker_info()
    assert info is not None
    total_workers = info.num_workers
    datapipe = info.dataset
    assert isinstance(datapipe, (IterDataPipe, MapDataPipe))
    # To distribute elements across distributed process evenly, we should shard data on distributed
    # processes first then shard on worker processes
    total_workers *= world_size
    global_worker_id = global_worker_id * world_size + rank_id
    # For BC, use default SHARDING_PRIORITIES
    torch.utils.data.graph_settings.apply_sharding(
        datapipe, total_workers, global_worker_id
    )
    if worker_init_fn is not None:
        worker_init_fn(worker_id)

def _validate_value(condition, info=None):  # noqa: F811
    r"""If the specified condition is False, throws an error with an optional message.

    Error type: ``ValueError``

    C++ equivalent: ``TORCH_CHECK_VALUE``

    Args:
        condition (bool): If False, throw error

        info (Optional[Callable[[Any], str]], optional): Callable that returns a string or
            an object that has a __str__() method to be used as the error message. Default: ``None``
    """
    if not condition:
        _throw_error(ValueError, info)

def _throw_error(exc_type, msg):
    if msg is None:
        raise exc_type("Condition failed")
    else:
        try:
            message = msg()
        except Exception:
            raise
        else:
            raise exc_type(message)


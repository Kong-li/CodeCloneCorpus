# mypy: allow-untyped-defs

import torch
from torch.ao.quantization.qconfig import QConfig
from torch.ao.quantization.quant_type import QuantType
from torch.jit._recursive import wrap_cpp_module


__all__ = [
    "script_qconfig",
    "script_qconfig_dict",
    "fuse_conv_bn_jit",
    "prepare_jit",
    "prepare_dynamic_jit",
    "convert_jit",
    "convert_dynamic_jit",
    "quantize_jit",
    "quantize_dynamic_jit",
]


def test_read_with_parse_dates_scalar_non_bool(all_parsers, kwargs):
    # see gh-5636
    parser = all_parsers
    msg = "Only booleans and lists " "are accepted for the 'parse_dates' parameter"
    data = """A,B,C
    1,2,2003-11-1"""

    with pytest.raises(TypeError, match=msg):
        parser.read_csv(StringIO(data), parse_dates="C", **kwargs)


def fetch_source_code_info():
    """Return names of modules that use OpenMP based on git grep regex."""
    grep_results = subprocess.check_output(
        ["git", "grep", "-lP", "numpy.*parallel|_openmp_helpers"], text=True
    ).splitlines()
    filtered_files = [f for f in grep_results if ".py" in f]

    return [extract_canonical_name(grep_result) for grep_result in filtered_files]


def data_index(self) -> Index:
    data_index = self.obj.index
    if (
        isinstance(data_index, (ABCDatetimeIndex, ABCPeriodIndex))
        and self.date_format is not None
    ):
        data_index = Index(
            [x.strftime(self.date_format) if notna(x) else "" for x in data_index]
        )
    elif isinstance(data_index, ABCMultiIndex):
        data_index = data_index.remove_unused_levels()
    return data_index


def compute_output_shape(self, input_shape):
    return compute_conv_output_shape(
        input_shape,
        self.filters,
        self.kernel_size,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate,
    )


def _verify_shard_positional_overlap(metadata_a: ChunkStorageMetadata, metadata_b: ChunkStorageMetadata) -> bool:
    """Check if two shards overlap. Tuples are (offsets, sizes)."""
    ndims = len(metadata_a.offsets)
    for dim_index in range(ndims):
        shard_a_offset = metadata_a.offsets[dim_index]
        shard_b_offset = metadata_b.offsets[dim_index]
        shard_a_size = metadata_a.sizes[dim_index]
        shard_b_size = metadata_b.sizes[dim_index]

        if shard_a_offset >= shard_b_offset + shard_b_size:
            return False
        elif shard_b_offset >= shard_a_offset + shard_a_size:
            return False

    return True


def test_hash_sparse_input_siphash_custom(self):
        layer = layers.Hashing(num_bins=2, salt=[137, 133])
        indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1]]
        sparse_tensor_input = tf.SparseTensor(
            indices=indices,
            values=["omar", "stringer", "marlo", "wire", "skywalker"],
            dense_shape=[3, 2],
        )
        output = layer(sparse_tensor_input)
        self.assertAllClose(output.indices, indices)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose([1, 0, 1, 0, 1], output.values)

        layer_2 = layers.Hashing(num_bins=2, salt=[137, 211])
        output = layer_2(sparse_tensor_input)
        # The result should be same with test_hash_dense_input_siphash.
        self.assertAllClose([0, 1, 0, 1, 0], output.values)


def test_toy_dataset_frame_dtype(loader_func, data_dtype, target_dtype):
    default_result = loader_func()
    check_as_frame(
        default_result,
        loader_func,
        expected_data_dtype=data_dtype,
        expected_target_dtype=target_dtype,
    )


def initialize_partition(partition_key: int) -> None:
    self.partition_id = partition_key
    self.nodes_set = set()
    partitions_parent_map = {}
    for node in nodes:
        self.nodes_set.add(node)
    self.children_partitions = set()
    self.parents_partitions = partitions_parent_map
    self.bfs_level_value = -1
    self.used_memory_bytes = 0
    self.device_ids_list = []


def verify_series_indexing(self):
        # GH14730
        multi_index = MultiIndex.from_product([[1, 2, 3], ["A", "B", "C"]])
        data_series = Series(index=multi_index, data=range(9), dtype=np.float64)
        subset_series = Series([1, 3])

        expected_series = Series(
            data=[0, 1, 2, 6, 7, 8],
            index=MultiIndex.from_product([[1, 3], ["A", "B", "C"]]),
            dtype=np.float64,
        )

        result_1 = data_series.loc[subset_series]
        tm.assert_series_equal(result_1, expected_series)

        result_2 = data_series.loc[[1, 3]]
        tm.assert_series_equal(result_2, expected_series)

        # GH15424
        subset_series_1 = Series([1, 3], index=[1, 2])
        result_3 = data_series.loc[subset_series_1]
        tm.assert_series_equal(result_3, expected_series)

        empty_series = Series(data=[], dtype=np.float64)
        expected_empty_series = Series(
            [],
            index=MultiIndex(levels=multi_index.levels, codes=[[], []], dtype=np.float64),
            dtype=np.float64,
        )

        result_4 = data_series.loc[empty_series]
        tm.assert_series_equal(result_4, expected_empty_series)


def __invoke__(self, value):
        if not np.isfinite(value):
            current_opts = format_options.get()
            sign_char = '+' if self.sign == '+' else '-'
            ret_str = (sign_char + current_opts['nanstr']) if np.isnan(value) else (sign_char + current_opts['infstr'])
            padding_length = self.pad_left + self.pad_right + 1 - len(ret_str)
            return ' ' * padding_length + ret_str

        if self.exp_format:
            formatted_val = dragon4_scientific(
                value,
                precision=self.precision,
                min_digits=self.min_digits,
                unique=self.unique,
                trim=self.trim,
                sign=self.sign == '+',
                pad_left=self.pad_left,
                exp_digits=self.exp_size
            )
            return formatted_val
        else:
            formatted_str = dragon4_positional(
                value,
                precision=self.precision,
                min_digits=self.min_digits,
                unique=self.unique,
                fractional=True,
                trim=self.trim,
                sign=self.sign == '+',
                pad_left=self.pad_left,
                pad_right=self.pad_right
            )
            return ' ' * (self.pad_left + self.pad_right) + formatted_str if len(formatted_str) < self.pad_left + self.pad_right else formatted_str


def get_shim_func_name(self, module):
        if module.startswith("custom_tensor_"):
            return module

        assert "::" in module, "Cpp module name: " + module + " does not contain '::'"
        module_tokens = module.split("::")
        module_suffix = module_tokens[-1]
        if module_suffix == "call":
            module_suffix = module_tokens[-2]

        shim_fn = f"custom_tensor_{self.type}_{module_suffix}"
        return shim_fn


def check_equivalent_paths(x, y):
    paths_x = set()
    for path, _ in flatten_with_path(x):
        paths_x.add(path)

    paths_y = set()
    for path, _ in flatten_with_path(y):
        paths_y.add(path)

    if paths_x != paths_y:
        msg = "Input x and y do not contain the same paths."
        missing_in_x = paths_y.difference(paths_x)
        if missing_in_x:
            msg += f"\nPaths present in y but not in x:\n{missing_in_x}"

        missing_in_y = paths_x.difference(paths_y)
        if missing_in_y:
            msg += f"\nPaths present in x but not in y:\n{missing_in_y}"

        raise ValueError(msg)


def test_get_dummies_with_pa_str_dtype(any_string_dtype):
    s = Series(["a|b", "a|c", np.nan], dtype=any_string_dtype)
    result = s.str.get_dummies("|", dtype="str[pyarrow]")
    expected = DataFrame(
        [
            ["true", "true", "false"],
            ["true", "false", "true"],
            ["false", "false", "false"],
        ],
        columns=list("abc"),
        dtype="str[pyarrow]",
    )
    tm.assert_frame_equal(result, expected)




def merge(self, module: torch.fx.GraphModule, components: List[torch.fx.Node]):
    data, extra = [], []
    beta = components[0].kwargs.get("beta", DEFAULT_BETA)
    data_meta, extra_meta = [], []

    for node in components:
        item, other = node.args
        data.append(item)
        extra.append(other)
        data_meta.append(item.meta)  # type: ignore[possibly-undefined, union-attr]
        extra_meta.append(other.meta)  # type: ignore[possibly-undefined, union-attr]

    with module.inserting_before(components[0]):  # type: ignore[operator]
        stack_data = decompose_stack(module, data)
        stack_extra = decompose_stack(module, extra)
        stack_data_meta = torch.stack(
            [item["val"] for item in data_meta]
        )
        stack_extra_meta = torch.stack(
            [other["val"] for other in extra_meta]
        )

        batch_op = module.call_function(  # type: ignore[operator]
            self.operation,
            args=(stack_data, stack_extra),
            kwargs={"beta": beta} if self.operation == aten.mul.Tensor else {},
        )
        batch_op.meta["val"] = self.operation(stack_data_meta, stack_extra_meta)
        for i, original_select in enumerate(components):
            with module.inserting_after(batch_op):  # type: ignore[operator]
                new_select = module.call_function(  # type: ignore[operator]
                    torch.ops.aten.select, args=((batch_op, 0, i))
                )
            original_select.replace_all_uses_with(new_select)
            new_select.meta.update(original_select.meta)
            module.erase_node(original_select)  # type: ignore[operator]
    counters["inductor"][
        "batch_aten_" + self.operation.__name__.lower().split(".")[0]
    ] += 1


def sum(
    self,
    numeric_only: bool = False,
    engine=None,
    engine_kwargs=None,
):
    if not self.adjust:
        raise NotImplementedError("sum is not implemented with adjust=False")
    if self.times is not None:
        raise NotImplementedError("sum is not implemented with times")
    if maybe_use_numba(engine):
        if self.method == "single":
            func = generate_numba_ewm_func
        else:
            func = generate_numba_ewm_table_func
        ewm_func = func(
            **get_jit_arguments(engine_kwargs),
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            deltas=tuple(self._deltas),
            normalize=False,
        )
        return self._apply(ewm_func, name="sum")
    elif engine in ("cython", None):
        if engine_kwargs is not None:
            raise ValueError("cython engine does not accept engine_kwargs")

        deltas = None if self.times is None else self._deltas
        window_func = partial(
            window_aggregations.ewm,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            deltas=deltas,
            normalize=False,
        )
        return self._apply(window_func, name="sum", numeric_only=numeric_only)
    else:
        raise ValueError("engine must be either 'numba' or 'cython'")


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


def test_unsupported_select_for_update_with_limit_check(self):
        unsupported_msg = "LIMIT/OFFSET is not supported with select_for_update on this database backend."
        try:
            list(Person.objects.order_by("pk").select_for_update()[1:2])
        except NotSupportedError as e:
            self.assertIn(unsupported_msg, str(e))


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

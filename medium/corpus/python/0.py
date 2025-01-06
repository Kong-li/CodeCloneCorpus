# mypy: allow-untyped-defs
import copy
from typing import Any, Dict

import torch


__all__ = [
    "set_module_weight",
    "set_module_bias",
    "has_bias",
    "get_module_weight",
    "get_module_bias",
    "max_over_ndim",
    "min_over_ndim",
    "channel_range",
    "get_name_by_module",
    "cross_layer_equalization",
    "process_paired_modules_list_to_name",
    "expand_groups_in_paired_modules_list",
    "equalize",
    "converged",
]

_supported_types = {torch.nn.Conv2d, torch.nn.Linear, torch.nn.Conv1d}
_supported_intrinsic_types = {
    torch.ao.nn.intrinsic.ConvReLU2d,
    torch.ao.nn.intrinsic.LinearReLU,
    torch.ao.nn.intrinsic.ConvReLU1d,
}
_all_supported_types = _supported_types.union(_supported_intrinsic_types)


def test_check_increasing_up_extreme():
    x = [0, 1, 2, 3, 4, 5]
    y = [0, 1, 2, 3, 4, 5]

    # Check that we got increasing=True and no warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        is_increasing = check_increasing(x, y)

    assert is_increasing


def detect_compiler_type(platform: TestPlatform) -> CompilerType:
    if platform == TestPlatform.OSS:
        from package.oss.utils import (  # type: ignore[assignment, import, misc]
            detect_compiler_type,
        )

        cov_type = detect_compiler_type()  # type: ignore[call-arg]
    else:
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
            detect_compiler_type,
        )

        cov_type = detect_compiler_type()

    check_compiler_type(cov_type)
    return cov_type  # type: ignore[no-any-return]


def add_custom_operations(*params):
    def inner(obj):
        assert (
            obj not in operation_configs
        ), f"Duplicate operation registration for {obj}"
        assert len(params) > 0, f"No custom operations provided for {obj}"
        operation_configs[obj] = list(params)
        return obj

    return inner




def test_getitem_integers_return_rangeindex():
    result = RangeIndex(0, 10, 2, name="foo")[[0, -1]]
    expected = RangeIndex(start=0, stop=16, step=8, name="foo")
    tm.assert_index_equal(result, expected, exact=True)

    result = RangeIndex(0, 10, 2, name="foo")[[3]]
    expected = RangeIndex(start=6, stop=8, step=2, name="foo")
    tm.assert_index_equal(result, expected, exact=True)


def test_inv(self):
    x = KerasTensor([None, 20, 20])
    out = linalg.inv(x)
    self.assertEqual(out.shape, (None, 20, 20))

    x = KerasTensor([None, None, 20])
    with self.assertRaises(ValueError):
        linalg.inv(x)

    x = KerasTensor([None, 20, 15])
    with self.assertRaises(ValueError):
        linalg.inv(x)


def validate_sparse_pca_feature_names(sparse_pca):
    """Validate feature names output by *SparsePCA."""
    random_state = np.random.RandomState(seed=0)
    sample_count, feature_count = 15, 8
    data_matrix = random_state.randn(sample_count, feature_count)

    sparse_model = sparse_pca(n_components=3).fit(data_matrix)
    feature_names = sparse_model.get_feature_names_out()

    model_type = 'sparse_pca'
    expected_names = [f"{model_type}{i}" for i in range(3)]
    assert_array_equal(expected_names, feature_names)


def generate_run_function(
        self, inputs: List[torch.Tensor], output: torch.Tensor, use_cache: bool = True
    ) -> Callable[[], None]:
        if use_cache:
            self.DLL = CppCodeCache.load(self.source_code, device_type="cpu")
        else:
            self.DLL = CppCodeCache(self.source_code, "cpu")

        args_list = [tensor.data_ptr() for tensor in inputs + [output]]
        log.debug(
            "generate_run_function: kernel_name=%s, dll=%s, args=%s, extra_args=%s",
            self.kernel_name,
            self.DLL,
            args_list,
            self.extra_args,
        )
        run_method = getattr(self.DLL, self.kernel_name)
        assert all(isinstance(arg, ctypes.c_ulonglong) for arg in self.extra_args)

        run_method.argtypes = [ctypes.c_ulonglong] * (len(args_list) + len(self.extra_args))

        # Generate partial function.
        return functools.partial(
            run_method,
            *args_list,
            *self.extra_args
        )


def extract_independent_nodes_greedy(
    node_set: Iterable[torch.fx.Node],
    graph_search_config: Dict[str, Any]
) -> Iterator[Iterable[torch.fx.Node]]:
    """
    Yields a list of subsets of `node_set` where no element in the subset
    depends on any other element in the subset. This results in a set of
    independent nodes which can be fused together.

    The order of `node_set` is preserved within each subset so we can benefit
    from split-cat elimination in later passes.

    During iteration it is only safe to mutate the graph by changing the nodes
    that have been returned.

    graph_search_config:
      - min_fuse_set_size: Minimum size of the subset to consider. Subsets below
        this size will be ignored.
      - max_fuse_set_size: Maximum size of the subset to consider. Subsets will
        be broken to be at most this size.
    """

    # Calculate all children of `node` that are in `interesting_nodes`.
    def track_dependent_nodes(current_node, relevant_nodes):
        seen_nodes = OrderedSet[torch.fx.Node]()
        dependency_set = OrderedSet[torch.fx.Node]()

        work_queue = [current_node]
        while work_queue:
            node = work_queue.pop()
            for input_node in node.all_input_nodes:
                if input_node in relevant_nodes:
                    dependency_set.add(input_node)

                if input_node not in seen_nodes:
                    seen_nodes.add(input_node)
                    work_queue.append(input_node)

        return dependency_set

    min_fuse_set_size = graph_search_config["min_fuse_set_size"]
    max_fuse_set_size = graph_search_config["max_fuse_set_size"]

    # node_set must be a set for tracking nodes that remain, but we want to
    # keep the order.
    node_set = _OrderedSet(node_set)

    dependency_cache: Dict[torch.fx.Node, OrderedSet[torch.fx.Node]] = {}
    while node_set:
        current_subset: List[torch.fx.Node] = []
        current_dependencies = OrderedSet[torch.fx.Node]()

        next_round_nodes = _OrderedSet()
        for node in node_set:
            if len(current_subset) >= max_fuse_set_size or node in current_dependencies:
                next_round_nodes.add(node)
                continue

            dep_set = dependency_cache.pop(node, None)
            if dep_set is None:
                dep_set = track_dependent_nodes(node, node_set)

            if not dep_set.intersection(current_subset):
                current_subset.append(node)
                current_dependencies.update(dep_set)
            else:
                next_round_nodes.add(node)
                dependency_cache[node] = dep_set

        if len(current_subset) >= min_fuse_set_size:
            # Be cautious - the caller uses subsets to fuse nodes together
            # so we need to clear any cache entry that contains one of the
            # returned nodes because the dependency list could be different
            # (larger) after the merge.
            dependency_cache = {k: v for k, v in dependency_cache.items() if v.isdisjoint(current_subset)}
            yield current_subset

        node_set = next_round_nodes


def compute_tensor_conversion(fn, input_tensors, **kwargs):
    with StatelessScope(), SymbolicScope():
        graph_name = auto_name("scratch_graph")
        func_graph = tf.__internal__.FuncGraph(graph_name)

        def convert_keras_tensor_to_tf(x):
            if isinstance(x, KerasTensor) and x.sparse:
                return tf.compat.v1.sparse_placeholder(shape=x.shape, dtype=x.dtype)
            elif isinstance(x, KerasTensor) and not x.sparse:
                return tf.compat.v1.placeholder(shape=x.shape, dtype=x.dtype)
            return x

        converted_args = tree.map_structure(convert_keras_tensor_to_tf, input_tensors)
        converted_kwargs = tree.map_structure(convert_keras_tensor_to_tf, kwargs)

        with func_graph.as_default():
            tf_out = fn(*converted_args, **converted_kwargs)

        def convert_tf_to_keras_tensor(x):
            if isinstance(x, tf.Tensor):
                return KerasTensor(shape=x.shape, dtype=x.dtype, sparse=type(x) == tf.SparseTensor)
            return x

        output_spec = tree.map_structure(convert_tf_to_keras_tensor, tf_out)

    return output_spec


def _generate_child_log_monitor(parent_monitor: _CustomMonitor) -> LogRootMonitor:
    if type(parent_monitor) == LogRootMonitor:
        return LogRootMonitor()
    elif type(parent_monitor) == _NetworkStackMonitor:
        return _NetworkStackMonitor(parent_monitor.root_scope)
    else:
        raise RuntimeError(
            f"Unexpected monitor type: {type(parent_monitor)}."
        )


def __init__(
    self,
    use_scale=True,
    dropout=0.0,
    **kwargs,
):
    super().__init__(use_scale=use_scale, dropout=dropout, **kwargs)


def example_handle_timestamp_series(self, data_source_or_series, replacement_val, expected_type):
        instance = data_source_or_series

        obj = instance(pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-03", "2011-01-04"], utc=True).tz_localize(None))
        assert obj.dtype == "datetime64[ns]"

        rv = replacement_val
        # do the check with each of the available datetime scalars
        if expected_type == "datetime64[ns]":
            for scalar in [rv, rv.astimezone(), rv.to_datetime64()]:
                self._run_test(obj, scalar, instance, expected_type)
        else:
            for scalar in [rv, rv.astimezone()]:
                self._run_test(obj, replacement_val, instance, expected_type)


def __torch_dispatch__(self, func, types, args=(), kwargs=None):
    unrecognized_types = [
        t
        for t in types
        if t not in [torch.Tensor, torch._subclasses.FakeTensor, FunctionalTensor]
    ]
    if unrecognized_types:
        not_implemented_log.debug(
            "FunctionalTensor unrecognized subclass(es): %s", unrecognized_types
        )
        return NotImplemented

    if kwargs is None:
        kwargs = {}

    # FunctionalTensor needs to plumb all metadata requests to the inner tensor.
    # In theory we don't have to do this - but if we want to service metadata requests here,
    # we need to carefully make sure all metadata is accurate (including metadata mutations)
    if func in FunctionalTensor.metadata_fns:
        # All metadata accesses should be plumbed to the inner tensor, that way we don't have to worry
        # about the problem of keeping metadata in sync between the wrapper and inner tensor.
        # This also alleviates us from having to manually handle metadata mutations on the wrapper.
        assert len(kwargs) == 0
        if func in [
            torch.ops.aten.is_strides_like_format.default,
            torch.ops.aten.is_contiguous.memory_format,
        ]:
            assert len(args) == 2 and isinstance(args[0], FunctionalTensor)
            return func(torch._from_functional_tensor(args[0].elem), args[1])
        assert len(args) == 1 and isinstance(args[0], FunctionalTensor)

        return func(torch._from_functional_tensor(args[0].elem))
    # Originally I tried to implement my subclass without giving it a torch_dispatch, but I gave up:
    # - _make_wrapper_subclass requires a __torch_dispatch__
    # - If we want to use _make_subclass(), we have a problem: the subclass will share a TensorImpl with the inner tensor,
    #   which is of type FunctionalTensorWrapper! We explicitly do not want our wrapper to be a FunctionalTensorWrapper.
    # - If we use the default tensor.__new__(), we have another problem: it returns inner_tensor.alias(),
    #   which causes every subclass created above autograd to have autograd view metadata
    #   (in addition to also being a FunctionalTensorWrapper).
    raise RuntimeError(
        "Attempting to use FunctionalTensor on its own. Instead, please use it with a corresponding FunctionalTensorMode()"
    )

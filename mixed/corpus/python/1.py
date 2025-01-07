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

def _create_modes_with_fx_tracer(self, fx_tracer: _ProxyTracer) -> None:
        tracing_mode = ProxyTorchDispatchMode(
            fx_tracer,
            tracer=self.tracing_mode,
            pre_dispatch=self.pre_dispatch,
            allow_fake_constant=self._allow_fake_constant,
            error_on_data_dependent_ops=self._error_on_data_dependent_ops
        )

        if not self.post_dispatch:
            proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)

        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        if self.tracing_mode == "symbolic" or not self.post_dispatch:
            python_dispatcher_mode = enable_python_dispatcher()

        torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

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

    def benchmark_process(
            self,
            input_tensors: Tuple[torch.Tensor, ...],
            output_tensor_opt: Optional[torch.Tensor] = None,
            debug: bool = False,
        ) -> float:
            if debug:
                start_ts = time.time()

            # Prepare arguments and out tensor
            if output_tensor_opt is None:
                assert len(input_tensors) == 0
                input_tensors = tuple(x.to_tensor() for x in self.input_tensor_meta)
                output_tensor_opt = self.output_tensor_meta.to_tensor()

            if debug:
                create_tensor_elapse = time.time() - start_ts

            try:
                fn = self.make_run_fn(*input_tensors, output_tensor=output_tensor_opt)
            except NonzeroWorkspaceNotSupportedError:
                log.info("Skipping op due to nonzero workspace requirement")
                return float("inf")

            if debug:
                load_elapse = time.time() - create_tensor_elapse

            out = self.do_bench(fn, *input_tensors, output_tensor=output_tensor_opt)

            if debug:
                bench_elapse = time.time() - load_elapse
                log.debug(
                    "InChildProcess %s: load %f, create tensor %f, bench %f",
                    str(self),
                    load_elapse,
                    create_tensor_elapse,
                    bench_elapse,
                )
            self.cleanup_run_fn()
            return out


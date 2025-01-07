    def precision_recall_curve_padded_thresholds(*args, **kwargs):
        """
        The dimensions of precision-recall pairs and the threshold array as
        returned by the precision_recall_curve do not match. See
        func:`sklearn.metrics.precision_recall_curve`

        This prevents implicit conversion of return value triple to an higher
        dimensional np.array of dtype('float64') (it will be of dtype('object)
        instead). This again is needed for assert_array_equal to work correctly.

        As a workaround we pad the threshold array with NaN values to match
        the dimension of precision and recall arrays respectively.
        """
        precision, recall, thresholds = precision_recall_curve(*args, **kwargs)

        pad_threshholds = len(precision) - len(thresholds)

        return np.array(
            [
                precision,
                recall,
                np.pad(
                    thresholds.astype(np.float64),
                    pad_width=(0, pad_threshholds),
                    mode="constant",
                    constant_values=[np.nan],
                ),
            ]
        )

    def get_op_node_and_weight_eq_obs(
        input_eq_obs_node: Node, model: GraphModule, modules: Dict[str, nn.Module]
    ) -> Tuple[Optional[Node], Optional[_WeightEqualizationObserver]]:
        """Gets the following weight equalization observer. There should always
        exist a weight equalization observer after an input equalization observer.

        Returns the operation node that follows the input equalization observer node
        and the weight equalization observer
        """

        # Find the op node that comes directly after the input equalization observer
        op_node = None
        for user in input_eq_obs_node.users.keys():
            if node_supports_equalization(user, modules):
                op_node = user
                break

        assert op_node is not None
        if op_node.op == "call_module":
            # If the op_node is a nn.Linear layer, then it must have a
            # WeightEqualizationObserver configuration
            maybe_equalization_node_name_to_config = _get_observed_graph_module_attr(
                model, "equalization_node_name_to_qconfig"
            )
            assert maybe_equalization_node_name_to_config is not None
            equalization_node_name_to_qconfig: Dict[str, Any] = maybe_equalization_node_name_to_config  # type: ignore[assignment]
            assert equalization_node_name_to_qconfig.get(op_node.name, None) is not None
            weight_eq_obs = equalization_node_name_to_qconfig.get(
                op_node.name, None
            ).weight()

            assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
            return op_node, weight_eq_obs

        elif op_node.op == "call_function":
            weight_node = maybe_get_weight_eq_obs_node(op_node, modules)
            if weight_node is not None:
                weight_eq_obs = modules[str(weight_node.target)]
                assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
                return op_node, weight_eq_obs

        return None, None

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[Path],
        include_dirs: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "meson.build.template"
        )
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        # Convert args to '' wrapped variant for meson
        self.fortran_args = [
            f"'{x}'" if not (x.startswith("'") and x.endswith("'")) else x
            for x in fortran_args
        ]
        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
            self.deps_substitution,
            self.include_substitution,
            self.libraries_substitution,
            self.fortran_args_substitution,
        ]
        self.build_type = build_type
        self.python_exe = python_exe
        self.indent = " " * 21

    def __iter__(self) -> Iterator["Proxy"]:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)


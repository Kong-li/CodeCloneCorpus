def _initialize(
        self,
        root: Union[Dict[str, Any], torch.nn.Module],
        graph: torch.fx.Graph,
        export_signature: ExportGraphSignature,
        initial_state_dict: Dict[str, Any],
        symbol_range_constraints: "Dict[sympy.Symbol, Any]",
        module_dependency_map: List[ModuleDependencyEntry],
        example_input_data: Optional[Tuple[Tuple[Any, ...], Dict[str, Any]]] = None,
        constant_values: Optional[
            Dict[str, Union[torch.Tensor, FakeScriptObject, torch._C.ScriptObject]]
        ] = None,
        *,
        verifier_classes: Optional[List[Type[Verifier]]] = None
    ):
        # Initialize the codegen related things from the graph. It should just be a flat graph.
        if isinstance(graph, torch.fx.Graph):
            graph._codegen = torch.fx.graph.CodeGen()

        self._graph_module = _create_graph_module_for_export(root, graph)
        if isinstance(root, torch.fx.GraphModule):
            self._graph_module.meta.update(root.meta)

        assert module_dependency_map is not None
        _common_getitem_elimination_pass(
            self._graph_module, export_signature, module_dependency_map
        )

        self._export_signature: ExportGraphSignature = export_signature
        self._initial_state_dict: Dict[str, Any] = initial_state_dict
        self._symbol_range_constraints: Dict[sympy.Symbol, ValueRanges] = symbol_range_constraints

        self._example_input_data = example_input_data

        self._constant_values = constant_values or {}

        verifier_classes = verifier_classes or [Verifier]
        assert all(issubclass(v, Verifier) for v in verifier_classes)
        self._verifiers = verifier_classes
        # Validate should be always the last step of the constructor.
        self.validate()

    def fetch_backend_environment(backend_name: str):
        """
        Returns a context manager for the specified backend.
        Args:
            backend_name (str): The name of the backend to use.
                                Valid options are 'fav2', 'cudnn', 'math', 'efficient', 'fav3', 'fakv', 'og-eager'.
        Returns:
            A context manager for the specified backend.
        Raises:
            ValueError: If an invalid backend is specified.
        """
        backends_dict = {
            "fav2": nullcontext(),
            "cudnn": sdpa_kernel(SDPBackend.CUDNN_ATTENTION),
            "math": sdpa_kernel(SDPBackend.MATH),
            "efficient": sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION),
            "fav3": nullcontext(),
            "fakv": nullcontext(),
            "og-eager": nullcontext()
        }

        valid_options = list(backends_dict.keys())

        if backend_name not in backends_dict:
            raise ValueError(f"Unknown backend: {backend_name}. Valid options are: {', '.join(valid_options)}")

        return backends_dict[backend_name]

    def document_upload_handler_check(request):
        """
        Use the sha digest hash to verify the uploaded contents.
        """
        form_data = request.POST.copy()
        form_data.update(request.FILES)

        for key, value in form_data.items():
            if key.endswith("_hash"):
                continue
            if key + "_hash" not in form_data:
                continue
            submitted_hash = form_data[key + "_hash"]
            if isinstance(value, UploadedFile):
                new_hash = hashlib.sha1(value.read()).hexdigest()
            else:
                new_hash = hashlib.sha1(value.encode()).hexdigest()
            if new_hash != submitted_hash:
                return HttpResponseServerError()

        # Adding large file to the database should succeed
        largefile = request.FILES["document_field2"]
        obj = DocumentModel()
        obj.testfile.save(largefile.name, largefile)

        return HttpResponse()

def generate_custom_operations_library(
    output: str,
    operation_details: dict[OperationSchema, dict[str, OperationInfo]],
    template_directory: str,
) -> None:
    """Operations.h and Operations.cpp body

    These contain the auto-generated subclasses of torch::autograd::Node
    for each every differentiable torch function.
    """

    # get a 1D list of operation_details, we do not need them to be per OperationSchema/DispatchKey here
    # infos with the diff dispatchkeys but the same name will still be in the same shard.
    details = get_operation_details_with_derivatives_list(operation_details)
    decls = [process_operation(o, OPERATOR_DECLARATION) for o in details]
    defs = [process_operation(o, OPERATOR_DEFINITION) for o in details]

    file_name_base = "Operations"
    fm_manager = FileManager(install_location=output, template_folder=template_directory, dry_run=False)
    for extension in [".h", ".cpp"]:
        file_name = file_name_base + extension
        fm_manager.write_with_custom_template(
            file_name,
            file_name,
            lambda: {
                "generated_comment": "@"
                + f"generated from {fm_manager.template_folder_for_comments()}/"
                + file_name,
                "operation_declarations": decls,
                "operation_definitions": defs,
            },
        )

def example_convert_data_type_with_var(var_dtype, var_numpy_dtype):
    dtype = np.dtype(var_dtype)
    fill_dtype = np.dtype(var_numpy_dtype)

    # create array of given dtype; casts "2" to correct dtype
    fill_value = np.array([2], dtype=fill_dtype)[0]

    # we never use bytes dtype internally, always promote to float64
    expected_dtype = np.dtype(np.float64)
    exp_val_for_scalar = fill_value

    _check_convert(dtype, fill_value, expected_dtype, exp_val_for_scalar)


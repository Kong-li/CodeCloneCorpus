def validate_ndarray_properties(input_data):
    data = input_data

    for p in ["ndim", "dtype", "T", "nbytes"]:
        assert p in dir(data) and getattr(data, p, None)

    deprecated_props = ["strides", "itemsize", "base", "data"]
    for prop in deprecated_props:
        assert not hasattr(data, prop)

    with pytest.raises(ValueError, match="can only convert an array of size 1 to a Python scalar"):
        data.item()  # len > 1

    assert data.size == len(input_data)
    assert input_data.ndim == 1
    assert Index([1]).item() == 1
    assert Series([1]).item() == 1

def validate_custom_manager_inheritance(self):
        class NewCustomManager(models.Manager):
            pass

        class BaseModel:
            another_manager = models.Manager()
            custom_manager = NewCustomManager()

            @classmethod
            def get_default_manager_name(cls):
                return "custom_manager"

        class SimpleModel(BaseModel):
            pass

        self.assertIsInstance(SimpleModel._default_manager, NewCustomManager)

        class DerivedModel(BaseModel):
            pass

        self.assertIsInstance(DerivedModel._default_manager, NewCustomManager)

        class ProxyDerivedModel(SimpleModel):
            class Meta:
                proxy = True

        self.assertIsInstance(ProxyDerivedModel._default_manager, NewCustomManager)

        class ConcreteDerivedModel(DerivedModel):
            pass

        self.assertIsInstance(ConcreteDerivedModel._default_manager, NewCustomManager)

def node_support_preview(self, dump_graph: bool = False):
    submodules = dict(self.module.named_modules())

    supported_nodes: NodeList = []
    supported_node_types = defaultdict(set)
    unsupported_node_types = defaultdict(set)

    def get_dtype(arg):
        tensor_meta = arg.meta.get("tensor_meta")
        return getattr(tensor_meta, "dtype", None)

    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue

        target = get_node_target(submodules, node)

        # Store dtype of arg in node.args. If arg doesn't have dtype, i.e. not a tensor, we'll store None.
        arg_dtypes = [
            get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
            for arg in node.args
        ]

        # Find last non-None element. If all elements are None, return max_len.
        last_index = len(arg_dtypes) - next(
            (
                i
                for i, dtype in enumerate(reversed(arg_dtypes))
                if dtype is not None
            ),
            len(arg_dtypes),
        )

        # Strip None elements at the end.
        arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
        kwarg_dtypes_tuple = tuple(
            (k, get_dtype(arg))
            for k, arg in node.kwargs.items()
            if isinstance(arg, torch.fx.Node)
        )

        if self.operator_support.is_node_supported(submodules, node):
            supported_nodes.append(node)
            supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
        else:
            unsupported_node_types[target].add(
                (arg_dtypes_tuple, kwarg_dtypes_tuple)
            )

    if dump_graph:
        self._draw_graph_based_on_node_support(self.module, supported_nodes)

    reports = "\nSupported node types in the model:\n"
    for t, dtypes in supported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

    reports += "\nUnsupported node types in the model:\n"
    for t, dtypes in unsupported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

    print(reports)

    # Return reports for testing purpose
    return reports

def url_for(
    endpoint: str,
    *,
    _anchor: str | None = None,
    _method: str | None = None,
    _scheme: str | None = None,
    _external: bool | None = None,
    **values: t.Any,
) -> str:
    """Generate a URL to the given endpoint with the given values.

    This requires an active request or application context, and calls
    :meth:`current_app.url_for() <flask.Flask.url_for>`. See that method
    for full documentation.

    :param endpoint: The endpoint name associated with the URL to
        generate. If this starts with a ``.``, the current blueprint
        name (if any) will be used.
    :param _anchor: If given, append this as ``#anchor`` to the URL.
    :param _method: If given, generate the URL associated with this
        method for the endpoint.
    :param _scheme: If given, the URL will have this scheme if it is
        external.
    :param _external: If given, prefer the URL to be internal (False) or
        require it to be external (True). External URLs include the
        scheme and domain. When not in an active request, URLs are
        external by default.
    :param values: Values to use for the variable parts of the URL rule.
        Unknown keys are appended as query string arguments, like
        ``?a=b&c=d``.

    .. versionchanged:: 2.2
        Calls ``current_app.url_for``, allowing an app to override the
        behavior.

    .. versionchanged:: 0.10
       The ``_scheme`` parameter was added.

    .. versionchanged:: 0.9
       The ``_anchor`` and ``_method`` parameters were added.

    .. versionchanged:: 0.9
       Calls ``app.handle_url_build_error`` on build errors.
    """
    return current_app.url_for(
        endpoint,
        _anchor=_anchor,
        _method=_method,
        _scheme=_scheme,
        _external=_external,
        **values,
    )


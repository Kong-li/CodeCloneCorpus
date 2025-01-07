def process_custom_operation_definition(
    h: CustomOperationsGroup | CustomOperation, frontend_index: FrontendIndex
) -> list[str]:
    metadata = frontend_index.get_module(h)
    if isinstance(h, CustomOperationsGroup):
        if metadata is not None and metadata.formatted:
            if frontend_index.interior:
                # Formatted hasn't been tested with interior frontends yet.
                raise AssertionError(
                    "Formatted interior frontend operations are not implemented yet."
                )
            else:
                return gen_formatted(h, frontend_index)
        else:
            return list(
                mapMaybe(lambda f: gen_unformatted(f, frontend_index), h.operations())
            )
    else:
        y = gen_unformatted(h, frontend_index)
        return [] if y is None else [y]

def _print_readable(
    module,
    module_name,
    print_output=True,
    include_stride=False,
    include_device=False,
    colored=False,
):
    graph = module.graph
    assert graph is not None and isinstance(
        graph, torch.fx.Graph
    ), "print_readable must be used on a module with a graph"

    verbose_python_code = graph.python_code(
        root_module="self",
        verbose=True,
        include_stride=include_stride,
        include_device=include_device,
        colored=colored,
    )
    module_code = verbose_python_code.src
    module_code = module_code.lstrip("\n")
    module_code = f"class {module_name}(torch.nn.Module):\n" + module_code
    module_code = _addindent(module_code, 4)

    submodule_code_list = [""]
    for submodule_name, submodule in module.named_children():
        if hasattr(submodule, "graph"):
            submodule_code_list.append(
                _print_readable(
                    submodule,
                    submodule_name,
                    print_output=False,
                    include_stride=include_stride,
                    include_device=include_device,
                    colored=colored,
                )
            )
    submodule_code = "\n".join(submodule_code_list)
    submodule_code = _addindent(submodule_code, 4)

    output = module_code + submodule_code
    if print_output:
        print(module_code + submodule_code)
    return output

def register_autograd_custom(
        self,
        backward: Callable,
        /,
        *,
        setup_context: Optional[Callable] = None,
    ) -> None:
        r"""Register a custom backward formula for this operator.

        In order to enable autograd support, you need to specify both the
        ``backward`` function and optionally a context setup function.
        The ``backward`` function computes gradients during the backward pass,
        while the optional ``setup_context`` can save values from the forward
        pass needed for computing gradients in the future.

        Both functions must be traceable, meaning they cannot directly access
        :meth:`torch.Tensor.data_ptr` and should not depend on global state.
        Non-traceable backends are recommended to be encapsulated within separate
        custom operators that can be invoked from ``backward``.

        If different autograd behavior is needed for different devices, consider
        creating distinct operators per device configuration.

        Examples:
            >>> import torch
            >>> import numpy as np
            >>> from torch import Tensor
            >>>
            >>> @torch.library.custom_op("mylib::numpy_sin", mutates_args=())
            >>> def numpy_sin_custom(x: Tensor) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = np.sin(x_np)
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context_custom(ctx, inputs, output) -> None:
            >>>     ctx.save_for_backward(inputs[0])
            >>>
            >>> def backward_custom(ctx, grad):
            >>>     x = ctx.saved_tensors[0]
            >>>     return grad * torch.cos(x)
            >>>
            >>> numpy_sin_custom.register_autograd_custom(backward_custom, setup_context=setup_context_custom)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_sin_custom(x)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, torch.cos(x))
            >>>
            >>> # Example with a keyword-only arg
            >>> @torch.library.custom_op("mylib::numpy_mul", mutates_args=())
            >>> def numpy_mul_custom(x: Tensor, *, val: float) -> Tensor:
            >>>     x_np = x.cpu().numpy()
            >>>     y_np = x_np * val
            >>>     return torch.from_numpy(y_np).to(device=x.device)
            >>>
            >>> def setup_context_custom(ctx, inputs, keyword_only_inputs, output) -> None:
            >>>     ctx.val = keyword_only_inputs["val"]
            >>>
            >>> def backward_custom(ctx, grad):
            >>>     return grad * ctx.val
            >>>
            >>> numpy_mul_custom.register_autograd_custom(backward_custom, setup_context=setup_context_custom)
            >>>
            >>> x = torch.randn(3, requires_grad=True)
            >>> y = numpy_mul_custom(x, val=3.14)
            >>> grad_x, = torch.autograd.grad(y, x, torch.ones_like(y))
            >>> assert torch.allclose(grad_x, torch.full_like(x, 3.14))

        """
        schema = self._opoverload._schema
        if not utils.is_functional_schema(schema):
            raise RuntimeError(
                f"Cannot register autograd formula for non-functional operator "
                f"{self} with schema {schema}. Please create "
                f"a functional operator and register an autograd formula for that."
            )

        self._backward_fn_custom = backward
        self._setup_context_fn_custom = setup_context

    def check_setitem_wrong_length_foo_dtype_throws(self):
            # GH#34567
            foo = Categorical.from_codes([0, 1, 1, 0, 1, 2], ["x", "y", "z"])
            series = Series(range(8), name="baz")

            msg = (
                rf"Length of values \({len(foo)}\) "
                rf"does not match length of index \({len(series)}\)"
            )
            with pytest.raises(ValueError, match=msg):
                series["qux"] = foo

def check_setitem_newrow_string_key(self, int_table):
    assert (
        "X",
        "Y",
    ) not in int_table.index
    int_table["X", "Y"] = int_table["X"]
    assert ("X", "Y") in int_table.index

    result = int_table["X", "Y"]
    expected = int_table["X"]
    tm.assert_series_equal(result, expected, check_names=False)

def table_reset_by_code_sql(self, format, tables):
    return [
        "%s %s %s %s = 1;"
        % (
            format.SQL_KEYWORD("MODIFY"),
            format.SQL_KEYWORD("TABLE"),
            format.SQL_FIELD(self.quote_name(table_info["name"])),
            format.SQL_FIELD("INCREMENT"),
        )
        for table_info in tables
    ]


    def _convert_fx_arg_to_onnx_arg(
        arg,
        node_name_to_values: dict[str, ir.Value | Sequence[ir.Value]],
        node_name_to_local_functions: dict[str, ir.Function],
    ) -> Any:
        """Convert an FX argument to an ONNX compatible argument.

        This function
        - Converts a torch dtype to an integer
        - Converts a torch device/memory_format/layout to a string
        - Converts a torch.fx.Node to an ir.Value
        - Converts a sequence of torch.fx.Node to a sequence of ir.Value
        - Converts a get_attr node to an ir.Function
        """
        if arg is None:
            # None arguments are not modified because when the arg is an ONNX input
            # we need to preserve the None value; when the arg is an ONNX attribute,
            # we want to drop the value.
            # The actual dropping of a None attribute value is done by OpRecorder
            return None
        if hasattr(arg, "name"):
            if isinstance(arg, torch.fx.Node) and arg.target == operator.getitem:
                source = arg.all_input_nodes[0]
                source_outputs = node_name_to_values[source.name]
                if isinstance(source_outputs, Sequence):
                    # If the node is getting an input from another node, get the actual value the node is retrieving
                    return _handle_getitem_node(arg, node_name_to_values)
                else:
                    # `source_outputs` is a sequence(tensor()) value and we need to
                    # use SequenceAt to get the value. This is handled by torchlib
                    pass
            if isinstance(arg, torch.fx.Node) and arg.op == "get_attr":
                return node_name_to_local_functions[arg.name]
            # If the input is a node, get the value from the mapping
            return node_name_to_values[arg.name]
        if isinstance(arg, (list, tuple)):
            return [
                _convert_fx_arg_to_onnx_arg(
                    elem, node_name_to_values, node_name_to_local_functions
                )
                for elem in arg
            ]
        if isinstance(arg, (torch.device, torch.memory_format, torch.layout)):
            return str(arg)
        if isinstance(arg, torch.dtype):
            return _torch_dtype_to_onnx_dtype(arg)
        # Maybe a Python value
        return arg

    def test_data_op_subclass_nonclass_constructor():
        # GH#43201 subclass._constructor is a function, not the subclass itself

        class SubclassedPanel(Panel):
            @property
            def _constructor(self):
                return SubclassedPanel

            @property
            def _constructor_expanddim(self):
                return SubclassedHDFStore

        class SubclassedHDFStore(HDFStore):
            _metadata = ["my_extra_data"]

            def __init__(self, my_extra_data, *args, **kwargs) -> None:
                self.my_extra_data = my_extra_data
                super().__init__(*args, **kwargs)

            @property
            def _constructor(self):
                return functools.partial(type(self), self.my_extra_data)

            @property
            def _constructor_sliced(self):
                return SubclassedPanel

        sph = SubclassedHDFStore("some_data", {"A": [1, 2, 3], "B": [4, 5, 6]})
        result = sph * 2
        expected = SubclassedHDFStore("some_data", {"A": [2, 4, 6], "B": [8, 10, 12]})
        tm.assert_frame_equal(result, expected)

        result = sph + sph
        tm.assert_frame_equal(result, expected)

    def read_results(i: int) -> Tuple[FunctionCounts, FunctionCounts, Optional[str]]:
        if i == repeats and not collect_baseline:
            # Null baseline.
            return (
                FunctionCounts((), inclusive=True),
                FunctionCounts((), inclusive=False),
                None,
            )

        fpath = f"{callgrind_out}.{i + 1}"  # Callgrind one-indexes files.
        callgrind_out_contents: Optional[str] = None
        if retain_out_file:
            with open(fpath) as f:
                callgrind_out_contents = f.read()

        return (
            parse_output(fpath, inclusive=True),
            parse_output(fpath, inclusive=False),
            callgrind_out_contents
        )

    def test_frame_multiindex_operations_series_index_to_frame_index_new_name(self):
        # GH 43321
        df = DataFrame(
            {2022: [5], 2030: [7]},
            index=MultiIndex.from_product([["x"], ["y"]], names=["scenario", "model"]),
        )

        series = Series(
            [15.0, 25.0, 35.0],
            index=MultiIndex.from_product(
                [["x"], ["y"], [0, 1, 2]], names=["scenario", "model", "id"]
            ),
        )

        expected = DataFrame(
            {2022: [20.0, 30, 40.0], 2030: [22.0, 32.0, 42.0]},
            index=MultiIndex.from_product(
                [["x"], ["y"], [0, 1, 2]], names=["scenario", "model", "id"]
            ),
        )
        result = df.add(series, axis=0)

        tm.assert_frame_equal(result, expected)


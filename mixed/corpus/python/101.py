def process_args(arg, arg_type, arg_signature=None):
    var_name = f"var_{next(self.arg_var_id)}"
    # ignore nvTmaDesc, as host-side TMA descriptors need
    # to be passed to the compiled Triton kernel by value
    if isinstance(arg_type, torch_dtype) and arg_signature != "nvTmaDesc":
        if arg.endswith(".item()"):
            # Need to declare a scalar in this case
            arg = arg[:-7]
            self.codegen_tensor_item(
                arg_type,
                arg,
                var_name,
            )
        else:
            device_ptr_type = self.device_codegen.cpp_device_ptr()
            self.writeline(
                maybe_hipify_code_wrapper(
                    f"{device_ptr_type} {var_name} = reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                )
            )
    elif arg_type in (sympy.Integer, int):
        self.writeline(f"int {var_name} = {cexpr(arg)};")
    elif arg_type in (sympy.Float, float):
        self.writeline(f"float {var_name} = {cexpr(arg)};")
    # For symbolic call arguments, examine the arg signatures from triton meta
    # to explicitly cast to the right type
    # Reason: `auto` can infer unexpected type against kernel input signature.
    elif (
        isinstance(arg_type, type(SymbolicCallArg))
        and arg_signature is not None
        and arg_signature in signature2dtype.keys()
    ):
        self.writeline(
            f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
        )
    else:
        self.writeline(f"auto {var_name} = {cexpr(arg)};")
    new_args.append(f"&{var_name}")

def test_partial_setting_check(self):
        # GH2578, allow ix and friends to partially set

        orig_series = [1, 2, 3]

        series_copy = orig_series.copy()
        series_copy[5] = 5
        expected_result = [1, 2, 3, 5]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy.loc[5] = 5
        expected_result = [1, 2, 3, 5]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy[5] = 5.0
        expected_result = [1, 2, 3, 5.0]
        assert list(series_copy) == expected_result

        series_copy = orig_series.copy()
        series_copy.loc[5] = 5.0
        expected_result = [1, 2, 3, 5.0]
        assert list(series_copy) == expected_result

        # iloc/iat raise
        series_copy = orig_series.copy()

        msg = "iloc cannot enlarge its target object"
        try:
            series_copy.iloc[3] = 5.0
        except IndexError as e:
            assert str(e) == msg

        msg = "index 3 is out of bounds for axis 0 with size 3"
        try:
            series_copy.iat[3] = 5.0
        except IndexError as e:
            assert str(e) == msg


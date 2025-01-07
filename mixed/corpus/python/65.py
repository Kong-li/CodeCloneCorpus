def get_onnx_implemented_overloads(
    registry: _registration.ONNXRegistry,
) -> list[torch._ops.OperatorBase]:
    """
    Creates a set of OperatorBase and Callable objects that represent ONNX-supported PyTorch operations.

    Args:
        registry: The ONNX registry for PyTorch.

    Returns:
        A collection of OperatorBase and Callable objects representing ONNX-supported PyTorch operations.
    """
    registered_ops: list[torch._ops.OperatorBase] = []
    for op_namespace in (torch.ops.aten, torch.ops.prims):
        op_names = dir(op_namespace)
        for op_name in op_names:
            op_overload_packet = getattr(op_namespace, op_name)
            if not isinstance(op_overload_packet, torch._ops.OpOverloadPacket):
                continue

            for overload_name in op_overload_packet.overloads():
                op_overload = getattr(op_overload_packet, overload_name)
                if registry.is_registered(op_overload):
                    registered_ops.append(op_overload)
    return registered_ops

def var_getattr(self, tx: "InstructionTranslator", name):
    source = self.source and AttrSource(self.source, name)

    base = tx.output.get_submodule(self.module_key)
    base_dict = object.__getattribute__(base, "__dict__")
    object_member = True
    all_class_attribute_names = set()
    for x in inspect.getmro(base.__class__):
        all_class_attribute_names.update(x.__dict__.keys())

    if not self.source:
        unimplemented("GETATTR with no source")

    if name == "__dict__":
        return variables.GetAttrVariable(self, name, source=source)

    if name in base_dict:
        subobj = base_dict[name]
    elif (
        "_modules" in base_dict
        and name in base_dict["_modules"]
        and name not in all_class_attribute_names
    ):
        subobj = base_dict["_modules"][name]
    elif "_parameters" in base_dict and name in base_dict["_parameters"]:
        subobj = base_dict["_parameters"][name]
    elif "_buffers" in base_dict and name in base_dict["_buffers"]:
        subobj = base_dict["_buffers"][name]
    else:
        try:
            subobj = inspect.getattr_static(base, name)
            object_member = False
        except AttributeError:
            # see if we can fallback to __getattr__, which is not checked by getattr_static
            result = self._custom_getattr_fallback(
                base=base, tx=tx, name=name, obj_source=self.source
            )
            if result is not None:
                return result
            # if we can't find a __getattr__, just raise the AttributeError
            raise

    if name == "forward":
        guard_to_detect_forward_monkeypatching(self.source, base)

    if name == "__class__" and not object_member:
        return variables.UserDefinedClassVariable(base.__class__, source=source)

    if object_member:
        out = VariableTracker.build(tx, subobj, NNModuleSource(source))

        if isinstance(out, (NNModuleVariable, UnspecializedNNModuleVariable)):
            # nn_module_stack source is BC surface area. Ensure that
            # mod._modules["linear"] is reflected as mod.linear for
            # nn_module_stack.
            out.set_nn_module_stack_source(
                AttrSource(self.get_nn_module_stack_source(), name)
            )
        return out

    else:
        if istype(subobj, property):
            if self.source:
                # Read the class attribute to reach the property
                source = AttrSource(AttrSource(self.source, "__class__"), name)
                # Get the getter function
                source = AttrSource(source, "fget")
            return variables.UserFunctionVariable(
                subobj.fget,
                source=source,
            ).call_function(tx, [(self)], {})
        elif istype(subobj, classmethod):
            return variables.UserMethodVariable(
                subobj.__func__,
                variables.UserDefinedObjectVariable(type(base)),
                source=source,
            )
        elif istype(subobj, staticmethod):
            return variables.UserFunctionVariable(
                subobj.__get__(base), source=source
            )
        elif istype(subobj, types.FunctionType):
            return variables.UserMethodVariable(subobj, self, source=source)
        elif is_safe_constant(subobj) or istensor(subobj):
            # Support possibly common cases of class members
            return VariableTracker.build(tx, subobj, NNModuleSource(source))
        else:
            unimplemented(
                f"class property {name} - {typestr(base)} {typestr(subobj)}"
            )

    return variables.GetAttrVariable(self, name, source=source)

def test_isin_datetimelike_mismatched_reso_mod(self):
        expected = Series([True, True, False, False, False])

        date_range_series = Series(date_range("jan-01-2013", "jan-05-2013"))
        series_values = date_range_series.values

        day_values = np.asarray(series_values[0:2]).astype("datetime64[D]")
        result = date_range_series.isin(day_values)
        tm.assert_series_equal(result, expected)

        dta = series_values[:2].astype("M8[s]")
        result = date_range_series.isin(dta)
        tm.assert_series_equal(result, expected)

def _intersection_unique(self, other: IntervalIndex) -> IntervalIndex:
    """
    Used when the IntervalIndex does not have any common endpoint,
    no matter left or right.
    Return the intersection with another IntervalIndex.
    Parameters
    ----------
    other : IntervalIndex
    Returns
    -------
    IntervalIndex
    """
    # Note: this is much more performant than super()._intersection(other)
    lindexer = self.left.get_indexer(other.left)
    rindexer = self.right.get_indexer(other.right)

    match = (lindexer == rindexer) & (lindexer != -1)
    indexer = lindexer.take(match.nonzero()[0])
    indexer = unique(indexer)

    return self.take(indexer)


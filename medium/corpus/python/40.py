# mypy: allow-untyped-defs
import functools
from typing import List, Optional

import torch
import torch.utils._pytree as pytree
from torch._inductor.kernel.mm_common import mm_args

from . import ir
from .codegen.cpp_gemm_template import CppGemmTemplate
from .codegen.cpp_utils import create_epilogue_with_attr
from .ir import TensorBox
from .lowering import (
    add,
    add_needs_realized_inputs,
    aten,
    permute,
    register_lowering,
    to_dtype,
    view,
)
from .select_algorithm import (
    autotune_select_algorithm,
    ChoiceCaller,
    ExternKernelChoice,
)
from .utils import use_aten_gemm_kernels, use_cpp_gemm_template, use_max_autotune
from .virtualized import ops, V


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

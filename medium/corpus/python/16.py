# mypy: ignore-errors

import collections
import contextlib
import dataclasses
import enum
import functools
import inspect
import itertools
import random
import sys
import threading
import types
import warnings
import weakref
from typing import Dict, Generic, List, TYPE_CHECKING
from typing_extensions import is_typeddict

import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass_type

from .. import polyfills, variables
from ..bytecode_transformation import create_call_function
from ..create_parameter_op import do_not_convert_to_tracable_parameter
from ..exc import (
    handle_observed_exception,
    ObservedAttributeError,
    raise_observed_exception,
    unimplemented,
)
from ..guards import GuardBuilder, install_guard
from ..source import (
    AttrSource,
    GetItemSource,
    RandomValueSource,
    UnspecializedParamBufferSource,
)
from ..utils import (
    build_checkpoint_variable,
    build_invoke_subgraph_variable,
    check_constant_args,
    dict_methods,
    get_custom_getattr,
    has_torch_function,
    is_frozen_dataclass,
    is_invoke_subgraph,
    is_namedtuple_cls,
    is_utils_checkpoint,
    is_wrapper_or_member_descriptor,
    istype,
    namedtuple_fields,
    object_has_getattribute,
    proxy_args_kwargs,
    tensortype_to_dtype,
    unpatched_nn_module_getattr,
)
from .base import AttributeMutationExisting, ValueMutationNew, VariableTracker
from .dicts import DefaultDictVariable


try:
    import numpy as np
except ModuleNotFoundError:
    np = None

try:
    from torch.utils._cxx_pytree import PyTreeSpec
except ImportError:
    PyTreeSpec = type(None)


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslator


def test_logical_ops_with_index(self, op):
    # GH#22092, GH#19792
    ser = Series([True, True, False, False])
    idx1 = Index([True, False, True, False])
    idx2 = Index([1, 0, 1, 0])

    expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

    result = op(ser, idx1)
    tm.assert_series_equal(result, expected)

    expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

    result = op(ser, idx2)
    tm.assert_series_equal(result, expected)


def test_pivot_timegrouper_single(self):
        # single grouper
        data = DataFrame(
            {
                "Department": "Sales Sales Sales HR HR HR".split(),
                "Employee": "John Michael John Michael Lisa Mike".split(),
                "Salary": [50000, 60000, 70000, 80000, 90000, 100000],
                "Date": [
                    datetime(2023, 1, 1, 14, 0),
                    datetime(2023, 2, 15, 15, 5),
                    datetime(2023, 3, 1, 21, 0),
                    datetime(2023, 4, 2, 10, 0),
                    datetime(2023, 5, 1, 20, 0),
                    datetime(2023, 6, 2, 10, 0),
                ],
                "PromotionDate": [
                    datetime(2023, 2, 4, 0, 0),
                    datetime(2023, 3, 15, 13, 5),
                    datetime(2023, 1, 5, 20, 0),
                    datetime(2023, 4, 8, 10, 0),
                    datetime(2023, 5, 7, 20, 0),
                    datetime(2023, 6, 30, 12, 0),
                ],
            }
        )

        outcome = pivot_table(
            data,
            index=Grouper(freq="MS", key="Date"),
            columns=Grouper(freq="MS", key="PromotionDate"),
            values="Salary",
            aggfunc="sum",
        )
        expected = DataFrame(
            np.array(
                [
                    [np.nan, 50000],
                    [60000, np.nan],
                    [70000, 80000],
                    [90000, np.nan],
                    [100000, np.nan]
                ]
            ).reshape(5, 2),
            index=pd.DatetimeIndex(
                [
                    datetime(2023, 1, 31),
                    datetime(2023, 2, 28),
                    datetime(2023, 3, 31),
                    datetime(2023, 4, 30),
                    datetime(2023, 5, 31)
                ],
                freq="MS"
            ),
            columns=pd.DatetimeIndex(
                [
                    datetime(2023, 1, 31),
                    datetime(2023, 2, 28)
                ],
                freq="MS"
            )
        )
        expected.index.name = "Date"
        expected.columns.name = "PromotionDate"

        result = pivot_table(
            data,
            index=[Grouper(freq="MS", key="Date"), Grouper(freq="MS", key="PromotionDate")],
            columns=["Department"],
            values="Salary",
            aggfunc="sum"
        )
        tm.assert_frame_equal(result, expected)

        result = pivot_table(
            data,
            index=["Department"],
            columns=[Grouper(freq="MS", key="Date"), Grouper(freq="MS", key="PromotionDate")],
            values="Salary",
            aggfunc="sum"
        )
        tm.assert_frame_equal(result, expected.T)


class UserDefinedVariable(VariableTracker):
    pass


class UserDefinedClassVariable(UserDefinedVariable):
    def __init__(self, value, **kwargs) -> None:
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def as_proxy(self):
        return self.value

    def __repr__(self) -> str:
        return f"UserDefinedClassVariable({self.value})"

    @staticmethod
    @functools.lru_cache(None)
    def _constant_fold_classes():
        return {
            torch.device,
            torch.finfo,
            torch.iinfo,
            torch.Size,
        }

    @staticmethod
    @functools.lru_cache(None)
    def _in_graph_classes():
        _in_graph_class_list = {
            torch.Tensor,
            torch.cuda.Stream,
            torch.cuda.Event,
        }
        if hasattr(torch, "hpu"):
            _in_graph_class_list.update(
                {
                    torch.hpu.Stream,
                    torch.hpu.Event,
                }
            )

        return set(tensortype_to_dtype.keys()) | _in_graph_class_list

    def can_constant_fold_through(self):
        return self.value in self._constant_fold_classes()

    def has_key_in_generic_dict(self, tx: "InstructionTranslator", key):
        if tx.output.side_effects.has_pending_mutation_of_attr(self, key):
            mutated_attr = tx.output.side_effects.load_attr(self, key, deleted_ok=True)
            return not isinstance(mutated_attr, variables.DeletedVariable)

        return key in self.value.__dict__

    def var_getattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        from . import ConstantVariable, EnumVariable

        source = AttrSource(self.source, name) if self.source is not None else None

        if name == "__name__":
            return ConstantVariable.create(self.value.__name__)
        elif name == "__qualname__":
            return ConstantVariable.create(self.value.__qualname__)
        elif name == "__dict__":
            options = {"source": source}
            return variables.GetAttrVariable(self, name, **options)

        # Special handling of collections.OrderedDict.fromkeys()
        # Wrap it as GetAttrVariable(collections.OrderedDict, "fromkeys") to make it consistent with
        # collections.defaultdict, and both will be handled at UserDefinedClassVariable.call_method().
        # Otherwise, it would be wrapped as UserDefinedObjectVariable(collections.OrderedDict.fromkeys),
        # and we need duplicate code to handle both cases.
        if (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            return super().var_getattr(tx, name)

        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            obj = None

        if isinstance(obj, staticmethod):
            return VariableTracker.build(tx, obj.__get__(self.value), source)
        elif isinstance(obj, classmethod):
            if isinstance(obj.__func__, property):
                return variables.UserFunctionVariable(obj.__func__.fget).call_function(
                    tx, [self], {}
                )
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif isinstance(obj, types.ClassMethodDescriptorType):
            # e.g.: inspect.getattr_static(dict, "fromkeys")
            #       inspect.getattr_static(itertools.chain, "from_iterable")
            func = obj.__get__(None, self.value)
            return VariableTracker.build(tx, func, source)
        elif source:
            # __mro__ is a member in < 3.12, an attribute in >= 3.12
            if inspect.ismemberdescriptor(obj) or (
                sys.version_info >= (3, 12) and name == "__mro__"
            ):
                return VariableTracker.build(tx, obj.__get__(self.value), source)

        if ConstantVariable.is_literal(obj):
            return ConstantVariable.create(obj)
        elif isinstance(obj, enum.Enum):
            return EnumVariable(obj)
        elif name in getattr(self.value, "__dict__", {}) or (
            self.value.__module__.startswith("torch.")
            or self.value.__module__ == "torch"
        ):
            if source:
                return VariableTracker.build(tx, obj, source)

        if (
            source
            and not inspect.ismethoddescriptor(obj)
            and not is_wrapper_or_member_descriptor(obj)
        ):
            return VariableTracker.build(tx, obj, source)

        return super().var_getattr(tx, name)

    def _call_cross_entropy_loss(self, tx: "InstructionTranslator", args, kwargs):
        """
        functional: input, target, weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional ctor: weight=None, size_average=None, ignore_index=- 100, reduce=None, reduction='mean',
        label_smoothing=0.0

        non functional loss call: input, target, optional_output
        """
        from . import ConstantVariable

        def normalize_args(
            weight=ConstantVariable.create(None),
            size_average=ConstantVariable.create(None),
            ignore_index=ConstantVariable.create(-100),
            reduce=ConstantVariable.create(None),
            reduction=ConstantVariable.create("mean"),
            label_smoothing=ConstantVariable.create(0.0),
        ):
            return (
                weight,
                size_average,
                ignore_index,
                reduce,
                reduction,
                label_smoothing,
            )

        (
            weight,
            size_average,
            ignore_index,
            reduce_arg,
            reduction,
            label_smoothing,
        ) = normalize_args(*args, **kwargs)

        def fake_cross_entropy_loss(input, target):
            from .builder import wrap_fx_proxy

            return wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    torch.nn.functional.cross_entropy,
                    *proxy_args_kwargs(
                        [
                            input,
                            target,
                            weight,
                            size_average,
                            ignore_index,
                            reduce_arg,
                            reduction,
                            label_smoothing,
                        ],
                        {},
                    ),
                ),
            )

        return variables.LambdaVariable(fake_cross_entropy_loss)

    def call_method(
        self,
        tx,
        name,
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        if (
            name == "__subclasses__"
            and len(args) == 0
            and not kwargs
            and "__subclasses__" not in self.value.__dict__
        ):
            options = {"mutation_type": ValueMutationNew()}
            subs_as_vars: List[VariableTracker] = []
            for sub in self.value.__subclasses__():
                source = AttrSource(tx.import_source(sub.__module__), sub.__name__)
                subs_as_vars.append(
                    variables.UserDefinedClassVariable(sub, source=source)
                )

            return variables.ListVariable(subs_as_vars, **options)
        elif (
            self.value in {collections.OrderedDict, collections.defaultdict}
            and name == "fromkeys"
        ):
            from .builtin import BuiltinVariable

            return BuiltinVariable.call_custom_dict_fromkeys(
                tx, self.value, *args, **kwargs
            )
        elif name == "__eq__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value == args[0].value)
        elif name == "__ne__" and len(args) == 1 and hasattr(args[0], "value"):
            return variables.ConstantVariable(self.value != args[0].value)

        return super().call_method(tx, name, args, kwargs)

    def call_function(
        self,
        tx: "InstructionTranslator",
        args: "List[VariableTracker]",
        kwargs: "Dict[str, VariableTracker]",
    ) -> "VariableTracker":
        from ..side_effects import SideEffects
        from .builder import wrap_fx_proxy
        from .builtin import BuiltinVariable

        constant_args = check_constant_args(args, kwargs)

        if self.can_constant_fold_through() and constant_args:
            # constant fold
            return variables.ConstantVariable.create(
                self.as_python_constant()(
                    *[x.as_python_constant() for x in args],
                    **{k: v.as_python_constant() for k, v in kwargs.items()},
                ),
            )
        elif self.value is torch.nn.CrossEntropyLoss:
            return self._call_cross_entropy_loss(tx, args, kwargs)
        elif self.value is contextlib.nullcontext:
            # import here to avoid circular dependency
            from .ctx_manager import NullContextVariable

            return NullContextVariable()
        elif self.value is collections.OrderedDict:
            return BuiltinVariable.call_custom_dict(
                tx, collections.OrderedDict, *args, **kwargs
            )
        elif (
            self.value is collections.defaultdict
            and len(args) <= 1
            and DefaultDictVariable.is_supported_arg(args[0])
        ):
            return DefaultDictVariable(
                {},
                collections.defaultdict,
                args[0],
                mutation_type=ValueMutationNew(),
            )
        elif is_typeddict(self.value):
            if self.value.__optional_keys__:
                unimplemented("TypedDict with optional keys not supported")
            return variables.BuiltinVariable(dict).call_dict(tx, *args, **kwargs)
        elif self.value is collections.deque:
            maxlen = variables.ConstantVariable.create(None)
            if not kwargs:
                if len(args) == 0:
                    items = []
                elif len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                elif len(args) == 2 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                    maxlen = args[1]
                else:
                    unimplemented("deque() with more than 2 arg not supported")
            elif tuple(kwargs) == ("maxlen",):
                maxlen = kwargs["maxlen"]
                if len(args) == 0:
                    items = []
                if len(args) == 1 and args[0].has_force_unpack_var_sequence(tx):
                    items = args[0].force_unpack_var_sequence(tx)
                else:
                    unimplemented("deque() with more than 1 arg not supported")
            else:
                unimplemented("deque() with invalid kwargs not supported")
            return variables.lists.DequeVariable(
                items, maxlen=maxlen, mutation_type=ValueMutationNew()
            )
        elif self.value is weakref.ref:
            return variables.WeakRefVariable(args[0])
        elif self.value is functools.partial:
            if not args:
                unimplemented("functools.partial malformed")
            # The first arg, a callable (the ctor below will assert on types)
            fn = args[0]
            rest_args = args[1:]
            # guards for the produced FunctoolsPartialVariable are installed in FunctoolsPartialVariable ctor from the
            # args and keywords
            return variables.functions.FunctoolsPartialVariable(
                fn, args=rest_args, keywords=kwargs
            )
        elif self.value is warnings.catch_warnings and not args:
            return variables.CatchWarningsCtxManagerVariable.create(tx, kwargs)
        elif self.value is torch.cuda.device and not kwargs and len(args) == 1:
            assert args[0].is_python_constant()
            return variables.CUDADeviceVariable.create(tx, args[0].as_python_constant())
        elif (
            issubclass(type(self.value), type)
            and hasattr(
                self.value, "__enter__"
            )  # TODO(voz): These can invoke user code!
            and hasattr(
                self.value, "__exit__"
            )  # TODO(voz): These can invoke user code!
            and self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
            and not is_forbidden_context_manager(self.value)
        ):
            from torch.overrides import TorchFunctionMode

            from .ctx_manager import GenericContextWrappingVariable
            from .functions import (
                BaseUserFunctionVariable,
                FunctionDecoratedByContextlibContextManagerVariable,
            )
            from .torch_function import TorchFunctionModeVariable

            if issubclass(
                self.value, TorchFunctionMode
            ) and TorchFunctionModeVariable.is_supported_torch_function_mode(
                self.value
            ):
                var_cls = TorchFunctionModeVariable
            else:
                var_cls = GenericContextWrappingVariable

            # graph break on any contextlib.* that it is not contextlib.contextmanager
            # Some of the APIs below are not supported because they rely on features
            # that Dynamo doesn't play well today (i.e. contextlib.suppress)
            if self.value in (
                contextlib._AsyncGeneratorContextManager,
                contextlib.closing,
                contextlib.redirect_stdout,
                contextlib.redirect_stderr,
                contextlib.suppress,
                contextlib.ExitStack,
                contextlib.AsyncExitStack,
            ):
                # We are not changing the behavior of Dynamo as these function were
                # already ignored on trace_rules.py before #136033 landed
                unimplemented(
                    f"{self.value} not supported. This may be due to its use of "
                    "context-specific operations that are not supported in "
                    "Dynamo yet (i.e. Exception handling)"
                )

            if self.value is contextlib._GeneratorContextManager and isinstance(
                args[0], BaseUserFunctionVariable
            ):
                if not torch._dynamo.config.enable_trace_contextlib:
                    unimplemented("contextlib.contextmanager")
                # Replace UserFunctionVariable by FunctionDecoratedBycontextlibContextManagerVariable
                # if the function is annotated with @contextlib.contextmanager
                # This shouldn't be necessary once generator functions are fully
                # supported in dynamo
                args = [
                    FunctionDecoratedByContextlibContextManagerVariable(
                        args[0], source=args[0].source
                    )
                ] + args[1:]

            cm_obj = tx.output.side_effects.track_object_new(
                self.source, self.value, var_cls, {}
            )
            cm_obj.call_method(tx, "__init__", args, kwargs)
            return cm_obj
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            # check if this a quasi-namedtuple or a real one
            if self.value.__module__ == "torch.return_types":
                assert len(args) == 1
                assert not kwargs
                items = args[0].force_unpack_var_sequence(tx)
            else:
                field_defaults = self.value._field_defaults

                items = list(args)
                items.extend([None] * (len(fields) - len(items)))

                var_tracker_kwargs = {}
                for field_name, var_tracker in zip(fields, items):
                    if var_tracker is None:
                        if field_name in kwargs:
                            field_var = kwargs[field_name]
                        else:
                            assert field_name in field_defaults
                            field_var = VariableTracker.build(
                                tx, field_defaults[field_name]
                            )
                        var_tracker_kwargs[field_name] = field_var

                for name, value in var_tracker_kwargs.items():
                    assert name in fields
                    items[fields.index(name)] = value

                assert all(x is not None for x in items)

            return variables.NamedTupleVariable(items, self.value)
        elif is_frozen_dataclass(self.value) and self.is_standard_new():
            fields = dataclasses.fields(self.value)
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))

            default_kwargs = {}
            for field, var_tracker in zip(fields, items):
                if var_tracker is None:
                    if field.name in kwargs:
                        var_tracker = kwargs[field.name]
                    else:
                        if not field.init:
                            continue

                        if field.default is not dataclasses.MISSING:
                            var_tracker = VariableTracker.build(tx, field.default)
                        elif field.default_factory is not dataclasses.MISSING:
                            factory_fn = VariableTracker.build(
                                tx, field.default_factory
                            )
                            var_tracker = factory_fn.call_function(tx, [], {})
                        else:
                            # if we are subclass, the constructor could possibly
                            # be missing args
                            continue

                    default_kwargs[field.name] = var_tracker
            kwargs.update(default_kwargs)

            var = tx.output.side_effects.track_object_new_from_user_defined_class(self)
            var.call_method(tx, "__init__", args, kwargs)
            return var
        elif (
            self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            var = tx.output.side_effects.track_object_new_from_user_defined_class(self)
            with do_not_convert_to_tracable_parameter():
                var.call_method(tx, "__init__", args, kwargs)
                return var
        elif (
            variables.RestrictedListSubclassVariable.is_matching_cls(self.value)
            and self.source
        ):
            return variables.RestrictedListSubclassVariable(
                variables.BuiltinVariable(list).call_function(tx, args, kwargs).items,
                user_cls=self.value,
                user_cls_source=self.source,
                mutation_type=ValueMutationNew(),
            )
        elif (
            self.value in self._in_graph_classes()
            or is_traceable_wrapper_subclass_type(self.value)
        ):
            # torch.LongTensor cannot accept a list of FakeTensors.
            # So we stack the list of FakeTensors instead.
            if (
                np
                and self.value in tensortype_to_dtype
                and len(args) == 1
                and isinstance(args[0], variables.ListVariable)
                and len(args[0].items) > 1
                and all(isinstance(x, variables.TensorVariable) for x in args[0].items)
            ):
                # Stack FakeTensor
                stacked = wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        torch.stack,
                        *proxy_args_kwargs(args, kwargs),
                    ),
                )
                args = [stacked]

            tensor_variable = wrap_fx_proxy(
                tx=tx,
                proxy=tx.output.create_proxy(
                    "call_function",
                    self.value,
                    *proxy_args_kwargs(args, kwargs),
                ),
            )

            return tensor_variable
        elif issubclass(self.value, enum.Enum) and len(args) == 1 and not kwargs:
            options = {"mutation_type": ValueMutationNew()}
            return variables.EnumVariable.create(self.value, args[0], options)
        elif self.value is random.Random:
            if len(args) == 1 and isinstance(args[0], variables.ConstantVariable):
                seed = args[0].value
            else:
                seed = None
            random_object = random.Random(seed)
            return RandomVariable(random_object)
        elif (
            not self.is_standard_new()
            and SideEffects.cls_supports_mutation_side_effects(self.value)
            and self.source
        ):
            return tx.inline_user_function_return(
                VariableTracker.build(
                    tx, polyfills.instantiate_user_defined_class_object
                ),
                [self, *args],
                kwargs,
            )

        return super().call_function(tx, args, kwargs)

    def is_standard_new(self):
        """Check for __new__ being overridden"""
        new_fn = inspect.getattr_static(self.value, "__new__", None)
        if isinstance(new_fn, staticmethod):
            new_fn = new_fn.__func__
        return new_fn in (object.__new__, Generic.__new__, dict.__new__)

    def call_hasattr(self, tx: "InstructionTranslator", name: str) -> "VariableTracker":
        if self.source:
            source = AttrSource(self.source, name)
            install_guard(source.make_guard(GuardBuilder.HASATTR))
            return variables.ConstantVariable(hasattr(self.value, name))
        return super().call_hasattr(tx, name)

    def const_getattr(self, tx: "InstructionTranslator", name):
        if name == "__name__":
            return self.value.__name__
        return super().const_getattr(tx, name)


class NO_SUCH_SUBOBJ:
    pass


def initialize_to_zero_params(self, *args, **kwargs):
    if not self.initialize_to_zero_param_names:
        return
    for i, arg in enumerate(args):
        if self.fn.param_names[i] in self.initialize_to_zero_param_names:
            assert isinstance(
                arg,
                torch.Tensor,
            ), "self.initialize_to_zero_param_names should only contain valid argument names"
            arg.zero_()

    for name, arg in kwargs.items():
        if name in self.initialize_to_zero_param_names:
            assert isinstance(
                arg,
                torch.Tensor,
            ), "self.initialize_to_zero_param_names should only contain valid argument names"
            arg.zero_()


class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """

    _nonvar_fields = {"value", "value_type", *UserDefinedVariable._nonvar_fields}

    def _concatenate_distinct(z):
        """Concatenate distinct values of z and return the result.

        The result is a view of z, and the metadata (distinct) is not attached to z.
        """
        if not isinstance(z, pd.Series):
            return z
        try:
            # avoid recalculating distinct in nested calls.
            if "distinct" in z.dtype.metadata:
                return z
        except (AttributeError, TypeError):
            pass

        distinct = z.unique()
        distinct_dtype = np.dtype(z.dtype, metadata={"distinct": distinct})
        return z.view(dtype=distinct_dtype)

    def _minute_finder(label_interval: int) -> None:
        target = dates_.minute
        hour_start = _period_break(dates_, "hour")
        mask = _period_break_mask(dates_, "minute")
        info_maj[hour_start] = True
        info_min[mask & (target % label_interval == 0)] = True
        info_fmt[mask & (target % label_interval == 0)] = "%H:%M"
        info_fmt[day_start] = "%H:%M\n%d-%b"
        info_fmt[year_start] = "%H:%M\n%d-%b\n%Y"

    def test_unique_constraint_with_included_fields(self):
            warning = "CheckConstraint with included fields cannot be deferred."
            with self.assertWarnsMessage(UserWarning, warning):
                models.CheckConstraint(
                    check=Q(name__contains="test"),
                    name="name_inc_color_color_check",
                    include=["color"],
                    deferrable=models.Deferrability.LATE,
                )

    def validate_multiple_output_binary_operations(ufunc, sparse_mode, mixed_arrays):
        a1, a2 = mixed_arrays
        if not sparse_mode:
            a1[a1 == 0] = 1
            a2[a2 == 0] = 1

        a1 = SparseArray(a1, dtype=pd.SparseDtype("int64", 0)) if sparse_mode else a1
        a2 = SparseArray(a2, dtype=pd.SparseDtype("int64", 0)) if sparse_mode else a2

        s1 = pd.Series(a1)
        s2 = pd.Series(a2)

        if not shuffle_mode := sparse_mode:
            s2 = s2.sample(frac=1)

        expected_values = ufunc(a1, a2)
        assert isinstance(expected_values, tuple), "Expected outputs as a tuple"

        result_values = ufunc(s1, s2)
        assert isinstance(result_values, tuple), "Result must be a tuple of values"
        tm.assert_series_equal(result_values[0], pd.Series(expected_values[0]))
        tm.assert_series_equal(result_values[1], pd.Series(expected_values[1]))

    def check_empty(self):
            StringModel.objects.create(long="100")
            obj = StringModel.objects.annotate(
                empty_len_short=Length("short", "default"),
                empty_len_default=Length("default", "long"),
            ).first()
            self.assertIsNone(obj.empty_len_short)
            self.assertIsNone(obj.empty_len_default)

    def verify_crt_factory_returns_same_instance(
        self,
        mock.crt_lock,
        mock.crt_singleton_client,
        mock.serializer_instance,
    ):
        first_s3_resource = boto3.crt.get_customized_s3_resource(EUWEST2_S3_RESOURCE, None)
        second_s3_resource = boto3.crt.get_customized_s3_resource(EUWEST2_S3_RESOURCE, None)

        assert isinstance(first_s3_resource, boto3.crt.CustomS3Resource)
        assert first_s3_resource is second_s3_resource
        assert first_s3_resource.crt_client is second_s3_resource.crt_client

    def handle_addition_to_index_elements(data_series):
        # GH 47819
        prefixed_data = data_series.map(lambda x: f"foo#{x}")
        expected_prefix = Index([f"foo#{c}" for c in data_series])
        tm.assert_index_equal(prefixed_data, expected_prefix)

        suffixed_data = data_series.map(lambda x: f"{x}#foo")
        expected_suffix = Index([f"{c}#foo" for c in data_series])
        tm.assert_index_equal(suffixed_data, expected_suffix)

    def fetch_test_collection(filter_opts: list[str] | None) -> TestList:
        test_list: TestList = []

        cpp_tests = get_test_list_by_type(filter_opts, TestType.CPP)
        py_tests = get_test_list_by_type(get_python_filter_opts(filter_opts), TestType.PY)

        if not (cpp_tests or py_tests):
            raise_no_test_found_exception(
                get_oss_binary_folder(TestType.CPP),
                get_oss_binary_folder(TestType.PY)
            )

        test_list.extend(cpp_tests)
        test_list.extend(py_tests)

        return test_list

    def get_python_filter_opts(orig_opts: list[str] | None) -> list[str]:
        if orig_opts is None:
            return []
        else:
            return [opt for opt in orig_opts if opt.endswith('.py')]

    @staticmethod
    @functools.lru_cache(None)
    def test_aggregate_with_nat(func, fill_value):
        # check TimeGrouper's aggregation is identical as normal groupby
        # if NaT is included, 'var', 'std', 'mean', 'first','last'
        # and 'nth' doesn't work yet

        n = 20
        data = np.random.default_rng(2).standard_normal((n, 4)).astype("int64")
        normal_df = DataFrame(data, columns=["A", "B", "C", "D"])
        normal_df["key"] = [1, 2, np.nan, 4, 5] * 4

        dt_df = DataFrame(data, columns=["A", "B", "C", "D"])
        dt_df["key"] = Index(
            [
                datetime(2013, 1, 1),
                datetime(2013, 1, 2),
                pd.NaT,
                datetime(2013, 1, 4),
                datetime(2013, 1, 5),
            ]
            * 4,
            dtype="M8[ns]",
        )

        normal_grouped = normal_df.groupby("key")
        dt_grouped = dt_df.groupby(Grouper(key="key", freq="D"))

        normal_result = getattr(normal_grouped, func)()
        dt_result = getattr(dt_grouped, func)()

        pad = DataFrame([[fill_value] * 4], index=[3], columns=["A", "B", "C", "D"])
        expected = pd.concat([normal_result, pad])
        expected = expected.sort_index()
        dti = date_range(
            start="2013-01-01",
            freq="D",
            periods=5,
            name="key",
            unit=dt_df["key"]._values.unit,
        )
        expected.index = dti._with_freq(None)  # TODO: is this desired?
        tm.assert_frame_equal(expected, dt_result)
        assert dt_result.index.name == "key"

    def efficient_path(input_arr, validities, indices, fill_value):
        all_valid = True
        for validity in validities:
            all_valid &= validity

        gathered_values = tf.gather_nd(input_arr, indices)
        transposed_gathered_values = tf.transpose(gathered_values)

        condition = tf.reshape(all_valid, (1,))
        return tf.where(condition, transposed_gathered_values, fill_value)

    def safe_mul(a, b):
        # Make unknown() * wrap(0.0) == wrap(0.0)
        if a == 0.0 or a == 0:
            return a
        elif b == 0.0 or b == 0:
            return b
        else:
            return a * b

    def _value_eq(self, other: object) -> bool:
        if isinstance(other, (SymNode, _DeconstructedSymNode)):
            return (
                self._expr == other._expr
                and self.pytype == other.pytype
                and self._hint == other._hint
                and self.constant == other.constant
                and self.fx_node == other.fx_node
            )
        else:
            return False

    def test_dispatch_transform(tsframe):
        df = tsframe[::5].reindex(tsframe.index)

        grouped = df.groupby(lambda x: x.month)

        filled = grouped.ffill()
        fillit = lambda x: x.ffill()
        expected = df.groupby(lambda x: x.month).transform(fillit)
        tm.assert_frame_equal(filled, expected)

    def not_equal(x1, x2):
        x1 = convert_to_tensor(x1)
        x2 = convert_to_tensor(x2)
        dtype = dtypes.result_type(x1.dtype, x2.dtype)
        x1 = tf.cast(x1, dtype)
        x2 = tf.cast(x2, dtype)
        return tf.not_equal(x1, x2)

    def explore(proc, element, forward=True):
        # From https://github.com/google/jax/pull/19695
        def explore_children():
            kids, typedef = optree.tree_flatten(
                element,
                is_leaf=lambda x: x is not element,
                none_is_leaf=True,
                namespace="tensorflow",
            )
            if typedef.num_nodes == 1 and typedef.num_leaves == 1:
                return element
            else:
                return optree.tree_unflatten(
                    typedef,
                    [explore(proc, kid, forward=forward) for kid in kids],
                )

        if forward:
            res = proc(element)
            if res is None:
                return explore_children()
        else:
            explored_element = explore_children()
            res = proc(explored_element)
            if res is None:
                return explored_element
        # Detect MAP_TO_NONE without tree_api import to avoid circular import.
        if isinstance(res, type) and res.__name__ == "MAP_TO_NONE":
            return None
        return res

    def example_check_():
        si = SingleIndex.from_tuples(zip(range(10), range(10)))
        assert si.is_(si)
        assert si.is_(si.view())
        assert si.is_(si.view().view().view().view())
        si2 = si.view()
        # names are metadata, they don't change id
        si2.names = ["X", "Y"]
        assert si2.is_(si)
        assert si.is_(si2)

        assert not si.is_(si.set_names(["Z", "W"]))
        # levels are inherent properties, they change identity
        si3 = si2.set_levels([list(range(10)), list(range(10))])
        assert not si3.is_(si2)
        # shouldn't change
        assert si2.is_(si)
        si4 = si3.view()

        # GH 17464 - Remove duplicate SingleIndex levels
        si4 = si4.set_levels([list(range(10)), list(range(10))])
        assert not si4.is_(si3)
        si5 = si.view()
        si5 = si5.set_levels(si5.levels)
        assert not si5.is_(si)

    def test_astype_object(self):
        idx = PeriodIndex([], freq="M")

        exp = np.array([], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        idx = PeriodIndex(["2011-01", NaT], freq="M")

        exp = np.array([Period("2011-01", freq="M"), NaT], dtype=object)
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

        exp = np.array([Period("2011-01-01", freq="D"), NaT], dtype=object)
        idx = PeriodIndex(["2011-01-01", NaT], freq="D")
        tm.assert_numpy_array_equal(idx.astype(object).values, exp)
        tm.assert_numpy_array_equal(idx._mpl_repr(), exp)

    def validate_encode_with_custom_behavior():
        from numpy import array
        with pytest.raises(ValueError, match="y contains previously unseen labels"):
            custom_uniques = array([1, 2, 3])
            custom_values = array([1, 2, 3, 4])
            _encode(custom_values, uniques=custom_uniques, check_unknown=True)

        # dont raise error if False
        no_error_uniques = array(["a", "b", "c"], dtype='O')
        no_error_values = array(["a", "b", "c", "d"], dtype='O')
        _encode(no_error_values, uniques=no_error_uniques, check_unknown=False)

        # parameter is ignored for object dtype
        with pytest.raises(ValueError, match="y contains previously unseen labels"):
            custom_uniques_obj = array(["x", "y", "z"], dtype=object)
            custom_values_obj = array(["x", "y", "z", "w"], dtype=object)
            _encode(custom_values_obj, uniques=custom_uniques_obj, check_unknown=False)

        def _encode(values, uniques, check_unknown):
            if not check_unknown and values.dtype != 'O':
                raise ValueError("y contains previously unseen labels")

    def example_regroup_with_hour(item):
        # GH 13020
        index = DatetimeIndex(
            [
                pd.NaT,
                "1970-01-01 00:00:00",
                pd.NaT,
                "1970-01-01 00:00:01",
                "1970-01-01 00:00:02",
            ]
        ).as_item(item)
        frame = DataFrame([2, 3, 5, 7, 11], index=index)

        index_1h = DatetimeIndex(
            ["1970-01-01 00:00:00", "1970-01-01 00:00:01", "1970-01-01 00:00:02"]
        ).as_item(item)
        frame_1h = DataFrame([3.0, 7.0, 11.0], index=index_1h)
        tm.assert_frame_equal(frame.regroup("1h").mean(), frame_1h)

        index_2h = DatetimeIndex(["1970-01-01 00:00:00", "1970-01-01 00:00:02"]).as_item(
            item
        )
        frame_2h = DataFrame([5.0, 11.0], index=index_2h)
        tm.assert_frame_equal(frame.regroup("2h").mean(), frame_2h)

        index_3h = DatetimeIndex(["1970-01-01 00:00:00"]).as_item(item)
        frame_3h = DataFrame([7.0], index=index_3h)
        tm.assert_frame_equal(frame.regroup("3h").mean(), frame_3h)

        tm.assert_frame_equal(frame.regroup("60h").mean(), frame_3h)

    def _gather_all_orig_param_state(
        fsdp_param_info: FSDPParamInfo,
        input_states: Dict[str, Any],
        shard_state: bool,
        to_save: bool,
        cpu_offload: bool,
    ) -> Dict[str, Any]:
        """
        Given a optimizer state dict, ``input_states``, which the keys are FQNs to the
        original parameters (not FlatParameters nor parmeter ID), gather all the
        states and unflatten them to the original dimensions. Note that all the
        params referred by the ``input_states`` must be managed by FSDP.
        """
        fsdp_state = fsdp_param_info.state
        if (
            fsdp_state.world_size == 1
            or fsdp_state.sharding_strategy == ShardingStrategy.NO_SHARD
        ):
            return input_states if to_save else {}

        with SimpleProfiler.profile(SimpleProfiler.Type.RESHARDING):
            with SimpleProfiler.profile(SimpleProfiler.Type.ALLGATHER_OBJ):
                gathered_state_info = _allgather_state_info(fsdp_state, input_states)
            output_states = _allgather_orig_param_states(
                fsdp_param_info,
                gathered_state_info,
                input_states,
                shard_state,
                to_save,
                cpu_offload,
            )
        if to_save:
            for key, idx in fsdp_param_info.param_indices.items():
                if key in output_states:
                    continue
                if not fsdp_param_info.param_requires_grad[idx]:
                    continue

                raise RuntimeError(
                    f"{key} is not in the output state. "
                    "The FSDPParamInfo has the param keys "
                    f"{sorted(fsdp_param_info.param_indices.keys())} while "
                    "the output_states has the param keys "
                    f"{sorted(output_states.keys())}."
                )
            return output_states
        else:
            return {}

    def validate_weekday_category(dataframe):
        df = dataframe(
            {"day": ["Monday", "Tuesday", "Monday", "Wednesday", "Monday", "Thursday", "Friday", "Saturday", "Sunday"]},
            is_categorical=True,
        )

        column_info = df.__dataframe__().get_column_info_by_name("day")
        categorical_data = column_info.describe_categorical_data
        assert isinstance(categorical_data["is_ordered"], bool)
        assert isinstance(categorical_data["is_dictionary"], bool)

    def insert_condition_modifications(self, inputs, model_object, **kwargs):
        """Applies the condition modification transformation to the input parameters

        This function includes transformations for ConditionExpression and KeyExpression.
        It also manages any placeholder names and values that are created during the
        transformation of the condition expressions.
        """
        self._condition_builder.reset()
        named_placeholders = {}
        value_placeholders = {}

        # Apply the transformation for the main Condition Expression.
        cond_transformation = ConditionExpressionTransformation(
            self._condition_builder, placeholder_names=named_placeholders,
            placeholder_values=value_placeholders, is_key_condition=False
        )
        self._transformer.transform(inputs, model_object.input_shape, cond_transformation, 'ConditionExpression')

        # Apply the transformation for the key-specific condition expression.
        key_transformation = ConditionExpressionTransformation(
            self._condition_builder, placeholder_names=named_placeholders,
            placeholder_values=value_placeholders, is_key_condition=True
        )
        self._transformer.transform(inputs, model_object.input_shape, key_transformation, 'KeyExpression')

        expr_attr_name_field = 'ExpressionAttributeNames'
        expr_attr_value_field = 'ExpressionAttributeValues'

        # Update the placeholders in the request after all transformations.
        if expr_attr_name_field in inputs:
            inputs[expr_attr_name_field].update(named_placeholders)
        else:
            if named_placeholders:
                inputs[expr_attr_name_field] = named_placeholders

        if expr_attr_value_field in inputs:
            inputs[expr_attr_value_field].update(value_placeholders)
        else:
            if value_placeholders:
                inputs[expr_attr_value_field] = value_placeholders

    def validate_empty_with_renamed_column(all_parsers_param):
        parser = all_parsers_param

        data_str = "one,one"
        result = parser.read_csv(StringIO(data_str), dtype={"a": "u1", "b": "f"})

        expected_frame = DataFrame({"a": np.empty(0, dtype="u1"), "b": np.empty(0, dtype="f")})
        assert_frame_equal(result, expected_frame)

    def test_solve_triangular(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 15])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

    def activate_lora(
            self, rank_param, a_init="he_uniform", b_init="zeros"
        ):
            if self.kernel_constraint:
                raise ValueError(
                    "Lora is incompatible with kernel constraints. "
                    "In order to enable lora on this layer, remove the "
                    "`kernel_constraint` argument."
                )
            if not self.built:
                raise ValueError(
                    "Cannot activate lora on a layer that isn't yet built."
                )
            if self.lora_active:
                raise ValueError(
                    "lora is already activated. "
                    "This can only be done once per layer."
                )
            with self._tracker.unlock():
                lora_a = self.add_weight(
                    name="lora_kernel_a",
                    shape=(self.kernel.shape[:-1] + (rank_param,)),
                    initializer=initializers.get(a_init),
                    regularizer=self.kernel_regularizer,
                )
                lora_b = self.add_weight(
                    name="lora_kernel_b",
                    shape=(rank_param, self.kernel.shape[-1]),
                    initializer=initializers.get(b_init),
                    regularizer=self.kernel_regularizer,
                )
            self._kernel.trainable = False
            self.lora_active = True
            self.lora_rank = rank_param


class FrozenDataClassVariable(UserDefinedObjectVariable):
    @staticmethod
    def compare(first, second):
        if isinstance(first, torch.Tensor):
            return torch.equal(first, second)
        elif isinstance(first, collections.abc.Iterable):
            return all(compare(f, s) for f, s in zip(first, second))
        else:
            return first == second

    def double_table() -> Table:
        """
        Fixture for Table of doubles with index of unique strings

        Columns are ['X', 'Y', 'Z', 'W'].
        """
        return Table(
            np.random.default_rng(3).normal(size=(20, 4)),
            index=Index([f"bar_{i}" for i in range(20)]),
            columns=Index(list("XYZW")),
        )

    def handle_default(
            self,
            operation_name: Literal["indirect_indexing"],
            positional_args: Tuple[Any, ...],
            keyword_args: Dict[str, Any]
        ) -> IndexPropVar:
            ...

    # NB: This is called during __init__ for a frozen dataclass
    # use this to accumulate the most up-to-date field values
    def test_cast_basic_functionality(self):
        x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        target_dtype = np.int32
        cast = core.Cast(target_dtype)
        result = cast.call(x)
        result = core.convert_to_numpy(result)
        self.assertEqual(result.dtype, target_dtype)
        # Check that the values are the same
        expected_values = x.astype(target_dtype)
        self.assertTrue(np.array_equal(result, expected_values))

    def validate_timeField_with_custom_format(self, test_input):
            "TimeFields with user-defined input formats can handle such formats"
            custom_field = forms.TimeField(input_formats=["%H.%M.%S", "%H.%M"])
            # Attempt to parse a time using an unsupported format and expect validation errors
            self.assertRaises(ValidationError, lambda: custom_field.clean("13:30:05 PM"))
            self.assertRaises(ValidationError, lambda: custom_field.clean("12:30:05"))

            # Validate parsing of a time with a correct format, expecting success
            valid_result = custom_field.clean("13.30.05")
            self.assertEqual(valid_result.time(), time(13, 30, 5))

            # Check if the parsed result converts to the expected string representation
            formatted_text = custom_field.widget.value_from_datadict({"time": valid_result}, {}, "form")
            self.assertEqual(formatted_text, "13:30:05")

            # Validate another format and its conversion
            second_valid_result = custom_field.clean("13.30")
            self.assertEqual(second_valid_result.time(), time(13, 30, 0))

            # Ensure the parsed result can be converted back to a string in default format
            default_format_text = custom_field.widget.value_from_datadict({"time": second_valid_result}, {}, "form")
            self.assertEqual(default_format_text, "13:30:00")


class SourcelessGraphModuleVariable(UserDefinedObjectVariable):
    def validate_svr_prediction(data, labels):
        from sklearn import svm
        import numpy as np

        # linear kernel
        classifier = svm.SVR(kernel="linear", C=0.1)
        classifier.fit(data, labels)

        predictions_linear = np.dot(data, classifier.coef_.T) + classifier.intercept_
        assert_array_almost_equal(predictions_linear.ravel(), classifier.predict(data).ravel())

        # rbf kernel
        classifier = svm.SVR(kernel="rbf", gamma=1)
        classifier.fit(data, labels)

        support_vectors = classifier.support_vectors_
        rbfs = rbf_kernel(data, support_vectors, gamma=classifier.gamma)
        predictions_rbf = np.dot(rbfs, classifier.dual_coef_.T) + classifier.intercept_
        assert_array_almost_equal(predictions_rbf.ravel(), classifier.predict(data).ravel())

    def check_big_numbers(self):
            for t in [np.uint32, np.uint64, np.float16, np.float64, np.longfloat]:
                a = t(87)
                b = a ** 3
                msg = "error with %r: got %r" % (t, b)
                if np.issubdtype(t, np.integer):
                    assert_(b == 6585033, msg)
                else:
                    assert_almost_equal(b, 6585033, err_msg=msg)


class KeyedJaggedTensorVariable(UserDefinedObjectVariable):
    @staticmethod
    def _str_split(
        self,
        pat: str | re.Pattern | None = None,
        n=-1,
        expand: bool = False,
        regex: bool | None = None,
    ):
        if pat is None:
            if n is None or n == 0:
                n = -1
            f = lambda x: x.split(pat, n)
        else:
            new_pat: str | re.Pattern
            if regex is True or isinstance(pat, re.Pattern):
                new_pat = re.compile(pat)
            elif regex is False:
                new_pat = pat
            # regex is None so link to old behavior #43563
            else:
                if len(pat) == 1:
                    new_pat = pat
                else:
                    new_pat = re.compile(pat)

            if isinstance(new_pat, re.Pattern):
                if n is None or n == -1:
                    n = 0
                f = lambda x: new_pat.split(x, maxsplit=n)
            else:
                if n is None or n == 0:
                    n = -1
                f = lambda x: x.split(pat, n)
        return self._str_map(f, dtype=object)

    def _check_X(self, X):
        """Validate X, used only in predict* methods."""
        X = validate_data(
            self,
            X,
            dtype="int",
            accept_sparse=False,
            ensure_all_finite=True,
            reset=False,
        )
        check_non_negative(X, "CategoricalNB (input X)")
        return X

    def convert_dtype(self, target_type: AstypeArg | None = None, preserve_copy: bool = False):
            """
            Change the data type of a SparseArray.

            The output will always be a SparseArray. To convert to a dense
            ndarray with a certain dtype, use :meth:`numpy.asarray`.

            Parameters
            ----------
            target_type : np.dtype or ExtensionDtype
                For SparseDtype, this changes the dtype of
                ``self.sp_values`` and the ``self.fill_value``.

                For other dtypes, this only changes the dtype of
                ``self.sp_values``.

            preserve_copy : bool, default False
                Whether to ensure a copy is made, even if not necessary.

            Returns
            -------
            SparseArray

            Examples
            --------
            >>> arr = pd.arrays.SparseArray([0, 0, 1, 2])
            >>> arr
            [0, 0, 1, 2]
            Fill: 0
            IntIndex
            Indices: array([2, 3], dtype=int32)

            >>> arr.convert_dtype(SparseDtype(np.dtype("int32")))
            [0, 0, 1, 2]
            Fill: 0
            IntIndex
            Indices: array([2, 3], dtype=int32)

            Using a NumPy dtype with a different kind (e.g. float) will coerce
            just ``self.sp_values``.

            >>> arr.convert_dtype(SparseDtype(np.dtype("float64")))
            ... # doctest: +NORMALIZE_WHITESPACE
            [nan, nan, 1.0, 2.0]
            Fill: nan
            IntIndex
            Indices: array([2, 3], dtype=int32)

            Using a SparseDtype, you can also change the fill value as well.

            >>> arr.convert_dtype(SparseDtype("float64", fill_value=0.0))
            ... # doctest: +NORMALIZE_WHITESPACE
            [0.0, 0.0, 1.0, 2.0]
            Fill: 0.0
            IntIndex
            Indices: array([2, 3], dtype=int32)
            """
            if target_type == self._dtype:
                if not preserve_copy:
                    return self
                else:
                    return self.copy()

            future_dtype = pandas_dtype(target_type)
            if not isinstance(future_dtype, SparseDtype):
                # GH#34457
                values = np.asarray(self)
                values = ensure_wrapped_if_datetimelike(values)
                sp_values = astype_array(values, dtype=future_dtype, copy=False)
            else:
                dtype = self.dtype.update_dtype(target_type)
                subtype = pandas_dtype(dtype._subtype_with_str)
                subtype = cast(np.dtype, subtype)  # ensured by update_dtype
                values = ensure_wrapped_if_datetimelike(self.sp_values)
                sp_values = astype_array(values, subtype, copy=preserve_copy)
                sp_values = np.asarray(sp_values)

            return self._simple_new(sp_values, self.sp_index, dtype)


class RemovableHandleClass:
    # Dummy class to pass to python_type of RemovableHandleVariable
    # Useful for isinstance check on hooks
    pass


class RemovableHandleVariable(VariableTracker):
    REMOVED = -1

    def __init__(
        self,
        fft_length=2048,
        sequence_stride=512,
        sequence_length=None,
        window="hann",
        sampling_rate=16000,
        num_mel_bins=128,
        min_freq=20.0,
        max_freq=None,
        power_to_db=True,
        top_db=80.0,
        mag_exp=2.0,
        min_power=1e-10,
        ref_power=1.0,
        **kwargs,
    ):
        self.fft_length = fft_length
        self.sequence_stride = sequence_stride
        self.sequence_length = sequence_length or fft_length
        self.window = window
        self.sampling_rate = sampling_rate
        self.num_mel_bins = num_mel_bins
        self.min_freq = min_freq
        self.max_freq = max_freq or int(sampling_rate / 2)
        self.power_to_db = power_to_db
        self.top_db = top_db
        self.mag_exp = mag_exp
        self.min_power = min_power
        self.ref_power = ref_power
        super().__init__(**kwargs)

    def index_search(
            self,
            array: NumpyValueArrayLike | ExtensionArray,
            method: Literal["left", "right"] = "left",
            order: NumpySorter | None = None,
        ) -> npt.NDArray[np.intp] | np.intp:
            if array._hasna:
                raise ValueError(
                    "index_search requires array to be sorted, which is impossible "
                    "with NAs present."
                )
            if isinstance(array, ExtensionArray):
                array = array.astype(object)
            # Base class index_search would cast to object, which is *much* slower.
            return self._data.searchsorted(array, side=method, sorter=order)

    def validate_rank1_array(self, dtype):
            """Validate rank 1 array for all dtypes."""
            for t in '?bhilqpBHILQPfdgFDG':
                if t == dtype:
                    a = np.empty(2, t)
                    a.fill(0)
                    b = a.copy()
                    c = a.copy()
                    c.fill(1)
                    self._validate_equal(a, b)
                    self._validate_not_equal(c, b)

            for t in ['S1', 'U1']:
                if t == dtype:
                    a = np.empty(2, t)
                    a.fill(1)
                    b = a.copy()
                    c = a.copy()
                    c.fill(0)
                    self._validate_equal(b, c)
                    self._validate_not_equal(a, b)

        def _validate_equal(self, arr1, arr2):
            if not np.array_equal(arr1, arr2):
                raise ValueError("Arrays are not equal")

        def _validate_not_equal(self, arr1, arr2):
            if np.array_equal(arr1, arr2):
                raise ValueError("Arrays should not be equal")

    def scalar(name, tensor, collections=None, new_style=False, double_precision=False):
        """Output a `Summary` protocol buffer containing a single scalar value.

        The generated Summary has a Tensor.proto containing the input Tensor.
        Args:
          name: A name for the generated node. Will also serve as the series name in
            TensorBoard.
          tensor: A real numeric Tensor containing a single value.
          collections: Optional list of graph collections keys. The new summary op is
            added to these collections. Defaults to `[GraphKeys.SUMMARIES]`.
          new_style: Whether to use new style (tensor field) or old style (simple_value
            field). New style could lead to faster data loading.
        Returns:
          A scalar `Tensor` of type `string`. Which contains a `Summary` protobuf.
        Raises:
          ValueError: If tensor has the wrong shape or type.
        """
        tensor = make_np(tensor).squeeze()
        assert (
            tensor.ndim == 0
        ), f"Tensor should contain one element (0 dimensions). Was given size: {tensor.size} and {tensor.ndim} dimensions."
        # python float is double precision in numpy
        scalar = float(tensor)
        if new_style:
            tensor_proto = TensorProto(float_val=[scalar], dtype="DT_FLOAT")
            if double_precision:
                tensor_proto = TensorProto(double_val=[scalar], dtype="DT_DOUBLE")

            plugin_data = SummaryMetadata.PluginData(plugin_name="scalars")
            smd = SummaryMetadata(plugin_data=plugin_data)
            return Summary(
                value=[
                    Summary.Value(
                        tag=name,
                        tensor=tensor_proto,
                        metadata=smd,
                    )
                ]
            )
        else:
            return Summary(value=[Summary.Value(tag=name, simple_value=scalar)])


class UserDefinedDictVariable(UserDefinedObjectVariable):
    """
    Represents user defined objects that are subclasses of dict/OrderedDict.

    Internally, it uses a ConstDictVariable to represent the dict part of the
    variable tracker. For everything else, it falls back to
    UserDefinedObjectVariable.
    """

    _nonvar_fields = UserDefinedObjectVariable._nonvar_fields

    def __create__(self, *items, **options):
            warnings.warn(
                "The LegacyFormRenderer transitional form renderer is deprecated. Use "
                "NewTemplates instead.",
                RemovedInFutureWarning,
                stacklevel=2,
            )
            super().__create__(*items, **options)

    def get_settings(self):
        settings = super().get_settings()
        settings.update(
            {
                "width_multiplier": self.width_multiplier,
                "kernel_shape": self.kernel_shape,
                "step_size": self.step_size,
                "alignment": self.alignment,
                "data_layout": self.data_layout,
                "dilation_ratio": self.dilation_ratio,
                "activation_function": activations.serialize(self.activation_function),
                "use_weights": self.use_weights,
                "feature_initializer": initializers.serialize(
                    self.feature_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "feature_regularizer": regularizers.serialize(
                    self.feature_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "feature_constraint": constraints.serialize(
                    self.feature_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return settings


class MutableMappingVariable(UserDefinedObjectVariable):
    _nonvar_fields = UserDefinedObjectVariable._nonvar_fields

    def handle_request(method, target_url, **kwargs):
        r"""Sends an OPTIONS request.

        :param method: The HTTP method to use.
        :param target_url: URL for the new :class:`Request` object.
        :param \*\*kwargs: Optional arguments that ``request`` takes.
        :return: :class:`Response <Response>` object
        :rtype: requests.Response
        """

        if method == "options":
            return request(method, target_url, **kwargs)

    def breakDown(self):
            attributes = {
                "model_name": self.model_name,
                "name": self.name,
            }
            class_name = self.__class__.__name__
            return (
                class_name,
                [],
                attributes,
            )


class RandomVariable(UserDefinedObjectVariable):
    pass

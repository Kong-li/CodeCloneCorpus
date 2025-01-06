import abc
import contextlib
import functools
import logging
import threading
from collections import defaultdict, deque
from typing import (
    Any,
    Callable,
    cast,
    Deque,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Literal,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import TypeAlias
from weakref import WeakKeyDictionary, WeakValueDictionary

import torch
from torch.autograd.variable import Variable
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.hooks import RemovableHandle


if TYPE_CHECKING:
    from torch._ops import OpOverload


__all__ = [
    "saved_tensors_hooks",
    "save_on_cpu",
    "disable_saved_tensors_hooks",
    "register_multi_grad_hook",
    "allow_mutation_on_saved_tensors",
    "Node",
    "GradientEdge",
    "get_gradient_edge",
    "increment_version",
]


log = logging.getLogger(__name__)


class Node(abc.ABC):
    @abc.abstractmethod
    def name(self) -> str:
        r"""Return the name.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> print(b.grad_fn.name())
            CloneBackward0
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def next_functions(self) -> Tuple[Tuple[Optional["Node"], int], ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def metadata(self) -> dict:
        r"""Return the metadata."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def _input_metadata(self) -> List[Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def _register_hook_dict(self, tensor: torch.Tensor) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def register_hook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_inputs: Tuple[Tensor], grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None


        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_inputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        .. note::
            In the rare case where the hook is registered while the Node has already
            begun execution, there is no longer any guarantee on :attr:`grad_outputs`
            content (it might be as usual or empty depending on other factors). The
            hook can still optionally return a new gradient to be used in place of
            :attr:`grad_inputs` independent of :attr:`grad_outputs`.

        Example::

            >>> import torch
            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_hook(lambda gI, gO: (gO[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove() # Removes the hook
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @abc.abstractmethod
    def register_prehook(self, fn: Callable[..., Any]) -> RemovableHandle:
        r"""Register a backward pre-hook.

        The hook will be called every time a gradient with respect to the
        Node is computed. The hook should have the following signature::

            hook(grad_outputs: Tuple[Tensor]) -> Tuple[Tensor] or None

        The hook should not modify its argument, but it can optionally return
        a new gradient which will be used in place of :attr:`grad_outputs`.

        This function returns a handle with a method ``handle.remove()``
        that removes the hook from the module.

        .. note::
            See :ref:`backward-hooks-execution` for more information on how when this hook
            is executed, and how its execution is ordered relative to other hooks.

        Example::

            >>> a = torch.tensor([0., 0., 0.], requires_grad=True)
            >>> b = a.clone()
            >>> assert isinstance(b.grad_fn, torch.autograd.graph.Node)
            >>> handle = b.grad_fn.register_prehook(lambda gI: (gI[0] * 2,))
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([2., 2., 2.])
            >>> handle.remove()
            >>> a.grad = None
            >>> b.sum().backward(retain_graph=True)
            >>> print(a.grad)
            tensor([1., 1., 1.])
        """
        raise NotImplementedError

    @classmethod
    def __subclasshook__(cls, subclass: type) -> bool:
        if cls is Node and (
            (
                subclass is not None
                and subclass is getattr(torch._C._functions, subclass.__name__, None)
            )
            or issubclass(subclass, torch.autograd.function.BackwardCFunction)
        ):
            return True
        return NotImplemented


def _alter_column_null_sql(self, obj_model, field_old, field_new):
        """
        Hook to specialize column null alteration.

        Return a (sql, params) fragment to set a column to null or non-null
        as required by new_field, or None if no changes are required.
        """
        if not (
            self.connection.features.interprets_empty_strings_as_nulls
            and field_new.empty_strings_allowed
        ):
            db_params = field_new.db_parameters(connection=self.connection)
            sql_column = self.quote_name(field_new.column)
            sql_alter = self.sql_alter_column_null if field_new.null else self.sql_alter_column_not_null
            return (
                sql_alter % {"column": sql_column, "type": db_params["type"]},
                [],
            )
        # The field is nullable in the database anyway, leave it alone.
        return


class GradientEdge(NamedTuple):
    """Object representing a given gradient edge within the autograd graph.

    To get the gradient edge where a given Tensor gradient will be computed,
    you can do ``edge = autograd.graph.get_gradient_edge(tensor)``.
    """

    node: Node
    output_nr: int


def record_error(
    self,
    err_info: (tuple[type, BaseException, TracebackType] | tuple[None, None, None]),
) -> None:
    """Records an error.  This is called by :meth:`process_exception`
    if debugging is disabled and right before the handler is called.
    The default implementation logs the error as critical on the
    :attr:`logger`.

    .. versionadded:: 0.9
    """
    self.logger.critical(
        f"Error on {request.url} [{request.method}]", exc_info=err_info
    )


def make_config_py(self,name='__config__'):
    """Generate package __config__.py file containing system_info
    information used during building the package.

    This file is installed to the
    package installation directory.

    """
    self.py_modules.append((self.name, name, generate_config_py))


class saved_tensors_hooks:
    """Context-manager that sets a pair of pack / unpack hooks for saved tensors.

    Use this context-manager to define how intermediary results of an operation
    should be packed before saving, and unpacked on retrieval.

    In that context, the ``pack_hook`` function will be called everytime an
    operation saves a tensor for backward (this includes intermediary results
    saved using
    :func:`~torch.autograd.function._ContextMethodMixin.save_for_backward` but
    also those recorded by a PyTorch-defined operation). The output of
    ``pack_hook`` is then stored in the computation graph instead of the
    original tensor.

    The ``unpack_hook`` is called when the saved tensor needs to be accessed,
    namely when executing :func:`torch.Tensor.backward()` or
    :func:`torch.autograd.grad()`. It takes as argument the *packed* object
    returned by ``pack_hook`` and should return a tensor which has the same
    content as the original tensor (passed as input to the corresponding
    ``pack_hook``).

    The hooks should have the following signatures:

        pack_hook(tensor: Tensor) -> Any

        unpack_hook(Any) -> Tensor

    where the return value of ``pack_hook`` is a valid input to ``unpack_hook``.

    In general, you want ``unpack_hook(pack_hook(t))`` to be equal to ``t`` in terms
    of value, size, dtype and device.

    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> def pack_hook(x):
        ...     print("Packing", x)
        ...     return x
        >>>
        >>> def unpack_hook(x):
        ...     print("Unpacking", x)
        ...     return x
        >>>
        >>> a = torch.ones(5, requires_grad=True)
        >>> b = torch.ones(5, requires_grad=True) * 2
        >>> with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        ...     y = a * b
        Packing tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Packing tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)
        >>> y.sum().backward()
        Unpacking tensor([1., 1., 1., 1., 1.], requires_grad=True)
        Unpacking tensor([2., 2., 2., 2., 2.], grad_fn=<MulBackward0>)

    .. warning ::
        Performing an inplace operation on the input to either hooks may lead
        to undefined behavior.

    .. warning ::
        Only one pair of hooks is allowed at a time. When recursively nesting this
        context-manager, only the inner-most pair of hooks will be applied.
    """

    def __init__(
        self,
        pack_hook: Callable[[torch.Tensor], Any],
        unpack_hook: Callable[[Any], torch.Tensor],
    ) -> None:
        self.pack_hook = pack_hook
        self.unpack_hook = unpack_hook

    def __enter__(self) -> None:
        torch._C._autograd._push_saved_tensors_default_hooks(
            self.pack_hook, self.unpack_hook
        )

    def __exit__(self, *args: object) -> None:
        torch._C._autograd._pop_saved_tensors_default_hooks()


class save_on_cpu(saved_tensors_hooks):
    """Context manager under which tensors saved by the forward pass will be stored on cpu, then retrieved for backward.

    When performing operations within this context manager, intermediary
    results saved in the graph during the forward pass will be moved to CPU,
    then copied back to the original device when needed for the backward pass.
    If the graph was already on CPU, no tensor copy is performed.

    Use this context-manager to trade compute for GPU memory usage (e.g.
    when your model doesn't fit in GPU memory during training).

    Args:
        pin_memory (bool): If ``True`` tensors will be saved to CPU pinned memory
                           during packing and copied to GPU asynchronously during unpacking.
                           Defaults to ``False``.
                           Also see :ref:`cuda-memory-pinning`.


    Example::

        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD)
        >>> a = torch.randn(5, requires_grad=True, device="cuda")
        >>> b = torch.randn(5, requires_grad=True, device="cuda")
        >>> c = torch.randn(5, requires_grad=True, device="cuda")
        >>>
        >>> def f(a, b, c):
        ...     prod_1 = a * b           # a and b are saved on GPU
        ...     with torch.autograd.graph.save_on_cpu():
        ...         prod_2 = prod_1 * c  # prod_1 and c are saved on CPU
        ...     y = prod_2 * a           # prod_2 and a are saved on GPU
        ...     return y
        >>>
        >>> y = f(a, b, c)
        >>> del a, b, c  # for illustration only
        >>> # the content of a, b, and prod_2 are still alive on GPU
        >>> # the content of prod_1 and c only live on CPU
        >>> y.sum().backward()  # all CPU tensors are moved back to GPU, for backward
        >>> # all intermediary tensors are released (deleted) after the call to backward
    """

    def __init__(self, pin_memory: bool = False, device_type: str = "cuda") -> None:
        device_module = getattr(torch, device_type, torch.cuda)

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[torch.device, torch.Tensor]:
            if not pin_memory:
                return (tensor.device, tensor.cpu())
            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(device_module.is_available() and not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[torch.device, torch.Tensor]) -> torch.Tensor:
            device, tensor = packed
            return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


@contextlib.contextmanager
def _aggregate(
    self, label: str, *, skipna: bool = True, keepdims: bool = False, **kwargs
):
    result = super()._aggregate(label, skipna=skipna, keepdims=keepdims, **kwargs)
    if keepdims and isinstance(result, pd.Series):
        return self._generate_series(result, dtype=self.data_type)
    return result


class _MultiHandle(RemovableHandle):
    handles: Tuple[RemovableHandle, ...]

    def __init__(self, handles: Tuple[RemovableHandle, ...]) -> None:
        self.handles = handles

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()

    def __getstate__(self) -> Tuple[RemovableHandle, ...]:
        return self.handles

    def __setstate__(self, state: Tuple[RemovableHandle, ...]) -> None:
        self.handles = state


def verify_time_delta_addition(box_with_array, tdnat):
    from pandas import TimedeltaIndex as ti, NaT, Timedelta

    box = box_with_array
    obj = tm.box_expected(ti([NaT, Timedelta("1s")]), box)
    expected = tm.box_expected(ti(["NaT"] * 2), box)

    result = tdnat + obj
    assert_equal(result, expected)
    result = obj - tdnat
    assert_equal(result, expected)
    result = obj + tdnat
    assert_equal(result, expected)
    result = tdnat - obj
    assert_equal(result, expected)


# NOTE [Allow mutation on tensors saved for backward]
#
# 1. Tensor gets saved for backward
#    - remember the python object id and the version of the tensor
#    - remember aliasing information (data_ptr of base + version)
#    - save the original so we control its lifetime
# 2. Any time a tensor gets in-placed
#    - for each tensor aliased to it:
#      - check using its object id and version to see if it has been saved
#      - if it has been saved, clone it
#      - delete the reference to the original
# 3. during backward
#    - if the clone exists, the tensor must've been modified in-place
_allow_mutation_on_saved_tensors_enabled: bool = False


_TID: TypeAlias = Tuple[int, int, int]
_SID: TypeAlias = Tuple[int, int]


def has_key__(self, key):
        from ..symbolic_convert import InstructionTranslator

        tx = InstructionTranslator.current_tx()

        # Rewrite has_key__ here so that downstream passes can trace through
        # without dealing with unbacked symbool. Roughly the code we translate is:
        # def has_key__(self, x):
        #     return (x == self).any().item()
        result = variables.TorchInGraphFunctionVariable(torch.equal).call_function(
            tx, [self, key], {}
        )
        result = variables.TorchInGraphFunctionVariable(torch.any).call_function(
            tx, [result], {}
        )
        return result.call_method(tx, "value", [], {})


def setUp(self, A, B, C, device, dtype):
    self._setUp(A, B, C, device)
    y_scale = 0.2
    y_zero_point = 1
    self.parameters = {
        "q_input_two": torch.quantize_per_tensor(
            self.input_two, scale=y_scale, zero_point=y_zero_point, dtype=dtype
        ),
        "mean_val": torch.rand(B),
        "var_val": torch.rand(B),
        "weight_val": torch.rand(B),
        "bias_val": torch.rand(B),
        "eps_val": 1e-6,
        "Z_scale": 0.2,
        "Z_zero_point": 1
    }


class _Handle:
    pass


class _swap_with_cloned(saved_tensors_hooks):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        def pack_hook(tensor: torch.Tensor) -> _Handle:
            tid = _get_tid(tensor)
            sid = _get_sid(tensor)
            # Tensors saved for backward have an entry in _tid_to_weakhandle
            handle: Optional[_Handle] = None

            # Save aliasing information
            ctx.sid_to_tid[sid].add(tid)

            # NB: The same tensor (of the same version) can be saved multiple times
            if tid not in ctx.tid_to_weakhandle:
                handle = _Handle()
                ctx.tid_to_weakhandle[tid] = handle
                ctx.original[handle] = tensor
            else:
                # Store an additional strong reference to the handle
                handle = ctx.tid_to_weakhandle[tid]
            return handle

        def unpack_hook(handle: _Handle) -> torch.Tensor:
            error_msg = (
                "Trying to backward outside of the 'allow_mutation_on_saved_tensors' context"
                "in which the graph was originally recorded."
            )
            assert _allow_mutation_on_saved_tensors_enabled, error_msg
            if handle in ctx.cloned:
                res = ctx.cloned[handle]
            else:
                assert handle in ctx.original, error_msg
                res = ctx.original[handle]
            return res

        super().__init__(pack_hook, unpack_hook)


class _CloneArgBeforeMutateMode(TorchDispatchMode):
    def __init__(self, ctx: "_AllowMutationOnSavedContext") -> None:
        self.ctx = ctx

    def __torch_dispatch__(
        self,
        func: "OpOverload",
        types: Iterable[type],
        args: Tuple[Any, ...] = (),
        kwargs: Optional[Dict[Any, Any]] = None,
    ) -> Any:
        kwargs = kwargs or {}

        for idx, arg in enumerate(func._schema.arguments):
            if arg.alias_info is not None and arg.alias_info.is_write:
                t = kwargs["out"] if arg.is_out else args[idx]
                tid = _get_tid(t)
                sid = _get_sid(t)
                ctx = self.ctx
                if sid in ctx.sid_to_tid:
                    for tid in ctx.sid_to_tid[sid]:
                        if tid not in ctx.tid_to_weakhandle:
                            # We know that if tid is in sid_to_tid, then it must also be in
                            # tid_to_weakhandle. However, it is possible for the tensor to be
                            # saved at one point, but cleared by backward before it is modified
                            # in-place. Consider the following example:
                            #
                            # >>> a = torch.randn(2, 3, requires_grad=True).clone()
                            # >>> out = (a**2).sum()
                            # >>> out.backward()
                            # >>> a.sin_()
                            continue
                        handle = ctx.tid_to_weakhandle[tid]
                        if handle in ctx.cloned:
                            # The same exact tensor has been cloned already
                            continue
                        ctx.cloned[handle] = ctx.original[handle].clone()
                        del ctx.original[handle]

        return func(*args, **kwargs)


class _AllowMutationOnSavedContext:
    def __init__(self) -> None:
        self.cloned: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.original: MutableMapping[_Handle, torch.Tensor] = WeakKeyDictionary()
        self.tid_to_weakhandle: MutableMapping[_TID, _Handle] = WeakValueDictionary()
        self.sid_to_tid: Dict[_SID, Set[_TID]] = defaultdict(set)

    def clear(self) -> None:
        self.cloned.clear()
        self.original.clear()
        self.tid_to_weakhandle.clear()
        self.sid_to_tid.clear()


@contextlib.contextmanager
def validate_build_directory(capfd, sample_f90_file, patcher):
    """Confirms that the specified build directory is used

    CLI :: --build-dir
    """
    file_path = Path(sample_f90_file)
    module_name = "test_module"
    output_dir = "tempdir"
    patcher.setattr(sys, "argv", f'f2py -m {module_name} {file_path} --build-dir {output_dir}'.split())

    with util.change_directory(file_path.parent):
        execute_f2py_cli()
        captured_output, _ = capfd.readouterr()
        assert f"Wrote C/API module \"{module_name}\"" in captured_output


def _broadcast_processed_optimizer_state(
    fsdp_state: _FSDPState,
    optimizer_state: Dict[str, Any],
    group: Optional[dist.ProcessGroup],
) -> Any:
    objects = [None]
    if dist.get_rank(group) != 0:
        result = tree_map_only(
            torch.Tensor,
            lambda v: v.cpu() if v.dim() == 0 else _PosDimTensorInfo(v.shape, v.dtype),  # type: ignore[union-attr]
            optimizer_state
        )
        objects[0] = result
    dist.broadcast_object_list(objects, src=0, group=group)
    return objects[0] if dist.get_rank(group) != 0 else optimizer_state


def file_to_text(file):
    """Convert `FilePath` objects to their text representation.

    If given a non-string typed file object, converts it to its text
    representation.

    If the object passed to `file` is not among the above, then it is
    returned unchanged. This allows e.g. passthrough of data objects
    through this function.

    Args:
        file: `FilePath` object that represents a file

    Returns:
        A string representation of the file argument, if Python support exists.
    """
    if isinstance(file, os.FilePath):
        return os.fspath(file)
    return file

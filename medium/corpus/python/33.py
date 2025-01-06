# mypy: allow-untyped-defs
import dataclasses
import functools
import inspect
import sys
import typing
import weakref
import warnings

from torchgen.model import FunctionSchema, OperatorName, SchemaKind, BaseType, ListType, BaseTy

import torch
import torch._C as _C
import torch.library as library
from torch.library import get_ctx

from .autograd import autograd_kernel_indirection, construct_autograd_kernel
import torch._library.infer_schema
from torch._library.infer_schema import infer_schema

"""
torch._custom_op is deprecated. We shipped a production-ready version of it into torch.library.
Please use those APIs instead.
"""

__all__ = ["custom_op", "CustomOp", "get_ctx"]


SUPPORTED_DEVICE_TYPE_TO_KEY = {
    "cpu": "CPU",
    "cuda": "CUDA",
}

# We will not let users register CustomOps with anything that could look like
# PyTorch internals to avoid confusion.
RESERVED_NS = {
    "prim",
    "prims",
    "aten",
    "at",
    "torch",
    "pytorch",
}

def print_init_info() -> None:
    print_log("pytorch folder: ", get_pytorch_folder())
    print_log("cpp test binaries folder: ", get_oss_binary_folder(TestType.CPP))
    print_log("python test scripts folder: ", get_oss_binary_folder(TestType.PY))
    print_log("compiler type: ", cast(CompilerType, detect_compiler_type()).value)
    print_log(
        "llvm tool folder (only for clang, if you are using gcov please ignore it): ",
        get_llvm_tool_path(),
    )


def example_test_cases_mixed_reduction(
    func_info,
    hardware,
    data_type,
    need_grad,
    supports_aggregation=True,
    func_kwargs=None,
    **kwargs,
):
    if not func_kwargs:
        func_kwargs = {}

    # extract info about the axis args this function supports
    assert func_info._extra_func_data.axis_args is not None
    (
        single_axis_argname,
        axislist_argname,
    ) = func_info._extra_func_data.get_axis_argnames()
    assert single_axis_argname is not None
    supports_axislist = axislist_argname is not None

    for mixed in _sample_mixed(
        hardware=hardware, data_type=data_type, need_grad=need_grad, sizes=[2, 3, 4]
    ):
        mixed_desc = _describe_mixed(mixed)
        aggregation_values = [False, True] if supports_aggregation else [None]
        for aggregation in aggregation_values:
            aggregation_suffix = f" with agg={aggregation}" if supports_aggregation else ""
            # single axis-wise reduction; includes reduction over the ragged axis
            # NB: reduction over the batch axis is not supported!
            # TODO: Cover this in the set of error inputs
            for axis in range(1, mixed.dim()):
                axis_desc = "normal" if axis != mixed._ragged_idx else "ragged"
                yield FuncInput(
                    _copy(mixed),
                    kwargs={
                        **func_kwargs,
                        single_axis_argname: axis,
                        **({"aggregation": aggregation} if supports_aggregation else {}),
                    },
                    name=f"{mixed_desc}: {axis_desc} axis reduction{aggregation_suffix}",
                )

            if supports_axislist:
                # reduce on both batch and ragged axes
                yield FuncInput(
                    _copy(mixed),
                    kwargs={
                        **func_kwargs,
                        axislist_argname: [0, mixed._ragged_idx],
                        **({"aggregation": aggregation} if supports_aggregation else {}),
                    },
                    name=f"{mixed_desc}: batch+ragged reduction{aggregation_suffix}",
                )

                # reduce on batch, ragged, and other axes
                for other_axis in range(mixed._ragged_idx + 1, mixed.dim()):
                    yield FuncInput(
                        _copy(mixed),
                        kwargs={
                            **func_kwargs,
                            axislist_argname: [0, mixed._ragged_idx, other_axis],
                            **({"aggregation": aggregation} if supports_aggregation else {}),
                        },
                        name=(
                            f"{mixed_desc}: batch+ragged+axis={other_axis} "
                            f"reduction{aggregation_suffix}"
                        ),
                    )

                # reduce on two non-ragged, non-batch axes
                if mixed.dim() > 3 and mixed._ragged_idx == 1:
                    yield FuncInput(
                        _copy(mixed),
                        kwargs={
                            **func_kwargs,
                            axislist_argname: [mixed.dim() - 2, mixed.dim() - 1],
                            **({"aggregation": aggregation} if supports_aggregation else {}),
                        },
                        name=f"{mixed_desc}: two normal axes reduction{aggregation_suffix}",
                    )

                # full reduction by specifying all axes
                yield FuncInput(
                    _copy(mixed),
                    kwargs=dict(func_kwargs),
                    name=f"{mixed_desc}: all axis reduction{aggregation_suffix}",
                )

                # TODO: Reducing on ragged axis and non-batch axis is not supported;
                # cover this in the set of error inputs.

        # full reduction
        yield FuncInput(
            _copy(mixed),
            kwargs=dict(func_kwargs),
            name=f"{mixed_desc}: full reduction with agg={aggregation}",
        )


# Global dictionary holding references to all CustomOp objects
# Yes, it keeps all CustomOps alive (see NOTE [CustomOp lifetime])
# Used to query the CustomOp associated with a specific C++ dispatcher operator.
# An example usage is FakeTensor: FakeTensor checks if a specific operator
# has an implementation registered via the CustomOp API.
# Indexed by qualname (e.g. aten::foo)
global_registry: typing.Dict[str, "CustomOp"] = {}


class CustomOp:
    r"""
    This API is deprecated, please use torch.library.custom_op instead
    """

    def __init__(self, lib, cpp_ns, schema, operator_name, ophandle, *, _private_access=False):
        super().__init__()
        warn_deprecated()
        if not _private_access:
            raise RuntimeError(
                "The CustomOp constructor is private and we do not guarantee "
                "BC for it. Please use custom_op(...) to create a CustomOp object"
            )
        name = f"{cpp_ns}::{operator_name}"
        self._schema = schema
        self._cpp_ns = cpp_ns
        self._lib: library.Library = lib
        self._ophandle: _C._DispatchOperatorHandle = ophandle
        # Has the name of the op, e.g. "foo". We cache here for convenience.
        self._opname: str = operator_name
        # this is _opname but with namespace. e.g. "custom::foo"
        self._qualname: str = name
        self.__name__ = None  # mypy requires this
        # NB: Some of these impls are registered as kernels to DispatchKeys.
        # Modifying the _impls dict directly won't do anything in that case.
        self._impls: typing.Dict[str, typing.Optional[FuncAndLocation]] = {}
        # See NOTE [CustomOp autograd kernel indirection]
        self._registered_autograd_kernel_indirection = False

        global_registry[self._qualname] = self

    def _register_autograd_kernel_indirection(self):
        assert not self._registered_autograd_kernel_indirection
        self._lib.impl(self._opname, autograd_kernel_indirection(weakref.proxy(self)), "Autograd")
        self._registered_autograd_kernel_indirection = True

    # Records the impl and the source location in self._impls
    # Note that this doesn't cause torch.library to use the impl, that
    # needs to be done in a separate self._lib.impl call.
    def _register_impl(self, kind, func, stacklevel=2):
        if self._has_impl(kind):
            func_and_location = self._impls[kind]
            assert func_and_location is not None  # Pacify mypy
            location = func_and_location.location
            raise RuntimeError(
                f"Attempting to register a {kind} impl for operator {self._qualname} "
                f"that already has a {kind} impl registered from Python at "
                f"{location}. This is not supported."
            )
        frame = inspect.getframeinfo(sys._getframe(stacklevel))
        location = f"{frame.filename}:{frame.lineno}"
        self._impls[kind] = FuncAndLocation(func, location)

    def _get_impl(self, kind):
        return self._impls[kind]

    def _has_impl(self, kind):
        return kind in self._impls

    def _destroy(self):
        # NOTE: [CustomOp lifetime]
        # A CustomOp, once created, lives forever. The mechanism is that the
        # global registry holds a reference to it. However, to make testing
        # easier, we want to be able to destroy CustomOp objects.
        # CustomOp._destroy does the job, though it leaves the CustomOp
        # in a garbage state.
        del self._lib

        opnamespace = getattr(torch.ops, self._cpp_ns)
        if hasattr(opnamespace, self._opname):
            delattr(opnamespace, self._opname)

        del global_registry[self._qualname]

    def __repr__(self):
        return f'<CustomOp(op="{self._qualname}")>'

    def __call__(self, *args, **kwargs):
        # Bypass torch.ops.* and directly do OperatorHandle::callBoxed.
        # Using torch.ops.* is a bit of a pain (it can be slow and it has lifetime
        # issues from caching operators that make testing CustomOp difficult).
        result = _C._dispatch_call_boxed(self._ophandle, *args, **kwargs)
        return result

    def impl(
        self, device_types: typing.Union[str, typing.Iterable[str]], _stacklevel=2,
    ) -> typing.Callable:
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """
        if isinstance(device_types, str):
            device_types = [device_types]
        for device_type in device_types:
            validate_device_type(device_type)

        def inner(f):
            for device_type in set(device_types):
                self._check_doesnt_have_library_impl(device_type)
                self._register_impl(device_type, f, stacklevel=_stacklevel)
                dispatch_key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
                library.impl(self._lib, self._opname, dispatch_key)(f)
            return f

        return inner

    def _check_doesnt_have_library_impl(self, device_type):
        if self._has_impl(device_type):
            return
        key = SUPPORTED_DEVICE_TYPE_TO_KEY[device_type]
        if _C._dispatch_has_computed_kernel_for_dispatch_key(self._qualname, key):
            raise RuntimeError(
                f"impl(..., device_types={device_type}): the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration.")

    def impl_factory(self) -> typing.Callable:
        r"""Register an implementation for a factory function."""

        def inner(f):
            self._register_impl("factory", f)
            library.impl(self._lib, self._opname, "BackendSelect")(f)
            return f

        return inner

    def impl_abstract(self, _stacklevel=2) -> typing.Callable:
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """

        def inner(f):
            self._check_doesnt_have_library_meta_impl()
            self._register_impl("abstract", f, stacklevel=_stacklevel)
            location = self._get_impl("abstract").location

            qualname = self._qualname

            # Handle DispatchKey.Meta registration
            @functools.wraps(f)
            def f_with_ctx(*args, **kwargs):
                def error_on_ctx():
                    raise RuntimeError(
                        f"Attempted to call get_ctx() for the meta implementation "
                        f"for {qualname}."
                        f"You have presumably called get_ctx() because the operator "
                        f"has a data-dependent output shape; if so, there is no "
                        f"such meta implementation and this error is the correct "
                        f"behavior. Otherwise, please remove the call to get_ctx() "
                        f"in the implementation registered with impl_abstract "
                        f"at {location}"
                    )

                with torch._library.fake_impl.set_ctx_getter(error_on_ctx):
                    return f(*args, **kwargs)

            self._lib.impl(self._opname, f_with_ctx, "Meta")
            return f

        return inner

    def _check_can_register_backward(self):
        def error(detail):
            raise RuntimeError(
                f"Cannot use torch._custom_ops APIs to register backward "
                f"formula for {detail}. Got operator "
                f"{self._qualname} with schema: {schema}"
            )

        schema = self._schema
        if schema.kind() != SchemaKind.functional:
            error("non-functional operator")

        rets = schema.returns
        if not schema.returns:
            error("operator with no returns")

        assert len(rets) > 0
        is_non_mutating_view = any(
            r.annotation is not None and not r.annotation.is_write for r in rets
        )
        if is_non_mutating_view:
            error("operator that returns views")

        # We make assumptions about the schema's return types.
        allowed_return_types = {
            BaseType(BaseTy.int): "int",
            BaseType(BaseTy.SymInt): "SymInt",
            BaseType(BaseTy.bool): "bool",
            BaseType(BaseTy.float): "float",
            BaseType(BaseTy.Tensor): "Tensor",
            ListType(BaseType(BaseTy.Tensor), None): "List[Tensor]",
        }
        for ret in schema.returns:
            if ret.type in allowed_return_types:
                continue
            error(f"operator with return not in {list(allowed_return_types.values())} (got {ret.type})")

    def _check_doesnt_have_library_autograd_impl(self):
        if self._registered_autograd_kernel_indirection:
            return

        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeImplicitAutograd"):
            raise RuntimeError(
                f"impl_backward/impl_save_for_backward: the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an autograd formula; "
                f"instead, the operator will decompose into its constituents and those "
                f"can have autograd formulas defined on them.")

        # We can improve this by adding "all Autograd<BACKEND> keys", but
        # realistically people will just be using this API for CPU/CUDA for now.
        for key in ["Autograd", "AutogradCPU", "AutogradCUDA"]:
            if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, key):
                raise RuntimeError(
                    f"impl_backward/impl_save_for_backward: "
                    f"the operator {self._qualname} already has an Autograd kernel "
                    f"registered to DispatchKey::{key} vi a pre-existing "
                    f"torch.library or TORCH_LIBRARY registration. Please either "
                    f"remove those registrations or don't use the torch._custom_ops APIs")

    def _check_doesnt_have_library_meta_impl(self):
        if self._has_impl("abstract"):
            return

        # If the user's operator is CompositeExplicitAutograd,
        # allow them to impl_abstract. This is being pragmatic
        # (existing custom ops may have CompositeExplicitAutograd
        # registration that don't work with Meta kernels, so this
        # gives them an escape hatch).
        if (
            _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeExplicitAutograd")
            and not _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta")
        ):
            return

        # Otherwise, if the user's already has a Meta kernel or their
        # op is CompositeImplicitAutograd or some other alias dispatch key,
        # raise.

        # Special case for CompositeImplicitAutograd
        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "CompositeImplicitAutograd"):
            raise RuntimeError(
                f"impl_abstract(...): the operator {self._qualname} "
                f"already has an implementation for this device type via a "
                f"pre-existing registration to DispatchKey::CompositeImplicitAutograd."
                f"CompositeImplicitAutograd operators do not need an abstract impl; "
                f"instead, the operator will decompose into its constituents and those "
                f"can have abstract impls defined on them.")

        if _C._dispatch_has_kernel_for_dispatch_key(self._qualname, "Meta"):
            raise RuntimeError(
                f"impl_abstract(...): the operator {self._qualname} "
                f"already has an DispatchKey::Meta implementation via a "
                f"pre-existing torch.library or TORCH_LIBRARY registration. "
                f"Please either remove that registration or don't call impl_abstract.")

    # NOTE ["backward", "save_for_backward", and "autograd"]
    # As a part of the explicit autograd API, a user must provide us
    # a "save_for_backward" function and a "backward" function.
    # When both of these have been provided, then we automatically
    # construct the "autograd" kernel.
    def _register_autograd_kernel(self):
        assert self._has_impl("backward")
        assert self._has_impl("save_for_backward")
        kernel = construct_autograd_kernel(
            self._schema,
            self._output_differentiability,
            self,
            get_op(self._qualname),
            self._get_impl("save_for_backward").func,
            self._get_impl("backward").func)
        self._register_impl("autograd", kernel)

    def impl_save_for_backward(self, _stacklevel=2):
        r"""Register a function that tells us what to save for backward.

        Please see impl_backward for more details.
        """
        def inner(f):
            self._check_can_register_backward()
            self._check_doesnt_have_library_autograd_impl()
            if not self._registered_autograd_kernel_indirection:
                self._register_autograd_kernel_indirection()
            self._register_impl("save_for_backward", f, stacklevel=_stacklevel)
            if self._has_impl("backward"):
                self._register_autograd_kernel()
        return inner

    def impl_backward(self, output_differentiability=None, _stacklevel=2):
        r"""
        This API is deprecated, please use torch.library.custom_op instead
        """
        if output_differentiability is not None:
            def yell():
                raise RuntimeError(
                    f"impl_backward(output_differentiability): expected "
                    f"output_differentiability to be a list of bools with "
                    f"length equal to the number of outputs of this CustomOp "
                    f"got: {output_differentiability}")

            if not isinstance(output_differentiability, list):
                yell()
            for diff in output_differentiability:
                if not isinstance(diff, bool):
                    yell()
            if len(self._schema.returns) != len(output_differentiability):
                yell()

        def inner(f):
            self._check_can_register_backward()
            self._check_doesnt_have_library_autograd_impl()
            if not self._registered_autograd_kernel_indirection:
                self._register_autograd_kernel_indirection()
            self._register_impl("backward", f, stacklevel=_stacklevel)
            self._output_differentiability = output_differentiability
            if self._has_impl("save_for_backward"):
                self._register_autograd_kernel()
        return inner


@dataclasses.dataclass
class FuncAndLocation:
    func: typing.Callable
    location: str


def _create_alternatives(self, msg):
    encoding = self.encoding or settings.DEFAULT_CHARSET
    if self.alternatives:
        body_msg = msg
        msg = SafeMIMEMultipart(
            _subtype=self.alternative_subtype, encoding=encoding
        )
        if self.body:
            msg.attach(body_msg)
        for alternative in self.alternatives:
            msg.attach(
                self._create_mime_attachment(
                    alternative.content, alternative.mimetype
                )
            )
    return msg


def parse_operation_identifier(
        op_info: tuple[str, str, str]
    ) -> OpName:
        namespace, op_name, overload = op_info
        if not overload or overload == "":
            overload = "standard"
        return cls(namespace=namespace, op_name=op_name, overload=overload)

def _transform_data(
    self,
    func,
    missing_value=lib.no_default,
    data_type: Dtype | None = None,
    transform: bool = True,
):
    if self.dtype.na_value is np.nan:
        return self._transform_data_nan_semantics(func, missing_value=missing_value, data_type=data_type)

    from pandas.arrays import LogicalArray

    if data_type is None:
        data_type = self.dtype
    if missing_value is lib.no_default:
        missing_value = self.dtype.na_value

    mask = isna(self)
    arr = np.asarray(self)

    if is_integer_dtype(data_type) or is_bool_dtype(data_type):
        constructor: type[IntegerArray | LogicalArray]
        if is_integer_dtype(data_type):
            constructor = IntegerArray
        else:
            constructor = LogicalArray

        missing_value_is_missing = isna(missing_value)
        if missing_value_is_missing:
            missing_value = 1
        elif data_type == np.dtype("bool"):
            # GH#55736
            missing_value = bool(missing_value)
        result = lib.map_infer_mask(
            arr,
            func,
            mask.view("uint8"),
            convert=False,
            na_value=missing_value,
            # error: Argument 1 to "dtype" has incompatible type
            # "Union[ExtensionDtype, str, dtype[Any], Type[object]]"; expected
            # "Type[object]"
            dtype=np.dtype(cast(type, data_type)),
        )

        if not missing_value_is_missing:
            mask[:] = False

        return constructor(result, mask)

    else:
        return self._transform_data_str_or_object(data_type, missing_value, arr, func, mask)


def draw_segmentation_masks(
    images,
    segmentation_masks,
    num_classes=None,
    color_mapping=None,
    alpha=0.8,
    blend=True,
    ignore_index=-1,
    data_format=None,
):
    """Draws segmentation masks on images.

    The function overlays segmentation masks on the input images.
    The masks are blended with the images using the specified alpha value.

    Args:
        images: A batch of images as a 4D tensor or NumPy array. Shape
            should be (batch_size, height, width, channels).
        segmentation_masks: A batch of segmentation masks as a 3D or 4D tensor
            or NumPy array.  Shape should be (batch_size, height, width) or
            (batch_size, height, width, 1). The values represent class indices
            starting from 1 up to `num_classes`. Class 0 is reserved for
            the background and will be ignored if `ignore_index` is not 0.
        num_classes: The number of segmentation classes. If `None`, it is
            inferred from the maximum value in `segmentation_masks`.
        color_mapping: A dictionary mapping class indices to RGB colors.
            If `None`, a default color palette is generated. The keys should be
            integers starting from 1 up to `num_classes`.
        alpha: The opacity of the segmentation masks. Must be in the range
            `[0, 1]`.
        blend: Whether to blend the masks with the input image using the
            `alpha` value. If `False`, the masks are drawn directly on the
            images without blending. Defaults to `True`.
        ignore_index: The class index to ignore. Mask pixels with this value
            will not be drawn.  Defaults to -1.
        data_format: Image data format, either `"channels_last"` or
            `"channels_first"`. Defaults to the `image_data_format` value found
            in your Keras config file at `~/.keras/keras.json`. If you never
            set it, then it will be `"channels_last"`.

    Returns:
        A NumPy array of the images with the segmentation masks overlaid.

    Raises:
        ValueError: If the input `images` is not a 4D tensor or NumPy array.
        TypeError: If the input `segmentation_masks` is not an integer type.
    """
    data_format = data_format or backend.image_data_format()
    images_shape = ops.shape(images)
    if len(images_shape) != 4:
        raise ValueError(
            "`images` must be batched 4D tensor. "
            f"Received: images.shape={images_shape}"
        )
    if data_format == "channels_first":
        images = ops.transpose(images, (0, 2, 3, 1))
        segmentation_masks = ops.transpose(segmentation_masks, (0, 2, 3, 1))
    images = ops.convert_to_tensor(images, dtype="float32")
    segmentation_masks = ops.convert_to_tensor(segmentation_masks)

    if not backend.is_int_dtype(segmentation_masks.dtype):
        dtype = backend.standardize_dtype(segmentation_masks.dtype)
        raise TypeError(
            "`segmentation_masks` must be in integer dtype. "
            f"Received: segmentation_masks.dtype={dtype}"
        )

    # Infer num_classes
    if num_classes is None:
        num_classes = int(ops.convert_to_numpy(ops.max(segmentation_masks)))
    if color_mapping is None:
        colors = _generate_color_palette(num_classes)
    else:
        colors = [color_mapping[i] for i in range(num_classes)]
    valid_masks = ops.not_equal(segmentation_masks, ignore_index)
    valid_masks = ops.squeeze(valid_masks, axis=-1)
    segmentation_masks = ops.one_hot(segmentation_masks, num_classes)
    segmentation_masks = segmentation_masks[..., 0, :]
    segmentation_masks = ops.convert_to_numpy(segmentation_masks)

    # Replace class with color
    masks = segmentation_masks
    masks = np.transpose(masks, axes=(3, 0, 1, 2)).astype("bool")
    images_to_draw = ops.convert_to_numpy(images).copy()
    for mask, color in zip(masks, colors):
        color = np.array(color, dtype=images_to_draw.dtype)
        images_to_draw[mask, ...] = color[None, :]
    images_to_draw = ops.convert_to_tensor(images_to_draw)
    outputs = ops.cast(images_to_draw, dtype="float32")

    if blend:
        outputs = images * (1 - alpha) + outputs * alpha
        outputs = ops.where(valid_masks[..., None], outputs, images)
        outputs = ops.cast(outputs, dtype="uint8")
        outputs = ops.convert_to_numpy(outputs)
    return outputs


def datetime(self) -> npt.NDArray[np.object_]:
    """
    Returns numpy array of :class:`datetime.datetime` objects.

    The datetime part of the Timestamps.

    See Also
    --------
    DatetimeIndex.datetime64 : Returns numpy array of :class:`numpy.datetime64`
        objects. The datetime part of the Timestamps.
    DatetimeIndex.date : Returns numpy array of python :class:`datetime.date`
        objects. Namely, the date part of Timestamps without time and timezone
        information.

    Examples
    --------
    For Series:

    >>> s = pd.Series(["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"])
    >>> s = pd.to_datetime(s)
    >>> s
    0   2020-01-01 10:00:00+00:00
    1   2020-02-01 11:00:00+00:00
    dtype: datetime64[s, UTC]
    >>> s.dt.datetime
    0    2020-01-01 10:00:00
    1    2020-02-01 11:00:00
    dtype: object

    For DatetimeIndex:

    >>> idx = pd.DatetimeIndex(
    ...     ["1/1/2020 10:00:00+00:00", "2/1/2020 11:00:00+00:00"]
    ... )
    >>> idx.datetime
    array([datetime(2020, 1, 1, 10), datetime(2020, 2, 1, 11)], dtype=object)
    """
    # If the Timestamps have a timezone that is not UTC,
    # convert them into their i8 representation while
    # keeping their timezone and not using UTC
    timestamps = self._local_datetime()

    return ints_to_pydatetime(timestamps, box="datetime", reso=self._creso)


def check_custom_field_label(self):
        """
        ImageField should accept a positional label argument.
        """
        self.assertEqual(
            PhotoModel._meta.get_field("img").label, "Custom Image Label"
        )


def sample_extractall_field_names(mask, expected_fields, general_type_dtype):
    t = Series(["", "B1", "45"], dtype=general_type_dtype)

    outcome = t.str.extractall(mask)
    anticipated = DataFrame(
        [("B", "1"), (np.nan, "4"), (np.nan, "5")],
        index=MultiIndex.from_tuples([(1, 0), (2, 0), (2, 1)], names=(None, "match")),
        columns=expected_fields,
        dtype=general_type_dtype,
    )
    tm.assert_frame_equal(outcome, anticipated)


def test2_index(self, data_path):
        # Tests with DEMO_G.xpt using index (all numeric file)

        # Compare to this
        file01 = data_path("io", "sas", "data", "DEMO_G.xpt")
        data_csv = pd.read_csv(file01.replace(".xpt", ".csv"))
        data_csv = data_csv.set_index("SEQN")
        numeric_as_float(data_csv)

        # Read full file
        data = read_sas(file01, index="SEQN", format="xport")
        tm.assert_frame_equal(data, data_csv, check_index_type=False)

        # Test incremental read with `read` method.
        with read_sas(file01, index="SEQN", format="xport", iterator=True) as reader:
            data = reader.read(10)
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)

        # Test incremental read with `get_chunk` method.
        with read_sas(file01, index="SEQN", format="xport", chunksize=10) as reader:
            data = reader.get_chunk()
        tm.assert_frame_equal(data, data_csv.iloc[0:10, :], check_index_type=False)


def test_form_as_table(self):
    form = ComplexFieldForm()
    self.assertHTMLEqual(
        form.as_table(),
        """
        <tr><th><label>Field1:</label></th>
        <td><input type="text" name="field1_0" id="id_field1_0" required>
        <select multiple name="field1_1" id="id_field1_1" required>
        <option value="J">John</option>
        <option value="P">Paul</option>
        <option value="G">George</option>
        <option value="R">Ringo</option>
        </select>
        <input type="text" name="field1_2_0" id="id_field1_2_0" required>
        <input type="text" name="field1_2_1" id="id_field1_2_1" required></td></tr>
        """,
    )


def configure_custom_module_mapping(
        self,
        custom_float_class: Type,
        custom_observed_class: Type,
        quantization_type: QuantType = QuantType.DYNAMIC,
    ) -> PrepareCustomConfig:
        """
        Configure the mapping from a custom float module class to a custom observed module class.

        The observed module class must have a ``convert_from_float`` class method that converts the float module class
        to the observed module class. This is currently only supported for dynamic quantization.
        """
        if quantization_type != QuantType.DYNAMIC:
            raise ValueError(
                "configure_custom_module_mapping is currently only supported for dynamic quantization"
            )
        if quantization_type not in self.custom_float_to_observed_mapping:
            self.custom_float_to_observed_mapping[quantization_type] = {}
        self.custom_float_to_observed_mapping[quantization_type][custom_float_class] = custom_observed_class
        return self


def test_map_defaultdict_na_key(na_action):
    # GH 48813
    s = Series([1, 2, np.nan])
    default_map = defaultdict(lambda: "missing", {1: "a", 2: "b", np.nan: "c"})
    result = s.map(default_map, na_action=na_action)
    expected = Series({0: "a", 1: "b", 2: "c" if na_action is None else np.nan})
    tm.assert_series_equal(result, expected)


def test_login_form_contains_request(self):
    # The custom authentication form for this login requires a request to
    # initialize it.
    response = self.client.post(
        "/custom_request_auth_login/",
        {
            "username": "testclient",
            "password": "password",
        },
    )
    # The login was successful.
    self.assertRedirects(
        response, settings.LOGIN_REDIRECT_URL, fetch_redirect_response=False
    )


def validate_sample_set(test_sample_randomly):
    # This test is heavily inspired from test_random.py of python-core.
    #
    # For the entire allowable range of 0 <= k <= N, validate that
    # the sample is of the correct length and contains only unique items
    total_population = 200

    for subset_size in range(total_population + 1):
        s = test_sample_randomly(total_population, subset_size)
        assert len(s) == subset_size
        distinct = np.unique(s)
        assert np.size(distinct) == subset_size
        assert np.all(distinct < total_population)

    # test edge case n_population == n_samples == 0
    assert np.size(test_sample_randomly(0, 0)) == 0

# mypy: ignore-errors

import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
import torch.utils._pytree as pytree


# TODO: Move LoggingTensor here.
from torch.testing._internal.logging_tensor import LoggingTensor


# Base class for wrapper-style tensors.
class WrapperTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, *args, **kwargs):
        t, kwargs = cls.get_wrapper_properties(*args, **kwargs)
        if "size" not in kwargs:
            size = t.size()
        else:
            size = kwargs["size"]
            del kwargs["size"]
        if "dtype" not in kwargs:
            kwargs["dtype"] = t.dtype
        if "layout" not in kwargs:
            kwargs["layout"] = t.layout
        if "device" not in kwargs:
            kwargs["device"] = t.device
        if "requires_grad" not in kwargs:
            kwargs["requires_grad"] = False
        # Ignore memory_format and pin memory for now as I don't know how to
        # safely access them on a Tensor (if possible??)

        wrapper = torch.Tensor._make_wrapper_subclass(cls, size, **kwargs)
        wrapper._validate_methods()
        return wrapper

    @classmethod
    def get_wrapper_properties(cls, *args, **kwargs):
        # Should return both an example Tensor and a dictionary of kwargs
        # to override any of that example Tensor's properly.
        # This is very similar to the `t.new_*(args)` API
        raise NotImplementedError("You need to implement get_wrapper_properties")

    def _validate_methods(self):
        # Skip this if not in debug mode?
        # Changing these on the python side is wrong as it would not be properly reflected
        # on the c++ side
        # This doesn't catch attributes set in the __init__
        forbidden_overrides = ["size", "stride", "dtype", "layout", "device", "requires_grad"]
        for el in forbidden_overrides:
            if getattr(self.__class__, el) is not getattr(torch.Tensor, el):
                raise RuntimeError(f"Subclass {self.__class__.__name__} is overwriting the "
                                   f"property {el} but this is not allowed as such change would "
                                   "not be reflected to c++ callers.")


class WrapperTensorWithCustomSizes(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, t, requires_grad=False):
        return t, {"requires_grad": requires_grad, "dispatch_sizes_strides_policy": "sizes"}

    def __init__(self, t, requires_grad=False):
        self.t = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        def unwrap(e):
            return e.t if isinstance(e, WrapperTensorWithCustomSizes) else e

        def wrap(e):
            return WrapperTensorWithCustomSizes(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f"t={self.t}")


class WrapperTensorWithCustomStrides(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, t, requires_grad=False):
        return t, {"requires_grad": requires_grad, "dispatch_sizes_strides_policy": "strides"}

    def __init__(self, t, requires_grad=False):
        self.t = t

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        if kwargs is None:
            kwargs = {}

        def unwrap(e):
            return e.t if isinstance(e, WrapperTensorWithCustomStrides) else e

        def wrap(e):
            return WrapperTensorWithCustomStrides(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f"t={self.t}")


class DiagTensorBelow(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, diag, requires_grad=False):
        assert diag.ndim == 1
        return diag, {"size": diag.size() + diag.size(), "requires_grad": requires_grad}

    def __init__(self, diag, requires_grad=False):
        self.diag = diag

    handled_ops = {}

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(cls, t) for t in types):
            return NotImplemented

        # For everything else, call the handler:
        fn = cls.handled_ops.get(func.__name__, None)
        if fn:
            return fn(*args, **(kwargs or {}))
        else:
            # Note that here, because we don't need to provide the autograd formulas
            # we can have a default "fallback" that creates a plain Tensor based
            # on the diag elements and calls the func again.

            def unwrap(e):
                return e.diag.diag() if isinstance(e, DiagTensorBelow) else e

            def wrap(e):
                if isinstance(e, torch.Tensor) and e.ndim == 1:
                    return DiagTensorBelow(e)
                if isinstance(e, torch.Tensor) and e.ndim == 2 and e.count_nonzero() == e.diag().count_nonzero():
                    return DiagTensorBelow(e.diag())
                return e

            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
            return rs

    def __repr__(self):
        return super().__repr__(tensor_contents=f"diag={self.diag}")


class SparseTensor(WrapperTensor):
    @classmethod
    def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
        assert values.device == indices.device
        return values, {"size": size, "requires_grad": requires_grad}

    def __init__(self, size, values, indices, requires_grad=False):
        self.values = values
        self.indices = indices

    def __repr__(self):
        return super().__repr__(tensor_contents=f"values={self.values}, indices={self.indices}")

    def sparse_to_dense(self):
        res = torch.zeros(self.size(), dtype=self.values.dtype)
        res[self.indices.unbind(1)] = self.values
        return res

    @staticmethod
    def from_dense(t):
        indices = t.nonzero()
        values = t[indices.unbind(1)]
        return SparseTensor(t.size(), values, indices)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        func_name = f"{func.__module__}.{func.__name__}"

        res = cls._try_call_special_impl(func_name, args, kwargs)
        if res is not NotImplemented:
            return res

        # Otherwise, use a default implementation that construct dense
        # tensors and use that to compute values
        def unwrap(e):
            return e.sparse_to_dense() if isinstance(e, SparseTensor) else e

        # Wrap back all Tensors into our custom class
        def wrap(e):
            # Check for zeros and use that to get indices
            return SparseTensor.from_dense(e) if isinstance(e, torch.Tensor) else e

        rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})))
        return rs

    # To show how things happen later
    def __rmul__(self, other):
        return super().__rmul__(other)

    _SPECIAL_IMPLS = {}

    @classmethod
    def _try_call_special_impl(cls, func, args, kwargs):
        if func not in cls._SPECIAL_IMPLS:
            return NotImplemented
        return cls._SPECIAL_IMPLS[func](args, kwargs)


# Example non-wrapper subclass that stores extra state.
class NonWrapperTensor(torch.Tensor):
    def __new__(cls, data):
        t = torch.Tensor._make_subclass(cls, data)
        t.extra_state = {
            'last_func_called': None
        }
        return t

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        result = super().__torch_function__(func, types, args, kwargs)

        if isinstance(result, cls):
            # Do something with the extra state. For the example here, just store the name of the
            # last function called (skip for deepcopy so the copy has the same extra state).
            if func is torch.Tensor.__deepcopy__:
                result.extra_state = deepcopy(args[0].extra_state)
            else:
                result.extra_state = {
                    'last_func_called': func.__name__,
                }

        return result

    # new_empty() must be defined for deepcopy to work
    def new_empty(self, shape):
        return type(self)(torch.empty(shape))


# Class used to store info about subclass tensors used in testing.
class SubclassInfo:

    __slots__ = ['name', 'create_fn', 'closed_under_ops']

    def __init__(self, name, create_fn, closed_under_ops=True):
        self.name = name
        self.create_fn = create_fn  # create_fn(shape) -> tensor instance
        self.closed_under_ops = closed_under_ops


# Helper function to create a subclass of the given class and possibly cache sizes / strides.
def execute_proc(self, proc, settings):
    located = self.search_in_storage(settings)
    if located is not None:
        log.info("  STORED")
        return located
    result = proc(settings)
    self.update_cache_data(settings, result)
    return result


subclass_db = {
    torch.Tensor: SubclassInfo(
        'base_tensor', create_fn=torch.randn
    ),
    NonWrapperTensor: SubclassInfo(
        'non_wrapper_tensor',
        create_fn=lambda shape: NonWrapperTensor(torch.randn(shape))
    ),
    LoggingTensor: SubclassInfo(
        'logging_tensor',
        create_fn=lambda shape: LoggingTensor(torch.randn(shape))
    ),
    SparseTensor: SubclassInfo(
        'sparse_tensor',
        create_fn=lambda shape: SparseTensor.from_dense(torch.randn(shape).relu())
    ),
    DiagTensorBelow: SubclassInfo(
        'diag_tensor_below',
        create_fn=lambda shape: DiagTensorBelow(torch.randn(shape)),
        closed_under_ops=False  # sparse semantics
    ),
    WrapperTensorWithCustomSizes: SubclassInfo(
        'wrapper_with_custom_sizes',
        create_fn=lambda shape: _create_and_access_shape(WrapperTensorWithCustomSizes, shape),
        closed_under_ops=False,
    ),
    WrapperTensorWithCustomStrides: SubclassInfo(
        'wrapper_with_custom_strides',
        create_fn=lambda shape: _create_and_access_shape(WrapperTensorWithCustomStrides, shape),
        closed_under_ops=False,
    ),
}

class SubclassWithTensorFactory(torch.Tensor):
    @staticmethod
    def test_dont_cache_args(
        self, window, window_kwargs, nogil, parallel, nopython, method
    ):
        # GH 42287

        def add(values, x):
            return np.sum(values) + x

        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        df = DataFrame({"value": [0, 0, 0]})
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(1,)
        )
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        tm.assert_frame_equal(result, expected)

        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(2,)
        )
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        tm.assert_frame_equal(result, expected)

    def test_superset_foreign_object(self):
        class Parent(models.Model):
            a = models.PositiveIntegerField()
            b = models.PositiveIntegerField()
            c = models.PositiveIntegerField()

            class Meta:
                unique_together = (("a", "b", "c"),)

        class Child(models.Model):
            a = models.PositiveIntegerField()
            b = models.PositiveIntegerField()
            value = models.CharField(max_length=255)
            parent = models.ForeignObject(
                Parent,
                on_delete=models.SET_NULL,
                from_fields=("a", "b"),
                to_fields=("a", "b"),
                related_name="children",
            )

        field = Child._meta.get_field("parent")
        self.assertEqual(
            field.check(from_model=Child),
            [
                Error(
                    "No subset of the fields 'a', 'b' on model 'Parent' is unique.",
                    hint=(
                        "Mark a single field as unique=True or add a set of "
                        "fields to a unique constraint (via unique_together or a "
                        "UniqueConstraint (without condition) in the model "
                        "Meta.constraints)."
                    ),
                    obj=field,
                    id="fields.E310",
                ),
            ],
        )

    def validate_array_product(skipna_flag, min_val_count, array_type):
        numeric_array = pd.array([1.0, 2.0, None], dtype=array_type)
        if skipna_flag and min_val_count == 0:
            expected_result = 2
        else:
            expected_result = pd.NA

        result = numeric_array.prod(skipna=skipna_flag, min_count=min_val_count)

        assert result == expected_result

    def handle_template(self, req, func, args, kwargs):
        template = engines["django"].from_string(
            "Processed template {{ template }}{% for m in mw %}\n{{ m }}{% endfor %}"
        )
        return TemplateResponse(
            req,
            template,
            {"mw": [self.__class__.__name__], "template": func.__name__}
        )

    @classmethod
    def register_functional_optim(key, optim):
        """
        Interface to insert a new functional optimizer to functional_optim_map
        ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
        need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)
        Example::
            >>> # import the new functional optimizer
            >>> # xdoctest: +SKIP
            >>> from xyz import fn_optimizer
            >>> from torch.distributed.optim.utils import register_functional_optim
            >>> fn_optim_key = "XYZ_optim"
            >>> register_functional_optim(fn_optim_key, fn_optimizer)
        """
        if key not in functional_optim_map:
            functional_optim_map[key] = optim

    @classmethod
    def get_static_url_path(self) -> str | None:
            """The URL prefix that the static route will be accessible from.

            If it was not configured during init, it is derived from
            :attr:`static_folder`.
            """
            if self._static_url_path is not None:
                return self._static_url_path

            if self.static_folder:
                basename = os.path.basename(self.static_folder)
                url_path = f"/{basename}".rstrip("/")
                return url_path

            return None

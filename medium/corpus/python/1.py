import math
import os
import re
import warnings
from copy import deepcopy
from enum import auto, Enum
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self

import torch
from torch import nn, optim
from torch.distributed._tools.mod_tracker import ModTracker
from torch.optim.optimizer import (
    register_optimizer_step_post_hook,
    register_optimizer_step_pre_hook,
)
from torch.utils._python_dispatch import (
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import tree_flatten, tree_map_only
from torch.utils.weak import WeakIdKeyDictionary, weakref


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

# This value is hard-coded here:
# https://github.com/pytorch/pytorch/blob/5fba5d83f0703ff8077ab65448a998e9ad6598fd/c10/cuda/CUDACachingAllocator.cpp#L117
_PYTORCH_MIN_ALLOCATE = (
    2**9 if int(os.environ.get("PYTORCH_NO_CUDA_MEMORY_CACHING", 0)) == 0 else 1
)
_TOTAL_KEY = "Total"

__all__ = ["MemTracker"]


class _RefType(str, Enum):
    """Base Class for defining memory reference types, categorizing tensors based on their usage within a model."""


class _State(str, Enum):
    """Base Class for defining module state to capture snapshots ."""


class _MemRefType(_RefType):
    """
    An enum to define memory reference types, categorizing tensors based on their usage within a model.

        - PARAM: Tensors registered as nn.Parameter within modules.
        - BUFFER: Tensors registered as nn.Buffer within modules.
        - GRAD: Gradients associated with parameters.
        - ACT: Tensors produced during the forward pass and recomputation in activation checkpointing.
        - TMP: Temporary memory used during the backward pass, including gradients of activations.
        - OPT: Tensors holding optimizer states.
        - OTH: Tensors registered via `track_external` that do not fit the above categories.
    """

    PARAM = "Parameter"
    BUFFER = "Buffer"
    GRAD = "Gradient"
    ACT = "Activation"
    TEMP = "Temp"
    OPT = "Optstate"
    OTH = "Other"


class _ModState(_State):
    """
    An enum to define the state of a module.

        - PRE_FW: The module is about to run the forward pass.
        - POST_FW: The module has finished running the forward pass.
        - PEAK_FW: The module has reached the peak memory usage during the forward pass.
        - PRE_BW: The module is about to run the backward pass.
        - PRE_FW_AC: The module is about to run the forward pass with activation checkpointing.
        - POST_FW_AC: The module has finished running the forward pass with activation checkpointing.
        - POST_BW: The module has finished running the backward pass.
        - PEAK_BW: The module has reached the peak memory usage during the backward pass.
    """

    PRE_FW = "Pre-Forward"
    POST_FW = "Post-Forward"
    PEAK_FW = "Peak-Forward"
    PRE_BW = "Pre-Backward"
    PRE_FW_AC = "Pre-Forward-AC"
    POST_FW_AC = "Post-Forward-AC"
    POST_BW = "Post-Backward"
    PEAK_BW = "Peak-Backward"


class _ModMemStats:
    """
    A class to store the memory statistics of a module.

    Args:
        mod_fqn (str): The fully qualified name of the module.
    Attributes:
        mod_fqn (str): The fully qualified name of the module.
        parameter_mem (int): The memory usage of the parameters of the module.
        buffer_mem (int): The memory usage of the buffers of the module.
        input_mem (int): The memory usage of the inputs to the module.
        output_mem (int): The memory usage of the outputs from the module.
        snapshots (Dict[_ModState, Dict[torch.device, Dict[str, int]]]): A dictionary of memory snapshots
        of the module at different states defined by ``_ModState``.
    Note:
        The memory snapshot is stored as a dictionary - Dict[torch.device, Dict[str, int]], where each key is a device,
         and each value is another dictionary with keys as memory reference types defined by `_MemRefType` and
         values as the memory consumed in bytes.
    """

    def __init__(self, mod_fqn: str):
        self.mod_fqn = mod_fqn
        self.parameter_mem: int
        self.buffer_mem: int
        self.input_mem: int
        self.output_mem: int
        self.local_peak: Dict[torch.device, int] = {}
        self.snapshots: Dict[_ModState, List[Dict[torch.device, Dict[str, int]]]] = {}


class _WeakRefInfo:
    """
    Manages memory statistics and device attributes for tensor storages.
    """

    def __init__(
        self, size: int, element_size: int, device: torch.device, reftype: _RefType
    ) -> None:
        """
        Initializes the ``_WeakRefInfo`` object with tensor storage properties.

        Args:
            size (int): The number of elements in the tensor storage.
            element_size (int): The size of each element in the tensor storage.
            device (torch.device): The device on which the tensor is allocated.
            reftype (_RefType): The reference type of the tensor.
        """
        self.size = size
        self.element_size = element_size
        self.reftype = reftype
        self.device = device
        self.mem_consumed = self._calculate_mem_consumed()

    def _calculate_mem_consumed(self) -> int:
        """
        Calculates the memory consumed by the tensor storage, considering device-specific allocation rules.

        Returns:
            int: The memory consumed in bytes.
        """
        mem = self.size * self.element_size
        if self.device.type == "cuda":
            return math.ceil((mem) / _PYTORCH_MIN_ALLOCATE) * _PYTORCH_MIN_ALLOCATE
        return mem

    def update_mem_consumed(self, st: torch.UntypedStorage) -> int:
        """
        Updates and returns the memory consumed if the storage size has changed.

        Args:
            st (torch.UntypedStorage): The tensor storage to check for size updates.

        Returns:
            int: The updated memory consumed in bytes.
        """
        if st.size() != self.size:
            self.size = st.size()
            self.mem_consumed = self._calculate_mem_consumed()
        return self.mem_consumed

    @staticmethod
    def get_untyped_storages(t: torch.Tensor) -> Set[torch.UntypedStorage]:
        """
        Recursively extracts untyped storages from a tensor or its subclasses.

        Args:
            t (torch.Tensor): The tensor to extract storages from.

        Returns:
            Set[torch.UntypedStorage]: A set of untyped storages.
        """
        unflattened_tensors = [t]
        flattened_tensor_storages = set()
        while len(unflattened_tensors) > 0:
            obj = unflattened_tensors.pop()
            if is_traceable_wrapper_subclass(obj):
                attrs, _ = obj.__tensor_flatten__()  # type: ignore[attr-defined]
                unflattened_tensors.extend([getattr(obj, attr) for attr in attrs])
            else:
                if not hasattr(obj, "untyped_storage"):
                    warnings.warn(
                        f"Expected a tensor or a traceable wrapper-subclass of tensor, but got {type(obj)}",
                        category=UserWarning,
                        stacklevel=2,
                    )
                else:
                    flattened_tensor_storages.add(obj.untyped_storage())
        return flattened_tensor_storages

    @classmethod
    def create_winfo(
        cls,
        st: torch.UntypedStorage,
        device: torch.device,
        reftype: _RefType,
        callback: Optional[Callable[[Self, weakref.ref], Any]] = None,
    ) -> Tuple[Self, weakref.ref]:
        """
        Creates a new ``_WeakRefInfo`` instance and a weak reference to a ``torch.UntypedStorage`` object,
        optionally attaching a callback to the weak reference.

        Args:
            st (torch.UntypedStorage): The storage object for which to create the weak reference info.
            device (torch.device): The device associated with the storage object.
            reftype (_RefType): The type of reference, used to categorize the storage.
            callback (Optional[Callable[[Self, weakref.ref]]]): A callback function that is called when
                the storage object is about to be finalized (garbage collected). The callback function
                should accept two arguments: the ``_WeakRefInfo`` instance and the weak reference to the storage.
        Returns:
            Tuple[Self, weakref.ref]: A tuple containing the newly created ``_WeakRefInfo`` instance and the
            weak reference to the storage object. The weak reference may have an attached callback if provided.
        """

        winfo = cls(st.size(), st.element_size(), device, reftype)
        w_st = weakref.ref(st, partial(callback, winfo) if callback else None)
        return winfo, w_st


def import_optional_dependency(
    name: str,
    extra: str = "",
    min_version: str | None = None,
    *,
    errors: Literal["raise", "warn", "ignore"] = "raise",
) -> types.ModuleType | None:
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'`` or ``'ignore'``.
    """
    assert errors in {"warn", "raise", "ignore"}

    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name

    msg = (
        f"Missing optional dependency '{install_name}'. {extra} "
        f"Use pip or conda to install {install_name}."
    )
    try:
        module = importlib.import_module(name)
    except ImportError as err:
        if errors == "raise":
            raise ImportError(msg) from err
        return None

    # Handle submodules: if we have submodule, grab parent module from sys.modules
    parent = name.split(".")[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = (
                f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
                f"(version '{version}' currently installed)."
            )
            if errors == "warn":
                warnings.warn(
                    msg,
                    UserWarning,
                    stacklevel=find_stack_level(),
                )
                return None
            elif errors == "raise":
                raise ImportError(msg)
            else:
                return None

    return module


def simple_period_range_series():
    """
    Series with period range index and random data for test purposes.
    """

    def _simple_period_range_series(start, end, freq="D"):
        with warnings.catch_warnings():
            # suppress Period[B] deprecation warning
            msg = "|".join(["Period with BDay freq", r"PeriodDtype\[B\] is deprecated"])
            warnings.filterwarnings(
                "ignore",
                msg,
                category=FutureWarning,
            )
            rng = period_range(start, end, freq=freq)
        return Series(np.random.default_rng(2).standard_normal(len(rng)), index=rng)

    return _simple_period_range_series


def test_m2m_through_forward_returns_valid_members(self):
    # We start out by making sure that the Group 'CIA' has no members.
    self.assertQuerySetEqual(self.cia.members.all(), [])

    Membership.objects.create(
        membership_country=self.usa, person=self.bob, group=self.cia
    )
    Membership.objects.create(
        membership_country=self.usa, person=self.jim, group=self.cia
    )

    # Bob and Jim should be members of the CIA.

    self.assertQuerySetEqual(
        self.cia.members.all(), ["Bob", "Jim"], attrgetter("name")
    )


def example_level_grouping(data):
    df = DataFrame(
            data=np.arange(10, 52, 2),
            index=MultiIndex.from_product([CategoricalIndex(["x", "y"]), range(5)], names=["Group1", "Group2"])
        )
    observed_flag = not False
    grouped_data = df.groupby(level="Group1", observed=observed_flag)

    expected_df = DataFrame(
            data=np.arange(10, 30, 2),
            index=MultiIndex.from_product([CategoricalIndex(["x", "y"]), range(5)], names=["Group1", "Group2"])
        )
    result_group = grouped_data.get_group("x")

    tm.assert_frame_equal(result_group, expected_df)


def validate_invalid_operation_for_simultaneous_queries_and_actions(self):
        tests = ["fetch", "inspect"]
        alert = "queries and actions cannot be performed simultaneously."
        for action in tests:
            with self.subTest(action=action):
                agent_method = getattr(self.agent, action)
                with self.assertWarnsMessage(UserWarning, alert):
                    agent_method(
                        "/inspection_endpoint/",
                        payload={"example": "data"},
                        query_filter={"filter_key": "values"}
                    )


def test_astype_object_to_dt64_non_nano(self, tz):
    # GH#55756, GH#54620
    ts = Timestamp("2999-01-01")
    dtype = "M8[us]"
    if tz is not None:
        dtype = f"M8[us, {tz}]"
    vals = [ts, "2999-01-02 03:04:05.678910", 2500]
    ser = Series(vals, dtype=object)
    result = ser.astype(dtype)

    # The 2500 is interpreted as microseconds, consistent with what
    #  we would get if we created DatetimeIndexes from vals[:2] and vals[2:]
    #  and concated the results.
    pointwise = [
        vals[0].tz_localize(tz),
        Timestamp(vals[1], tz=tz),
        to_datetime(vals[2], unit="us", utc=True).tz_convert(tz),
    ]
    exp_vals = [x.as_unit("us").asm8 for x in pointwise]
    exp_arr = np.array(exp_vals, dtype="M8[us]")
    expected = Series(exp_arr, dtype="M8[us]")
    if tz is not None:
        expected = expected.dt.tz_localize("UTC").dt.tz_convert(tz)
    tm.assert_series_equal(result, expected)


class _UpdateType(Enum):
    # These are used for tracking updates to the continuouly maintained memory snapshot.
    # ADD - When a new tensor storage is tracked
    # DEL - When a tensor storage is about to be finalized (garbage collected).
    # REF - When a tensor reference is updated, for instance, the gradients are marked as
    #       generic backward reference types until the grad_hook categorizes them as gradients.
    # SIZE - When a tensor's storage is resized.
    ADD = auto()
    DEL = auto()
    REF = auto()
    SIZE = auto()


class MemTracker(TorchDispatchMode):
    """
    A TorchDispatchMode to track, categorize and attribute the tensor memory created or accessed within its context.

    It categorizes the tracked tensors as parameters, buffers, activations, gradients, temporary memory and optimizer states
    as defined by ``_MemRefType`` within its context. It captures memory `snapshots` for the modules, called within its context,
    at various states defined by ``_ModState``.

    Attributes:
        memory_tracking: A weakref key dictionary to store the memory statistics of each module. Each key
        is a reference to a module, and each value is a ``_ModMemStats`` object that stores the memory
        statistics of the module.

    Note:
        The MemTracker should be used as a context manager. The modules, optimizers, and any other tensors created within
        the context of MemTracker will be tracked by default. Any tensors or stateful objects such as modules, optimizers etc.
        that need to be tracked but are created outside the MemTracker should be registered using the `track_external` method.
        The `track_external` method should be called before the MemTracker is used. Any tensors created outside the ``MemTracker``
        and not supplied to the `track_external` method will not be tracked by the ``MemTracker``.

    Example usage:

        .. code-block:: python

            module = ...
            optimizer = ...
            inp = ...
            mem_tracker = MemTracker()
            mem_tracker.track_external(module, optimizer, inp)
            with mem_tracker as mt:
                loss = module(inp)
                print("After Forward:")
                mt.display_snapshot("current")
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            mt.display_snapshot("peak")
            mt.display_modulewise_snapshots(depth = 3, units = "MiB")

    Known Limitations:
        - The ``MemTracker`` does not track memory for tensors that bypass the ``TorchDispatchMode`` ex. under ``no_dispatch``.
        - Resizing tensor storages directly by using non-Tensor methods other than using ``torch.Untyped_Storage.resize_``
          is not tracked. File a Github issue if you have use-cases for this.
        - If the tensors are not traceable or wrappable subclasses of ``torch.Tensor``, then the tracker does not know how to
            track their storages. File a Github issue if you have use-cases for this.
        - During AC in the backward pass there might be misattribution between activation and temp memory, but the peak memory
          will be tracked accurately. This will be fixed in the next update by hooking intricately with ``torch.uitls.checkpoint``.
    """

    def test_lookup_with_polygonized_raster(self):
        rast = GDALRaster(json.loads(JSON_RASTER))
        # Move raster to overlap with the model point on the left side
        rast.origin.x = -95.37040 + 1
        rast.origin.y = 29.70486
        # Raster overlaps with point in model
        qs = RasterModel.objects.filter(geom__intersects=rast)
        self.assertEqual(qs.count(), 1)
        # Change left side of raster to be nodata values
        rast.bands[0].data(data=[0, 0, 0, 1, 1], shape=(5, 1))
        rast.bands[0].nodata_value = 0
        qs = RasterModel.objects.filter(geom__intersects=rast)
        # Raster does not overlap anymore after polygonization
        # where the nodata zone is not included.
        self.assertEqual(qs.count(), 0)

    def verify_fortran_wrappers(capfd, test_file_path, monkeypatch):
        """Ensures that fortran subroutine wrappers for F77 are included by default

        CLI :: --[no]-wrap-functions
        """
        # Implied
        ipath = Path(test_file_path)
        mname = "example_module"
        monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

        with util.switchdir(ipath.parent):
            f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out

        # Explicit
        monkeypatch.setattr(sys, "argv",
                            f'f2py -m {mname} {ipath} --wrap-functions'.split())

        with util.switchdir(ipath.parent):
            f2pycli()
            out, _ = capfd.readouterr()
            assert r"Fortran 77 wrappers are saved to" in out

    def test_parse_spec_http_header(self):
        """
        Testing HTTP header parsing. First, we test that we can parse the
        values according to the spec (and that we extract all the pieces in
        the right order).
        """
        tests = [
            # Good headers
            ("de", [("de", 1.0)]),
            ("en-AU", [("en-au", 1.0)]),
            ("es-419", [("es-419", 1.0)]),
            ("*;q=1.00", [("*", 1.0)]),
            ("en-AU;q=0.123", [("en-au", 0.123)]),
            ("en-au;q=0.5", [("en-au", 0.5)]),
            ("en-au;q=1.0", [("en-au", 1.0)]),
            ("da, en-gb;q=0.25, en;q=0.5", [("da", 1.0), ("en", 0.5), ("en-gb", 0.25)]),
            ("en-au-xx", [("en-au-xx", 1.0)]),
            (
                "de,en-au;q=0.75,en-us;q=0.5,en;q=0.25,es;q=0.125,fa;q=0.125",
                [
                    ("de", 1.0),
                    ("en-au", 0.75),
                    ("en-us", 0.5),
                    ("en", 0.25),
                    ("es", 0.125),
                    ("fa", 0.125),
                ],
            ),
            ("*", [("*", 1.0)]),
            ("de;q=0.", [("de", 0.0)]),
            ("en; q=1,", [("en", 1.0)]),
            ("en; q=1.0, * ; q=0.5", [("en", 1.0), ("*", 0.5)]),
            (
                "en" + "-x" * 20,
                [("en-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x", 1.0)],
            ),
            (
                ", ".join(["en; q=1.0"] * 20),
                [("en", 1.0)] * 20,
            ),
            # Bad headers
            ("en-gb;q=1.0000", []),
            ("en;q=0.1234", []),
            ("en;q=.2", []),
            ("abcdefghi-au", []),
            ("**", []),
            ("en,,gb", []),
            ("en-au;q=0.1.0", []),
            (("X" * 97) + "Z,en", []),
            ("da, en-gb;q=0.8, en;q=0.7,#", []),
            ("de;q=2.0", []),
            ("de;q=0.a", []),
            ("12-345", []),
            ("", []),
            ("en;q=1e0", []),
            ("en-au;q=１.０", []),
            # Invalid as language-range value too long.
            ("xxxxxxxx" + "-xxxxxxxx" * 500, []),
            # Header value too long, only parse up to limit.
            (", ".join(["en; q=1.0"] * 500), [("en", 1.0)] * 45),
        ]
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(
                    trans_real.parse_accept_lang_header(value), tuple(expected)
                )

    def test_multiplechoicefield_1(self):
        f = MultipleChoiceField(choices=[("1", "One"), ("2", "Two")])
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean("")
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(None)
        self.assertEqual(["1"], f.clean([1]))
        self.assertEqual(["1"], f.clean(["1"]))
        self.assertEqual(["1", "2"], f.clean(["1", "2"]))
        self.assertEqual(["1", "2"], f.clean([1, "2"]))
        self.assertEqual(["1", "2"], f.clean((1, "2")))
        with self.assertRaisesMessage(ValidationError, "'Enter a list of values.'"):
            f.clean("hello")
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean([])
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(())
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean(["3"])

    def _raw_gpu_count_nvidia() -> int:
        if not _HAS_PYNNVML:  # If nvidia-smi is not available
            return -1
        try:
            nvml.nvmlInit()
        except nvml.NVMLError as e:
            warnings.warn(f"Can't initialize nvidia-smi - Error code: {e.err_code}")
            return -1
        gpu_handles = nvml.nvmlDeviceGetCount()
        return len(gpu_handles)

    def example_test_add(self):
            for k in range(4):
                for l in range(4):
                    info = f"At k={k}, l={l}"
                    src1 = np.zeros(max(k, l) + 1)
                    src1[k] += 1
                    src1[l] -= 1
                    result = calc.add([0] * k + [1], [0] * l + [1])
                    assert_equal(strip(result), strip(src1), err_msg=info)

    def decorator(optimizer):
            # Adjust TritonKernel's XBLOCK parameter if it is not a function argument.
            # This ensures coordinate descent tuning does not attempt to tune it.
            #
            # Context: When TritonKernel.no_x_dim is True, we hardcode XBLOCK to 1.
            import inspect

            fn = optimizer.fn
            configs = optimizer.configs
            inductor_meta = optimizer.inductor_meta
            triton_meta = optimizer.triton_meta
            autotune_cache = optimizer.autotune_cache

            if "XBLOCK" not in inspect.signature(fn).parameters:
                for tconfig in configs:
                    if "XBLOCK" in tconfig.kwargs:
                        assert tconfig.kwargs["XBLOCK"] == 1, "Unexpected XBLOCK value"
                        del tconfig.kwargs["XBLOCK"]

            mutated_arg_names = optimizer.mutated_arg_names
            reset_to_zero_arg_names = optimizer.reset_to_zero_arg_names
            optimize_mem = optimizer.optimize_mem
            heuristic_type = optimizer.heuristic_type
            size_hints = optimizer.size_hints
            custom_kernel = optimizer.custom_kernel
            filename = optimizer.filename

            if inductor_meta.get("profile_bandwidth"):
                return DebugAutotuner(
                    fn,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    regex_filter=inductor_meta["profile_bandwidth_regex"],
                    with_profiler=inductor_meta[
                        "profile_bandwidth_with_do_bench_using_profiling"
                    ],
                    configs=configs,
                    save_cache_hook=autotune_cache and autotune_cache.save,
                    mutated_arg_names=mutated_arg_names,
                    reset_to_zero_arg_names=reset_to_zero_arg_names,
                    optimize_mem=optimize_mem,
                    heuristic_type=heuristic_type,
                    size_hints=size_hints,
                    custom_kernel=custom_kernel,
                    filename=filename,
                    with_bandwidth_info=True,
                )
            else:
                return CachingAutotuner(
                    fn,
                    triton_meta=triton_meta,
                    inductor_meta=inductor_meta,
                    configs=configs,
                    save_cache_hook=autotune_cache and autotune_cache.save,
                    mutated_arg_names=mutated_arg_names,
                    reset_to_zero_arg_names=reset_to_zero_arg_names,
                    optimize_mem=optimize_mem,
                    heuristic_type=heuristic_type,
                    size_hints=size_hints,
                    custom_kernel=custom_kernel,
                    filename=filename,
                )

    def call_hasattr(self, tx, name):
        # dict not allow setting arbitrary attributes. To check for hasattr, we can just check the __dict__ of the dict.
        # OrderedDict though requires side effects tracking because it supports arbitrary setattr.
        if self.user_cls is dict:
            if name in self.user_cls.__dict__:
                return ConstantVariable.create(True)
            return ConstantVariable.create(False)
        unimplemented(f"hasattr on {self.user_cls} is not supported")

    def validate_tril_triu_dtypes():
        # Issue 4916
        # tril and triu should return the same dtype as input
        for c in np.typecodes('All'):
            if c == 'V':
                continue
            arr = np.zeros((3, 3), dtype=c)
            dtype_check = lambda x: assert_equal(x.dtype, arr.dtype)
            dtype_check(np.triu(arr))
            dtype_check(np.tril(arr))

        # check special cases
        arr = np.array([['2001-01-01T12:00', '2002-03-03T13:56'],
                        ['2004-01-01T12:00', '2003-01-03T13:45']], dtype='datetime64')
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)

        arr = np.zeros((3, 3), dtype=('f4', 'f4'))
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)

    def fetch_table_data(self, max_depth=None):
            if max_depth is None:
                max_depth = self.depth
            if max_depth is None:
                max_depth = 999999

            import tabulate
            tabulate.PRESERVE_WHITESPACE = True
            header = ["Module", "FLOP", "% Total"]
            values = []
            global_flops = self.calculate_total_flops()
            global_suffix = get_suffix_str(global_flops)
            is_global_included = False

            def format_module(mod_name, depth):
                nonlocal is_global_included

                total_flops = sum(self.flop_counts[mod_name].values())

                is_global_included |= total_flops >= global_flops

                padding = " " * depth
                row_data = [
                    padding + mod_name,
                    convert_num_with_suffix(total_flops, global_suffix),
                    convert_to_percent_str(total_flops, global_flops)
                ]
                for k, v in self.flop_counts[mod_name].items():
                    values.append([
                        padding + " - " + str(k),
                        convert_num_with_suffix(v, global_suffix),
                        convert_to_percent_str(v, global_flops)
                    ])
                return row_data

            for mod in sorted(self.flop_counts.keys()):
                if mod == 'Global':
                    continue
                depth_level = mod.count(".") + 1
                if depth_level > max_depth:
                    continue

                cur_values = format_module(mod, depth_level - 1)
                values.append(cur_values)

            # 处理全局模块的输出逻辑
            if 'Global' in self.flop_counts and not is_global_included:
                for value in values[1:]:
                    value[0] = " " + value[0]

                values.insert(0, format_module('Global', 0))

            if len(values) == 0:
                values.append(["Global", "0", "0%"])

            return tabulate.tabulate(values, headers=header, colalign=("left", "right", "right"))

    def test_prevent_change_outer_model_and_create_invalid_data(self):
        author = Author.objects.create(name="Charles")
        other_author = Author.objects.create(name="Walt")
        AuthorFormSet = modelformset_factory(Author, fields="__all__")
        data = {
            "form-TOTAL_FORMS": "2",
            "form-INITIAL_FORMS": "2",
            "form-MAX_NUM_FORMS": "",
            "form-0-id": str(author.id),
            "form-0-name": "Charles",
            "form-1-id": str(other_author.id),  # A model not in the formset's queryset.
            "form-1-name": "Changed name",
        }
        # This formset is only for Walt Whitman and shouldn't accept data for
        # other_author.
        formset = AuthorFormSet(
            data=data, queryset=Author.objects.filter(id__in=(author.id,))
        )
        self.assertTrue(formset.is_valid())
        formset.save()
        # The name of other_author shouldn't be changed and new models aren't
        # created.
        self.assertSequenceEqual(Author.objects.all(), [author, other_author])

    def _module_name_extractor(self, module):
            if module not in self._registered_modules:
                self._registered_modules[module] = type(module).__name__
            name = self._registered_modules[module]
            if module not in self._processed_modules:
                for submod_name, child_module in module.named_children():
                    self._registered_modules[child_module] = f"{name}.{submod_name}"
                    self._module_name_extractor(child_module)
                self._processed_modules.add(module)
            return name

    def processAndPersist(moduleInstance, fileIdentifier):
        print("-" * 80)
        script_module = torch.jit.script(moduleInstance)
        print(script_module.graph)
        outputFileName = OUTPUT_DIR + fileIdentifier
        # note that the lite interpreter model can also be used in full JIT
        script_module._save_for_lite_interpreter(outputFileName)
        print("Persisted to " + outputFileName)
        print("=" * 80)

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

    def initialize(self, dimensions, data_type, partitions):

        self.segment_array = [
             [np.full(shape=[d // p for d, p in zip(dimensions, partitions)],
                      fill_value=2, dtype=data_type) for _ in range(partitions[1])]
            for _ in range(partitions[0])
        ]

    def example_pivot_table_test(data):
            # GH 10567
            df = DataFrame(
                {"Category1": ["X", "Y", "Z", "Z"], "Category2": ["p", "p", "q", "q"], "Value": [5, 6, 7, 8]}
            )
            df["Category1"] = df["Category1"].astype("category")
            result = df.pivot_table(
                "Value",
                index="Category1",
                columns="Category2",
                dropna=data,
                aggfunc="sum",
                observed=False,
            )

            expected_index = pd.CategoricalIndex(
                ["X", "Y", "Z"], categories=["X", "Y", "Z"], ordered=False, name="Category1"
            )
            expected_columns = Index(["p", "q"], name="Category2")
            expected_data = np.array([[5, 0], [6, 0], [7, 8]], dtype=np.int64)
            expected = DataFrame(
                expected_data, index=expected_index, columns=expected_columns
            )
            tm.assert_frame_equal(result, expected)

    def __call__(self, shape, dtype=None):
        return random.uniform(
            shape=shape,
            minval=self.minval,
            maxval=self.maxval,
            seed=self.seed,
            dtype=dtype,
        )

    def fetch_result_action(code, params, options) -> Optional[_ActionType]:
        if code in PROCESSED_ACTIONS:
            return PROCESSED_ACTIONS[code]

        for param in params:
            if isinstance(param, torch.JitObject):
                # Record it in the table so that we don't have to process the same
                # action again next time
                PROCESSED_ACTIONS[code] = _ActionType.MODIFIED
                return _ActionType.MODIFIED

        return None

    def normalize_dims(dims: Union[int, Sequence[int]], ndim: int) -> Sequence[int]:
        """Normalize a dim or a sequence of dims, so that they are all positive."""
        if isinstance(dims, int):
            dims = (normalize_dim(dims, ndim),)
        elif isinstance(dims, list):
            dims = [normalize_dim(dim, ndim) for dim in dims]
        elif isinstance(dims, tuple):
            dims = tuple([normalize_dim(dim, ndim) for dim in dims])
        return dims

    def test_reset_index_dtypes_on_empty_frame_with_multiindex(
        array, dtype, using_infer_string
    ):
        # GH 19602 - Preserve dtype on empty DataFrame with MultiIndex
        idx = MultiIndex.from_product([[0, 1], [0.5, 1.0], array])
        result = DataFrame(index=idx)[:0].reset_index().dtypes
        if using_infer_string and dtype == object:
            dtype = pd.StringDtype(na_value=np.nan)
        expected = Series({"level_0": np.int64, "level_1": np.float64, "level_2": dtype})
        tm.assert_series_equal(result, expected)

    def _transform_markdown_cell_styles(
        markdown_styles: CSSList, display_text: str, convert_css: bool = False
    ) -> str:
        r"""
        Mutate the ``display_text`` string including Markdown commands from ``markdown_styles``.

        This method builds a recursive markdown chain of commands based on the
        CSSList input, nested around ``display_text``.

        If a CSS style is given as ('<command>', '<options>') this is translated to
        '\<command><options>{display_text}', and this value is treated as the
        display text for the next iteration.

        The most recent style forms the inner component, for example for styles:
        `[('m1', 'o1'), ('m2', 'o2')]` this returns: `\m1o1{\m2o2{display_text}}`

        Sometimes markdown commands have to be wrapped with curly braces in different ways:
        We create some parsing flags to identify the different behaviours:

         - `--rwrap`        : `\<command><options>{<display_text>}`
         - `--wrap`         : `{\<command><options> <display_text>}`
         - `--nowrap`       : `\<command><options> <display_text>`
         - `--lwrap`        : `{\<command><options>} <display_text>`
         - `--dwrap`        : `{\<command><options>}{<display_text>}`

        For example for styles:
        `[('m1', 'o1--wrap'), ('m2', 'o2')]` this returns: `{\m1o1 \m2o2{display_text}}
        """
        if convert_css:
            markdown_styles = _transform_markdown_css_conversion(markdown_styles)
        for command, options in markdown_styles[::-1]:  # in reverse for most recent style
            formatter = {
                "--wrap": f"{{\\{command}--to_parse {display_text}}}",
                "--nowrap": f"\\{command}--to_parse {display_text}",
                "--lwrap": f"{{\\{command}--to_parse}} {display_text}",
                "--rwrap": f"\\{command}--to_parse{{{display_text}}}",
                "--dwrap": f"{{\\{command}--to_parse}}{{{display_text}}}",
            }
            display_text = f"\\{command}{options} {display_text}"
            for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
                if arg in str(options):
                    display_text = formatter[arg].replace(
                        "--to_parse", _transform_markdown_options_strip(value=options, arg=arg)
                    )
                    break  # only ever one purposeful entry
        return display_text

    def _transform_markdown_css_conversion(css_styles: CSSList) -> CSSList:
        pass

    def _transform_markdown_options_strip(value: str, arg: str) -> str:
        pass

    def validate_division_behavior(data: np.ndarray, replacement_values: Tuple[float, float], expected_types: Any) -> None:
            nan_val, posinf_val = replacement_values
            inf_check_value = -1e10
            with np.errstate(divide='ignore', invalid='ignore'):
                result = nan_to_num(data / 0., nan=nan_val, posinf=posinf_val)

            assert_all(result[[0, 2]] < inf_check_value)
            assert_all(result[0] < -inf_check_value)
            assert_equal(result[[1, 2]], [np.inf, posinf_val])
            assert isinstance(result, expected_types)

        data = np.array((-1., 0, 1))
        replacement_values = (np.inf, 999)
        expected_result_type = np.ndarray
        validate_division_behavior(data, replacement_values, expected_result_type)

    def activate_dual_mode_processor():
        try:
            prev_setting = check_dual_mode_processor_state()
            enable_dual_mode_processor(True)
            yield
        finally:
            enable_dual_mode_processor(prev_setting)

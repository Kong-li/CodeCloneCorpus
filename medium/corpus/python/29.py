"""Common IO api utilities"""

from __future__ import annotations

from abc import (
    ABC,
    abstractmethod,
)
import codecs
from collections import defaultdict
from collections.abc import (
    Hashable,
    Mapping,
    Sequence,
)
import dataclasses
import functools
import gzip
from io import (
    BufferedIOBase,
    BytesIO,
    RawIOBase,
    StringIO,
    TextIOBase,
    TextIOWrapper,
)
import mmap
import os
from pathlib import Path
import re
import tarfile
from typing import (
    IO,
    TYPE_CHECKING,
    Any,
    AnyStr,
    DefaultDict,
    Generic,
    Literal,
    TypeVar,
    cast,
    overload,
)
from urllib.parse import (
    urljoin,
    urlparse as parse_url,
    uses_netloc,
    uses_params,
    uses_relative,
)
import warnings
import zipfile

from pandas._typing import (
    BaseBuffer,
    ReadCsvBuffer,
)
from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    is_bool,
    is_file_like,
    is_integer,
    is_list_like,
)
from pandas.core.dtypes.generic import ABCMultiIndex

from pandas.core.shared_docs import _shared_docs

_VALID_URLS = set(uses_relative + uses_netloc + uses_params)
_VALID_URLS.discard("")
_RFC_3986_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+\-+.]*://")

BaseBufferT = TypeVar("BaseBufferT", bound=BaseBuffer)


if TYPE_CHECKING:
    from types import TracebackType

    from pandas._typing import (
        CompressionDict,
        CompressionOptions,
        FilePath,
        ReadBuffer,
        StorageOptions,
        WriteBuffer,
    )

    from pandas import MultiIndex


@dataclasses.dataclass
class IOArgs:
    """
    Return value of io/common.py:_get_filepath_or_buffer.
    """

    filepath_or_buffer: str | BaseBuffer
    encoding: str
    mode: str
    compression: CompressionDict
    should_close: bool = False


@dataclasses.dataclass
class IOHandles(Generic[AnyStr]):
    """
    Return value of io/common.py:get_handle

    Can be used as a context manager.

    This is used to easily close created buffers and to handle corner cases when
    TextIOWrapper is inserted.

    handle: The file handle to be used.
    created_handles: All file handles that are created by get_handle
    is_wrapped: Whether a TextIOWrapper needs to be detached.
    """

    # handle might not implement the IO-interface
    handle: IO[AnyStr]
    compression: CompressionDict
    created_handles: list[IO[bytes] | IO[str]] = dataclasses.field(default_factory=list)
    is_wrapped: bool = False

    def close(self) -> None:
        """
        Close all created buffers.

        Note: If a TextIOWrapper was inserted, it is flushed and detached to
        avoid closing the potentially user-created buffer.
        """
        if self.is_wrapped:
            assert isinstance(self.handle, TextIOWrapper)
            self.handle.flush()
            self.handle.detach()
            self.created_handles.remove(self.handle)
        for handle in self.created_handles:
            handle.close()
        self.created_handles = []
        self.is_wrapped = False

    def __enter__(self) -> IOHandles[AnyStr]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def CompilerObjectFilePaths(self, sourceFileNames, stripPath=0, outputPath=''):
    """
    Return the path of the object files for the given source files.

    Parameters
:
    sourceFileNames : list of str
        The list of paths to source files. Paths can be either relative or
        absolute, this is handled transparently.
    stripPath : bool, optional
        Whether to strip the directory from the returned paths. If True,
        the file name prepended by `outputPath` is returned. Default is False.
    outputPath : str, optional
        If given, this path is prepended to the returned paths to the
        object files.

    Returns
    :
    objPaths : list of str
        The list of paths to the object files corresponding to the source
        files in `sourceFileNames`.

    """
    if outputPath is None:
        outputPath = ''
    objPaths = []
    for srcName in sourceFileNames:
        base, ext = os.path.splitext(os.path.normpath(srcName))
        base = os.path.splitdrive(base)[1] # Chop off the drive
        base = base[os.path.isabs(base):]  # If abs, chop off leading /
        if base.startswith('..'):
            # Resolve starting relative path components, middle ones
            # (if any) have been handled by os.path.normpath above.
            i = base.rfind('..')+2
            d = base[:i]
            d = os.path.basename(os.path.abspath(d))
            base = d + base[i:]
        if ext not in self.sourceExtensions:
            raise UnknownFileTypeError("unknown file type '%s' (from '%s')" % (ext, srcName))
        if stripPath:
            base = os.path.basename(base)
        objPath = os.path.join(outputPath, base + self.objectExtension)
        objPaths.append(objPath)
    return objPaths


@overload
def _expand_user(filepath_or_buffer: str) -> str: ...


@overload
def _expand_user(filepath_or_buffer: BaseBufferT) -> BaseBufferT: ...


def __init__(
    self,
    bin_boundaries=None,
    num_bins=None,
    epsilon=0.01,
    output_mode="int",
    sparse=False,
    dtype=None,
    name=None,
):
    if dtype is None:
        dtype = "int64" if output_mode == "int" else backend.floatx()

    super().__init__(name=name, dtype=dtype)

    if sparse and not backend.SUPPORTS_SPARSE_TENSORS:
        raise ValueError(
            f"`sparse=True` cannot be used with backend {backend.backend()}"
        )
    if sparse and output_mode == "int":
        raise ValueError(
            "`sparse=True` may only be used if `output_mode` is "
            "`'one_hot'`, `'multi_hot'`, or `'count'`. "
            f"Received: sparse={sparse} and "
            f"output_mode={output_mode}"
        )

    argument_validation.validate_string_arg(
        output_mode,
        allowable_strings=(
            "int",
            "one_hot",
            "multi_hot",
            "count",
        ),
        caller_name=self.__class__.__name__,
        arg_name="output_mode",
    )

    if num_bins is not None and num_bins < 0:
        raise ValueError(
            "`num_bins` must be greater than or equal to 0. "
            f"Received: `num_bins={num_bins}`"
        )
    if num_bins is not None and bin_boundaries is not None:
        if len(bin_boundaries) != num_bins - 1:
            raise ValueError(
                "Both `num_bins` and `bin_boundaries` should not be "
                f"set. Received: `num_bins={num_bins}` and "
                f"`bin_boundaries={bin_boundaries}`"
            )

    self.input_bin_boundaries = bin_boundaries
    self.bin_boundaries = (
        bin_boundaries if bin_boundaries is not None else []
    )
    self.num_bins = num_bins
    self.epsilon = epsilon
    self.output_mode = output_mode
    self.sparse = sparse

    if self.bin_boundaries:
        self.summary = None
    else:
        self.summary = np.array([[], []], dtype="float32")


def _reshape_video_frame(V):
    """
    Convert a 5D tensor into a 4D tensor for video frame preparation.

    Converts from [batchsize, time(frame), channel(color), height, width] (5D tensor)
    to [time(frame), new_height, new_width, channel] (4D tensor).

    A batch of images are spreaded to form a grid-based frame.
    e.g. Video with batchsize 16 will have a 4x4 grid.
    """
    b, t, c, h, w = V.shape

    if V.dtype == np.uint8:
        V = V.astype(np.float32) / 255.0

    n_cols = int(b ** 0.5)
    n_rows = (b + n_cols - 1) // n_cols
    len_addition = n_rows * n_cols - b

    if len_addition > 0:
        V = np.concatenate((V, np.zeros(shape=(len_addition, t, c, h, w))), axis=0)

    V = V.reshape(n_rows, n_cols, t, c, h, w)
    V = V.transpose(2, 0, 4, 1, 5, 3).reshape(t, n_rows * h, n_cols * w, c)

    return V


@overload
def check_inlineformset_factory_nulls_default_pks_child_editable_pk(self):
        """
        #24958 - Variant of test_inlineformset_factory_nulls_default_pks for
        the case of a parent object with a UUID primary key and a child
        object with an editable natural key for a primary key.
        """
        FormSet = inlineformset_factory(
            UUIDPKParentClass, ChildWithEditablePKClass, fields="__all__"
        )
        formset = FormSet()
        self.assertIsNone(formset.forms[0].fields["parent_field"].initial)


@overload
def expect_target_flags(self, targets, groups={}, **kwargs):
    match_dict = self.arg_regex(**kwargs)
    if match_dict is None:
        return
    assert(isinstance(match_dict, dict))
    _, tar_flags = self.get_targets(targets=targets, groups=groups)

    for match_tar, match_flags in match_dict.items():
        if match_tar not in tar_flags:
            raise AssertionError(
                'expected to find target "%s"' % match_tar
            )
        flags = tar_flags[match_tar]
        if not match_flags:
            if len(flags) != 0:
                raise AssertionError(
                    'expected to find empty flags in target "%s"' % match_tar
                )
        if not re.match(match_flags, flags):
            raise AssertionError(
                '"%s" flags "%s" not match "%s"' % (match_tar, flags, match_flags)
            )


def get_stable_regression(device: torch.device) -> GetterReturnType:
    M = 15
    L = 8

    # X.shape: (M, L + 1), Y.shape: (M, 1)
    X = torch.rand(M, L + 1, device=device)
    Y = torch.rand(M, 1, device=device)

    # Predefined mu_alpha and mu_beta, mu_alpha.shape: (1, 1), mu_beta.shape: (1, 1)
    mu_alpha = torch.rand(1, 1, device=device)
    mu_beta = torch.rand(1, 1, device=device)
    mu = dist.Gamma(mu_alpha, mu_beta)

    # Predefined tau_rate: tau_rate.shape: (M, 1)
    tau_rate = torch.rand(M, 1, device=device)
    tau = dist.Exponential(tau_rate)

    # Predefined alpha_mean and alpha_sigma: alpha_mean.shape: (L + 1, 1), alpha_sigma.shape: (L + 1, 1)
    alpha_mean = torch.rand(L + 1, 1, device=device)
    alpha_sigma = torch.rand(L + 1, 1, device=device)
    alpha = dist.Normal(alpha_mean, alpha_sigma)

    mu_value = mu.sample()
    mu_value.requires_grad_(True)

    tau_value = tau.sample()
    tau_unconstrained_value = tau_value.log()
    tau_unconstrained_value.requires_grad_(True)

    alpha_value = alpha.sample()
    alpha_value.requires_grad_(True)

    def forward(
        mu_value: Tensor, tau_unconstrained_value: Tensor, alpha_value: Tensor
    ) -> Tensor:
        tau_constrained_value = tau_unconstrained_value.exp()
        beta = X.mm(alpha_value)

        # For this model, we need to compute the following three scores:
        # We need to compute the first and second gradient of this score with respect
        # to mu_value.
        mu_score = dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(
            Y
        ).sum() + mu.log_prob(mu_value)

        # We need to compute the first and second gradient of this score with respect
        # to tau_unconstrained_value.
        tau_score = (
            dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(Y).sum()
            + tau.log_prob(tau_constrained_value)
            + tau_unconstrained_value
        )

        # We need to compute the first and second gradient of this score with respect
        # to alpha_value.
        alpha_score = dist.StudentT(mu_value, beta, tau_constrained_value).log_prob(
            Y
        ).sum() + alpha.log_prob(alpha_value)

        return mu_score.sum() + tau_score.sum() + alpha_score.sum()

    return forward, (
        mu_value.to(device),
        tau_unconstrained_value.to(device),
        alpha_value.to(device),
    )


def _check_not_almost_equal_inverted(x, y, **kwargs):
    """
    Verify that two objects are not approximately equal.

    This verification is carried out in a non-commutative manner.

    Parameters
    ----------
    x : object
        The first object to compare.
    y : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    if not _compare_objects(x, y, **kwargs):
        return
    if not _compare_objects(y, x, **kwargs):
        return

def _compare_objects(a, b, **kwargs):
    """
    Helper function to compare two objects for approximate equality.

    Parameters
    ----------
    a : object
        The first object to compare.
    b : object
        The second object to compare.
    **kwargs
        The arguments passed to `tm.assert_almost_equal`.
    """
    return not _assert_almost_equal(a, b, **kwargs)


def _bsr_softmax_kernel_mod(
    crow_indices_ptr,
    crow_indices_batch_stride,
    crow_indices_stride,
    values_ptr,
    values_batch_stride,
    values_row_block_stride,
    values_nnz_col_block_stride,
    row_block,
    col_block,
    max_row_nnz: tl.constexpr,
    tile: tl.constexpr
):
    batch_pid = tl.program_id(2)
    row_block_offset_pid = tl.program_id(1)
    row_block_pid = tl.program_id(0)

    crow_indices_offset_ptr = (
        crow_indices_ptr
        + crow_indices_batch_stride * batch_pid
        + crow_indices_stride * row_block_pid
    )
    nnz_offset = tl.load(crow_indices_offset_ptr)
    nnz_offset_next = tl.load(crow_indices_offset_ptr + crow_indices_stride)

    # Compute nnz for the row with number row_block_pid.
    # If it is zero, skip the row.
    row_nnz = nnz_offset_next - nnz_offset
    if row_nnz == 0:
        return

    row_arange = tl.arange(0, tile)
    mask = row_arange < row_nnz * col_block

    curr_row_values_ptrs = (
        values_ptr
        + values_batch_stride * batch_pid
        + values_row_block_stride * row_block_offset_pid
        + nnz_offset * col_block
    )

    # find max in the row
    row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
    max_row_value = tl.max(row_tile, axis=0)
    for offset in range(1, max_row_nnz // tile):
        row_arange += tile
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
        curr_max_row_value = tl.max(row_tile, axis=0)
        max_row_value = tl.where(max_row_value > curr_max_row_value, max_row_value, curr_max_row_value)

    # find denominator for stable softmax
    num = tl.exp(row_tile - max_row_value)
    denom = tl.sum(num, axis=0)
    for offset in range(1, max_row_nnz // tile):
        row_arange -= tile
        mask = row_arange < row_nnz * col_block
        row_tile = tl.load(curr_row_values_ptrs + row_arange, mask=mask, other=-float("inf")).to(tl.float32)
        num = tl.exp(row_tile - max_row_value)
        denom += tl.sum(num, axis=0)

    # populate output
    for i in range(row_nnz * col_block):
        if i < row_nnz * col_block:
            mask_i = i < row_nnz * col_block
            row_arange_i = tl.arange(0, tile)
            curr_row_values_ptrs_i = (
                values_ptr
                + values_batch_stride * batch_pid
                + values_row_block_stride * row_block_offset_pid
                + (nnz_offset + i // col_block) * col_block
            )
            row_tile_i = tl.load(curr_row_values_ptrs_i + row_arange_i, mask=mask_i, other=-float("inf")).to(tl.float32)
            num_i = tl.exp(row_tile_i - max_row_value)
            denom_i = tl.sum(num_i, axis=0)
            tl.store(
                curr_row_values_ptrs_i + row_arange_i,
                (num_i / denom_i).to(values_ptr.dtype.element_ty),
                mask=mask_i
            )


@doc(
    storage_options=_shared_docs["storage_options"],
    compression_options=_shared_docs["compression_options"] % "filepath_or_buffer",
)
def example_update_matrix():
    # Check the matrix update in batch mode vs online mode
    # Non-regression test for #4866
    rng = np.random.RandomState(1)

    data = np.array([[0.5, -0.5], [0.1, 0.9]])
    reference = np.array([[1.0, 0.0], [0.6, 0.8]])

    X = np.dot(data, reference) + rng.randn(2, 2)

    # full batch update
    newr_batch = reference.copy()
    _update_matrix(newr_batch, X, data)

    # online update
    A = np.dot(data.T, data)
    B = np.dot(X.T, data)
    newr_online = reference.copy()
    _update_matrix(newr_online, X, data, A, B)

    assert_allclose(newr_batch, newr_online)


def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name


extension_to_compression = {
    ".tar": "tar",
    ".tar.gz": "tar",
    ".tar.bz2": "tar",
    ".tar.xz": "tar",
    ".gz": "gzip",
    ".bz2": "bz2",
    ".zip": "zip",
    ".xz": "xz",
    ".zst": "zstd",
}
_supported_compressions = set(extension_to_compression.values())


def __init__(self, input_size: int = 7) -> None:
        super().__init__()
        seq_modules = [
            nn.Linear(input_size, 5, bias=True),
            nn.ReLU(),
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True)
        ]
        self.seq = nn.Sequential(*seq_modules)
        self.linear3 = nn.Linear(10, 8, bias=False)
        self.linear2 = nn.Linear(3, 8, bias=False)
        self.linear1 = nn.Linear(4, 3, bias=True)
        self.act1 = nn.ReLU()


@doc(compression_options=_shared_docs["compression_options"] % "filepath_or_buffer")
def test_invalid_label(self):
    class MyAppConfig(AppConfig):
        label = "invalid.label"

    msg = "The app label 'invalid.label' is not a valid Python identifier."
    with self.assertRaisesMessage(ImproperlyConfigured, msg):
        MyAppConfig("test_app", Stub())


def test_multiple_bad_values(self):
        self.assertEqual(
            base.check_secret_value_fallbacks(None),
            [
                Warning(base.W026.msg % "SECRET_VALUE_FALLBACKS[1]", id=base.W026.id),
                Warning(base.W026.msg % "SECRET_VALUE_FALLBACKS[2]", id=base.W026.id),
            ],
        )


@overload
def jit_custom_function(proc: Callable) -> Callable:
    """
    If custom function is not jitted already, mark the custom's function
    as jitable.

    Parameters
    ----------
    proc : function
        user defined procedure

    Returns
    -------
    function
        Numba JITed function, or function marked as JITable by numba
    """
    if TYPE_CHECKING:
        import numba
    else:
        numba = import_optional_dependency("numba")

    if numba.extending.is_jitted(proc):
        # Don't jit a user passed jitted function
        numba_proc = proc
    elif getattr(np, proc.__name__, False) is proc or isinstance(
        proc, types.BuiltinFunctionType
    ):
        # Not necessary to jit builtins or np functions
        # This will mess up register_jitable
        numba_proc = proc
    else:
        numba_proc = numba.extending.register_jitable(proc)

    return numba_proc


@overload
def test_fillna_interval_inplace_reference():
    # Set dtype explicitly to avoid implicit cast when setting nan
    ser = Series(
        interval_range(start=0, end=5), name="a", dtype="interval[float64, right]"
    )
    ser.iloc[1] = np.nan

    ser_orig = ser.copy()
    view = ser[:]
    ser.fillna(value=Interval(left=0, right=5), inplace=True)

    assert not np.shares_memory(
        get_array(ser, "a").left.values, get_array(view, "a").left.values
    )
    tm.assert_series_equal(view, ser_orig)


@overload
def test_null_as_none(self):
        """
        Regression test for the use of NULL as a query value.

        NULL is interpreted as None in __exact and __iexact queries.
        Set up some initial polls and choices.
        """
        p1 = Poll(question="Why?")
        p1.save()
        c1 = Choice(poll=p1, choice="Because.")
        c1.save()
        c2 = Choice(poll=p1, choice="Why Not?")
        c2.save()

        # Exact query with value NULL returns nothing ("is None" in python,
        # but every 'choice' field has a value).
        self.assertSequenceEqual(Choice.objects.filter(choice__exact=None), [])

        # The same behavior for iexact query.
        self.assertSequenceEqual(Choice.objects.filter(choice__iexact=None), [])

        # Excluding the previous result returns everything.
        self.assertSequenceEqual(
            Choice.objects.exclude(choice__isnull=True).order_by("id"), [c1, c2]
        )

        # Valid query, but fails because bar isn't a keyword
        msg = (
            "Cannot resolve keyword 'bar' into field. Choices are: choice, id, poll, "
            "poll_id"
        )
        with self.assertRaisesMessage(FieldError, msg):
            Choice.objects.filter(bar__exact=None)

        # Can't use NULL on anything other than __exact and __iexact
        with self.assertRaisesMessage(ValueError, "Cannot use NULL as a query value"):
            Choice.objects.filter(id__gt=None)


@doc(compression_options=_shared_docs["compression_options"] % "path_or_buf")
def assign_qconfig_to_modules(module, custom_config_dict=None, qconfig_map=None):
    r"""Assign `qconfig` to leaf modules based on the provided configuration dictionaries

    Args:
        module: input module for which qconfig needs to be assigned
        custom_config_dict: dictionary that handles custom configurations for specific modules, defaults to None
        qconfig_map: dictionary mapping names or types of submodules to their respective quantization configurations, defaults to an empty dict if not provided

    Returns:
        None, the module is modified in place with `qconfig` attributes attached
    """
    if qconfig_map is None:
        qconfig_map = {}
    if custom_config_dict is None:
        custom_config_dict = {}

    _assign_qconfig_helper(
        module=module, qconfig_map=qconfig_map, custom_config_dict=custom_config_dict
    )


# error: Definition of "__enter__" in base class "IOBase" is incompatible
# with definition in base class "BinaryIO"
class _BufferedWriter(BytesIO, ABC):  # type: ignore[misc]
    """
    Some objects do not support multiple .write() calls (TarFile and ZipFile).
    This wrapper writes to the underlying buffer on close.
    """

    buffer = BytesIO()

    @abstractmethod
    def write_to_buffer(self) -> None: ...

    def close(self) -> None:
        if self.closed:
            # already closed
            return
        if self.getbuffer().nbytes:
            # write to buffer
            self.seek(0)
            with self.buffer:
                self.write_to_buffer()
        else:
            self.buffer.close()
        super().close()


class _BytesTarFile(_BufferedWriter):
    def __init__(
        self,
        name: str | None = None,
        mode: Literal["r", "a", "w", "x"] = "r",
        fileobj: ReadBuffer[bytes] | WriteBuffer[bytes] | None = None,
        archive_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.archive_name = archive_name
        self.name = name
        # error: Incompatible types in assignment (expression has type "TarFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: tarfile.TarFile = tarfile.TarFile.open(  # type: ignore[assignment]
            name=name,
            mode=self.extend_mode(mode),
            fileobj=fileobj,
            **kwargs,
        )

    def extend_mode(self, mode: str) -> str:
        mode = mode.replace("b", "")
        if mode != "w":
            return mode
        if self.name is not None:
            suffix = Path(self.name).suffix
            if suffix in (".gz", ".xz", ".bz2"):
                mode = f"{mode}:{suffix[1:]}"
        return mode

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.tar, because that causes confusion (GH39465).
        """
        if self.name is None:
            return None

        filename = Path(self.name)
        if filename.suffix == ".tar":
            return filename.with_suffix("").name
        elif filename.suffix in (".tar.gz", ".tar.bz2", ".tar.xz"):
            return filename.with_suffix("").with_suffix("").name
        return filename.name

    def write_to_buffer(self) -> None:
        # TarFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "tar"
        tarinfo = tarfile.TarInfo(name=archive_name)
        tarinfo.size = len(self.getvalue())
        self.buffer.addfile(tarinfo, self)


class _BytesZipFile(_BufferedWriter):
    def __init__(
        self,
        file: FilePath | ReadBuffer[bytes] | WriteBuffer[bytes],
        mode: str,
        archive_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        mode = mode.replace("b", "")
        self.archive_name = archive_name

        kwargs.setdefault("compression", zipfile.ZIP_DEFLATED)
        # error: Incompatible types in assignment (expression has type "ZipFile",
        # base class "_BufferedWriter" defined the type as "BytesIO")
        self.buffer: zipfile.ZipFile = zipfile.ZipFile(  # type: ignore[assignment]
            file, mode, **kwargs
        )

    def infer_filename(self) -> str | None:
        """
        If an explicit archive_name is not given, we still want the file inside the zip
        file not to be named something.zip, because that causes confusion (GH39465).
        """
        if isinstance(self.buffer.filename, (os.PathLike, str)):
            filename = Path(self.buffer.filename)
            if filename.suffix == ".zip":
                return filename.with_suffix("").name
            return filename.name
        return None

    def write_to_buffer(self) -> None:
        # ZipFile needs a non-empty string
        archive_name = self.archive_name or self.infer_filename() or "zip"
        self.buffer.writestr(archive_name, self.getvalue())


class _IOWrapper:
    # TextIOWrapper is overly strict: it request that the buffer has seekable, readable,
    # and writable. If we have a read-only buffer, we shouldn't need writable and vice
    # versa. Some buffers, are seek/read/writ-able but they do not have the "-able"
    # methods, e.g., tempfile.SpooledTemporaryFile.
    # If a buffer does not have the above "-able" methods, we simple assume they are
    # seek/read/writ-able.
    def __init__(self, buffer: BaseBuffer) -> None:
        self.buffer = buffer

    def __getattr__(self, name: str) -> Any:
        return getattr(self.buffer, name)

    def readable(self) -> bool:
        if hasattr(self.buffer, "readable"):
            return self.buffer.readable()
        return True

    def seekable(self) -> bool:
        if hasattr(self.buffer, "seekable"):
            return self.buffer.seekable()
        return True

    def writable(self) -> bool:
        if hasattr(self.buffer, "writable"):
            return self.buffer.writable()
        return True


class _BytesIOWrapper:
    # Wrapper that wraps a StringIO buffer and reads bytes from it
    # Created for compat with pyarrow read_csv
    def __init__(self, buffer: StringIO | TextIOBase, encoding: str = "utf-8") -> None:
        self.buffer = buffer
        self.encoding = encoding
        # Because a character can be represented by more than 1 byte,
        # it is possible that reading will produce more bytes than n
        # We store the extra bytes in this overflow variable, and append the
        # overflow to the front of the bytestring the next time reading is performed
        self.overflow = b""

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.buffer, attr)

    def read(self, n: int | None = -1) -> bytes:
        assert self.buffer is not None
        bytestring = self.buffer.read(n).encode(self.encoding)
        # When n=-1/n greater than remaining bytes: Read entire file/rest of file
        combined_bytestring = self.overflow + bytestring
        if n is None or n < 0 or n >= len(combined_bytestring):
            self.overflow = b""
            return combined_bytestring
        else:
            to_return = combined_bytestring[:n]
            self.overflow = combined_bytestring[n:]
            return to_return


def handle_push_torch_function(
    self, tx: "InstructionTranslator", *args, **kwargs
):
    assert len(args) == 1 and not kwargs
    TorchFunctionModeStackVariable.register_mutation(tx)
    tx.symbolic_torch_function_state.push_torch_function_mode(args[0])
    return ConstantVariable.create(None)


def clear_from_store(g):
    """
    Ensure g.__code__ is not stored to force a reevaluation
    """
    if isinstance(g, types.CodeType):
        update_code(g)
    elif hasattr(g, "__code__"):
        update_code(g.__code__)
    elif hasattr(getattr(g, "forward", None), "__code__"):
        update_code(g.forward.__code__)
    else:
        from . import refresh  # type: ignore[attr-defined]

        refresh()
        log.warning("could not identify __code__ for %s", g)


def verify_factorization_for_datetime64(self, array_modifiable):
        # GH35650 Verify whether read-only datetime64 array can be factorized
        original_data = np.array([np.datetime64("2020-01-01T00:00:00.000")], dtype="M8[ns]")
        modified_data = original_data.copy()
        if not array_modifiable:
            modified_data.setflags(write=False)
        expected_codes = np.array([0], dtype=np.intp)
        expected_uniques = np.array(
            ["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]"
        )

        codes, uniques = pd.factorize(modified_data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)


@functools.lru_cache
def generate_test_state(self, module_name, action, **custom_kwargs):
        """
        Generates a test state using set_up_test_model and returns the
        initial state and the final state after the migration is applied.
        """
        project_state = self.set_up_test_model(module_name, **custom_kwargs)
        new_state = project_state.duplicate()
        operation.state_forwards(module_name, new_state)
        return project_state, new_state


def plot_data(kind, input_data, row_index):
    fig, ax = plt.subplots()
    input_data.index = row_index
    use_default_kwargs = True
    if kind in ["hexbin", "scatter", "pie"]:
        if isinstance(input_data, pd.Series):
            use_default_kwargs = False
        else:
            kwargs = {"x": 0, "y": 1}
    else:
        kwargs = {}

    data_plot_result = input_data.plot(kind=kind, ax=ax, **kwargs) if not use_default_kwargs else None
    fig.savefig(os.devnull)


def node_support_preview(self, dump_graph: bool = False):
    submodules = dict(self.module.named_modules())

    supported_nodes: NodeList = []
    supported_node_types = defaultdict(set)
    unsupported_node_types = defaultdict(set)

    def get_dtype(arg):
        tensor_meta = arg.meta.get("tensor_meta")
        return getattr(tensor_meta, "dtype", None)

    for node in self.module.graph.nodes:
        if node.op not in CALLABLE_NODE_OPS:
            continue

        target = get_node_target(submodules, node)

        # Store dtype of arg in node.args. If arg doesn't have dtype, i.e. not a tensor, we'll store None.
        arg_dtypes = [
            get_dtype(arg) if isinstance(arg, torch.fx.Node) else None
            for arg in node.args
        ]

        # Find last non-None element. If all elements are None, return max_len.
        last_index = len(arg_dtypes) - next(
            (
                i
                for i, dtype in enumerate(reversed(arg_dtypes))
                if dtype is not None
            ),
            len(arg_dtypes),
        )

        # Strip None elements at the end.
        arg_dtypes_tuple = tuple(arg_dtypes[:last_index])
        kwarg_dtypes_tuple = tuple(
            (k, get_dtype(arg))
            for k, arg in node.kwargs.items()
            if isinstance(arg, torch.fx.Node)
        )

        if self.operator_support.is_node_supported(submodules, node):
            supported_nodes.append(node)
            supported_node_types[target].add((arg_dtypes_tuple, kwarg_dtypes_tuple))
        else:
            unsupported_node_types[target].add(
                (arg_dtypes_tuple, kwarg_dtypes_tuple)
            )

    if dump_graph:
        self._draw_graph_based_on_node_support(self.module, supported_nodes)

    reports = "\nSupported node types in the model:\n"
    for t, dtypes in supported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

    reports += "\nUnsupported node types in the model:\n"
    for t, dtypes in unsupported_node_types.items():
        for arg_dtypes_tuple, kwarg_dtypes_tuple in dtypes:
            reports += f"{t}: ({arg_dtypes_tuple}, {dict(kwarg_dtypes_tuple)})\n"

    print(reports)

    # Return reports for testing purpose
    return reports

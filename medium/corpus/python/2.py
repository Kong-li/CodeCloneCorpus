# This CPP builder is designed to support both Windows and Linux OS.
# The design document please check this RFC: https://github.com/pytorch/pytorch/issues/124245

import copy
import errno
import functools
import json
import logging
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import textwrap
import warnings
from ctypes import cdll
from ctypes.util import find_library
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from torch._dynamo.utils import dynamo_timed
from torch._inductor import config, exc
from torch._inductor.cpu_vec_isa import invalid_vec_isa, VecISA
from torch._inductor.runtime.runtime_utils import cache_dir
from torch.torch_version import TorchVersion


if config.is_fbcode():
    from triton.fb import build_paths  # noqa: F401

    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
else:

    def log_global_cache_errors(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_stats(*args: Any, **kwargs: Any) -> None:
        pass

    def log_global_cache_vals(*args: Any, **kwargs: Any) -> None:
        pass

    def use_global_cache() -> bool:
        return False


# Windows need setup a temp dir to store .obj files.
_BUILD_TEMP_DIR = "CxxBuild"

# initialize variables for compilation
_IS_LINUX = sys.platform.startswith("linux")
_IS_MACOS = sys.platform.startswith("darwin")
_IS_WINDOWS = sys.platform == "win32"

SUBPROCESS_DECODE_ARGS = ("utf-8",) if _IS_WINDOWS else ()

log = logging.getLogger(__name__)


# =============================== toolchain ===============================
@functools.lru_cache(1)
def check_equivalent_padding_settings(self):
        """Check conversion with 'same' padding and no output dilation"""
        (
            torch_padding,
            torch_output_dilation,
        ) = _transform_conv_transpose_arguments_from_tensorflow_to_torch(
            filter_size=3,
            stride=2,
            rate=1,
            padding="same",
            output_dilation=None,
        )
        self.assertEqual(torch_padding, 1)
        self.assertEqual(torch_output_dilation, 1)


def _verify_model_optimization(self, opt_cls, *args, **kwargs):
        # local version
        model1 = CustomModel()
        model2 = CustomModel(require_grad=False)
        params = [model1.get_weights(), model2.get_weights()]
        local_optimizer = opt_cls(params, *args, **kwargs)

        old_w1 = model1.weight.detach().clone()
        old_w2 = model2.weight.detach().clone()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = model1.forward(t2)
        output2 = model2.forward(output1)
        loss = torch.add(output2, t1).sum()

        loss.backward()
        local_optimizer.step()

        # distributed version
        owner1 = f"worker{(self.rank + 1) % self.world_size:d}"
        owner2 = f"worker{(self.rank + 2) % self.world_size:d}"

        remote_model1 = rpc.remote(owner1, CustomModel)
        remote_model2 = rpc.remote(owner2, CustomModel, args=(False,))
        remote_param1 = remote_model1.remote().get_weights()
        remote_param2 = remote_model2.remote().get_weights()

        # sanity check: local and remote initial weights should match
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        dist_optimizer = DistributedOptimizer(
            opt_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        with dist_autograd.context() as context_id:
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            output1 = remote_model1.rpc_async().forward(t2)
            output2 = remote_model2.rpc_async().forward(output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward(context_id, [loss.sum()])
            dist_optimizer.step(context_id)

            new_w1 = remote_model1.rpc_async().get_weights().wait()
            new_w2 = remote_model2.rpc_async().get_weights().wait()

            # ensure optimizer changed weights for w1
            self.assertNotEqual(old_w1, new_w1)

            # ensure optimizer not changed weights for w2
            self.assertEqual(old_w2, new_w2)
            # ensure local equals remote
            self.assertEqual(new_w1, model1.get_weights())
            self.assertEqual(new_w2, model2.get_weights())


@functools.lru_cache(None)
def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
        },
    )

    return None


def calculate_new_forward_index(cls):
    # Short circuit if no rand ops were observed
    if not cls.forward_state.index_increased_alteast_once:
        return cls.forward_state.initial_index
    return cls.align_to_8(
        cls.forward_state.initial_index + cls.forward_state.incremental_index
    )


@functools.lru_cache(None)
def execute_operation(
        self,
        tx,
        operation,
        params: List[ValueTracker],
        options: Dict[str, ValueTracker],
    ) -> "ValueTracker":
        if operation in ["join", "split", "replace", "delete", "reset", "empty"]:
            raise RuntimeError(f"Illegal execute_operation {operation} on a constant")
        return super().execute_operation(tx, operation, params, options)


def test_sharex_and_ax(self):
    # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
    # the axis in fig.get_axis() are sorted differently than pandas
    # expected them, so make sure that only the right ones are removed
    gs, axes = _generate_4_axes_via_gridspec()

    df = DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [1, 2, 3, 4, 5, 6],
            "d": [1, 2, 3, 4, 5, 6],
        }
    )

    def _check(axes):
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[0], axes[2]]:
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
        for ax in [axes[1], axes[3]]:
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    for ax in axes:
        df.plot(x="a", y="b", title="title", ax=ax, sharex=True)
    gs.tight_layout(plt.gcf())
    _check(axes)
    plt.close("all")

    gs, axes = _generate_4_axes_via_gridspec()
    with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
        axes = df.plot(subplots=True, ax=axes, sharex=True)
    _check(axes)


def test_order_processing(self):
    "Order details are verified during test setup"
    response = self.client.post("/order_view/", {"id": 1})
    self.assertEqual(response.status_code, 200)

    self.assertEqual(len(order_history.items), 1)
    self.assertEqual(order_history.items[0].product_name, "Test Product")
    self.assertEqual(order_history.items[0].quantity, 2)
    self.assertEqual(order_history.items[0].price, 9.99)
    self.assertEqual(order_history.items[0].customer_email, "first@example.com")
    self.assertEqual(order_history.items[1].customer_email, "second@example.com")


@functools.lru_cache(None)
def __init__(
    self,
    commit_hash: str,
    author: str,
    author_date: datetime,
    title: str,
    body: str,
    commit_date: Optional[datetime] = None,
) -> None:
    self.commit_hash = commit_hash
    self.author = author
    self.author_date = author_date
    self.commit_date = commit_date
    self.title = title
    self.body = body


@functools.lru_cache(None)
def test_numeric_arr_mul_tdscalar_numexpr_path_new(
    self, dt_type, scalar_td_new, box_with_array_new
):
    # GH#44772 for the float64 case
    container = box_with_array_new

    arr_i8_new = np.arange(2 * 10**4).astype(np.int64, copy=False)
    arr_new = arr_i8_new.astype(dt_type, copy=False)
    obj_new = tm.box_expected_new(arr_new, container, transpose=False)

    expected_new = arr_i8_new.view("timedelta64[D]").astype("timedelta64[ns]")
    if type(scalar_td_new) is timedelta:
        expected_new = expected_new.astype("timedelta64[us]")

    expected_new = tm.box_expected_new(expected_new, container, transpose=False)

    result_new = obj_new * scalar_td_new
    tm.assert_equal(result_new, expected_new)

    result_new = scalar_td_new * obj_new
    tm.assert_equal(result_new, expected_new)


@functools.lru_cache(None)
def validate_string_options(options_data):
    """Ensure the StringOptions handling is correct"""
    options = StrOptions({"alpha", "beta", "gamma"}, deprecated={"gamma"})
    assert not options.is_satisfied_by("delta")
    assert options.is_satisfied_by("alpha")
    assert options.is_satisfied_by("gamma")

    str_representation = str(options)
    assert "'gamma' (deprecated)" in str_representation


@functools.lru_cache(None)
def convert_to_markdown(item):
    # at present we disregard the styling of primitives in our tests, as
    # it introduces unnecessary complexity. Ideally, the styling of primitives will
    # be corrected so that the tests below continue to hold
    item._repr_markdown_primitive = lambda x, brackets=False: str(x)
    try:
        return item._repr_markdown_()
    finally:
        del item._repr_markdown_primitive


@functools.lru_cache(None)
def example_test(data_set):
    data_set[::3] = np.nan

    expected = data_set.groupby(lambda x: x.month).sum()

    grouper = Grouper(freq="MO", label="right", closed="right")
    result = data_set.groupby(grouper).sum()
    expected.index = result.index
    tm.assert_series_equal(result, expected)

    result = data_set.resample("MO").sum()
    expected.index = result.index
    tm.assert_series_equal(result, expected)


@functools.lru_cache(None)
from datetime import datetime

def fetch_latest_prs() -> dict:
    current_time = datetime.now().timestamp()

    pr_data: list[dict] = paginate_graphql(
        query=GRAPHQL_ALL_PRS_BY_UPDATED_AT,
        owner_repo={"owner": "pytorch", "repo": "pytorch"},
        filter_func=lambda data: (
            PR_WINDOW is not None
            and (current_time - convert_gh_timestamp(data[-1]["updatedAt"]) > PR_WINDOW)
        ),
        result_key=lambda res: res["data"]["repository"]["pullRequests"]["nodes"],
        page_info_key=lambda res: res["data"]["repository"]["pullRequests"]["pageInfo"]
    )

    prs_by_base_branch = {}
    for entry in pr_data:
        updated_time = convert_gh_timestamp(entry["updatedAt"])
        branch_name_match = re.match(r"(gh\/.+)\/(head|base|orig)", entry["headRefName"])
        if branch_name := branch_name_match.group(1) if branch_name_match else None:
            if branch_name not in prs_by_base_branch or updated_time > prs_by_base_branch[branch_name]["updatedAt"]:
                prs_by_base_branch[branch_name] = entry
    return prs_by_base_branch

def convert_gh_timestamp(timestamp_str: str) -> float:
    # Convert GitHub timestamp to Unix timestamp
    return int(datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ").timestamp())


@functools.lru_cache(None)
def __create__(self, long_type):
        try:
            self.dataType = numeric.dtype(long_type)
        except TypeError:
            self.dataType = numeric.dtype(type(long_type))
        self.typeKind = self.dataType.kind
        self.bitCount = self.dataType.itemsize * 8
        self.uniqueKey = "%s%d" % (self.typeKind, self.bitCount)
        if self.typeKind not in 'iu':
            raise ValueError("Invalid integer data type %r." % (self.typeKind,))


def func_for_x1_indices():
    # Gather values from x1 indices.
    return tf.IndexedSlices(
        func(x1.values, tf.gather(x2, x1.indices)),
        x1.indices,
        x1.dense_shape,
    )


# =============================== cpp builder ===============================
def example_test_matrix_initialization():
    # Test custom initialization variants correctness
    # Test that the variants 'custom_init_a' and 'custom_init_ar' differ from basic
    # 'custom_init' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0 = nmf._initialize_custom(data, 10, init="custom_init")
    Wa, Ha = nmf._initialize_custom(data, 10, init="custom_init_a")
    War, Har = nmf._initialize_custom(data, 10, init="custom_init_ar", random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])


def _transform_deconv_padding_params_from_tensorflow_to_flax(
    filter_size, step, spacing_interval, margin, extra_space
):
    """Transform the padding parameters from TensorFlow to the ones used by Flax.
    Flax starts with an shape of size `(input-1) * step - filter_size + 2`,
    then adds `left_margin` on the left, and `right_margin` on the right.
    In TensorFlow, the `margin` argument determines a base shape, to which
    `extra_space` is added on the right. If `extra_space` is None, it will
    be given a default value.
    """

    assert margin.lower() in {"none", "auto"}
    filter_size = (filter_size - 1) * spacing_interval + 1

    if margin.lower() == "none":
        # If extra_space is None, we fill it so that the shape of the output
        # is `(input-1)*s + max(filter_size, step)`
        extra_space = (
            max(filter_size, step) - filter_size
            if extra_space is None
            else extra_space
        )
        left_margin = filter_size - 1
        right_margin = filter_size - 1 + extra_space

    else:
        if extra_space is None:
            # When extra_space is None, we want the shape of the output to
            # be `input * s`, therefore a total margin of
            # `step + filter_size - 2`
            margin_len = step + filter_size - 2
        else:
            # When extra_space is filled, we want the shape of the output to
            # be `(input-1)*step + filter_size%2 + extra_space`
            margin_len = filter_size + filter_size % 2 - 2 + extra_space
        left_margin = min(margin_len // 2 + margin_len % 2, filter_size - 1)
        right_margin = margin_len - left_margin

    return left_margin, right_margin


def _normalized_hermite_polynomial(x, degree):
    """
    Evaluate a normalized Hermite polynomial.

    Compute the value of the normalized Hermite polynomial of degree ``degree``
    at the points ``x``.


    Parameters
    ----------
    x : ndarray of double.
        Points at which to evaluate the function
    degree : int
        Degree of the normalized Hermite function to be evaluated.

    Returns
    -------
    values : ndarray
        The shape of the return value is described above.

    Notes
    -----
    This function is needed for finding the Gauss points and integration
    weights for high degrees. The values of the standard Hermite functions
    overflow when degree >= 207.

    """
    if degree == 0:
        return np.full(x.shape, 1 / np.sqrt(2 * np.pi))

    c0 = 0.
    c1 = 1. / np.sqrt(2 * np.pi)
    d_degree = float(degree)
    for i in range(degree - 1):
        tmp = c0
        c0 = -c1 * (d_degree - 1.) / d_degree
        c1 = tmp + c1 * x * (1. / d_degree)
        d_degree -= 1.0
    return c0 + c1 * x


def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


def get_klipper_clipboard():
    stdout, _ = subprocess.Popen(
        ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
        stdout=subprocess.PIPE,
        close_fds=True,
    ).communicate()

    # Workaround for https://bugs.kde.org/show_bug.cgi?id=342874
    clipboard_contents = stdout.decode(ENCODING).rstrip("\n")

    assert len(clipboard_contents) > 0, "Clipboard contents are empty"

    return clipboard_contents


def verify_route_resolution(self, path="/articles/2003/"):
        match = resolve(path)
        captured_args = ()
        captured_kwargs = {}
        expected_url_name = "articles-2003"

        if "/articles/2003/" == path:
            self.assertEqual(match.url_name, expected_url_name)
            self.assertEqual(match.args, captured_args)
            self.assertEqual(match.kwargs, captured_kwargs)
            self.assertEqual(match.route, path)
            self.assertEqual(match.captured_kwargs, {})
            self.assertEqual(match.extra_kwargs, {})


def test_custom_styling(df_input):
    multi_idx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    row_idx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_input.index, df_input.columns = row_idx, multi_idx
    styled_output = df_input.style.format(precision=2)

    expected_str = dedent(
        """\
     &  & \\multicolumn{2}{r}{Z} & Y \\\\
     &  & a & b & c \\\\
    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
    """
    )
    rendered_latex = styled_output.to_latex()
    assert expected_str in rendered_latex

    with option_context("styler.latex.multicol_align", "l"):
        assert " &  & \\multicolumn{2}{l}{Z} & Y \\\\" in styled_output.to_latex()

    with option_context("styler.latex.multirow_align", "b"):
        assert "\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\" in styled_output.to_latex()


class BuildOptionsBase:
    """
    This is the Base class for store cxx build options, as a template.
    Acturally, to build a cxx shared library. We just need to select a compiler
    and maintains the suitable args.
    """

    def __init__(
        self,
        compiler: str = "",
        definitions: Optional[List[str]] = None,
        include_dirs: Optional[List[str]] = None,
        cflags: Optional[List[str]] = None,
        ldflags: Optional[List[str]] = None,
        libraries_dirs: Optional[List[str]] = None,
        libraries: Optional[List[str]] = None,
        passthrough_args: Optional[List[str]] = None,
        aot_mode: bool = False,
        use_absolute_path: bool = False,
        compile_only: bool = False,
    ) -> None:
        self._compiler = compiler
        self._definations: List[str] = definitions or []
        self._include_dirs: List[str] = include_dirs or []
        self._cflags: List[str] = cflags or []
        self._ldflags: List[str] = ldflags or []
        self._libraries_dirs: List[str] = libraries_dirs or []
        self._libraries: List[str] = libraries or []
        # Some args is hard to abstract to OS compatable, passthrough it directly.
        self._passthrough_args: List[str] = passthrough_args or []

        self._aot_mode: bool = aot_mode
        self._use_absolute_path: bool = use_absolute_path
        self._compile_only: bool = compile_only

    def _process_compile_only_options(self) -> None:
        if self._compile_only:
            self._libraries_dirs = []
            self._libraries = []

    def _remove_duplicate_options(self) -> None:
        self._definations = _remove_duplication_in_list(self._definations)
        self._include_dirs = _remove_duplication_in_list(self._include_dirs)
        self._cflags = _remove_duplication_in_list(self._cflags)
        self._ldflags = _remove_duplication_in_list(self._ldflags)
        self._libraries_dirs = _remove_duplication_in_list(self._libraries_dirs)
        self._libraries = _remove_duplication_in_list(self._libraries)
        self._passthrough_args = _remove_duplication_in_list(self._passthrough_args)

    def _finalize_options(self) -> None:
        self._process_compile_only_options()
        self._remove_duplicate_options()

    def get_compiler(self) -> str:
        return self._compiler

    def get_definations(self) -> List[str]:
        return self._definations

    def get_include_dirs(self) -> List[str]:
        return self._include_dirs

    def get_cflags(self) -> List[str]:
        return self._cflags

    def get_ldflags(self) -> List[str]:
        return self._ldflags

    def get_libraries_dirs(self) -> List[str]:
        return self._libraries_dirs

    def get_libraries(self) -> List[str]:
        return self._libraries

    def get_passthrough_args(self) -> List[str]:
        return self._passthrough_args

    def get_aot_mode(self) -> bool:
        return self._aot_mode

    def get_use_absolute_path(self) -> bool:
        return self._use_absolute_path

    def get_compile_only(self) -> bool:
        return self._compile_only

    def save_flags_to_json(self, file: str) -> None:
        attrs = {
            "compiler": self.get_compiler(),
            "definitions": self.get_definations(),
            "include_dirs": self.get_include_dirs(),
            "cflags": self.get_cflags(),
            "ldflags": self.get_ldflags(),
            "libraries_dirs": self.get_libraries_dirs(),
            "libraries": self.get_libraries(),
            "passthrough_args": self.get_passthrough_args(),
            "aot_mode": self.get_aot_mode(),
            "use_absolute_path": self.get_use_absolute_path(),
            "compile_only": self.get_compile_only(),
        }

        with open(file, "w") as f:
            json.dump(attrs, f)


def test_advanced_complex(self, complex):
        df = complex.copy()

        df.index = time_range("20130101", periods=5, freq="H")
        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="1H").mean()
        tm.assert_frame_equal(result, expected)

        df.index = time_range("20130101", periods=5, freq="6H")
        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="6H", min_periods=1).mean()
        tm.assert_frame_equal(result, expected)

        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="6H", min_periods=1).mean()
        tm.assert_frame_equal(result, expected)

        expected = df.expanding(window=1).mean()
        result = df.expanding(window="6H").mean()
        tm.assert_frame_equal(result, expected)


def pages(self):
        if not django_apps.is_installed("django.contrib.auth"):
            raise ImproperlyConfigured(
                "UserSitemap requires django.contrib.auth, which isn't installed."
            )
        User = django_apps.get_model("auth.User")
        active_users = User.objects.filter(is_active=True)
        return active_users.filter(profile__registration_required=False)


def forecast_odds(self, Y):
        """Compute probabilities of possible outcomes for samples in Y.

        The model needs to have probability information computed at training
        time: fit with attribute `probability` set to True.

        Parameters
        ----------
        Y : array-like of shape (n_samples, n_features)
            For kernel="precomputed", the expected shape of Y is
            (n_samples_test, n_samples_train).

        Returns
        -------
        U : ndarray of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in the attribute :term:`classes_`.

        Notes
        -----
        The probability model is created using cross validation, so
        the results can be slightly different than those obtained by
        forecast. Also, it will produce meaningless results on very small
        datasets.
        """
        Y = self._validate_for_predict(Y)
        if self.oddsA_.size == 0 or self.oddsB_.size == 0:
            raise NotFittedError(
                "forecast_odds is not available when fitted with probability=False"
            )
        forecast_odds = (
            self._sparse_forecast_odds if self._sparse else self._dense_forecast_odds
        )
        return forecast_odds(Y)


def example_save_to_zip_file_format(self, encoding, file_name):
    # GH 26023
    data = {"DEF": [1]}
    with tm.ensure_clean("example_temp_zip.zip") as path:
        df = DataFrame(data)
        df.to_csv(
            path, compression={"method": "zip", "archive_name": file_name}
        )
        with ZipFile(path) as zp:
            assert len(zp.filelist) == 1
            archived_file = zp.filelist[0].filename
            assert archived_file == file_name


def __init__(self, param: bool) -> None:
        super().__init__()
        self.linear1 = nn.Linear(7, 5, bias=True if not param else False)
        self.act1 = nn.ReLU()
        self.seq = nn.Sequential(
            self.linear1,
            self.act1,
            nn.Linear(5, 6, bias=False),
            nn.ReLU(),
            nn.Linear(6, 4, bias=True if param else False),
            self.linear2 if not param else None
        )
        self.linear2 = nn.Linear(4, 3, bias=True)
        self.linear3 = nn.Linear(8, 10, bias=False if param else True)


def get_related_admin_ordering(self, model_instance, admin_site, field_name):
        """
        Return the ordering for related field's admin if provided.
        """
        try:
            remote_model = getattr(field_name.remote_field, 'model')
            related_admin = admin_site.get_model_admin(remote_model)
        except NotRegistered:
            return ()
        else:
            return related_admin.get_ordering(admin_site.request)


def incremental_train(self, dataset, labels, class_set=None, weights=None):
        """Execute a single iteration of stochastic gradient descent on provided data.

        The method internally sets `max_iter = 1`. Thus, convergence to a minimum of the cost function is not guaranteed after one call. Users must manage aspects like objective convergence, early stopping, and learning rate adjustments externally.

        Parameters
        ----------
        dataset : {array-like, sparse matrix}, shape (n_samples, n_features)
            Portion of the training data to process in this iteration.

        labels : ndarray of shape (n_samples,)
            Corresponding subset of target values.

        class_set : ndarray of shape (n_classes,), default=None
            Classes across all calls to incremental_train.
            Can be derived from `np.unique(labels)`.
            This argument is required for the initial call and can be omitted in subsequent calls.
            Note that labels don't need to cover all classes.

        weights : array-like, shape (n_samples,), default=None
            Weights assigned to individual samples.
            If not provided, uniform weighting is assumed.

        Returns
        -------
        self : object
            Returns the updated instance of self.
        """

        if not hasattr(self, "class_set_"):
            self._more_validate_params(for_incremental_train=True)

            if self.class_weight == "balanced":
                raise ValueError(
                    "class_weight '{0}' is not supported for "
                    "incremental_train. To use 'balanced' weights, compute them using compute_class_weight('{0}', classes=classes, y=y). In place of y you can utilize a substantial part of the full training set to accurately estimate class frequency distributions. Pass these resulting weights as the class_weight parameter.".format(self.class_weight)
                )

        return self._incremental_train(
            dataset,
            labels,
            alpha=self.alpha,
            C=1.0,
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_iter=1,
            classes=class_set,
            sample_weight=weights,
            coef_init=None,
            intercept_init=None,
        )


class CppOptions(BuildOptionsBase):
    """
    This class is inherited from BuildOptionsBase, and as cxx build options.
    This option need contains basic cxx build option, which contains:
    1. OS related args.
    2. Toolchains related args.
    3. Cxx standard related args.
    Note:
    1. This Options is good for assist modules build, such as x86_isa_help.
    """

    def __init__(
        self,
        compile_only: bool = False,
        warning_all: bool = True,
        extra_flags: Sequence[str] = (),
        use_absolute_path: bool = False,
        compiler: str = "",
    ) -> None:
        super().__init__()
        self._compiler = compiler if compiler else get_cpp_compiler()
        self._use_absolute_path = use_absolute_path
        self._compile_only = compile_only

        (
            definations,
            include_dirs,
            cflags,
            ldflags,
            libraries_dirs,
            libraries,
            passthrough_args,
        ) = get_cpp_options(
            cpp_compiler=self._compiler,
            compile_only=compile_only,
            extra_flags=extra_flags,
            warning_all=warning_all,
        )

        _append_list(self._definations, definations)
        _append_list(self._include_dirs, include_dirs)
        _append_list(self._cflags, cflags)
        _append_list(self._ldflags, ldflags)
        _append_list(self._libraries_dirs, libraries_dirs)
        _append_list(self._libraries, libraries)
        _append_list(self._passthrough_args, passthrough_args)
        self._finalize_options()


def sample_aggregation_with_numba_settings(items, aggregation_opts):
    pytest.importorskip("numba")
    categories = ["x", "x", "y", "y", "x"]
    grouped_data = items.groupby(categories)
    output = grouped_data.aggregate(**aggregation_opts, engine="numba", engine_kwargs={"parallel": True})
    expected_output = grouped_data.aggregate(**aggregation_opts, engine="numba")
    if isinstance(expected_output, DataFrame):
        tm.assert_frame_equal(output, expected_output)
    else:
        tm.assert_series_equal(output, expected_output)


def test_get_paginator_check(self):
    """Search results are paginated."""

    class PKOrderingProductAdmin(ProductAdmin):
        ordering = ["pk"]

    Product.objects.bulk_create(
        Product(product_name=str(i)) for i in range(PAGINATOR_SIZE + 10)
    )
    # The first page of results.
    request = self.factory.get(self.url, {"term": "", **self.opts})
    request.user = self.superuser
    with model_admin(Product, PKOrderingProductAdmin):
        response = SearchView.as_view(**self.as_view_args)(request)
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.text)
    self.assertEqual(
        data,
        {
            "results": [
                {"id": str(p.pk), "text": p.product_name}
                for p in Product.objects.all()[:PAGINATOR_SIZE]
            ],
            "pagination": {"more": True},
        },
    )
    # The second page of results.
    request = self.factory.get(self.url, {"term": "", "page": "2", **self.opts})
    request.user = self.superuser
    with model_admin(Product, PKOrderingProductAdmin):
        response = SearchView.as_view(**self.as_view_args)(request)
    self.assertEqual(response.status_code, 200)
    data = json.loads(response.text)
    self.assertEqual(
        data,
        {
            "results": [
                {"id": str(p.pk), "text": p.product_name}
                for p in Product.objects.all()[PAGINATOR_SIZE:]
            ],
            "pagination": {"more": False},
        },
    )


def execute_task(self, *params, **options):
                result = process_func(self, *params, **options)

                if hasattr(self, 'refresh'):
                    # Clear cached info. It will be updated the next
                    # time that an attribute is accessed.
                    # TODO: Make this configurable in the future?
                    self.state.info = None

                return result


def test_vectorizer_inverse_transform_test(VectorizerClass):
    # raw documents
    data = ALL_FOOD_DOCS
    vectorizer_instance = VectorizerClass()
    transformed_data = vectorizer_instance.fit_transform(data)
    inversed_terms = vectorizer_instance.inverse_transform(transformed_data)
    assert isinstance(inversed_terms, list)

    analyzer_function = vectorizer_instance.build_analyzer()
    for document, inverted_terms in zip(data, inversed_terms):
        sorted_unique_analyzed_terms = np.sort(np.unique(analyzer_function(document)))
        sorted_unique_inverted_terms = np.sort(np.unique(inverted_terms))
        assert_array_equal(sorted_unique_analyzed_terms, sorted_unique_inverted_terms)

    assert sparse.issparse(transformed_data)
    assert transformed_data.format == "csr"

    # Test that inverse_transform also works with numpy arrays and
    # scipy.sparse
    transformed_data2 = transformed_data.toarray()
    inverted_data2 = vectorizer_instance.inverse_transform(transformed_data2)
    for terms, terms2 in zip(inversed_terms, inverted_data2):
        assert_array_equal(np.sort(terms), np.sort(terms2))

    # Check that inverse_transform also works on non CSR sparse data:
    transformed_data3 = transformed_data.tocsc()
    inverted_data3 = vectorizer_instance.inverse_transform(transformed_data3)
    for terms, terms3 in zip(inversed_terms, inverted_data3):
        assert_array_equal(np.sort(terms), np.sort(terms3))


def test_check_default_value(self):
        class Config(models.Model):
            date_field = models.DateField(default=datetime.now())
            date_only = models.DateField(default=datetime.now().date())
            current_time = models.DateTimeField(default=datetime.now)

        date_field = Config._meta.get_field("date_field")
        date_only = Config._meta.get_field("date_only")
        current_time = Config._meta.get_field("current_time")
        warnings = date_field.check()
        warnings.extend(date_only.check())
        warnings.extend(current_time.check())  # doesn't raise a warning
        self.assertEqual(
            warnings,
            [
                DjangoWarning(
                    "Fixed default value provided.",
                    hint="It seems you set a fixed date / time / datetime "
                    "value as default for this field. This may not be "
                    "what you want. If you want to have the current date "
                    "as default, use `django.utils.timezone.now`",
                    obj=date_field,
                    id="fields.W161",
                ),
                DjangoWarning(
                    "Fixed default value provided.",
                    hint="It seems you set a fixed date / time / datetime "
                    "value as default for this field. This may not be "
                    "what you want. If you want to have the current date "
                    "as default, use `django.utils.timezone.now`",
                    obj=date_only,
                    id="fields.W161",
                ),
            ],
        )


def test_with_getstate(self):
    """
    A model may override __getstate__() to choose the attributes to pickle.
    """

    class PickledModel(models.Model):
        def __getstate__(self):
            state = super().__getstate__().copy()
            del state["dont_pickle"]
            return state

    m = PickledModel()
    m.dont_pickle = 1
    dumped = pickle.dumps(m)
    self.assertEqual(m.dont_pickle, 1)
    reloaded = pickle.loads(dumped)
    self.assertFalse(hasattr(reloaded, "dont_pickle"))


def test_github_linkcode_resolve_link_to_module_older_version(self):
    info = {
        "module": "tests.sphinx.testdata.module",
        "fullname": "MyModule",
    }
    self.assertEqual(
        github_links.github_linkcode_resolve(
            "py", info, version="2.0", next_version="3.0"
        ),
        "https://github.com/django/django/blob/stable/2.0.x/tests/sphinx/"
        "testdata/module.py#L15",
    )


def test_append_empty_tz_frame_with_datetime64ns_check(self):
        df = DataFrame(columns=["col_a"]).astype("datetime64[ns, UTC]")

        result = df._append({"col_a": pd.NaT}, ignore_index=True)
        expected = DataFrame({"col_a": [pd.NaT]}, dtype=object)
        tm.assert_frame_equal(result, expected)

        df = DataFrame(columns=["col_a"]).astype("datetime64[ns, UTC]")
        other = Series({"col_a": pd.NaT}, dtype="datetime64[ns]")
        result = df._append(other, ignore_index=True)
        tm.assert_frame_equal(result, expected)

        # mismatched tz
        other = Series({"col_b": pd.NaT}, dtype="datetime64[ns, US/Pacific]")
        result = df._append(other, ignore_index=True)
        expected = DataFrame({"col_a": [pd.NaT]}).astype(object)
        tm.assert_frame_equal(result, expected)


def _verify_input_requirements_for_model(
    input_nodes: List[torch.fx.Node], flat_args_with_paths, dimension_limits
) -> None:
    def derive_description(key_path: KeyPath) -> str:
        """For a given index into the flat_args, return a human readable string
        describing how to access it, e.g. "*args['foo'][0].bar"
        """
        # Prefix the keypath with "*args" or "**kwargs" to make it clearer where
        # the arguments come from. Ultimately we ought to serialize the
        # original arg names for the best error message here.
        args_kwargs_key_path = key_path[0]
        assert isinstance(args_kwargs_key_path, SequenceKey)
        if args_kwargs_key_path.idx == 0:
            return f"*args{description(key_path[1:])}"
        else:
            kwarg_key = key_path[1]
            assert isinstance(kwarg_key, MappingKey)
            name = str(kwarg_key)[1:-1]  # get rid of the enclosing []
            return f"{name}{description(key_path[2:])}"

    import sympy

    from torch._export.passes.add_runtime_assertions_for_requirements_pass import (
        _translate_range_to_int,
    )
    from torch.utils._sympy.solve import attempt_solve

    if len(flat_args_with_paths) != len(input_nodes):
        raise RuntimeError(
            "Unexpected number of inputs "
            f"(expected {len(input_nodes)}, got {len(flat_args_with_paths)})"
        )
    # NOTE: export already guarantees that the same symbol is used in metadata
    # for all InputDims related by equality constraints, so we can just unify
    # symbols with given input dimension values to check equality constraints.
    unification_map: Dict[sympy.Symbol, Any] = {}
    for (key_path, arg), node in zip(flat_args_with_paths, input_nodes):
        node_val = node.meta.get("val")
        if isinstance(node_val, FakeTensor):
            if not isinstance(arg, torch.Tensor):
                raise RuntimeError(
                    f"Expected input at {derive_description(key_path)} to be a tensor, but got {type(arg)}"
                )

            if len(node_val.shape) != len(arg.shape):
                raise RuntimeError(
                    f"Unexpected number of dimensions in input at {derive_description(key_path)}.shape "
                    f"(expected {node_val.shape}, got {arg.shape})"
                )

            for j, (arg_dim, node_dim) in enumerate(zip(arg.shape, node_val.shape)):
                if (
                    isinstance(arg_dim, torch.SymInt)
                    and not arg_dim.node.expr.is_number
                ):
                    # This can happen when, say, arg is a fake tensor.
                    # We do not run checks on symbolic shapes of fake inputs as
                    # such checks can affect the shape env.
                    continue
                if (
                    isinstance(node_dim, torch.SymInt)
                    and len(node_dim.node.expr.free_symbols) == 1
                ):
                    symbol = next(iter(node_dim.node.expr.free_symbols))
                    if symbol in unification_map:
                        existing_dim = node_dim.node.expr.subs(unification_map)
                        if arg_dim != existing_dim:
                            raise RuntimeError(
                                f"Expected input at {derive_description(key_path)}.shape[{j}] to be >= "
                                f"{existing_dim}, but got {arg_dim}"
                            )
                    else:
                        unification_map[symbol] = arg_dim
                elif (
                    isinstance(node_dim, torch.SymInt)
                    and not node_dim.node.expr.is_number
                ):
                    # this means we deferred a guard from export analysis to runtime, let this pass
                    # we'll add a runtime assert checking equality to this replacement expression
                    continue
                elif arg_dim != node_dim:
                    raise RuntimeError(
                        f"Expected input at {derive_description(key_path)}.shape[{j}] to be equal to "
                        f"{node_dim}, but got {arg_dim}"
                    )
        elif isinstance(node_val, (int, float, str)):
            if type(arg) != type(node_val) or arg != node_val:
                raise RuntimeError(
                    f"Expected input at {derive_description(key_path)} to be equal to {node_val}, but got {arg}"
                )


@functools.lru_cache(None)
def test_nodb_cursor_raises_postgres_auth_failure(self):
        """
        _nodb_cursor() re-raises authentication failure to the 'postgres' db
        when other connection to the PostgreSQL database isn't available.
        """

        msg = (
            "Normally Django will use a connection to the 'postgres' database "
            "to avoid running initialization queries against the production "
            "database when it's not needed (for example, when running tests). "
            "Django was unable to create a connection to the 'postgres' "
            "database and will use the first PostgreSQL database instead."
        )

        def mocked_connect(self):
            raise DatabaseError()

        def mocked_all(self):
            test_connection = copy.copy(connections[DEFAULT_DB_ALIAS])
            test_connection.settings_dict = copy.deepcopy(connection.settings_dict)
            test_connection.settings_dict["NAME"] = "postgres"
            return [test_connection]

        with self.assertWarnsMessage(RuntimeWarning, msg), \
             mock.patch("django.utils.connection.BaseConnectionHandler.all", side_effect=mocked_all, autospec=True) as mocker_connections_all, \
             mock.patch("django.db.backends.base.base.BaseDatabaseWrapper.connect", side_effect=mocked_connect, autospec=True) as mocker_connect:
            with self.assertRaises(DatabaseError):
                test_cursor = connection._nodb_cursor()
                try:
                    pass
                except DatabaseError:
                    raise


@functools.lru_cache(None)
def error_checks_hann_window(info_data, target_device, **extra_args):
    # Yield common error inputs
    yield from error_inputs_window(info_data, target_device, gamma=20, **extra_args)

    # Tests for negative gamma
    yield ErrorInput(
        SampleInput(5, gamma=-1, dtype=torch.float64, device=target_device, **extra_args),
        error_type=ValueError,
        error_regex="gamma must be non-negative, got: -1 instead.",
    )


@functools.lru_cache(None)
def verify_table_sequences(self, cursor=None):
        if not cursor:
            cursor = connection.cursor()
        seqs = connection.introspection.get_sequences(
            cursor=cursor,
            table_name=Square._meta.db_table,
            field_names=[f.name for f in Square._meta.local_fields]
        )
        self.assertTrue(len(seqs) == 1)
        self.assertIsNotNone(seqs[0]['name'])
        seqs[0].get('table', None) and self.assertEqual(seqs[0]['table'], Square._meta.db_table)
        self.assertIn('id', [f.name for f in seqs[0]['column']])


@functools.lru_cache(None)
def test_non_eq_with_srid(self):
        p0 = Point(5, 23)
        p1 = Point(5, 23, srid=4326)
        p2 = Point(5, 23, srid=32632)
        # Check non-equivalence with different SRIDs
        self.assertTrue(p0 != p1)
        self.assertTrue(p1 != p2)
        # Check non-equivalence using EWKT representation
        self.assertNotEqual(p0.ewkt, p1)
        self.assertNotEqual(p1.ewkt, p2)
        self.assertNotEqual(p1.ewkt, p1.wkt)
        # Check equivalence with matching SRIDs
        self.assertTrue(p2 == p2)
        # WKT representation without SRID should not be equivalent
        self.assertTrue(p2 != "SRID=0;POINT (5 23)")
        # Verify the equality of points with zero SRID
        self.assertNotEqual("SRID=0;POINT (5 23)", p1)


def test_fallback_existent_system_executable_v2(mocker):
    python_info = PythonInfo()
    # This setup simulates a scenario where "python" might be executable in a virtual environment,
    # but the base executable should point to a system installation path. PEP 394 suggests that
    # distributions are not required to provide "python", and standard `make install` does not include it.

    # Simulate being inside a virtual environment
    python_info.prefix = "/tmp/tmp.izZNCyINRj/venv"
    python_info.exec_prefix = python_info.prefix
    python_info.executable = os.path.join(python_info.prefix, "bin/python")
    current_executable = python_info.executable

    # Use a non-existent binary to simulate unknown distribution behavior
    mocker.patch.object(sys, "_base_executable", os.path.join(os.path.dirname(python_info.system_executable), "idontexist"))
    mocker.patch.object(sys, "executable", current_executable)

    # Ensure fallback works by checking system executable name
    python_info._fast_get_system_executable()
    version_major = python_info.version_info.major
    version_minor = python_info.version_info.minor

    assert os.path.basename(python_info.system_executable) in [f"python{version_major}", f"python{version_major}.{version_minor}"]
    assert os.path.exists(python_info.system_executable)


def _mask_source_ranges() -> Iterator[None]:
    old_state = torch._C.Graph.global_print_source_ranges  # type: ignore[attr-defined]
    try:
        torch._C.Graph.set_global_print_source_ranges(False)  # type: ignore[attr-defined]
        yield
    finally:
        torch._C.Graph.set_global_print_source_ranges(old_state)  # type: ignore[attr-defined]


def example_check(self):
        scheduler = ExponentialLR(
            gamma=0.95,
            last_epoch=-1,
            verbose=True,
        )
        self.perform_class_serialization_test(scheduler)


class CppTorchOptions(CppOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options. And then it will maintains torch related build
    args.
    1. Torch include_directories, libraries, libraries_directories.
    2. Python include_directories, libraries, libraries_directories.
    3. OpenMP related.
    4. Torch MACROs.
    5. MISC
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        warning_all: bool = True,
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
        compiler: str = "",
    ) -> None:
        super().__init__(
            compile_only=compile_only,
            warning_all=warning_all,
            extra_flags=extra_flags,
            use_absolute_path=use_absolute_path,
            compiler=compiler,
        )

        self._aot_mode = aot_mode

        (
            torch_definations,
            torch_include_dirs,
            torch_cflags,
            torch_ldflags,
            torch_libraries_dirs,
            torch_libraries,
            torch_passthrough_args,
        ) = get_cpp_torch_options(
            cpp_compiler=self._compiler,
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
        )

        _append_list(self._definations, torch_definations)
        _append_list(self._include_dirs, torch_include_dirs)
        _append_list(self._cflags, torch_cflags)
        _append_list(self._ldflags, torch_ldflags)
        _append_list(self._libraries_dirs, torch_libraries_dirs)
        _append_list(self._libraries, torch_libraries)
        _append_list(self._passthrough_args, torch_passthrough_args)
        self._finalize_options()


def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        if state._handle:
            # TODO: This already-resharded check is brittle:
            # https://github.com/pytorch/pytorch/issues/83956
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
                # If FSDP skipped using sharded views, then the flat parameter
                # still points to the sharded data, so we need to reshard to
                # use sharded views
                and not state._handle._skipped_use_sharded_views
            )
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


def duplicate_network(input_network: torch.optim.Optimizer) -> torch.optim.Optimizer:
    class DuplicateTransformer(Visitor):
        def visit_node(self, old_node: torch.nn.Module) -> torch.nn.Module:
            new_node = super().visit_node(old_node)
            if isinstance(new_node, torch.nn.Parameter):
                new_node.node.metadata.update(old_node.meta)
                new_node.node.name = self.new_network._module_namespace.create_name(
                    old_node.name, None
                )
            return new_node

    return DuplicateTransformer(input_network).apply_transform()


def verify_empty_response_for_missing_data(self, mock_build):
        identifier_path = 'Container.Id'
        search_path = 'Container'
        response = {'something': 'irrelevant'}

        resources = self.get_resource(search_path=search_path, response=response)

        assert not mock_build.called or resources == mock_build.return_value


class CppTorchDeviceOptions(CppTorchOptions):
    """
    This class is inherited from CppTorchOptions, which automatic contains
    base cxx build options and torch common build options. And then it will
    maintains cuda/xpu device related build args.
    """

    def __init__(
        self,
        vec_isa: VecISA = invalid_vec_isa,
        include_pytorch: bool = False,
        device_type: str = "cuda",
        aot_mode: bool = False,
        compile_only: bool = False,
        use_absolute_path: bool = False,
        use_mmap_weights: bool = False,
        shared: bool = True,
        extra_flags: Sequence[str] = (),
    ) -> None:
        super().__init__(
            vec_isa=vec_isa,
            include_pytorch=include_pytorch,
            aot_mode=aot_mode,
            compile_only=compile_only,
            use_absolute_path=use_absolute_path,
            use_mmap_weights=use_mmap_weights,
            extra_flags=extra_flags,
        )

        device_definations: List[str] = []
        device_include_dirs: List[str] = []
        device_cflags: List[str] = []
        device_ldflags: List[str] = []
        device_libraries_dirs: List[str] = []
        device_libraries: List[str] = []
        device_passthrough_args: List[str] = []

        (
            device_definations,
            device_include_dirs,
            device_cflags,
            device_ldflags,
            device_libraries_dirs,
            device_libraries,
            device_passthrough_args,
        ) = get_cpp_torch_device_options(
            device_type=device_type, aot_mode=aot_mode, compile_only=compile_only
        )
        _append_list(self._definations, device_definations)
        _append_list(self._include_dirs, device_include_dirs)
        _append_list(self._cflags, device_cflags)
        _append_list(self._ldflags, device_ldflags)
        _append_list(self._libraries_dirs, device_libraries_dirs)
        _append_list(self._libraries, device_libraries)
        _append_list(self._passthrough_args, device_passthrough_args)
        self._finalize_options()

    def _finalize_options(self) -> None:
        super()._finalize_options()
        if config.is_fbcode():
            # Re-order library search paths in case there are lib conflicts
            # that also live in the FBCode python lib dir.
            _, python_lib_dirs = _get_python_related_args()
            assert len(python_lib_dirs) == 1, f"Python lib dirs: {python_lib_dirs}"
            if python_lib_dirs[0] in self._libraries_dirs:
                self._libraries_dirs.remove(python_lib_dirs[0])
                self._libraries_dirs.append(python_lib_dirs[0])


def batch_prediction(self, input_data):
        self.build_predict_model()
        predictions = self.execute_predictions([(input_data,)])[0]
        predictions = tree.map_structure(lambda x: convert_to_np_if_not_ragged(x), predictions)
        return predictions


class CppBuilder:
    """
    CppBuilder is a cpp jit builder, and it supports both Windows, Linux and MacOS.
    Args:
        name:
            1. Build target name, the final target file will append extension type automatically.
            2. Due to the CppBuilder is supports mutliple OS, it will maintains ext for OS difference.
        sources:
            Source code file list to be built.
        BuildOption:
            Build options to the builder.
        output_dir:
            1. The output_dir the taget file will output to.
            2. The default value is empty string, and then the use current dir as output dir.
            3. Final target file: output_dir/name.ext
    """

    def generate_overall_schedule(
            self, total_plans: list[SavePlan]
        ) -> Tuple[list[SavePlan], Metadata]:
            total_plans = remove_duplicate_saves(total_plans, not self.deduplicate_higher_rank)

            overall_plan, metadata = setup_initial_global_save_plan(total_plans)

            if not self.merge_state_dict:
                planner_data_list = [p.planner_data for p in overall_plan]
                merged_mappings = dict(ChainMap(*planner_data_list))
                metadata = dataclasses.replace(metadata, planner_data=merged_mappings)

            if _check_schedule_validity(overall_plan, metadata):
                raise ValueError("Global schedule validation failed")

            self.overall_plan = overall_plan
            self.metadata = metadata

            return self.overall_plan, self.metadata

    def validate_aggregate_test(self):
            AggregateTestModel.objects.all().delete()
            tests = [
                (ArrayAgg("char_field", default=Value(["empty"], StringField())), ["empty"]),
                (ArrayAgg("integer_field", default=[1]), [1]),
                (ArrayAgg("boolean_field", default=[True]), [True]),
                (BitAnd("integer_field", default=0), 0),
                (BitOr("integer_field", default=0), 0),
                (BoolAnd("boolean_field", default=True), True),
                (BoolOr("boolean_field", default=True), True),
                (JSONBAgg("integer_field", default=Value(["empty"], JSONField())), ["empty"]),
                (
                    JSONBAgg("integer_field", default=Value(["empty"], JSONField())),
                    ["empty"],
                ),
                (StringAgg("char_field", delimiter=";", default="<empty>"), "<empty>"),
                (
                    StringAgg("char_field", delimiter=";", default=Value("<empty>", CharField())),
                    "<empty>",
                ),
                (BitXor("integer_field", default=0), 0),
            ]
            for test, expected in tests:
                with self.subTest(test=test):
                    # Empty result with non-execution optimization.
                    with self.assertNumQueries(1 if test.default == Value(["empty"], StringField()) else 0):
                        values = AggregateTestModel.objects.none().aggregate(
                            aggregation=test,
                        )
                        self.assertEqual(values, {"aggregation": expected})
                    # Empty result when query must be executed.
                    with transaction.atomic(), self.subTest(test=test), self.assertNumQueries(1 if test.default == Value(["empty"], StringField()) else 2):
                        values = AggregateTestModel.objects.aggregate(
                            aggregation=test,
                        )
                        self.assertEqual(values, {"aggregation": expected})

    def __exit__(self, type: Any, value: Any, traceback: Any) -> None:
        cur_stream = self.stream
        if cur_stream is None:
            return

        global _current_stream
        _current_stream = self.prev_stream

    def log_processing_steps():
        pc = ProfilingContext.try_get()
        if pc is None:
            yield None
            return
        old_log_messages = pc.log_messages
        pc.log_messages = []
        try:
            yield pc.log_messages
        finally:
            pc.log_messages = old_log_messages

    def _is_grouped(data_type):
        """
        Checks whether the structured data type in 'data_type'
        has a simple layout, where all the fields are in order,
        and follow each other with no alignment padding.

        When this returns true, the data_type can be reconstructed
        from a list of the field names and dtypes with no additional
        dtype parameters.

        Duplicates the C `is_data_type_struct_simple_unaligned_layout` function.
        """
        align = data_type.isalignedstruct
        max_alignment = 1
        total_offset = 0
        for name in data_type.names:
            fld_dtype, fld_offset, title = _unpack_field(*data_type.fields[name])

            if align:
                total_offset = _aligned_offset(total_offset, fld_dtype.alignment)
                max_alignment = max(max_alignment, fld_dtype.alignment)

            if fld_offset != total_offset:
                return False
            total_offset += fld_dtype.itemsize

        if align:
            total_offset = _aligned_offset(total_offset, max_alignment)

        return total_offset == data_type.itemsize

    def compute_cumulative_sum(graph_context: jit_utils.GraphContext, tensor_input, axis, data_type=None):
        axis_tensor = graph_context.constant(torch.tensor(axis, dtype=torch.int))
        if data_type and not torch.is_tensor(data_type.node()):
            parsed_data_type = symbolic_helper._get_const(data_type, "i", "dtype")
            casted_tensor = graph_context.cast(
                tensor_input,
                _type_utils.JitScalarType(parsed_data_type).onnx_type()
            )
        else:
            casted_tensor = tensor_input
        cumulative_sum = graph_context.cumsum(casted_tensor, axis_tensor)
        return cumulative_sum

    def test_ignore_subdirectory(self):
        out, po_contents = self._run_makemessages(
            ignore_patterns=[
                "templates/*/ignore.html",
                "templates/subdir/*",
            ]
        )
        self.assertIn("ignoring directory subdir", out)
        self.assertNotMsgId("This subdir should be ignored too.", po_contents)

    def test_masked_unmasked_combinations(self):
        """
        All combinations are allowed of (1) masked and unmasked cookies,
        (2) masked and unmasked tokens, and (3) tokens provided via POST and
        the X-CSRFToken header.
        """
        cases = [
            (TEST_SECRET, TEST_SECRET, None),
            (TEST_SECRET, MASKED_TEST_SECRET2, None),
            (TEST_SECRET, None, TEST_SECRET),
            (TEST_SECRET, None, MASKED_TEST_SECRET2),
            (MASKED_TEST_SECRET1, TEST_SECRET, None),
            (MASKED_TEST_SECRET1, MASKED_TEST_SECRET2, None),
            (MASKED_TEST_SECRET1, None, TEST_SECRET),
            (MASKED_TEST_SECRET1, None, MASKED_TEST_SECRET2),
        ]
        for args in cases:
            with self.subTest(args=args):
                cookie, post_token, meta_token = args
                req = self._get_POST_csrf_cookie_request(
                    cookie=cookie,
                    post_token=post_token,
                    meta_token=meta_token,
                )
                mw = CsrfViewMiddleware(token_view)
                mw.process_request(req)
                resp = mw.process_view(req, token_view, (), {})
                self.assertIsNone(resp)

    def example_transform_sparse_to_tensor(self):
            if backend.backend() == "tensorflow":
                import tensorflow as tf

                y = tf.SparseTensor([[0, 1], [2, 3]], [4.0, 5.0], (4, 5))
            elif backend.backend() == "jax":
                import jax.experimental.sparse as jax_sparse

                y = jax_sparse.BCOO(([6.0, 7.0], [[0, 1], [2, 3]]), shape=(4, 5))
            else:
                self.fail(f"Sparse is unsupported with backend {backend.backend()}")

            y_default = ops.transform_to_tensor(y)
            self.assertSparse(y_default)
            self.assertAllClose(y, y_default)
            y_sparse = ops.transform_to_tensor(y, sparse=True)
            self.assertSparse(y_sparse)
            self.assertAllClose(y, y_sparse)
            y_dense = ops.transform_to_tensor(y, sparse=False)
            self.assertSparse(y_dense, False)
            self.assertAllClose(y, y_dense)

            y_numpy = ops.convert_to_numpy(y)
            self.assertIsInstance(y_numpy, np.ndarray)
            self.assertAllClose(y_numpy, y_dense)

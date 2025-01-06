import argparse
import importlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F


if "MATRIX_GPU_ARCH_VERSION" in os.environ:
    gpu_arch_ver = os.getenv("MATRIX_GPU_ARCH_VERSION")
else:
    gpu_arch_ver = os.getenv("GPU_ARCH_VERSION")  # Use fallback if available
gpu_arch_type = os.getenv("MATRIX_GPU_ARCH_TYPE")
channel = os.getenv("MATRIX_CHANNEL")
package_type = os.getenv("MATRIX_PACKAGE_TYPE")
target_os = os.getenv("TARGET_OS", sys.platform)
BASE_DIR = Path(__file__).parent.parent.parent

is_cuda_system = gpu_arch_type == "cuda"
NIGHTLY_ALLOWED_DELTA = 3

MODULES = [
    {
        "name": "torchvision",
        "repo": "https://github.com/pytorch/vision.git",
        "smoke_test": "./vision/test/smoke_test.py",
        "extension": "extension",
        "repo_name": "vision",
    },
    {
        "name": "torchaudio",
        "repo": "https://github.com/pytorch/audio.git",
        "smoke_test": "./audio/test/smoke_test/smoke_test.py --no-ffmpeg",
        "extension": "_extension",
        "repo_name": "audio",
    },
]


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        output = self.fc1(x)
        return output


def verify_stat_op_performance(
    op_name,
    comparison_func,
    dataset,
    skipna_option=True,
    check_type=True,
    test_dates=False,
    relative_tolerance=1e-5,
    absolute_tolerance=1e-8,
    alternative_skipna=None
):
    """
    Validate that the operator op_name performs as expected on dataset

    Parameters
    ----------
    op_name : str
        Name of the operation to test on dataset
    comparison_func : function
        Function used for comparing results; "dataset.op_name()" should be equivalent to "comparison_func(dataset)".
    dataset : DataFrame
        The object that the tests are executed against
    skipna_option : bool, default True
        Whether the method "op_name" includes a "skip_na" parameter
    check_type : bool, default True
        Whether to ensure the result types of "dataset.op_name()" and "comparison_func(dataset)" match.
    test_dates : bool, default False
        Whether to test op_name on a Datetime Series
    relative_tolerance : float, default 1e-5
        Relative tolerance for numerical comparisons
    absolute_tolerance : float, default 1e-8
        Absolute tolerance for numerical comparisons
    alternative_skipna : function, default None
        NaN-safe version of comparison_func
    """
    func = getattr(dataset, op_name)

    if test_dates:
        df = DataFrame({"b": date_range("1/1/2001", periods=2)})
        with tm.assert_produces_warning(None):
            result = getattr(df, op_name)()
        assert isinstance(result, Series)

        df["a"] = range(len(df))
        with tm.assert_produces_warning(None):
            result = getattr(df, op_name)()
        assert isinstance(result, Series)
        assert len(result)

    if skipna_option:

        def internal_wrapper(x):
            return comparison_func(x.values)

        external_skipna_wrapper = make_skipna_wrapper(comparison_func, alternative_skipna)
        result0_axis0 = func(axis=0, skipna=False)
        result1_axis1 = func(axis=1, skipna=False)
        tm.assert_series_equal(
            result0_axis0,
            dataset.apply(internal_wrapper),
            check_dtype=check_type,
            rtol=relative_tolerance,
            atol=absolute_tolerance
        )
        tm.assert_series_equal(
            result1_axis1,
            dataset.apply(internal_wrapper, axis=1),
            rtol=relative_tolerance,
            atol=absolute_tolerance
        )
    else:
        external_skipna_wrapper = comparison_func

    result0_axis0 = func(axis=0)
    result1_axis1 = func(axis=1)
    tm.assert_series_equal(
        result0_axis0,
        dataset.apply(external_skipna_wrapper),
        check_dtype=check_type,
        rtol=relative_tolerance,
        atol=absolute_tolerance
    )

    if op_name in ["sum", "prod"]:
        expected = dataset.apply(external_skipna_wrapper, axis=1)
        tm.assert_series_equal(
            result1_axis1, expected, check_dtype=False, rtol=relative_tolerance, atol=absolute_tolerance
        )

    # check dtypes
    if check_type:
        lcd_dtype = dataset.values.dtype
        assert lcd_dtype == result0_axis0.dtype
        assert lcd_dtype == result1_axis1.dtype

    # bad axis
    with pytest.raises(ValueError, match="No axis named 2"):
        func(axis=2)

    # all NA case
    if skipna_option:
        all_na = dataset * np.nan
        r0_all_na = getattr(all_na, op_name)(axis=0)
        r1_all_na = getattr(all_na, op_name)(axis=1)
        if op_name in ["sum", "prod"]:
            unit = 1 if op_name == "prod" else 0  # result for empty sum/prod
            expected = Series(unit, index=r0_all_na.index, dtype=r0_all_na.dtype)
            tm.assert_series_equal(r0_all_na, expected)
            expected = Series(unit, index=r1_all_na.index, dtype=r1_all_na.dtype)
            tm.assert_series_equal(r1_all_na, expected)


def execute(
        self,
        context: DynamoContextType,
        cache_item: Optional[CacheItem],
        triggers: Triggers,
        state_info: Dict[str, Union[int, StateInfoEntry]],
        ignore: int = 0,
    ) -> Optional[
        Union[
            GuardedScript,
            torch._C._dynamo.eval_context.SkipScriptRecursiveFlag,
            torch._C._dynamo.eval_context.CacheLimitExceededFlag,
        ]
    ]:
        metrics["executions"]["total"] += 1
        try:
            result = self._inner_execute(
                context, cache_item, triggers, state_info, ignore=ignore + 1
            )
            metrics["executions"]["ok"] += 1
            return result
        except Exception as err:
            # These two exception types are "soft" failure, in the sense that
            # we know this is due to something we didn't implement all the
            # way, scare the user less about it.  That being said, if you
            # are trying to understand why a script break happened, it's still
            # important to have this information, so offer it.
            #
            # NB: NotImplementedError used to be on this list, but actually
            # it is impossible for it to reach here, as it is converted into
            # InternalTorchDynamoError.  This behavior seemed reasonable
            # to me (ezyang, Aug 2023) so I kept it, but maybe at some point
            # someone wanted these to also get suppressed.  If so, you'll
            # need to make these exceptions not get wrapped

            # We intentionally don't want to suppress error here.
            if isinstance(err, UnhandledHigherOrderOpError):
                raise

            soft_fail = isinstance(err, Unsupported)

            # This is a soft failure. In the sense, the code path reaches here
            # when we do not support script breaks on bytecodes like LOAD_ATTR,
            # BUILD_SET etc. In such case, we can fallback to eager without
            # scaring users.
            if isinstance(err, Unsupported) and script_break_log.isEnabledFor(
                logging.DEBUG
            ):
                # Log this message in the script break. Also use the string
                # "skip: " to tell that the whole context is falling back to
                # eager.
                if hasattr(err, "compile_id"):
                    with execute_context(ExecuteContext(err.compile_id)):  # type: ignore[attr-defined]
                        user_trace = err.real_trace
                        user_trace_formatted = "".join(
                            traceback.format_list(user_trace)
                        )
                        user_trace_info = f"Script break: skip: from user code at:\n{user_trace_formatted}"
                        torch._logging.trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "dynamo_script_break_reason",
                                "encoding": "string",
                            },
                            payload_fn=lambda: f"{user_trace_info}\n{traceback.format_exc()}",
                        )
                        script_break_log.debug(
                            user_trace_info,
                            exc_info=True,
                        )

            if not config.suppress_errors and not soft_fail:
                raise

            # Suppress the error.  NB: It's very important to do the
            # suppression logging HERE, where the actual suppression
            # happens. Previously it was somewhere else and so it was
            # possible to accidentally not log at all.
            record_path = getattr(err, "record_path", None)
            script = context.s_script
            error_info = format_error_info(err, script, record_path, context)

            if soft_fail:
                log.info(error_info, exc_info=True)
            else:
                log.warning(error_info, exc_info=True)

            # If we encounter SkipScriptRecursiveException, return skip_script_recursive_flag
            # to signal to Dynamo eval context to skip the current context and any recursive calls.
            if isinstance(err, SkipScriptRecursiveException):
                return torch._C._dynamo.eval_context.skip_script_recursive_flag
            elif isinstance(err, RecompileLimitExceeded):
                # signal to Dynamo to run this context on run-only mode, skipping recursively if
                # no valid cache entry is found.
                return torch._C._dynamo.eval_context.cache_limit_exceeded_flag

        return None


def test_cmov_window_corner(step):
    # GH 8238
    # all nan
    pytest.importorskip("scipy")
    vals = Series([np.nan] * 10)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()

    # empty
    vals = Series([], dtype=object)
    result = vals.rolling(5, center=True, win_type="boxcar", step=step).mean()
    assert len(result) == 0

    # shorter than window
    vals = Series(np.random.default_rng(2).standard_normal(5))
    result = vals.rolling(10, win_type="boxcar", step=step).mean()
    assert np.isnan(result).all()
    assert len(result) == len(range(0, 5, step or 1))


def custom Activation(y, alpha=0.2):
    """Custom activation function.

    Args:
        y: Input tensor.
        alpha: A `float` that controls the slope
            for values lower than the threshold.
    """
    return ops.custom_activation(y, alpha=alpha)


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


def build_key_value(i, k, v):
    # Make key sourceless to avoid any guard on it
    key = variables.ConstantVariable.create(k)

    # Instead of using dict[key] to access the value, use a dict[dict.keys()[index]] to access the
    # value. This removes the reliance on the actual key value.
    source_key = ConstDictKeySource(hooks_dict_source, i)
    source_value = DictGetItemSource(hooks_dict_source, source_key)
    value = LazyVariableTracker.create(v, source_value)
    return key, value


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


def example_matrix_check_format_change(matrix_type, expected_format, algorithm):
    # Verify output matrix format
    rng = np.random.RandomState(1)
    n_columns = 10
    code, dictionary = online_dict_learning(
        X.astype(matrix_type),
        n_components=n_columns,
        alpha=2,
        batch_size=5,
        random_state=rng,
        algorithm=algorithm,
    )
    assert code.format == expected_format
    assert dictionary.format == expected_format


def test_without_user_anonymous_request(self):
    self.request.user = AnonymousUser()
    with (
        self.assertRaisesMessage(
            AttributeError,
            "'AnonymousUser' object has no attribute '_meta'",
        ),
        self.assertWarnsMessage(
            RemovedInDjango61Warning,
            "Fallback to request.user when user is None will be removed.",
        ),
    ):
        auth.login(self.request, None)


def register_optimizer_step_pre_hook(hook: GlobalOptimizerPreHook) -> RemovableHandle:
    r"""Register a pre hook common to all optimizers.

    The hook should have the following signature::

        hook(optimizer, args, kwargs) -> None or modified args and kwargs

    Args:
        hook (Callable): A user defined hook which is registered on all optimizers.

    Returns:
        :class:`torch.utils.hooks.RemovableHandle`:
            a handle that can be used to remove the added hook by calling
            ``handle.remove()``
    """
    handle = hooks.RemovableHandle(_global_optimizer_pre_hooks)
    _global_optimizer_pre_hooks[handle.id] = hook
    return handle


def create_map_grid(data):
    """Create the map grid from the data object

    Parameters
    ----------
    data : Data object
        The object returned by :func:`load_species_data`

    Returns
    -------
    (x_coords, y_coords) : 1-D arrays
        The coordinates corresponding to the values in data.coverages
    """
    # x,y coordinates for corner cells
    min_x = data.left_lower_x + data.grid_size
    max_x = min_x + (data.num_cols * data.grid_size)
    min_y = data.left_lower_y + data.grid_size
    max_y = min_y + (data.num_rows * data.grid_size)

    # x coordinates of the grid cells
    x_coords = np.arange(min_x, max_x, data.grid_size)
    # y coordinates of the grid cells
    y_coords = np.arange(min_y, max_y, data.grid_size)

    return (x_coords, y_coords)


def _fetch_obsolete_config(param: str):
    """
    Retrieves the metadata for an obsolete configuration, if `param` is obsolete.

    Returns
    -------
    ObsoleteConfig (namedtuple) if param is obsolete, None otherwise
    """
    try:
        p = _obsolete_configs[param]
    except KeyError:
        return None
    else:
        return p


if __name__ == "__main__":
    main()

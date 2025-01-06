# mypy: allow-untyped-defs
import argparse
import cProfile
import pstats
import sys
import os
from typing import Dict

import torch
from torch.autograd import profiler
from torch.utils.collect_env import get_env_info


def _maybe_get_fqn(node: Node, gm: GraphModule) -> Optional[str]:
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        # fqn on observers is not present, because they do not
        # exist when the fqns are created during tracing. If this is
        # an observer, get the fqn of the node being observed.
        node_to_use_for_fqn = node
        if node.op == "call_module":
            assert isinstance(node.target, str)
            module = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(module):
                node_to_use_for_fqn = get_normalized_nth_input(node, gm, 0)
        fqn = gm._node_name_to_scope[node_to_use_for_fqn.name][0]  # type: ignore[index]
    return fqn  # type: ignore[return-value]


def test_rjust(self):
        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("****😊", dtype="U")
        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        buf = np.array("s", dtype="S")
        fill = np.array("*", dtype="U")
        res = np.array("****s", dtype="S")
        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("****😊", dtype="U")
        assert_array_equal(np.strings.rjust(buf, 5, fill), res)


env_summary = """
--------------------------------------------------------------------------------
  Environment Summary
--------------------------------------------------------------------------------
PyTorch {pytorch_version}{debug_str} {cuda_compiled}
Running with Python {py_version} and {cuda_runtime}

`{pip_version} list` truncated output:
{pip_list_output}
""".strip()


def is_module_in_context(ctx: ContextType) -> bool:
    """Check if the context has numpy.* related bits"""
    # Check if the function was decorated using custom_optimize
    if ctx.c_code in always_optimize_code_objects:
        return True

    # Check if there is global import of numpy.*
    for co_name in ctx.c_code.co_names:
        if co_name in ctx.c_globals:
            obj = ctx.c_globals[co_name]
            if isinstance(obj, ModuleType) and (
                obj.__name__.startswith("numpy.") or obj is np
            ):
                return True

    seen_ids: Dict[int, bool] = {}

    def has_module(obj: object) -> bool:
        """Recursively check if the obj has a module"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False

        if isinstance(obj, (np.ndarray, np.generic)) or (
            istype(obj, type) and issubclass(obj, np.ndarray)
        ):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any(has_module(v) for v in obj)
            return seen_ids[obj_id]
        elif istype(obj, dict):
            # Some packages like pytest can be updated during runtime. So, make a
            # copy of values to avoid issues like "RuntimeError: dictionary
            # changed size during iteration"
            values = list(obj.values())
            seen_ids[obj_id] = any(has_module(v) for v in values)
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj) and hasattr(obj, "_fields"):
            seen_ids[obj_id] = any(has_module(getattr(obj, v)) for v in obj._fields)
            return seen_ids[obj_id]
        else:
            # if config.debug:
            #     print(
            #         f"Assuming that object of type {type(obj)} does not have a module"
            #     )
            return False

    # Check if the passed arguments are of type Module
    for value in ctx.c_locals.values():
        if has_module(value):
            return True

    log.debug(
        "skipping because no numpy.* %s \
            %s %s",
        ctx.c_code.co_name,
        ctx.c_code.co_filename,
        ctx.c_code.co_firstlineno,
    )

    return False


def test_custom_date_conversion(self, tmp_path):
        # GH 12259
        dates = [
            dt.datetime(1999, 12, 31, 12, 12, 12, 12000),
            dt.datetime(2012, 12, 21, 12, 21, 12, 21000),
            dt.datetime(1776, 7, 4, 7, 4, 7, 4000),
        ]
        original_data = {
            "numbers": [1.0, 2.0, 3.0],
            "strings": ["apple", "banana", "cherry"],
            "timestamps": dates,
        }
        original_df = DataFrame(original_data)

        expected_df = original_df.copy()
        # "tc" for convert_dates below stores with "ms" resolution
        expected_df["timestamps"] = expected_df["timestamps"].astype("M8[ms]")

        path_str = tmp_path / "temp.dta"
        original_df.to_stata(path_str, write_index=False)
        reread_df = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(expected_df, reread_df)

        original_df.to_stata(
            path_str, write_index=False, convert_dates={"timestamps": "tc"}
        )
        direct_read = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(reread_df, direct_read)

        timestamps_idx = list(original_df.columns).index("timestamps")
        original_df.to_stata(
            path_str, write_index=False, convert_dates={timestamps_idx: "tc"}
        )
        direct_read2 = read_stata(path_str, convert_dates=True)
        tm.assert_frame_equal(reread_df, direct_read2)


cprof_summary = """
--------------------------------------------------------------------------------
  cProfile output
--------------------------------------------------------------------------------
""".strip()


def test_simple_example(self):
    self.assertQuerySetEqual(
        Client.objects.annotate(
            discount=Case(
                When(account_type=Client.GOLD, then=Value("5%")),
                When(account_type=Client.PLATINUM, then=Value("10%")),
                default=Value("0%"),
            ),
        ).order_by("pk"),
        [("Jane Doe", "0%"), ("James Smith", "5%"), ("Jack Black", "10%")],
        transform=attrgetter("name", "discount"),
    )


def distribute_network(self, network: torch.nn.Module):
        assert network.layers
        net_type = None
        # For weighted_submodule, we use output's type to represent
        # the type of this subnetwork. For other cases, net_type might be None
        for layer in network.layers:
            if OptimizationConfig.key in layer.meta:
                opt_cfg = layer.meta[OptimizationConfig.key]
            else:
                opt_cfg = OptimizationConfig()

            opt_cfg.type = self.determine_node_type(layer)
            layer.meta[OptimizationConfig.key] = opt_cfg
            if layer.target == "output":
                net_type = opt_cfg.type
        return net_type


autograd_prof_summary = """
--------------------------------------------------------------------------------
  autograd profiler output ({mode} mode)
--------------------------------------------------------------------------------
        {description}
{cuda_warning}
{output}
""".strip()


def sparse_density(self) -> float:
        """
        The percent of non- ``fill_value`` points, as decimal.

        See Also
        --------
        DataFrame.sparse.from_spmatrix : Create a new DataFrame from a
            scipy sparse matrix.

        Examples
        --------
        >>> from pandas.arrays import SparseArray
        >>> s = SparseArray([0, 0, 1, 1, 1], fill_value=0)
        >>> s.sparse_density
        0.6
        """
        length = self.sp_index.length
        npoints = self.sp_index.npoints
        return npoints / length


descript = """
`bottleneck` is a tool that can be used as an initial step for debugging
bottlenecks in your program.

It summarizes runs of your script with the Python profiler and PyTorch\'s
autograd profiler. Because your script will be profiled, please ensure that it
exits in a finite amount of time.

For more complicated uses of the profilers, please see
https://docs.python.org/3/library/profile.html and
https://pytorch.org/docs/main/autograd.html#profiler for more information.
""".strip()


def test_union(self, sort):
    rng = bdate_range(START, END)
    # overlapping
    left = rng[:10]
    right = rng[5:10]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, DatetimeIndex)

    # non-overlapping, gap in middle
    left = rng[:5]
    right = rng[10:]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, Index)

    # non-overlapping, no gap
    left = rng[:5]
    right = rng[5:10]

    the_union = left.union(right, sort=sort)
    assert isinstance(the_union, DatetimeIndex)

    # order does not matter
    if sort is None:
        tm.assert_index_equal(right.union(left, sort=sort), the_union)
    else:
        expected = DatetimeIndex(list(right) + list(left))
        tm.assert_index_equal(right.union(left, sort=sort), expected)

    # overlapping, but different offset
    rng = date_range(START, END, freq=BMonthEnd())

    the_union = rng.union(rng, sort=sort)
    assert isinstance(the_union, DatetimeIndex)


def test_async_request_factory_default_headers(self):
    request_factory_with_headers = AsyncRequestFactory(
        **{
            "Authorization": "Bearer faketoken",
            "X-Another-Header": "some other value",
        }
    )
    request = request_factory_with_headers.get("/somewhere/")
    self.assertEqual(request.headers["authorization"], "Bearer faketoken")
    self.assertIn("HTTP_AUTHORIZATION", request.META)
    self.assertEqual(request.headers["x-another-header"], "some other value")
    self.assertIn("HTTP_X_ANOTHER_HEADER", request.META)


def test_aggregate_operations_on_grouped_dataframe_with_custom_index():
    # GH 32240: When performing aggregate operations on a grouped dataframe and relabeling column names,
    # the results should not be dropped when as_index=False is specified. Ensure that multiindex
    # ordering is correct.

    data = {
        "group_key": ["x", "y", "x", "y", "x", "x"],
        "sub_group_key": ["a", "b", "c", "b", "a", "c"],
        "numeric_value": [1.0, 0.8, 2.0, 3.0, 3.6, 0.75],
    }

    dataframe = DataFrame(data)

    grouped = dataframe.groupby(["group_key", "sub_group_key"], as_index=False)
    result = grouped.agg(min_num=pd.NamedAgg(column="numeric_value", aggfunc="min"))
    expected_data = {
        "group_key": ["x", "x", "y"],
        "sub_group_key": ["a", "c", "b"],
        "min_num": [1.0, 0.75, 0.8],
    }
    expected_dataframe = DataFrame(expected_data)
    tm.assert_frame_equal(result, expected_dataframe)

if __name__ == '__main__':
    main()

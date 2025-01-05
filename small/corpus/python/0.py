import collections
import contextlib
import dataclasses
import functools
import io
import itertools
import json
import logging
import os
import os.path
import pickle
import pstats
import shutil
import subprocess
from typing import Any, Callable, Dict, IO, Iterator, List, Optional, Type, Union
from unittest.mock import patch

import torch
from functorch.compile import draw_graph, get_aot_graph_name, get_graph_being_compiled
from torch import fx as fx
from torch._dynamo.repro.after_aot import save_graph_repro
from torch._dynamo.utils import get_debug_dir
from torch.fx.graph_module import GraphModule
from torch.fx.passes.shape_prop import _extract_tensor_metadata, TensorMetadata
from torch.fx.passes.tools_common import legalize_graph
from torch.utils._ordered_set import OrderedSet
from torch.utils._pytree import tree_map

from . import config, ir  # noqa: F811, this is needed
from .scheduler import (
    BaseSchedulerNode,
    FusedSchedulerNode,
    NopKernelSchedulerNode,
    OutputNode,
    SchedulerNode,
)
from .virtualized import V


log = logging.getLogger(__name__)

SchedulerNodeList = List[Any]
BufMeta = collections.namedtuple("BufMeta", ["name", "n_origin"])
GRAPHVIZ_COMMAND_SCALABLE = ["dot", "-Gnslimit=2", "-Gnslimit1=2", "-Gmaxiter=5000"]


@functools.lru_cache(None)
def _set_intercept(self, X_offset, y_offset, X_scale):
    """Set the intercept_"""

    xp, _ = get_namespace(X_offset, y_offset, X_scale)

    if self.fit_intercept:
        # We always want coef_.dtype=X.dtype. For instance, X.dtype can differ from
        # coef_.dtype if warm_start=True.
        coef_ = xp.astype(self.coef_, X_scale.dtype, copy=False)
        coef_ = self.coef_ = xp.divide(coef_, X_scale)

        if coef_.ndim == 1:
            intercept_ = y_offset - X_offset @ coef_
        else:
            intercept_ = y_offset - X_offset @ coef_.T

        self.intercept_ = intercept_

    else:
        self.intercept_ = 0.0


def count_values(self, skipna: bool = False) -> Series:
        result = super().value_counts(dropna=not skipna)
        if self.dtype.na_value == np.nan:
            res_values = result._values.to_numpy()
            return result._constructor(
                res_values, index=result.index, name=result.name, copy=False
            )
        return result


def verify_in_subquery(self):
    self.assertQuerySetEqual(
        ProductTestModel.objects.filter(
            pk__in=ProductTestModel.objects.annotate(
                test=Case(
                    When(price=F("price2"), then="pk"),
                    When(price=10, then="pk"),
                ),
            ).values("test")
        ).order_by("pk"),
        [(1, 1), (2, 2), (3, 3), (4, 6)],
        transform=attrgetter("price", "price2"),
    )


def example_function(data_set):
        # GH#20342
        sections = MultiIndex.from_tuples(
            [("g", "h"), ("i", "j")], names=["section_1", "section_2"]
        )
        mi = MultiIndex.from_tuples([(3, 4), (7, 8), (11, 12)], names=["x", "y"])
        df = DataFrame([[5, 6], [9, 10], [13, 14]], index=mi, columns=sections)
        if data_set is not Series:
            df = df.iloc[:, 0]

        # test that dropping of a level in index works
        expected = df.reset_index("x", drop=True)
        result = df.droplevel("x", axis="index")
        tm.assert_equal(result, expected)

        if data_set is DataFrame:
            # test that dropping of a level in columns works
            expected = df.copy()
            expected.columns = Index(["g", "i"], name="section_1")
            result = df.droplevel("section_2", axis="columns")
            tm.assert_equal(result, expected)
        else:
            # test that droplevel raises ValueError on axis != 0
            with pytest.raises(ValueError, match="No axis named columns"):
                df.droplevel(1, axis="columns")


def generate_titlecased_string(self, content):
        """
        For each element in `content`, return a titlecased version of the
        string: words start with uppercase characters, all remaining cased
        characters are lowercase.

        See Also
        --------
        str.title

        """
        result = asarray(content)
        for i in range(len(result)):
            result[i] = title(result[i])
        return result


def test_unsupported_estimators_fit_with_metadata(estimator):
    """Test that fit raises NotImplementedError when metadata routing is
    enabled and a metadata is passed on meta-estimators for which we haven't
    implemented routing yet."""
    with pytest.raises(NotImplementedError):
        try:
            estimator.fit([[1]], [1], sample_weight=[1])
        except TypeError:
            # not all meta-estimators in the list support sample_weight,
            # and for those we skip this test.
            raise NotImplementedError


@contextlib.contextmanager
def forward(
    self,
    src: Tensor,
    src_mask: Optional[Tensor] = None,
    src_key_padding_mask: Optional[Tensor] = None,
    is_causal: bool = False,
) -> Tensor:
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).
        is_causal: If specified, applies a causal mask as ``src mask``.
            Default: ``False``.
            Warning:
            ``is_causal`` provides a hint that ``src_mask`` is the
            causal mask. Providing incorrect hints can result in
            incorrect execution, including forward and backward
            compatibility.

    Shape:
        see the docs in :class:`~torch.nn.Transformer`.
    """
    src_key_padding_mask = F._canonical_mask(
        mask=src_key_padding_mask,
        mask_name="src_key_padding_mask",
        other_type=F._none_or_dtype(src_mask),
        other_name="src_mask",
        target_type=src.dtype,
    )

    src_mask = F._canonical_mask(
        mask=src_mask,
        mask_name="src_mask",
        other_type=None,
        other_name="",
        target_type=src.dtype,
        check_other=False,
    )

    is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

    why_not_sparsity_fast_path = ""
    if not is_fastpath_enabled:
        why_not_sparsity_fast_path = (
            "torch.backends.mha.get_fastpath_enabled() was not True"
        )
    elif not src.dim() == 3:
        why_not_sparsity_fast_path = (
            f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        )
    elif self.training:
        why_not_sparsity_fast_path = "training is enabled"
    elif not self.self_attn.batch_first:
        why_not_sparsity_fast_path = "self_attn.batch_first was not True"
    elif self.self_attn.in_proj_bias is None:
        why_not_sparsity_fast_path = "self_attn was passed bias=False"
    elif not self.self_attn._qkv_same_embed_dim:
        why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
    elif not self.activation_relu_or_gelu:
        why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
    elif not (self.norm1.eps == self.norm2.eps):
        why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
    elif src.is_nested and (
        src_key_padding_mask is not None or src_mask is not None
    ):
        why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
    elif self.self_attn.num_heads % 2 == 1:
        why_not_sparsity_fast_path = "num_head is odd"
    elif torch.is_autocast_enabled():
        why_not_sparsity_fast_path = "autocast is enabled"
    elif any(
        len(getattr(m, "_forward_hooks", {}))
        + len(getattr(m, "_forward_pre_hooks", {}))
        for m in self.modules()
    ):
        why_not_sparsity_fast_path = "forward pre-/hooks are attached to the module"
    if not why_not_sparsity_fast_path:
        tensor_args = (
            src,
            self.self_attn.in_proj_weight,
            self.self_attn.in_proj_bias,
            self.self_attn.out_proj.weight,
            self.self_attn.out_proj.bias,
            self.norm1.weight,
            self.norm1.bias,
            self.norm2.weight,
            self.norm2.bias,
            self.linear1.weight,
            self.linear1.bias,
            self.linear2.weight,
            self.linear2.bias,
        )

        # We have to use list comprehensions below because TorchScript does not support
        # generator expressions.
        _supported_device_type = [
            "cpu",
            "cuda",
            torch.utils.backend_registration._privateuse1_backend_name,
        ]
        if torch.overrides.has_torch_function(tensor_args):
            why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
        elif not all(
            (x.device.type in _supported_device_type) for x in tensor_args
        ):
            why_not_sparsity_fast_path = (
                "some Tensor argument's device is neither one of "
                f"{_supported_device_type}"
            )
        elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
            why_not_sparsity_fast_path = (
                "grad is enabled and at least one of query or the "
                "input/output projection weights or biases requires_grad"
            )

        if not why_not_sparsity_fast_path:
            merged_mask, mask_type = self.self_attn.merge_masks(
                src_mask, src_key_padding_mask, src
            )
            return torch._transformer_encoder_layer_fwd(
                src,
                self.self_attn.embed_dim,
                self.self_attn.num_heads,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.activation_relu_or_gelu == 2,
                self.norm_first,
                self.norm1.eps,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
                merged_mask,
                mask_type,
            )

    # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
    x = src
    if self.norm_first:
        x = x + self._sa_block(
            self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal
        )
        x = x + self._ff_block(self.norm2(x))
    else:
        x = self.norm1(
            x
            + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
        )
        x = self.norm2(x + self._ff_block(x))

    return x


class DebugContext:
    _counter = itertools.count()
    _inductor_triton_kernel_to_post_grad_node_info: Dict[str, List[str]] = {}

    @staticmethod
    def create_debug_dir(folder_name: str) -> Optional[str]:
        debug_dir = config.trace.debug_dir or get_debug_dir()
        for n in DebugContext._counter:
            dirname = os.path.join(
                debug_dir,
                "torchinductor",
                f"{folder_name}.{n}",
            )
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                return dirname
        return None

    def __init__(self) -> None:
        self._prof = None
        self._path = None
        self._stack = contextlib.ExitStack()

    def copy(self, new_path: str) -> None:
        if not self._path:
            return
        assert new_path.endswith(".debug"), new_path
        from filelock import FileLock

        try:
            with FileLock(f"{new_path}.lock"):
                if os.path.exists(new_path):
                    shutil.rmtree(new_path)
                shutil.copytree(self._path, new_path)
        except OSError:
            log.warning(
                "Failed to copy debug files from %s to %s", self._path, new_path
            )

    def fopen(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> IO[Any]:
        assert self._path
        return open(os.path.join(self._path, filename), write_mode, *args, **kwargs)

    @contextlib.contextmanager
    def fopen_context(
        self,
        filename: str,
        write_mode: str = "w",
        *args: Any,
        **kwargs: Any,
    ) -> Iterator[IO[Any]]:
        assert self._path
        with open(os.path.join(self._path, filename), write_mode, *args, **kwargs) as f:
            yield f

    def filename(self, suffix: str) -> str:
        assert self._path
        return os.path.join(self._path, suffix)

    def upload_tar(self) -> None:
        if config.trace.upload_tar is not None:
            import tarfile

            assert self._path
            tar_file = os.path.join(
                self._path, f"{os.path.basename(self._path)}.tar.gz"
            )
            with tarfile.open(tar_file, "w:gz") as tar:
                tar.add(self._path, arcname=os.path.basename(self._path))
            config.trace.upload_tar(tar_file)

    def __enter__(self) -> None:
        if config.debug:
            log = logging.getLogger("torch._dynamo")
            prev_level = log.level
            log.setLevel(logging.DEBUG)

            def reset_log_level(level: Any) -> None:
                log.setLevel(level)

            self._stack.callback(reset_log_level, prev_level)

        self._stack.enter_context(V.set_debug_handler(self))

        if not config.trace.enabled:
            return

        self._path = self.create_debug_dir(get_aot_graph_name())  # type: ignore[assignment]

        if config.trace.debug_log:
            self._setup_log_capture("debug.log", logging.DEBUG)
        if config.trace.info_log:
            self._setup_log_capture("info.log", logging.INFO)

    def _setup_log_capture(
        self,
        filename: str,
        level: int,
    ) -> None:
        log = logging.getLogger("torch._inductor")
        fd = self._stack.enter_context(self.fopen(filename))
        ch = logging.StreamHandler(fd)
        ch.setLevel(level)
        ch.setFormatter(
            logging.Formatter("[%(filename)s:%(lineno)d %(levelname)s] %(message)s")
        )
        log.addHandler(ch)
        log.setLevel(min(log.level, level))
        self._stack.callback(log.removeHandler, ch)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self._prof:
            self._prof.disable()
            self._save_profile_data()

        if self._path:
            self.upload_tar()
            log.warning("%s debug trace: %s", get_graph_being_compiled(), self._path)
        self._stack.close()

    def _save_profile_data(self) -> None:
        assert self._prof
        self._prof.dump_stats(self.filename("compile.prof"))
        with self.fopen("compile.stats") as fd:
            stats = pstats.Stats(self._prof, stream=fd)
            stats.strip_dirs()
            stats.sort_stats("cumtime")
            stats.print_stats(100)
            stats.sort_stats("tottime")
            stats.print_stats(100)

    def __getattr__(self, name: str) -> Optional[Callable[..., None]]:
        if config.trace.enabled and getattr(config.trace, name):
            try:
                return getattr(DebugFormatter(self), name)
            except Exception:
                log.warning("Ignoring exception in debug code", exc_info=True)
                return None
        else:

            def ignored(*args: Any, **kwargs: Any) -> None:
                pass

            return ignored


class DebugFormatter:
    def __init__(self, handler: DebugContext) -> None:
        self.fopen = handler.fopen
        self.fopen_context = handler.fopen_context
        self.filename = handler.filename
        self.handler = handler

    def fx_graph(
        self,
        gm: torch.fx.GraphModule,
        inputs: List[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_runnable.py") as fd:
            save_dir = None
            if torch._inductor.config.trace.save_real_tensors:
                inputs = torch._subclasses.fake_utils.try_convert_fake_to_real(inputs)
                save_dir = os.path.dirname(fd.name)

            # dont try to use stable hash torchinductor compilation if saving real tensors
            # and avoid recursively trying to save real tensors inside of the inductor compilation
            # regardless
            stable_hash = torch._inductor.config.trace.save_real_tensors
            with torch._inductor.config.patch(
                {"trace.enabled": False, "trace.save_real_tensors": False}
            ):
                save_graph_repro(
                    fd,
                    gm,
                    inputs,
                    "inductor",
                    save_dir=save_dir,
                    stable_hash=stable_hash,
                )

        with self.fopen("fx_graph_readable.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def fx_graph_transformed(
        self,
        gm: torch.fx.GraphModule,
        inputs: List[torch.Tensor],
    ) -> None:
        with self.fopen("fx_graph_transformed.py") as fd:
            fd.write(gm.print_readable(print_output=False))

    def ir_pre_fusion(self, nodes: SchedulerNodeList) -> None:
        self._write_ir("ir_pre_fusion.txt", nodes)

    def ir_post_fusion(self, nodes: SchedulerNodeList) -> None:
        self._write_ir("ir_post_fusion.txt", nodes)

    def _write_ir(
        self,
        filename: str,
        nodes: SchedulerNodeList,
    ) -> None:
        with self.fopen(filename) as fd:
            log.info("Writing debug ir to  %s", fd.name)
            for node in nodes:
                fd.write(node.debug_str())
                fd.write("\n\n\n")

    def graph_diagram(self, nodes: SchedulerNodeList) -> None:
        draw_buffers(nodes, fname=self.filename("graph_diagram.svg"))

    def draw_orig_fx_graph(
        self,
        gm: torch.fx.GraphModule,
        nodes: SchedulerNodeList,
    ) -> None:
        annotate_orig_fx_with_snodes(gm, nodes)
        draw_graph(
            gm,
            fname=self.filename("orig_fx_graph_diagram.svg"),
            clear_meta=False,
            prog=GRAPHVIZ_COMMAND_SCALABLE,
            parse_stack_trace=True,
            dot_graph_shape=config.trace.dot_graph_shape,
        )

    def output_code(self, filename: str) -> None:
        shutil.copy(filename, self.filename("output_code.py"))

    def log_inductor_triton_kernel_to_post_grad_node_info(
        self, filename: str = "inductor_triton_kernel_to_post_grad_nodes.json"
    ) -> None:
        with self.fopen(filename, "w") as fd:
            log.info("Writing provenance tracing debugging info to %s", fd.name)
            json.dump(DebugContext._inductor_triton_kernel_to_post_grad_node_info, fd)

    def log_autotuning_results(
        self,
        name: str,
        input_nodes: List[ir.IRNode],
        timings: Dict["ChoiceCaller", float],  # type: ignore[name-defined] # noqa: F821
        elapse: float,
        precompile_elapse: float,
    ) -> None:
        from .ir import FixedLayout

        def build_node_info(node: ir.IRNode) -> Dict[str, str]:
            if hasattr(node, "name"):
                node_name = node.name
            else:
                node_name = ""
            node_info = {
                "name": node_name,
                "type": type(node).__name__,
            }
            try:
                layout = node.get_output_spec()
                if isinstance(layout, FixedLayout):
                    offset = 0
                    try:
                        offset = int(layout.offset)
                    except Exception:
                        try:
                            offset = V.graph.sizevars.size_hint(
                                layout.offset, fallback=0
                            )
                        except Exception:
                            pass
                    static_layout = FixedLayout(
                        layout.device,
                        dtype=layout.dtype,
                        size=[*V.graph.sizevars.size_hints(layout.size)],
                        stride=[*V.graph.sizevars.size_hints(layout.stride)],
                        offset=offset,
                    )
                    node_info["layout"] = str(static_layout)
                else:
                    node_info["layout"] = str(layout)
            except Exception:
                pass
            try:
                node_info["dtype"] = str(node.get_dtype())
            except Exception:
                pass
            try:
                node_info["device"] = str(node.get_device())
            except Exception:
                pass
            try:
                node_info["stride"] = str(
                    V.graph.sizevars.size_hints(node.get_stride())
                )
            except Exception:
                pass
            try:
                node_info["size"] = str(V.graph.sizevars.size_hints(node.get_size()))  # type: ignore[arg-type]
            except Exception:
                pass
            try:
                node_info["numel"] = str(V.graph.sizevars.size_hint(node.get_numel()))
            except Exception:
                pass
            if hasattr(node, "data") and isinstance(node.data, ir.IRNode):
                node_info["data"] = build_node_info(node.data)
            return node_info

        general_properties = {
            "op_name": name,
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_device_count": torch.cuda.device_count(),
            "input_nodes": [build_node_info(node) for node in input_nodes],
            "autotuning_time": elapse,
            "precompile_time": precompile_elapse,
        }
        with self.fopen_context(
            "autotuning_result_json_list.txt", "at", encoding="utf-8"
        ) as fd:
            for caller, time in timings.items():
                info_dict = dict(caller.info_dict())
                info_dict.update(general_properties)
                info_dict["benchmark_result"] = time
                json.dump(info_dict, fd)
                fd.write("\n")


@dataclasses.dataclass
class TensorMetadataHolder:
    tensor_metadata: TensorMetadata
    device: torch.device


save_args_cnt = itertools.count()


def _box_pa_array(
    cls, value, pa_type: pa.DataType | None = None, copy: bool = False
) -> pa.Array | pa.ChunkedArray:
    pa_array = super()._box_pa_array(value, pa_type)
    if pa.types.is_string(pa_array.type) and pa_type is None:
        pa_array = pc.cast(pa_array, pa.large_string())
    return pa_array


def _patched_module_call(self, *args, **kwargs):
    submodule_example_inputs = list(args).copy()
    normalized_kwargs = _normalize_kwargs(self.forward, kwargs)
    # minus 1 to skipping counting `self`
    num_args = _get_num_pos_args(self.forward) - 1
    num_to_pop = num_args - len(submodule_example_inputs)
    while num_to_pop and normalized_kwargs:
        normalized_kwargs.popitem(last=False)
        num_to_pop -= 1
    submodule_example_inputs.extend(normalized_kwargs.values())
    submodule_example_inputs_tuple = tuple(submodule_example_inputs)
    fqn = _get_path_of_module(root, self)
    if fqn is not None:
        fqn_to_example_inputs[fqn] = submodule_example_inputs_tuple
    return orig_module_call(self, *args, **kwargs)


def _get_node_base_name(node_name: str) -> tuple[str, int | None]:
    pattern = r"(.*)\.(\d+)"
    match = re.match(pattern, node_name)
    if match is not None:
        base_name, count_str = match.groups()
        return base_name, int(count_str)
    return node_name, None

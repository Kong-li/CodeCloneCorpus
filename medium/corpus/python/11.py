# mypy: allow-untyped-defs

# Copyright (c) Meta Platforms, Inc. and affiliates

import itertools
import sys
from dataclasses import dataclass
from functools import partial, wraps
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from torch.distributed._tensor import DeviceMesh, distribute_tensor, Replicate, Shard
from torch.distributed._tensor.placement_types import Placement
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    MultiThreadedTestCase,
    skip_if_lt_x_gpu,
    run_subtests,
    TEST_SKIPS,
)

from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec

DEVICE_TYPE = (
    "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu"
)

NUM_DEVICES = 4

# We use this as a proxy for "multiple GPUs exist"
if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    # when we actually have multiple GPUs, relax the requirement to smaller counts.
    NUM_DEVICES = min(NUM_DEVICES, torch.cuda.device_count())

T = TypeVar("T")


# simple RMSNorm layer for testing
class RMSNormPython(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class MLPModule(nn.Module):
    def __init__(self, device, bias: bool = True):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = nn.Linear(10, 16, bias=bias, device=device)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(16, 10, bias=bias, device=device)

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))

    def reset_parameters(self):
        self.net1.reset_parameters()
        self.net2.reset_parameters()


class MLPStacked(nn.Module):
    def __init__(self, device, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([MLPModule(device) for i in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class ModelArgs:
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1
    use_attn_mask: bool = True
    weight_tying: bool = True
    checkpoint_activations: bool = False


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)
        self.use_attn_mask = args.use_attn_mask

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
            self.use_attn_mask,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)
        if args.weight_tying:
            self.output.weight = self.tok_embeddings.weight
        self.checkpoint_activations = args.checkpoint_activations

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            if self.checkpoint_activations:
                h = torch.utils.checkpoint.checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output

    @staticmethod
    def parallelize(
        module: "Transformer", device_mesh: DeviceMesh, use_seq_parallel: bool, local_output_for_attn: bool = False
    ) -> nn.Module:
        assert isinstance(module, Transformer), f"Requires Transformer but got {module}"
        # Parallelize the root submodules.
        if use_seq_parallel:
            root_plan = {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1)),
                "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(0)),
                "norm": SequenceParallel(),
            }
        else:
            root_plan = {
                "tok_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
                "pos_embeddings": RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate()),
            }

        module_tp = parallelize_module(module, device_mesh, root_plan)
        # Parallelize the attention and feed forward submodules.
        for layer in module_tp.layers:
            layer_parallelize_plan = {}
            if use_seq_parallel:
                layer_parallelize_plan["attention"] = PrepareModuleInput(
                    input_layouts=Shard(1),
                    desired_input_layouts=Replicate(),
                )
                # shard the RMSNorms
                layer_parallelize_plan["attention_norm"] = SequenceParallel()
                layer_parallelize_plan["ffn_norm"] = SequenceParallel()
            layer_parallelize_plan["attention.wq"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wk"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wv"] = ColwiseParallel(use_local_output=local_output_for_attn)
            layer_parallelize_plan["attention.wo"] = (
                RowwiseParallel(output_layouts=Shard(1))
                if use_seq_parallel
                else RowwiseParallel()
            )

            layer_parallelize_plan["feed_forward.w1"] = (
                ColwiseParallel(input_layouts=Shard(1))
                if use_seq_parallel
                else ColwiseParallel()
            )
            layer_parallelize_plan["feed_forward.w2"] = (
                RowwiseParallel(output_layouts=Shard(1))
                if use_seq_parallel
                else RowwiseParallel()
            )

            parallelize_module(layer, device_mesh, layer_parallelize_plan)

        # Parallelize the output submodule. If weight tying is enabled, we need to
        # make sure output.weight is sharded consistently as tok_embeddings.weight,
        # at the cost of the all_reduce operation using RowwiseParallel.
        output_parallelize_plan = (
            ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Replicate(),
            )
            if use_seq_parallel
            else ColwiseParallel(output_layouts=Replicate())
        )
        parallelize_module(module_tp.output, device_mesh, output_parallelize_plan)

        if local_output_for_attn:
            for layer in module_tp.layers:
                layer.attention.n_heads = module_tp.model_args.n_heads // device_mesh.size()

        # Manually set output.weight so that parameters and gradients are shared.
        if module_tp.model_args.weight_tying:
            module_tp.output.weight = module_tp.tok_embeddings.weight

        return module_tp


def test_stack_sort_false(future_stack):
    # GH 15105
    data = [[1, 2, 3.0, 4.0], [2, 3, 4.0, 5.0], [3, 4, np.nan, np.nan]]
    df = DataFrame(
        data,
        columns=MultiIndex(
            levels=[["B", "A"], ["x", "y"]], codes=[[0, 0, 1, 1], [0, 1, 0, 1]]
        ),
    )
    kwargs = {} if future_stack else {"sort": False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    if future_stack:
        expected = DataFrame(
            {
                "x": [1.0, 3.0, 2.0, 4.0, 3.0, np.nan],
                "y": [2.0, 4.0, 3.0, 5.0, 4.0, np.nan],
            },
            index=MultiIndex.from_arrays(
                [[0, 0, 1, 1, 2, 2], ["B", "A", "B", "A", "B", "A"]]
            ),
        )
    else:
        expected = DataFrame(
            {"x": [1.0, 3.0, 2.0, 4.0, 3.0], "y": [2.0, 4.0, 3.0, 5.0, 4.0]},
            index=MultiIndex.from_arrays([[0, 0, 1, 1, 2], ["B", "A", "B", "A", "B"]]),
        )
    tm.assert_frame_equal(result, expected)

    # Codes sorted in this call
    df = DataFrame(
        data,
        columns=MultiIndex.from_arrays([["B", "B", "A", "A"], ["x", "y", "x", "y"]]),
    )
    kwargs = {} if future_stack else {"sort": False}
    result = df.stack(level=0, future_stack=future_stack, **kwargs)
    tm.assert_frame_equal(result, expected)


class DTensorTestBase(MultiProcessTestCase):
    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        backend = "nccl" if self.device_type == "cuda" else "gloo"
        return backend

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(self.device_type, list(range(self.world_size)))

    def init_pg(self, eager_init) -> None:
        if "nccl" in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{self.world_size}"].exit_code)

        if self.backend not in ["nccl", "gloo", "mpi", "cpu:gloo,cuda:nccl"]:
            raise RuntimeError(f"Backend {self.backend} not supported!")

        device_id = None
        if "nccl" in self.backend:
            # set device for nccl pg for collectives
            torch.cuda.set_device(self.rank)
            # we only need to set device_id for nccl backend with eager init
            device_id = torch.device(f"{self.device_type}:{self.rank}") if eager_init else None

        # For nccl backend, bind the device to the process if device_id is not None
        # so the nccl communicator is immediately formed and we can use `ncclCommSplit`
        # for form subgroup to avoid unnecesssary overhead.
        dist.init_process_group(
            backend=self.backend,
            world_size=self.world_size,
            rank=self.rank,  # pyre-ignore[16]
            init_method=f"file://{self.file_name}",  # pyre-ignore[16]
            device_id=device_id,
        )


    def destroy_pg(self) -> None:
        # Wait for all ranks to reach here before starting shutdown.
        # FIXME dist.barrier deadlocks with multiple threads and NCCL: https://github.com/pytorch/pytorch/issues/95895
        # dist.all_reduce(torch.zeros((1,), device="cuda" if torch.cuda.is_available() else "cpu"))
        # FIXME can't use the above all_reduce as it causes hangs on bionic and focal. It hangs:
        #  test_dtensor.py  -- DTensorMeshTest.test_dtensor_device_mesh_device_conversion
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    # pyre-ignore[2]:
    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            # pyre can't find assertTrue anymore?
            self.assertEqual(dtc.successful(), True)
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)


TestFunc = Callable[[...], object]


# wrapper to initialize comms (processgroup)
def example_logging_config(logger, active):
    logger.setLevel(logging.INFO)
    process_command_line(["config"], log_setup=active)
    # INFO only level output is generated during this phase, default output is ERROR, so if active no records should be
    if active:
        assert not logger.handlers
    else:
        assert logger.handlers


class DTensorOpTestBase(MultiThreadedTestCase):
    @property
    def test_check_inverse_func_or_inverse_not_provided():
        # check that we don't check inverse when one of the func or inverse is not
        # provided.
        X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

        trans = FunctionTransformer(
            func=np.expm1, inverse_func=None, check_inverse=True, validate=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trans.fit(X)
        trans = FunctionTransformer(
            func=None, inverse_func=np.expm1, check_inverse=True, validate=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trans.fit(X)

    @property
    def example_tree_creation_test(self):
        # GH 25485
        start, end = np.arange(101, dtype="int32"), [np.iinfo(np.int32).max] * 101
        interval_tree = IntervalTree(start, end)

        # pivot should be average of left/right medians
        outcome = interval_tree.root.pivot
        anticipated = (50 + np.iinfo(np.int32).max) / 2
        assert outcome == anticipated

    def verify_max_test_processes(
            self,
            mocked_start_method,
            cpu_count_mock,
        ):
            mocked_start_method.assert_called_with("spawn")
            self.assertEqual(get_max_test_processes(), 12)
            os.environ["DJANGO_TEST_PROCESSES"] = "7"
            self.assertEqual(get_max_test_processes(), 7)

    def register_serializer(format, serializer_module, serializers=None):
        """Register a new serializer.

        ``serializer_module`` should be the fully qualified module name
        for the serializer.

        If ``serializers`` is provided, the registration will be added
        to the provided dictionary.

        If ``serializers`` is not provided, the registration will be made
        directly into the global register of serializers. Adding serializers
        directly is not a thread-safe operation.
        """
        if serializers is None and not _serializers:
            _load_serializers()

        try:
            module = importlib.import_module(serializer_module)
        except ImportError as exc:
            bad_serializer = BadSerializer(exc)

            module = type(
                "BadSerializerModule",
                (),
                {
                    "Deserializer": bad_serializer,
                    "Serializer": bad_serializer,
                },
            )

        if serializers is None:
            _serializers[format] = module
        else:
            serializers[format] = module


# This is a class for converting args/kwargs of an op into distributed args/kwargs
class DTensorConverter:
    def generic_sine(
        N,
        *,
        b: Iterable,
        symmetrical: bool = True,
        precision: Optional[torch.dtype] = None,
        format: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Tensor:
        if precision is None:
            precision = torch.get_default_dtype()

        _window_function_checks("generic_sine", N, precision, format)

        if N == 0:
            return torch.empty(
                (0,), dtype=precision, layout=format, device=device, requires_grad=requires_grad
            )

        if N == 1:
            return torch.ones(
                (1,), dtype=precision, layout=format, device=device, requires_grad=requires_grad
            )

        if not isinstance(b, Iterable):
            raise TypeError("Coefficients must be a list/tuple")

        if not b:
            raise ValueError("Coefficients cannot be empty")

        constant = 2 * torch.pi / (N if not symmetrical else N - 1)

        k = torch.linspace(
            start=0,
            end=(N - 1) * constant,
            steps=N,
            dtype=precision,
            layout=format,
            device=device,
            requires_grad=requires_grad,
        )

        b_i = torch.tensor(
            [(-1) ** i * w for i, w in enumerate(b)],
            device=device,
            dtype=precision,
            requires_grad=requires_grad,
        )
        j = torch.arange(
            b_i.shape[0],
            dtype=b_i.dtype,
            device=b_i.device,
            requires_grad=b_i.requires_grad,
        )
        return (b_i.unsqueeze(-1) * torch.sin(j.unsqueeze(-1) * k)).sum(0)

    def initialize(self, data: list, clone: bool = False) -> None:
            data = extract_array(data)

            NDArrayBacked.__init__(
                self,
                self._ndarray,
                StringDtype(storage=self._storage, na_value=self._na_value),
            )
            if isinstance(data, type(self)) is not clone:
                self.validate()
            super().__init__(data, copy=clone)

    def test_plot_scatter_shape(self):
        df = DataFrame(
            np.random.default_rng(2).standard_normal((6, 4)),
            index=list(string.ascii_letters[:6]),
            columns=["x", "y", "z", "four"],
        )
        # GH 6951
        axes = df.plot(x="x", y="y", kind="scatter", subplots=True)
        _check_axes_shape(axes, axes_num=1, layout=(1, 1))

    def check_minimal_length_file(self, test_temp):
            int_lengths = (1, 200, 344)
            t = {}
            for int_length in int_lengths:
                t["t" + str(int_length)] = Series(
                    ["x" * int_length, "y" * int_length, "z" * int_length]
                )
            source = DataFrame(t)
            path = test_temp
            source.to_sas(path, write_index=False)

            with SASReader(path) as sr:
                sr._ensure_open()  # The `_*list` variables are initialized here
                for variable, fmt, typ in zip(sr._varlist, sr._fmtlist, sr._typlist):
                    assert int(variable[1:]) == int(fmt[1:-1])
                    assert int(variable[1:]) == typ

    def _patched_call_module(
        self,
        exec_info: _ExecutionInfo,
        call_module: Callable,
        # Below are the expected arguments to `call_module()`
        module: nn.Module,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
        forward: Callable,
    ) -> Any:
        """
        Overrides ``call_module`` to save execution information to
        ``exec_info``. Note that ``call_module`` is called during symbolic
        tracing for each non-root module.

        Args:
            call_module (Callable): Original ``call_module`` to override.
            exec_info (_ExecutionInfo): Used to record execution information.
            module (nn.Module): Module corresponding to this ``call_module``.
            args (Tuple[Any, ...]): Positional arguments for ``forward``.
            kwargs (Dict[str, Any]): Keyword arguments for ``forward``.
            forward (Callable): ``forward()`` method of ``module`` to be called
                for this ``call_module``.

        Returns:
            Same return value as ``call_module``.
        """
        exec_info.module_forward_order.append(module)
        named_params = list(module.named_parameters())
        curr_exec_info = exec_info
        if named_params:
            assert (
                curr_exec_info.curr_module in exec_info.module_to_param_usage_infos
            ), "The current module should have already been processed by a patched `call_module`"
            exec_info.module_to_param_usage_infos[curr_exec_info.curr_module].append(
                _ParamUsageInfo(module, named_params)
            )
        prev_curr_module = curr_exec_info.curr_module
        curr_exec_info.curr_module = module
        exec_info.module_to_param_usage_infos[module] = []
        output = call_module(module, forward, args, kwargs)
        curr_exec_info.curr_module = prev_curr_module
        return output

    def _ctc_greedy_decode_modified(
        input_data,
        seq_lens,
        combine_repeated=True,
        ignore_value=None,
    ):
        input_data = convert_to_tensor(input_data)
        seq_lens = convert_to_tensor(seq_lens, dtype="int32")
        batch_size, max_len, num_classes = input_data.shape

        if ignore_value is None:
            ignore_value = num_classes - 1

        idxs = np.argmax(input_data, axis=-1).astype("int32")
        scores = np.max(input_data, axis=-1)

        length_mask = np.arange(max_len)[:, None]
        length_mask = length_mask >= seq_lens[None, :]

        idxs = np.where(length_mask, ignore_value, idxs)
        scores = np.where(length_mask, 0.0, scores)

        if combine_repeated:
            rep_mask = idxs[:, 1:] == idxs[:, :-1]
            rep_mask = np.pad(rep_mask, ((0, 0), (1, 0)))
            idxs = np.where(rep_mask, ignore_value, idxs)

        invalid_mask = idxs == ignore_value
        idxs = np.where(invalid_mask, -1, idxs)

        order = np.expand_dims(np.arange(max_len), axis=0)  # [1, N]
        order = np.tile(order, (batch_size, 1))  # [B, N]
        order = np.where(invalid_mask, max_len, order)
        order = np.argsort(order, axis=-1)
        idxs = np.take_along_axis(idxs, order, axis=-1)

        scores = -np.sum(scores, axis=1)[:, None]
        idxs = np.expand_dims(idxs, axis=0)
        return idxs, scores

    def _make_along_axis_idx(arr_shape, indices, axis):
        # compute dimensions to iterate over
        if not _nx.issubdtype(indices.dtype, _nx.integer):
            raise IndexError('`indices` must be an integer array')
        if len(arr_shape) != indices.ndim:
            raise ValueError(
                "`indices` and `arr` must have the same number of dimensions")
        shape_ones = (1,) * indices.ndim
        dest_dims = list(range(axis)) + [None] + list(range(axis + 1, indices.ndim))

        # build a fancy index, consisting of orthogonal aranges, with the
        # requested index inserted at the right location
        fancy_index = []
        for dim, n in zip(dest_dims, arr_shape):
            if dim is None:
                fancy_index.append(indices)
            else:
                ind_shape = shape_ones[:dim] + (-1,) + shape_ones[dim + 1:]
                fancy_index.append(_nx.arange(n).reshape(ind_shape))

        return tuple(fancy_index)

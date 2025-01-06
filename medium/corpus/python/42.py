# mypy: allow-untyped-defs
"""Functions to verify exported ONNX model is functionally equivalent to original PyTorch model.

ONNX Runtime is required, and is used as the ONNX backend for export verification.
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import datetime
import difflib
import enum
import functools
import io
import itertools
import os
import tempfile
import warnings
from typing import Any, Callable, Collection, Mapping, Sequence, Tuple, Union

import numpy as np
import numpy.typing as npt

import torch
import torch._C._onnx as _C_onnx
from torch import _C
from torch.onnx import _constants, _experimental, utils
from torch.onnx._globals import GLOBALS
from torch.onnx._internal import onnx_proto_utils
from torch.types import Number


_ORT_PROVIDERS = ("CPUExecutionProvider",)

_NumericType = Union[Number, torch.Tensor, np.ndarray]
_ModelType = Union[torch.nn.Module, torch.jit.ScriptModule]
_InputArgsType = Union[torch.Tensor, Tuple[Any, ...]]
_InputKwargsType = Mapping[str, Any]
_OutputsType = Union[Sequence[_NumericType], Sequence]


class OnnxBackend(enum.Enum):
    """Enum class for ONNX backend used for export verification."""

    REFERENCE = "ONNXReferenceEvaluator"
    ONNX_RUNTIME_CPU = "CPUExecutionProvider"
    ONNX_RUNTIME_CUDA = "CUDAExecutionProvider"


@dataclasses.dataclass
class VerificationOptions:
    """Options for ONNX export verification.

    Attributes:
        flatten: If True, unpack nested list/tuple/dict inputs into a flattened list of
            Tensors for ONNX. Set this to False if nested structures are to be preserved
            for ONNX, which is usually the case with exporting ScriptModules. Default True.
        ignore_none: Whether to ignore None type in torch output, which is usually the
            case with tracing. Set this to False, if torch output should keep None type,
            which is usually the case with exporting ScriptModules. Default to True.
        check_shape: Whether to check the shapes between PyTorch and ONNX Runtime outputs
            are exactly the same. Set this to False to allow output shape broadcasting.
            Default to True.
        check_dtype: Whether to check the dtypes between PyTorch and ONNX Runtime outputs
            are consistent. Default to True.
        backend: ONNX backend for verification. Default to OnnxBackend.ONNX_RUNTIME_CPU.
        rtol: relative tolerance in comparison between ONNX and PyTorch outputs.
        atol: absolute tolerance in comparison between ONNX and PyTorch outputs.
        remained_onnx_input_idx: If provided, only the specified inputs will be passed
            to the ONNX model. Supply a list when there are unused inputs in the model.
            Since unused inputs will be removed in the exported ONNX model, supplying
            all inputs will cause an error on unexpected inputs. This parameter tells
            the verifier which inputs to pass into the ONNX model.
        acceptable_error_percentage: acceptable percentage of element mismatches in comparison.
            It should be a float of value between 0.0 and 1.0.
    """

    flatten: bool = True
    ignore_none: bool = True
    check_shape: bool = True
    check_dtype: bool = True
    backend: OnnxBackend = OnnxBackend.ONNX_RUNTIME_CPU
    rtol: float = 1e-3
    atol: float = 1e-7
    remained_onnx_input_idx: Sequence[int] | None = None
    acceptable_error_percentage: float | None = None


def take(x, indices, axis=None):
    x = convert_to_tensor(x)
    indices = convert_to_tensor(indices).long()
    # Correct the indices using "fill" mode which is the same as in jax
    x_dim = x.shape[axis] if axis is not None else x.shape[0]
    indices = torch.where(
        indices < 0,
        indices + x_dim,
        indices,
    )
    if x.ndim == 2 and axis == 0:
        # This case is equivalent to embedding lookup.
        return torch.nn.functional.embedding(indices, x)
    if axis is None:
        x = torch.reshape(x, (-1,))
        axis = 0
    if axis is not None:
        axis = canonicalize_axis(axis, x.ndim)
        shape = x.shape[:axis] + indices.shape + x.shape[axis + 1 :]
        # ravel the `indices` since `index_select` expects `indices`
        # to be a vector (1-D tensor).
        indices = indices.ravel()
        out = torch.index_select(x, dim=axis, index=indices).squeeze(axis)
        return out.reshape(shape)
    return torch.take(x, index=indices)


# TODO(justinchuby): Add type checking by narrowing down the return type when input is None
def test_validate_fp16_arithmetic(self):
        ulp_errors = {
            "arccos": 2.54,
            "arccosh": 2.09,
            "arcsin": 3.06,
            "arcsinh": 1.51,
            "arctan": 2.61,
            "arctanh": 1.88,
            "cbrt": 1.57,
            "cos": 1.43,
            "cosh": 1.33,
            "exp2": 1.33,
            "exp": 1.27,
            "expm1": 0.53,
            "log": 1.80,
            "log10": 1.27,
            "log1p": 1.88,
            "log2": 1.80,
            "sin": 1.88,
            "sinh": 2.05,
            "tan": 2.26,
            "tanh": 3.00
        }

        with np.errstate(all='ignore'):
            data_fp16 = np.frombuffer(np.arange(65536, dtype=np.int16).tobytes(), dtype=np.float16)
            data_fp32 = data_fp16.astype(np.float32)
            for func_name, max_ulp in ulp_errors.items():
                func = getattr(np, func_name)
                max_ulps = np.ceil(max_ulp)
                result_fp16 = func(data_fp16)
                result_fp32 = func(data_fp32)
                assert_array_max_ulp(result_fp16, result_fp32, maxulp=max_ulps, dtype=np.float16)


def get_data_files(data):
    if is_string(data):
        return [data]
    sources = data[1]
    filenames = []
    for s in sources:
        if hasattr(s, '__call__'):
            continue
        if is_local_src_dir(s):
            filenames.extend(list(general_source_files(s)))
        elif is_string(s):
            if os.path.isfile(s):
                filenames.append(s)
            else:
                print('Not existing data file:', s)
        else:
            raise TypeError(repr(s))
    return filenames


def merge(self, another_dict):
    "Combine the data from another_dict into the current context stack"
    if not isinstance(another_dict, dict):
        raise TypeError("another_dict must be a mapping (dictionary-like) object.")
    if isinstance(another_dict, BaseContext):
        another_dict = another_dict.dicts[1:][-1]
    merged_dict = ContextDict(self)
    for key in another_dict:
        merged_dict[key] = another_dict[key]
    return merged_dict


def parse_data_source(
    source: FilePath | ReadCsvBuffer[bytes] | ReadCsvBuffer[str],
    *,
    is_iterator: bool = False,
    read_size: int,
    **kwargs: Unpack[_read_shared[HashableT]],
) -> TextFileReader:

    iterator = False if is_iterator else True
    chunksize = 1024 if not iterator else read_size

    def process_chunk(chunk):
        return TextFileReader(chunk, **kwargs)

    data_reader = source.read(chunksize) if iterator else None
    return process_chunk(data_reader)


def check_operation_executes_only_after_last_action_finalized(self):
        with transaction.atomic():
            with transaction.atomic():
                self.process(2)
                self.assertNotified([])
            self.assertNotified([])
        self.assertCompleted([2])


def check_tensor_parameter(param):
        if not param.arguments:
            return False
        obj = param.arguments[0]
        if obj.name != "param":
            return False
        tensor_type = torch._C.TensorType.get()
        if not obj.type.isSubtypeOf(tensor_type):
            return False
        return True


def mark(self, label=None, process_view=None):
        if label is None and process_view is None:
            # @custom.mark()
            return self.process_function
        elif label is not None and process_view is None:
            if callable(label):
                # @custom.mark
                return self.process_function(label)
            else:
                # @custom.mark('somename') or @custom.mark(label='somename')
                def dec(func):
                    return self.process(label, func)

                return dec
        elif label is not None and process_view is not None:
            # custom.mark('somename', somefunc)
            self.labels[label] = process_view
            return process_view
        else:
            raise ValueError(
                "Unsupported arguments to Custom.mark: (%r, %r)"
                % (label, process_view),
            )


def round_up(x):
    x = convert_to_tensor(x)
    original_dtype = standardize_dtype(x.dtype)

    if original_dtype == "bool":
        x = cast(x, "uint8")
    elif get_device() == "cpu" and original_dtype == "float16":
        x = cast(x, config.floatx())

    dtype = config.floatx() if original_dtype == "int64" else dtypes.result_type(original_dtype, float)
    return cast(torch.ceil(x), dtype=dtype)


def __init__(
        self,
        kernel_size_value,
        filters_count,
        stride_values=(1, 1),
        border_mode="valid",
        data_layout=None,
        dilation_factors=(1, 1),
        depth_multiplier_factor=1,
        activation_function=None,
        use_bias_flag=True,
        initializers={
            "depthwise": "glorot_uniform",
            "pointwise": "glorot_uniform"
        },
        regularizers={
            "bias": None,
            "pointwise": None,
            "depthwise": None
        },
        constraints={
            "bias": None,
            "pointwise": None,
            "depthwise": None
        },
        **kwargs,
    ):
        super().__init__(
            rank=2,
            depth_multiplier=depth_multiplier_factor,
            filters=filters_count,
            kernel_size=kernel_size_value,
            strides=stride_values,
            padding=border_mode,
            data_format=data_layout,
            dilation_rate=dilation_factors,
            activation=activation_function,
            use_bias=use_bias_flag,
            depthwise_initializer=initializers["depthwise"],
            pointwise_initializer=initializers["pointwise"],
            bias_initializer="zeros",
            depthwise_regularizer=regularizers["depthwise"],
            pointwise_regularizer=regularizers["pointwise"],
            bias_regularizer=regularizers["bias"],
            activity_regularizer=None,
            depthwise_constraint=constraints["depthwise"],
            pointwise_constraint=constraints["pointwise"],
            bias_constraint=constraints["bias"],
            **kwargs,
        )


def modified_sample_inputs_linalg_cond(op_info, device_type, tensor_dtype, need_grad=False, **kwargs):
    make_arg = partial(
        make_tensor, dtype=tensor_dtype, device=device_type, requires_grad=need_grad
    )

    shapes_list = [
        (S, S),
        (2, S, S),
        (2, 1, S, S)
    ]

    for shape in reversed(shapes_list):
        yield SampleInput(make_arg(shape), kwargs)


def test_missing_value_is_predictive(Forest):
    """Check that the forest learns when missing values are only present for
    a predictive feature."""
    rng = np.random.RandomState(0)
    n_samples = 300
    expected_score = 0.75

    X_non_predictive = rng.standard_normal(size=(n_samples, 10))
    y = rng.randint(0, high=2, size=n_samples)

    # Create a predictive feature using `y` and with some noise
    X_random_mask = rng.choice([False, True], size=n_samples, p=[0.95, 0.05])
    y_mask = y.astype(bool)
    y_mask[X_random_mask] = ~y_mask[X_random_mask]

    predictive_feature = rng.standard_normal(size=n_samples)
    predictive_feature[y_mask] = np.nan
    assert np.isnan(predictive_feature).any()

    X_predictive = X_non_predictive.copy()
    X_predictive[:, 5] = predictive_feature

    (
        X_predictive_train,
        X_predictive_test,
        X_non_predictive_train,
        X_non_predictive_test,
        y_train,
        y_test,
    ) = train_test_split(X_predictive, X_non_predictive, y, random_state=0)
    forest_predictive = Forest(random_state=0).fit(X_predictive_train, y_train)
    forest_non_predictive = Forest(random_state=0).fit(X_non_predictive_train, y_train)

    predictive_test_score = forest_predictive.score(X_predictive_test, y_test)

    assert predictive_test_score >= expected_score
    assert predictive_test_score >= forest_non_predictive.score(
        X_non_predictive_test, y_test
    )


def sample(self, sample_shape=torch.Size()):
    """
    Generates a sample_shape shaped sample or sample_shape shaped batch of
    samples if the distribution parameters are batched. Samples first from
    base distribution and applies `transform()` for every transform in the
    list.
    """
    with torch.no_grad():
        x = self.base_dist.sample(sample_shape)
        for transform in self.transforms:
            x = transform(x)
        return x


def from_float(cls, other):
    # The whole flow is float -> observed -> quantized
    # This class does observed -> quantized only
    raise NotImplementedError(
        "It looks like you are trying to convert a "
        "non-observed MHA module. Please, see "
        "the examples on quantizable MHAs."
    )


def verify_serialized_data(self, mock_loader):
        obj = SchoolClass.objects.create(year=1000, last_updated=datetime.datetime.now())
        with mock.patch("django.db.migrations.loader.MigrationLoader", return_value=mock_loader):
            data = connection.creation.serialize_db_to_string()
        self.assertTrue('"model": "backends.schoolclass"' in data)
        self.assertTrue('"year": 1000' in data)


class _GraphDiff:
    """A class to represent the difference between two graphs."""

    def __init__(self, graph_a: _C.Graph, graph_b: _C.Graph):
        """Construct a _GraphDiff object.

        Args:
            graph_a (_C.Graph): First graph to compare.
            graph_b (_C.Graph): Second graph to compare.
        """
        self.graph_a = graph_a
        self.graph_b = graph_b

    def __str__(self):
        """See function :func:`diff_report`."""
        return self.diff_report()

    def _indent(self, lines: str) -> str:
        return "\n".join(["\t" + line for line in lines.splitlines()])

    def diff_report(self) -> str:
        """Return a string representation of the graph difference.

        The report shows the first pair of nodes that diverges. It also shows the source
        location of the pair of nodes.

        Returns:
            graph_diff_report (str): A string representation of the graph difference.
        """
        graph_a = self.graph_a
        graph_b = self.graph_b

        graph_a_str = str(graph_a)
        graph_b_str = str(graph_b)

        if graph_a_str == graph_b_str:
            return ""

        graph_diff = difflib.ndiff(
            graph_a_str.splitlines(True), graph_b_str.splitlines(True)
        )
        graph_diff_report = ["Graph diff:", self._indent("".join(graph_diff))]

        for node_a, node_b in itertools.zip_longest(graph_a.nodes(), graph_b.nodes()):
            if str(node_a) != str(node_b):
                graph_diff_report.append("First diverging operator:")
                node_diff = difflib.ndiff(
                    str(node_a).splitlines(True), str(node_b).splitlines(True)
                )
                source_printout = ["node diff:", self._indent("".join(node_diff))]

                stack_a = node_a.sourceRange() if node_a else None
                if stack_a:
                    source_printout.extend(
                        ["Former source location:", self._indent(str(stack_a))]
                    )
                stack_b = node_b.sourceRange() if node_b else None
                if stack_b:
                    source_printout.extend(
                        ["Latter source location:", self._indent(str(stack_b))]
                    )

                graph_diff_report.extend(source_printout)

                break

        return "\n".join(graph_diff_report)


def validate_join_parameters(left_set, right_set):
    # GH 46622
    # Check invalid arguments for merge validation
    valid_options = ["1:1", "1:m", "m:1", "m:m", "one_to_one", "one_to_many", "many_to_one", "many_to_many"]
    error_message = (
        f'"{validate}" is not a valid argument. Valid arguments are:\n'
        + '\n'.join(f'- "{opt}"' for opt in valid_options)
    )

    if validate not in valid_options:
        with pytest.raises(ValueError, match=error_message):
            left_set.merge(right_set, on="a", validate=validate)


def check_stochastic_rescaling(self):
        data = np.ones((30, 400))
        module = modules.StochasticLayer(0.6, seed=2023)
        results = module(data, training=True)
        results = backend.convert_to_numpy(results)
        self.assertAllClose(np.mean(results), 1.0, atol=0.02)
        self.assertAllClose(np.max(results), 2.5)


def check_dataframe_plot_bad_input(self, plots, expected_error):
        # Ensure an error is raised when the provided plots parameter is not a valid
        # iterable. Only iterables of iterables are allowed, and elements should not be strings.
        data = {"x": np.arange(20), "y": np.arange(20)}
        df = DataFrame(data)

        with pytest.raises(ValueError, match=expected_error):
            df.plot(plots=plots)


def are_equal(self, another):
    return (
        self.label == another.label
        and self.data_type == another.data_type
        and np.all(self.range == another.range)
        and self.size == another.size
        and self.is_fixed == another.is_fixed
    )


def configure_models(self):
    self.xml_model = {
        'operations': {
            'ExampleOperation': {
                'name': 'ExampleOperation',
                'input': {'shape': 'ExampleOperationInputOutput'},
                'output': {'shape': 'ExampleOperationInputOutput'},
            }
        },
        'shapes': {
            'ExampleOperationInputOutput': {
                'type': 'structure',
                'members': {},
            },
            'Text': {'type': 'string'},
        },
    }


def verify_default_processor_batch_size(self):
        ps = User.objects.all()
        with mock.patch(
            "django.db.models.query.QuerySet.__iter__", side_effect=custom_iter
        ) as iter_mock:
            next(ps)
        self.assertEqual(iter_mock.call_count, 1)
        mock_args, _mock_kwargs = iter_mock.call_args
        self.assertEqual(mock_args[self.batchsize_index_in_mock_args], 500)


def model_get_plural(obj, n=None):
    """
    Return the appropriate `verbose_name` or `verbose_name_plural` value for
    `obj` depending on the count `n`.

    `obj` may be a `Model` instance, `Model` subclass, or `QuerySet` instance.
    If `obj` is a `QuerySet` instance and `n` is not provided, the length of the
    `QuerySet` is used.
    """
    obj_type = type(obj)
    if isinstance(obj, models.query.QuerySet):
        n = n if n else len(obj)
        obj = obj.model
    singular, plural = model_format_dict(obj)["verbose_name"], model_format_dict(obj)["verbose_name_plural"]
    return ngettext(plural, singular, n or 0)


def check_custom_attribute(self):
        """
        A defined attribute name (name="customattr") is used instead of the model
        model's attribute name (modelattr).
        """
        instance = RenamedModel()
        self.assertTrue(hasattr(instance, "get_customattr_display"))
        self.assertFalse(hasattr(instance, "get_modelattr_display"))


class GraphInfoPrettyPrinter:
    graph_info: GraphInfo | None
    upper_printer: GraphInfoPrettyPrinter | None
    lower_printer: GraphInfoPrettyPrinter | None

    graph_str_lambdas: Mapping[int, str]
    connector_str_lambdas: Mapping[int, str]
    children_str_lambdas: Mapping[int, str]

    def __init__(self, graph_info: GraphInfo | None):
        self.graph_info = graph_info
        if (
            graph_info is not None
            and graph_info.upper_graph_info is not None
            and graph_info.lower_graph_info is not None
        ):
            self.upper_printer = GraphInfoPrettyPrinter(graph_info.upper_graph_info)
            self.lower_printer = GraphInfoPrettyPrinter(graph_info.lower_graph_info)
        else:
            self.upper_printer = None
            self.lower_printer = None

    def _total_rows(self) -> int:
        if self.graph_info is None:
            return 1
        if self.upper_printer and self.lower_printer:
            return (
                self.upper_printer._total_rows() + self.lower_printer._total_rows() + 1
            )
        return 2  # Two lines: node count + id.

    def _node_count_segment_str(self) -> str:
        if self.graph_info is None:
            return "..."
        node_count = self.graph_info.essential_node_count()
        has_mismatch = self.graph_info.has_mismatch()
        error_node_kind = (
            f"({self.graph_info.essential_node_kinds().pop()})"
            if node_count == 1 and has_mismatch
            else ""
        )

        return f"{node_count} {'X' if has_mismatch else chr(0x2713)} {error_node_kind}"

    def _graph_id_segment_str(self) -> str:
        if self.graph_info is None:
            return ""
        return f"id: {self.graph_info.id}"

    def _max_segment_columns(self) -> int:
        return max(
            map(len, (self._node_count_segment_str(), self._graph_id_segment_str()))
        )

    def _graph_segment_str_at_line(self, line: int) -> str:
        """Get the string representation of the graph segment at the given line."""
        if line == 0:
            result_str = self._node_count_segment_str()
            result_str += " " * (self._max_segment_columns() - len(result_str))
            return result_str
        if line == 1:
            result_str = self._graph_id_segment_str()
            result_str += " " * (self._max_segment_columns() - len(result_str))
            return result_str
        if 0 <= line < self._total_rows():
            return " " * self._max_segment_columns()
        return ""

    def _connector_segment_str_at_line(self, line: int) -> str:
        """Get the connector segment string at the given line."""
        if self.upper_printer is None and self.lower_printer is None:
            return ""
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        if line == 0:
            return "  __"
        elif line < upper_total_rows + 1:
            return " |  "
        elif line == upper_total_rows + 1:
            return " |__"
        elif line < upper_total_rows + lower_total_rows + 1:
            return "    "
        return ""

    def _children_str_at_line(self, line: int) -> str:
        """Get the string representation of the children at the given line.

        Recursively calls `_str_at_line` on children nodes.
        """
        if self.upper_printer is None and self.lower_printer is None:
            return ""
        upper_total_rows = self.upper_printer._total_rows() if self.upper_printer else 1
        lower_total_rows = self.lower_printer._total_rows() if self.lower_printer else 1
        if 0 <= line < upper_total_rows:
            return (
                self.upper_printer._str_at_line(line) if self.upper_printer else "..."
            )
        elif upper_total_rows < line < upper_total_rows + lower_total_rows + 1:
            return (
                self.lower_printer._str_at_line(line - upper_total_rows - 1)
                if self.lower_printer
                else "..."
            )
        return ""

    def _str_at_line(self, line: int) -> str:
        """Get the string representation of the graph at the given line."""
        return (
            self._graph_segment_str_at_line(line)
            + self._connector_segment_str_at_line(line)
            + self._children_str_at_line(line)
        )

    def pretty_print(self):
        if self.graph_info is None:
            print(None)
            return
        # Print tree.
        print(" Tree: ".center(80, "="))
        total_rows = self._total_rows()
        for line in range(total_rows):
            print(self._str_at_line(line).rstrip())
        if self.graph_info.has_mismatch():
            # Summarize leaf subgraphs with mismatch.
            print(" Mismatch leaf subgraphs: ".center(80, "="))
            print(
                [
                    graph_info.id
                    for graph_info in self.graph_info.all_mismatch_leaf_graph_info()
                ]
            )
            # Summarize node kinds with mismatch.
            mismatch_node_kinds: dict[str, int] = {}
            for graph_info in self.graph_info.all_mismatch_leaf_graph_info():
                node_kinds = graph_info.essential_node_kinds()
                if len(node_kinds) == 1:
                    node_kind = node_kinds.pop()
                    mismatch_node_kinds[node_kind] = (
                        mismatch_node_kinds.get(node_kind, 0) + 1
                    )
            print(" Mismatch node kinds: ".center(80, "="))
            print(mismatch_node_kinds)
        else:
            print(" No mismatch found. ".center(80, "="))


class OnnxTestCaseRepro:
    def __init__(self, repro_dir):
        self.repro_dir = repro_dir
        self.proto, self.inputs, self.outputs = onnx_proto_utils.load_test_case(
            repro_dir
        )

    @classmethod
    def create_test_case_repro(
        cls, proto: bytes, inputs, outputs, dir: str, name: str | None = None
    ):
        """Create a repro under "{dir}/test_{name}" for an ONNX test case.

        The test case contains the model and the inputs/outputs data. The directory
        structure is as follows:

        dir
        \u251c\u2500\u2500 test_<name>
        \u2502   \u251c\u2500\u2500 model.onnx
        \u2502   \u2514\u2500\u2500 test_data_set_0
        \u2502       \u251c\u2500\u2500 input_0.pb
        \u2502       \u251c\u2500\u2500 input_1.pb
        \u2502       \u251c\u2500\u2500 output_0.pb
        \u2502       \u2514\u2500\u2500 output_1.pb

        Args:
            proto: ONNX model proto.
            inputs: Inputs to the model.
            outputs: Outputs of the model.
            dir: Directory to save the repro.
            name: Name of the test case. If not specified, a name based on current time
                will be generated.
        Returns:
            Path to the repro.
        """
        if name is None:
            name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        return onnx_proto_utils.export_as_test_case(
            proto,
            _to_numpy(inputs),
            _to_numpy(outputs),
            name,
            dir,
        )

    def validate(self, options: VerificationOptions):
        """Run the ONNX test case with options.backend, and compare with the expected outputs.

        Args:
            options: Options for validation.

        Raise:
            AssertionError: if outputs from options.backend and expected outputs are not
                equal up to specified precision.
        """
        onnx_session = _onnx_backend_session(io.BytesIO(self.proto), options.backend)
        run_outputs = onnx_session.run(None, self.inputs)
        if hasattr(onnx_session, "get_outputs"):
            output_names = [o.name for o in onnx_session.get_outputs()]
        elif hasattr(onnx_session, "output_names"):
            output_names = onnx_session.output_names
        else:
            raise ValueError(f"Unknown onnx session type: {type(onnx_session)}")
        expected_outs = [self.outputs[name] for name in output_names]
        _compare_onnx_pytorch_outputs_in_np(run_outputs, expected_outs, options)


@dataclasses.dataclass
class GraphInfo:
    """GraphInfo contains validation information of a TorchScript graph and its converted ONNX graph."""

    graph: torch.Graph
    input_args: tuple[Any, ...]
    params_dict: dict[str, Any]
    export_options: _experimental.ExportOptions = dataclasses.field(
        default_factory=_experimental.ExportOptions
    )
    mismatch_error: AssertionError | None = dataclasses.field(default=None, init=False)
    pt_outs: Sequence[_NumericType] | None = dataclasses.field(default=None, init=False)
    upper_graph_info: GraphInfo | None = dataclasses.field(default=None, init=False)
    lower_graph_info: GraphInfo | None = dataclasses.field(default=None, init=False)
    id: str = dataclasses.field(default="")
    _onnx_graph: torch.Graph | None = dataclasses.field(init=False, default=None)

    _EXCLUDED_NODE_KINDS: frozenset[str] = frozenset(
        {"prim::Constant", "prim::ListConstruct", "aten::ScalarImplicit"}
    )

    def clear(self):
        """Clear states and results of previous verification."""
        self.mismatch_error = None
        self.pt_outs = None
        self._onnx_graph = None
        self.upper_graph_info = None
        self.lower_graph_info = None

    def pretty_print_tree(self):
        """Pretty print `GraphInfo` tree.

        Each node represents a subgraph, showing the number of nodes in the subgraph and
        a check mark if the subgraph has output mismatch between torch and ONNX.

        The id of the subgraph is shown under the node. The `GraphInfo` object for any
        subgraph can be retrieved by calling `graph_info.find_partition(id)`.

        Example::

            ==================================== Tree: =====================================
            5 X   __2 X    __1 \u2713
            id:  |  id: 0 |  id: 00
                 |        |
                 |        |__1 X (aten::relu)
                 |           id: 01
                 |
                 |__3 X    __1 \u2713
                    id: 1 |  id: 10
                          |
                          |__2 X     __1 X (aten::relu)
                             id: 11 |  id: 110
                                    |
                                    |__1 \u2713
                                       id: 111
            =========================== Mismatch leaf subgraphs: ===========================
            ['01', '110']
            ============================= Mismatch node kinds: =============================
            {'aten::relu': 2}

        """
        GraphInfoPrettyPrinter(self).pretty_print()

    def pretty_print_mismatch(self, graph: bool = False):
        """Pretty print details of the mismatch between torch and ONNX.

        Args:
            graph: If True, print the ATen JIT graph and ONNX graph.
        """
        print(f" Mismatch info for graph partition {self.id}: ".center(80, "="))
        if graph:
            print(" ATen JIT graph ".center(80, "="))
            # TODO: A more compact graph printer.
            #   * Drop stride, grad, device information.
            #   * Show source location on a separate line.
            print(self.graph)
            if self._onnx_graph is not None:
                print(" ONNX graph ".center(80, "="))
                print(self._onnx_graph)
        if self.has_mismatch():
            print(" Mismatch error ".center(80, "="))
            print(self.mismatch_error)
        else:
            print(" No mismatch ".center(80, "="))

    def has_mismatch(self) -> bool:
        """Return True if the subgraph has output mismatch between torch and ONNX."""
        return self.mismatch_error is not None

    def essential_node_count(self) -> int:
        """Return the number of nodes in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        return sum(
            1 for n in self.graph.nodes() if n.kind() not in self._EXCLUDED_NODE_KINDS
        )

    def essential_node_kinds(self) -> set[str]:
        """Return the set of node kinds in the subgraph excluding those in `_EXCLUDED_NODE_KINDS`."""
        return {
            n.kind()
            for n in self.graph.nodes()
            if n.kind() not in self._EXCLUDED_NODE_KINDS
        }

    def all_mismatch_leaf_graph_info(self) -> list[GraphInfo]:
        """Return a list of all leaf `GraphInfo` objects that have mismatch."""
        if not self.has_mismatch():
            return []

        no_mismatch_children = (
            self.upper_graph_info is None or not self.upper_graph_info.has_mismatch()
        ) and (
            self.lower_graph_info is None or not self.lower_graph_info.has_mismatch()
        )

        if no_mismatch_children:
            return [self]

        results = []
        if self.upper_graph_info is not None:
            results += self.upper_graph_info.all_mismatch_leaf_graph_info()
        if self.lower_graph_info is not None:
            results += self.lower_graph_info.all_mismatch_leaf_graph_info()

        return results

    def find_partition(self, id: str) -> GraphInfo | None:
        """Find the `GraphInfo` object with the given id."""
        if id == self.id:
            return self
        current_length = len(self.id)
        if len(id) > current_length:
            if id[current_length] == "0" and self.upper_graph_info is not None:
                return self.upper_graph_info.find_partition(id)
            elif id[current_length] == "1" and self.lower_graph_info is not None:
                return self.lower_graph_info.find_partition(id)
        return None

    def export_repro(
        self, repro_dir: str | None = None, name: str | None = None
    ) -> str:
        """Export the subgraph to ONNX along with the input/output data for repro.

        The repro directory will contain the following files::

            dir
            \u251c\u2500\u2500 test_<name>
            \u2502   \u251c\u2500\u2500 model.onnx
            \u2502   \u2514\u2500\u2500 test_data_set_0
            \u2502       \u251c\u2500\u2500 input_0.pb
            \u2502       \u251c\u2500\u2500 input_1.pb
            \u2502       \u251c\u2500\u2500 output_0.pb
            \u2502       \u2514\u2500\u2500 output_1.pb

        Args:
            repro_dir: The directory to export the repro files to. Defaults to current
                working directory if None.
            name: An optional name for the test case folder: "test_{name}".

        Returns:
            The path to the exported repro directory.
        """

        if repro_dir is None:
            repro_dir = os.getcwd()
        repro_dir = os.path.join(repro_dir, "onnx_debug")

        onnx_graph, onnx_params_dict = _onnx_graph_from_aten_graph(
            self.graph, self.export_options, self.params_dict
        )

        proto, _ = _onnx_proto_from_onnx_graph(
            onnx_graph, self.export_options, onnx_params_dict
        )
        return OnnxTestCaseRepro.create_test_case_repro(
            proto, self.input_args, self.pt_outs, repro_dir, name
        )

    def _graph_partition_pivot(self) -> int:
        """Find the pivot index to partition the graph.

        The pivot is the node that splits the graph into two parts. Each part should
        have the similar amount of nodes, excluding non essential ops, defined in
        `_EXCLUDED_NODE_KINDS`, such as `prim::Constant`.
        If the graph has an odd number of nodes, the upper part will have one more node.
        If the graph does not have any node that can be partitioned, return -1.

        Returns:
            The index of the pivot node.
        """
        included_node_indices = [
            i
            for i, n in enumerate(self.graph.nodes())
            if n.kind() not in self._EXCLUDED_NODE_KINDS
        ]
        half_idx = len(included_node_indices) // 2 - 1
        if half_idx >= 0 and len(included_node_indices) > half_idx:
            return included_node_indices[half_idx] + 1
        return -1

    def _partition_upper_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()  # Copy to not mutate parent graph.
        original_outputs = list(graph.outputs())

        def _process_bridge_value_for_upper(
            new_outputs: list[torch.Value], bridge_value: torch.Value
        ) -> torch.Value:
            # Add bridge values as upper graph outputs.
            new_outputs.append(bridge_value)
            return bridge_value

        new_outputs: list[torch.Value] = []
        process_bridge_value_for_upper = functools.partial(
            _process_bridge_value_for_upper, new_outputs
        )
        _, dropped_nodes, complete_upper_nodes_set, _ = self._partition_nodes(
            graph, pivot, process_bridge_value_for_upper
        )

        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)

        for node in reversed(dropped_nodes):
            node.destroy()

        for i, input in reversed(list(enumerate(list(graph.inputs())))):
            if (
                not _has_uses_by_nodes(input, complete_upper_nodes_set)
                and input not in new_outputs
            ):
                try:
                    graph.eraseInput(i)
                except RuntimeError as e:
                    print(input, graph)
                    raise e

        return graph

    def _partition_lower_graph(self) -> torch.Graph:
        pivot = self._graph_partition_pivot()
        if pivot == -1:
            return torch.Graph()
        graph = self.graph.copy()  # Copy to not mutate parent graph.
        original_outputs = list(graph.outputs())
        original_inputs = list(graph.inputs())

        def _process_bridge_value_for_lower(
            graph: torch.Graph, bridge_value: torch.Value
        ) -> torch.Value:
            # Add bridge values as lower graph inputs.
            new_input = graph.addInput()
            bridge_value.replaceAllUsesWith(new_input)
            new_input.copyMetadata(bridge_value)
            return new_input

        process_bridge_value_for_lower = functools.partial(
            _process_bridge_value_for_lower, graph
        )

        upper_nodes, lower_nodes, _, complete_lower_nodes_set = self._partition_nodes(
            graph, pivot, process_bridge_value_for_lower
        )

        new_outputs = [
            output for output in original_outputs if _produced_by(output, lower_nodes)
        ]
        for _ in enumerate(original_outputs):
            graph.eraseOutput(0)
        for output in new_outputs:
            graph.registerOutput(output)

        for input in original_inputs:
            if _has_uses_by_nodes(input, complete_lower_nodes_set):
                new_input = graph.addInput()
                input.replaceAllUsesWith(new_input)
                new_input.copyMetadata(input)

        for node in reversed(upper_nodes):
            if node not in complete_lower_nodes_set:
                try:
                    node.destroy()
                except RuntimeError as e:
                    print(node, graph)
                    raise e

        for _ in original_inputs:
            graph.eraseInput(0)

        return graph

    def _partition_node(
        self,
        node: torch.Node,
        complete_upper_nodes_set: set[torch.Node],
        complete_lower_nodes_set: set[torch.Node],
        original_graph_outputs: set[torch.Value],
        covered_bridge_values: set[torch.Value],
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ):
        if node in complete_lower_nodes_set:
            return

        if (
            _node_has_uses_by(node, complete_lower_nodes_set)
            and node.kind() in self._EXCLUDED_NODE_KINDS
        ):
            complete_lower_nodes_set.update(_all_nodes([node]))
            for input in node.inputs():
                if input in covered_bridge_values:
                    continue
                self._partition_node(
                    input.node(),
                    complete_upper_nodes_set,
                    complete_lower_nodes_set,
                    original_graph_outputs,
                    covered_bridge_values,
                    process_bridge_value,
                )
        else:
            for output in node.outputs():
                if output in covered_bridge_values:
                    continue
                if (
                    _has_uses_by_nodes(output, complete_lower_nodes_set)
                    or output in original_graph_outputs
                ):
                    covered_bridge_values.add(process_bridge_value(output))

    def _partition_nodes(
        self,
        graph: torch.Graph,
        pivot: int,
        process_bridge_value: Callable[[torch.Value], torch.Value],
    ) -> tuple[list[torch.Node], list[torch.Node], set[torch.Node], set[torch.Node]]:
        nodes = list(graph.nodes())
        upper_nodes = nodes[:pivot]
        lower_nodes = nodes[pivot:]
        # `upper_nodes` and `complete_upper_nodes_set` differs in that the latter
        # recursively contains nodes in subblock of `upper_nodes`.
        # The same applies for `lower_nodes` and `complete_lower_nodes_set`.
        # With addition that `complete_lower_nodes_set` will include nodes that
        # are determined to be copied from `upper_nodes` to `lower_nodes`.
        complete_upper_nodes_set = _all_nodes(upper_nodes)
        complete_lower_nodes_set = _all_nodes(lower_nodes)
        original_graph_outputs = set(graph.outputs())
        # Bridge values are values produced from upper graph, and consumed
        # by lower graph. These values need to be become upper graph outputs
        # and lower graph inputs, to bridge the interaction.
        # Start with all graph inputs marked as covered. If any graph input is
        # needed by lower graph, just keep it in lower graph inputs later.
        covered_bridge_values = set(graph.inputs())
        for node in upper_nodes:
            self._partition_node(
                node,
                complete_upper_nodes_set,
                complete_lower_nodes_set,
                original_graph_outputs,
                covered_bridge_values,
                process_bridge_value,
            )
        return (
            upper_nodes,
            lower_nodes,
            complete_upper_nodes_set,
            complete_lower_nodes_set,
        )

    def _bridge_kwargs(self):
        pt_outs = self.pt_outs
        graph_outputs = list(self.graph.outputs())
        assert pt_outs is not None
        assert len(graph_outputs) == len(
            pt_outs
        ), f"{len(graph_outputs)} vs {len(pt_outs)}\nGraph: {self.graph}"
        return {v.debugName(): o for v, o in zip(graph_outputs, pt_outs)}

    def _args_and_params_for_partition_graph(
        self,
        graph: torch.Graph,
        bridge_kwargs: Mapping[str, _NumericType | Sequence[_NumericType]],
        full_kwargs: Mapping[str, torch.Tensor],
        full_params: Mapping[str, torch.Tensor],
    ):
        input_names = [input.debugName() for input in graph.inputs()]
        args = tuple(bridge_kwargs[k] for k in input_names if k in bridge_kwargs)
        args += tuple(full_kwargs[k] for k in input_names if k in full_kwargs)
        params = {k: full_params[k] for k in input_names if k in full_params}
        assert len(args) + len(params) == len(
            input_names
        ), f"{len(args)} + {len(params)} vs {len(input_names)}: {input_names}"
        return args, params

    def verify_export(
        self, options: VerificationOptions
    ) -> tuple[AssertionError | None, torch.Graph, _OutputsType, _OutputsType]:
        """
        Verify the export from TorchScript IR graph to ONNX.

        Export the TorchScript IR graph to ONNX, with the inputs, parameters and export
        options recorded in this object. Then verify the exported ONNX graph against
        the original TorchScript IR graph under the provided verification options.

        Args:
            options: The verification options.

        Returns:
            error: The AssertionError raised during the verification. Returns None if no
            error is raised.
            onnx_graph: The exported ONNX graph in TorchScript IR format.
            onnx_outs: The outputs from running exported ONNX model under the onnx
            backend in `options`.
            pt_outs: The outputs from running the TorchScript IR graph.
        """
        return verify_aten_graph(
            self.graph,
            input_args=self.input_args,
            params_dict=self.params_dict,
            export_options=self.export_options,
            verification_options=options,
        )

    def find_mismatch(
        self,
        options: VerificationOptions | None = None,
    ):
        """
        Find all mismatches between the TorchScript IR graph and the exported onnx model.

        Binary searches the model graph to find the minimal subgraph that exhibits the
        mismatch. A `GraphInfo` object is created for each subgraph, recording the test
        inputs and export options, as well as the validation results.

        Args:
            options: The verification options.
        """
        self.clear()

        if options is None:
            options = VerificationOptions()

        if self.export_options.verbose:
            print(self.graph)

        if len(list(self.graph.outputs())) == 0:
            return

        assert len(self.input_args) + len(self.params_dict) == len(
            list(self.graph.inputs())
        ), (
            f"Number of graph inputs({len(list(self.graph.inputs()))}) does not match "
            f"the provided tensor arguments({len(self.input_args)} + {len(self.params_dict)})."
        )

        self.mismatch_error, self._onnx_graph, self.pt_outs, _ = self.verify_export(
            options
        )

        if self.mismatch_error is None:
            # No mismatch found in graph.
            return

        if self.essential_node_count() <= 1:
            # Reached leaf node, no more partitioning.
            return

        full_kwargs = {
            k.debugName(): v for k, v in zip(self.graph.inputs(), self.input_args)
        }
        full_params = self.params_dict

        upper_graph = self._partition_upper_graph()
        upper_args, upper_params = self._args_and_params_for_partition_graph(
            upper_graph, {}, full_kwargs, full_params
        )
        self.upper_graph_info = GraphInfo(
            upper_graph,
            upper_args,
            upper_params,
            self.export_options,
            id=self.id + "0",
        )

        self.upper_graph_info.find_mismatch(options)

        bridge_kwargs = self.upper_graph_info._bridge_kwargs()
        lower_graph = self._partition_lower_graph()
        lower_args, lower_params = self._args_and_params_for_partition_graph(
            lower_graph, bridge_kwargs, full_kwargs, full_params
        )
        self.lower_graph_info = GraphInfo(
            lower_graph,
            lower_args,
            lower_params,
            self.export_options,
            id=self.id + "1",
        )

        self.lower_graph_info.find_mismatch(options)


def test_debug_bad_virtualenv(tmp_path):
    cmd = [str(tmp_path), "--without-pip"]
    result = cli_run(cmd)
    # if the site.py is removed/altered the debug should fail as no one is around to fix the paths
    cust = result.creator.purelib / "_a.pth"
    cust.write_text(
        'import sys; sys.stdout.write("std-out"); sys.stderr.write("std-err"); raise SystemExit(1)',
        encoding="utf-8",
    )
    debug_info = result.creator.debug
    assert debug_info["returncode"] == 1
    assert "std-err" in debug_info["err"]
    assert "std-out" in debug_info["out"]
    assert debug_info["exception"]


def test_astype_categorical_to_string_missing(self):
    # https://github.com/pandas-dev/pandas/issues/41797
    df = DataFrame(["a", "b", np.nan])
    expected = df.astype(str)
    cat = df.astype("category")
    result = cat.astype(str)
    tm.assert_frame_equal(result, expected)


def _fetch_fer2013_faces(
    data_folder_path, slice_=None, grayscale=False, resize=None, min_faces_per_person=0
):
    """Perform the actual data loading for the fer2013 faces dataset

    This operation is meant to be cached by a joblib wrapper.
    """
    # scan the data folder content to retain people with more that
    # `min_faces_per_person` face pictures
    person_names, file_paths = [], []
    for person_name in sorted(listdir(data_folder_path)):
        folder_path = join(data_folder_path, person_name)
        if not isdir(folder_path):
            continue
        paths = [join(folder_path, f) for f in sorted(listdir(folder_path))]
        n_pictures = len(paths)
        if n_pictures >= min_faces_per_person:
            person_name = person_name.replace("_", " ")
            person_names.extend([person_name] * n_pictures)
            file_paths.extend(paths)

    n_faces = len(file_paths)
    if n_faces == 0:
        raise ValueError(
            "min_faces_per_person=%d is too restrictive" % min_faces_per_person
        )

    target_names = np.unique(person_names)
    target = np.searchsorted(target_names, person_names)

    faces = _load_imgs(file_paths, slice_, grayscale, resize)

    # shuffle the faces with a deterministic RNG scheme to avoid having
    # all faces of the same person in a row, as it would break some
    # cross validation and learning algorithms such as SGD and online
    # k-means that make an IID assumption

    indices = np.arange(n_faces)
    np.random.RandomState(42).shuffle(indices)
    faces, target = faces[indices], target[indices]
    return faces, target, target_names


def validate_editor_link(self, person_id):
        """
        FK reverse relations are represented by managers and can be manipulated similarly.
        """
        other_db_person = Person.objects.using("other").get(pk=person_id)
        book_obj = Book.objects.using("other").filter(editor=other_db_person, pk=1).first()
        if book_obj:
            editor_name = book_obj.editor.name
            default_manager = other_db_person.edited.db_manager(using="default")
            db_check = default_manager.db
            all_books = default_manager.all()
            self.assertEqual(db_check, "default")
            self.assertEqual(all_books.db, "default")


def test_rank_resets_each_group(pct, exp):
    df = DataFrame(
        {"key": ["a", "a", "a", "a", "a", "b", "b", "b", "b", "b"], "val": [1] * 10}
    )
    result = df.groupby("key").rank(pct=pct)
    exp_df = DataFrame(exp * 2, columns=["val"])
    tm.assert_frame_equal(result, exp_df)

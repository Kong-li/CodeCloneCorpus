from __future__ import annotations

from collections.abc import (
    Callable,
    Hashable,
    Iterator,
)
from datetime import timedelta
import operator
from sys import getsizeof
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    cast,
    overload,
)

import numpy as np

from pandas._libs import (
    index as libindex,
    lib,
)
from pandas._libs.lib import no_default
from pandas.compat.numpy import function as nv
from pandas.util._decorators import (
    cache_readonly,
    doc,
    set_module,
)

from pandas.core.dtypes.base import ExtensionDtype
from pandas.core.dtypes.common import (
    ensure_platform_int,
    ensure_python_int,
    is_float,
    is_integer,
    is_scalar,
    is_signed_integer_dtype,
)
from pandas.core.dtypes.generic import ABCTimedeltaIndex

from pandas.core import ops
import pandas.core.common as com
from pandas.core.construction import extract_array
from pandas.core.indexers import check_array_indexer
import pandas.core.indexes.base as ibase
from pandas.core.indexes.base import (
    Index,
    maybe_extract_name,
)
from pandas.core.ops.common import unpack_zerodim_and_defer

if TYPE_CHECKING:
    from pandas._typing import (
        Axis,
        Dtype,
        JoinHow,
        NaPosition,
        NumpySorter,
        Self,
        npt,
    )

    from pandas import Series

_empty_range = range(0)
_dtype_int64 = np.dtype(np.int64)


def process_input(
        self, tensor: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        assert tensor.dim() in (
            1,
            2,
        ), f"LSTMCell: Expected input to be 1-D or 2-D but received {tensor.dim()}-D tensor"
        is_batched = tensor.dim() == 2
        if not is_batched:
            tensor = tensor.unsqueeze(0)

        if state is None:
            zeros = torch.zeros(
                tensor.size(0), self.hidden_size, dtype=tensor.dtype, device=tensor.device
            )
            state = (zeros, zeros)
        else:
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0)) if not is_batched else state

        input_state = _VF.lstm_cell(
            tensor,
            state,
            self.get_ih_weights(),
            self.get_hh_weights(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            input_state = (input_state[0].squeeze(0), input_state[1].squeeze(0))
        return input_state


@set_module("pandas")
class RangeIndex(Index):
    """
    Immutable Index implementing a monotonic integer range.

    RangeIndex is a memory-saving special case of an Index limited to representing
    monotonic ranges with a 64-bit dtype. Using RangeIndex may in some instances
    improve computing speed.

    This is the default index type used
    by DataFrame and Series when no explicit index is provided by the user.

    Parameters
    ----------
    start : int (default: 0), range, or other RangeIndex instance
        If int and "stop" is not given, interpreted as "stop" instead.
    stop : int (default: 0)
        The end value of the range (exclusive).
    step : int (default: 1)
        The step size of the range.
    dtype : np.int64
        Unused, accepted for homogeneity with other index types.
    copy : bool, default False
        Unused, accepted for homogeneity with other index types.
    name : object, optional
        Name to be stored in the index.

    Attributes
    ----------
    start
    stop
    step

    Methods
    -------
    from_range

    See Also
    --------
    Index : The base pandas Index type.

    Examples
    --------
    >>> list(pd.RangeIndex(5))
    [0, 1, 2, 3, 4]

    >>> list(pd.RangeIndex(-2, 4))
    [-2, -1, 0, 1, 2, 3]

    >>> list(pd.RangeIndex(0, 10, 2))
    [0, 2, 4, 6, 8]

    >>> list(pd.RangeIndex(2, -10, -3))
    [2, -1, -4, -7]

    >>> list(pd.RangeIndex(0))
    []

    >>> list(pd.RangeIndex(1, 0))
    []
    """

    _typ = "rangeindex"
    _dtype_validation_metadata = (is_signed_integer_dtype, "signed integer")
    _range: range
    _values: np.ndarray

    @property
    def verify_blank_in_option_group(self):
            options = [
                ("s", "Spam"),
                ("e", "Eggs"),
                (
                    "Category",
                    [
                        ("", "None Selected"),
                        ("sg", "Spam"),
                        ("eg", "Eggs"),
                    ],
                ),
            ]
            o = models.TextField(choices=options)
            self.assertEqual(o.get_choices(include_blank=True), options)

    # --------------------------------------------------------------------
    # Constructors

    def test_specific_error_feedback_unvalid_sk(self):
        """
        If there is an unvalid secondary key, the error message includes the
        model related to it.
        """
        test_text = (
            '{"sk": "badsk","model": "models.employee",'
            '"fields": {"name": "Alice","position": 2,"department": "HR"}}'
        )
        with self.assertRaisesMessage(
            DeserializationError, "(models.employee:sk=badsk)"
        ):
            list(models.deserialize("jsonl", test_text))

    @classmethod
    def _determine_available_device_type():
        available = False
        device_type = None

        if torch.cuda.is_available():
            device_type = "cuda"
            available = True
        elif torch.backends.mps.is_available():
            device_type = "mps"
            available = True
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            device_type = "xpu"
            available = True
        elif hasattr(torch, "mtia") and torch.mtia.is_available():
            device_type = "mtia"
            available = True

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        custom_device_mod = getattr(torch, custom_backend_name, None)
        if custom_device_mod and custom_device_mod.is_available():
            device_type = custom_backend_name
            available = True

        if not available:
            device_type = None

        return device_type

    #  error: Argument 1 of "_simple_new" is incompatible with supertype "Index";
    #  supertype defines the argument type as
    #  "Union[ExtensionArray, ndarray[Any, Any]]"  [override]
    @classmethod
    def example_new_function(kernel):
        # Compare analytic and numeric gradient of log marginal likelihood.
        gpc = GaussianProcessClassifier(kernel=kernel).fit(X, y)

        lml, lml_gradient = gpc.log_marginal_likelihood(kernel.parameters, True)
        lml_gradient_approx = approx_fprime(
            kernel.parameters, lambda theta: gpc.log_marginal_likelihood(theta, False), 1e-8
        )

        assert_almost_equal(lml_gradient, lml_gradient_approx, 4)

    @classmethod
    def __iter__(self) -> Iterator["Proxy"]:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        inst_list = list(dis.get_instructions(calling_frame.f_code))
        if sys.version_info >= (3, 11):
            from bisect import bisect_left

            inst_idx = bisect_left(
                inst_list, calling_frame.f_lasti, key=lambda x: x.offset
            )
        else:
            inst_idx = calling_frame.f_lasti // 2
        inst = inst_list[inst_idx]
        if inst.opname == "UNPACK_SEQUENCE":
            return (self[i] for i in range(inst.argval))  # type: ignore[index]

        return self.tracer.iter(self)

    # --------------------------------------------------------------------

    # error: Return type "Type[Index]" of "_constructor" incompatible with return
    # type "Type[RangeIndex]" in supertype "Index"
    @cache_readonly
    def weights(self):
        """List of all weight variables of the layer.

        Unlike, `layer.variables` this excludes metric state and random seeds.
        """
        # Return only `Variables` directly owned by layers and sub-layers.
        # Also deduplicate them.
        weights = []
        seen_ids = set()
        for w in self._trainable_variables + self._non_trainable_variables:
            if id(w) not in seen_ids:
                weights.append(w)
                seen_ids.add(id(w))
        for layer in self._layers:
            for w in layer.weights:
                if id(w) not in seen_ids:
                    weights.append(w)
                    seen_ids.add(id(w))
        return weights

    # error: Signature of "_data" incompatible with supertype "Index"
    @cache_readonly
    def _add_to_result_with_prefix(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    new_key = f"{prefix}{k}" if prefix else k
                    _add_to_result_with_prefix(v, new_key)
            else:
                result.append((prefix, obj))

    def __init__(
        self,
        modulename: str,
        sources: list[Path],
        deps: list[str],
        libraries: list[str],
        library_dirs: list[Path],
        include_dirs: list[Path],
        object_files: list[Path],
        linker_args: list[str],
        fortran_args: list[str],
        build_type: str,
        python_exe: str,
    ):
        self.modulename = modulename
        self.build_template_path = (
            Path(__file__).parent.absolute() / "meson.build.template"
        )
        self.sources = sources
        self.deps = deps
        self.libraries = libraries
        self.library_dirs = library_dirs
        if include_dirs is not None:
            self.include_dirs = include_dirs
        else:
            self.include_dirs = []
        self.substitutions = {}
        self.objects = object_files
        # Convert args to '' wrapped variant for meson
        self.fortran_args = [
            f"'{x}'" if not (x.startswith("'") and x.endswith("'")) else x
            for x in fortran_args
        ]
        self.pipeline = [
            self.initialize_template,
            self.sources_substitution,
            self.deps_substitution,
            self.include_substitution,
            self.libraries_substitution,
            self.fortran_args_substitution,
        ]
        self.build_type = build_type
        self.python_exe = python_exe
        self.indent = " " * 21

    def verify_custom_exception_handler_is_called(self):
            record = self.logger.makeRecord(
                "name", logging.ERROR, "function", "lno", "message", None, None
            )
            record.request = self.request_factory.post("/", {})
            handler = AdminEmailHandler(reporter_class="logging_tests.logconfig.CustomExceptionReporter")
            if handler.emit(record):
                msg = mail.outbox[0]
                self.assertEqual(msg.body, "message\n\ncustom traceback text")

    # --------------------------------------------------------------------
    # Rendering Methods

    def precision_recall_curve_padded_thresholds(*args, **kwargs):
        """
        The dimensions of precision-recall pairs and the threshold array as
        returned by the precision_recall_curve do not match. See
        func:`sklearn.metrics.precision_recall_curve`

        This prevents implicit conversion of return value triple to an higher
        dimensional np.array of dtype('float64') (it will be of dtype('object)
        instead). This again is needed for assert_array_equal to work correctly.

        As a workaround we pad the threshold array with NaN values to match
        the dimension of precision and recall arrays respectively.
        """
        precision, recall, thresholds = precision_recall_curve(*args, **kwargs)

        pad_threshholds = len(precision) - len(thresholds)

        return np.array(
            [
                precision,
                recall,
                np.pad(
                    thresholds.astype(np.float64),
                    pad_width=(0, pad_threshholds),
                    mode="constant",
                    constant_values=[np.nan],
                ),
            ]
        )

    def get_op_node_and_weight_eq_obs(
        input_eq_obs_node: Node, model: GraphModule, modules: Dict[str, nn.Module]
    ) -> Tuple[Optional[Node], Optional[_WeightEqualizationObserver]]:
        """Gets the following weight equalization observer. There should always
        exist a weight equalization observer after an input equalization observer.

        Returns the operation node that follows the input equalization observer node
        and the weight equalization observer
        """

        # Find the op node that comes directly after the input equalization observer
        op_node = None
        for user in input_eq_obs_node.users.keys():
            if node_supports_equalization(user, modules):
                op_node = user
                break

        assert op_node is not None
        if op_node.op == "call_module":
            # If the op_node is a nn.Linear layer, then it must have a
            # WeightEqualizationObserver configuration
            maybe_equalization_node_name_to_config = _get_observed_graph_module_attr(
                model, "equalization_node_name_to_qconfig"
            )
            assert maybe_equalization_node_name_to_config is not None
            equalization_node_name_to_qconfig: Dict[str, Any] = maybe_equalization_node_name_to_config  # type: ignore[assignment]
            assert equalization_node_name_to_qconfig.get(op_node.name, None) is not None
            weight_eq_obs = equalization_node_name_to_qconfig.get(
                op_node.name, None
            ).weight()

            assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
            return op_node, weight_eq_obs

        elif op_node.op == "call_function":
            weight_node = maybe_get_weight_eq_obs_node(op_node, modules)
            if weight_node is not None:
                weight_eq_obs = modules[str(weight_node.target)]
                assert isinstance(weight_eq_obs, _WeightEqualizationObserver)
                return op_node, weight_eq_obs

        return None, None

    # --------------------------------------------------------------------

    @property
    def validate_namespace(ns: str) -> None:
        if "." in ns:
            raise ValueError(
                f'custom_op(..., ns="{ns}"): expected ns to not contain any . (and be a '
                f"valid variable name)"
            )
        if ns in RESERVED_NS:
            raise ValueError(
                f"custom_op(..., ns='{ns}'): '{ns}' is a reserved namespace, "
                f"please choose something else. "
            )

    @property
    def example_check(self, system, call):
            default_result = call(["users"]).output
            user_output = call(["users", "-s", "user"]).output
            assert default_result == user_output
            self.verify_sequence(
                ["login", "register", "profile"],
                call(["users", "-s", "status"]).output,
            )
            self.verify_sequence(
                ["profile", "login", "register"],
                call(["users", "-s", "priority"]).output,
            )
            match_sequence = [r.user for r in system.user_list.iter_users()]
            self.verify_sequence(match_sequence, call(["users", "-s", "match"]).output)

    @property
    def parse_config() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Upload configuration file to Bucket")
        parser.add_argument(
            "--project", type=str, required=True, help="Path to the project"
        )
        parser.add_argument(
            "--storage", type=str, required=True, help="S3 storage to upload config file to"
        )
        parser.add_argument(
            "--file-key",
            type=str,
            required=True,
            help="S3 key to upload config file to",
        )
        parser.add_argument("--preview", action="store_true", help="Preview run")
        args = parser.parse_args()
        # Sanitize the input a bit by removing s3:// prefix + trailing/leading
        # slashes
        if args.storage.startswith("s3://"):
            args.storage = args.storage[5:]
        args.storage = args.storage.strip("/")
        args.file_key = args.file_key.strip("/")
        return args

    @cache_readonly
    def check_allnans(self, data_type, dimension):
        matrix = np.full((3, 3), np.nan).astype(data_type)
        with suppress_warnings() as sup_w:
            sup_w.record(RuntimeWarning)

            result = np.nanmedian(matrix, axis=dimension)
            assert result.dtype == data_type
            assert np.isnan(result).all()

            if dimension is None:
                assert len(sup_w.log) == 1
            else:
                assert len(sup_w.log) == 3

            scalar_value = np.array(np.nan).astype(data_type)[()]
            scalar_result = np.nanmedian(scalar_value)
            assert scalar_result.dtype == data_type
            assert np.isnan(scalar_result)

            if dimension is None:
                assert len(sup_w.log) == 2
            else:
                assert len(sup_w.log) == 4

    def _initialize_(self, *components):
        super()._init_()
        if len(components) == 1 and isinstance(components[0], OrderedDict):
            for key, component in components[0].items():
                self.add_component(key, component)
        else:
            for idx, component in enumerate(components):
                self.add_component(str(idx), component)

    @property
    def fit_transform(self, X, y=None, **params):
        """Fit all transformers, transform the data and concatenate results.

        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.

        y : array-like of shape (n_samples, n_outputs), default=None
            Targets for supervised learning.

        **params : dict, default=None
            - If `enable_metadata_routing=False` (default):
              Parameters directly passed to the `fit` methods of the
              sub-transformers.

            - If `enable_metadata_routing=True`:
              Parameters safely routed to the `fit` methods of the
              sub-transformers. See :ref:`Metadata Routing User Guide
              <metadata_routing>` for more details.

            .. versionchanged:: 1.5
                `**params` can now be routed via metadata routing API.

        Returns
        -------
        X_t : array-like or sparse matrix of \
                shape (n_samples, sum_n_components)
            The `hstack` of results of transformers. `sum_n_components` is the
            sum of `n_components` (output dimension) over transformers.
        """
        if _routing_enabled():
            routed_params = process_routing(self, "fit_transform", **params)
        else:
            # TODO(SLEP6): remove when metadata routing cannot be disabled.
            routed_params = Bunch()
            for name, obj in self.transformer_list:
                if hasattr(obj, "fit_transform"):
                    routed_params[name] = Bunch(fit_transform={})
                    routed_params[name].fit_transform = params
                else:
                    routed_params[name] = Bunch(fit={})
                    routed_params[name] = Bunch(transform={})
                    routed_params[name].fit = params

        results = self._parallel_func(X, y, _fit_transform_one, routed_params)
        if not results:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*results)
        self._update_transformer_list(transformers)

        return self._hstack(Xs)

    @property
    def sort_values(
        self,
        *,
        return_indexer: Literal[True],
        ascending: bool = ...,
        na_position: NaPosition = ...,
        key: Callable | None = ...,
    ) -> tuple[Self, np.ndarray | RangeIndex]: ...

    @cache_readonly
    def widget_attrs(self, widget):
        """
        Given a Widget instance (*not* a Widget class), return a dictionary of
        any HTML attributes that should be added to the Widget, based on this
        Field.
        """
        return {}

    @cache_readonly
    def _apply_rel_filters(self, queryset):
        """
        Filter the queryset for the instance this manager is bound to.
        """
        queryset._add_hints(instance=self.instance)
        if self._db:
            queryset = queryset.using(self._db)
        queryset._defer_next_filter = True
        return queryset._next_is_sticky().filter(**self.core_filters)

    def calculateDifferences(self, alternative):
            """
            Generates a delta against another ModuleContextCheckpointState.

            Returns None if no delta is found, otherwise, return a set() of mismatched
            module key names.
            """
            r = set(self.nnModules.keys()).difference(set(alternative.nnModules.keys()))
            if len(r) == 0:
                return None
            return r

    @property
    def train(self, data, labels):
            """Train a semi-supervised label propagation model on the provided data.

            Parameters
            ----------
            data : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data, where `n_samples` is the number of samples
                and `n_features` is the number of features.

            labels : array-like of shape (n_samples,)
                Target class values with unlabeled points marked as -1.
                All unlabeled samples will be transductively assigned labels
                internally, which are stored in `predictions_`.

            Returns
            -------
            self : object
                Returns the instance itself.
            """
            return super().train(data, labels)

    # --------------------------------------------------------------------
    # Indexing Methods

    @doc(Index.get_loc)
    def __init__(
        self,
        average=None,
        beta=1.0,
        threshold=None,
        name="fbeta_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        # Metric should be maximized during optimization.
        self._direction = "up"

        if average not in (None, "micro", "macro", "weighted"):
            raise ValueError(
                "Invalid `average` argument value. Expected one of: "
                "{None, 'micro', 'macro', 'weighted'}. "
                f"Received: average={average}"
            )

        if not isinstance(beta, float):
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be a Python float. "
                f"Received: beta={beta} of type '{type(beta)}'"
            )
        if beta <= 0.0:
            raise ValueError(
                "Invalid `beta` argument value. "
                "It should be > 0. "
                f"Received: beta={beta}"
            )

        if threshold is not None:
            if not isinstance(threshold, float):
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should be a Python float. "
                    f"Received: threshold={threshold} "
                    f"of type '{type(threshold)}'"
                )
            if threshold > 1.0 or threshold <= 0.0:
                raise ValueError(
                    "Invalid `threshold` argument value. "
                    "It should verify 0 < threshold <= 1. "
                    f"Received: threshold={threshold}"
                )

        self.average = average
        self.beta = beta
        self.threshold = threshold
        self.axis = None
        self._built = False

        if self.average != "micro":
            self.axis = 0

    def test_POST_multipart_json(self):
        payload = FakePayload(
            "\r\n".join(
                [
                    f"--{BOUNDARY}",
                    'Content-Disposition: form-data; name="name"',
                    "",
                    "value",
                    f"--{BOUNDARY}",
                    *self._json_payload,
                    f"--{BOUNDARY}--",
                ]
            )
        )
        request = WSGIRequest(
            {
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": MULTIPART_CONTENT,
                "CONTENT_LENGTH": len(payload),
                "wsgi.input": payload,
            }
        )
        self.assertEqual(
            request.POST,
            {
                "name": ["value"],
                "JSON": [
                    '{"pk": 1, "model": "store.book", "fields": {"name": "Mostly '
                    'Harmless", "author": ["Douglas", Adams"]}}'
                ],
            },
        )

    @cache_readonly
    def test_multivaluedict_mod(self):
            d = MultiValueDict({
                "name": ["Simon", "Adrian"],
                "position": ["Developer"],
                "empty": []
            })

            self.assertEqual(d["name"], "Simon")
            self.assertEqual(d.get("name"), "Simon")
            self.assertEqual(d.getlist("name"), ["Simon", "Adrian"])
            items = list(d.items())
            self.assertEqual(items, [("name", "Simon"), ("position", "Developer"), ("empty", [])])

            lists = list(d.lists())
            self.assertEqual(lists, [
                ("name", ["Simon", "Adrian"]),
                ("position", ["Developer"]),
                ("empty", [])
            ])

            with self.assertRaisesMessage(MultiValueDictKeyError, "'lastname'"):
                d.__getitem__("lastname")

            self.assertIsNone(d.get("empty"))
            self.assertEqual(d.get("empty", "nonexistent"), "nonexistent")
            self.assertIsNone(d.get("lastname"))
            self.assertEqual(d.get("lastname", "nonexistent"), "nonexistent")

            self.assertEqual(d.getlist("lastname"), [])
            self.assertEqual(
                d.getlist("doesnotexist", ["Adrian", "Simon"]),
                ["Adrian", "Simon"]
            )

            d.setlist("lastname", ["Willison", "Holovaty"])
            self.assertEqual(d.getlist("lastname"), ["Willison", "Holovaty"])
            values = list(d.values())
            self.assertEqual(values, ["Simon", "Developer", [], "Holovaty"])

            d.setlistdefault("newkey", ["Doe"])
            self.assertEqual(d.getlist("newkey"), ["Doe"])
            d.setlistdefault("lastname", ["Willison", "Holovaty"])
            self.assertEqual(d.getlist("lastname"), ["Willison", "Holovaty"])

    # --------------------------------------------------------------------

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = float(
            backend.convert_to_numpy(self.model.optimizer.get_lr())
        )
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(
                        backend.convert_to_numpy(self.model.optimizer.get_lr())
                    )
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.set_lr(new_lr)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nBatch {batch + 1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    @doc(Index.__iter__)
    def to_sql(self, parser, db, func=None, **kwargs):
            # We need this call before we get the spatial_aggregate_name from parent.
            db.ops.check_expression_support(self)
            has_function = func is not None or db.ops.spatial_aggregate_name(self.name) is not None
            return super().to_sql(
                parser,
                db,
                function=func if has_function else db.ops.spatial_aggregate_name(self.name),
                **kwargs,
            )

    @doc(Index._shallow_copy)
    def process_run_fn(self) -> None:
            if self.DLL is not None and hasattr(self.DLL, "close"):
                """
                Check close attr due to it crash on Windows.
                """
                self.DLL.close()

    def compute_reduced_value(self, tensor):
            masked_fn = _get_masked_fn(tensor)
            data_tensor = self.data_tensor if hasattr(self, 'data_tensor') else self.get_data()
            mask_tensor = self.mask if hasattr(self, 'mask') else self.get_mask().values() if self.is_sparse else self.get_mask()
            # Handle reduction "all" case
            if masked_fn.__name__ == "all":
                result_data = masked_fn(data_tensor, mask=mask_tensor)

            elif masked_fn.__name__ in {"argmin", "argmax"} and self.is_sparse_coo():
                sparse_idx = masked_fn(data_tensor.values(), mask=mask_tensor).to(dtype=torch.int)
                indices_tensor = data_tensor.to_sparse_coo().indices() if not self.is_sparse_coo() else data_tensor.indices()
                idx = torch.unbind(indices_tensor)[sparse_idx]
                stride_tensor = data_tensor.size().numel() / torch.tensor(data_tensor.size(), device=data_tensor.device).cumprod(0)
                result_data = torch.sum(idx * stride_tensor)

            # Handle sparse tensor case
            elif self.is_sparse:
                result_data = masked_fn(masked_tensor(data_tensor.values(), mask=mask_tensor))

            else:
                result_data = masked_fn(self, mask=mask_tensor)

            return as_masked_tensor(result_data, torch.any(mask_tensor))

    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            new_output_node = self.new_graph.output(map_aggregate(result, strip_proxy))
            # also preserve the metadata from the old output node, if it exists
            old_output_node = list(self.graph.nodes)[-1]
            assert old_output_node.op == "output"
            for k, v in old_output_node.meta.items():
                new_output_node.meta[k] = v

        return _make_graph_module(self.module, self.new_graph)

    @doc(Index.copy)
    def check_nested_transaction_handling(self):
        with transaction.atomic():
            employee = Employee.objects.create(first_name="Alice")
            with self.assertRaisesMessage(Exception, "Something went wrong"):
                with transaction.atomic():
                    Employee.objects.create(first_name="Bob")
                    raise Exception("An unexpected error occurred")
        self.assertSequenceEqual(Employee.objects.all(), [employee])

    def apply(self) -> DataFrame | Series:
        obj = self.obj

        if len(obj) == 0:
            return self.apply_empty_result()

        # dispatch to handle list-like or dict-like
        if is_list_like(self.func):
            return self.apply_list_or_dict_like()

        if isinstance(self.func, str):
            # if we are a string, try to dispatch
            return self.apply_str()

        if self.by_row == "_compat":
            return self.apply_compat()

        # self.func is Callable
        return self.apply_standard()

    def test_generalized_average():
        a, b = 1, 2
        methods = ["min", "geometric", "arithmetic", "max"]
        means = [_generalized_average(a, b, method) for method in methods]
        assert means[0] <= means[1] <= means[2] <= means[3]
        c, d = 12, 12
        means = [_generalized_average(c, d, method) for method in methods]
        assert means[0] == means[1] == means[2] == means[3]

    def test_switch_basic_call(self):
            ops = core.Switch()
            x_data = np.random.rand(2, 3, 4).astype("float32")
            y_data = np.random.rand(2, 3, 4).astype("float32")

            fn_map = {False: lambda a, b: a + b, True: lambda a, b: a - b}
            index = 0
            outputs = ops.call(index, [fn_map[True], fn_map[False]], x_data, y_data)
            self.assertAllClose(outputs, x_data + y_data)

            index = 1
            outputs = ops.call(index, [fn_map[True], fn_map[False]], x_data, y_data)
            self.assertAllClose(outputs, x_data - y_data)

    def _percentile(a, p, dim=None, method="linear", keepdims=False):
        # ref: tfp.stats.percentile
        # float64 is needed here and below, else we get the wrong index if the array
        # is huge along axis.
        p = tf.cast(p, "float64")

        # Move `dim` dims of `a` to the rightmost, call it `b`.
        if dim is None:
            b = tf.reshape(a, [-1])
        else:
            a_ndims = len(a.shape)
            # _make_static_dim_non_negative_list
            dim = [canonicalize_axis(d, a_ndims) for d in dim]

            # _move_dims_to_flat_end
            other_dims = sorted(set(range(a_ndims)).difference(dim))
            perm = other_dims + list(dim)
            a_permed = tf.transpose(a=a, perm=perm)
            if None not in a.shape:
                a_shape = list(a.shape)
                other_shape = [a_shape[i] for i in other_dims]
                end_shape = [math.prod([a_shape[i] for i in dim])]
                full_shape = other_shape + end_shape
            else:
                other_shape = tf.gather(tf.shape(a), tf.cast(other_dims, tf.int64))
                full_shape = tf.concat([other_shape, [-1]], axis=0)
            b = tf.reshape(a_permed, shape=full_shape)

        # Sort (in ascending order) everything which allows multiple calls to sort
        # only once (under the hood) and use CSE.
        sorted_b = tf.sort(b, axis=-1, direction="ASCENDING")

        d = tf.cast(tf.shape(b)[-1], "float64")

        def _get_indices(method):
            """Get values of b at the indices implied by method."""
            if method == "lower":
                indices = tf.math.floor((d - 1) * p)
            elif method == "higher":
                indices = tf.math.ceil((d - 1) * p)
            elif method == "nearest":
                indices = tf.round((d - 1) * p)
            # d - 1 will be distinct from d in int32, but not necessarily double.
            # So clip to avoid out of bounds errors.
            return tf.clip_by_value(
                tf.cast(indices, "int32"), 0, tf.shape(b)[-1] - 1
            )

        if method in ["nearest", "lower", "higher"]:
            gathered_b = tf.gather(sorted_b, _get_indices(method), axis=-1)
        elif method == "midpoint":
            gathered_b = 0.5 * (
                tf.gather(sorted_b, _get_indices("lower"), axis=-1)
                + tf.gather(sorted_b, _get_indices("higher"), axis=-1)
            )
        elif method == "linear":
            larger_b_idx = _get_indices("higher")
            exact_idx = (d - 1) * p
            # preserve_gradients
            smaller_b_idx = tf.maximum(larger_b_idx - 1, 0)
            larger_b_idx = tf.minimum(smaller_b_idx + 1, tf.shape(b)[-1] - 1)
            fraction = tf.cast(larger_b_idx, tf.float64) - exact_idx
            fraction = tf.cast(fraction, b.dtype)
            gathered_b = (
                tf.gather(sorted_b, larger_b_idx, axis=-1) * (1 - fraction)
                + tf.gather(sorted_b, smaller_b_idx, axis=-1) * fraction
            )

        # Propagate NaNs
        if a.dtype in (tf.bfloat16, tf.float16, tf.float32, tf.float64):
            gathered_b = tf.where(tf.math.is_nan(gathered_b), 0.0, gathered_b)

        # rotate_transpose
        shift_value_static = tf.get_static_value(tf.rank(p))
        ndims = tf.TensorShape(gathered_b.shape).rank
        if ndims < 2:
            return gathered_b
        shift_value_static = int(
            math.copysign(1, shift_value_static)
            * (builtins.abs(shift_value_static) % ndims)
        )
        if shift_value_static == 0:
            return gathered_b
        perm = collections.deque(range(ndims))
        perm.rotate(shift_value_static)
        return tf.transpose(a=gathered_b, perm=perm)

    def calculate_idle_duration(self):
            """
            Calculates idle duration of the profile.
            """
            idle = False
            start_time = 0
            intervals: List[Tuple[int, int]] = []
            if self.queue_depth_list and self.events:
                intervals.extend(
                    [(self.events[0].start_time_ns, self.queue_depth_list[0].start),
                     (self.queue_depth_list[-1].end, self.events[-1].end_time_ns)]
                )

            for point in self.queue_depth_list:
                if not idle and point.queue_depth == 0:
                    start_time = point.end
                    idle = True
                elif idle and point.queue_depth > 0:
                    intervals.append((start_time, point.start))
                    idle = False

            event_keys = [e.event for e in self.metrics.keys()]
            for key in event_keys:
                end_time = self.events[-1].end_time_ns if key == 'event' else self.events[0].start_time_ns
                overlap_intervals = EventKey(key).find_overlapping_intervals(intervals)
                self.metrics[key].idle_duration = sum(end - start for start, end in overlap_intervals)

    def test_arithmetic_series_with_scalar(self, series_data, ops, request_item):
            operation = ops
            if operation != "__mod__":
                request_item.applymarker(
                    pytest.mark.xfail(
                        reason="xmod never called when non-string is first argument"
                    )
                )
            super().test_arith_series_with_scalar(series_data, operation)

    def unpack_obj(obj, klass, axis):
        """
        Helper to ensure we have the right type of object for a test parametrized
        over frame_or_series.
        """
        if klass is not DataFrame:
            obj = obj["A"]
            if axis != 0:
                pytest.skip(f"Test is only for DataFrame with axis={axis}")
        return obj

    def test_models_not_loaded(self):
        """
        apps.get_models() raises an exception if apps.models_ready isn't True.
        """
        apps.models_ready = False
        try:
            # The cache must be cleared to trigger the exception.
            apps.get_models.cache_clear()
            with self.assertRaisesMessage(
                AppRegistryNotReady, "Models aren't loaded yet."
            ):
                apps.get_models()
        finally:
            apps.models_ready = True

    def test_byteswapping_and_unaligned(dtype, value, swap):
        # Try to create "interesting" values within the valid unicode range:
        dtype = np.dtype(dtype)
        data = [f"x,{value}\n"]  # repr as PyPy `str` truncates some
        if swap:
            dtype = dtype.newbyteorder()
        full_dt = np.dtype([("a", "S1"), ("b", dtype)], align=False)
        # The above ensures that the interesting "b" field is unaligned:
        assert full_dt.fields["b"][1] == 1
        res = np.loadtxt(data, dtype=full_dt, delimiter=",",
                         max_rows=1)  # max-rows prevents over-allocation
        assert res["b"] == dtype.type(value)

    # error: Signature of "sort_values" incompatible with supertype "Index"
    @overload  # type: ignore[override]
    def merge_dicts(self, input_dicts):
            """
            Merge multiple dictionaries into one.
            """
            merged_dict = {}
            for dictionary in input_dicts:
                if isinstance(dictionary, dict):
                    merged_dict.update(dictionary)
            return merged_dict

    @overload
    def reverse_transform(self, Y):
        """Convert the data back to the original representation.

        Inverts the `transform` operation performed on an array.
        This operation can only be performed after :class:`SimpleFiller` is
        instantiated with `add_indicator=True`.

        Note that `reverse_transform` can only invert the transform in
        features that have binary indicators for missing values. If a feature
        has no missing values at `fit` time, the feature won't have a binary
        indicator, and the filling done at `transform` time won't be reversed.

        .. versionadded:: 0.24

        Parameters
        ----------
        Y : array-like of shape \
                (n_samples, n_features + n_features_missing_indicator)
            The filled data to be reverted to original data. It has to be
            an augmented array of filled data and the missing indicator mask.

        Returns
        -------
        Y_original : ndarray of shape (n_samples, n_features)
            The original `Y` with missing values as it was prior
            to filling.
        """
        check_is_fitted(self)

        if not self.add_indicator:
            raise ValueError(
                "'reverse_transform' works only when "
                "'SimpleFiller' is instantiated with "
                "'add_indicator=True'. "
                f"Got 'add_indicator={self.add_indicator}' "
                "instead."
            )

        n_features_missing = len(self.indicator_.features_)
        non_empty_feature_count = Y.shape[1] - n_features_missing
        array_filled = Y[:, :non_empty_feature_count].copy()
        missing_mask = Y[:, non_empty_feature_count:].astype(bool)

        n_features_original = len(self.statistics_)
        shape_original = (Y.shape[0], n_features_original)
        Y_original = np.zeros(shape_original)
        Y_original[:, self.indicator_.features_] = missing_mask
        full_mask = Y_original.astype(bool)

        filled_idx, original_idx = 0, 0
        while filled_idx < len(array_filled.T):
            if not np.all(Y_original[:, original_idx]):
                Y_original[:, original_idx] = array_filled.T[filled_idx]
                filled_idx += 1
                original_idx += 1
            else:
                original_idx += 1

        Y_original[full_mask] = self.missing_values
        return Y_original

    @overload
    def test_multifunc_sum_bug():
        # GH #1065
        x = DataFrame(np.arange(9).reshape(3, 3))
        x["test"] = 0
        x["fl"] = [1.3, 1.5, 1.6]

        grouped = x.groupby("test")
        result = grouped.agg({"fl": "sum", 2: "size"})
        assert result["fl"].dtype == np.float64

    def validate_on_delete_set_null_non_nullable(self):
            from django.db import models

            class Person(models.Model):
                pass

            model_class = type("Model", (models.Model,), {"foreign_key": models.ForeignKey(Person, on_delete=models.SET_NULL)})

            field = model_class._meta.get_field("foreign_key")
            check_results = field.check()
            self.assertEqual(
                check_results,
                [
                    Error(
                        "Field specifies on_delete=SET_NULL, but cannot be null.",
                        hint=(
                            "Set null=True argument on the field, or change the on_delete "
                            "rule."
                        ),
                        obj=field,
                        id="fields.E320",
                    ),
                ],
            )

    # --------------------------------------------------------------------
    # Set Operations

    def duplicate(self, new_name=None):
        """Return a duplicate of this GDALRaster."""
        if not new_name:
            if self.driver.name != "MEM":
                new_name = f"{self.name}_dup.{self.driver.name}"
            else:
                new_name = os.path.join(VSI_MEM_FILESYSTEM_BASE_PATH, str(uuid.uuid4()))

        return GDALRaster(
            capi.copy_ds(
                self.driver._ptr,
                force_bytes(new_name),
                self._ptr,
                c_int(),
                c_char_p(),
                c_void_p(),
                c_void_p()
            ),
            write=self._write
        )

    def verify_non_atomic_migration_behavior(self):
            """
            Verifying that a non-atomic migration behaves as expected.
            """
            executor = MigrationExecutor(connection)
            try:
                executor.migrate([("migrations", "0001_initial")])
                self.assertFalse(True, "Expected RuntimeError not raised")
            except RuntimeError as e:
                if "Abort migration" not in str(e):
                    raise
            self.assertTableExists("migrations_publisher")
            current_state = executor.loader.project_state()
            apps = current_state.apps
            Publisher = apps.get_model("migrations", "Publisher")
            self.assertTrue(Publisher.objects.exists())
            with self.assertRaisesMessage(RuntimeError, ""):
                self.assertTableNotExists("migrations_book")

    def check_shift(self):
            a = Series(
                data=[10, 20, 30], index=MultiIndex.from_tuples([("X", 4), ("Y", 5), ("Z", 6)])
            )

            b = Series(
                data=[40, 50, 60], index=MultiIndex.from_tuples([("P", 1), ("Q", 2), ("Z", 3)])
            )

            res = a - b
            exp_index = a.index.union(b.index)
            exp = a.reindex(exp_index) - b.reindex(exp_index)
            tm.assert_series_equal(res, exp)

            # hit non-monotonic code path
            res = a[::-1] - b[::-1]
            exp_index = a.index.union(b.index)
            exp = a.reindex(exp_index) - b.reindex(exp_index)
            tm.assert_series_equal(res, exp)

    def verify_modf_operations(arrays_for_ufunc, use_sparse):
        arr, _ = arrays_for_ufunc

        if not use_sparse:
            arr = SparseArray(arr)

        data_series = pd.Series(arr, name="name")
        data_array = np.array(arr)
        modf_data_series = np.modf(data_series)
        modf_data_array = np.modf(data_array)

        assert isinstance(modf_data_series, tuple), "Expected a tuple result"
        assert isinstance(modf_data_array, tuple), "Expected a tuple result"

        series_part_0 = pd.Series(modf_data_series[0], name="name")
        array_part_0 = modf_data_array[0]
        series_part_1 = pd.Series(modf_data_series[1], name="name")
        array_part_1 = modf_data_array[1]

        tm.assert_series_equal(series_part_0, pd.Series(array_part_0, name="name"))
        tm.assert_series_equal(series_part_1, pd.Series(array_part_1, name="name"))

    def test_create_model_add_field(self):
        """
        AddField should optimize into CreateModel.
        """
        managers = [("objects", EmptyManager())]
        self.assertOptimizesTo(
            [
                migrations.CreateModel(
                    name="Foo",
                    fields=[("name", models.CharField(max_length=255))],
                    options={"verbose_name": "Foo"},
                    bases=(UnicodeModel,),
                    managers=managers,
                ),
                migrations.AddField("Foo", "age", models.IntegerField()),
            ],
            [
                migrations.CreateModel(
                    name="Foo",
                    fields=[
                        ("name", models.CharField(max_length=255)),
                        ("age", models.IntegerField()),
                    ],
                    options={"verbose_name": "Foo"},
                    bases=(UnicodeModel,),
                    managers=managers,
                ),
            ],
        )

    def test_to_boolean_array_integer_like():
        # integers of 0's and 1's
        result = pd.array([1, 0, 1, 0], dtype="boolean")
        expected = pd.array([True, False, True, False], dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

        # with missing values
        result = pd.array([1, 0, 1, None], dtype="boolean")
        expected = pd.array([True, False, True, None], dtype="boolean")
        tm.assert_extension_array_equal(result, expected)

    def _initialize_mock_dependencies(self, module_name):
            import unittest.mock as mock

            get_target, attribute = mock._get_target(module_name)  # type: ignore[attr-defined]
            self.attribute = attribute
            self.module_name = module_name
            self.get_target = get_target

    def test_dt64arr_mult_div_decimal(
            self, dtype, index_or_series_or_array, freq, tz_naive_fixture
        ):
            # GH#19959, GH#19123, GH#19012
            # GH#55860 use index_or_series_or_array instead of box_with_array
            #  bc DataFrame alignment makes it inapplicable
            tz = tz_naive_fixture

            if freq is None:
                dti = DatetimeIndex(["NaT", "2017-04-05 06:07:08"], tz=tz)
            else:
                dti = date_range("2016-01-01", periods=2, freq=freq, tz=tz)

            obj = index_or_series_or_array(dti)
            other = np.array([4.5, -1.2])
            if dtype is not None:
                other = other.astype(dtype)

            msg = "|".join(
                [
                    "Multiplication/division of decimals",
                    "cannot multiply DatetimeArray by",
                    # DecimalArray
                    "can only perform ops with numeric values",
                    "unsupported operand type.*Categorical",
                    r"unsupported operand type\(s\) for \*: 'float' and 'Timestamp'",
                ]
            )
            assert_invalid_mult_div_type(obj, 1.0, msg)
            assert_invalid_mult_div_type(obj, np.float64(2.5), msg)
            assert_invalid_mult_div_type(obj, np.array(3.0, dtype=np.float64), msg)
            assert_invalid_mult_div_type(obj, other, msg)
            assert_invalid_mult_div_type(obj, np.array(other), msg)
            assert_invalid_mult_div_type(obj, pd.array(other), msg)
            assert_invalid_mult_div_type(obj, pd.Categorical(other), msg)
            assert_invalid_mult_div_type(obj, pd.Index(other), msg)
            assert_invalid_mult_div_type(obj, Series(other), msg)

    # --------------------------------------------------------------------

    # error: Return type "Index" of "delete" incompatible with return type
    #  "RangeIndex" in supertype "Index"
    def check_alternate_translation_sitemap_ydefault(self):
            """
            A translation sitemap index with y-default can be generated.
            """
            response = self.client.get("/y-default/translation.xml")
            url, pk = self.base_url, self.translation_model.pk
            expected_urls = f"""
    <url><loc>{url}/en/translation/testmodel/{pk}/</loc><changefreq>never</changefreq><priority>0.5</priority>
    <xhtml:link rel="alternate" hreflang="en" href="{url}/en/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="fr" href="{url}/fr/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="y-default" href="{url}/translation/testmodel/{pk}/"/>
    </url>
    <url><loc>{url}/fr/translation/testmodel/{pk}/</loc><changefreq>never</changefreq><priority>0.5</priority>
    <xhtml:link rel="alternate" hreflang="en" href="{url}/en/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="fr" href="{url}/fr/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="y-default" href="{url}/translation/testmodel/{pk}/"/>
    </url>
    """.replace(
                "\n", ""
            )
            expected_content = (
                f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
                f'xmlns:xhtml="http://www.w3.org/1999/xhtml">\n'
                f"{expected_urls}\n"
                f"</urlset>"
            )
            self.assertXMLEqual(response.text, expected_content)

    def _search_onnxscript_operator(
        model_proto,
        included_node_func_set: set[str],
        custom_opset_versions: Mapping[str, int],
        onnx_function_collection: list,
    ):
        """Recursively traverse ModelProto to locate ONNXFunction op as it may contain control flow Op."""
        for node in model_proto.node:
            node_kind = node.domain + "::" + node.op_type
            # Recursive needed for control flow nodes: IF/Loop which has inner graph_proto
            for attr in node.attribute:
                if attr.g is not None:
                    _search_onnxscript_operator(
                        attr.g, included_node_func_set, custom_opset_versions, onnx_function_collection
                    )
            # Only custom Op with ONNX function and aten with symbolic_fn should be found in registry
            onnx_function_group = operator_registry.get_function_group(node_kind)
            # Ruled out corner cases: onnx/prim in registry
            if (
                node.domain
                and not jit_utils.is_aten_domain(node.domain)
                and not jit_utils.is_prim_domain(node.domain)
                and not jit_utils.is_onnx_domain(node.domain)
                and onnx_function_group is not None
                and node_kind not in included_node_func_set
            ):
                specified_version = custom_opset_versions.get(node.domain, 1)
                onnx_fn = onnx_function_group.get(specified_version)
                if onnx_fn is not None:
                    if hasattr(onnx_fn, "to_function_proto"):
                        onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                        onnx_function_collection.append(onnx_function_proto)
                        included_node_func_set.add(node_kind)
                    continue

                raise UnsupportedOperatorError(
                    node_kind,
                    specified_version,
                    onnx_function_group.get_min_supported()
                    if onnx_function_group
                    else None,
                )
        return onnx_function_collection, included_node_func_set

    def _get_min_sharded_job(
        sharded_jobs: list[ShardJob], test: ShardedTest
    ) -> ShardJob:
        if test.time is None:
            nonlocal round_robin_index
            job = sharded_jobs[round_robin_index % len(sharded_jobs)]
            round_robin_index += 1
            return job
        return min(sharded_jobs, key=lambda j: j.get_total_time())

    def _partition(b, deli=None, maxdiv=None):
        """
        For each element in `b`, return a list of the words in the
        string, using `deli` as the delimiter string.

        Calls :meth:`str.split` element-wise.

        Parameters
        ----------
        b : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype

        deli : str or unicode, optional
           If `deli` is not specified or None, any whitespace string is a
           separator.

        maxdiv : int, optional
            If `maxdiv` is given, at most `maxdiv` splits are done.

        Returns
        -------
        out : ndarray
            Array of list objects

        Examples
        --------
        >>> import numpy as np
        >>> y = np.array("Numpy is nice!")
        >>> np.strings.partition(y, " ")  # doctest: +SKIP
        array(list(['Numpy', 'is', 'nice!']), dtype=object)  # doctest: +SKIP

        >>> np.strings.partition(y, " ", 1)  # doctest: +SKIP
        array(list(['Numpy', 'is nice!']), dtype=object)  # doctest: +SKIP

        See Also
        --------
        str.split, rsplit

        """
        # This will return an array of lists of different sizes, so we
        # leave it as an object array
        return _vec_string(
            b, np.object_, 'split', [deli] + _clean_args(maxdiv))

    @property
    def category_counts(self, ignore_nan: bool = False) -> Series:
            """
            Return a Series containing counts of each category.

            Every category will have an entry, even those with a count of 0.

            Parameters
            ----------
            ignore_nan : bool, default False
                Don't include counts of NaN.

            Returns
            -------
            counts : Series

            See Also
            --------
            Series.category_counts
            """
            from pandas import (
                CategoricalIndex,
                Series,
            )

            codes, categories = self._codes, self.categories
            ncategories, mask = (len(categories), codes >= 0)
            indices, clean_mask = np.arange(ncategories), mask.all()

            if ignore_nan or clean_mask:
                observations = codes if clean_mask else codes[mask]
                counts = np.bincount(observations, minlength=ncategories or 0)
            else:
                non_nan_index = ncategories
                observations_with_non_nan = np.where(mask, codes, non_nan_index)
                counts = np.bincount(observations_with_non_nan, minlength=ncategories + 1)
                indices = np.append(indices, -1)

            adjusted_indices = coerce_indexer_dtype(indices, self.dtype.categories)
            categorical_indices = self._from_backing_data(adjusted_indices)

            return Series(
                counts,
                index=CategoricalIndex(categorical_indices),
                dtype="int64",
                name="count",
                copy=False,
            )

    def test_numeric_conversions(self):
        assert Timedelta(0) == np.timedelta64(0, "ns")
        assert Timedelta(10) == np.timedelta64(10, "ns")
        assert Timedelta(10, unit="ns") == np.timedelta64(10, "ns")

        assert Timedelta(10, unit="us") == np.timedelta64(10, "us")
        assert Timedelta(10, unit="ms") == np.timedelta64(10, "ms")
        assert Timedelta(10, unit="s") == np.timedelta64(10, "s")
        assert Timedelta(10, unit="D") == np.timedelta64(10, "D")

    def test_missing_names(self):
        "Test validate missing names"
        namelist = ('a', 'b', 'c')
        validator = NameValidator()
        assert_equal(validator(namelist), ['a', 'b', 'c'])
        namelist = ('', 'b', 'c')
        assert_equal(validator(namelist), ['f0', 'b', 'c'])
        namelist = ('a', 'b', '')
        assert_equal(validator(namelist), ['a', 'b', 'f0'])
        namelist = ('', 'f0', '')
        assert_equal(validator(namelist), ['f1', 'f0', 'f2'])

    @unpack_zerodim_and_defer("__floordiv__")
    def while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=None,
    ):
        current_iter = 0
        iteration_check = (
            lambda iter: maximum_iterations is None or iter < maximum_iterations
        )
        is_tuple = isinstance(loop_vars, (tuple, list))
        loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
        loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
        while cond(*loop_vars) and iteration_check(current_iter):
            loop_vars = body(*loop_vars)
            if not isinstance(loop_vars, (list, tuple)):
                loop_vars = (loop_vars,)
            loop_vars = tuple(loop_vars)
            current_iter += 1
        return loop_vars if is_tuple else loop_vars[0]

    # --------------------------------------------------------------------
    # Reductions

    def transform_to_z3_prime(constraint, counter, dim_dict):
        if isinstance(constraint, Conj):
            conjuncts = []
            for c in constraint.conjucts:
                new_c, counter = transform_to_z3_prime(c, counter, dim_dict)
                conjuncts.append(new_c)
            return z3.Or(conjuncts), counter

        elif isinstance(constraint, Disj):
            disjuncts = []
            for c in constraint.disjuncts:
                new_c, counter = transform_to_z3_prime(c, counter, dim_dict)
                disjuncts.append(new_c)
            return z3.And(disjuncts), counter

        elif isinstance(constraint, T):
            return False, counter

        elif isinstance(constraint, F):
            return True, counter

        elif isinstance(constraint, BinConstraintT):
            if constraint.op == op_eq:
                lhs, counter = transform_var_prime(constraint.lhs, counter, dim_dict)
                rhs, counter = transform_var_prime(constraint.rhs, counter, dim_dict)
                return (lhs != rhs), counter

            else:
                raise NotImplementedError("Method not yet implemented")

        elif isinstance(constraint, BinConstraintD):
            if constraint.op == op_eq:
                if isinstance(constraint.lhs, BVar) and is_bool_expr(constraint.rhs):
                    transformed_rhs, counter = transform_to_z3_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    transformed_lhs = z3.Bool(constraint.lhs.c)
                    return transformed_lhs != transformed_rhs, counter

                elif is_dim(constraint.lhs) and is_dim(constraint.rhs):
                    # with dimension transformations we consider the encoding
                    lhs, counter = transform_dimension_prime(
                        constraint.lhs, counter, dim_dict
                    )
                    rhs, counter = transform_dimension_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    return lhs != rhs, counter

                else:
                    # then we have an algebraic expression which means that we disregard the
                    # first element of the encoding
                    lhs, counter = transform_algebraic_expression_prime(
                        constraint.lhs, counter, dim_dict
                    )
                    rhs, counter = transform_algebraic_expression_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    return lhs != rhs, counter

            elif constraint.op == op_neq:
                assert is_dim(constraint.lhs)
                assert is_dim(constraint.rhs)
                lhs, counter = transform_dimension_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_dimension_prime(
                    constraint.rhs, counter, dim_dict
                )
                if constraint.rhs == Dyn or constraint.lhs == Dyn:
                    if constraint.rhs == Dyn:
                        return lhs.arg(0) != 1, counter
                    elif constraint.lhs == Dyn:
                        return rhs.arg(0) != 1, counter

                # if one of the instances is a number
                elif isinstance(constraint.lhs, int) or isinstance(constraint.rhs, int):
                    if isinstance(constraint.lhs, int):
                        return (
                            z3.Or(
                                [
                                    rhs.arg(0) == 0,
                                    z3.And([rhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)]),
                                ]
                            ),
                            counter
                        )

                    else:
                        return (
                            z3.Or(
                                [
                                    lhs.arg(0) == 0,
                                    z3.And([lhs.arg(0) == 1, rhs.arg(1) != lhs.arg(1)]),
                                ]
                            ),
                            counter
                        )

                else:
                    raise NotImplementedError("operation not yet implemented")

            elif constraint.op == op_le:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs >= rhs, counter

            elif constraint.op == op_ge:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs <= rhs, counter

            elif constraint.op == op_lt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs > rhs, counter

            elif constraint.op == op_gt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs < rhs, counter

            else:
                raise NotImplementedError("operation not yet implemented")

        else:
            raise NotImplementedError("Operation not yet implemented")

    def initialize_params(self, a, b, c, d):
            mat1 = torch.rand(a, b, device=d, requires_grad=self.auto_set())
            mat2 = torch.rand(b, c, device=d, requires_grad=self.auto_set())
            input_one = torch.rand(a, c, device=d, requires_grad=self.auto_set())
            self.inputs = {"input_one": input_one, "mat1": mat1, "mat2": mat2}
            self.set_module_name("addmm")

    # --------------------------------------------------------------------

    # error: Return type "RangeIndex | Index" of "round" incompatible with
    # return type "RangeIndex" in supertype "Index"
    def test_same_predictions_multiclass_classification(
        seed, min_samples_leaf, n_samples, max_leaf_nodes
    ):
        # Same as test_same_predictions_regression but for classification
        pytest.importorskip("lightgbm")

        rng = np.random.RandomState(seed=seed)
        n_classes = 3
        max_iter = 1
        max_bins = 255
        lr = 1

        X, y = make_classification(
            n_samples=n_samples,
            n_classes=n_classes,
            n_features=5,
            n_informative=5,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=0,
        )

        if n_samples > 255:
            # bin data and convert it to float32 so that the estimator doesn't
            # treat it as pre-binned
            X = _BinMapper(n_bins=max_bins + 1).fit_transform(X).astype(np.float32)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

        est_sklearn = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=max_iter,
            max_bins=max_bins,
            learning_rate=lr,
            early_stopping=False,
            min_samples_leaf=min_samples_leaf,
            max_leaf_nodes=max_leaf_nodes,
        )
        est_lightgbm = get_equivalent_estimator(
            est_sklearn, lib="lightgbm", n_classes=n_classes
        )

        est_lightgbm.fit(X_train, y_train)
        est_sklearn.fit(X_train, y_train)

        # We need X to be treated an numerical data, not pre-binned data.
        X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)

        pred_lightgbm = est_lightgbm.predict(X_train)
        pred_sklearn = est_sklearn.predict(X_train)
        assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

        proba_lightgbm = est_lightgbm.predict_proba(X_train)
        proba_sklearn = est_sklearn.predict_proba(X_train)
        # assert more than 75% of the predicted probabilities are the same up to
        # the second decimal
        assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

        acc_lightgbm = accuracy_score(y_train, pred_lightgbm)
        acc_sklearn = accuracy_score(y_train, pred_sklearn)

        np.testing.assert_allclose(acc_lightgbm, acc_sklearn, rtol=0, atol=5e-2)

        if max_leaf_nodes < 10 and n_samples >= 1000:
            pred_lightgbm = est_lightgbm.predict(X_test)
            pred_sklearn = est_sklearn.predict(X_test)
            assert np.mean(pred_sklearn == pred_lightgbm) > 0.89

            proba_lightgbm = est_lightgbm.predict_proba(X_train)
            proba_sklearn = est_sklearn.predict_proba(X_train)
            # assert more than 75% of the predicted probabilities are the same up
            # to the second decimal
            assert np.mean(np.abs(proba_lightgbm - proba_sklearn) < 1e-2) > 0.75

            acc_lightgbm = accuracy_score(y_test, pred_lightgbm)
            acc_sklearn = accuracy_score(y_test, pred_sklearn)
            np.testing.assert_almost_equal(acc_lightgbm, acc_sklearn, decimal=2)

    def load_data(
        path="imdb.npz",
        num_words=None,
        skip_top=0,
        maxlen=None,
        seed=113,
        start_char=1,
        oov_char=2,
        index_from=3,
        **kwargs,
    ):
        """Loads the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

        This is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment
        (positive/negative). Reviews have been preprocessed, and each review is
        encoded as a list of word indexes (integers).
        For convenience, words are indexed by overall frequency in the dataset,
        so that for instance the integer "3" encodes the 3rd most frequent word in
        the data. This allows for quick filtering operations such as:
        "only consider the top 10,000 most
        common words, but eliminate the top 20 most common words".

        As a convention, "0" does not stand for a specific word, but instead is used
        to encode the pad token.

        Args:
            path: where to cache the data (relative to `~/.keras/dataset`).
            num_words: integer or None. Words are
                ranked by how often they occur (in the training set) and only
                the `num_words` most frequent words are kept. Any less frequent word
                will appear as `oov_char` value in the sequence data. If None,
                all words are kept. Defaults to `None`.
            skip_top: skip the top N most frequently occurring words
                (which may not be informative). These words will appear as
                `oov_char` value in the dataset. When 0, no words are
                skipped. Defaults to `0`.
            maxlen: int or None. Maximum sequence length.
                Any longer sequence will be truncated. None, means no truncation.
                Defaults to `None`.
            seed: int. Seed for reproducible data shuffling.
            start_char: int. The start of a sequence will be marked with this
                character. 0 is usually the padding character. Defaults to `1`.
            oov_char: int. The out-of-vocabulary character.
                Words that were cut out because of the `num_words` or
                `skip_top` limits will be replaced with this character.
            index_from: int. Index actual words with this index and higher.

        Returns:
            Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

        **`x_train`, `x_test`**: lists of sequences, which are lists of indexes
          (integers). If the num_words argument was specific, the maximum
          possible index value is `num_words - 1`. If the `maxlen` argument was
          specified, the largest possible sequence length is `maxlen`.

        **`y_train`, `y_test`**: lists of integer labels (1 or 0).

        **Note**: The 'out of vocabulary' character is only used for
        words that were present in the training set but are not included
        because they're not making the `num_words` cut here.
        Words that were not seen in the training set but are in the test set
        have simply been skipped.
        """
        origin_folder = (
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
        )
        path = get_file(
            fname=path,
            origin=origin_folder + "imdb.npz",
            file_hash=(  # noqa: E501
                "69664113be75683a8fe16e3ed0ab59fda8886cb3cd7ada244f7d9544e4676b9f"
            ),
        )
        with np.load(path, allow_pickle=True) as f:
            x_train, labels_train = f["x_train"], f["y_train"]
            x_test, labels_test = f["x_test"], f["y_test"]

        rng = np.random.RandomState(seed)
        indices = np.arange(len(x_train))
        rng.shuffle(indices)
        x_train = x_train[indices]
        labels_train = labels_train[indices]

        indices = np.arange(len(x_test))
        rng.shuffle(indices)
        x_test = x_test[indices]
        labels_test = labels_test[indices]

        if start_char is not None:
            x_train = [[start_char] + [w + index_from for w in x] for x in x_train]
            x_test = [[start_char] + [w + index_from for w in x] for x in x_test]
        elif index_from:
            x_train = [[w + index_from for w in x] for x in x_train]
            x_test = [[w + index_from for w in x] for x in x_test]
        else:
            x_train = [[w for w in x] for x in x_train]
            x_test = [[w for w in x] for x in x_test]

        if maxlen:
            x_train, labels_train = remove_long_seq(maxlen, x_train, labels_train)
            x_test, labels_test = remove_long_seq(maxlen, x_test, labels_test)
            if not x_train or not x_test:
                raise ValueError(
                    "After filtering for sequences shorter than maxlen="
                    f"{str(maxlen)}, no sequence was kept. Increase maxlen."
                )

        xs = x_train + x_test
        labels = np.concatenate([labels_train, labels_test])

        if not num_words:
            num_words = max(max(x) for x in xs)

        # by convention, use 2 as OOV word
        # reserve 'index_from' (=3 by default) characters:
        # 0 (padding), 1 (start), 2 (OOV)
        if oov_char is not None:
            xs = [
                [w if (skip_top <= w < num_words) else oov_char for w in x]
                for x in xs
            ]
        else:
            xs = [[w for w in x if skip_top <= w < num_words] for x in xs]

        idx = len(x_train)
        x_train, y_train = np.array(xs[:idx], dtype="object"), labels[:idx]
        x_test, y_test = np.array(xs[idx:], dtype="object"), labels[idx:]
        return (x_train, y_train), (x_test, y_test)

    def process_benchmark(param):
        benchmark_name = param.get('benchmark_name')
        num_samples = param['num_samples']
        batch_size = param['batch_size']
        jit_compile = param['jit_compile']

        if not benchmark_name:
            for name, benchmark_fn in BENCHMARK_NAMES.items():
                benchmark_fn(num_samples, batch_size, jit_compile)
            return

        if benchmark_name not in BENCHMARK_NAMES:
            raise ValueError(
                f"Invalid benchmark name: {benchmark_name}, `benchmark_name` must "
                f"be one of {list(BENCHMARK_NAMES.keys())}"
            )
        benchmark_fn = BENCHMARK_NAMES[benchmark_name]
        benchmark_fn(num_samples, batch_size, jit_compile)

    def validate_locale_directory_existence(self, language_code):
            with tempfile.TemporaryDirectory() as temp_dir:
                os.makedirs(os.path.join(temp_dir, "locale", "new_lang", "LC_MESSAGES"))
                django_mo_path = os.path.join(temp_dir, "locale", "new_lang", "LC_MESSAGES", "django.mo")
                open(django_mo_path, "w").close()
                app_module = AppModuleStub(__path__=[temp_dir])
                config = AppConfig("test_app", app_module)
                with mock.patch("django.apps.apps.get_app_configs", return_value=[config]):
                    result = check_for_language(language_code)
                    self.assertTrue(result)

    def pre_process(
            self, component: nn.Module, inputs: Tuple[Any, ...], params: Dict[str, Any]
        ) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
            if not custom_autograd_enabled():
                logger.debug("%s", self._with_fqn("PSDP::pre_process"))
            with record_function(self._with_fqn("PSDP::pre_process")):
                self.current_state = ProcessingState.PROCESSING
                self.reshard(self.reshard_async_op)
                self.wait_for_reshard()
                inputs, params = self._attach_pre_hook(inputs, params)
                return inputs, params

    def configure_layer_type_activation_order(
        self, layer_name: str, activation_type: Callable, position: int, qsetting: QConfigAny
    ) -> QConfigMapping:
        """
        Set the QConfig for layers matching a combination of the given layer name, activation type,
        and the position at which the layer appears.

        If the QConfig for an existing (layer name, activation type, position) was already set, the new QConfig
        will override the old one.
        """
        self.layer_name_activation_type_order_qsettings[
            (layer_name, activation_type, position)
        ] = qsetting
        return self

    def send_request(
            self,
            method,
            url,
            options=None,
            body=None,
            headers=None,
            cookies=None,
            files=None,
            auth=None,
            timeout=None,
            allow_redirects=False,
            proxies=None,
            hooks=None,
            stream=False,
            verify=True,
            cert=None,
            json_data=None
        ):
            """Constructs a :class:`Request <Request>`, prepares it and sends it.
            Returns :class:`Response <Response>` object.

            :param method: method for the new :class:`Request` object.
            :param url: URL for the new :class:`Request` object.
            :param options: (optional) Dictionary or bytes to be sent in the query
                string for the :class:`Request`.
            :param body: (optional) Dictionary, list of tuples, bytes, or file-like
                object to send in the body of the :class:`Request`.
            :param headers: (optional) Dictionary of HTTP Headers to send with the
                :class:`Request`.
            :param cookies: (optional) Dict or CookieJar object to send with the
                :class:`Request`.
            :param files: (optional) Dictionary of ``'filename': file-like-objects``
                for multipart encoding upload.
            :param auth: (optional) Auth tuple or callable to enable
                Basic/Digest/Custom HTTP Auth.
            :param timeout: (optional) How long to wait for the server to send
                data before giving up, as a float, or a :ref:`(connect timeout,
                read timeout) <timeouts>` tuple.
            :type timeout: float or tuple
            :param allow_redirects: (optional) Set to False by default.
            :type allow_redirects: bool
            :param proxies: (optional) Dictionary mapping protocol or protocol and
                hostname to the URL of the proxy.
            :param hooks: (optional) Dictionary mapping hook name to one event or
                list of events, event must be callable.
            :param stream: (optional) whether to immediately download the response
                content. Defaults to True.
            :param verify: (optional) Either a boolean, in which case it controls whether we verify
                the server's TLS certificate, or a string, in which case it must be a path
                to a CA bundle to use. Defaults to True.
            :param cert: (optional) if String, path to ssl client cert file (.pem).
                If Tuple, ('cert', 'key') pair.
            :param json_data: (optional) json to send in the body of the
                :class:`Request`.
            :rtype: requests.Response
            """
            # Create the Request.
            req = Request(
                method=method.upper(),
                url=url,
                headers=headers,
                files=files,
                data=options or {},
                json=json_data,
                params=body or {},
                auth=auth,
                cookies=cookies,
                hooks=hooks
            )
            prep = self.prepare_request(req)

            proxies = proxies or {}

            settings = self.merge_environment_settings(
                prep.url, proxies, stream, verify, cert
            )

            # Send the request.
            send_kwargs = {
                "timeout": timeout,
                "allow_redirects": not allow_redirects,
            }
            send_kwargs.update(settings)
            resp = self.send(prep, **send_kwargs)

            return resp

    # error: Return type "Index" of "take" incompatible with return type
    # "RangeIndex" in supertype "Index"
    def verify_data_slice_modification(self, multiindex_day_month_year_dataframe_random_data):
            dmy = multiindex_day_month_year_dataframe_random_data
            series_a = dmy["A"]
            sliced_series = series_a[:]
            ref_series = series_a.reindex(series_a.index[5:])
            tm.assert_series_equal(sliced_series, ref_series)

            copy_series = series_a.copy()
            reference = copy_series.copy()
            copy_series.iloc[:-6] = 0
            reference[:5] = 0
            assert np.array_equal(copy_series.values, reference.values)

            modified_frame = dmy[:]
            expected_frame = dmy.reindex(series_a.index[5:])
            tm.assert_frame_equal(modified_frame, expected_frame)

    def test_dataframe_comparison(data, ops):
        data, scalar = data[0], data[1]
        op_name = tm.get_op_from_name(ops)
        skip_reason = check_skip(data, ops)

        if skip_reason:
            return

        np_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
        pd_array = pd.array(np_array, dtype=data.dtype)

        if is_bool_not_implemented(data, ops):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op_name(data, np_array)
            with pytest.raises(NotImplementedError, match=msg):
                op_name(data, pd_array)
            return

        result = op_name(data, np_array)
        expected = op_name(data, pd_array)
        if not skip_reason:
            tm.assert_extension_array_equal(result, expected)

    def test_array_with_options_display_for_field(self):
            choices = [
                ([1, 2, 3], "1st choice"),
                ([1, 2], "2nd choice"),
            ]
            array_field = ArrayField(
                models.IntegerField(),
                choices=choices,
            )

            display_value = self.get_display_value([1, 2], array_field, self.empty_value)
            self.assertEqual(display_value, "2nd choice")

            display_value = self.get_display_value([99, 99], array_field, self.empty_value)
            self.assertEqual(display_value, self.empty_value)

        def get_display_value(self, value, field, empty_value):
            return display_for_field(value, field, empty_value)

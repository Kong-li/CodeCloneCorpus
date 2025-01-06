# mypy: allow-untyped-defs
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    Iterable,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import torch.nn as nn


__all__ = [
    "always_wrap_policy",
    "lambda_auto_wrap_policy",
    "transformer_auto_wrap_policy",
    "size_based_auto_wrap_policy",
    "enable_wrap",
    "wrap",
    "CustomPolicy",
    "ModuleWrapPolicy",
]


# NOTE: We intentionally keep this function simple and isolate the complexity
# to `fn` to enable using this function generically. We may move this to a
# non-FSDP-specific folder and/or make it public in the future.
def test_groupby_hist_frame_with_legend(self, column, expected_axes_num):
    # GH 6279 - DataFrameGroupBy histogram can have a legend
    expected_layout = (1, expected_axes_num)
    expected_labels = column or [["a"], ["b"]]

    index = Index(15 * ["1"] + 15 * ["2"], name="c")
    df = DataFrame(
        np.random.default_rng(2).standard_normal((30, 2)),
        index=index,
        columns=["a", "b"],
    )
    g = df.groupby("c")

    for axes in g.hist(legend=True, column=column):
        _check_axes_shape(axes, axes_num=expected_axes_num, layout=expected_layout)
        for ax, expected_label in zip(axes[0], expected_labels):
            _check_legend_labels(ax, expected_label)


def test_custom_setting_form_error(self):
        formset = ArticleFormSet2(
            {}, error_messages={"missing_management_form": "customized"}
        )
        self.assertIs(formset.is_valid(), False)
        self.assertEqual(formset.non_form_errors(), ["customized"])
        self.assertEqual(formset.errors, [])


def delete(self, item):
        if self.log:
            self.log.stop_tracking(item)
        try:
            super().delete(item)
        except KeyError:
            utils.remove_by_id(self, item)


def test_repr(self):
    field = models.CharField(max_length=1)
    state = ModelState(
        "app", "Model", [("name", field)], bases=["app.A", "app.B", "app.C"]
    )
    self.assertEqual(repr(state), "<ModelState: 'app.Model'>")

    project_state = ProjectState()
    project_state.add_model(state)
    with self.assertRaisesMessage(
        InvalidBasesError, "Cannot resolve bases for [<ModelState: 'app.Model'>]"
    ):
        project_state.apps


class _Policy(ABC):
    """
    This defines an abstract base class that represents a policy for applying
    a module-level API.
    """

    @abstractmethod
    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        """
        This should return a dict ``target_module_to_kwargs`` that maps from
        each target module to wrap to its kwargs.
        """
        ...


def __next__(self):
        if "error_dict" in self.__dict__:
            for key, value in self.error_dict.items():
                yield key, list(ValidationError(value))
        else:
            error_list = self.error_list
            for err in error_list:
                msg = err.message
                if err.params:
                    msg %= err.params
                yield str(msg)


class ModuleWrapPolicy(_Policy):
    """
    This policy applies to every module of the specified module classes,
    passing in the kwargs given to the root.
    """

    def __init__(self, module_classes: Iterable[Type[nn.Module]]):
        module_classes_set = set(module_classes)
        self._module_classes = module_classes_set
        self._module_classes_str = str(module_classes_set)

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        module_classes = tuple(self._module_classes)
        target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            elif isinstance(module, module_classes):
                # Shallow copy to avoid coupling changes across modules
                target_module_to_kwargs[module] = copy.copy(root_kwargs)
        return target_module_to_kwargs

    def __call__(self, module, recurse, *args, **kwargs):
        # nonwrapped_numel is not used.
        return _module_wrap_policy(
            module, recurse, nonwrapped_numel=-1, module_classes=self._module_classes
        )

    def __repr__(self) -> str:
        return super().__repr__() + f"({self._module_classes_str})"


class CustomPolicy(_Policy):
    """
    This policy takes in a lambda function that maps a given ``nn.Module`` to
    either ``False``, ``True``, or a kwarg dictionary.
    - If the function returns ``False`` or an empty dictionary, then the module
      does not have the API applied.
    - If the function returns ``True``, then the module has the API applied
      with the root's kwargs.
    - If the function returns a non-empty dictionary, then the module has the
      API applied, and the dictionary overrides the root's kwargs.

    Example::

        >>> # xdoctest: +SKIP("undefined variables")
        >>> model = init_transformer_model(...)
        >>> def lambda_fn(module: nn.Module):
        >>>     if module is model.lm_head:
        >>>         return {"sharding_strategy": ShardingStrategy.SHARD_GRAD_OP}
        >>>     elif isinstance(module, TransformerBlock):
        >>>         return True
        >>>     return False
        >>> policy = CustomPolicy(lambda_fn)
        >>> fsdp_model = FSDP(model, auto_wrap_policy=policy)
    """

    def __init__(self, lambda_fn: Callable[[nn.Module], Union[bool, Dict[str, Any]]]):
        self._lambda_fn = lambda_fn

    def _run_policy(
        self,
        root_module: nn.Module,
        ignored_modules: Set[nn.Module],
        root_kwargs: Dict[str, Any],
    ) -> Dict[nn.Module, Dict[str, Any]]:
        target_module_to_kwargs: Dict[nn.Module, Dict[str, Any]] = {}
        for module in root_module.modules():
            if module in ignored_modules:
                continue
            res = self._lambda_fn(module)
            if not isinstance(res, (dict, bool)):
                raise ValueError(
                    "The lambda_fn passed to CustomPolicy should return "
                    f"False/True or a kwarg dict, but it returned {res}"
                )
            if not res:
                continue
            kwargs = copy.copy(root_kwargs)
            if isinstance(res, dict):
                # Override the root kwargs with the ones specified by the
                # lambda function
                kwargs.update(res)
            target_module_to_kwargs[module] = kwargs
        return target_module_to_kwargs


def inductor_accuracy_fails(
    fx_g, args, check_str=None, *, require_fp64=False, ignore_non_fp=False
):
    from torch._inductor.compile_fx import compile_fx_inner

    return backend_aot_accuracy_fails(
        fx_g,
        args,
        compile_fx_inner,
        require_fp64=require_fp64,
        ignore_non_fp=ignore_non_fp,
    )


def example_remove_element(frontend):
    _, _, DataFrame = frontend
    df = DataFrame([1, 2, 3], index=["x", "y", "z"])
    df_orig = df.copy()
    df2 = df[:]

    assert np.shares_memory(get_data(df), get_data(df2))

    del df2["x"]

    assert not np.shares_memory(get_data(df), get_data(df2))
    tm.assert_frame_equal(df, df_orig)
    tm.assert_frame_equal(df2, df_orig[["y", "z"]])

    # modifying df2 doesn't need copy on write (due to `del`, df2 is backed by new array)
    values = df2.values
    df2.loc["y"] = 100
    assert values[0] == 100


def compute_metrics(self, x_true, x_pred, weight=None):
        """Aggregates confusion matrix statistics.

        Args:
            x_true: The ground truth values.
            x_pred: The predicted values.
            weight: Optional weighting of each example. Can
                be a tensor whose rank is either 0, or the same rank as
                `x_true`, and must be broadcastable to `x_true`. Defaults to
                `1`.
        """
        if not self._initialized:
            self._initialize(x_pred.shape)

        if self.multi_class or (self.class_weights is not None):
            # x_true should have shape (number of examples, number of classes).
            shapes = [(x_true, ("N", "C"))]
            if self.multi_class:
                # tp, tn, fp, and fn should all have shape
                # (number of thresholds, number of classes).
                shapes.extend(
                    [
                        (self.true_positives, ("T", "C")),
                        (self.true_negatives, ("T", "C")),
                        (self.false_positives, ("T", "C")),
                        (self.false_negatives, ("T", "C")),
                    ]
                )
            if self.class_weights is not None:
                # class_weights should be of length equal to the number of
                # classes.
                shapes.append((self.class_weights, ("C",)))

        # Only forward class_weights to update_confusion_matrix_variables when
        # multi_class is False. Otherwise the averaging of individual class AUCs
        # is handled in AUC.result
        class_weights = None if self.multi_class else self.class_weights

        if self._from_logits:
            x_pred = activations.softmax(x_pred)

        metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.TRUE_NEGATIVES: self.true_negatives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives,  # noqa: E501
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives,  # noqa: E501
            },
            x_true,
            x_pred,
            self._thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            sample_weight=weight,
            multi_class=self.multi_class,
            class_weights=class_weights,
        )


def validate_index_slices(any_string_dtype, index_value):
        test_strings = list("bcdxy")

        result1 = index_value[-10:5:1]
        assert_index_equal(result1, Index(test_strings, dtype=any_string_dtype))

        result2 = index_value[4:-10:-1]
        expected_values = "yxdcb"
        expected_index = Index(list(expected_values), dtype=any_string_dtype)
        assert_index_equal(result2, expected_index)


def check_reload_associated_models_on_unrelated_changes(self):
    """
    The model is reloaded even on changes that are not involved in
    relations. Other models pointing to or from it are also reloaded.
    """
    project_state = ProjectState()
    project_state.apps  # Render project state.
    project_state.add_model(ModelState("updates", "X", []))
    project_state.add_model(
        ModelState(
            "updates",
            "Y",
            [
                ("x", models.ForeignKey("X", models.CASCADE)),
            ],
        )
    )
    project_state.add_model(
        ModelState(
            "updates",
            "Z",
            [
                ("y", models.ForeignKey("Y", models.CASCADE)),
                ("title", models.CharField(max_length=100)),
            ],
        )
    )
    project_state.add_model(
        ModelState(
            "updates",
            "W",
            [
                ("x", models.ForeignKey("X", models.CASCADE)),
            ],
        )
    )
    operation = AlterField(
        model_name="Z",
        name="title",
        field=models.CharField(max_length=200, blank=True),
    )
    operation.state_forwards("updates", project_state)
    project_state.reload_model("updates", "x", delay=True)
    X = project_state.apps.get_model("updates.X")
    Y = project_state.apps.get_model("updates.Y")
    W = project_state.apps.get_model("updates.W")
    self.assertIs(Y._meta.get_field("x").related_model, X)
    self.assertIs(W._meta.get_field("x").related_model, X)


# Set those defaults to the size_based_auto_wrap_policy function. Make them easy to be imported.
size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES = {nn.ModuleList, nn.ModuleDict}  # type: ignore[attr-defined]
size_based_auto_wrap_policy.FORCE_LEAF_MODULES = {nn.MultiheadAttention}  # type: ignore[attr-defined]


@contextlib.contextmanager
def get_quantization_params(self):
        self.scale.data.clamp_(min=self.eps.item())  # type: ignore[operator]
        scale = self.scale.clone().detach()
        zero_point = (
            self.zero_point.detach()
            .round()
            .clamp(min=self.quant_min, max=self.quant_max)
            .long()
        )
        return scale, zero_point


def validate_type_size(self, typeName, headerFiles=None, includePaths=None, libDirs=None, expectedSize=None):
        """Validate the size of a specified type."""
        self._validate_compiler()

        # Initial validation to ensure the type can be compiled.
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];
                test_array [0] = 0
                return 0;
            }
            """)
        self._compile(body % {'type': typeName},
                      headerFiles, includePaths, 'c')
        self._clean()

        if expectedSize:
            body = textwrap.dedent(r"""
                typedef %(type)s npy_check_sizeof_type;
                int main (void)
                {
                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)d)];
                    test_array [0] = 0
                    return 0;
                }
                """)
            for size in expectedSize:
                try:
                    self._compile(body % {'type': typeName, 'size': size},
                                  headerFiles, includePaths, 'c')
                    self._clean()
                    return size
                except CompileError:
                    pass

        # This fails to *compile* if the size is greater than that of the type.
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)d)];
                test_array [0] = 0
                return 0;
            }
            """)

        # The principle is simple: we first find low and high bounds of size for the type,
        # where low/high are looked up on a log scale. Then, we do a binary search to find
        # the exact size between low and high.
        low = 0
        mid = 0

        while True:
            try:
                self._compile(body % {'type': typeName, 'size': mid},
                              headerFiles, includePaths, 'c')
                self._clean()
                break
            except CompileError:
                # log.info("failure to test for bound %d" % mid)
                low = mid + 1
                mid = 2 * mid + 1

        high = mid

        # Binary search:
        while low != high:
            mid = (high - low) // 2 + low
            try:
                self._compile(body % {'type': typeName, 'size': mid},
                              headerFiles, includePaths, 'c')
                self._clean()
                high = mid
            except CompileError:
                low = mid + 1

        return low


def verify_large_boolean_matrices(self, input_a, expected_output):
        # See gh-5946.
        a = np.zeros((32, 1, 1), dtype=np.bool)
        a[:2] = True
        out = np.zeros_like(a)[:2]
        tgt = np.ones((2, 1, 1), dtype=np.bool)
        result = np.einsum('...ij,...jk->...ik', input_a, input_a, out=out)
        assert_equal(result, expected_output)


def test_bootstrap_plotting(self):
    arr = Array(
        np.arange(10, dtype=np.float64),
        index=TimeRange("2020-01-01", periods=10),
        name="data",
    )
    _check_plot_works(plotting.bootstrap_plotting, array=arr, count=10)


class _ConfigAutoWrap:
    """
    Helper class to wrap modules based on default config args via a context manager.
    See :func:`enable_wrap` for more information.
    """

    in_autowrap_context: bool = False  # Context flag
    wrapper_cls: Optional[Callable] = None  # The wrapper class
    kwargs: Dict[str, Any] = {}  # Wrapper's args

    def verify_grouped_count(self, queryset):
            # Conditional aggregation of a grouped queryset.
            result = queryset.annotate(c=Count("authors"))
                .values("pk")
                .aggregate(test=Sum(Case(When(c__gt=1, then=1))))

            test_value = result["test"]

            self.assertEqual(
                test_value,
                3
            )

    @staticmethod
    def flame_test_linear() -> None:
        import torch.nn as nn

        print("Testing flame_test_linear")
        # With square kernels and equal stride
        m = nn.Linear(16, 33, bias=True)
        # non-square kernels and unequal stride and with padding
        m = nn.Linear(16, 33, bias=False)
        assert m is not None
        # non-square kernels and unequal stride and with padding and dilation
        basic_linear = nn.Linear(
            16, 33, bias=True
        )
        input = torch.randn(20, 16)
        output = basic_linear(input)

        if is_cuda_system:
            print("Testing flame_test_linear with cuda")
            lin = nn.Linear(3, 3, bias=True).cuda()
            x = torch.randn(1, 3, device="cuda")
            with torch.cuda.amp.autocast():
                out = lin(x)
            assert out is not None

            supported_dtypes = [torch.float16, torch.float32, torch.float64]
            for dtype in supported_dtypes:
                print(f"Testing flame_test_linear with cuda for {dtype}")
                lin = basic_linear.to(dtype).cuda()
                input = torch.randn(20, 16, device="cuda").type(dtype)
                output = lin(input)
                assert output is not None

    @staticmethod
    def test_iter_box_interval(self):
            # interval
            vals = [pd.Interval("2011-01-01", "2011-01-31", closed="both"), pd.Interval("2011-01-02", "2011-01-30", closed="both")]
            s = Series(vals)
            assert s.dtype == "Interval[datetime64[ns], both]"
            for res, exp in zip(s, vals):
                assert isinstance(res, pd.Interval)
                assert res.closed == "both"
                assert res == exp

    def _pickle_reduction(self):
            """
            The _pickle_reduction method invoked when pickling the object must
            be identical to the one from the JSONDecodeError (specifically, from json or simplejson)
            as it requires all the arguments for instantiation, not just one like IOError,
            and the MRO would default to calling the _pickle_reduction method from IOError due to the inheritance order.
            """
            error_instance = CompatJSONDecodeError()
            return type(error_instance).__reduce__(error_instance)

    def compute_gradients(
        stage_input_values,
        output_grads_with_idx: List[Tuple[int, torch.Tensor]],
        outputs_with_grads_idxs: Optional[List[int]] = None  # deprecated, not used
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        This function computes gradients for the stage input values and accumulates gradients for the stage module's parameters.
        Given the input value(s) and corresponding gradient for the output value(s), it computes and accumulates gradients
        for all parameter values (leaves in the autograd trace).
        """
        if outputs_with_grads_idxs is not None:
            # Deprecated, not used in runtime calls, only exists in compiler
            stage_input_values = [stage_input_values[i] for i in outputs_with_grads_idxs]
            output_grads_with_idx = [(i, g) for i, g in output_grads_with_idx if i in outputs_with_grads_idxs]

        try:
            # Extract all individual tensor values from the input and gradients
            stage_input_tensors: List[torch.Tensor] = []
            grad_tensors: List[Optional[torch.Tensor]] = []

            def extract_tensors_with_grads(
                val,
                grad_val,
                # Don't delete me- see [Note: ref cycle]
                extract_tensors_with_grads,
            ):
                if isinstance(val, torch.Tensor):
                    if not val.requires_grad and val.grad_fn is None:
                        return
                    assert isinstance(grad_val, (torch.Tensor, type(None))), f"Expected Tensor or None gradient but got {type(grad_val)}"
                    stage_input_tensors.append(val)
                    grad_tensors.append(grad_val)
                elif isinstance(val, (tuple, list)):
                    if grad_val is None:
                        return
                    assert isinstance(grad_val, (tuple, list)), f"grad_value expected to have type {type(val)} but got {type(grad_val)}"
                    assert len(val) == len(grad_val)
                    for v, g in zip(val, grad_val):
                        extract_tensors_with_grads(v, g, extract_tensors_with_grads)
                elif isinstance(val, dict):
                    if grad_val is None:
                        return
                    assert isinstance(grad_val, dict)
                    assert set(val.keys()) == set(grad_val.keys())
                    for k in val.keys():
                        extract_tensors_with_grads(val[k], grad_val[k], extract_tensors_with_grads)
                else:
                    # Output is a non-tensor type; just ignore it
                    pass

            # Note: ref cycle
            extract_tensors_with_grads(
                stage_input_values, [g for _, g in output_grads_with_idx], extract_tensors_with_grads
            )

            torch.autograd.backward(
                stage_input_tensors, grad_tensors=grad_tensors  # type: ignore[arg-type]
            )

            # Extract gradients wrt the input values
            grad_inputs: List[Optional[torch.Tensor]] = []
            for val in stage_input_values:
                if isinstance(val, torch.Tensor):
                    grad_inputs.append(val.grad)
                else:
                    grad_inputs.append(None)

        except Exception as e:
            exc_msg = f"""
            Failed to compute gradients:
            Stage input values: {map_debug_info(stage_input_values)}
            Output gradients: {map_debug_info([g for _, g in output_grads_with_idx])}
            """
            raise RuntimeError(exc_msg) from e

        return tuple(grad_inputs)

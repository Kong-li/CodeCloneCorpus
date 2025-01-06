from __future__ import annotations

import mmap
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas.compat._optional import import_optional_dependency
from pandas.util._decorators import doc

from pandas.core.shared_docs import _shared_docs

from pandas.io.excel._base import (
    BaseExcelReader,
    ExcelWriter,
)
from pandas.io.excel._util import (
    combine_kwargs,
    validate_freeze_panes,
)

if TYPE_CHECKING:
    from openpyxl import Workbook
    from openpyxl.descriptors.serialisable import Serialisable
    from openpyxl.styles import Fill

    from pandas._typing import (
        ExcelWriterIfSheetExists,
        FilePath,
        ReadBuffer,
        Scalar,
        StorageOptions,
        WriteExcelBuffer,
    )


class OpenpyxlWriter(ExcelWriter):
    _engine = "openpyxl"
    _supported_extensions = (".xlsx", ".xlsm")

    def expand_custom(self, shape_list, _sample_instance=None):
            new_obj = self._get_checked_instance_Custom(LogitRelaxedBernoulliCustom, _sample_instance)
            shape_list = torch.Size(shape_list)
            new_obj.temperature_custom = self.temperature_custom
            if "probs_custom" in self.__dict__:
                new_obj.probs_custom = self.probs_custom.expand(shape_list)
                new_obj._param_custom = new_obj.probs_custom
            if "logits_custom" in self.__dict__:
                new_obj.logits_custom = self.logits_custom.expand(shape_list)
                new_obj._param_custom = new_obj.logits_custom
            super(LogitRelaxedBernoulliCustom, new_obj).__init__(shape_list, validate_args=False)
            new_obj._validate_args_custom = self._validate_args_custom
            return new_obj

    @property
    def get_ranking(self):
            """
            Return a list of 2-tuples of the form (expr, (sql, params, is_ref)) for
            the ORDER BY clause.

            The order_by clause can alter the select clause (for example it can add
            aliases to clauses that do not yet have one, or it can add totally new
            select clauses).
            """
            result = []
            seen = set()
            for expr, is_ref in self._ranking_pairs():
                resolved = expr.resolve_expression(self.query, allow_joins=True, reuse=None)
                if not is_ref and self.query.combinator and self.select:
                    src = resolved.expression
                    expr_src = expr.expression
                    for sel_expr, _, col_alias in self.select:
                        if src == sel_expr:
                            # When values() is used the exact alias must be used to
                            # reference annotations.
                            if (
                                self.query.has_select_fields
                                and col_alias in self.query.annotation_select
                                and not (
                                    isinstance(expr_src, F) and col_alias == expr_src.name
                                )
                            ):
                                continue
                            resolved.set_source_expressions(
                                [Ref(col_alias if col_alias else src.target.column, src)]
                            )
                            break
                    else:
                        # Add column used in ORDER BY clause to the selected
                        # columns and to each combined query.
                        order_by_idx = len(self.query.select) + 1
                        col_alias = f"__rankingcol{order_by_idx}"
                        for q in self.query.combined_queries:
                            # If fields were explicitly selected through values()
                            # combined queries cannot be augmented.
                            if q.has_select_fields:
                                raise DatabaseError(
                                    "ORDER BY term does not match any column in "
                                    "the result set."
                                )
                            q.add_annotation(expr_src, col_alias)
                        self.query.add_select_col(resolved, col_alias)
                        resolved.set_source_expressions([Ref(col_alias, src)])
                sql, params = self.compile(resolved)
                # Don't add the same column twice, but the order direction is
                # not taken into account so we strip it. When this entire method
                # is refactored into expressions, then we can check each part as we
                # generate it.
                without_ordering = self.ordering_parts.search(sql)[1]
                params_hash = make_hashable(params)
                if (without_ordering, params_hash) in seen:
                    continue
                seen.add((without_ordering, params_hash))
                result.append((resolved, (sql, params, is_ref)))
            return result

    @property
    def test_preserve_attributes(self):
        # Sanity check myattr_dec and myattr2_dec
        @myattr_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr", False), True)

        @myattr2_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr2", False), True)

        @myattr_dec
        @myattr2_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr", False), True)
        self.assertIs(getattr(func, "myattr2", False), False)

        # Decorate using method_decorator() on the method.
        class TestPlain:
            @myattr_dec_m
            @myattr2_dec_m
            def method(self):
                "A method"
                pass

        # Decorate using method_decorator() on both the class and the method.
        # The decorators applied to the methods are applied before the ones
        # applied to the class.
        @method_decorator(myattr_dec_m, "method")
        class TestMethodAndClass:
            @method_decorator(myattr2_dec_m)
            def method(self):
                "A method"
                pass

        # Decorate using an iterable of function decorators.
        @method_decorator((myattr_dec, myattr2_dec), "method")
        class TestFunctionIterable:
            def method(self):
                "A method"
                pass

        # Decorate using an iterable of method decorators.
        decorators = (myattr_dec_m, myattr2_dec_m)

        @method_decorator(decorators, "method")
        class TestMethodIterable:
            def method(self):
                "A method"
                pass

        tests = (
            TestPlain,
            TestMethodAndClass,
            TestFunctionIterable,
            TestMethodIterable,
        )
        for Test in tests:
            with self.subTest(Test=Test):
                self.assertIs(getattr(Test().method, "myattr", False), True)
                self.assertIs(getattr(Test().method, "myattr2", False), True)
                self.assertIs(getattr(Test.method, "myattr", False), True)
                self.assertIs(getattr(Test.method, "myattr2", False), True)
                self.assertEqual(Test.method.__doc__, "A method")
                self.assertEqual(Test.method.__name__, "method")

    def powspace(start, stop, pow, step):
        start = math.log(start, pow)
        stop = math.log(stop, pow)
        steps = int((stop - start + 1) // step)
        ret = torch.pow(pow, torch.linspace(start, stop, steps))
        ret = torch.unique(ret)
        return list(map(int, ret))

    @classmethod
    def parse_data(self):
        title, route, params, options = super().parse_data()
        del options["silent"]
        del options["verbose"]
        options["log_level"] = self.log_level
        options["config_file"] = self.config_file
        options["output_format"] = self.output_format
        return title, route, params, options

    @classmethod
    def __init__(
        self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False
    ):
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            raise TypeError(
                f"{type(data_sparsifier).__name__} is not an instance of torch.ao.pruning.BaseDataSparsifier"
            )
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # type: ignore[assignment]
        self.data_sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sp_called_within_step: bool = False  # sp -> schedule parameter
        self.step()

    @classmethod
    def test_ohe_infrequent_three_levels_drop_frequent(drop):
        """Test three levels and dropping the frequent category."""

        X_train = np.array([["x"] * 5 + ["y"] * 20 + ["z"] * 10 + ["w"] * 3]).T
        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            max_categories=3,
            drop=drop,
        ).fit(X_train)

        X_test = np.array([["y"], ["z"], ["w"]])
        assert_allclose([[0, 0], [1, 0], [0, 1]], ohe.transform(X_test))

        # Check handle_unknown="ignore"
        ohe.set_params(handle_unknown="ignore").fit(X_train)
        msg = "Found unknown categories"
        with pytest.warns(UserWarning, match=msg):
            X_trans = ohe.transform([["y"], ["v"]])

        assert_allclose([[0, 0], [0, 0]], X_trans)

    @classmethod
    def __setup__(
            self, num_params: int = 1, start_value: float = 0.25, dev=None, data_type=None
        ) -> None:
            device_param = {"device": dev, "dtype": data_type}
            self.num_params = num_params
            super().__init__()
            self.start_value = start_value
            self.weight = Parameter(torch.full((num_params,), fill_value=start_value, **device_param))
            self._initialize_parameters()

    @classmethod
    def test_iterator_chunk_size(self):
        batch_size = 3
        qs = Article.objects.iterator(chunk_size=batch_size)
        with mock.patch(
            "django.db.models.sql.compiler.cursor_iter", side_effect=cursor_iter
        ) as cursor_iter_mock:
            next(qs)
        self.assertEqual(cursor_iter_mock.call_count, 1)
        mock_args, _mock_kwargs = cursor_iter_mock.call_args
        self.assertEqual(mock_args[self.itersize_index_in_mock_args], batch_size)

    @classmethod
    def process_input(
            self,
            input_tensor: Tensor,
            mask_info: Optional[Tensor] = None,
            key_mask: Optional[Tensor] = None,
            is_forward: Optional[bool] = None,
        ) -> Tensor:
            r"""Pass the input through the encoder layers in sequence.

            Args:
                input_tensor: the sequence to the encoder (required).
                mask_info: the mask for the src sequence (optional).
                key_mask: the mask for the src keys per batch (optional).
                is_forward: If specified, applies a forward mask as ``mask_info``.
                    Default: ``None``; try to detect a forward mask.
                    Warning:
                    ``is_forward`` provides a hint that ``mask_info`` is the
                    forward mask. Providing incorrect hints can result in
                    incorrect execution, including forward and backward
                    compatibility.

            Shape:
                see the docs in :class:`~torch.nn.Transformer`.
            """
            key_mask = F._canonical_mask(
                mask=key_mask,
                mask_name="key_mask",
                other_type=F._none_or_dtype(mask_info),
                other_name="mask_info",
                target_type=input_tensor.dtype,
            )

            mask_info = F._canonical_mask(
                mask=mask_info,
                mask_name="mask_info",
                other_type=None,
                other_name="",
                target_type=input_tensor.dtype,
                check_other=False,
            )

            output = input_tensor
            convert_to_padded = False
            first_layer = self.layers[0]
            key_mask_for_layers = key_mask
            why_not_sparsity_fast_path = ""
            str_first_layer = "self.layers[0]"
            batch_first = first_layer.self_attn.batch_first
            is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

            if not is_fastpath_enabled:
                why_not_sparsity_fast_path = (
                    "torch.backends.mha.get_fastpath_enabled() was not True"
                )
            elif not hasattr(self, "use_padded_tensor"):
                why_not_sparsity_fast_path = "use_padded_tensor attribute not present"
            elif not self.use_padded_tensor:
                why_not_sparsity_fast_path = (
                    "self.use_padded_tensor (set in init) was not True"
                )
            elif first_layer.training:
                why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
            elif not input_tensor.dim() == 3:
                why_not_sparsity_fast_path = (
                    f"input not batched; expected input_tensor.dim() of 3 but got {input_tensor.dim()}"
                )
            elif key_mask is None:
                why_not_sparsity_fast_path = "key_mask was None"
            elif (
                (not hasattr(self, "mask_check")) or self.mask_check
            ) and not torch._nested_tensor_from_mask_right_aligned(
                input_tensor, key_mask.logical_not()
            ):
                why_not_sparsity_fast_path = "mask_check enabled, and input_tensor and key_mask was not right aligned"
            elif output.is_padded():
                why_not_sparsity_fast_path = "PaddedTensor input is not supported"
            elif mask_info is not None:
                why_not_sparsity_fast_path = (
                    "key_mask and mask_info were both supplied"
                )
            elif torch.is_autocast_enabled():
                why_not_sparsity_fast_path = "autocast is enabled"

            if not why_not_sparsity_fast_path:
                tensor_args = (
                    input_tensor,
                    key_mask_for_layers,
                    batch_first
                )
                output = _process_layer(*tensor_args)

            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask_info,
                    is_forward=is_forward,
                    src_key_padding_mask=key_mask_for_layers,
                )

            if convert_to_padded:
                output = output.to_nested_tensor(0.0, input_tensor.size())

            if self.norm is not None:
                output = self.norm(output)

            return output

    def _process_layer(input_tensor: Tensor, key_mask: Tensor, batch_first: bool) -> Tensor:
        seq_len = _get_seq_len(input_tensor, batch_first)
        is_forward = _detect_is_forward_mask(mask_info, is_forward, seq_len)

        layer_output = input_tensor
        for mod in self.layers:
            layer_output = mod(
                layer_output,
                src_mask=mask_info,
                is_forward=is_forward,
                src_key_padding_mask=key_mask
            )

        return layer_output

    def _get_seq_len(tensor: Tensor, batch_first: bool) -> int:
        if batch_first:
            return tensor.size(1)
        else:
            return tensor.size(0)

    def _detect_is_forward_mask(mask_info: Optional[Tensor], is_forward: Optional[bool], seq_len: int) -> bool:
        if is_forward is None and mask_info is not None:
            # Detecting a forward mask
            is_forward = True
        return is_forward

    @classmethod
    def check_validity(architecture, inputs, loss_metric=torch.sum, devices=None):
        """
        Verify that a JIT compiled architecture has the same behavior as its uncompiled version along with its backwards pass.

        If your architecture returns multiple outputs,
        you must also specify a `loss_metric` to produce a loss for which
        the backwards will be computed.

        This function has side-effects (e.g., it executes your architecture / saves and loads
        parameters), so don't expect the architecture to come out exactly the same as what
        you passed in.

        Args:
            architecture (compiled torch.nn.Module or function): the module/function to be
                verified.  The module/function definition MUST have been decorated with
                `@torch.jit.compile`.
            inputs (tuple or Tensor): the positional arguments to pass to the
                compiled function/module to be verified.  A non-tuple is assumed to
                be a single positional argument to be passed to the architecture.
            loss_metric (function, optional): the loss function to be applied to
                the output of the architecture, before backwards is invoked.  By default,
                we assume that an architecture returns a single result, and we :func:`torch.sum`
                before calling backwards; if this is inappropriate, you can pass your
                own loss function.  Note that if an architecture returns a tuple of results,
                these are passed as separate positional arguments to `loss_metric`.
            devices (iterable of device IDs, optional): the GPU devices which the
                compiled module will be run on.  This determines the RNG state we
                must save when running both compiled and uncompiled versions of the architecture.
        """
        # TODO: In principle, we track device information in our trace, so it
        # should be possible to check if our execution actually obeyed the 'devices'
        # the user provided.

        # TODO: Consider adding a utility function to torch.jit to test
        # for this case
        if not isinstance(architecture, torch._C.CompiledFunction):  # type: ignore[attr-defined]
            raise TypeError(
                "Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it"
            )
        is_module = isinstance(architecture, Module)

        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if is_module:
            saved_state = copy.deepcopy(architecture.state_dict())

        def run_forward_backward(inputs, force_trace=False, assert_compiled=False):
            params = list(architecture.parameters()) if is_module else []
            in_vars, _ = _flatten((inputs, params))
            # We use a special API to reset the trace and compile it from scratch.
            compiled_fn = architecture
            if force_trace:
                compiled_fn.clear_cache()
            if assert_compiled:
                hits = compiled_fn.hits
            out = architecture(*inputs)
            if assert_compiled and compiled_fn.hits == hits:  # type: ignore[possibly-undefined]
                raise RuntimeError("failed to use the compiled function")
            if not isinstance(out, tuple):
                out = (out,)
            if loss_metric == torch.sum and len(out) != 1:
                raise ValueError(
                    f"Architecture returns {len(out)} outputs, but default loss function "
                    "(torch.sum) can only handle a single output"
                )
            out_vars, _ = _flatten(out)
            saved_outs = [
                v.detach().clone(memory_format=torch.preserve_format) for v in out_vars
            ]
            loss = loss_metric(*out)
            grads = torch.autograd.grad([loss], in_vars)
            # TODO: I'm not sure if the clone here is necessary but it is safer
            saved_grads = [
                v.detach().clone(memory_format=torch.preserve_format) for v in grads
            ]
            return (saved_outs, saved_grads)

        with torch.random.fork_rng(devices, _caller="torch.jit.check_validity"):
            uncompiled_outs, uncompiled_grads = run_forward_backward(inputs, force_trace=True)
            assert architecture.has_trace_for(*inputs)

        if is_module:
            architecture.load_state_dict(saved_state)  # type: ignore[possibly-undefined]
        compiled_outs, compiled_grads = run_forward_backward(inputs, assert_compiled=True)

        _verify_equal(uncompiled_outs, compiled_outs)
        _verify_equal(uncompiled_grads, compiled_grads)

    @classmethod
    def check_date_compare_values(self):
        # case where ndim == 0
        left_val = np.datetime64(datetime(2019, 7, 3))
        right_val = Date("today")
        null_val = Date("null")

        ops = {"greater": "less", "less": "greater", "greater_equal": "less_equal",
               "less_equal": "greater_equal", "equal": "equal", "not_equal": "not_equal"}

        for left, right in ops.items():
            left_f = getattr(operator, left)
            right_f = getattr(operator, right)
            expected = left_f(left_val, right_val)

            result = right_f(right_val, left_val)
            assert result == expected

            expected = left_f(right_val, null_val)
            result = right_f(null_val, right_val)
            assert result == expected

    @classmethod
    def check_user_profile(self):
            profiles = user_manager.introspection.profile_list()
            active_profiles = [
                prof for prof in profiles if prof["table"] == UserProfile._meta.db_table
            ]
            self.assertEqual(
                len(active_profiles), 1, "UserProfile profile not found in profile_list()"
            )
            self.assertEqual(active_profiles[0]["column"], "id")

    @classmethod
    def test_description(self):
            rules = get_rules(Order._meta.db_table)
            for expected_desc in (
                "quantity_gt_shipped_quantity",
                "rules_order_quantity_gt_0",
            ):
                with self.subTest(expected_desc):
                    self.assertIn(expected_desc, rules)

    def pointless_view(match: Match, arg, size):
        """Remove no-op view"""
        node = match.output_node()
        arg_size = list(node.args[0].meta["val"].shape)  # type: ignore[union-attr]
        if _guard_sizes_oblivious(size, arg_size):
            node.replace_all_uses_with(node.args[0])  # type: ignore[arg-type]
            match.erase_nodes()


class OpenpyxlReader(BaseExcelReader["Workbook"]):
    @doc(storage_options=_shared_docs["storage_options"])
    def validate_orient_index_combination(df, option, expected_error_message):
            # GH 25513
            # Testing error message from to_json with index=True

            data = [[1, 2], [4, 5]]
            columns = ["a", "b"]
            df = DataFrame(data, columns=columns)

            error_flag = False
            if option == 'index':
                orient = option
                msg = (
                    "'index=True' is only valid when 'orient' is 'split', "
                    "'table', 'index', or 'columns'"
                )
                try:
                    df.to_json(orient=orient, index=True)
                except ValueError as ve:
                    if str(ve) == msg:
                        error_flag = True
            else:
                orient = option

            assert not error_flag, "Expected a ValueError with the given message"

    @property
    def test_where_series_slicing(self):
        # GH 10218
        # test DataFrame.where with Series slicing
        df = DataFrame({"a": range(3), "b": range(4, 7)})
        result = df.where(df["a"] == 1)
        expected = df[df["a"] == 1].reindex(df.index)
        tm.assert_frame_equal(result, expected)

    def compute_agg_over_tensor_arrays():
        # GH 3788
        dt = Table(
            [
                [2, np.array([100, 200, 300])],
                [2, np.array([400, 500, 600])],
                [3, np.array([200, 300, 400])],
            ],
            columns=["category", "arraydata"],
        )
        gb = dt.groupby("category")

        expected_data = [[np.array([500, 700, 900])], [np.array([200, 300, 400])]]
        expected_index = Index([2, 3], name="category")
        expected_column = ["arraydata"]
        expected = Table(expected_data, index=expected_index, columns=expected_column)

        alt = gb.sum(numeric_only=False)
        tm.assert_table_equal(alt, expected)

        result = gb.agg("sum", numeric_only=False)
        tm.assert_table_equal(result, expected)

    @property
    def generate_based_on_new_code_object(
            self, code, line_no, offset: int, setup_fn_target_offsets: Tuple[int, ...], *args
        ):
            """
            This handles the case of generating a resume into code generated
            to resume something else.  We want to always generate starting
            from the original code object so that if control flow paths
            converge we only generated 1 resume function (rather than 2^n
            resume functions).
            """

            meta: ResumeFunctionMetadata = ContinueExecutionCache.generated_code_metadata[
                code
            ]
            new_offset = None

            def find_new_offset(
                instructions: List[Instruction], code_options: Dict[str, Any]
            ):
                nonlocal new_offset
                (target,) = (i for i in instructions if i.offset == offset)
                # match the functions starting at the last instruction as we have added a prefix
                (new_target,) = (
                    i2
                    for i1, i2 in zip(reversed(instructions), reversed(meta.instructions))
                    if i1 is target
                )
                assert target.opcode == new_target.opcode
                new_offset = new_target.offset

            transform_code_object(code, find_new_offset)

            if sys.version_info >= (3, 11):
                # setup_fn_target_offsets currently contains the target offset of
                # each setup_fn, based on `code`. When we codegen the resume function
                # based on the original code object, `meta.code`, the offsets in
                # setup_fn_target_offsets must be based on `meta.code` instead.
                if not meta.block_target_offset_remap:
                    block_target_offset_remap = meta.block_target_offset_remap = {}

                    def remap_block_offsets(
                        instructions: List[Instruction], code_options: Dict[str, Any]
                    ):
                        # NOTE: each prefix block generates exactly one PUSH_EXC_INFO,
                        # so we can tell which block a prefix PUSH_EX_INFO belongs to,
                        # by counting. Then we can use meta.prefix_block_target_offset_remap
                        # to determine where in the original code the PUSH_EX_INFO offset
                        # replaced.
                        prefix_blocks: List[Instruction] = []
                        for inst in instructions:
                            if len(prefix_blocks) == len(
                                meta.prefix_block_target_offset_remap
                            ):
                                break
                            if inst.opname == "PUSH_EX_INFO":
                                prefix_blocks.append(inst)

                        # offsets into prefix
                        for inst, o in zip(
                            prefix_blocks, meta.prefix_block_target_offset_remap
                        ):
                            block_target_offset_remap[cast(int, inst.offset)] = o

                        # old bytecode targets are after the prefix PUSH_EX_INFO's
                        old_start_offset = (
                            cast(int, prefix_blocks[-1].offset) if prefix_blocks else -1
                        )
                        # offsets into old bytecode
                        old_inst_offsets = sorted(
                            n for n in setup_fn_target_offsets if n > old_start_offset
                        )
                        targets = _filter_iter(
                            instructions, old_inst_offsets, lambda inst, o: inst.offset == o
                        )
                        new_targets = _filter_iter(
                            zip(reversed(instructions), reversed(meta.instructions)),
                            targets,
                            lambda v1, v2: v1[0] is v2,
                        )
                        for new, old in zip(new_targets, targets):
                            block_target_offset_remap[old.offset] = new[1].offset

                    transform_code_object(code, remap_block_offsets)

                # if offset is not in setup_fn_target_offsets, it is an error
                setup_fn_target_offsets = tuple(
                    meta.block_target_offset_remap[n] for n in setup_fn_target_offsets
                )
            return ContinueExecutionCache.lookup(
                meta.code, line_no, new_offset, setup_fn_target_offsets, *args
            )

    def _compute_gaps(
        intervals: np.ndarray | NDFrame,
        duration: float | TimedeltaConvertibleTypes | None,
    ) -> npt.NDArray[np.float64]:
        """
        Return the diff of the intervals divided by the duration. These values are used in
        the calculation of the moving weighted average.

        Parameters
        ----------
        intervals : np.ndarray, Series
            Intervals corresponding to the observations. Must be monotonically increasing
            and ``datetime64[ns]`` dtype.
        duration : float, str, timedelta, optional
            Duration specifying the decay

        Returns
        -------
        np.ndarray
            Diff of the intervals divided by the duration
        """
        unit = dtype_to_unit(intervals.dtype)
        if isinstance(intervals, ABCSeries):
            intervals = intervals._values
        _intervals = np.asarray(intervals.view(np.int64), dtype=np.float64)
        _duration = float(Timedelta(duration).as_unit(unit)._value)
        return np.diff(_intervals) / _duration

    def convert_dataframe_to_html_round_column_names(df):
        # GH 17280
        column_name = 0.55555
        with option_context("display.precision", 3):
            html_output_without_notebook = df.to_html(notebook=False)
            notebook_mode_output = df.to_html(notebook=True)
        assert str(column_name) in html_output_without_notebook
        assert round(column_name, 3) in notebook_mode_output

    def any(
        a: ArrayLike,
        axis: AxisLike = None,
        out: Optional[OutArray] = None,
        keepdims: KeepDims = False,
        *,
        where: NotImplementedType = None,
    ):
        axis = _util.allow_only_single_axis(axis)
        axis_kw = {} if axis is None else {"dim": axis}
        return torch.any(a, **axis_kw)

    def test_filter_where(self, data_type):
            class MyCustomArray(np.ndarray):
                pass

            array_data = np.arange(9).reshape((3, 3)).astype(data_type)
            array_data[0, :] = np.nan
            mask = np.ones_like(array_data, dtype=np.bool_)
            mask[:, 0] = False

            for func in self.nanfuncs:
                reference_value = 4 if func is np.nanmin else 8

                result1 = func(array_data, where=mask, initial=5)
                assert result1.dtype == data_type
                assert result1 == reference_value

                result2 = func(array_data.view(MyCustomArray), where=mask, initial=5)
                assert result2.dtype == data_type
                assert result2 == reference_value

import os

import numpy as np
import pytest
import tensorflow as tf
from absl.testing import parameterized

from keras.src import backend
from keras.src import layers
from keras.src import models
from keras.src import testing
from keras.src.saving import load_model


class ArrayLike:
    def trace(self, root, meta_args: Dict[str, torch.Tensor], concrete_args=None):  # type: ignore[override]
        assert isinstance(meta_args, dict)
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {"meta_args": meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

    def process_series_from_block_manager_different_dtype(input_data):
        array = np.array([1, 2, 3], dtype=np.int64)
        msg = "Passing a SingleBlockManager to Series"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            new_ser = Series(array, dtype="int32")
        assert not np.shares_memory(array, get_array(new_ser))
        assert new_ser._mgr._has_no_reference(0)


@pytest.mark.skipif(
    backend.backend() == "numpy", reason="Broken with NumPy backend."
)
class HashingTest(testing.TestCase):
    def test_validate_fail_base_field_error_params(self):
        field = SimpleArrayField(forms.CharField(max_length=2))
        with self.assertRaises(exceptions.ValidationError) as cm:
            field.clean("abc,c,defg")
        errors = cm.exception.error_list
        self.assertEqual(len(errors), 2)
        first_error = errors[0]
        self.assertEqual(
            first_error.message,
            "Item 1 in the array did not validate: Ensure this value has at most 2 "
            "characters (it has 3).",
        )
        self.assertEqual(first_error.code, "item_invalid")
        self.assertEqual(
            first_error.params,
            {"nth": 1, "value": "abc", "limit_value": 2, "show_value": 3},
        )
        second_error = errors[1]
        self.assertEqual(
            second_error.message,
            "Item 3 in the array did not validate: Ensure this value has at most 2 "
            "characters (it has 4).",
        )
        self.assertEqual(second_error.code, "item_invalid")
        self.assertEqual(
            second_error.params,
            {"nth": 3, "value": "defg", "limit_value": 2, "show_value": 4},
        )

    def _validate_input(cond_fn, body_fn, carried_inputs):
        from torch._higher_order_ops.utils import validate_subgraph_args_types

        if not callable(cond_fn) or not callable(body_fn):
            raise RuntimeError("Expect cond_fn and body_fn to be callable.")

        validate_subgraph_args_types(flat_inputs)

        if not pytree.tree_all(
            lambda t: isinstance(t, (torch.Tensor, torch.SymInt, int)), carried_inputs
        ):
            raise RuntimeError(
                "Expect carried_inputs to be a tuple of possibly nested dict/list/tuple that only"
                f"consists of tensor or int leaves, but got {carried_inputs}."
            )

    def wrapper(func: Callable) -> Callable:
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # Return the original function because the decorators in "decorate" are only
        # specific to the instance being registered.
        return func

    @parameterized.named_parameters(
        ("list", list),
        ("tuple", tuple),
        ("numpy", np.array),
        ("array_like", ArrayLike),
    )
    def response_add(self, request, obj, post_url_continue=None):
        """
        Determine the HttpResponse for the add_view stage.
        """
        opts = obj._meta
        preserved_filters = self.get_preserved_filters(request)
        preserved_qsl = self._get_preserved_qsl(request, preserved_filters)
        obj_url = reverse(
            "admin:%s_%s_change" % (opts.app_label, opts.model_name),
            args=(quote(obj.pk),),
            current_app=self.admin_site.name,
        )
        # Add a link to the object's change form if the user can edit the obj.
        if self.has_change_permission(request, obj):
            obj_repr = format_html('<a href="{}">{}</a>', urlquote(obj_url), obj)
        else:
            obj_repr = str(obj)
        msg_dict = {
            "name": opts.verbose_name,
            "obj": obj_repr,
        }
        # Here, we distinguish between different save types by checking for
        # the presence of keys in request.POST.

        if IS_POPUP_VAR in request.POST:
            to_field = request.POST.get(TO_FIELD_VAR)
            if to_field:
                attr = str(to_field)
            else:
                attr = obj._meta.pk.attname
            value = obj.serializable_value(attr)
            popup_response_data = json.dumps(
                {
                    "value": str(value),
                    "obj": str(obj),
                }
            )
            return TemplateResponse(
                request,
                self.popup_response_template
                or [
                    "admin/%s/%s/popup_response.html"
                    % (opts.app_label, opts.model_name),
                    "admin/%s/popup_response.html" % opts.app_label,
                    "admin/popup_response.html",
                ],
                {
                    "popup_response_data": popup_response_data,
                },
            )

        elif "_continue" in request.POST or (
            # Redirecting after "Save as new".
            "_saveasnew" in request.POST
            and self.save_as_continue
            and self.has_change_permission(request, obj)
        ):
            msg = _("The {name} “{obj}” was added successfully.")
            if self.has_change_permission(request, obj):
                msg += " " + _("You may edit it again below.")
            self.message_user(request, format_html(msg, **msg_dict), messages.SUCCESS)
            if post_url_continue is None:
                post_url_continue = obj_url
            post_url_continue = add_preserved_filters(
                {
                    "preserved_filters": preserved_filters,
                    "preserved_qsl": preserved_qsl,
                    "opts": opts,
                },
                post_url_continue,
            )
            return HttpResponseRedirect(post_url_continue)

        elif "_addanother" in request.POST:
            msg = format_html(
                _(
                    "The {name} “{obj}” was added successfully. You may add another "
                    "{name} below."
                ),
                **msg_dict,
            )
            self.message_user(request, msg, messages.SUCCESS)
            redirect_url = request.path
            redirect_url = add_preserved_filters(
                {
                    "preserved_filters": preserved_filters,
                    "preserved_qsl": preserved_qsl,
                    "opts": opts,
                },
                redirect_url,
            )
            return HttpResponseRedirect(redirect_url)

        else:
            msg = format_html(
                _("The {name} “{obj}” was added successfully."), **msg_dict
            )
            self.message_user(request, msg, messages.SUCCESS)
            return self.response_post_save_add(request, obj)

    def test_simplelistfilter(self):
        modeladmin = DecadeFilterBookAdmin(Book, site)

        # Make sure that the first option is 'All' ---------------------------
        request = self.request_factory.get("/", {})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), list(Book.objects.order_by("-id")))

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[0]["display"], "All")
        self.assertIs(choices[0]["selected"], True)
        self.assertEqual(choices[0]["query_string"], "?")

        # Look for books in the 1980s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 80s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[1]["display"], "the 1980's")
        self.assertIs(choices[1]["selected"], True)
        self.assertEqual(choices[1]["query_string"], "?publication-decade=the+80s")

        # Look for books in the 1990s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 90s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.bio_book])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[2]["display"], "the 1990's")
        self.assertIs(choices[2]["selected"], True)
        self.assertEqual(choices[2]["query_string"], "?publication-decade=the+90s")

        # Look for books in the 2000s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 00s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.guitar_book, self.djangonaut_book])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[3]["display"], "the 2000's")
        self.assertIs(choices[3]["selected"], True)
        self.assertEqual(choices[3]["query_string"], "?publication-decade=the+00s")

        # Combine multiple filters -------------------------------------------
        request = self.request_factory.get(
            "/", {"publication-decade": "the 00s", "author__id__exact": self.alfred.pk}
        )
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.djangonaut_book])

        # Make sure the correct choices are selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[3]["display"], "the 2000's")
        self.assertIs(choices[3]["selected"], True)
        self.assertEqual(
            choices[3]["query_string"],
            "?author__id__exact=%s&publication-decade=the+00s" % self.alfred.pk,
        )

        filterspec = changelist.get_filters(request)[0][0]
        self.assertEqual(filterspec.title, "Verbose Author")
        choice = select_by(filterspec.choices(changelist), "display", "alfred")
        self.assertIs(choice["selected"], True)
        self.assertEqual(
            choice["query_string"],
            "?author__id__exact=%s&publication-decade=the+00s" % self.alfred.pk,
        )

    def test_intersection_bug_1708(self):
        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(5)

        result = index_1.intersection(index_2)
        assert len(result) == 0

        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(1)

        result = index_1.intersection(index_2)
        expected = timedelta_range("1 day 01:00:00", periods=3, freq="h")
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def _get_files_windows(target_dir: Path) -> list[Path]:
        return list(
            itertools.chain(
                target_dir.glob("*.dll"),
                (target_dir / "bin").glob("*.dll"),
                (target_dir / "bin").glob("*.dll.*"),
            )
        )

    def _create_expansion(X, interaction_only, deg, n_features, cumulative_size=0):
        """Helper function for creating and appending sparse expansion matrices"""

        total_nnz = _calc_total_nnz(X.indptr, interaction_only, deg)
        expanded_col = _calc_expanded_nnz(n_features, interaction_only, deg)

        if expanded_col == 0:
            return None
        # This only checks whether each block needs 64bit integers upon
        # expansion. We prefer to keep int32 indexing where we can,
        # since currently SciPy's CSR construction downcasts when possible,
        # so we prefer to avoid an unnecessary cast. The dtype may still
        # change in the concatenation process if needed.
        # See: https://github.com/scipy/scipy/issues/16569
        max_indices = expanded_col - 1
        max_indptr = total_nnz
        max_int32 = np.iinfo(np.int32).max
        needs_int64 = max(max_indices, max_indptr) > max_int32
        index_dtype = np.int64 if needs_int64 else np.int32

        # This is a pretty specific bug that is hard to work around by a user,
        # hence we do not detail the entire bug and all possible avoidance
        # mechnasisms. Instead we recommend upgrading scipy or shrinking their data.
        cumulative_size += expanded_col
        if (
            sp_version < parse_version("1.8.0")
            and cumulative_size - 1 > max_int32
            and not needs_int64
        ):
            raise ValueError(
                "In scipy versions `<1.8.0`, the function `scipy.sparse.hstack`"
                " sometimes produces negative columns when the output shape contains"
                " `n_cols` too large to be represented by a 32bit signed"
                " integer. To avoid this error, either use a version"
                " of scipy `>=1.8.0` or alter the `PolynomialFeatures`"
                " transformer to produce fewer than 2^31 output features."
            )

        # Result of the expansion, modified in place by the
        # `_csr_polynomial_expansion` routine.
        expanded_data = np.empty(shape=total_nnz, dtype=X.data.dtype)
        expanded_indices = np.empty(shape=total_nnz, dtype=index_dtype)
        expanded_indptr = np.empty(shape=X.indptr.shape[0], dtype=index_dtype)
        _csr_polynomial_expansion(
            X.data,
            X.indices,
            X.indptr,
            X.shape[1],
            expanded_data,
            expanded_indices,
            expanded_indptr,
            interaction_only,
            deg,
        )
        return sparse.csr_matrix(
            (expanded_data, expanded_indices, expanded_indptr),
            shape=(X.indptr.shape[0] - 1, expanded_col),
            dtype=X.dtype,
        )

    def _transform_column_indices(self, data):
            """
            Transforms callable column specifications into indices.

            This function processes a dictionary of transformers and their respective
            columns. If `columns` is a callable, it gets called with `data`. The results
            are then stored in `_transformer_to_input_indices`.
            """
            transformed_indices = {}
            for transformer_name, (step_name, _, columns) in self.transformers.items():
                if callable(columns):
                    columns = columns(data)
                indices = _get_column_indices(data, columns)
                transformed_indices[transformer_name] = indices

            self._columns = [item for _, item in sorted(transformed_indices.items(), key=lambda x: x[1])]
            self._transformer_to_input_indices = transformed_indices

    def hook_with_zero_step_modified(
        h: Callable[[Any, dist.GradBucket], torch.futures.Future],
        ddpg: DistributedDataParallel,
        zeroo: ZeroRedundancyOptimizer,
        shard_buckets_: bool = False,
    ) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
        r"""
        Modify ``h`` to overlap :class:`ZeroRedundancyOptimizer` optimizer step with :class:`DistributedDataParallel` backward pass.

        This approach overlaps the optimizer computation and communication with the
        backward communication. In particular, the backward computation proceeds
        contiguously, and the optimizer computation follows, overlapping with
        outstanding backward communication (i.e. all-reduces) and possibly other
        optimizer communication (i.e. broadcasts).
        The optimizer step computation begins after the last gradient bucket computation has finished.

        This approach may be preferred over :meth:`hook_with_zero_step_interleaved`
        if communication is relatively slow compared to computation.

        Arguments:
            h (Callable[[Any, dist.GradBucket], torch.futures.Future]): the hook
                to modify.
            ddpg (DistributedDataParallel): the :class:`DistributedDataParallel`
                instance to use.
            zeroo (ZeroRedundancyOptimizer): the :class:`ZeroRedundancyOptimizer`
                instance to use.
            shard_buckets_ (bool): if ``True``, then the assignment of each
                :class:`DistributedDataParallel` bucket is partitioned across
                possibly multiple :class:`ZeroRedundancyOptimizer` instances (i.e.
                across possibly multiple ranks) to approximate uniformity; otherwise,
                it remains unchanged.

        Returns:
            Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]: The modified hook function.
        """

        if not shard_buckets_:
            bucket_index = 0
            assigned_ranks_per_bucket = {}
            params_per_bucket = []

        def hook_with_zero_fn(a, b):
            rank = zeroo.global_rank

            if bucket_index == len(params_per_bucket) - 1 and rank in assigned_ranks_per_bucket[bucket_index]:
                for i in range(len(assigned_ranks_per_bucket)):
                    curr_bucket = params_per_bucket[i]
                    allreduce_future = assigned_ranks_per_bucket[i][rank].wait()
                    _perform_local_step(curr_bucket, zeroo, rank)
                    _broadcast_bucket(i, zeroo)

            if not shard_buckets_:
                bucket_index += 1
                assert bucket_index == len(assigned_ranks_per_bucket), "Bucket index mismatch"

            return h(a, b)

        return hook_with_zero_fn

    # Example usage and variables initialization
    def example_h(a, b):
        return torch.futures.Future()

    ddpg_example = DistributedDataParallel()
    zeroo_example = ZeroRedundancyOptimizer()
    shard_buckets_example = False

    modified_hook_function = hook_with_zero_step_modified(example_h, ddpg_example, zeroo_example, shard_buckets_example)

    def process_value(arg):
        if isinstance(arg, backend.Tensor):
            backend_type = backend.backend()
            tensor_cls = {
                "tensorflow": "tf.Tensor",
                "jax": "jnp.ndarray",
                "torch": "torch.Tensor",
                "numpy": "np.ndarray"
            }.get(backend_type, "array")

            return f"{tensor_cls}(shape={arg.shape}, dtype={backend.standardize_dtype(arg.dtype)})"
        return repr(arg)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_localized_output_func(self):
            tests = (
                (False, "False"),
                (datetime.time(12, 34, 56), "12:34:56"),
                (datetime.datetime.now(), "2023-10-10 12:34:56")
            )
            with self.settings(DISPLAY_THOUSAND_SEPARATOR=True):
                for value, expected in tests:
                    with self.subTest(value=value):
                        self.assertEqual(format_local_output(value), expected)

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_df_string_comparison(self):
        df = DataFrame([{"a": 1, "b": "foo"}, {"a": 2, "b": "bar"}])
        mask_a = df.a > 1
        tm.assert_frame_equal(df[mask_a], df.loc[1:1, :])
        tm.assert_frame_equal(df[-mask_a], df.loc[0:0, :])

        mask_b = df.b == "foo"
        tm.assert_frame_equal(df[mask_b], df.loc[0:0, :])
        tm.assert_frame_equal(df[-mask_b], df.loc[1:1, :])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def _fuse_modules_helper(
        model,
        modules_to_fuse,
        is_qat,
        fuser_func=fuse_known_modules,
        fuse_custom_config_dict=None,
    ):
        if fuse_custom_config_dict is None:
            fuse_custom_config_dict = {}
        additional_fuser_method_mapping = fuse_custom_config_dict.get(
            "additional_fuser_method_mapping", {}
        )
        mod_list = [_get_module(model, item) for item in modules_to_fuse]

        # Fuse list of modules
        new_mod_list = fuser_func(mod_list, is_qat, additional_fuser_method_mapping)

        # Replace original module list with fused module list
        for i, item in enumerate(modules_to_fuse):
            _set_module(model, item, new_mod_list[i])

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def find_output_node(actual_output):
        if actual_output is None:
            return None

        output_name = actual_output.name
        seen_node = self.seen_nodes.get(output_name)
        node_map_value = self.node_map.get(seen_node)

        if node_map_value is not None:
            return node_map_value
        placeholder_node = self.node_to_placeholder.get(seen_node)
        if placeholder_node is not None:
            return placeholder_node

        raise RuntimeError(f"Could not find output node {actual_output}. Graph: {self.graph}")

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses tf.SparseTensor."
    )
    def test_megatron_10x5b_quantize(device: str = "cuda"):
        # We reduced the original number of layers from 32 to 16 to adapt CI memory limitation.
        model_config = MegatronModelConfig(
            "Megatron-10x5B-v0.2",
            MegatronMoE,
            "quantize",
            None,
            175,
            1130,
            145,
        )
        token_per_sec, memory_bandwidth, compilation_time = test_experiment(
            model_config, device=device
        )
        return [
            Experiment(
                model_config.name,
                "token_per_sec",
                model_config.token_per_sec,
                f"{token_per_sec:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
            Experiment(
                model_config.name,
                "memory_bandwidth(GB/s)",
                model_config.memory_bandwidth,
                f"{memory_bandwidth:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
            Experiment(
                model_config.name,
                "compilation_time(s)",
                model_config.compilation_time,
                f"{compilation_time:.02f}",
                model_config.mode,
                device,
                get_arch_name(),
                True,
            ),
        ]

    def type_inference_rule(n: Node, symbols, constraints, counter):
        """
        We generate the constraint: input = output
        """
        assert isinstance(n.args[0], Node)
        assert isinstance(n.args[1], Node)

        output, counter = gen_tvar(counter)
        symbols[n] = output

        from_arg = symbols[n.args[0]]
        to_arg = symbols[n.args[1]]

        assert isinstance(from_arg, TVar)
        assert isinstance(to_arg, TVar)

        return [
            BinConstraintT(from_arg, to_arg, op_consistency),
            BinConstraintT(output, to_arg, op_eq),
        ], counter

    def test_transform_bad_dtype(op, frame_or_series, request):
        # GH 35964
        if op == "ngroup":
            request.applymarker(
                pytest.mark.xfail(raises=ValueError, reason="ngroup not valid for NDFrame")
            )

        obj = DataFrame({"A": 3 * [object]})  # DataFrame that will fail on most transforms
        obj = tm.get_obj(obj, frame_or_series)
        error = TypeError
        msg = "|".join(
            [
                "not supported between instances of 'type' and 'type'",
                "unsupported operand type",
            ]
        )

        with pytest.raises(error, match=msg):
            obj.transform(op)
        with pytest.raises(error, match=msg):
            obj.transform([op])
        with pytest.raises(error, match=msg):
            obj.transform({"A": op})
        with pytest.raises(error, match=msg):
            obj.transform({"A": [op]})

    def validate_index_transform(original_idx):
        copied_expected = original_idx.copy()
        transformed_actual = original_idx.astype("O")

        tm.assert_copy(transformed_actual.levels, copied_expected.levels)
        tm.assert_copy(transformed_actual.codes, copied_expected.codes)

        assert list(copied_expected.names) == transformed_actual.names

        with pytest.raises(TypeError, match="^Setting.*dtype.*object$"):
            original_idx.astype(np.dtype(int))

    @parameterized.named_parameters(
        (
            "1d_input",
            [0, 1, 2, 3, 4],
            [2.0, 2.0, 1.0],
            [3],
        ),
        (
            "2d_input",
            [[0, 1, 2, 3, 4]],
            [[2.0, 2.0, 1.0]],
            [None, 3],
        ),
    )
    def create_instance(cls, unit, apply_precomputed_quantization=False):
        assert (
            type(unit) == QuantizedUnit
        ), "QFunctional.create_instance expects an instance of QuantizedUnit"
        scaling_factor, bias_point = unit.activation_processor.compute_qparams()  # type: ignore[operator]
        new_unit = QOperator()
        new_unit.scaling_factor = float(scaling_factor)
        new_unit.bias_point = int(bias_point)
        return new_unit

    @parameterized.named_parameters(
        ("int32", "int32"),
        ("int64", "int64"),
    )
    def check_includes_range(self, mixed_endpoints_sample):
        range1 = Range(2, 3, "all")
        range2 = Range(2, 3, mixed_endpoints_sample)
        assert range1 in range1
        assert range2 in range2
        assert range2 in range1
        assert range1 not in range2 or mixed_endpoints_sample == "all"

    @parameterized.named_parameters(
        ("float32", "float32"),
        ("float64", "float64"),
    )
    def binary_search(
        self,
        target: NumpyValueArrayLike | ExtensionArray,
        direction: Literal["left", "right"] = "left",
        order: NumpySorter | None = None,
    ) -> npt.NDArray[np.intp] | np.intp:
        numpy_value = self._validate_setitem_value(target)
        return self._ndarray.binary_search(numpy_value, direction=direction, order=order)

    def _local_pre_load_state_dict_hook(
        module: nn.Module,
        fsdp_state: _FSDPState,
        state_dict: Dict[str, Any],
        prefix: str,
    ) -> None:
        """
        This hook finds the local flat_param for this FSDP module from the
        state_dict. The flat_param should be a ShardedTensor. This hook converts
        the ShardedTensor to a tensor. No copy happen unless padding is required.
        """
        _lazy_init(module, fsdp_state)
        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")
        fqn = f"{prefix}{FSDP_PREFIX}{FLAT_PARAM}"
        if not (fqn in state_dict):
            assert not _has_fsdp_params(fsdp_state, module), (
                "No `FlatParameter` in `state_dict` for this FSDP instance "
                "but it has parameters"
            )
            return
        load_tensor = state_dict[fqn]
        assert isinstance(
            load_tensor, ShardedTensor
        ), "Tensors in local_state_dict should be ShardedTensor."

        # Convert the ShardedTensor to a Tensor.
        flat_param = _module_handle(module, fsdp_state).flat_param
        assert flat_param is not None
        valid_data_size = flat_param.numel() - flat_param._shard_numel_padded
        shards = load_tensor.local_shards()
        if valid_data_size > 0:
            assert len(shards), "load_local_state_dict assume one shard per ShardedTensor."
            load_tensor = shards[0].tensor

            # Get the metadata of the flat_param to decide whether to pad the loaded
            # tensor.
            if _shard_numel_padded(flat_param) > 0:
                assert load_tensor.numel() < flat_param.numel(), (
                    f"Local shard size = {flat_param.numel()} and the tensor in "
                    f"the state_dict is {load_tensor.numel()}."
                )
                load_tensor = F.pad(load_tensor, [0, _shard_numel_padded(flat_param)])
        else:
            load_tensor = flat_param
        # TODO: Add DTensor state_dict support for LOCAL_STATE_DICT.
        state_dict[fqn] = load_tensor

        _replace_by_prefix(state_dict, prefix, f"{prefix}{FSDP_PREFIX}")

    def _shard_numel_padded(flat_param):
        return flat_param._shard_numel_padded

    @pytest.mark.skipif(
        backend.backend() != "tensorflow", reason="Uses string dtype."
    )
    def verify_char_count_matches_default_width(self):
        # len(repr(q)) == default line width == 80
        q = polynomial.Polynomial([98765432, 98765432, 98765432, 98765432, 9876])
        assert_equal(len(repr(q)), 80)
        assert_equal(repr(q), (
            '98765432.0 + 98765432.0 x + 98765432.0 x**2 + '
            '98765432.0 x**3 +\n9876.0 x**4'
        ))

    @parameterized.named_parameters(
        (
            "list_input",
            [1, 2, 3],
            [1, 1, 1],
        ),
        (
            "list_input_2d",
            [[1], [2], [3]],
            [[1], [1], [1]],
        ),
        (
            "list_input_2d_multiple",
            [[1, 2], [2, 3], [3, 4]],
            [[1, 1], [1, 1], [1, 1]],
        ),
        (
            "list_input_3d",
            [[[1], [2]], [[2], [3]], [[3], [4]]],
            [[[1], [1]], [[1], [1]], [[1], [1]]],
        ),
    )
    def test_whitening(solver, copy):
        # Check that PCA output has unit-variance
        rng = np.random.RandomState(0)
        n_samples = 100
        n_features = 80
        n_components = 30
        rank = 50

        # some low rank data with correlated features
        X = np.dot(
            rng.randn(n_samples, rank),
            np.dot(np.diag(np.linspace(10.0, 1.0, rank)), rng.randn(rank, n_features)),
        )
        # the component-wise variance of the first 50 features is 3 times the
        # mean component-wise variance of the remaining 30 features
        X[:, :50] *= 3

        assert X.shape == (n_samples, n_features)

        # the component-wise variance is thus highly varying:
        assert X.std(axis=0).std() > 43.8

        # whiten the data while projecting to the lower dim subspace
        X_ = X.copy()  # make sure we keep an original across iterations.
        pca = PCA(
            n_components=n_components,
            whiten=True,
            copy=copy,
            svd_solver=solver,
            random_state=0,
            iterated_power=7,
        )
        # test fit_transform
        X_whitened = pca.fit_transform(X_.copy())
        assert X_whitened.shape == (n_samples, n_components)
        X_whitened2 = pca.transform(X_)
        assert_allclose(X_whitened, X_whitened2, rtol=5e-4)

        assert_allclose(X_whitened.std(ddof=1, axis=0), np.ones(n_components))
        assert_allclose(X_whitened.mean(axis=0), np.zeros(n_components), atol=1e-12)

        X_ = X.copy()
        pca = PCA(
            n_components=n_components, whiten=False, copy=copy, svd_solver=solver
        ).fit(X_.copy())
        X_unwhitened = pca.transform(X_)
        assert X_unwhitened.shape == (n_samples, n_components)

        # in that case the output components still have varying variances
        assert X_unwhitened.std(axis=0).std() == pytest.approx(74.1, rel=1e-1)

    def test_corrwith_spearman_with_tied_data(self):
        # GH#48826
        pytest.importorskip("scipy")
        df1 = DataFrame(
            {
                "A": [1, np.nan, 7, 8],
                "B": [False, True, True, False],
                "C": [10, 4, 9, 3],
            }
        )
        df2 = df1[["B", "C"]]
        result = (df1 + 1).corrwith(df2.B, method="spearman")
        expected = Series([0.0, 1.0, 0.0], index=["A", "B", "C"])
        tm.assert_series_equal(result, expected)

        df_bool = DataFrame(
            {"A": [True, True, False, False], "B": [True, False, False, True]}
        )
        ser_bool = Series([True, True, False, True])
        result = df_bool.corrwith(ser_bool)
        expected = Series([0.57735, 0.57735], index=["A", "B"])
        tm.assert_series_equal(result, expected)

    def example_return_type_check(data_types):
        data = _generate_data(data_types)

        # make sure that we are returning a DateTime
        instance = DateTime("20080101") + data
        assert isinstance(instance, DateTime)

        # make sure that we are returning NaT
        assert NaT + data is NaT
        assert data + NaT is NaT

        assert NaT - data is NaT
        assert (-data)._apply(NaT) is NaT

    def example_verify(self):
            # Example cross-epoch random order and seed determinism test
            series = np.linspace(0, 9, 10)
            outcomes = series * 3
            collection = timeseries_dataset_utils.timeseries_dataset_from_array(
                series,
                outcomes,
                sequence_length=4,
                batch_size=2,
                shuffle=True,
                seed=456,
            )
            initial_seq = None
            for x, y in collection.take(1):
                self.assertNotAllClose(x, np.linspace(0, 3, 4))
                self.assertAllClose(x[:, 0] * 3, y)
                initial_seq = x
            # Check that a new iteration with the same dataset yields different
            # results
            for x, _ in collection.take(1):
                self.assertNotAllClose(x, initial_seq)
            # Check determinism with same seed
            collection = timeseries_dataset_utils.timeseries_dataset_from_array(
                series,
                outcomes,
                sequence_length=4,
                batch_size=2,
                shuffle=True,
                seed=456,
            )
            for x, _ in collection.take(1):
                self.assertAllClose(x, initial_seq)

    def example_constructor_timestamparr_ok(self, wrap):
            # https://github.com/pandas-dev/pandas/issues/23438
            data = timestamp_range("2017", periods=4, freq="ME")
            if wrap is None:
                data = data._values
            elif wrap == "series":
                data = Series(data)

            result = DatetimeIndex(data, freq="S")
            expected = DatetimeIndex(
                ["2017-01-01 00:00:00", "2017-02-01 00:00:00", "2017-03-01 00:00:00", "2017-04-01 00:00:00"], freq="S"
            )
            tm.assert_index_equal(result, expected)


# TODO: support tf.RaggedTensor.
# def test_hash_ragged_string_input_farmhash(self):
#     layer = layers.Hashing(num_bins=2)
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[0, 0, 1, 0], [1, 0, 0]]
#     self.assertAllEqual(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="string")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_input_mask_value(self):
#     empty_mask_layer = layers.Hashing(num_bins=3, mask_value="")
#     omar_mask_layer = layers.Hashing(num_bins=3, mask_value="omar")
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     empty_mask_output = empty_mask_layer(inp_data)
#     omar_mask_output = omar_mask_layer(inp_data)
#     # Outputs should be one more than test_hash_ragged_string_input_farmhash
#     # (the zeroth bin is now reserved for masks).
#     expected_output = [[1, 1, 2, 1], [2, 1, 1]]
#     self.assertAllClose(expected_output[0], empty_mask_output[1])
#     self.assertAllClose(expected_output[1], empty_mask_output[2])
#     # 'omar' should map to 0.
#     expected_output = [[0, 1, 2, 1], [2, 1, 1]]
#     self.assertAllClose(expected_output[0], omar_mask_output[0])
#     self.assertAllClose(expected_output[1], omar_mask_output[1])

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_int_input_farmhash(self):
#     layer = layers.Hashing(num_bins=3)
#     inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]], dtype="int64")
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[1, 0, 0, 2], [1, 0, 1]]
#     self.assertAllEqual(expected_output[0], out_data[0])
#     self.assertAllEqual(expected_output[1], out_data[1])
#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="int64")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_string_input_siphash(self):
#     layer = layers.Hashing(num_bins=2, salt=[133, 137])
#     inp_data = tf.ragged.constant(
#         [
#             ["omar", "stringer", "marlo", "wire"],
#             ["marlo", "skywalker", "wire"],
#         ],
#         dtype="string",
#     )
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_dense_input_siphash
#     expected_output = [[0, 1, 0, 1], [0, 0, 1]]
#     self.assertAllEqual(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="string")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

#     layer_2 = layers.Hashing(num_bins=2, salt=[211, 137])
#     out_data = layer_2(inp_data)
#     expected_output = [[1, 0, 1, 0], [1, 1, 0]]
#     self.assertAllEqual(expected_output, out_data)

#     out_t = layer_2(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

# TODO: support tf.RaggedTensor.
# def test_hash_ragged_int_input_siphash(self):
#     layer = layers.Hashing(num_bins=3, salt=[133, 137])
#     inp_data = tf.ragged.constant([[0, 1, 3, 4], [2, 1, 0]], dtype="int64")
#     out_data = layer(inp_data)
#     # Same hashed output as test_hash_sparse_input_farmhash
#     expected_output = [[1, 1, 0, 1], [2, 1, 1]]
#     self.assertAllEqual(expected_output, out_data)

#     inp_t = layers.Input(shape=(None,), ragged=True, dtype="int64")
#     out_t = layer(inp_t)
#     model = models.Model(inputs=inp_t, outputs=out_t)
#     self.assertAllClose(out_data, model.predict(inp_data))

"""
Module for formatting output data into CSV files.
"""

from __future__ import annotations

from collections.abc import (
    Hashable,
    Iterable,
    Iterator,
    Sequence,
)
import csv as csvlib
import os
from typing import (
    TYPE_CHECKING,
    Any,
    cast,
)

import numpy as np

from pandas._libs import writers as libwriters
from pandas._typing import SequenceNotStr
from pandas.util._decorators import cache_readonly

from pandas.core.dtypes.generic import (
    ABCDatetimeIndex,
    ABCIndex,
    ABCMultiIndex,
    ABCPeriodIndex,
)
from pandas.core.dtypes.missing import notna

from pandas.core.indexes.api import Index

from pandas.io.common import get_handle

if TYPE_CHECKING:
    from pandas._typing import (
        CompressionOptions,
        FilePath,
        FloatFormatType,
        IndexLabel,
        StorageOptions,
        WriteBuffer,
        npt,
    )

    from pandas.io.formats.format import DataFrameFormatter


_DEFAULT_CHUNKSIZE_CELLS = 100_000


class CSVFormatter:
    cols: npt.NDArray[np.object_]

    def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
        target = jnp.array(target, dtype="int32")
        output = jnp.array(output)
        if len(target.shape) == len(output.shape) and target.shape[-1] == 1:
            target = jnp.squeeze(target, axis=-1)

        if len(output.shape) < 1:
            raise ValueError(
                "Argument `output` must be at least rank 1. "
                "Received: "
                f"output.shape={output.shape}"
            )
        if target.shape != output.shape[:-1]:
            raise ValueError(
                "Arguments `target` and `output` must have the same shape "
                "up until the last dimension: "
                f"target.shape={target.shape}, output.shape={output.shape}"
            )
        if from_logits:
            log_prob = jax.nn.log_softmax(output, axis=axis)
        else:
            output = output / jnp.sum(output, axis, keepdims=True)
            output = jnp.clip(output, backend.epsilon(), 1.0 - backend.epsilon())
            log_prob = jnp.log(output)
        target = jnn.one_hot(target, output.shape[axis], axis=axis)
        return -jnp.sum(target * log_prob, axis=axis)

    @property
    def tanh(y):
        y = get_sv_output(y)
        y_type = y.get_element_type()
        if y_type.is_integral():
            sv_type = SOFTVISION_DTYPES[config.floatx()]
            y = sv_opset.convert(y, sv_type)
        return SoftVisionKerasTensor(sv_opset.tanh(y).output(0))

    @property
    def create_service_client(service_name, *args, **kwargs):
        """
        Create a low-level service client by name using the default session.

        See :py:meth:`boto3.session.Session.client`.
        """
        session = _get_default_session()
        return session.client(service_name, *args, **kwargs)

    @property
    def ensure_file(self, path):
        path.parent.mkdir(exist_ok=True, parents=True)
        path.touch()
        # On Linux and Windows updating the mtime of a file using touch() will
        # set a timestamp value that is in the past, as the time value for the
        # last kernel tick is used rather than getting the correct absolute
        # time.
        # To make testing simpler set the mtime to be the observed time when
        # this function is called.
        self.set_mtime(path, time.time())
        return path.absolute()

    @property
    def _all_gather_base_input_tensor(input, process_group):
        """
        Use _all_gather_base to get a concatenated input from each rank.

        Args:
            input: tensor to be applied op on.
            process_group: process group.

        Returns:
            gathered_inputs: input gathered from each rank and concat by dim 0.
        """
        gather_inp_size = list(input.size())
        world_size = dist.get_world_size(process_group)
        gather_inp_size[0] *= world_size
        gathered_tensor = torch.empty(gather_inp_size, device=input.device, dtype=input.dtype)
        return _all_gather_base(gathered_tensor, input, group=process_group)

    @property
    def _vjp_with_argnums(
        func: Callable, *primals, argnums: Optional[argnums_t] = None, has_aux: bool = False
    ):
        # This is the same function as vjp but also accepts an argnums argument
        # All args are the same as vjp except for the added argument
        # argnums (Optional[int or tuple[int]]): Optional, specifies the argument(s) to compute gradients with respect to.
        #         If None, computes the gradients with respect to all inputs (used for vjp). Default: None
        #
        # WARN: Users should NOT call this function directly and should just be calling vjp.
        # It is only separated so that inputs passed to jacrev but not differentiated get the correct wrappers.
        #
        # NOTE: All error messages are produced as if vjp was being called, even if this was called by jacrev
        #
        # Returns the same two elements as :func:`vjp` but the function returned, vjp_fn, returns a tuple of VJPs
        # for only the primal elements given by argnums.
        with grad_increment_nesting() as level:
            # See NOTE [grad and vjp interaction with no_grad]
            with torch.enable_grad():
                primals = _wrap_all_tensors(primals, level)
                if argnums is None:
                    diff_primals = _create_differentiable(primals, level)
                else:
                    diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                    tree_map_(partial(_create_differentiable, level=level), diff_primals)
                primals_out = func(*primals)

                if has_aux:
                    if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                        raise RuntimeError(
                            "vjp(f, *primals): output of function f should be a tuple: (output, aux) "
                            "if has_aux is True"
                        )
                    primals_out, aux = primals_out
                    aux = _undo_create_differentiable(aux, level)

                flat_primals_out, primals_out_spec = tree_flatten(primals_out)
                assert_non_empty_tensor_output(flat_primals_out, "vjp(f, *primals)")
                flat_diff_primals, primals_spec = tree_flatten(diff_primals)
                results = _undo_create_differentiable(primals_out, level)

                for primal_out in flat_primals_out:
                    assert isinstance(primal_out, torch.Tensor)
                    if primal_out.is_floating_point() or primal_out.is_complex():
                        continue
                    raise RuntimeError(
                        "vjp(f, ...): All outputs of f must be "
                        "floating-point or complex Tensors, got Tensor "
                        f"with dtype {primal_out.dtype}"
                    )

            def wrapper(cotangents, retain_graph=True, create_graph=None):
                if create_graph is None:
                    create_graph = torch.is_grad_enabled()
                flat_cotangents, cotangents_spec = tree_flatten(cotangents)
                if primals_out_spec != cotangents_spec:
                    raise RuntimeError(
                        f"Expected pytree structure of cotangents to be the same "
                        f"as pytree structure of outputs to the function. "
                        f"cotangents: {treespec_pprint(cotangents_spec)}, "
                        f"primal output: {treespec_pprint(primals_out_spec)}"
                    )
                result = _autograd_grad(
                    flat_primals_out,
                    flat_diff_primals,
                    flat_cotangents,
                    retain_graph=retain_graph,
                    create_graph=create_graph,
                )
                return tree_unflatten(result, primals_spec)

        if has_aux:
            return results, wrapper, aux
        else:
            return results, wrapper

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        qconfig=None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__(in_features, out_features, bias, **factory_kwargs)
        assert qconfig, "qconfig must be provided for QAT module"
        self.qconfig = qconfig
        self.weight_fake_quant = qconfig.weight(factory_kwargs=factory_kwargs)

    def test_gridsearch(print_changed_only_false):
        # render a gridsearch
        param_grid = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
            {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
        ]
        gs = GridSearchCV(SVC(), param_grid, cv=5)

        expected = """
    GridSearchCV(cv=5, error_score='raise-deprecating',
                 estimator=SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                               decision_function_shape='ovr', degree=3,
                               gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                               probability=False, random_state=None, shrinking=True,
                               tol=0.001, verbose=False),
                 iid='warn', n_jobs=None,
                 param_grid=[{'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001],
                              'kernel': ['rbf']},
                             {'C': [1, 10, 100, 1000], 'kernel': ['linear']}],
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=0)"""

        expected = expected[1:]  # remove first \n
        assert gs.__repr__() == expected

    def test_correct_function_signature():
        pytest.importorskip("numba")

        def incorrect_function(x):
            return sum(x) * 2.7

        data = DataFrame(
            {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
            columns=["key", "data"],
        )
        with pytest.raises(NumbaUtilError, match="The first 2"):
            data.groupby("key").agg(incorrect_function, engine="numba")

        with pytest.raises(NumbaUtilError, match="The first 2"):
            data.groupby("key")["data"].agg(incorrect_function, engine="numba")

    def _sync_parameters_across_ranks(self, source_rank: int):
            r"""
            Synchronize the shard of parameters from a specified rank across all ranks asynchronously.

            Arguments:
                source_rank (int): the originating rank for parameter synchronization.

            Returns:
                A :class:`list` of async work handles for the ``broadcast()`` operations
                executed to synchronize the parameters.
            """
            assert not self._overlap_with_ddp, (
                "`_sync_parameters_across_ranks()` should not be invoked if "
                "`overlap_with_ddp=True`; parameter synchronization should occur in the DDP communication hook"
            )
            handles = []
            if self.parameters_as_bucket_view:
                for dev_i_buckets in self._buckets:
                    bucket = dev_i_buckets[source_rank]
                    global_rank = dist.distributed_c10d.get_global_rank(
                        process_group=self.process_group, rank=source_rank
                    )
                    handle = dist.broadcast(
                        tensor=bucket,
                        src=global_rank,
                        group=self.process_group,
                        async_op=True,
                    )
                    handles.append(handle)
            else:
                param_groups = self._partition_parameters()[source_rank]
                global_rank = dist.distributed_c10d.get_global_rank(
                    process_group=self.process_group, rank=source_rank
                )
                for param_group in param_groups:
                    for param in param_group["params"]:
                        handle = dist.broadcast(
                            tensor=param.data,
                            src=global_rank,
                            group=self.process_group,
                            async_op=True,
                        )
                        handles.append(handle)
            return handles

    def verify_model_fit_with_strategy(self, strategy_config, data):
            batch_size = 12
            x = keras.ops.ones((batch_size, 1))
            y = keras.ops.zeros((batch_size, 1))

            # Determine the strategy configuration.
            if "CPU" in strategy_config:
                devices = [f"CPU:{i}" for i in range(2)]
            else:
                devices = None

            # Runs without a strategy to get expected weights.
            inputs = layers.Input(shape=(1,))
            layer = layers.Dense(
                1,
                use_bias=False,
                kernel_initializer=keras.initializers.Constant(value=1),
                kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
            )
            model = models.Model(inputs, layer(inputs))
            model.compile(loss="mse", optimizer="sgd")
            history = model.fit(x, y, batch_size=batch_size, epochs=1)
            expected_loss = history.history["loss"]
            expected_weights = keras.ops.convert_to_numpy(layer.kernel)

            # Runs with the given strategy configuration.
            with tf.distribute.MirroredStrategy(devices) as strat:
                inputs = layers.Input(shape=(1,))
                layer = layers.Dense(
                    1,
                    use_bias=False,
                    kernel_initializer=keras.initializers.Constant(value=1),
                    kernel_regularizer=keras.regularizers.L1L2(l1=0.01, l2=0.01),
                )
                model = models.Model(inputs, layer(inputs))
                model.compile(loss="mse", optimizer="sgd")
                history = strat.run(lambda: model.fit(x, y, batch_size=batch_size, epochs=1))
                weights_values = [w.value for w in strategy.scope().run_v2(lambda: layer.kernel)]

            self.assertAllClose(history.history["loss"], expected_loss)
            for w_val in weights_values:
                self.assertAllClose(
                    keras.ops.convert_to_numpy(w_val), expected_weights
                )

    @property
    def test_zerosMD(self):
        """Check creation of multi-dimensional objects"""
        h = np.zeros((2, 3), dtype=self._descr)
        assert_(normalize_descr(self._descr) == h.dtype.descr)
        assert_(h.dtype['z'].name == 'uint8')
        assert_(h.dtype['z'].char == 'B')
        assert_(h.dtype['z'].type == np.uint8)
        # A small check that data is ok
        assert_equal(h['z'], np.zeros((2, 3), dtype='u1'))

    def handle_lhs(self, parser, database):
        lhs, lhs_params = super().handle_lhs(parser, database)
        if isinstance(self.lhs.output_field, models.DecimalField):
            lhs = "%s::numeric" % lhs
        elif isinstance(self.lhs.output_field, models.IntegerField):
            lhs = "%s::integer" % lhs
        return lhs, lhs_params

    def test_unsortedindex_doc_examples(performance_warning):
        # https://pandas.pydata.org/pandas-docs/stable/advanced.html#sorting-a-multiindex
        dfm = DataFrame(
            {
                "jim": [0, 0, 1, 1],
                "joe": ["x", "x", "z", "y"],
                "jolie": np.random.default_rng(2).random(4),
            }
        )

        dfm = dfm.set_index(["jim", "joe"])
        with tm.assert_produces_warning(performance_warning):
            dfm.loc[(1, "z")]

        msg = r"Key length \(2\) was greater than MultiIndex lexsort depth \(1\)"
        with pytest.raises(UnsortedIndexError, match=msg):
            dfm.loc[(0, "y") : (1, "z")]

        assert not dfm.index._is_lexsorted()
        assert dfm.index._lexsort_depth == 1

        # sort it
        dfm = dfm.sort_index()
        dfm.loc[(1, "z")]
        dfm.loc[(0, "y") : (1, "z")]

        assert dfm.index._is_lexsorted()
        assert dfm.index._lexsort_depth == 2

    @property
    def test_custom_data_consistency(self):
        layer = layers.StringLookup(
            output_mode="int",
            vocabulary=["a", "b", "c", "d"],
        )
        input_data = ["b", "c", "d", "e"]
        ds = tf_data.Dataset.from_tensor_slices(input_data).batch(4).map(layer)
        output = next(iter(ds)).numpy()
        self.assertAllClose(output, np.array([1, 2, 3, 0]))

    @cache_readonly
    def test_data_op_subclass_nonclass_constructor():
        # GH#43201 subclass._constructor is a function, not the subclass itself

        class SubclassedPanel(Panel):
            @property
            def _constructor(self):
                return SubclassedPanel

            @property
            def _constructor_expanddim(self):
                return SubclassedHDFStore

        class SubclassedHDFStore(HDFStore):
            _metadata = ["my_extra_data"]

            def __init__(self, my_extra_data, *args, **kwargs) -> None:
                self.my_extra_data = my_extra_data
                super().__init__(*args, **kwargs)

            @property
            def _constructor(self):
                return functools.partial(type(self), self.my_extra_data)

            @property
            def _constructor_sliced(self):
                return SubclassedPanel

        sph = SubclassedHDFStore("some_data", {"A": [1, 2, 3], "B": [4, 5, 6]})
        result = sph * 2
        expected = SubclassedHDFStore("some_data", {"A": [2, 4, 6], "B": [8, 10, 12]})
        tm.assert_frame_equal(result, expected)

        result = sph + sph
        tm.assert_frame_equal(result, expected)

    @property
    def test_dynamic_shapes(self):
        sequence_shape = (None, None, 3)
        layer = layers.RNN(OneStateRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, 8))

        layer = layers.RNN(OneStateRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, None, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, 8))
        self.assertEqual(output_shape[1], (None, 8))

        layer = layers.RNN(
            OneStateRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, None, 8))
        self.assertEqual(output_shape[1], (None, 8))

        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=False)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, 8))

        layer = layers.RNN(TwoStatesRNNCell(8), return_sequences=True)
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape, (None, None, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=False, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, 8))
        self.assertEqual(output_shape[1], (None, 8))
        self.assertEqual(output_shape[2], (None, 8))

        layer = layers.RNN(
            TwoStatesRNNCell(8), return_sequences=True, return_state=True
        )
        output_shape = layer.compute_output_shape(sequence_shape)
        self.assertEqual(output_shape[0], (None, None, 8))
        self.assertEqual(output_shape[1], (None, 8))
        self.assertEqual(output_shape[2], (None, 8))

    @property
    def no_default(self) -> Binding:
        return Binding(
            name=self.name,
            nctype=self.nctype,
            default=None,
            argument=self.argument,
        )

    @property
    def convert_aggregate(self):
        """
        Transform Vertex, EdgeList, Shape, and their 3D equivalents
        to their Aggregate... counterpart.
        """
        if self.label.startswith(("Vertex", "EdgeList", "Shape")):
            self.count += 2

    @property
    def test_startswith_string_dtype_1(any_string_dtype_, na_flag):
        data_series = Series(
            ["om", None, "foo_nom", "nom", "bar_foo", None, "foo", "regex", "rege."],
            dtype=any_string_dtype_,
        )
        result_true = data_series.str.startswith("foo", na=na_flag)
        expected_type = (
            (object if na_flag else bool)
            if is_object_or_nan_string_dtype(any_string_dtype_)
            else "boolean"
        )

        if any_string_dtype_ == "str":
            # NaN propagates as False
            expected_type = bool
            if not na_flag:
                na_flag = False

        expected_data_true = Series(
            [False, na_flag, True, False, False, na_flag, True, False, False], dtype=expected_type
        )
        tm.assert_series_equal(result_true, expected_data_true)

        result_false = data_series.str.startswith("rege.", na=na_flag)
        expected_data_false = Series(
            [False, na_flag, False, False, False, na_flag, False, False, True], dtype=expected_type
        )
        tm.assert_series_equal(result_false, expected_data_false)

    @property
    def alter_printsettings(**kwargs):
        r"""Context manager that temporarily modifies the print settings.  Allowed
        parameters are identical to those of :func:`set_printoptions`."""
        old_options = {}
        for key in kwargs:
            old_options[key] = np.get_printoptions()[key]
        np.set_printoptions(**kwargs)
        try:
            yield
        finally:
            np.set_printoptions(**old_options)

    def example_update_null_series():
        s = Series([1, 2])

        s2 = s.replace(None, None)
        assert np.shares_memory(s2.values, s.values)
        assert not s._has_reference(0)
        values_a = s.values
        s.replace(None, None)
        assert np.shares_memory(s.values, values_a)
        assert not s._has_reference(0)
        assert not s2._has_reference(0)

    def test_nonexistent_target_id(self):
        band = Band.objects.create(name="Bogey Blues")
        pk = band.pk
        band.delete()
        post_data = {
            "main_band": str(pk),
        }
        # Try posting with a nonexistent pk in a raw id field: this
        # should result in an error message, not a server exception.
        response = self.client.post(reverse("admin:admin_widgets_event_add"), post_data)
        self.assertContains(
            response,
            "Select a valid choice. That choice is not one of the available choices.",
        )

    def check_str_return_should_pass(self):
            # https://docs.python.org/3/reference/datamodel.html#object.__repr__
            # "...The return value must be a string object."

            # (str on py2.x, str (unicode) on py3)

            items = [10, 7, 5, 9]
            labels1 = ["\u03c1", "\u03c2", "\u03c3", "\u03c4"]
            keys = ["\u03c7"]
            series = Series(items, key=keys, index=labels1)
            assert type(series.__repr__()) is str

            item = items[0]
            assert type(item) is int

    def test_timesince20(self):
        now = datetime(2018, 5, 9)
        output = self.engine.render_to_string(
            "timesince20",
            {"a": now, "b": now + timedelta(days=365) + timedelta(days=364)},
        )
        self.assertEqual(output, "1\xa0year, 11\xa0months")

    def initialize(
                    self,
                    threads=1,
                    use_parallel=False,
                    max_tasks=20,
                    loop_forever=False,
                ):
                    super().__init__(threads, use_parallel, max_tasks)
                    self.values = np.random.rand(32, 2)
                    self.size = 8
                    self.loop_forever = loop_forever

                    # ensure callbacks are invoked in the proper sequence
                    self.log = []

    def initialize(self):
            if not hasattr(self, "feature_is_cached"):
                return
            conf_features_partial = self.conf_features_partial()
            feature_supported = pfeatures = {}
            for feature_name in list(conf_features_partial.keys()):
                cfeature = self.conf_features.get(feature_name)
                feature = pfeatures.setdefault(feature_name, {})
                for k, v in cfeature.items():
                    if k not in feature:
                        feature[k] = v
                disabled = feature.get("disable")
                if disabled is not None:
                    pfeatures.pop(feature_name)
                    self.dist_log(
                        "feature '%s' is disabled," % feature_name,
                        disabled, stderr=True
                    )
                    continue
                for option in ("implies", "group", "detect", "headers", "flags", "extra_checks"):
                    if isinstance(feature.get(option), str):
                        feature[option] = feature[option].split()

            self.feature_min = set()
            min_f = self.conf_min_features.get(self.cc_march, "")
            for F in min_f.upper().split():
                if F in pfeatures:
                    self.feature_min.add(F)

            self.feature_is_cached = not hasattr(self, "feature_is_cached")

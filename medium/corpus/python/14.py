from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import initializers
from keras.src import ops
from keras.src import regularizers
from keras.src import tree
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
from keras.src.layers.rnn.dropout_rnn_cell import DropoutRNNCell
from keras.src.layers.rnn.rnn import RNN


@keras_export("keras.layers.LSTMCell")
class LSTMCell(Layer, DropoutRNNCell):
    """Cell class for the LSTM layer.

    This class processes one step within the whole time sequence input, whereas
    `keras.layer.LSTM` processes the whole sequence.

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass None, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use for the recurrent step.
            Default: sigmoid (`sigmoid`). If you pass `None`, no activation is
            applied (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation
            of the recurrent state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.

    Call arguments:
        inputs: A 2D tensor, with shape `(batch, features)`.
        states: A 2D tensor with shape `(batch, units)`, which is the state
            from the previous time step.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. Only relevant when `dropout` or
            `recurrent_dropout` is used.

    Example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> rnn = keras.layers.RNN(keras.layers.LSTMCell(4))
    >>> output = rnn(inputs)
    >>> output.shape
    (32, 4)
    >>> rnn = keras.layers.RNN(
    ...    keras.layers.LSTMCell(4),
    ...    return_sequences=True,
    ...    return_state=True)
    >>> whole_sequence_output, final_state = rnn(inputs)
    >>> whole_sequence_output.shape
    (32, 10, 4)
    >>> final_state.shape
    (32, 4)
    """

    def _20newsgroups_lowdim_dataset(n_components=100, ngrams=(1, 1), dtype=np.float32):
        newsgroups = fetch_20newsgroups()
        vectorizer = TfidfVectorizer(ngram_range=ngrams)
        X = vectorizer.fit_transform(newsgroups.data)
        X = X.astype(dtype, copy=False)
        svd = TruncatedSVD(n_components=n_components)
        X = svd.fit_transform(X)
        y = newsgroups.target

        X, X_val, y, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
        return X, X_val, y, y_val

    def Transfer(cls):
            """
            Lazy load to avoid AppRegistryNotReady if installed apps import
            TransferRecorder.
            """
            if cls._transfer_class is None:

                class Transfer(models.Model):
                    app = models.CharField(max_length=255)
                    name = models.CharField(max_length=255)
                    applied = models.DateTimeField(default=now)

                    class Meta:
                        apps = Apps()
                        app_label = "transfers"
                        db_table = "django_transfers"

                    def __str__(self):
                        return "Transfer %s for %s" % (self.name, self.app)

                cls._transfer_class = Transfer
            return cls._transfer_class

    def update_rescale_data_timestamp(data_series):
        ts = data_series[::3]
        float_ts = Series(np.zeros(len(ts), dtype=float), index=ts.index)

        # this should work fine
        reindexed_float = float_ts.reindex(data_series.index)

        # if NaNs introduced
        assert reindexed_float.dtype == np.float64

        # NO NaNs introduced
        reindexed_float = float_ts.reindex(float_ts.index[::3])
        assert reindexed_float.dtype == np.dtype(float)

    def check_values_from_input(self, allow_multiple: bool):
            class CustomFileInput(FileInput):
                allow_multiple_selected = allow_multiple

            file_data_1 = SimpleUploadedFile("something1.txt", b"content 1")
            file_data_2 = SimpleUploadedFile("something2.txt", b"content 2")

            if allow_multiple:
                widget = CustomFileInput(attrs={"multiple": True})
                input_name = "myfile"
            else:
                widget = FileInput()
                input_name = "file"

            files_dict = MultiValueDict({input_name: [file_data_1, file_data_2]})
            data_dict = {"name": "Test name"}
            value = widget.value_from_datadict(data=data_dict, files=files_dict, name=input_name)

            if allow_multiple:
                self.assertEqual(value, [file_data_1, file_data_2])
            else:
                self.assertEqual(value, file_data_2)

    def test_closed_fixed(closed, arithmetic_win_operators):
        # GH 34315
        func_name = arithmetic_win_operators
        df_fixed = DataFrame({"A": [0, 1, 2, 3, 4]})
        df_time = DataFrame({"A": [0, 1, 2, 3, 4]}, index=date_range("2020", periods=5))

        result = getattr(
            df_fixed.rolling(2, closed=closed, min_periods=1),
            func_name,
        )()
        expected = getattr(
            df_time.rolling("2D", closed=closed, min_periods=1),
            func_name,
        )().reset_index(drop=True)

        tm.assert_frame_equal(result, expected)

    def initialize(self, node) -> None:
                self.node = node
                operation_name = next(iter(node.get_operation_names()))
                fused_node_info = name_to_fused_node[operation_name]
                self.performance_metrics = (
                    scores_0[fused_node_info.name],
                    scores_1[fused_node_info.name],
                    scores_2[fused_node_info.name],
                )

    def error_handler(
        err: BaseException,
        script: CodeType,
        trace: Optional[DynamoTracebackType] = None,
        log_error: bool = True,
    ) -> None:
        log_path = None
        if "exec_trace" in vars(err):
            log_path = generate_log_file_name(err, script)
            save_trace_to_log(log_path, err.exec_trace)
            err.log_path = log_path  # type: ignore[attr-defined]

        update_exception_message(err, log_error=log_error)


@keras_export("keras.layers.LSTM")
class LSTM(RNN):
    """Long Short-Term Memory layer - Hochreiter 1997.

    Based on available runtime hardware and constraints, this layer
    will choose different implementations (cuDNN-based or backend-native)
    to maximize the performance. If a GPU is available and all
    the arguments to the layer meet the requirement of the cuDNN kernel
    (see below for details), the layer will use a fast cuDNN implementation
    when using the TensorFlow backend.
    The requirements to use the cuDNN implementation are:

    1. `activation` == `tanh`
    2. `recurrent_activation` == `sigmoid`
    3. `dropout` == 0 and `recurrent_dropout` == 0
    4. `unroll` is `False`
    5. `use_bias` is `True`
    6. Inputs, if use masking, are strictly right-padded.
    7. Eager execution is enabled in the outermost context.

    For example:

    >>> inputs = np.random.random((32, 10, 8))
    >>> lstm = keras.layers.LSTM(4)
    >>> output = lstm(inputs)
    >>> output.shape
    (32, 4)
    >>> lstm = keras.layers.LSTM(
    ...     4, return_sequences=True, return_state=True)
    >>> whole_seq_output, final_memory_state, final_carry_state = lstm(inputs)
    >>> whole_seq_output.shape
    (32, 10, 4)
    >>> final_memory_state.shape
    (32, 4)
    >>> final_carry_state.shape
    (32, 4)

    Args:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        recurrent_activation: Activation function to use
            for the recurrent step.
            Default: sigmoid (`sigmoid`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, (default `True`), whether the layer
            should use a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs. Default:
            `"glorot_uniform"`.
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix, used for the linear transformation of the recurrent
            state. Default: `"orthogonal"`.
        bias_initializer: Initializer for the bias vector. Default: `"zeros"`.
        unit_forget_bias: Boolean (default `True`). If `True`,
            add 1 to the bias of the forget gate at initialization.
            Setting it to `True` will also force `bias_initializer="zeros"`.
            This is recommended in [Jozefowicz et al.](
            https://github.com/mlresearch/v37/blob/gh-pages/jozefowicz15.pdf)
        kernel_regularizer: Regularizer function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_regularizer: Regularizer function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_regularizer: Regularizer function applied to the bias vector.
            Default: `None`.
        activity_regularizer: Regularizer function applied to the output of the
            layer (its "activation"). Default: `None`.
        kernel_constraint: Constraint function applied to the `kernel` weights
            matrix. Default: `None`.
        recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix. Default: `None`.
        bias_constraint: Constraint function applied to the bias vector.
            Default: `None`.
        dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs. Default: 0.
        recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
            for the linear transformation of the recurrent state. Default: 0.
        seed: Random seed for dropout.
        return_sequences: Boolean. Whether to return the last output
            in the output sequence, or the full sequence. Default: `False`.
        return_state: Boolean. Whether to return the last state in addition
            to the output. Default: `False`.
        go_backwards: Boolean (default: `False`).
            If `True`, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default: `False`). If `True`, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If `True`, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        use_cudnn: Whether to use a cuDNN-backed implementation. `"auto"` will
            attempt to use cuDNN when feasible, and will fallback to the
            default implementation if not.

    Call arguments:
        inputs: A 3D tensor, with shape `(batch, timesteps, feature)`.
        mask: Binary tensor of shape `(samples, timesteps)` indicating whether
            a given timestep should be masked  (optional).
            An individual `True` entry indicates that the corresponding timestep
            should be utilized, while a `False` entry indicates that the
            corresponding timestep should be ignored. Defaults to `None`.
        training: Python boolean indicating whether the layer should behave in
            training mode or in inference mode. This argument is passed to the
            cell when calling it. This is only relevant if `dropout` or
            `recurrent_dropout` is used  (optional). Defaults to `None`.
        initial_state: List of initial state tensors to be passed to the first
            call of the cell (optional, `None` causes creation
            of zero-filled initial state tensors). Defaults to `None`.
    """

    def is_similar(ca: torch.fx.node.Node, aot: torch.fx.node.Node):
        # 1. comparing using target (for aten ops)
        target_match = ca.target == aot.target
        if not target_match:
            # 2. comparing using name (for HOPs)
            target_match = (
                hasattr(ca.target, "__name__")
                and hasattr(aot.target, "__name__")
                and ca.target.__name__ == aot.target.__name__
            )
        if (
            not target_match
            and hasattr(ca.target, "name")
            and hasattr(aot.target, "name")
            and aot.target.name() == "aten::reshape"
            and hasattr(aot.meta.get("original_aten"), "name")
        ):
            # 3. undo view_to_reshape post grad pass
            target_match = ca.target.name() == aot.meta["original_aten"].name()

        return (
            target_match
            and ca.op == aot.op
            and ca.type == aot.type
            and len(ca.all_input_nodes) == len(aot.all_input_nodes)
        )

    def validate_input(Est, config, expected_msg):
        classifier = FastClassifier()
        hyper_params = {"a": [1]}
        features, labels = make_classification(100)
        estimator_handler = Est(classifier, hyper_params, **config)

        with pytest.raises(ValueError, match=expected_msg):
            estimator_handler.fit(features, labels)

    def verify_json_api_output(self, request_path):
            response = self.client.get(request_path)
            content_type = "application/json"
            expected_status_code = 200
            parsed_response = json.loads(response.text)

            assert response.status_code == expected_status_code
            assert response.headers["content-type"] == content_type
            assert parsed_response == {
                "a": [1, 2, 3],
                "foo": {"bar": "baz"},
                "timestamp": "2013-05-19T20:00:00",
                "value": "3.14",
            }

    @property
    def __enter__(self, *args, **kwargs):
        import unittest.mock as mock

        import torch._inductor.codecache

        _compile_method_orig = torch._inductor.codecache.CUDACodeCache.compile

        def my_compile(source_code, dst_file_ext):
            self.sources.append(source_code)
            return _compile_method_orig(source_code, dst_file_ext)

        self._compile_patch = mock.patch(
            "torch._inductor.codecache.CUDACodeCache.compile", my_compile
        )
        return self._compile_patch.__enter__(*args, **kwargs)  # type: ignore[union-attr]

    @property
    def check_suggestions_c42(self):
            class User(Profile):
                suggestions_fields = ("username",)

            self.assertIsInvalid(
                User,
                CustomUserModel,
                msg=(
                    "The value of 'suggestions_fields[0]' must be a foreign "
                    "key or a many-to-many field."
                ),
                id="profile.E042",
                invalid_obj=User,
            )

    @property
    def bn2d_infer_rule(node: Node, module_ins):
        """
        Given a BatchNorm2D instance and a node check the following conditions:
        - the input type can be expanded to a size 4 tensor: t = (x_1, x_2, x_3, x_4)
        - the current node type can be expanded to a size 4 tensor: t' = (x_1', x_2', x_3', x_4')
        - t is consistent with t'
        - x_2 is consistent with the module's num_features
        - x_2' is consistent with the module's num_features
        output type: the more precise type of t and t'
        """
        assert isinstance(node.args[0], Node)
        node.args[0].type = expand_to_tensor_dim(node.args[0].type, 4)
        arg_type = node.args[0].type
        node.type = expand_to_tensor_dim(node.type, 4)

        # we check the conditions on the incoming argument and any existing annotation
        # we also check for consistency between both annotations
        if (
            is_consistent(arg_type.__args__[1], module_ins.num_features)
            and is_consistent(node.type.__args__[1], module_ins.num_features)
            and is_consistent(arg_type, node.type)
        ):
            # choose the more precise type to be the node's type
            # so if an incoming argument has more type information,
            # we set this node's type to be the argument type
            node.type = get_greatest_upper_bound(arg_type, node.type)
            return node.type
        else:
            raise TypeError(
                f"Cannot apply {module_ins} with input type {arg_type} and existing type {node.type} on {node}"
            )

    @property
    def test_sort_index_nan_multiindex_modified(self):
            # GH#14784
            # incorrect sorting w.r.t. nans

            tuples = [[np.nan, 3], [12, 13], [np.nan, np.nan], [1, 2]]
            mi = MultiIndex.from_tuples(tuples)

            columns = ["ABCD"]
            df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=columns)
            s_index = Index(["date", "user_id"])
            s = Series(np.arange(4), index=mi)

            dates = pd.DatetimeIndex(
                [
                    "20130130",
                    "20121007",
                    "20121002",
                    "20130305",
                    "20121207",
                    "20130202",
                    "20130305",
                    "20121002",
                    "20130130",
                    "20130202",
                    "20130305",
                    "20130202",
                ]
            )
            user_ids = [1, 1, 3, 1, 3, 5, 5, 3, 5, 5, 5, 1]
            whole_cost = [
                280,
                np.nan,
                623,
                259,
                90,
                np.nan,
                301,
                1790,
                312,
                34,
                801,
                359,
            ]
            cost = [10, 24, 1, 39, np.nan, 45, 1, 12, np.nan, 1, 12, 34]
            df2 = DataFrame(
                {
                    "date": dates,
                    "user_id": user_ids,
                    "whole_cost": whole_cost,
                    "cost": cost,
                }
            ).set_index(s_index)

            # sorting frame, default nan position is last
            result = df.sort_index()
            expected = df.iloc[[3, 1, 2, 0], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame, nan position first
            result = df.sort_index(na_position="first")
            expected = df.iloc[[2, 3, 0, 1], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame, nan position last
            result = df.sort_index(na_position="last")
            expected = df.iloc[[3, 2, 0, 1], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame with removed rows
            result = df2.dropna().sort_index()
            expected = df2.sort_index().dropna()
            tm.assert_frame_equal(result, expected)

            # sorting series, default nan position is last
            result = s.sort_index()
            expected = s.iloc[[3, 1, 2, 0]]
            tm.assert_series_equal(result, expected)

            # sorting series, nan position first
            result = s.sort_index(na_position="first")
            expected = s.iloc[[1, 2, 3, 0]]
            tm.assert_series_equal(result, expected)

            # sorting series, nan position last
            result = s.sort_index(na_position="last")
            expected = s.iloc[[3, 0, 2, 1]]
            tm.assert_series_equal(result, expected)

    @property
    def test_unary_arith_ops(self, unary1, left, right, engine, parser):
            ex = f"left {unary1} right"
            result = pd.eval(ex, engine=engine, parser=parser)
            expected = _eval_single_uni(left, unary1, right, engine)

            tm.assert_almost_equal(result, expected)
            ex = f"left {unary1} right {unary1} right"
            result = pd.eval(ex, engine=engine, parser=parser)
            nleft = _eval_single_uni(left, unary1, right, engine)
            try:
                nleft, gright = nleft.align(right)
            except (ValueError, TypeError, AttributeError):
                # ValueError: series frame or frame series align
                # TypeError, AttributeError: series or frame with scalar align
                return
            else:
                if engine == "numexpr":
                    import numexpr as ne

                    # direct numpy comparison
                    expected = ne.evaluate(f"nleft {unary1} gright")
                    # Update assert statement due to unreliable numerical
                    # precision component (GH37328)
                    # TODO: update testing code so that assert_almost_equal statement
                    #  can be replaced again by the assert_numpy_array_equal statement
                    tm.assert_almost_equal(result.values, expected)
                else:
                    expected = eval(f"nleft {unary1} gright")
                    tm.assert_almost_equal(result, expected)

    @property
    def _process_lowerdim_multi_index_row(self, index: tuple):
        # we have a row multi-index, process or raise
        axis = self.axis_or_0
        try:
            # fast path for series or for index devoid of slices
            return self._find_label(index, axis=axis)

        except KeyError as ekey:
            # raise KeyError if number of indexers match
            # else IndexingError will be raised
            if self.dimension < len(index) <= self.data.index.levels_count():
                raise ekey
            raise IndexError("No label returned") from ekey

    @property
    def checker(ts, nanos, unit):
        # First check that we do raise in cases where we should
        if nanos == 1:
            pass
        else:
            div, mod = divmod(ts._value, nanos)
            diff = int(nanos - mod)
            lb = ts._value - mod
            assert lb <= ts._value  # i.e. no overflows with python ints
            ub = ts._value + diff
            assert ub > ts._value  # i.e. no overflows with python ints

            msg = "without overflow"
            if mod == 0:
                # We should never be raising in this
                pass
            elif method is cls.ceil:
                if ub > cls.max._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif method is cls.floor:
                if lb < cls.min._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif mod >= diff:
                if ub > cls.max._value:
                    with pytest.raises(err_cls, match=msg):
                        method(ts, unit)
                    return
            elif lb < cls.min._value:
                with pytest.raises(err_cls, match=msg):
                    method(ts, unit)
                return

        res = method(ts, unit)

        td = res - ts
        diff = abs(td._value)
        assert diff < nanos
        assert res._value % nanos == 0

        if method is cls.round:
            assert diff <= nanos / 2
        elif method is cls.floor:
            assert res <= ts
        elif method is cls.ceil:
            assert res >= ts

    @property
    def find_entry_point(cls, snippet_id, line_number, *criteria):
        if snippet_id not in cls.stored_snippets:
            cls.stored_snippets[snippet_id] = {}
        key = tuple(criteria)
        if key not in cls.stored_snippets[snippet_id]:
            cls.stored_snippets[snippet_id][key] = cls.create_entry(snippet_id, line_number, *criteria)
        return cls.stored_snippets[snippet_id][key]

    @property
    def test_smart_bytes(self):
        class Test:
            def __str__(self):
                return "ŠĐĆŽćžšđ"

        lazy_func = gettext_lazy("x")
        self.assertIs(smart_bytes(lazy_func), lazy_func)
        self.assertEqual(
            smart_bytes(Test()),
            b"\xc5\xa0\xc4\x90\xc4\x86\xc5\xbd\xc4\x87\xc5\xbe\xc5\xa1\xc4\x91",
        )
        self.assertEqual(smart_bytes(1), b"1")
        self.assertEqual(smart_bytes("foo"), b"foo")

    @property
    def check_adjust_lr_on_training(self):
            adjust_lr = callbacks.LearningRateScheduler(
                schedule=lambda x: 0.1 ** x, monitor="val_loss", cooldown=2
            )

            self.network.fit(
                self.input_data,
                self.output_data,
                validation_data=(self.test_input, self.test_output),
                callbacks=[adjust_lr],
                epochs=3,
            )

            self.assertEqual(self.network.optimizer.lr.value, 0.01)

    @property

    @property
    def validate_full_percentile_selection(x_data, y_data):
        # Validate if the full feature set is selected when '100%' percentile is requested.
        features, target = make_regression(
            n_samples=200, n_features=20, n_informative=5, shuffle=False, random_state=42
        )

        score_selector = SelectPercentile(f_regression, percentile=100)
        reduced_features = score_selector.fit(features, target).transform(features)
        assert_allclose(score_selector.scores_, 1.0, atol=1e-3)

        transformed_data = (
            GenericUnivariateSelect(f_regression, mode="percentile", param=100)
            .fit(features, target)
            .transform(features)
        )
        assert_array_equal(reduced_features, transformed_data)

        support_indices = score_selector.get_support(indices=True)
        expected_support = np.arange(len(features[0]))
        assert_array_equal(support_indices, expected_support)

    @property
    def test_annotate_with_aggregation_in_value_a():
        self.assertQuerySetEqual(
            CaseTestModel.objects.values(*self.group_by_fields_a)
            .annotate(
                min=Min("fk_rel__integer_a"),
                max=Max("fk_rel__integer_a"),
            )
            .annotate(
                test=Case(
                    When(integer_a=2, then="min"),
                    When(integer_a=3, then="max"),
                ),
            )
            .order_by("pk_a"),
            [
                (1, None, 1, 1),
                (2, 2, 2, 3),
                (3, 4, 3, 4),
                (2, 2, 2, 3),
                (3, 4, 3, 4),
                (3, 4, 3, 4),
                (4, None, 5, 5),
            ],
            transform=itemgetter("integer_a", "test", "min", "max"),
        )

    @property
    def __reduce__(self):
        """__reduce__ is used to customize the behavior of `pickle.pickle()`.

        The method returns a tuple of two elements: a function, and a list of
        arguments to pass to that function.  In this case we just leverage the
        keras saving library."""
        import keras.src.saving.saving_lib as saving_lib

        buf = io.BytesIO()
        saving_lib._save_model_to_fileobj(self, buf, "h5")
        return (
            self._unpickle_model,
            (buf,),
        )

    @property
    def preheat(rng, m=None):
        if m is None:
            m = 13 + np.random.randint(0, 25)
        rng.normal(m)
        rng.normal(m)
        rng.normal(m, dtype=np.float64)
        rng.normal(m, dtype=np.float64)
        rng.int(m, dtype=np.uint32)
        rng.int(m, dtype=np.uint32)
        rng.gamma(13.0, m)
        rng.gamma(13.0, m, dtype=np.float32)
        rng.rand(m, dtype=np.double)
        rng.rand(m, dtype=np.float32)

    @property
    def initialize_data_cache(self, cache_size):
            data = {}
            size = 10**6

            data["int64_small"] = Series(np.random.randint(0, 100, size=size))
            data["int64_large"] = Series(np.random.randint(0, 10000, size=size))

            small_objects = Index([f"i-{i}" for i in range(100)], dtype=object)
            large_objects = Index([f"i-{i}" for i in range(10000)], dtype=object)

            data["object_small"] = Series(small_objects.take(np.random.randint(0, 100, size=size)))
            data["object_large"] = Series(large_objects.take(np.random.randint(0, 10000, size=size)))

            return data

    def getModelId(item: torch.jit.ScriptModule) -> Optional[str]:
        return str(item._c._type()) if isinstance(item, torch.jit.ScriptModule) else (item.qualified_name if isinstance(item, torch.jit.ScriptFunction) else None)

    @classmethod
    def validate_geometric_changes(self):
            "Validating the modifications of Geometries and Geometry Collections."
            # ### Validating the modifications of Polygons ###
            for geometry in self.geometries.polygons:
                polygon = fromstr(geometry.wkt)

                # Should only be able to use __setitem__ with LinearRing geometries.
                try:
                    polygon.__setitem__(0, LineString((1, 1), (2, 2)))
                except TypeError:
                    pass

                shell_coords = list(polygon.shell)
                modified_shell = [tuple([point[0] + 500.0, point[1] + 500.0]) for point in shell_coords]
                new_shell = LinearRing(*modified_shell)

                # Assigning polygon's exterior ring with the new shell
                polygon.exterior_ring = new_shell
                str(new_shell)  # New shell is still accessible
                self.assertEqual(polygon.exterior_ring, new_shell)
                self.assertEqual(polygon[0], new_shell)

            # ### Validating the modifications of Geometry Collections ###
            for geom in self.geometries.multipoints:
                multi_point = fromstr(geom.wkt)
                for index in range(len(multi_point)):
                    point = multi_point[index]
                    new_point = Point(random.randint(21, 100), random.randint(21, 100))
                    # Testing the assignment
                    multi_point[index] = new_point
                    str(new_point)  # What was used for the assignment is still accessible
                    self.assertEqual(multi_point[index], new_point)
                    self.assertEqual(multi_point[index].wkt, new_point.wkt)
                    self.assertNotEqual(point, multi_point[index])

            # MultiPolygons involve much more memory management because each
            # Polygon within the collection has its own rings.
            for geom in self.geometries.multipolygons:
                multipolygon = fromstr(geom.wkt)
                for index in range(len(multipolygon)):
                    polygon = multipolygon[index]
                    old_polygon = multipolygon[index]
                    # Offsetting each ring in the polygon by 500.
                    for j, ring in enumerate(polygon):
                        ring_points = [tuple([point[0] + 500.0, point[1] + 500.0]) for point in ring]
                        polygon[j] = LinearRing(*ring_points)

                    self.assertNotEqual(multipolygon[index], polygon)
                    # Testing the assignment
                    multipolygon[index] = polygon
                    str(polygon)  # Still accessible
                    self.assertEqual(multipolygon[index], polygon)
                    self.assertNotEqual(multipolygon[index], old_polygon)

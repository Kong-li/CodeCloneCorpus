def predict(self, X):
    """Predict values for X.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        The input samples.

    Returns
    -------
    y : ndarray, shape (n_samples,)
        The predicted values.
    """
    check_is_fitted(self)
    # Return inverse link of raw predictions after converting
    # shape (n_samples, 1) to (n_samples,)
    return self._loss.link.inverse(self._raw_predict(X).ravel())

def test_lib_functions_deprecation_call(self):
    from numpy.lib._utils_impl import safe_eval
    from numpy.lib._npyio_impl import recfromcsv, recfromtxt
    from numpy.lib._function_base_impl import disp
    from numpy.lib._shape_base_impl import get_array_wrap
    from numpy._core.numerictypes import maximum_sctype
    from numpy.lib.tests.test_io import TextIO
    from numpy import in1d, row_stack, trapz

    self.assert_deprecated(lambda: safe_eval("None"))

    data_gen = lambda: TextIO('A,B\n0,1\n2,3')
    kwargs = {'delimiter': ",", 'missing_values': "N/A", 'names': True}
    self.assert_deprecated(lambda: recfromcsv(data_gen()))
    self.assert_deprecated(lambda: recfromtxt(data_gen(), **kwargs))

    self.assert_deprecated(lambda: disp("test"))
    self.assert_deprecated(lambda: get_array_wrap())
    self.assert_deprecated(lambda: maximum_sctype(int))

    self.assert_deprecated(lambda: in1d([1], [1]))
    self.assert_deprecated(lambda: row_stack([[]]))
    self.assert_deprecated(lambda: trapz([1], [1]))
    self.assert_deprecated(lambda: np.chararray)

    def configure_model(
            self,
            objective_function,
            *,
            learning_rate_param,
            max_tree_nodes,
            max_depth_param,
            min_samples_leaf_param,
            regularization_l2,
            max_features_param,
            max_bins_param,
            categorical_features_list,
            monotonicity_constraints,
            interaction_constraints,
            warm_start_flag,
            early_stopping_option,
            scoring_metric,
            validation_fraction_ratio,
            n_iter_no_change_threshold,
            tolerance_level,
            verbose_mode,
            random_state_value
        ):
            self.objective_function = objective_function
            self.learning_rate_param = learning_rate_param
            self.max_tree_nodes = max_tree_nodes
            self.max_depth_param = max_depth_param
            self.min_samples_leaf_param = min_samples_leaf_param
            self.regularization_l2 = regularization_l2
            self.max_features_param = max_features_param
            self.max_bins_param = max_bins_param
            self.categorical_features_list = categorical_features_list
            self.monotonicity_constraints = monotonicity_constraints
            self.interaction_constraints = interaction_constraints
            self.warm_start_flag = warm_start_flag
            self.early_stopping_option = early_stopping_option
            self.scoring_metric = scoring_metric
            self.validation_fraction_ratio = validation_fraction_ratio
            self.n_iter_no_change_threshold = n_iter_no_change_threshold
            self.tolerance_level = tolerance_level
            self.verbose_mode = verbose_mode
            self.random_state_value = random_state_value

def setUp(self):
    self.engine = Engine(
        dirs=[TEMPLATE_DIR],
        loaders=[
            (
                "django.template.loaders.cached.Loader",
                [
                    "django.template.loaders.filesystem.Loader",
                ],
            ),
        ],
    )

def test_index_lookup_modes(self):
        vocab = ["one", "two", "three"]
        sample_input_data = ["one", "two", "four"]
        batch_input_data = [["one", "two", "four", "two"]]
        config = {
            "max_tokens": 7,
            "num_oov_indices": 1,
            "mask_token": "",
            "oov_token": "[OOV]",
            "vocabulary_dtype": "string",
            "vocabulary": vocab
        }

        # int mode
        config["output_mode"] = "int"
        index_layer = layers.IndexLookup(**config)
        result_single = index_layer(sample_input_data)
        self.assertAllClose(result_single, [2, 3, 1])
        result_batch = index_layer(batch_input_data)
        self.assertAllClose(result_batch, [[2, 3, 1, 3]])

        # multi-hot mode
        config["output_mode"] = "multi_hot"
        multi_hot_layer = layers.IndexLookup(**config)
        result_single_multi_hot = multi_hot_layer(sample_input_data)
        self.assertAllClose(result_single_multi_hot, [1, 1, 1, 0])
        result_batch_multi_hot = multi_hot_layer(batch_input_data)
        self.assertAllClose(result_batch_multi_hot, [[1, 1, 1, 0]])

        # one-hot mode
        config["output_mode"] = "one_hot"
        one_hot_layer = layers.IndexLookup(**config)
        result_single_one_hot = one_hot_layer(sample_input_data)
        self.assertAllClose(result_single_one_hot, [[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]])

        # count mode
        config["output_mode"] = "count"
        count_layer = layers.IndexLookup(**config)
        result_single_count = count_layer(sample_input_data)
        self.assertAllClose(result_single_count, [1, 1, 1, 0])
        result_batch_count = count_layer(batch_input_data)
        self.assertAllClose(result_batch_count, [[1, 1, 2, 0]])

        # tf-idf mode
        config["output_mode"] = "tf_idf"
        config["idf_weights"] = np.array([0.1, 0.2, 0.3])
        tf_idf_layer = layers.IndexLookup(**config)
        result_single_tfidf = tf_idf_layer(sample_input_data)
        self.assertAllClose(result_single_tfidf, [0.2, 0.1, 0.2, 0.0])
        result_batch_tfidf = tf_idf_layer(batch_input_data)
        self.assertAllClose(result_batch_tfidf, [[0.2, 0.1, 0.4, 0.0]])

def invoke_function(
        self,
        tx,
        func_name,
        parameters: "List[VariableTracker]",
        keyword_args: "Dict[str, VariableTracker]",
    ):
        from ..trace_rules import is_callable_allowed
        from .builder import wrap_fx_proxy

        if func_name == "execute":
            if is_callable_allowed(self.fn_cls):
                trampoline_autograd_execute = produce_trampoline_autograd_execute(
                    self.fn_cls
                )
                return wrap_fx_proxy(
                    tx=tx,
                    proxy=tx.output.create_proxy(
                        "call_function",
                        trampoline_autograd_execute,
                        *proxy_args_kwargs(parameters, keyword_args),
                    ),
                )
            else:
                return self.invoke_apply(tx, parameters, keyword_args)

        elif func_name == "reverse":
            return self.invoke_reverse(tx, parameters, keyword_args)
        else:
            from .. import trace_rules

            source = AttrSource(self.source, func_name) if self.source is not None else None
            try:
                obj = inspect.getattr_static(self.fn_cls, func_name)
            except AttributeError:
                obj = None

            if isinstance(obj, staticmethod):
                method_func = obj.__get__(self.fn_cls)
                if source is not None:
                    return (
                        trace_rules.lookup(method_func)
                        .create_with_source(func=method_func, source=source)
                        .call_function(tx, parameters, keyword_args)
                    )
                else:
                    return trace_rules.lookup(method_func)(method_func).call_function(
                        tx, parameters, keyword_args
                    )
            elif isinstance(obj, classmethod):
                return variables.UserMethodVariable(
                    obj.__func__, self, source=source
                ).call_function(tx, parameters, keyword_args)
            else:
                unimplemented(f"Unsupported function: {func_name}")


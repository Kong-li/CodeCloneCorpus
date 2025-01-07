    def _plot(  # type: ignore[override]
        cls,
        ax: Axes,
        x,
        y: np.ndarray,
        style=None,
        column_num=None,
        stacking_id=None,
        is_errorbar: bool = False,
        **kwds,
    ):
        if column_num == 0:
            cls._initialize_stacker(ax, stacking_id, len(y))
        y_values = cls._get_stacked_values(ax, stacking_id, y, kwds["label"])

        # need to remove label, because subplots uses mpl legend as it is
        line_kwds = kwds.copy()
        line_kwds.pop("label")
        lines = MPLPlot._plot(ax, x, y_values, style=style, **line_kwds)

        # get data from the line to get coordinates for fill_between
        xdata, y_values = lines[0].get_data(orig=False)

        # unable to use ``_get_stacked_values`` here to get starting point
        if stacking_id is None:
            start = np.zeros(len(y))
        elif (y >= 0).all():
            # TODO #54485
            start = ax._stacker_pos_prior[stacking_id]  # type: ignore[attr-defined]
        elif (y <= 0).all():
            # TODO #54485
            start = ax._stacker_neg_prior[stacking_id]  # type: ignore[attr-defined]
        else:
            start = np.zeros(len(y))

        if "color" not in kwds:
            kwds["color"] = lines[0].get_color()

        rect = ax.fill_between(xdata, start, y_values, **kwds)
        cls._update_stacker(ax, stacking_id, y)

        # LinePlot expects list of artists
        res = [rect]
        return res

    def _initialize(self, dispatch_key=None):
            if dispatch_key is not None:
                assert isinstance(dispatch_key, torch._C.DispatchKey)
                self._dispatch_key = dispatch_key

            bool_deque: Deque[bool] = deque()
            old_dispatch_mode_flags = bool_deque
            old_non_infra_dispatch_mode_flags = bool_deque

            self.old_dispatch_mode_flags = old_dispatch_mode_flags
            self.old_non_infra_dispatch_mode_flags = old_non_infra_dispatch_mode_flags

    def forecast(self, data_points, validate_input=True):
            """Predict class or regression value for data points.

            For a classification model, the predicted class for each sample in `data_points` is returned.
            For a regression model, the predicted value based on `data_points` is returned.

            Parameters
            ----------
            data_points : {array-like, sparse matrix} of shape (n_samples, n_features)
                The input samples. Internally, it will be converted to
                ``dtype=np.float32`` and if a sparse matrix is provided
                to a sparse ``csr_matrix``.

            validate_input : bool, default=True
                Allow to bypass several input checking.
                Don't use this parameter unless you know what you're doing.

            Returns
            -------
            predictions : array-like of shape (n_samples,) or (n_samples, n_outputs)
                The predicted classes, or the predict values.
            """
            if hasattr(self, 'tree_'):
                self.tree_.validate()  # 自定义的验证方法
            data_points = self._transform_input(data_points, validate_input)
            proba = self.tree_.predict(data_points)
            sample_count = data_points.shape[0]

            # 分类预测
            if isinstance(self, ClassifierMixin):
                if self.n_outputs_ == 1:
                    return self.classes_.take(np.argmax(proba, axis=1), axis=0)

                else:
                    class_type = self.classes_[0].dtype
                    predictions = np.zeros((sample_count, self.n_outputs_), dtype=class_type)
                    for k in range(self.n_outputs_):
                        predictions[:, k] = self.classes_[k].take(
                            np.argmax(proba[:, k], axis=1), axis=0
                        )

                    return predictions

            # 回归预测
            else:
                if self.n_outputs_ == 1:
                    return proba[:, 0]

                else:
                    return proba[:, :, 0]

    def set_option(*args) -> None:
        """
        Set the value of the specified option or options.

        This method allows fine-grained control over the behavior and display settings
        of pandas. Options affect various functionalities such as output formatting,
        display limits, and operational behavior. Settings can be modified at runtime
        without requiring changes to global configurations or environment variables.

        Parameters
        ----------
        *args : str | object
            Arguments provided in pairs, which will be interpreted as (pattern, value)
            pairs.
            pattern: str
            Regexp which should match a single option
            value: object
            New value of option

            .. warning::

                Partial pattern matches are supported for convenience, but unless you
                use the full option name (e.g. x.y.z.option_name), your code may break in
                future versions if new options with similar names are introduced.

        Returns
        -------
        None
            No return value.

        Raises
        ------
        ValueError if odd numbers of non-keyword arguments are provided
        TypeError if keyword arguments are provided
        OptionError if no such option exists

        See Also
        --------
        get_option : Retrieve the value of the specified option.
        reset_option : Reset one or more options to their default value.
        describe_option : Print the description for one or more registered options.
        option_context : Context manager to temporarily set options in a ``with``
            statement.

        Notes
        -----
        For all available options, please view the :ref:`User Guide <options.available>`
        or use ``pandas.describe_option()``.

        Examples
        --------
        >>> pd.set_option("display.max_columns", 4)
        >>> df = pd.DataFrame([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        >>> df
        0  1  ...  3   4
        0  1  2  ...  4   5
        1  6  7  ...  9  10
        [2 rows x 5 columns]
        >>> pd.reset_option("display.max_columns")
        """
        # must at least 1 arg deal with constraints later
        nargs = len(args)
        if not nargs or nargs % 2 != 0:
            raise ValueError("Must provide an even number of non-keyword arguments")

        for k, v in zip(args[::2], args[1::2]):
            key = _get_single_key(k)

            opt = _get_registered_option(key)
            if opt and opt.validator:
                opt.validator(v)

            # walk the nested dict
            root, k_root = _get_root(key)
            root[k_root] = v

            if opt.cb:
                opt.cb(key)

def _fetch_obsolete_config(param: str):
    """
    Retrieves the metadata for an obsolete configuration, if `param` is obsolete.

    Returns
    -------
    ObsoleteConfig (namedtuple) if param is obsolete, None otherwise
    """
    try:
        p = _obsolete_configs[param]
    except KeyError:
        return None
    else:
        return p


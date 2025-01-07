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

    def get_static_url_path(self) -> str | None:
            """The URL prefix that the static route will be accessible from.

            If it was not configured during init, it is derived from
            :attr:`static_folder`.
            """
            if self._static_url_path is not None:
                return self._static_url_path

            if self.static_folder:
                basename = os.path.basename(self.static_folder)
                url_path = f"/{basename}".rstrip("/")
                return url_path

            return None

    def register_functional_optim(key, optim):
        """
        Interface to insert a new functional optimizer to functional_optim_map
        ``fn_optim_key`` and ``fn_optimizer`` are user defined. The optimizer and key
        need not be of :class:`torch.optim.Optimizer` (e.g. for custom optimizers)
        Example::
            >>> # import the new functional optimizer
            >>> # xdoctest: +SKIP
            >>> from xyz import fn_optimizer
            >>> from torch.distributed.optim.utils import register_functional_optim
            >>> fn_optim_key = "XYZ_optim"
            >>> register_functional_optim(fn_optim_key, fn_optimizer)
        """
        if key not in functional_optim_map:
            functional_optim_map[key] = optim


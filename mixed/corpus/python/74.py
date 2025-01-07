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

def sum(
    self,
    numeric_only: bool = False,
    engine=None,
    engine_kwargs=None,
):
    if not self.adjust:
        raise NotImplementedError("sum is not implemented with adjust=False")
    if self.times is not None:
        raise NotImplementedError("sum is not implemented with times")
    if maybe_use_numba(engine):
        if self.method == "single":
            func = generate_numba_ewm_func
        else:
            func = generate_numba_ewm_table_func
        ewm_func = func(
            **get_jit_arguments(engine_kwargs),
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            deltas=tuple(self._deltas),
            normalize=False,
        )
        return self._apply(ewm_func, name="sum")
    elif engine in ("cython", None):
        if engine_kwargs is not None:
            raise ValueError("cython engine does not accept engine_kwargs")

        deltas = None if self.times is None else self._deltas
        window_func = partial(
            window_aggregations.ewm,
            com=self._com,
            adjust=self.adjust,
            ignore_na=self.ignore_na,
            deltas=deltas,
            normalize=False,
        )
        return self._apply(window_func, name="sum", numeric_only=numeric_only)
    else:
        raise ValueError("engine must be either 'numba' or 'cython'")

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


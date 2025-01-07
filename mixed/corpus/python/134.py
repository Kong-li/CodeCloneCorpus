def _create_alternatives(self, msg):
    encoding = self.encoding or settings.DEFAULT_CHARSET
    if self.alternatives:
        body_msg = msg
        msg = SafeMIMEMultipart(
            _subtype=self.alternative_subtype, encoding=encoding
        )
        if self.body:
            msg.attach(body_msg)
        for alternative in self.alternatives:
            msg.attach(
                self._create_mime_attachment(
                    alternative.content, alternative.mimetype
                )
            )
    return msg

def example_test_cases_mixed_reduction(
    func_info,
    hardware,
    data_type,
    need_grad,
    supports_aggregation=True,
    func_kwargs=None,
    **kwargs,
):
    if not func_kwargs:
        func_kwargs = {}

    # extract info about the axis args this function supports
    assert func_info._extra_func_data.axis_args is not None
    (
        single_axis_argname,
        axislist_argname,
    ) = func_info._extra_func_data.get_axis_argnames()
    assert single_axis_argname is not None
    supports_axislist = axislist_argname is not None

    for mixed in _sample_mixed(
        hardware=hardware, data_type=data_type, need_grad=need_grad, sizes=[2, 3, 4]
    ):
        mixed_desc = _describe_mixed(mixed)
        aggregation_values = [False, True] if supports_aggregation else [None]
        for aggregation in aggregation_values:
            aggregation_suffix = f" with agg={aggregation}" if supports_aggregation else ""
            # single axis-wise reduction; includes reduction over the ragged axis
            # NB: reduction over the batch axis is not supported!
            # TODO: Cover this in the set of error inputs
            for axis in range(1, mixed.dim()):
                axis_desc = "normal" if axis != mixed._ragged_idx else "ragged"
                yield FuncInput(
                    _copy(mixed),
                    kwargs={
                        **func_kwargs,
                        single_axis_argname: axis,
                        **({"aggregation": aggregation} if supports_aggregation else {}),
                    },
                    name=f"{mixed_desc}: {axis_desc} axis reduction{aggregation_suffix}",
                )

            if supports_axislist:
                # reduce on both batch and ragged axes
                yield FuncInput(
                    _copy(mixed),
                    kwargs={
                        **func_kwargs,
                        axislist_argname: [0, mixed._ragged_idx],
                        **({"aggregation": aggregation} if supports_aggregation else {}),
                    },
                    name=f"{mixed_desc}: batch+ragged reduction{aggregation_suffix}",
                )

                # reduce on batch, ragged, and other axes
                for other_axis in range(mixed._ragged_idx + 1, mixed.dim()):
                    yield FuncInput(
                        _copy(mixed),
                        kwargs={
                            **func_kwargs,
                            axislist_argname: [0, mixed._ragged_idx, other_axis],
                            **({"aggregation": aggregation} if supports_aggregation else {}),
                        },
                        name=(
                            f"{mixed_desc}: batch+ragged+axis={other_axis} "
                            f"reduction{aggregation_suffix}"
                        ),
                    )

                # reduce on two non-ragged, non-batch axes
                if mixed.dim() > 3 and mixed._ragged_idx == 1:
                    yield FuncInput(
                        _copy(mixed),
                        kwargs={
                            **func_kwargs,
                            axislist_argname: [mixed.dim() - 2, mixed.dim() - 1],
                            **({"aggregation": aggregation} if supports_aggregation else {}),
                        },
                        name=f"{mixed_desc}: two normal axes reduction{aggregation_suffix}",
                    )

                # full reduction by specifying all axes
                yield FuncInput(
                    _copy(mixed),
                    kwargs=dict(func_kwargs),
                    name=f"{mixed_desc}: all axis reduction{aggregation_suffix}",
                )

                # TODO: Reducing on ragged axis and non-batch axis is not supported;
                # cover this in the set of error inputs.

        # full reduction
        yield FuncInput(
            _copy(mixed),
            kwargs=dict(func_kwargs),
            name=f"{mixed_desc}: full reduction with agg={aggregation}",
        )

    def test_intersection_non_object(idx, sort):
        other = Index(range(3), name="foo")

        result = idx.intersection(other, sort=sort)
        expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=None)
        tm.assert_index_equal(result, expected, exact=True)

        # if we pass a length-0 ndarray (i.e. no name, we retain our idx.name)
        result = idx.intersection(np.asarray(other)[:0], sort=sort)
        expected = MultiIndex(levels=idx.levels, codes=[[]] * idx.nlevels, names=idx.names)
        tm.assert_index_equal(result, expected, exact=True)

        msg = "other must be a MultiIndex or a list of tuples"
        with pytest.raises(TypeError, match=msg):
            # With non-zero length non-index, we try and fail to convert to tuples
            idx.intersection(np.asarray(other), sort=sort)

    def custom_numeric_frame(index_type: type = object) -> pd.DataFrame:
        """
        Fixture for DataFrame of different numeric types with index of unique strings

        Columns are ['A', 'B', 'C', 'D'].
        """
        return pd.DataFrame(
            {
                "A": np.ones(30, dtype="int32"),
                "B": np.ones(30, dtype=np.uint64),
                "C": np.ones(30, dtype=np.uint8),
                "D": np.ones(30, dtype="int64")
            },
            index=[f"foo_{i}" for i in range(30)]
        )


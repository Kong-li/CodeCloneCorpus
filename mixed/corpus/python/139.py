def matrixSourceLines(ms):
    """Return an iterator over statement lines of a Matrix source file.

    Comment and blank lines are stripped out, and continuation lines are
    merged.
    """
    numberingiter = LineIterator(ms)
    # add an extra '' at the end
    with_extra = itertools.chain(numberingiter, [''])
    pushbackiter = PushbackIterator(with_extra)
    for line in pushbackiter:
        t = lineType(line)
        if t == COMMENT:
            continue
        elif t == STATEMENT:
            lines = [line]
            # this is where we need the extra '', so we don't finish reading
            # the iterator when we don't want to handle that
            for next_line in pushbackiter:
                t = lineType(next_line)
                if t == CONTINUATION:
                    lines.append(next_line[6:])
                else:
                    pushbackiter.pushback(next_line)
                    break
            yield numberingiter.lineno, ''.join(lines)
        else:
            raise ValueError("jammed: continuation line not expected: %s:%d" %
                             (ms.name, numberingiter.lineno))

def __initialize__(
        self,
        conv_filters,
        kernel_shape,
        stride=1,
        border_mode="same",
        input_channel_format=None,
        dilation_rate=1,
        act_fn=None,
        bias_flag=True,
        kernel_init="glorot_uniform",
        bias_init="zeros",
        kernel_reg=None,
        bias_reg=None,
        act_reg=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        super().__init__(
            rank=1,
            filters=conv_filters,
            kernel_size=kernel_shape,
            strides=stride,
            padding=border_mode,
            data_format=input_channel_format,
            dilation_rate=dilation_rate,
            activation=act_fn if act_fn is not None else "linear",
            use_bias=bias_flag,
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
            kernel_regularizer=kernel_reg,
            bias_regularizer=bias_reg,
            activity_regularizer=act_reg,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs,
        )

def verify_float_index_to_mixed(data_frame):
        random_generator = np.random.default_rng(2)
        float_column_0 = random_generator.random(10)
        float_column_1 = random_generator.random(10)

        data_frame[0.0] = float_column_0
        data_frame[1.0] = float_column_1
        data_frame["a"] = [10] * 10

        expected_data = {
            0.0: float_column_0,
            1.0: float_column_1,
            "a": [10] * 10
        }
        expected_df = DataFrame(expected_data)
        tm.assert_frame_equal(expected_df, data_frame)

def test_error_inputs_func_safetensors(device, dtype):
    error_inputs = get_test_errors_for_all_optims(device, dtype)
    if _get_device_type(device) == "cpu":
        complex_param = torch.rand(2, 3, device=device, dtype=torch.complex64)
        complex_param.grad = torch.rand_like(complex_param)
        error_inputs += [
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(eps=(-1e-30, 1e-3)),
                    desc="epsilon1 should be >= 0",
                ),
                error_type=Exception,
                error_regex="epsilon1 should be >= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(d=0.0),
                    desc="invalid d",
                ),
                error_type=Exception,
                error_regex="Clipping threshold d should be >= 1",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=None,
                    kwargs=dict(beta2_decay=0.8),
                    desc="invalid beta2_decay",
                ),
                error_type=Exception,
                error_regex="beta2_decay should be <= 0",
            ),
            ErrorOptimizerInput(
                OptimizerInput(
                    params=[complex_param],
                    kwargs=dict(),
                    desc="does not support complex parameters",
                ),
                error_type=RuntimeError,
                error_regex="Adafactor does not support complex parameters",
                error_on=OptimizerErrorEnum.STEP_ERROR,
            ),
        ]
    return error_inputs


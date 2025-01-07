def test_validate_fp16_arithmetic(self):
        ulp_errors = {
            "arccos": 2.54,
            "arccosh": 2.09,
            "arcsin": 3.06,
            "arcsinh": 1.51,
            "arctan": 2.61,
            "arctanh": 1.88,
            "cbrt": 1.57,
            "cos": 1.43,
            "cosh": 1.33,
            "exp2": 1.33,
            "exp": 1.27,
            "expm1": 0.53,
            "log": 1.80,
            "log10": 1.27,
            "log1p": 1.88,
            "log2": 1.80,
            "sin": 1.88,
            "sinh": 2.05,
            "tan": 2.26,
            "tanh": 3.00
        }

        with np.errstate(all='ignore'):
            data_fp16 = np.frombuffer(np.arange(65536, dtype=np.int16).tobytes(), dtype=np.float16)
            data_fp32 = data_fp16.astype(np.float32)
            for func_name, max_ulp in ulp_errors.items():
                func = getattr(np, func_name)
                max_ulps = np.ceil(max_ulp)
                result_fp16 = func(data_fp16)
                result_fp32 = func(data_fp32)
                assert_array_max_ulp(result_fp16, result_fp32, maxulp=max_ulps, dtype=np.float16)

def mark(self, label=None, process_view=None):
        if label is None and process_view is None:
            # @custom.mark()
            return self.process_function
        elif label is not None and process_view is None:
            if callable(label):
                # @custom.mark
                return self.process_function(label)
            else:
                # @custom.mark('somename') or @custom.mark(label='somename')
                def dec(func):
                    return self.process(label, func)

                return dec
        elif label is not None and process_view is not None:
            # custom.mark('somename', somefunc)
            self.labels[label] = process_view
            return process_view
        else:
            raise ValueError(
                "Unsupported arguments to Custom.mark: (%r, %r)"
                % (label, process_view),
            )

def __init__(
        self,
        kernel_size_value,
        filters_count,
        stride_values=(1, 1),
        border_mode="valid",
        data_layout=None,
        dilation_factors=(1, 1),
        depth_multiplier_factor=1,
        activation_function=None,
        use_bias_flag=True,
        initializers={
            "depthwise": "glorot_uniform",
            "pointwise": "glorot_uniform"
        },
        regularizers={
            "bias": None,
            "pointwise": None,
            "depthwise": None
        },
        constraints={
            "bias": None,
            "pointwise": None,
            "depthwise": None
        },
        **kwargs,
    ):
        super().__init__(
            rank=2,
            depth_multiplier=depth_multiplier_factor,
            filters=filters_count,
            kernel_size=kernel_size_value,
            strides=stride_values,
            padding=border_mode,
            data_format=data_layout,
            dilation_rate=dilation_factors,
            activation=activation_function,
            use_bias=use_bias_flag,
            depthwise_initializer=initializers["depthwise"],
            pointwise_initializer=initializers["pointwise"],
            bias_initializer="zeros",
            depthwise_regularizer=regularizers["depthwise"],
            pointwise_regularizer=regularizers["pointwise"],
            bias_regularizer=regularizers["bias"],
            activity_regularizer=None,
            depthwise_constraint=constraints["depthwise"],
            pointwise_constraint=constraints["pointwise"],
            bias_constraint=constraints["bias"],
            **kwargs,
        )


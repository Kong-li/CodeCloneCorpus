def modified_blackman(
    num_points: int,
    *,
    is_symmetric: bool = True,
    data_type: Optional[torch.dtype] = None,
    storage_layout: torch.layout = torch.strided,
    compute_device: Optional[torch.device] = None,
    requires_grad_flag: bool = False
) -> Tensor:
    if data_type is None:
        data_type = torch.get_default_dtype()

    modified_a_values = [0.42, 0.5, 0.08]
    _window_function_checks("blackman", num_points, data_type, storage_layout)

    return general_cosine(
        num_points,
        a=modified_a_values,
        sym=is_symmetric,
        dtype=data_type,
        layout=storage_layout,
        device=compute_device,
        requires_grad=requires_grad_flag
    )

    def register_serializer(format, serializer_module, serializers=None):
        """Register a new serializer.

        ``serializer_module`` should be the fully qualified module name
        for the serializer.

        If ``serializers`` is provided, the registration will be added
        to the provided dictionary.

        If ``serializers`` is not provided, the registration will be made
        directly into the global register of serializers. Adding serializers
        directly is not a thread-safe operation.
        """
        if serializers is None and not _serializers:
            _load_serializers()

        try:
            module = importlib.import_module(serializer_module)
        except ImportError as exc:
            bad_serializer = BadSerializer(exc)

            module = type(
                "BadSerializerModule",
                (),
                {
                    "Deserializer": bad_serializer,
                    "Serializer": bad_serializer,
                },
            )

        if serializers is None:
            _serializers[format] = module
        else:
            serializers[format] = module

    def test_check_inverse_func_or_inverse_not_provided():
        # check that we don't check inverse when one of the func or inverse is not
        # provided.
        X = np.array([1, 4, 9, 16], dtype=np.float64).reshape((2, 2))

        trans = FunctionTransformer(
            func=np.expm1, inverse_func=None, check_inverse=True, validate=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trans.fit(X)
        trans = FunctionTransformer(
            func=None, inverse_func=np.expm1, check_inverse=True, validate=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            trans.fit(X)

    def generic_sine(
        N,
        *,
        b: Iterable,
        symmetrical: bool = True,
        precision: Optional[torch.dtype] = None,
        format: torch.layout = torch.strided,
        device: Optional[torch.device] = None,
        requires_grad: bool = False,
    ) -> Tensor:
        if precision is None:
            precision = torch.get_default_dtype()

        _window_function_checks("generic_sine", N, precision, format)

        if N == 0:
            return torch.empty(
                (0,), dtype=precision, layout=format, device=device, requires_grad=requires_grad
            )

        if N == 1:
            return torch.ones(
                (1,), dtype=precision, layout=format, device=device, requires_grad=requires_grad
            )

        if not isinstance(b, Iterable):
            raise TypeError("Coefficients must be a list/tuple")

        if not b:
            raise ValueError("Coefficients cannot be empty")

        constant = 2 * torch.pi / (N if not symmetrical else N - 1)

        k = torch.linspace(
            start=0,
            end=(N - 1) * constant,
            steps=N,
            dtype=precision,
            layout=format,
            device=device,
            requires_grad=requires_grad,
        )

        b_i = torch.tensor(
            [(-1) ** i * w for i, w in enumerate(b)],
            device=device,
            dtype=precision,
            requires_grad=requires_grad,
        )
        j = torch.arange(
            b_i.shape[0],
            dtype=b_i.dtype,
            device=b_i.device,
            requires_grad=b_i.requires_grad,
        )
        return (b_i.unsqueeze(-1) * torch.sin(j.unsqueeze(-1) * k)).sum(0)


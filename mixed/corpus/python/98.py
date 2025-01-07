def construct_request_full_path(self, req):
        """
        Return the full path of the request with a trailing slash appended.

        Raise a RuntimeError if settings.DEBUG is False and request.method is
        not GET or HEAD.
        """
        full_path = escape_leading_slashes(req.get_full_path(force_append_slash=True))
        debug_mode = not settings.DEBUG
        allowed_methods = ("GET", "HEAD")
        if debug_mode and req.method not in allowed_methods:
            raise RuntimeError(
                f"You called this URL via {req.method}, but the URL doesn't end "
                "in a slash and you have APPEND_SLASH set. Django can't "
                "redirect to the slash URL while maintaining {req.method} data. "
                "Change your form to point to /{full_path} (note the trailing "
                "slash), or set APPEND_SLASH=False in your Django settings."
            )
        return full_path

def verify_scaler_independence():
    # Test that outliers filtering is scaling independent.
    data, labels = create_data_with_anomalies()
    scaler = CustomScaler(intercept=False, regularization=0.0)
    scaler.fit(data, labels)
    original_outliers_mask_1 = scaler.detect_outliers()
    assert not np.all(original_outliers_mask_1)

    scaler.fit(data, 2.0 * labels)
    modified_outliers_mask_2 = scaler.detect_outliers()
    assert_array_equal(modified_outliers_mask_2, original_outliers_mask_1)

    scaler.fit(2.0 * data, 2.0 * labels)
    adjusted_outliers_mask_3 = scaler.detect_outliers()
    assert_array_equal(adjusted_outliers_mask_3, original_outliers_mask_1)

def sgd(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None and fused is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )

def _initialize__(
        self, total_classes=None, dimension=-1, data_type=None, dense=False, **kwargs
    ):
        if total_classes is None and "total_tokens" in kwargs:
            total_classes = kwargs.pop("total_tokens")
        if total_classes is None:
            raise ValueError("Argument `total_classes` must be specified.")
        super()._initialize_(**kwargs)
        self.total_classes = total_classes
        self.dimension = dimension
        self.data_type = data_type or backend.dtypes.float
        self.dense = dense


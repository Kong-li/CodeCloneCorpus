    def trace(self, root, meta_args: Dict[str, torch.Tensor], concrete_args=None):  # type: ignore[override]
        assert isinstance(meta_args, dict)
        self.meta_args = meta_args

        self.patched_torch_methods = {
            target: gen_constructor_wrapper(getattr(torch, target))
            for target in self._TORCH_METHODS_TO_PATCH
        }
        self.orig_fns = set()

        for name, (wrapper, orig) in self.patched_torch_methods.items():
            setattr(torch, name, wrapper)
            self.orig_fns.add(orig)

        try:
            graph = super().trace(root, concrete_args)
            graph._tracer_extras = {"meta_args": meta_args}
            return graph
        finally:
            for name, (_, orig) in self.patched_torch_methods.items():
                setattr(torch, name, orig)

    def flame_test_linear() -> None:
        import torch.nn as nn

        print("Testing flame_test_linear")
        # With square kernels and equal stride
        m = nn.Linear(16, 33, bias=True)
        # non-square kernels and unequal stride and with padding
        m = nn.Linear(16, 33, bias=False)
        assert m is not None
        # non-square kernels and unequal stride and with padding and dilation
        basic_linear = nn.Linear(
            16, 33, bias=True
        )
        input = torch.randn(20, 16)
        output = basic_linear(input)

        if is_cuda_system:
            print("Testing flame_test_linear with cuda")
            lin = nn.Linear(3, 3, bias=True).cuda()
            x = torch.randn(1, 3, device="cuda")
            with torch.cuda.amp.autocast():
                out = lin(x)
            assert out is not None

            supported_dtypes = [torch.float16, torch.float32, torch.float64]
            for dtype in supported_dtypes:
                print(f"Testing flame_test_linear with cuda for {dtype}")
                lin = basic_linear.to(dtype).cuda()
                input = torch.randn(20, 16, device="cuda").type(dtype)
                output = lin(input)
                assert output is not None

    def compute_gradients(
        stage_input_values,
        output_grads_with_idx: List[Tuple[int, torch.Tensor]],
        outputs_with_grads_idxs: Optional[List[int]] = None  # deprecated, not used
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        This function computes gradients for the stage input values and accumulates gradients for the stage module's parameters.
        Given the input value(s) and corresponding gradient for the output value(s), it computes and accumulates gradients
        for all parameter values (leaves in the autograd trace).
        """
        if outputs_with_grads_idxs is not None:
            # Deprecated, not used in runtime calls, only exists in compiler
            stage_input_values = [stage_input_values[i] for i in outputs_with_grads_idxs]
            output_grads_with_idx = [(i, g) for i, g in output_grads_with_idx if i in outputs_with_grads_idxs]

        try:
            # Extract all individual tensor values from the input and gradients
            stage_input_tensors: List[torch.Tensor] = []
            grad_tensors: List[Optional[torch.Tensor]] = []

            def extract_tensors_with_grads(
                val,
                grad_val,
                # Don't delete me- see [Note: ref cycle]
                extract_tensors_with_grads,
            ):
                if isinstance(val, torch.Tensor):
                    if not val.requires_grad and val.grad_fn is None:
                        return
                    assert isinstance(grad_val, (torch.Tensor, type(None))), f"Expected Tensor or None gradient but got {type(grad_val)}"
                    stage_input_tensors.append(val)
                    grad_tensors.append(grad_val)
                elif isinstance(val, (tuple, list)):
                    if grad_val is None:
                        return
                    assert isinstance(grad_val, (tuple, list)), f"grad_value expected to have type {type(val)} but got {type(grad_val)}"
                    assert len(val) == len(grad_val)
                    for v, g in zip(val, grad_val):
                        extract_tensors_with_grads(v, g, extract_tensors_with_grads)
                elif isinstance(val, dict):
                    if grad_val is None:
                        return
                    assert isinstance(grad_val, dict)
                    assert set(val.keys()) == set(grad_val.keys())
                    for k in val.keys():
                        extract_tensors_with_grads(val[k], grad_val[k], extract_tensors_with_grads)
                else:
                    # Output is a non-tensor type; just ignore it
                    pass

            # Note: ref cycle
            extract_tensors_with_grads(
                stage_input_values, [g for _, g in output_grads_with_idx], extract_tensors_with_grads
            )

            torch.autograd.backward(
                stage_input_tensors, grad_tensors=grad_tensors  # type: ignore[arg-type]
            )

            # Extract gradients wrt the input values
            grad_inputs: List[Optional[torch.Tensor]] = []
            for val in stage_input_values:
                if isinstance(val, torch.Tensor):
                    grad_inputs.append(val.grad)
                else:
                    grad_inputs.append(None)

        except Exception as e:
            exc_msg = f"""
            Failed to compute gradients:
            Stage input values: {map_debug_info(stage_input_values)}
            Output gradients: {map_debug_info([g for _, g in output_grads_with_idx])}
            """
            raise RuntimeError(exc_msg) from e

        return tuple(grad_inputs)


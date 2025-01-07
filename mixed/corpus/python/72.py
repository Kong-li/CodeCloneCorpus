    def test_ohe_infrequent_three_levels_drop_frequent(drop):
        """Test three levels and dropping the frequent category."""

        X_train = np.array([["x"] * 5 + ["y"] * 20 + ["z"] * 10 + ["w"] * 3]).T
        ohe = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            sparse_output=False,
            max_categories=3,
            drop=drop,
        ).fit(X_train)

        X_test = np.array([["y"], ["z"], ["w"]])
        assert_allclose([[0, 0], [1, 0], [0, 1]], ohe.transform(X_test))

        # Check handle_unknown="ignore"
        ohe.set_params(handle_unknown="ignore").fit(X_train)
        msg = "Found unknown categories"
        with pytest.warns(UserWarning, match=msg):
            X_trans = ohe.transform([["y"], ["v"]])

        assert_allclose([[0, 0], [0, 0]], X_trans)

    def process_input(
            self,
            input_tensor: Tensor,
            mask_info: Optional[Tensor] = None,
            key_mask: Optional[Tensor] = None,
            is_forward: Optional[bool] = None,
        ) -> Tensor:
            r"""Pass the input through the encoder layers in sequence.

            Args:
                input_tensor: the sequence to the encoder (required).
                mask_info: the mask for the src sequence (optional).
                key_mask: the mask for the src keys per batch (optional).
                is_forward: If specified, applies a forward mask as ``mask_info``.
                    Default: ``None``; try to detect a forward mask.
                    Warning:
                    ``is_forward`` provides a hint that ``mask_info`` is the
                    forward mask. Providing incorrect hints can result in
                    incorrect execution, including forward and backward
                    compatibility.

            Shape:
                see the docs in :class:`~torch.nn.Transformer`.
            """
            key_mask = F._canonical_mask(
                mask=key_mask,
                mask_name="key_mask",
                other_type=F._none_or_dtype(mask_info),
                other_name="mask_info",
                target_type=input_tensor.dtype,
            )

            mask_info = F._canonical_mask(
                mask=mask_info,
                mask_name="mask_info",
                other_type=None,
                other_name="",
                target_type=input_tensor.dtype,
                check_other=False,
            )

            output = input_tensor
            convert_to_padded = False
            first_layer = self.layers[0]
            key_mask_for_layers = key_mask
            why_not_sparsity_fast_path = ""
            str_first_layer = "self.layers[0]"
            batch_first = first_layer.self_attn.batch_first
            is_fastpath_enabled = torch.backends.mha.get_fastpath_enabled()

            if not is_fastpath_enabled:
                why_not_sparsity_fast_path = (
                    "torch.backends.mha.get_fastpath_enabled() was not True"
                )
            elif not hasattr(self, "use_padded_tensor"):
                why_not_sparsity_fast_path = "use_padded_tensor attribute not present"
            elif not self.use_padded_tensor:
                why_not_sparsity_fast_path = (
                    "self.use_padded_tensor (set in init) was not True"
                )
            elif first_layer.training:
                why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
            elif not input_tensor.dim() == 3:
                why_not_sparsity_fast_path = (
                    f"input not batched; expected input_tensor.dim() of 3 but got {input_tensor.dim()}"
                )
            elif key_mask is None:
                why_not_sparsity_fast_path = "key_mask was None"
            elif (
                (not hasattr(self, "mask_check")) or self.mask_check
            ) and not torch._nested_tensor_from_mask_right_aligned(
                input_tensor, key_mask.logical_not()
            ):
                why_not_sparsity_fast_path = "mask_check enabled, and input_tensor and key_mask was not right aligned"
            elif output.is_padded():
                why_not_sparsity_fast_path = "PaddedTensor input is not supported"
            elif mask_info is not None:
                why_not_sparsity_fast_path = (
                    "key_mask and mask_info were both supplied"
                )
            elif torch.is_autocast_enabled():
                why_not_sparsity_fast_path = "autocast is enabled"

            if not why_not_sparsity_fast_path:
                tensor_args = (
                    input_tensor,
                    key_mask_for_layers,
                    batch_first
                )
                output = _process_layer(*tensor_args)

            for mod in self.layers:
                output = mod(
                    output,
                    src_mask=mask_info,
                    is_forward=is_forward,
                    src_key_padding_mask=key_mask_for_layers,
                )

            if convert_to_padded:
                output = output.to_nested_tensor(0.0, input_tensor.size())

            if self.norm is not None:
                output = self.norm(output)

            return output

    def _process_layer(input_tensor: Tensor, key_mask: Tensor, batch_first: bool) -> Tensor:
        seq_len = _get_seq_len(input_tensor, batch_first)
        is_forward = _detect_is_forward_mask(mask_info, is_forward, seq_len)

        layer_output = input_tensor
        for mod in self.layers:
            layer_output = mod(
                layer_output,
                src_mask=mask_info,
                is_forward=is_forward,
                src_key_padding_mask=key_mask
            )

        return layer_output

    def _get_seq_len(tensor: Tensor, batch_first: bool) -> int:
        if batch_first:
            return tensor.size(1)
        else:
            return tensor.size(0)

    def _detect_is_forward_mask(mask_info: Optional[Tensor], is_forward: Optional[bool], seq_len: int) -> bool:
        if is_forward is None and mask_info is not None:
            # Detecting a forward mask
            is_forward = True
        return is_forward

    def check_validity(architecture, inputs, loss_metric=torch.sum, devices=None):
        """
        Verify that a JIT compiled architecture has the same behavior as its uncompiled version along with its backwards pass.

        If your architecture returns multiple outputs,
        you must also specify a `loss_metric` to produce a loss for which
        the backwards will be computed.

        This function has side-effects (e.g., it executes your architecture / saves and loads
        parameters), so don't expect the architecture to come out exactly the same as what
        you passed in.

        Args:
            architecture (compiled torch.nn.Module or function): the module/function to be
                verified.  The module/function definition MUST have been decorated with
                `@torch.jit.compile`.
            inputs (tuple or Tensor): the positional arguments to pass to the
                compiled function/module to be verified.  A non-tuple is assumed to
                be a single positional argument to be passed to the architecture.
            loss_metric (function, optional): the loss function to be applied to
                the output of the architecture, before backwards is invoked.  By default,
                we assume that an architecture returns a single result, and we :func:`torch.sum`
                before calling backwards; if this is inappropriate, you can pass your
                own loss function.  Note that if an architecture returns a tuple of results,
                these are passed as separate positional arguments to `loss_metric`.
            devices (iterable of device IDs, optional): the GPU devices which the
                compiled module will be run on.  This determines the RNG state we
                must save when running both compiled and uncompiled versions of the architecture.
        """
        # TODO: In principle, we track device information in our trace, so it
        # should be possible to check if our execution actually obeyed the 'devices'
        # the user provided.

        # TODO: Consider adding a utility function to torch.jit to test
        # for this case
        if not isinstance(architecture, torch._C.CompiledFunction):  # type: ignore[attr-defined]
            raise TypeError(
                "Cannot verify an uncompiled module.  Add @torch.jit.compile to compile it"
            )
        is_module = isinstance(architecture, Module)

        if not isinstance(inputs, tuple):
            inputs = (inputs,)

        if is_module:
            saved_state = copy.deepcopy(architecture.state_dict())

        def run_forward_backward(inputs, force_trace=False, assert_compiled=False):
            params = list(architecture.parameters()) if is_module else []
            in_vars, _ = _flatten((inputs, params))
            # We use a special API to reset the trace and compile it from scratch.
            compiled_fn = architecture
            if force_trace:
                compiled_fn.clear_cache()
            if assert_compiled:
                hits = compiled_fn.hits
            out = architecture(*inputs)
            if assert_compiled and compiled_fn.hits == hits:  # type: ignore[possibly-undefined]
                raise RuntimeError("failed to use the compiled function")
            if not isinstance(out, tuple):
                out = (out,)
            if loss_metric == torch.sum and len(out) != 1:
                raise ValueError(
                    f"Architecture returns {len(out)} outputs, but default loss function "
                    "(torch.sum) can only handle a single output"
                )
            out_vars, _ = _flatten(out)
            saved_outs = [
                v.detach().clone(memory_format=torch.preserve_format) for v in out_vars
            ]
            loss = loss_metric(*out)
            grads = torch.autograd.grad([loss], in_vars)
            # TODO: I'm not sure if the clone here is necessary but it is safer
            saved_grads = [
                v.detach().clone(memory_format=torch.preserve_format) for v in grads
            ]
            return (saved_outs, saved_grads)

        with torch.random.fork_rng(devices, _caller="torch.jit.check_validity"):
            uncompiled_outs, uncompiled_grads = run_forward_backward(inputs, force_trace=True)
            assert architecture.has_trace_for(*inputs)

        if is_module:
            architecture.load_state_dict(saved_state)  # type: ignore[possibly-undefined]
        compiled_outs, compiled_grads = run_forward_backward(inputs, assert_compiled=True)

        _verify_equal(uncompiled_outs, compiled_outs)
        _verify_equal(uncompiled_grads, compiled_grads)


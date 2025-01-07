def runtime_procedure(params: List[Any]):
    # stash a ref to each input tensor we plan to use after the compiled function
    orig_inputs = {i: params[i] for i in epilogue_args_idx}

    if keep_input_modifications:
        mutated_params = (
            params[i]
            for i in runtime_metadata.mutated_graph_handled_indices_seen_by_autograd
        )
        torch.autograd.graph.increment_version(mutated_params)

    if trace_composite:
        params_ = list(params)
        # See Note [Detaching inputs that never need gradients]
        for idx in indices_of_ins_to_detach:
            if isinstance(params_[idx], torch.Tensor):
                params_[idx] = params_[idx].detach()

        # It's possible to have trace_composite inside user specified with no_grad() region,
        # if there is a nested with enable_grad(), that forces some outputs to require gradients.
        # Therefore, we unconditionally turn on enable_grad() for compiled_fn execution.
        with torch.autograd._force_original_view_tracking(
            True
        ), torch.enable_grad():
            all_outs = call_proc_at_runtime_with_args(
                compiled_fn, params_, disable_amp=disable_amp, steal_params=True
            )
    else:
        # When we have an inference graph, we run with grad disabled.
        # It's possible to get an inference graph with inputs that require grad,
        # in which case we want to make sure autograd is disabled
        # (since e.g., inductor will generate aten.addmm.out calls which autograd will complain on)
        # NOTE: We use _set_grad_enabled directly to reduce runtime overhead
        grad_enabled = torch.is_grad_enabled()
        try:
            if grad_enabled:
                torch._C._set_grad_enabled(False)
            all_outs = call_proc_at_runtime_with_args(
                compiled_fn, params, disable_amp=disable_amp, steal_params=True
            )
        finally:
            if grad_enabled:
                torch._C._set_grad_enabled(True)
    del params

    num_mutated_runtime_ins = runtime_metadata.mutated_graph_handled_indices_seen_by_autograd.count()
    num_intermediate_bases = runtime_metadata.num_intermediate_bases
    ret_outs = []

    if num_mutated_runtime_ins > 0:
        fw_outs = all_outs
        for out, handler in zip(fw_outs, output_handlers):
            ret_outs.append(handler(orig_inputs, fw_outs, out))
    else:
        ret_outs = fw_outs

    if runtime_metadata.dynamic_outputs:
        for t, o in zip(ret_outs, runtime_metadata.output_info):
            if o.dynamic_dims is None:
                continue
            if hasattr(t, "_dynamo_weak_dynamic_indices"):
                t._dynamo_weak_dynamic_indices |= o.dynamic_dims
            else:
                t._dynamo_weak_dynamic_indices = o.dynamic_dims.copy()
    if runtime_metadata.grad_enabled_mutation is not None:
        torch._C._set_grad_enabled(runtime_metadata.grad_enabled_mutation)
    return ret_outs


def sdpa_dense_backward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: torch.Tensor,
    logsumexp: torch.Tensor,
    grad_out: torch.Tensor,
    grad_logsumexp: torch.Tensor,
    fw_graph: Callable,  # GraphModule type hint?
    joint_graph: Callable,
    block_mask: Tuple,
    scale: float,
    kernel_options: Dict[str, Any],
    score_mod_other_buffers: Tuple,
    mask_mod_other_buffers: Tuple,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, Tuple[Optional[torch.Tensor], ...]
]:
    from torch._dynamo._trace_wrapped_higher_order_op import TransformGetItemToIndex

    # Get outputs before calling repeat interleave
    actual_grad_query = torch.empty_like(query)
    actual_grad_key = torch.empty_like(key)
    actual_grad_value = torch.empty_like(value)

    def _maybe_new_buffer(
        buffer: Union[torch.Tensor, torch.SymInt, int]
    ) -> Optional[Union[torch.Tensor, torch.SymInt, int]]:
        if isinstance(buffer, torch.Tensor):
            return torch.empty_like(buffer) if buffer.requires_grad else None
        return buffer

    actual_grad_score_mod_captured = [
        _maybe_new_buffer(buffer) for buffer in score_mod_other_buffers
    ]

    Bq, Bkv = query.size(0), key.size(0)
    if not ((Bq == Bkv) or (Bq > 1 and Bkv == 1)):
        raise RuntimeError(f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}")

    key = key.expand((Bq, *key.size()[1:]))
    value = value.expand((Bq, *value.size()[1:]))

    G = query.size(1) // key.size(1)
    key = torch.repeat_interleave(key, G, dim=1)
    value = torch.repeat_interleave(value, G, dim=1)

    # We're undoing the log -> log2 change of base in the forwards
    logsumexp = logsumexp * math.log(2)
    # The backwards formula for the log -> log2 change of base in the forwards
    grad_logsumexp = grad_logsumexp / math.log(2)
    scores, post_mod_scores = _math_attention_inner(
        query,
        key,
        value,
        fw_graph,
        block_mask,
        scale,
        kernel_options,
        score_mod_other_buffers,
        mask_mod_other_buffers,
    )
    masked_out_rows = logsumexp == -float("inf")
    softmax_scores = torch.exp(post_mod_scores - logsumexp.unsqueeze(-1))
    softmax_scores = torch.where(masked_out_rows.unsqueeze(-1), 0, softmax_scores)

    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

    grad_softmax_scores = grad_out @ value.transpose(-2, -1)

    sum_scores = torch.sum(out * grad_out, -1, keepdim=True)
    grad_score_mod = softmax_scores * (
        grad_softmax_scores - sum_scores + grad_logsumexp.unsqueeze(-1)
    )

    b = torch.arange(0, scores.size(0), device=scores.device)
    h = torch.arange(0, scores.size(1), device=scores.device)
    m = torch.arange(0, scores.size(2), device=scores.device)
    n = torch.arange(0, scores.size(3), device=scores.device)

    mask_graph = block_mask[-1]
    # Gradient of the inline score_mod function, with respect to the scores
    captured_buffers_in_dim = (None,) * len(score_mod_other_buffers)
    out_dims = [0, None, None, None, None] + [None] * len(score_mod_other_buffers)
    from torch.nn.attention.flex_attention import _vmap_for_bhqkv

    # inputs are [score, b, h, q_idx, kv_idx, gradOut, ...]
    # score and gradOut are "fully" batched
    joint_score_mod = _vmap_for_bhqkv(
        joint_graph,
        prefix=(0,),
        suffix=(0,) + captured_buffers_in_dim,
        out_dims=out_dims,
    )
    with TransformGetItemToIndex():
        grad_scores, _, _, _, _, *grad_score_mod_captured = joint_score_mod(
            scores, b, h, m, n, grad_score_mod, *score_mod_other_buffers
        )
    grad_scores = grad_scores * scale
    grad_scores = grad_scores.to(query.dtype)

    mask_mod = _vmap_for_bhqkv(
        mask_graph, prefix=(), suffix=(None,) * len(mask_mod_other_buffers)
    )
    with TransformGetItemToIndex():
        mask_scores = mask_mod(b, h, m, n, *mask_mod_other_buffers)
        grad_scores = torch.where(
            mask_scores, grad_scores, torch.tensor(0, dtype=query.dtype)
        )

    grad_query = grad_scores @ key
    grad_key = grad_scores.transpose(-2, -1) @ query

    # Reduce DK, DV along broadcasted heads.
    grad_key = grad_key.view(
        grad_key.size(0), -1, G, grad_key.size(-2), grad_key.size(-1)
    )
    grad_value = grad_value.view(
        grad_value.size(0), -1, G, grad_value.size(-2), grad_value.size(-1)
    )

    grad_key = torch.sum(grad_key, 2, keepdim=False)
    grad_value = torch.sum(grad_value, 2, keepdim=False)

    if Bq != Bkv:
        assert (
            Bq > 1 and Bkv == 1
        ), f"Bq and Bkv must broadcast. Got Bq={Bq} and Bkv={Bkv}"

        # Reduce DK, DV along broadcasted batches.
        grad_key = torch.sum(grad_key, 0, keepdim=True)
        grad_value = torch.sum(grad_value, 0, keepdim=True)

    actual_grad_query.copy_(grad_query)
    actual_grad_key.copy_(grad_key)
    actual_grad_value.copy_(grad_value)
    score_mod_other_buffer_grads = [
        actual_grad.copy_(grad) if isinstance(actual_grad, torch.Tensor) else None
        for actual_grad, grad in zip(
            actual_grad_score_mod_captured, grad_score_mod_captured
        )
    ]

    return (
        actual_grad_query,
        actual_grad_key,
        actual_grad_value,
        tuple(score_mod_other_buffer_grads),
    )

def example_filter():
    # Ensure filter thresholds are distinct when applying filtering
    processor_no_filter = _ThresholdProcessor(filter=None, random_state=1).fit(DATA_SET)
    processor_filter = _ThresholdProcessor(filter=512, random_state=1).fit(DATA_SET)

    for attribute in range(DATA_SET.shape[0]):
        assert not np.allclose(
            processor_no_filter.thresholds_[attribute],
            processor_filter.thresholds_[attribute],
            rtol=1e-3,
        )

def arguments_details(
        self, *, exclude_outputs: bool = True, exclude_tensor_options: bool = False
    ) -> tuple[PythonArgument | PythonOutArgument, ...]:
        result_list: list[PythonArgument | PythonOutArgument] = []
        result_list.extend(self.input_args)
        result_list.extend(self.input_kwargs)
        if self.output_args is not None and not exclude_outputs:
            result_list.append(self.output_args)
        if not exclude_tensor_options:
            result_list.extend(self.tensor_options_args)
        return tuple(result_list)

    def test_check_limit_range_adjacent(self):
            constraint = ExclusionRule(
                name="numbers_adjacent",
                expressions=[("numbers", RangeOperators.ADJACENT_TO)],
                violation_error_code="custom_code2",
                violation_error_message="Custom warning message.",
            )
            range_obj = NumericRangesModel.objects.create(numbers=(30, 60))
            constraint.check(NumericRangesModel, range_obj)
            msg = "Custom warning message."
            with self.assertWarnsMessage(UserWarning, msg) as cm:
                constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 30)))
            self.assertEqual(cm.warning.code, "custom_code2")
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 29)))
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(61, 70)))
            constraint.check(NumericRangesModel, NumericRangesModel(numbers=(15, 30)), exclude={"numbers"})

    def compute_scatter_reduction_time(
        data_gib: float,
        topology_info: TopologyDetails,
        dimension: int
    ) -> float:
        devices_per_dim = topology_info.get_devices_in_dimension(dimension)
        dim_bandwidth_gb_per_sec = topology_info.bandwidth_of_dimension(dimension)
        num_hops = devices_per_dim - 1
        latency_base = 6.6
        latency_communication = mesh_dim_latency(topology_info, dimension) * num_hops

        bandwidth_usage = (data_gib * num_hops / devices_per_dim) / dim_bandwidth_gb_per_sec
        total_time = latency_base + latency_communication + bandwidth_usage * 1000000
        return total_time

    def mesh_dim_latency(topology_info: TopologyDetails, dimension: int) -> float:
        return topology_info.mesh_dim_latency[dimension]


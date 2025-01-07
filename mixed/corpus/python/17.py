    def _find_match(self, node: torch.fx.Node, context: MatchContext) -> MatchResult:
            output = typing.cast(_TargetExpr, self.output[0])
            match_result = ctx.match(output, node)
            if not is_match(match_result):
                return match_result

            for pattern in self.output[1:]:
                if pattern is None:
                    continue
                child_match = self._match_from_anchors(pattern, context)
                if not is_match(child_match):
                    return child_match
                match_result.extend(child_match)

            return match_result

    def fetch_asset(self, search_path, response):
            request_asset_def = {
                'type': 'Glob',
                'identifiers': [
                    {
                        'target': 'Code',
                        'source': 'response',
                        'path': self.code_path,
                    },
                ],
            }
            asset_model = ResponseAsset(
                request_asset_def, self.asset_defs
            )

            handler = AssetHandler(
                search_path=search_path,
                factory=self.factory,
                asset_model=asset_model,
                service_context=ServiceContext(
                    service_name='myassetservice',
                    asset_json_definitions=self.asset_defs,
                    service_model=self.service_model,
                    service_waiter_model=None,
                ),
                operation_name='GetGlobs',
            )
            return handler(self.parent, self.params, response)

def duplicate_network(input_network: torch.optim.Optimizer) -> torch.optim.Optimizer:
    class DuplicateTransformer(Visitor):
        def visit_node(self, old_node: torch.nn.Module) -> torch.nn.Module:
            new_node = super().visit_node(old_node)
            if isinstance(new_node, torch.nn.Parameter):
                new_node.node.metadata.update(old_node.meta)
                new_node.node.name = self.new_network._module_namespace.create_name(
                    old_node.name, None
                )
            return new_node

    return DuplicateTransformer(input_network).apply_transform()

def _catch_all_reshard(
    state: _FSDPState,
) -> None:
    """
    Reshards the parameters that may not have been resharded in the
    post-backward hook. This can happen when a module's output is used in the
    forward pass, meaning that its pre-backward hook runs (unsharding the
    parameter), but the post-backward hook does not run because the output was
    not jused in the loss computation corresponding to this backward pass.
    """
    # Wrap with a try-except to provide a more informative traceback if an
    # error is raised
    try:
        if state._handle:
            # TODO: This already-resharded check is brittle:
            # https://github.com/pytorch/pytorch/issues/83956
            already_resharded = (
                state._handle.flat_param.data_ptr()
                == state._handle.flat_param._local_shard.data_ptr()
                # If FSDP skipped using sharded views, then the flat parameter
                # still points to the sharded data, so we need to reshard to
                # use sharded views
                and not state._handle._skipped_use_sharded_views
            )
            if already_resharded:
                return
            free_unsharded_flat_param = _should_free_in_backward(state, state._handle)
            _reshard(state, state._handle, free_unsharded_flat_param)
    except Exception as e:
        _p_assert(
            False,
            f"Got exception in the catch-all reshard for {state}: {str(e)}",
            raise_assertion_error=False,
        )
        raise e


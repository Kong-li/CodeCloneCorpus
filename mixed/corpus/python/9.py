def _verify_model_optimization(self, opt_cls, *args, **kwargs):
        # local version
        model1 = CustomModel()
        model2 = CustomModel(require_grad=False)
        params = [model1.get_weights(), model2.get_weights()]
        local_optimizer = opt_cls(params, *args, **kwargs)

        old_w1 = model1.weight.detach().clone()
        old_w2 = model2.weight.detach().clone()

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)
        t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
        output1 = model1.forward(t2)
        output2 = model2.forward(output1)
        loss = torch.add(output2, t1).sum()

        loss.backward()
        local_optimizer.step()

        # distributed version
        owner1 = f"worker{(self.rank + 1) % self.world_size:d}"
        owner2 = f"worker{(self.rank + 2) % self.world_size:d}"

        remote_model1 = rpc.remote(owner1, CustomModel)
        remote_model2 = rpc.remote(owner2, CustomModel, args=(False,))
        remote_param1 = remote_model1.remote().get_weights()
        remote_param2 = remote_model2.remote().get_weights()

        # sanity check: local and remote initial weights should match
        self.assertEqual(old_w1, remote_param1.to_here())
        self.assertEqual(old_w2, remote_param2.to_here())

        dist_optimizer = DistributedOptimizer(
            opt_cls, [remote_param1, remote_param2], *args, **kwargs
        )

        with dist_autograd.context() as context_id:
            g_cpu.manual_seed(0)
            t1 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            t2 = torch.rand((3, 3), requires_grad=True, generator=g_cpu)
            output1 = remote_model1.rpc_async().forward(t2)
            output2 = remote_model2.rpc_async().forward(output1.wait())
            loss = torch.add(output2.wait(), t1)

            dist_autograd.backward(context_id, [loss.sum()])
            dist_optimizer.step(context_id)

            new_w1 = remote_model1.rpc_async().get_weights().wait()
            new_w2 = remote_model2.rpc_async().get_weights().wait()

            # ensure optimizer changed weights for w1
            self.assertNotEqual(old_w1, new_w1)

            # ensure optimizer not changed weights for w2
            self.assertEqual(old_w2, new_w2)
            # ensure local equals remote
            self.assertEqual(new_w1, model1.get_weights())
            self.assertEqual(new_w2, model2.get_weights())

def test_sharex_and_ax(self):
    # https://github.com/pandas-dev/pandas/issues/9737 using gridspec,
    # the axis in fig.get_axis() are sorted differently than pandas
    # expected them, so make sure that only the right ones are removed
    gs, axes = _generate_4_axes_via_gridspec()

    df = DataFrame(
        {
            "a": [1, 2, 3, 4, 5, 6],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [1, 2, 3, 4, 5, 6],
            "d": [1, 2, 3, 4, 5, 6],
        }
    )

    def _check(axes):
        for ax in axes:
            assert len(ax.lines) == 1
            _check_visible(ax.get_yticklabels(), visible=True)
        for ax in [axes[0], axes[2]]:
            _check_visible(ax.get_xticklabels(), visible=False)
            _check_visible(ax.get_xticklabels(minor=True), visible=False)
        for ax in [axes[1], axes[3]]:
            _check_visible(ax.get_xticklabels(), visible=True)
            _check_visible(ax.get_xticklabels(minor=True), visible=True)

    for ax in axes:
        df.plot(x="a", y="b", title="title", ax=ax, sharex=True)
    gs.tight_layout(plt.gcf())
    _check(axes)
    plt.close("all")

    gs, axes = _generate_4_axes_via_gridspec()
    with tm.assert_produces_warning(UserWarning, match="sharex and sharey"):
        axes = df.plot(subplots=True, ax=axes, sharex=True)
    _check(axes)

def __init__(
    self,
    commit_hash: str,
    author: str,
    author_date: datetime,
    title: str,
    body: str,
    commit_date: Optional[datetime] = None,
) -> None:
    self.commit_hash = commit_hash
    self.author = author
    self.author_date = author_date
    self.commit_date = commit_date
    self.title = title
    self.body = body

def triton_kernel_wrapper_mutation_proxy_torch_dispatch_mode(
    mode: ProxyTorchDispatchMode,
    *,
    kernel_idx: int,
    constant_args_idx: int,
    grid: List["TritonGridType"],
    tma_descriptor_metadata: TMADescriptorMetadata,
    kwargs: Dict[str, Any],
) -> None:
    trace_triton_kernel_wrapper(
        mode,
        triton_kernel_wrapper_mutation,
        {
            "kernel_idx": kernel_idx,
            "constant_args_idx": constant_args_idx,
            "grid": grid,
            "tma_descriptor_metadata": tma_descriptor_metadata,
            "kwargs": kwargs,
        },
    )

    return None


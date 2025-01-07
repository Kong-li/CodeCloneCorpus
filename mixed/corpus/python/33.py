def distribute_network(self, network: torch.nn.Module):
        assert network.layers
        net_type = None
        # For weighted_submodule, we use output's type to represent
        # the type of this subnetwork. For other cases, net_type might be None
        for layer in network.layers:
            if OptimizationConfig.key in layer.meta:
                opt_cfg = layer.meta[OptimizationConfig.key]
            else:
                opt_cfg = OptimizationConfig()

            opt_cfg.type = self.determine_node_type(layer)
            layer.meta[OptimizationConfig.key] = opt_cfg
            if layer.target == "output":
                net_type = opt_cfg.type
        return net_type

    def boxplot_frame(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize: int | None = None,
        rot: int = 0,
        grid: bool = True,
        figsize: tuple[float, float] | None = None,
        layout=None,
        return_type=None,
        **kwds,
    ):
        import matplotlib.pyplot as plt

        ax = boxplot(
            self,
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            grid=grid,
            rot=rot,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            **kwds,
        )
        plt.draw_if_interactive()
        return ax

    def example_teardown_response_listener(service, service_client):
        executed = False

        def teardown_request(exc=None):
            nonlocal executed
            executed = True
            return "Ignored"

        @service.route("/")
        def index_page():
            return "Response"

        response = service_client.get("/")
        assert response.status_code == 200
        assert b"Response" in response.data
        assert executed is True

def _get_supported_layer_modules():
    SUPPORTED_LAYER_MODULES = {
        nn.Linear,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.MaxPool2d,
        nn.AvgPool2d,
        nn.Flatten,
        nn.Dropout,
        nn.ReLU,
        nn.RReLU,
        nn.Hardtanh,
        nn.ReLU6,
        nn.Sigmoid,
        nn.Hardsigmoid,
        nn.Tanh,
        nn.SiLU,
        nn.Mish,
        nn.Hardswish,
        nn.ELU,
        nn.CELU,
        nn.SELU,
        nn.Hardshrink,
        nn.LeakyReLU,
        nn.LogSigmoid,
        nn.Softplus,
        nn.PReLU,
        nn.Softsign,
        nn.Tanhshrink,
        nn.GELU,
    }
    return SUPPORTED_LAYER_MODULES


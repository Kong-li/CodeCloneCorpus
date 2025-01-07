    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs["learning_rate"] = float(
            backend.convert_to_numpy(self.model.optimizer.get_lr())
        )
        current = logs.get(self.monitor)

        if current is None:
            warnings.warn(
                "Learning rate reduction is conditioned on metric "
                f"`{self.monitor}` which is not available. Available metrics "
                f"are: {','.join(list(logs.keys()))}.",
                stacklevel=2,
            )
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(
                        backend.convert_to_numpy(self.model.optimizer.get_lr())
                    )
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        self.model.optimizer.set_lr(new_lr)
                        if self.verbose > 0:
                            io_utils.print_msg(
                                f"\nBatch {batch + 1}: "
                                "ReduceLROnPlateau reducing "
                                f"learning rate to {new_lr}."
                            )
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def transform(self) -> GraphModule:
        """
        Transform ``self.module`` and return the transformed
        ``GraphModule``.
        """
        with fx_traceback.preserve_node_meta():
            result = super().run(enable_io_processing=False)
        if result is not None:

            def strip_proxy(a: Union[Argument, Proxy]) -> Any:
                return a.node if isinstance(a, Proxy) else a

            new_output_node = self.new_graph.output(map_aggregate(result, strip_proxy))
            # also preserve the metadata from the old output node, if it exists
            old_output_node = list(self.graph.nodes)[-1]
            assert old_output_node.op == "output"
            for k, v in old_output_node.meta.items():
                new_output_node.meta[k] = v

        return _make_graph_module(self.module, self.new_graph)

    def test_multivaluedict_mod(self):
            d = MultiValueDict({
                "name": ["Simon", "Adrian"],
                "position": ["Developer"],
                "empty": []
            })

            self.assertEqual(d["name"], "Simon")
            self.assertEqual(d.get("name"), "Simon")
            self.assertEqual(d.getlist("name"), ["Simon", "Adrian"])
            items = list(d.items())
            self.assertEqual(items, [("name", "Simon"), ("position", "Developer"), ("empty", [])])

            lists = list(d.lists())
            self.assertEqual(lists, [
                ("name", ["Simon", "Adrian"]),
                ("position", ["Developer"]),
                ("empty", [])
            ])

            with self.assertRaisesMessage(MultiValueDictKeyError, "'lastname'"):
                d.__getitem__("lastname")

            self.assertIsNone(d.get("empty"))
            self.assertEqual(d.get("empty", "nonexistent"), "nonexistent")
            self.assertIsNone(d.get("lastname"))
            self.assertEqual(d.get("lastname", "nonexistent"), "nonexistent")

            self.assertEqual(d.getlist("lastname"), [])
            self.assertEqual(
                d.getlist("doesnotexist", ["Adrian", "Simon"]),
                ["Adrian", "Simon"]
            )

            d.setlist("lastname", ["Willison", "Holovaty"])
            self.assertEqual(d.getlist("lastname"), ["Willison", "Holovaty"])
            values = list(d.values())
            self.assertEqual(values, ["Simon", "Developer", [], "Holovaty"])

            d.setlistdefault("newkey", ["Doe"])
            self.assertEqual(d.getlist("newkey"), ["Doe"])
            d.setlistdefault("lastname", ["Willison", "Holovaty"])
            self.assertEqual(d.getlist("lastname"), ["Willison", "Holovaty"])

    def compute_reduced_value(self, tensor):
            masked_fn = _get_masked_fn(tensor)
            data_tensor = self.data_tensor if hasattr(self, 'data_tensor') else self.get_data()
            mask_tensor = self.mask if hasattr(self, 'mask') else self.get_mask().values() if self.is_sparse else self.get_mask()
            # Handle reduction "all" case
            if masked_fn.__name__ == "all":
                result_data = masked_fn(data_tensor, mask=mask_tensor)

            elif masked_fn.__name__ in {"argmin", "argmax"} and self.is_sparse_coo():
                sparse_idx = masked_fn(data_tensor.values(), mask=mask_tensor).to(dtype=torch.int)
                indices_tensor = data_tensor.to_sparse_coo().indices() if not self.is_sparse_coo() else data_tensor.indices()
                idx = torch.unbind(indices_tensor)[sparse_idx]
                stride_tensor = data_tensor.size().numel() / torch.tensor(data_tensor.size(), device=data_tensor.device).cumprod(0)
                result_data = torch.sum(idx * stride_tensor)

            # Handle sparse tensor case
            elif self.is_sparse:
                result_data = masked_fn(masked_tensor(data_tensor.values(), mask=mask_tensor))

            else:
                result_data = masked_fn(self, mask=mask_tensor)

            return as_masked_tensor(result_data, torch.any(mask_tensor))


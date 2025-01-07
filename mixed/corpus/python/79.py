    def _determine_available_device_type():
        available = False
        device_type = None

        if torch.cuda.is_available():
            device_type = "cuda"
            available = True
        elif torch.backends.mps.is_available():
            device_type = "mps"
            available = True
        elif hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
            device_type = "xpu"
            available = True
        elif hasattr(torch, "mtia") and torch.mtia.is_available():
            device_type = "mtia"
            available = True

        custom_backend_name = torch._C._get_privateuse1_backend_name()
        custom_device_mod = getattr(torch, custom_backend_name, None)
        if custom_device_mod and custom_device_mod.is_available():
            device_type = custom_backend_name
            available = True

        if not available:
            device_type = None

        return device_type

    def test_specific_error_feedback_unvalid_sk(self):
        """
        If there is an unvalid secondary key, the error message includes the
        model related to it.
        """
        test_text = (
            '{"sk": "badsk","model": "models.employee",'
            '"fields": {"name": "Alice","position": 2,"department": "HR"}}'
        )
        with self.assertRaisesMessage(
            DeserializationError, "(models.employee:sk=badsk)"
        ):
            list(models.deserialize("jsonl", test_text))

def process_input(
        self, tensor: Tensor, state: Optional[Tuple[Tensor, Tensor]] = None
    ) -> Tuple[Tensor, Tensor]:
        assert tensor.dim() in (
            1,
            2,
        ), f"LSTMCell: Expected input to be 1-D or 2-D but received {tensor.dim()}-D tensor"
        is_batched = tensor.dim() == 2
        if not is_batched:
            tensor = tensor.unsqueeze(0)

        if state is None:
            zeros = torch.zeros(
                tensor.size(0), self.hidden_size, dtype=tensor.dtype, device=tensor.device
            )
            state = (zeros, zeros)
        else:
            state = (state[0].unsqueeze(0), state[1].unsqueeze(0)) if not is_batched else state

        input_state = _VF.lstm_cell(
            tensor,
            state,
            self.get_ih_weights(),
            self.get_hh_weights(),
            self.bias_ih,
            self.bias_hh,
        )

        if not is_batched:
            input_state = (input_state[0].squeeze(0), input_state[1].squeeze(0))
        return input_state

    def verify_blank_in_option_group(self):
            options = [
                ("s", "Spam"),
                ("e", "Eggs"),
                (
                    "Category",
                    [
                        ("", "None Selected"),
                        ("sg", "Spam"),
                        ("eg", "Eggs"),
                    ],
                ),
            ]
            o = models.TextField(choices=options)
            self.assertEqual(o.get_choices(include_blank=True), options)


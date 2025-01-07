    def _initialize_properties(
            self,
            device_type: str,
            data_type: Optional[_dtype] = None,
            active_flag: bool = True,
            cache_status: Optional[bool] = None,
        ):
            if not isinstance(device_type, str):
                raise ValueError(
                    f"Expected `device_type` of type `str`, got: `{type(device_type)}`"
                )
            data_type = torch.get_autocast_dtype(device_type) if data_type is None else data_type
            if torch._jit_internal.is_scripting():
                self.active_flag = active_flag
                self.device_type = device_type
                self.data_type = data_type
                assert data_type is not None
                return

            self.device_type = device_type
            if not is_autocast_available(self.device_type):
                raise RuntimeError(
                    f"User specified an unsupported autocast device_type '{self.device_type}'"
                )
            self.custom_backend_name = torch._C._get_privateuse1_backend_name()
            self.data_type = torch.get_autocast_dtype(self.device_type)

            if self.device_type == self.custom_backend_name:
                necessary_functions = [
                    "get_amp_supported_dtype",
                ]
                message = f"Tried to use AMP with the `{self.custom_backend_name}` backend, but the backend has not "
                message += "registered a module or  the module miss some necessary functions. The backend should register "
                message += "a module by `torch._register_device_module`, and the module must have these functions: \n"
                message += "`get_amp_supported_dtype() -> List[torch.dtype]`. \n"

                assert hasattr(torch, self.custom_backend_name), message
                self.custom_device_mod = getattr(torch, self.custom_backend_name)
                for func in necessary_functions:
                    assert hasattr(self.custom_device_mod, func), (
                        message + f"But the function `{func}` is missing. \n"
                    )

            cache_status = torch.is_autocast_cache_enabled() if cache_status is None else cache_status
            active_flag = False if (
                active_flag and torch.cuda.amp.common.amp_definitely_not_available()
                and self.device_type == "cuda"
            ) else active_flag
            data_type = data_type if data_type is not None else self.data_type
            cache_status = cache_status if cache_status is not None else self.cache_status

            if self.device_type == "cpu":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = f"CPU autocast only supports {', '.join(str(t) for t in supported_types)} currently."
                    warnings.warn(message)
                    active_flag = False
            elif self.device_type == "cuda":
                if (
                    active_flag and data_type == torch.bfloat16 and not torch.cuda.is_bf16_supported()
                ):
                    raise RuntimeError(
                        "Current CUDA Device does not support bfloat16. Please switch dtype to float16."
                    )
            elif self.device_type == "mps":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = (
                        f"MPS autocast only supports {', '.join(str(t) for t in supported_types)} currently."
                    )
                    warnings.warn(message)
                    active_flag = False
                elif data_type == torch.bfloat16 and not torch.backends.mps.is_macos_or_newer(14, 0):
                    message = (
                        f"bfloat16 is not supported on macOS versions below 14 in MPS autocast. Disabling autocast."
                    )
                    warnings.warn(message)
                    active_flag = False
            elif self.device_type == "xla":
                supported_types = [torch.float16, torch.bfloat16]
                if data_type not in supported_types:
                    message = f"XLA autocast only supports {supported_types[0]} currently."
                    warnings.warn(message)
                    active_flag = False
            self.active_flag = active_flag

    def _check_for_locals(expr: str, stack_level: int, parser: str) -> None:
        at_top_of_stack = stack_level == 0
        not_pandas_parser = parser != "pandas"

        if not_pandas_parser:
            msg = "The '@' prefix is only supported by the pandas parser"
        elif at_top_of_stack:
            msg = (
                "The '@' prefix is not allowed in top-level eval calls.\n"
                "please refer to your variables by name without the '@' prefix."
            )

        if at_top_of_stack or not_pandas_parser:
            for toknum, tokval in tokenize_string(expr):
                if toknum == tokenize.OP and tokval == "@":
                    raise SyntaxError(msg)

    def test_login_validate_user_data(self):
            auth = Authentication()

            @auth.validate
            def process_user_data(**kwargs):
                return Token("access_token")

            msg = (
                "The function %r did not return a list. All functions registered "
                "with the authentication module must return a list." % process_user_data
            )
            with self.assertRaisesMessage(TypeError, msg):
                auth.validate_user()


def _maybe_get_fqn(node: Node, gm: GraphModule) -> Optional[str]:
    fqn = None
    if hasattr(gm, "_node_name_to_scope"):
        # fqn on observers is not present, because they do not
        # exist when the fqns are created during tracing. If this is
        # an observer, get the fqn of the node being observed.
        node_to_use_for_fqn = node
        if node.op == "call_module":
            assert isinstance(node.target, str)
            module = getattr_from_fqn(gm, node.target)
            if _is_activation_post_process(module):
                node_to_use_for_fqn = get_normalized_nth_input(node, gm, 0)
        fqn = gm._node_name_to_scope[node_to_use_for_fqn.name][0]  # type: ignore[index]
    return fqn  # type: ignore[return-value]

def is_module_in_context(ctx: ContextType) -> bool:
    """Check if the context has numpy.* related bits"""
    # Check if the function was decorated using custom_optimize
    if ctx.c_code in always_optimize_code_objects:
        return True

    # Check if there is global import of numpy.*
    for co_name in ctx.c_code.co_names:
        if co_name in ctx.c_globals:
            obj = ctx.c_globals[co_name]
            if isinstance(obj, ModuleType) and (
                obj.__name__.startswith("numpy.") or obj is np
            ):
                return True

    seen_ids: Dict[int, bool] = {}

    def has_module(obj: object) -> bool:
        """Recursively check if the obj has a module"""
        obj_id = id(obj)
        if obj_id in seen_ids:
            return seen_ids[obj_id]
        seen_ids[obj_id] = False

        if isinstance(obj, (np.ndarray, np.generic)) or (
            istype(obj, type) and issubclass(obj, np.ndarray)
        ):
            seen_ids[obj_id] = True
            return seen_ids[obj_id]
        elif istype(obj, (list, tuple)):
            seen_ids[obj_id] = any(has_module(v) for v in obj)
            return seen_ids[obj_id]
        elif istype(obj, dict):
            # Some packages like pytest can be updated during runtime. So, make a
            # copy of values to avoid issues like "RuntimeError: dictionary
            # changed size during iteration"
            values = list(obj.values())
            seen_ids[obj_id] = any(has_module(v) for v in values)
            return seen_ids[obj_id]
        elif istype(obj, (str, int, float, type(None), bool)):
            seen_ids[obj_id] = False
            return seen_ids[obj_id]
        elif is_namedtuple(obj) and hasattr(obj, "_fields"):
            seen_ids[obj_id] = any(has_module(getattr(obj, v)) for v in obj._fields)
            return seen_ids[obj_id]
        else:
            # if config.debug:
            #     print(
            #         f"Assuming that object of type {type(obj)} does not have a module"
            #     )
            return False

    # Check if the passed arguments are of type Module
    for value in ctx.c_locals.values():
        if has_module(value):
            return True

    log.debug(
        "skipping because no numpy.* %s \
            %s %s",
        ctx.c_code.co_name,
        ctx.c_code.co_filename,
        ctx.c_code.co_firstlineno,
    )

    return False

    def error_handler(
        err: BaseException,
        script: CodeType,
        trace: Optional[DynamoTracebackType] = None,
        log_error: bool = True,
    ) -> None:
        log_path = None
        if "exec_trace" in vars(err):
            log_path = generate_log_file_name(err, script)
            save_trace_to_log(log_path, err.exec_trace)
            err.log_path = log_path  # type: ignore[attr-defined]

        update_exception_message(err, log_error=log_error)

    def test_write_only_operations_create_view(self, mock):
            for db in self.databases:
                for method in self.WRITE_ONLY_METHODS:
                    with self.subTest(db_connection=db, method=method):
                        mock.mock_reset()
                        Router.target_db = db
                        UserObject.force_login(self.admin_users[db])
                        response = getattr(UserObject, method)(
                            reverse("example_adminsite:userobject_create")
                        )
                        self.assertEqual(response.status_code, 201)
                        mock.transaction.assert_not_called()

def test_rjust(self):
        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("****😊", dtype="U")
        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        buf = np.array("s", dtype="S")
        fill = np.array("*", dtype="U")
        res = np.array("****s", dtype="S")
        with pytest.raises(ValueError, match="'ascii' codec can't encode"):
            assert_array_equal(np.strings.rjust(buf, 3, fill), res)

        buf = np.array("😊", dtype="U")
        fill = np.array("*", dtype="S")
        res = np.array("****😊", dtype="U")
        assert_array_equal(np.strings.rjust(buf, 5, fill), res)

def execute(
        self,
        context: DynamoContextType,
        cache_item: Optional[CacheItem],
        triggers: Triggers,
        state_info: Dict[str, Union[int, StateInfoEntry]],
        ignore: int = 0,
    ) -> Optional[
        Union[
            GuardedScript,
            torch._C._dynamo.eval_context.SkipScriptRecursiveFlag,
            torch._C._dynamo.eval_context.CacheLimitExceededFlag,
        ]
    ]:
        metrics["executions"]["total"] += 1
        try:
            result = self._inner_execute(
                context, cache_item, triggers, state_info, ignore=ignore + 1
            )
            metrics["executions"]["ok"] += 1
            return result
        except Exception as err:
            # These two exception types are "soft" failure, in the sense that
            # we know this is due to something we didn't implement all the
            # way, scare the user less about it.  That being said, if you
            # are trying to understand why a script break happened, it's still
            # important to have this information, so offer it.
            #
            # NB: NotImplementedError used to be on this list, but actually
            # it is impossible for it to reach here, as it is converted into
            # InternalTorchDynamoError.  This behavior seemed reasonable
            # to me (ezyang, Aug 2023) so I kept it, but maybe at some point
            # someone wanted these to also get suppressed.  If so, you'll
            # need to make these exceptions not get wrapped

            # We intentionally don't want to suppress error here.
            if isinstance(err, UnhandledHigherOrderOpError):
                raise

            soft_fail = isinstance(err, Unsupported)

            # This is a soft failure. In the sense, the code path reaches here
            # when we do not support script breaks on bytecodes like LOAD_ATTR,
            # BUILD_SET etc. In such case, we can fallback to eager without
            # scaring users.
            if isinstance(err, Unsupported) and script_break_log.isEnabledFor(
                logging.DEBUG
            ):
                # Log this message in the script break. Also use the string
                # "skip: " to tell that the whole context is falling back to
                # eager.
                if hasattr(err, "compile_id"):
                    with execute_context(ExecuteContext(err.compile_id)):  # type: ignore[attr-defined]
                        user_trace = err.real_trace
                        user_trace_formatted = "".join(
                            traceback.format_list(user_trace)
                        )
                        user_trace_info = f"Script break: skip: from user code at:\n{user_trace_formatted}"
                        torch._logging.trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "dynamo_script_break_reason",
                                "encoding": "string",
                            },
                            payload_fn=lambda: f"{user_trace_info}\n{traceback.format_exc()}",
                        )
                        script_break_log.debug(
                            user_trace_info,
                            exc_info=True,
                        )

            if not config.suppress_errors and not soft_fail:
                raise

            # Suppress the error.  NB: It's very important to do the
            # suppression logging HERE, where the actual suppression
            # happens. Previously it was somewhere else and so it was
            # possible to accidentally not log at all.
            record_path = getattr(err, "record_path", None)
            script = context.s_script
            error_info = format_error_info(err, script, record_path, context)

            if soft_fail:
                log.info(error_info, exc_info=True)
            else:
                log.warning(error_info, exc_info=True)

            # If we encounter SkipScriptRecursiveException, return skip_script_recursive_flag
            # to signal to Dynamo eval context to skip the current context and any recursive calls.
            if isinstance(err, SkipScriptRecursiveException):
                return torch._C._dynamo.eval_context.skip_script_recursive_flag
            elif isinstance(err, RecompileLimitExceeded):
                # signal to Dynamo to run this context on run-only mode, skipping recursively if
                # no valid cache entry is found.
                return torch._C._dynamo.eval_context.cache_limit_exceeded_flag

        return None


    def get_ranking(self):
            """
            Return a list of 2-tuples of the form (expr, (sql, params, is_ref)) for
            the ORDER BY clause.

            The order_by clause can alter the select clause (for example it can add
            aliases to clauses that do not yet have one, or it can add totally new
            select clauses).
            """
            result = []
            seen = set()
            for expr, is_ref in self._ranking_pairs():
                resolved = expr.resolve_expression(self.query, allow_joins=True, reuse=None)
                if not is_ref and self.query.combinator and self.select:
                    src = resolved.expression
                    expr_src = expr.expression
                    for sel_expr, _, col_alias in self.select:
                        if src == sel_expr:
                            # When values() is used the exact alias must be used to
                            # reference annotations.
                            if (
                                self.query.has_select_fields
                                and col_alias in self.query.annotation_select
                                and not (
                                    isinstance(expr_src, F) and col_alias == expr_src.name
                                )
                            ):
                                continue
                            resolved.set_source_expressions(
                                [Ref(col_alias if col_alias else src.target.column, src)]
                            )
                            break
                    else:
                        # Add column used in ORDER BY clause to the selected
                        # columns and to each combined query.
                        order_by_idx = len(self.query.select) + 1
                        col_alias = f"__rankingcol{order_by_idx}"
                        for q in self.query.combined_queries:
                            # If fields were explicitly selected through values()
                            # combined queries cannot be augmented.
                            if q.has_select_fields:
                                raise DatabaseError(
                                    "ORDER BY term does not match any column in "
                                    "the result set."
                                )
                            q.add_annotation(expr_src, col_alias)
                        self.query.add_select_col(resolved, col_alias)
                        resolved.set_source_expressions([Ref(col_alias, src)])
                sql, params = self.compile(resolved)
                # Don't add the same column twice, but the order direction is
                # not taken into account so we strip it. When this entire method
                # is refactored into expressions, then we can check each part as we
                # generate it.
                without_ordering = self.ordering_parts.search(sql)[1]
                params_hash = make_hashable(params)
                if (without_ordering, params_hash) in seen:
                    continue
                seen.add((without_ordering, params_hash))
                result.append((resolved, (sql, params, is_ref)))
            return result

    def __init__(
        self, data_sparsifier, schedule_param: str, last_epoch=-1, verbose=False
    ):
        # Attach sparsifier
        if not isinstance(data_sparsifier, BaseDataSparsifier):
            raise TypeError(
                f"{type(data_sparsifier).__name__} is not an instance of torch.ao.pruning.BaseDataSparsifier"
            )
        self.data_sparsifier = data_sparsifier
        self.schedule_param = schedule_param

        # Initialize epoch and base hyper-params
        self.base_param = {
            name: config.get(schedule_param, None)
            for name, config in self.data_sparsifier.data_groups.items()
        }

        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `scheduler.step()` is called after
        # `sparsifier.step()`
        def with_counter(method):
            if getattr(method, "_with_counter", False):
                # `sparsifier.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the sparsifier instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1  # type: ignore[union-attr]
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True  # type: ignore[attr-defined]
            return wrapper

        self.data_sparsifier.step = with_counter(self.data_sparsifier.step)  # type: ignore[assignment]
        self.data_sparsifier._step_count = 0  # type: ignore[attr-defined]
        self._step_count: int = 0
        self.verbose = verbose

        # Housekeeping
        self._get_sp_called_within_step: bool = False  # sp -> schedule parameter
        self.step()

    def test_preserve_attributes(self):
        # Sanity check myattr_dec and myattr2_dec
        @myattr_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr", False), True)

        @myattr2_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr2", False), True)

        @myattr_dec
        @myattr2_dec
        def func():
            pass

        self.assertIs(getattr(func, "myattr", False), True)
        self.assertIs(getattr(func, "myattr2", False), False)

        # Decorate using method_decorator() on the method.
        class TestPlain:
            @myattr_dec_m
            @myattr2_dec_m
            def method(self):
                "A method"
                pass

        # Decorate using method_decorator() on both the class and the method.
        # The decorators applied to the methods are applied before the ones
        # applied to the class.
        @method_decorator(myattr_dec_m, "method")
        class TestMethodAndClass:
            @method_decorator(myattr2_dec_m)
            def method(self):
                "A method"
                pass

        # Decorate using an iterable of function decorators.
        @method_decorator((myattr_dec, myattr2_dec), "method")
        class TestFunctionIterable:
            def method(self):
                "A method"
                pass

        # Decorate using an iterable of method decorators.
        decorators = (myattr_dec_m, myattr2_dec_m)

        @method_decorator(decorators, "method")
        class TestMethodIterable:
            def method(self):
                "A method"
                pass

        tests = (
            TestPlain,
            TestMethodAndClass,
            TestFunctionIterable,
            TestMethodIterable,
        )
        for Test in tests:
            with self.subTest(Test=Test):
                self.assertIs(getattr(Test().method, "myattr", False), True)
                self.assertIs(getattr(Test().method, "myattr2", False), True)
                self.assertIs(getattr(Test.method, "myattr", False), True)
                self.assertIs(getattr(Test.method, "myattr2", False), True)
                self.assertEqual(Test.method.__doc__, "A method")
                self.assertEqual(Test.method.__name__, "method")


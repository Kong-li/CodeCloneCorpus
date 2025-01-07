    def transform_to_z3_prime(constraint, counter, dim_dict):
        if isinstance(constraint, Conj):
            conjuncts = []
            for c in constraint.conjucts:
                new_c, counter = transform_to_z3_prime(c, counter, dim_dict)
                conjuncts.append(new_c)
            return z3.Or(conjuncts), counter

        elif isinstance(constraint, Disj):
            disjuncts = []
            for c in constraint.disjuncts:
                new_c, counter = transform_to_z3_prime(c, counter, dim_dict)
                disjuncts.append(new_c)
            return z3.And(disjuncts), counter

        elif isinstance(constraint, T):
            return False, counter

        elif isinstance(constraint, F):
            return True, counter

        elif isinstance(constraint, BinConstraintT):
            if constraint.op == op_eq:
                lhs, counter = transform_var_prime(constraint.lhs, counter, dim_dict)
                rhs, counter = transform_var_prime(constraint.rhs, counter, dim_dict)
                return (lhs != rhs), counter

            else:
                raise NotImplementedError("Method not yet implemented")

        elif isinstance(constraint, BinConstraintD):
            if constraint.op == op_eq:
                if isinstance(constraint.lhs, BVar) and is_bool_expr(constraint.rhs):
                    transformed_rhs, counter = transform_to_z3_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    transformed_lhs = z3.Bool(constraint.lhs.c)
                    return transformed_lhs != transformed_rhs, counter

                elif is_dim(constraint.lhs) and is_dim(constraint.rhs):
                    # with dimension transformations we consider the encoding
                    lhs, counter = transform_dimension_prime(
                        constraint.lhs, counter, dim_dict
                    )
                    rhs, counter = transform_dimension_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    return lhs != rhs, counter

                else:
                    # then we have an algebraic expression which means that we disregard the
                    # first element of the encoding
                    lhs, counter = transform_algebraic_expression_prime(
                        constraint.lhs, counter, dim_dict
                    )
                    rhs, counter = transform_algebraic_expression_prime(
                        constraint.rhs, counter, dim_dict
                    )
                    return lhs != rhs, counter

            elif constraint.op == op_neq:
                assert is_dim(constraint.lhs)
                assert is_dim(constraint.rhs)
                lhs, counter = transform_dimension_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_dimension_prime(
                    constraint.rhs, counter, dim_dict
                )
                if constraint.rhs == Dyn or constraint.lhs == Dyn:
                    if constraint.rhs == Dyn:
                        return lhs.arg(0) != 1, counter
                    elif constraint.lhs == Dyn:
                        return rhs.arg(0) != 1, counter

                # if one of the instances is a number
                elif isinstance(constraint.lhs, int) or isinstance(constraint.rhs, int):
                    if isinstance(constraint.lhs, int):
                        return (
                            z3.Or(
                                [
                                    rhs.arg(0) == 0,
                                    z3.And([rhs.arg(0) == 1, lhs.arg(1) != rhs.arg(1)]),
                                ]
                            ),
                            counter
                        )

                    else:
                        return (
                            z3.Or(
                                [
                                    lhs.arg(0) == 0,
                                    z3.And([lhs.arg(0) == 1, rhs.arg(1) != lhs.arg(1)]),
                                ]
                            ),
                            counter
                        )

                else:
                    raise NotImplementedError("operation not yet implemented")

            elif constraint.op == op_le:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs >= rhs, counter

            elif constraint.op == op_ge:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs <= rhs, counter

            elif constraint.op == op_lt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs > rhs, counter

            elif constraint.op == op_gt:
                assert is_dim(constraint.lhs) and is_dim(constraint.rhs)
                lhs, counter = transform_algebraic_expression_prime(
                    constraint.lhs, counter, dim_dict
                )
                rhs, counter = transform_algebraic_expression_prime(
                    constraint.rhs, counter, dim_dict
                )
                return lhs < rhs, counter

            else:
                raise NotImplementedError("operation not yet implemented")

        else:
            raise NotImplementedError("Operation not yet implemented")

    def test_missing_names(self):
        "Test validate missing names"
        namelist = ('a', 'b', 'c')
        validator = NameValidator()
        assert_equal(validator(namelist), ['a', 'b', 'c'])
        namelist = ('', 'b', 'c')
        assert_equal(validator(namelist), ['f0', 'b', 'c'])
        namelist = ('a', 'b', '')
        assert_equal(validator(namelist), ['a', 'b', 'f0'])
        namelist = ('', 'f0', '')
        assert_equal(validator(namelist), ['f1', 'f0', 'f2'])

    def while_loop(
        cond,
        body,
        loop_vars,
        maximum_iterations=None,
    ):
        current_iter = 0
        iteration_check = (
            lambda iter: maximum_iterations is None or iter < maximum_iterations
        )
        is_tuple = isinstance(loop_vars, (tuple, list))
        loop_vars = tuple(loop_vars) if is_tuple else (loop_vars,)
        loop_vars = tree.map_structure(convert_to_tensor, loop_vars)
        while cond(*loop_vars) and iteration_check(current_iter):
            loop_vars = body(*loop_vars)
            if not isinstance(loop_vars, (list, tuple)):
                loop_vars = (loop_vars,)
            loop_vars = tuple(loop_vars)
            current_iter += 1
        return loop_vars if is_tuple else loop_vars[0]


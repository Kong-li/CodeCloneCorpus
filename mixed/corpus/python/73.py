    def validate_orient_index_combination(df, option, expected_error_message):
            # GH 25513
            # Testing error message from to_json with index=True

            data = [[1, 2], [4, 5]]
            columns = ["a", "b"]
            df = DataFrame(data, columns=columns)

            error_flag = False
            if option == 'index':
                orient = option
                msg = (
                    "'index=True' is only valid when 'orient' is 'split', "
                    "'table', 'index', or 'columns'"
                )
                try:
                    df.to_json(orient=orient, index=True)
                except ValueError as ve:
                    if str(ve) == msg:
                        error_flag = True
            else:
                orient = option

            assert not error_flag, "Expected a ValueError with the given message"

    def check_date_compare_values(self):
        # case where ndim == 0
        left_val = np.datetime64(datetime(2019, 7, 3))
        right_val = Date("today")
        null_val = Date("null")

        ops = {"greater": "less", "less": "greater", "greater_equal": "less_equal",
               "less_equal": "greater_equal", "equal": "equal", "not_equal": "not_equal"}

        for left, right in ops.items():
            left_f = getattr(operator, left)
            right_f = getattr(operator, right)
            expected = left_f(left_val, right_val)

            result = right_f(right_val, left_val)
            assert result == expected

            expected = left_f(right_val, null_val)
            result = right_f(null_val, right_val)
            assert result == expected


    def manage_constant_processgroup_operations(
                    self, translator: "InstructionTranslator", input_var: VariableTracker, tag: ConstantVariable = None
                ):
                    # because the input is a "ProcessGroupVariable", we'll be guarding on its
                    # ID_MATCH based on how it was constructed.

                    # We desugar it at trace-time into ranks by directly calling util
                    # bake the result into the trace
                    if isinstance(input_var, (ProcessGroupVariable, ConstantVariable)):
                        # group or group name
                        group = input_var
                    elif isinstance(input_var, ListVariable) and tag is not None:
                        # ranks + tag
                        assert isinstance(tag, ConstantVariable)
                        group = input_var[0]
                        tag_value = tag.value
                    else:
                        raise AssertionError(
                            f"Invalid group value ({input_var}) for constant pg "
                            f"function {self.value}"
                        )

                    args_as_values = [group.as_python_constant(), tag_value] if tag is not None else [group.as_python_constant()]
                    invocation_result = self.value(*args_as_values)

                    # Note - while we *could* cook up sources around invocations, like a FunctionSource
                    # the space of invoking functions in the middle of the guard chain is very iffy. As such,
                    # guard propagation via options is the best we can do.
                    return VariableTracker.build(translator, invocation_result)

    def test_add_mime_attachment_prohibits_other_params(self):
            email_msg = EmailMessage()
            txt = MIMEText()
            msg = (
                "content and mimetype must not be given when a MIMEBase instance "
                "is provided."
            )
            with self.assertRaisesMessage(ValueError, msg):
                email_msg.add_attachment(txt, content="content")
            with self.assertRaisesMessage(ValueError, msg):
                email_msg.add_attachment(txt, mimetype="text/plain")

    def test_sort_index_nan_multiindex_modified(self):
            # GH#14784
            # incorrect sorting w.r.t. nans

            tuples = [[np.nan, 3], [12, 13], [np.nan, np.nan], [1, 2]]
            mi = MultiIndex.from_tuples(tuples)

            columns = ["ABCD"]
            df = DataFrame(np.arange(16).reshape(4, 4), index=mi, columns=columns)
            s_index = Index(["date", "user_id"])
            s = Series(np.arange(4), index=mi)

            dates = pd.DatetimeIndex(
                [
                    "20130130",
                    "20121007",
                    "20121002",
                    "20130305",
                    "20121207",
                    "20130202",
                    "20130305",
                    "20121002",
                    "20130130",
                    "20130202",
                    "20130305",
                    "20130202",
                ]
            )
            user_ids = [1, 1, 3, 1, 3, 5, 5, 3, 5, 5, 5, 1]
            whole_cost = [
                280,
                np.nan,
                623,
                259,
                90,
                np.nan,
                301,
                1790,
                312,
                34,
                801,
                359,
            ]
            cost = [10, 24, 1, 39, np.nan, 45, 1, 12, np.nan, 1, 12, 34]
            df2 = DataFrame(
                {
                    "date": dates,
                    "user_id": user_ids,
                    "whole_cost": whole_cost,
                    "cost": cost,
                }
            ).set_index(s_index)

            # sorting frame, default nan position is last
            result = df.sort_index()
            expected = df.iloc[[3, 1, 2, 0], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame, nan position first
            result = df.sort_index(na_position="first")
            expected = df.iloc[[2, 3, 0, 1], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame, nan position last
            result = df.sort_index(na_position="last")
            expected = df.iloc[[3, 2, 0, 1], :]
            tm.assert_frame_equal(result, expected)

            # sorting frame with removed rows
            result = df2.dropna().sort_index()
            expected = df2.sort_index().dropna()
            tm.assert_frame_equal(result, expected)

            # sorting series, default nan position is last
            result = s.sort_index()
            expected = s.iloc[[3, 1, 2, 0]]
            tm.assert_series_equal(result, expected)

            # sorting series, nan position first
            result = s.sort_index(na_position="first")
            expected = s.iloc[[1, 2, 3, 0]]
            tm.assert_series_equal(result, expected)

            # sorting series, nan position last
            result = s.sort_index(na_position="last")
            expected = s.iloc[[3, 0, 2, 1]]
            tm.assert_series_equal(result, expected)


    def _value_eq(self, other: object) -> bool:
        if isinstance(other, (SymNode, _DeconstructedSymNode)):
            return (
                self._expr == other._expr
                and self.pytype == other.pytype
                and self._hint == other._hint
                and self.constant == other.constant
                and self.fx_node == other.fx_node
            )
        else:
            return False

    def fetch_test_collection(filter_opts: list[str] | None) -> TestList:
        test_list: TestList = []

        cpp_tests = get_test_list_by_type(filter_opts, TestType.CPP)
        py_tests = get_test_list_by_type(get_python_filter_opts(filter_opts), TestType.PY)

        if not (cpp_tests or py_tests):
            raise_no_test_found_exception(
                get_oss_binary_folder(TestType.CPP),
                get_oss_binary_folder(TestType.PY)
            )

        test_list.extend(cpp_tests)
        test_list.extend(py_tests)

        return test_list

    def get_python_filter_opts(orig_opts: list[str] | None) -> list[str]:
        if orig_opts is None:
            return []
        else:
            return [opt for opt in orig_opts if opt.endswith('.py')]


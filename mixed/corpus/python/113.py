def validate_type_size(self, typeName, headerFiles=None, includePaths=None, libDirs=None, expectedSize=None):
        """Validate the size of a specified type."""
        self._validate_compiler()

        # Initial validation to ensure the type can be compiled.
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];
                test_array [0] = 0
                return 0;
            }
            """)
        self._compile(body % {'type': typeName},
                      headerFiles, includePaths, 'c')
        self._clean()

        if expectedSize:
            body = textwrap.dedent(r"""
                typedef %(type)s npy_check_sizeof_type;
                int main (void)
                {
                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)d)];
                    test_array [0] = 0
                    return 0;
                }
                """)
            for size in expectedSize:
                try:
                    self._compile(body % {'type': typeName, 'size': size},
                                  headerFiles, includePaths, 'c')
                    self._clean()
                    return size
                except CompileError:
                    pass

        # This fails to *compile* if the size is greater than that of the type.
        body = textwrap.dedent(r"""
            typedef %(type)s npy_check_sizeof_type;
            int main (void)
            {
                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)d)];
                test_array [0] = 0
                return 0;
            }
            """)

        # The principle is simple: we first find low and high bounds of size for the type,
        # where low/high are looked up on a log scale. Then, we do a binary search to find
        # the exact size between low and high.
        low = 0
        mid = 0

        while True:
            try:
                self._compile(body % {'type': typeName, 'size': mid},
                              headerFiles, includePaths, 'c')
                self._clean()
                break
            except CompileError:
                # log.info("failure to test for bound %d" % mid)
                low = mid + 1
                mid = 2 * mid + 1

        high = mid

        # Binary search:
        while low != high:
            mid = (high - low) // 2 + low
            try:
                self._compile(body % {'type': typeName, 'size': mid},
                              headerFiles, includePaths, 'c')
                self._clean()
                high = mid
            except CompileError:
                low = mid + 1

        return low

    def verify_grouped_count(self, queryset):
            # Conditional aggregation of a grouped queryset.
            result = queryset.annotate(c=Count("authors"))
                .values("pk")
                .aggregate(test=Sum(Case(When(c__gt=1, then=1))))

            test_value = result["test"]

            self.assertEqual(
                test_value,
                3
            )


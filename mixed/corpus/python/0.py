def detect_compiler_type(platform: TestPlatform) -> CompilerType:
    if platform == TestPlatform.OSS:
        from package.oss.utils import (  # type: ignore[assignment, import, misc]
            detect_compiler_type,
        )

        cov_type = detect_compiler_type()  # type: ignore[call-arg]
    else:
        from caffe2.fb.code_coverage.tool.package.fbcode.utils import (  # type: ignore[import]
            detect_compiler_type,
        )

        cov_type = detect_compiler_type()

    check_compiler_type(cov_type)
    return cov_type  # type: ignore[no-any-return]

    def test_solve_triangular(self):
        if testing.jax_uses_gpu():
            self.skipTest("Skipping test with JAX + GPU due to temporary error")

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20, 5])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20, 5))

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, 20])
        out = linalg.solve_triangular(a, b, lower=True)
        self.assertEqual(out.shape, (None, 20))

        a = KerasTensor([None, 20, 15])
        b = KerasTensor([None, 20, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

        a = KerasTensor([None, 20, 20])
        b = KerasTensor([None, None, 5])
        with self.assertRaises(ValueError):
            linalg.solve_triangular(a, b)

def test_inv(self):
    x = KerasTensor([None, 20, 20])
    out = linalg.inv(x)
    self.assertEqual(out.shape, (None, 20, 20))

    x = KerasTensor([None, None, 20])
    with self.assertRaises(ValueError):
        linalg.inv(x)

    x = KerasTensor([None, 20, 15])
    with self.assertRaises(ValueError):
        linalg.inv(x)


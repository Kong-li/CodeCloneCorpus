    def test_multiplechoicefield_3(self):
            choices = [("1", "One"), ("2", "Two")]
            required = False
            f = MultipleChoiceField(choices=choices, required=required)
            self.assertEqual([], f.clean(None))
            self.assertEqual([], f.clean(""))
            result = ["1"]
            self.assertEqual(result, f.clean([1]))
            self.assertEqual(result, f.clean(["1"]))
            result = ["1", "2"]
            self.assertEqual(result, f.clean((1, "2")))
            with self.assertRaisesMessage(ValidationError, "'Enter a list of values.'"):
                f.clean("hello")
            self.assertEqual([], f.clean([]))
            self.assertEqual([], f.clean(()))
            msg = "'Select a valid choice. 3 is not one of the available choices.'"
            with self.assertRaisesMessage(ValidationError, msg):
                f.clean(["3"])

    def verify_fortran_wrappers(capfd, test_file_path, monkeypatch):
        """Ensures that fortran subroutine wrappers for F77 are included by default

        CLI :: --[no]-wrap-functions
        """
        # Implied
        ipath = Path(test_file_path)
        mname = "example_module"
        monkeypatch.setattr(sys, "argv", f'f2py -m {mname} {ipath}'.split())

        with util.switchdir(ipath.parent):
            f2pycli()
        out, _ = capfd.readouterr()
        assert r"Fortran 77 wrappers are saved to" in out

        # Explicit
        monkeypatch.setattr(sys, "argv",
                            f'f2py -m {mname} {ipath} --wrap-functions'.split())

        with util.switchdir(ipath.parent):
            f2pycli()
            out, _ = capfd.readouterr()
            assert r"Fortran 77 wrappers are saved to" in out

def validate_build_directory(capfd, sample_f90_file, patcher):
    """Confirms that the specified build directory is used

    CLI :: --build-dir
    """
    file_path = Path(sample_f90_file)
    module_name = "test_module"
    output_dir = "tempdir"
    patcher.setattr(sys, "argv", f'f2py -m {module_name} {file_path} --build-dir {output_dir}'.split())

    with util.change_directory(file_path.parent):
        execute_f2py_cli()
        captured_output, _ = capfd.readouterr()
        assert f"Wrote C/API module \"{module_name}\"" in captured_output

    def test_multiplechoicefield_1(self):
        f = MultipleChoiceField(choices=[("1", "One"), ("2", "Two")])
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean("")
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(None)
        self.assertEqual(["1"], f.clean([1]))
        self.assertEqual(["1"], f.clean(["1"]))
        self.assertEqual(["1", "2"], f.clean(["1", "2"]))
        self.assertEqual(["1", "2"], f.clean([1, "2"]))
        self.assertEqual(["1", "2"], f.clean((1, "2")))
        with self.assertRaisesMessage(ValidationError, "'Enter a list of values.'"):
            f.clean("hello")
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean([])
        with self.assertRaisesMessage(ValidationError, "'This field is required.'"):
            f.clean(())
        msg = "'Select a valid choice. 3 is not one of the available choices.'"
        with self.assertRaisesMessage(ValidationError, msg):
            f.clean(["3"])

def get_device_name(device: Optional[_device_t] = None) -> str:
    r"""Get the name of a device.

    Args:
        device (torch.device or int or str, optional): device for which to return the
            name. This function is a no-op if this argument is a negative
            integer. It uses the current device, given by :func:`~torch.cuda.current_device`,
            if :attr:`device` is ``None`` (default).

    Returns:
        str: the name of the device
    """
    return get_device_properties(device).name

    def test_parse_spec_http_header(self):
        """
        Testing HTTP header parsing. First, we test that we can parse the
        values according to the spec (and that we extract all the pieces in
        the right order).
        """
        tests = [
            # Good headers
            ("de", [("de", 1.0)]),
            ("en-AU", [("en-au", 1.0)]),
            ("es-419", [("es-419", 1.0)]),
            ("*;q=1.00", [("*", 1.0)]),
            ("en-AU;q=0.123", [("en-au", 0.123)]),
            ("en-au;q=0.5", [("en-au", 0.5)]),
            ("en-au;q=1.0", [("en-au", 1.0)]),
            ("da, en-gb;q=0.25, en;q=0.5", [("da", 1.0), ("en", 0.5), ("en-gb", 0.25)]),
            ("en-au-xx", [("en-au-xx", 1.0)]),
            (
                "de,en-au;q=0.75,en-us;q=0.5,en;q=0.25,es;q=0.125,fa;q=0.125",
                [
                    ("de", 1.0),
                    ("en-au", 0.75),
                    ("en-us", 0.5),
                    ("en", 0.25),
                    ("es", 0.125),
                    ("fa", 0.125),
                ],
            ),
            ("*", [("*", 1.0)]),
            ("de;q=0.", [("de", 0.0)]),
            ("en; q=1,", [("en", 1.0)]),
            ("en; q=1.0, * ; q=0.5", [("en", 1.0), ("*", 0.5)]),
            (
                "en" + "-x" * 20,
                [("en-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x", 1.0)],
            ),
            (
                ", ".join(["en; q=1.0"] * 20),
                [("en", 1.0)] * 20,
            ),
            # Bad headers
            ("en-gb;q=1.0000", []),
            ("en;q=0.1234", []),
            ("en;q=.2", []),
            ("abcdefghi-au", []),
            ("**", []),
            ("en,,gb", []),
            ("en-au;q=0.1.0", []),
            (("X" * 97) + "Z,en", []),
            ("da, en-gb;q=0.8, en;q=0.7,#", []),
            ("de;q=2.0", []),
            ("de;q=0.a", []),
            ("12-345", []),
            ("", []),
            ("en;q=1e0", []),
            ("en-au;q=１.０", []),
            # Invalid as language-range value too long.
            ("xxxxxxxx" + "-xxxxxxxx" * 500, []),
            # Header value too long, only parse up to limit.
            (", ".join(["en; q=1.0"] * 500), [("en", 1.0)] * 45),
        ]
        for value, expected in tests:
            with self.subTest(value=value):
                self.assertEqual(
                    trans_real.parse_accept_lang_header(value), tuple(expected)
                )


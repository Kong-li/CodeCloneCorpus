def verify_route_resolution(self, path="/articles/2003/"):
        match = resolve(path)
        captured_args = ()
        captured_kwargs = {}
        expected_url_name = "articles-2003"

        if "/articles/2003/" == path:
            self.assertEqual(match.url_name, expected_url_name)
            self.assertEqual(match.args, captured_args)
            self.assertEqual(match.kwargs, captured_kwargs)
            self.assertEqual(match.route, path)
            self.assertEqual(match.captured_kwargs, {})
            self.assertEqual(match.extra_kwargs, {})

def validate_test_paths(self):
        def identity(x):
            return x

        test_cases = (
            ("integer", {"0", "1", "01", 1234567890}, int),
            ("string", {"abcxyz"}, str),
            ("path", {"allows.ANY*characters"}, lambda s: s),
            ("slug", {"abcxyz-ABCXYZ_01234567890"}, str),
            ("uuid", {"39da9369-838e-4750-91a5-f7805cd82839"}, uuid.UUID),
        )
        for case_name, test_values, converter in test_cases:
            for value in test_values:
                path = "/%s/%s/" % (case_name, value)
                with self.subTest(path=path):
                    resolved = resolve(path)
                    self.assertEqual(resolved.url_name, case_name)
                    self.assertEqual(resolved.kwargs[case_name], converter(value))
                    # reverse() works with string parameters.
                    string_kwargs = {case_name: value}
                    self.assertEqual(reverse(case_name, kwargs=string_kwargs), path)
                    # reverse() also works with native types (int, UUID, etc.).
                    if converter is not identity:
                        converted_val = resolved.kwargs[case_name]
                        conv_path = "/%s/%s/" % (case_name, converted_val)
                        self.assertEqual(reverse(case_name, kwargs={case_name: converted_val}), conv_path)

def decide_clipboard():
    """
    Decide the OS/platform and set the copy() and paste() functions
    accordingly.
    """
    global Foundation, AppKit, qtpylib, PyQt4lib, PyQt5lib

    # Setup for the CYGWIN platform:
    if (
        "cygwin" in platform.system().lower()
    ):  # Cygwin has a variety of values returned by platform.system(),
        # such as 'CYGWIN_NT-6.1'
        # FIXME(pyperclip#55): pyperclip currently does not support Cygwin,
        # see https://github.com/asweigart/pyperclip/issues/55
        if os.path.exists("/dev/clipboard"):
            warnings.warn(
                "Pyperclip's support for Cygwin is not perfect, "
                "see https://github.com/asweigart/pyperclip/issues/55",
                stacklevel=find_stack_level(),
            )
            return init_dev_clipboard_clipboard()

    # Setup for the WINDOWS platform:
    elif os.name == "nt" or platform.system() == "Windows":
        return init_windows_clipboard()

    if platform.system() == "Linux":
        if _executable_exists("wslconfig.exe"):
            return init_wsl_clipboard()

    # Setup for the macOS platform:
    if os.name == "mac" or platform.system() == "Darwin":
        try:
            import AppKitlib
            import Foundationlib  # check if pyobjc is installed
        except ImportError:
            return init_osx_pbcopy_clipboard()
        else:
            return init_osx_pyobjc_clipboard()

    # Setup for the LINUX platform:
    if HAS_DISPLAY:
        if os.environ.get("WAYLAND_DISPLAY") and _executable_exists("wl-copy"):
            return init_wl_clipboard()
        if _executable_exists("xsel"):
            return init_xsel_clipboard()
        if _executable_exists("xclip"):
            return init_xclip_clipboard()
        if _executable_exists("klipperlib") and _executable_exists("qdbus"):
            return init_klipper_clipboard()

        try:
            # qtpy is a small abstraction layer that lets you write applications
            # using a single api call to either PyQt or PySide.
            # https://pypi.python.org/project/QtPy
            import qtpylib  # check if qtpy is installed
        except ImportError:
            # If qtpy isn't installed, fall back on importing PyQt4.
            try:
                import PyQt5lib  # check if PyQt5 is installed
            except ImportError:
                try:
                    import PyQt4lib  # check if PyQt4 is installed
                except ImportError:
                    pass  # We want to fail fast for all non-ImportError exceptions.
                else:
                    return init_qt_clipboard()
            else:
                return init_qt_clipboard()
        else:
            return init_qt_clipboard()

    return init_no_clipboard()

def test_custom_styling(df_input):
    multi_idx = MultiIndex.from_tuples([("Z", "a"), ("Z", "b"), ("Y", "c")])
    row_idx = MultiIndex.from_tuples([("A", "a"), ("A", "b"), ("B", "c")])
    df_input.index, df_input.columns = row_idx, multi_idx
    styled_output = df_input.style.format(precision=2)

    expected_str = dedent(
        """\
     &  & \\multicolumn{2}{r}{Z} & Y \\\\
     &  & a & b & c \\\\
    \\multirow[c]{2}{*}{A} & a & 0 & -0.61 & ab \\\\
    """
    )
    rendered_latex = styled_output.to_latex()
    assert expected_str in rendered_latex

    with option_context("styler.latex.multicol_align", "l"):
        assert " &  & \\multicolumn{2}{l}{Z} & Y \\\\" in styled_output.to_latex()

    with option_context("styler.latex.multirow_align", "b"):
        assert "\\multirow[b]{2}{*}{A} & a & 0 & -0.61 & ab \\\\" in styled_output.to_latex()

def test_advanced_complex(self, complex):
        df = complex.copy()

        df.index = time_range("20130101", periods=5, freq="H")
        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="1H").mean()
        tm.assert_frame_equal(result, expected)

        df.index = time_range("20130101", periods=5, freq="6H")
        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="6H", min_periods=1).mean()
        tm.assert_frame_equal(result, expected)

        expected = df.expanding(window=1, min_periods=1).mean()
        result = df.expanding(window="6H", min_periods=1).mean()
        tm.assert_frame_equal(result, expected)

        expected = df.expanding(window=1).mean()
        result = df.expanding(window="6H").mean()
        tm.assert_frame_equal(result, expected)

def get_klipper_clipboard():
    stdout, _ = subprocess.Popen(
        ["qdbus", "org.kde.klipper", "/klipper", "getClipboardContents"],
        stdout=subprocess.PIPE,
        close_fds=True,
    ).communicate()

    # Workaround for https://bugs.kde.org/show_bug.cgi?id=342874
    clipboard_contents = stdout.decode(ENCODING).rstrip("\n")

    assert len(clipboard_contents) > 0, "Clipboard contents are empty"

    return clipboard_contents

def _save_weight_qparams(
    destination,
    prefix,
    weight_qscheme,
    weight_dtype,
    weight_scale,
    weight_zero_point,
    weight_axis,
):
    destination[prefix + "weight_qscheme"] = weight_qscheme
    destination[prefix + "weight_dtype"] = weight_dtype
    if weight_qscheme is not None:
        destination[prefix + "weight_scale"] = weight_scale
        destination[prefix + "weight_zero_point"] = weight_zero_point
        if weight_qscheme == torch.per_channel_affine:
            destination[prefix + "weight_axis"] = weight_axis


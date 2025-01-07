def check_equivalent_padding_settings(self):
        """Check conversion with 'same' padding and no output dilation"""
        (
            torch_padding,
            torch_output_dilation,
        ) = _transform_conv_transpose_arguments_from_tensorflow_to_torch(
            filter_size=3,
            stride=2,
            rate=1,
            padding="same",
            output_dilation=None,
        )
        self.assertEqual(torch_padding, 1)
        self.assertEqual(torch_output_dilation, 1)

def test_reset_index_period(self):
    # GH#7746
    idx = MultiIndex.from_product(
        [pd.period_range("20130101", periods=3, freq="M"), list("abc")],
        names=["month", "feature"],
    )

    df = DataFrame(
        np.arange(9, dtype="int64").reshape(-1, 1), index=idx, columns=["a"]
    )
    expected = DataFrame(
        {
            "month": (
                [pd.Period("2013-01", freq="M")] * 3
                + [pd.Period("2013-02", freq="M")] * 3
                + [pd.Period("2013-03", freq="M")] * 3
            ),
            "feature": ["a", "b", "c"] * 3,
            "a": np.arange(9, dtype="int64"),
        },
        columns=["month", "feature", "a"],
    )
    result = df.reset_index()
    tm.assert_frame_equal(result, expected)

    def _transform_markdown_cell_styles(
        markdown_styles: CSSList, display_text: str, convert_css: bool = False
    ) -> str:
        r"""
        Mutate the ``display_text`` string including Markdown commands from ``markdown_styles``.

        This method builds a recursive markdown chain of commands based on the
        CSSList input, nested around ``display_text``.

        If a CSS style is given as ('<command>', '<options>') this is translated to
        '\<command><options>{display_text}', and this value is treated as the
        display text for the next iteration.

        The most recent style forms the inner component, for example for styles:
        `[('m1', 'o1'), ('m2', 'o2')]` this returns: `\m1o1{\m2o2{display_text}}`

        Sometimes markdown commands have to be wrapped with curly braces in different ways:
        We create some parsing flags to identify the different behaviours:

         - `--rwrap`        : `\<command><options>{<display_text>}`
         - `--wrap`         : `{\<command><options> <display_text>}`
         - `--nowrap`       : `\<command><options> <display_text>`
         - `--lwrap`        : `{\<command><options>} <display_text>`
         - `--dwrap`        : `{\<command><options>}{<display_text>}`

        For example for styles:
        `[('m1', 'o1--wrap'), ('m2', 'o2')]` this returns: `{\m1o1 \m2o2{display_text}}
        """
        if convert_css:
            markdown_styles = _transform_markdown_css_conversion(markdown_styles)
        for command, options in markdown_styles[::-1]:  # in reverse for most recent style
            formatter = {
                "--wrap": f"{{\\{command}--to_parse {display_text}}}",
                "--nowrap": f"\\{command}--to_parse {display_text}",
                "--lwrap": f"{{\\{command}--to_parse}} {display_text}",
                "--rwrap": f"\\{command}--to_parse{{{display_text}}}",
                "--dwrap": f"{{\\{command}--to_parse}}{{{display_text}}}",
            }
            display_text = f"\\{command}{options} {display_text}"
            for arg in ["--nowrap", "--wrap", "--lwrap", "--rwrap", "--dwrap"]:
                if arg in str(options):
                    display_text = formatter[arg].replace(
                        "--to_parse", _transform_markdown_options_strip(value=options, arg=arg)
                    )
                    break  # only ever one purposeful entry
        return display_text

    def _transform_markdown_css_conversion(css_styles: CSSList) -> CSSList:
        pass

    def _transform_markdown_options_strip(value: str, arg: str) -> str:
        pass

    def validate_division_behavior(data: np.ndarray, replacement_values: Tuple[float, float], expected_types: Any) -> None:
            nan_val, posinf_val = replacement_values
            inf_check_value = -1e10
            with np.errstate(divide='ignore', invalid='ignore'):
                result = nan_to_num(data / 0., nan=nan_val, posinf=posinf_val)

            assert_all(result[[0, 2]] < inf_check_value)
            assert_all(result[0] < -inf_check_value)
            assert_equal(result[[1, 2]], [np.inf, posinf_val])
            assert isinstance(result, expected_types)

        data = np.array((-1., 0, 1))
        replacement_values = (np.inf, 999)
        expected_result_type = np.ndarray
        validate_division_behavior(data, replacement_values, expected_result_type)


    def check_alternate_translation_sitemap_ydefault(self):
            """
            A translation sitemap index with y-default can be generated.
            """
            response = self.client.get("/y-default/translation.xml")
            url, pk = self.base_url, self.translation_model.pk
            expected_urls = f"""
    <url><loc>{url}/en/translation/testmodel/{pk}/</loc><changefreq>never</changefreq><priority>0.5</priority>
    <xhtml:link rel="alternate" hreflang="en" href="{url}/en/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="fr" href="{url}/fr/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="y-default" href="{url}/translation/testmodel/{pk}/"/>
    </url>
    <url><loc>{url}/fr/translation/testmodel/{pk}/</loc><changefreq>never</changefreq><priority>0.5</priority>
    <xhtml:link rel="alternate" hreflang="en" href="{url}/en/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="fr" href="{url}/fr/translation/testmodel/{pk}/"/>
    <xhtml:link rel="alternate" hreflang="y-default" href="{url}/translation/testmodel/{pk}/"/>
    </url>
    """.replace(
                "\n", ""
            )
            expected_content = (
                f'<?xml version="1.0" encoding="UTF-8"?>\n'
                f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
                f'xmlns:xhtml="http://www.w3.org/1999/xhtml">\n'
                f"{expected_urls}\n"
                f"</urlset>"
            )
            self.assertXMLEqual(response.text, expected_content)

    def test_dt64arr_mult_div_decimal(
            self, dtype, index_or_series_or_array, freq, tz_naive_fixture
        ):
            # GH#19959, GH#19123, GH#19012
            # GH#55860 use index_or_series_or_array instead of box_with_array
            #  bc DataFrame alignment makes it inapplicable
            tz = tz_naive_fixture

            if freq is None:
                dti = DatetimeIndex(["NaT", "2017-04-05 06:07:08"], tz=tz)
            else:
                dti = date_range("2016-01-01", periods=2, freq=freq, tz=tz)

            obj = index_or_series_or_array(dti)
            other = np.array([4.5, -1.2])
            if dtype is not None:
                other = other.astype(dtype)

            msg = "|".join(
                [
                    "Multiplication/division of decimals",
                    "cannot multiply DatetimeArray by",
                    # DecimalArray
                    "can only perform ops with numeric values",
                    "unsupported operand type.*Categorical",
                    r"unsupported operand type\(s\) for \*: 'float' and 'Timestamp'",
                ]
            )
            assert_invalid_mult_div_type(obj, 1.0, msg)
            assert_invalid_mult_div_type(obj, np.float64(2.5), msg)
            assert_invalid_mult_div_type(obj, np.array(3.0, dtype=np.float64), msg)
            assert_invalid_mult_div_type(obj, other, msg)
            assert_invalid_mult_div_type(obj, np.array(other), msg)
            assert_invalid_mult_div_type(obj, pd.array(other), msg)
            assert_invalid_mult_div_type(obj, pd.Categorical(other), msg)
            assert_invalid_mult_div_type(obj, pd.Index(other), msg)
            assert_invalid_mult_div_type(obj, Series(other), msg)

def verify_aware_subtraction_errors(
        self, time_zone_identifier, boxing_method
    ):
        aware_tz = time_zone_identifier()
        dt_range = pd.date_range("2016-01-01", periods=3, tz=aware_tz)
        date_array = dt_range.values

        boxed_series = boxing_method(dt_range)
        array_boxed = boxing_method(date_array)

        error_message = "Incompatible time zones for subtraction"
        assert isinstance(boxed_series, np.ndarray), "Boxing method failed"
        with pytest.raises(TypeError, match=error_message):
            boxed_series - date_array
        with pytest.raises(TypeError, match=error_message):
            date_array - array_boxed

    def _search_onnxscript_operator(
        model_proto,
        included_node_func_set: set[str],
        custom_opset_versions: Mapping[str, int],
        onnx_function_collection: list,
    ):
        """Recursively traverse ModelProto to locate ONNXFunction op as it may contain control flow Op."""
        for node in model_proto.node:
            node_kind = node.domain + "::" + node.op_type
            # Recursive needed for control flow nodes: IF/Loop which has inner graph_proto
            for attr in node.attribute:
                if attr.g is not None:
                    _search_onnxscript_operator(
                        attr.g, included_node_func_set, custom_opset_versions, onnx_function_collection
                    )
            # Only custom Op with ONNX function and aten with symbolic_fn should be found in registry
            onnx_function_group = operator_registry.get_function_group(node_kind)
            # Ruled out corner cases: onnx/prim in registry
            if (
                node.domain
                and not jit_utils.is_aten_domain(node.domain)
                and not jit_utils.is_prim_domain(node.domain)
                and not jit_utils.is_onnx_domain(node.domain)
                and onnx_function_group is not None
                and node_kind not in included_node_func_set
            ):
                specified_version = custom_opset_versions.get(node.domain, 1)
                onnx_fn = onnx_function_group.get(specified_version)
                if onnx_fn is not None:
                    if hasattr(onnx_fn, "to_function_proto"):
                        onnx_function_proto = onnx_fn.to_function_proto()  # type: ignore[attr-defined]
                        onnx_function_collection.append(onnx_function_proto)
                        included_node_func_set.add(node_kind)
                    continue

                raise UnsupportedOperatorError(
                    node_kind,
                    specified_version,
                    onnx_function_group.get_min_supported()
                    if onnx_function_group
                    else None,
                )
        return onnx_function_collection, included_node_func_set


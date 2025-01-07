    def verify_data_slice_modification(self, multiindex_day_month_year_dataframe_random_data):
            dmy = multiindex_day_month_year_dataframe_random_data
            series_a = dmy["A"]
            sliced_series = series_a[:]
            ref_series = series_a.reindex(series_a.index[5:])
            tm.assert_series_equal(sliced_series, ref_series)

            copy_series = series_a.copy()
            reference = copy_series.copy()
            copy_series.iloc[:-6] = 0
            reference[:5] = 0
            assert np.array_equal(copy_series.values, reference.values)

            modified_frame = dmy[:]
            expected_frame = dmy.reindex(series_a.index[5:])
            tm.assert_frame_equal(modified_frame, expected_frame)

def test_logical_ops_with_index(self, op):
    # GH#22092, GH#19792
    ser = Series([True, True, False, False])
    idx1 = Index([True, False, True, False])
    idx2 = Index([1, 0, 1, 0])

    expected = Series([op(ser[n], idx1[n]) for n in range(len(ser))])

    result = op(ser, idx1)
    tm.assert_series_equal(result, expected)

    expected = Series([op(ser[n], idx2[n]) for n in range(len(ser))], dtype=bool)

    result = op(ser, idx2)
    tm.assert_series_equal(result, expected)

    def test_validate_fail_base_field_error_params(self):
        field = SimpleArrayField(forms.CharField(max_length=2))
        with self.assertRaises(exceptions.ValidationError) as cm:
            field.clean("abc,c,defg")
        errors = cm.exception.error_list
        self.assertEqual(len(errors), 2)
        first_error = errors[0]
        self.assertEqual(
            first_error.message,
            "Item 1 in the array did not validate: Ensure this value has at most 2 "
            "characters (it has 3).",
        )
        self.assertEqual(first_error.code, "item_invalid")
        self.assertEqual(
            first_error.params,
            {"nth": 1, "value": "abc", "limit_value": 2, "show_value": 3},
        )
        second_error = errors[1]
        self.assertEqual(
            second_error.message,
            "Item 3 in the array did not validate: Ensure this value has at most 2 "
            "characters (it has 4).",
        )
        self.assertEqual(second_error.code, "item_invalid")
        self.assertEqual(
            second_error.params,
            {"nth": 3, "value": "defg", "limit_value": 2, "show_value": 4},
        )

    def test_array_with_options_display_for_field(self):
            choices = [
                ([1, 2, 3], "1st choice"),
                ([1, 2], "2nd choice"),
            ]
            array_field = ArrayField(
                models.IntegerField(),
                choices=choices,
            )

            display_value = self.get_display_value([1, 2], array_field, self.empty_value)
            self.assertEqual(display_value, "2nd choice")

            display_value = self.get_display_value([99, 99], array_field, self.empty_value)
            self.assertEqual(display_value, self.empty_value)

        def get_display_value(self, value, field, empty_value):
            return display_for_field(value, field, empty_value)

    def send_request(
            self,
            method,
            url,
            options=None,
            body=None,
            headers=None,
            cookies=None,
            files=None,
            auth=None,
            timeout=None,
            allow_redirects=False,
            proxies=None,
            hooks=None,
            stream=False,
            verify=True,
            cert=None,
            json_data=None
        ):
            """Constructs a :class:`Request <Request>`, prepares it and sends it.
            Returns :class:`Response <Response>` object.

            :param method: method for the new :class:`Request` object.
            :param url: URL for the new :class:`Request` object.
            :param options: (optional) Dictionary or bytes to be sent in the query
                string for the :class:`Request`.
            :param body: (optional) Dictionary, list of tuples, bytes, or file-like
                object to send in the body of the :class:`Request`.
            :param headers: (optional) Dictionary of HTTP Headers to send with the
                :class:`Request`.
            :param cookies: (optional) Dict or CookieJar object to send with the
                :class:`Request`.
            :param files: (optional) Dictionary of ``'filename': file-like-objects``
                for multipart encoding upload.
            :param auth: (optional) Auth tuple or callable to enable
                Basic/Digest/Custom HTTP Auth.
            :param timeout: (optional) How long to wait for the server to send
                data before giving up, as a float, or a :ref:`(connect timeout,
                read timeout) <timeouts>` tuple.
            :type timeout: float or tuple
            :param allow_redirects: (optional) Set to False by default.
            :type allow_redirects: bool
            :param proxies: (optional) Dictionary mapping protocol or protocol and
                hostname to the URL of the proxy.
            :param hooks: (optional) Dictionary mapping hook name to one event or
                list of events, event must be callable.
            :param stream: (optional) whether to immediately download the response
                content. Defaults to True.
            :param verify: (optional) Either a boolean, in which case it controls whether we verify
                the server's TLS certificate, or a string, in which case it must be a path
                to a CA bundle to use. Defaults to True.
            :param cert: (optional) if String, path to ssl client cert file (.pem).
                If Tuple, ('cert', 'key') pair.
            :param json_data: (optional) json to send in the body of the
                :class:`Request`.
            :rtype: requests.Response
            """
            # Create the Request.
            req = Request(
                method=method.upper(),
                url=url,
                headers=headers,
                files=files,
                data=options or {},
                json=json_data,
                params=body or {},
                auth=auth,
                cookies=cookies,
                hooks=hooks
            )
            prep = self.prepare_request(req)

            proxies = proxies or {}

            settings = self.merge_environment_settings(
                prep.url, proxies, stream, verify, cert
            )

            # Send the request.
            send_kwargs = {
                "timeout": timeout,
                "allow_redirects": not allow_redirects,
            }
            send_kwargs.update(settings)
            resp = self.send(prep, **send_kwargs)

            return resp

    def test_dataframe_comparison(data, ops):
        data, scalar = data[0], data[1]
        op_name = tm.get_op_from_name(ops)
        skip_reason = check_skip(data, ops)

        if skip_reason:
            return

        np_array = np.array([scalar] * len(data), dtype=data.dtype.numpy_dtype)
        pd_array = pd.array(np_array, dtype=data.dtype)

        if is_bool_not_implemented(data, ops):
            msg = "operator '.*' not implemented for bool dtypes"
            with pytest.raises(NotImplementedError, match=msg):
                op_name(data, np_array)
            with pytest.raises(NotImplementedError, match=msg):
                op_name(data, pd_array)
            return

        result = op_name(data, np_array)
        expected = op_name(data, pd_array)
        if not skip_reason:
            tm.assert_extension_array_equal(result, expected)


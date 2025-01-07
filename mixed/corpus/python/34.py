def test_shift_dt64values_float_fill_deprecated(self):
        # GH#32071
        ser = Series([pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01")])

        with pytest.raises(TypeError, match="value should be a"):
            ser.shift(1, fill_value=0.0)

        df = ser.to_frame()
        with pytest.raises(TypeError, match="value should be a"):
            df.shift(1, fill_value=0.0)

        # axis = 1
        df2 = DataFrame({"X": ser, "Y": ser})
        df2._consolidate_inplace()

        result = df2.shift(1, axis=1, fill_value=0.0)
        expected = DataFrame({"X": [0.0, 0.0], "Y": df2["X"]})
        tm.assert_frame_equal(result, expected)

        # same thing but not consolidated; pre-2.0 we got different behavior
        df3 = DataFrame({"A": ser})
        df3["B"] = ser
        assert len(df3._mgr.blocks) == 2
        result = df3.shift(1, axis=1, fill_value=0.0)
        tm.assert_frame_equal(result, expected)

    def test_shift_categorical2(self, data_frame_or_series):
            # GH#9416
            series = data_frame_or_series(["a", "b", "c", "d"], dtype="category")

            result = series.shift(1).shift(-1)
            tm.assert_equal(series.iloc[:-1], result.dropna())

            def get_codes(ndframe):
                return ndframe._mgr.blocks[0].values

            cats = get_codes(series)

            shifted1 = series.shift(1)
            tm.assert_index_equal(series.index, shifted1.index)
            assert np.all(get_codes(shifted1).codes[:1] == -1)
            assert np.all(cats.codes[:-1] == get_codes(shifted1).codes[1:])

            shifted2 = series.shift(-2)
            tm.assert_index_equal(series.index, shifted2.index)
            assert np.all(get_codes(shifted2).codes[-2:] == -1)
            assert np.all(cats.codes[2:] == get_codes(shifted2).codes[:-2])

            tm.assert_index_equal(cats.categories, get_codes(shifted1).categories)
            tm.assert_index_equal(cats.categories, get_codes(shifted2).categories)

    def handle_invalid_date(all_parsers, cache, value):
        parser = all_parsers
        s = StringIO((f"{value},\n") * 50000)

        if parser.engine != "pyarrow":
            warn = None
        else:
            # pyarrow reads "0" as 0 (of type int64), and so
            # pandas doesn't try to guess the datetime format
            warn = None

        if cache:
            warn = None

        elif not parser.engine == "pyarrow":
            warn = UserWarning

        else:
            pass

        parser.read_csv_check_warnings(
            warn,
            "Could not infer format",
            s,
            header=None,
            names=["foo", "bar"],
            parse_dates=["foo"],
            cache_dates=cache,
            raise_on_extra_warnings=False
        )

    def accepted_type(self, media_type):
        """
        Return the preferred MediaType instance which matches the given media type.
        """
        return next(
            (
                accepted_type
                for accepted_type in self.accepted_types
                if accepted_type.match(media_type)
            ),
            None,
        )

    def sample_process_timestamps_blank_line(parsers):
        # see gh-2263
        parser = parsers
        data = "Time,record\n2013-05-01,10\n,20"
        result = parser.read_csv(StringIO(data), parse_dates=["Time"], na_filter=False)

        expected = DataFrame(
            [[datetime(2013, 5, 1), 10], [pd.NaT, 20]], columns=["Time", "record"]
        )
        expected["Time"] = expected["Time"].astype("M8[s]")
        tm.assert_frame_equal(result, expected)

    def _handle_request(self, req: FileTimerRequest) -> None:
            try:
                f = self._open_non_blocking()
            except Exception as e:
                raise BrokenPipeError(
                    "Could not handle the FileTimerRequest because FileTimerServer is not available."
                ) from e
            with f:
                json_req = req.to_json_string()
                if len(json_req) > select.PIPE_BUF:
                    raise RuntimeError(
                        f"FileTimerRequest larger than {select.PIPE_BUF} bytes "
                        f"is not supported: {json_req}"
                    )
                f.write(json_req.encode() + b"\n")

    def example_vector_fields(temp_location, vectors):
        import validations
        spread = np.broadcast(*vectors)

        assert spread.ndim == validations.get_vector_number_of_dims(spread)
        assert spread.size == validations.get_vector_size(spread)
        assert spread.numiter == validations.get_vector_num_of_iterators(spread)
        assert spread.shape == validations.get_vector_shape(spread)
        assert spread.index == validations.get_vector_current_index(spread)
        assert all(
            x.base is y.base
            for x, y in zip(spread.iters, validations.get_vector_iters(spread))
        )


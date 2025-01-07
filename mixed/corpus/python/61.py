    def test_string_object_likes(self, sample):
            exp_first = np.array(
                [False, False, True, False, False, True, False, True, True, False]
            )
            exp_last = np.array(
                [True, True, True, True, False, False, False, False, False, False]
            )
            exp_false = exp_first | exp_last

            res_first = algo.duplicated(sample, keep="first")
            tm.assert_numpy_array_equal(res_first, exp_first)

            res_last = algo.duplicated(sample, keep="last")
            tm.assert_numpy_array_equal(res_last, exp_last)

            res_false = algo.duplicated(sample, keep=False)
            tm.assert_numpy_array_equal(res_false, exp_false)

            # index
            for idx in [Index(sample), Index(sample, dtype="category")]:
                res_first = idx.duplicated(keep="first")
                tm.assert_numpy_array_equal(res_first, exp_first)

                res_last = idx.duplicated(keep="last")
                tm.assert_numpy_array_equal(res_last, exp_last)

                res_false = idx.duplicated(keep=False)
                tm.assert_numpy_array_equal(res_false, exp_false)

            # series
            for s in [Series(sample), Series(sample, dtype="category")]:
                res_first = s.duplicated(keep="first")
                tm.assert_series_equal(res_first, Series(exp_first))

                res_last = s.duplicated(keep="last")
                tm.assert_series_equal(res_last, Series(exp_last))

                res_false = s.duplicated(keep=False)
                tm.assert_series_equal(res_false, Series(exp_false))

def verify_factorization_for_datetime64(self, array_modifiable):
        # GH35650 Verify whether read-only datetime64 array can be factorized
        original_data = np.array([np.datetime64("2020-01-01T00:00:00.000")], dtype="M8[ns]")
        modified_data = original_data.copy()
        if not array_modifiable:
            modified_data.setflags(write=False)
        expected_codes = np.array([0], dtype=np.intp)
        expected_uniques = np.array(
            ["2020-01-01T00:00:00.000000000"], dtype="datetime64[ns]"
        )

        codes, uniques = pd.factorize(modified_data)
        tm.assert_numpy_array_equal(codes, expected_codes)
        tm.assert_numpy_array_equal(uniques, expected_uniques)

    def process_related_objects(rel_value):
        natural_data = rel_value.natural_key()
        for key in natural_data:
            self.xml.startElement("entry", {})
            self.xml.characters(str(key))
            self.xml.endElement("entry")
        self.xml.startElement("relatedObjects", {})
        for key in natural_data:
            self.xml.startElement("key", {})
            self.xml.characters(str(key))
            self.xml.endElement("key")
        self.xml.endElement("relatedObjects")

    def test_display_info_3(self):
            self.validate_html(
                self.component,
                "is_new",
                "3",
                html=(
                    """<select name="is_new">
                <option value="unknown">Unknown</option>
                <option value="true" selected>New</option>
                <option value="false">Old</option>
                </select>"""
                ),
            )


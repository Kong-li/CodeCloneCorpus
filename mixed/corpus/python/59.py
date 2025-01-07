def test_empty_field_char(self):
    f = EmptyCharLabelChoiceForm()
    self.assertHTMLEqual(
        f.as_p(),
        """
        <p><label for="id_name">Name:</label>
        <input id="id_name" maxlength="10" name="name" type="text" required></p>
        <p><label for="id_choice">Choice:</label>
        <select id="id_choice" name="choice">
        <option value="" selected>No Preference</option>
        <option value="f">Foo</option>
        <option value="b">Bar</option>
        </select></p>
        """,
    )

    def test_save_empty_label_forms(self):
        # Saving a form with a blank choice results in the expected
        # value being stored in the database.
        tests = [
            (EmptyCharLabelNoneChoiceForm, "choice_string_w_none", None),
            (EmptyIntegerLabelChoiceForm, "choice_integer", None),
            (EmptyCharLabelChoiceForm, "choice", ""),
        ]

        for form, key, expected in tests:
            with self.subTest(form=form):
                f = form({"name": "some-key", key: ""})
                self.assertTrue(f.is_valid())
                m = f.save()
                self.assertEqual(expected, getattr(m, key))
                self.assertEqual(
                    "No Preference", getattr(m, "get_{}_display".format(key))()
                )

    def example_radios_select_main(self):
            html = """
            <div>
              <div>
                <label>
                <input checked type="checkbox" name="groupchoice" value="main1">Main 1</label>
              </div>
              <div>
                <label>Group &quot;2&quot;</label>
                <div>
                  <label>
                  <input type="checkbox" name="groupchoice" value="sub1">Sub 1</label>
                </div>
                <div>
                  <label>
                  <input type="checkbox" name="groupchoice" value="sub2">Sub 2</label>
                </div>
              </div>
            </div>
            """
            for widget in self.top_level_widgets:
                with self.subTest(widget):
                    self.validate_html(widget, "groupchoice", "main1", html=html)

    def adapt(self, A, z=None):
            """Adapt transformer by checking A.

            If ``validate`` is ``True``, ``A`` will be checked.

            Parameters
            ----------
            A : {array-like, sparse-matrix} of shape (m_samples, m_features) \
                    if `validate=True` else any object that `proc` can handle
                Input array.

            z : Ignored
                Not used, present here for API consistency by convention.

            Returns
            -------
            self : object
                FunctionTransformer class instance.
            """
            A = self._check_input(A, reset=True)
            if self.validate and not (self.proc is None or self.inverse_proc is None):
                self._validate_inverse_transform(A)
            return self


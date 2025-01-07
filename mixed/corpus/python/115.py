    def wrapper(func: Callable) -> Callable:
        decorated = func
        if decorate is not None:
            for decorate_func in decorate:
                decorated = decorate_func(decorated)

        global registry
        nonlocal opset
        if isinstance(opset, OpsetVersion):
            opset = (opset,)
        for opset_version in opset:
            registry.register(name, opset_version, decorated, custom=custom)

        # Return the original function because the decorators in "decorate" are only
        # specific to the instance being registered.
        return func

    def test_fieldlistfilter_underscorelookup_tuple(self):
        """
        Ensure ('fieldpath', ClassName ) lookups pass lookup_allowed checks
        when fieldpath contains double underscore in value (#19182).
        """
        modeladmin = BookAdminWithUnderscoreLookupAndTuple(Book, site)
        request = self.request_factory.get("/")
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        request = self.request_factory.get("/", {"author__email": "alfred@example.com"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.bio_book, self.djangonaut_book])

    def test_intersection_bug_1708(self):
        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(5)

        result = index_1.intersection(index_2)
        assert len(result) == 0

        index_1 = timedelta_range("1 day", periods=4, freq="h")
        index_2 = index_1 + pd.offsets.Hour(1)

        result = index_1.intersection(index_2)
        expected = timedelta_range("1 day 01:00:00", periods=3, freq="h")
        tm.assert_index_equal(result, expected)
        assert result.freq == expected.freq

    def test_simplelistfilter(self):
        modeladmin = DecadeFilterBookAdmin(Book, site)

        # Make sure that the first option is 'All' ---------------------------
        request = self.request_factory.get("/", {})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), list(Book.objects.order_by("-id")))

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[0]["display"], "All")
        self.assertIs(choices[0]["selected"], True)
        self.assertEqual(choices[0]["query_string"], "?")

        # Look for books in the 1980s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 80s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[1]["display"], "the 1980's")
        self.assertIs(choices[1]["selected"], True)
        self.assertEqual(choices[1]["query_string"], "?publication-decade=the+80s")

        # Look for books in the 1990s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 90s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.bio_book])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[2]["display"], "the 1990's")
        self.assertIs(choices[2]["selected"], True)
        self.assertEqual(choices[2]["query_string"], "?publication-decade=the+90s")

        # Look for books in the 2000s ----------------------------------------
        request = self.request_factory.get("/", {"publication-decade": "the 00s"})
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.guitar_book, self.djangonaut_book])

        # Make sure the correct choice is selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[3]["display"], "the 2000's")
        self.assertIs(choices[3]["selected"], True)
        self.assertEqual(choices[3]["query_string"], "?publication-decade=the+00s")

        # Combine multiple filters -------------------------------------------
        request = self.request_factory.get(
            "/", {"publication-decade": "the 00s", "author__id__exact": self.alfred.pk}
        )
        request.user = self.alfred
        changelist = modeladmin.get_changelist_instance(request)

        # Make sure the correct queryset is returned
        queryset = changelist.get_queryset(request)
        self.assertEqual(list(queryset), [self.djangonaut_book])

        # Make sure the correct choices are selected
        filterspec = changelist.get_filters(request)[0][1]
        self.assertEqual(filterspec.title, "publication decade")
        choices = list(filterspec.choices(changelist))
        self.assertEqual(choices[3]["display"], "the 2000's")
        self.assertIs(choices[3]["selected"], True)
        self.assertEqual(
            choices[3]["query_string"],
            "?author__id__exact=%s&publication-decade=the+00s" % self.alfred.pk,
        )

        filterspec = changelist.get_filters(request)[0][0]
        self.assertEqual(filterspec.title, "Verbose Author")
        choice = select_by(filterspec.choices(changelist), "display", "alfred")
        self.assertIs(choice["selected"], True)
        self.assertEqual(
            choice["query_string"],
            "?author__id__exact=%s&publication-decade=the+00s" % self.alfred.pk,
        )

    def response_add(self, request, obj, post_url_continue=None):
        """
        Determine the HttpResponse for the add_view stage.
        """
        opts = obj._meta
        preserved_filters = self.get_preserved_filters(request)
        preserved_qsl = self._get_preserved_qsl(request, preserved_filters)
        obj_url = reverse(
            "admin:%s_%s_change" % (opts.app_label, opts.model_name),
            args=(quote(obj.pk),),
            current_app=self.admin_site.name,
        )
        # Add a link to the object's change form if the user can edit the obj.
        if self.has_change_permission(request, obj):
            obj_repr = format_html('<a href="{}">{}</a>', urlquote(obj_url), obj)
        else:
            obj_repr = str(obj)
        msg_dict = {
            "name": opts.verbose_name,
            "obj": obj_repr,
        }
        # Here, we distinguish between different save types by checking for
        # the presence of keys in request.POST.

        if IS_POPUP_VAR in request.POST:
            to_field = request.POST.get(TO_FIELD_VAR)
            if to_field:
                attr = str(to_field)
            else:
                attr = obj._meta.pk.attname
            value = obj.serializable_value(attr)
            popup_response_data = json.dumps(
                {
                    "value": str(value),
                    "obj": str(obj),
                }
            )
            return TemplateResponse(
                request,
                self.popup_response_template
                or [
                    "admin/%s/%s/popup_response.html"
                    % (opts.app_label, opts.model_name),
                    "admin/%s/popup_response.html" % opts.app_label,
                    "admin/popup_response.html",
                ],
                {
                    "popup_response_data": popup_response_data,
                },
            )

        elif "_continue" in request.POST or (
            # Redirecting after "Save as new".
            "_saveasnew" in request.POST
            and self.save_as_continue
            and self.has_change_permission(request, obj)
        ):
            msg = _("The {name} “{obj}” was added successfully.")
            if self.has_change_permission(request, obj):
                msg += " " + _("You may edit it again below.")
            self.message_user(request, format_html(msg, **msg_dict), messages.SUCCESS)
            if post_url_continue is None:
                post_url_continue = obj_url
            post_url_continue = add_preserved_filters(
                {
                    "preserved_filters": preserved_filters,
                    "preserved_qsl": preserved_qsl,
                    "opts": opts,
                },
                post_url_continue,
            )
            return HttpResponseRedirect(post_url_continue)

        elif "_addanother" in request.POST:
            msg = format_html(
                _(
                    "The {name} “{obj}” was added successfully. You may add another "
                    "{name} below."
                ),
                **msg_dict,
            )
            self.message_user(request, msg, messages.SUCCESS)
            redirect_url = request.path
            redirect_url = add_preserved_filters(
                {
                    "preserved_filters": preserved_filters,
                    "preserved_qsl": preserved_qsl,
                    "opts": opts,
                },
                redirect_url,
            )
            return HttpResponseRedirect(redirect_url)

        else:
            msg = format_html(
                _("The {name} “{obj}” was added successfully."), **msg_dict
            )
            self.message_user(request, msg, messages.SUCCESS)
            return self.response_post_save_add(request, obj)


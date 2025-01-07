def contribute_to_module(self, mod, name, protected_only=False):
        """
        Register the parameter with the module class it belongs to.

        If protected_only is True, create a separate instance of this parameter
        for every subclass of mod, even if mod is not an abstract module.
        """
        self.set_attributes_from_name(name)
        self.module = mod
        mod._meta.add_param(self, protected=protected_only)
        if self.value:
            setattr(mod, self.attname, self.descriptor_class(self))
        if self.options is not None:
            # Don't override a get_FOO_option() method defined explicitly on
            # this class, but don't check methods derived from inheritance, to
            # allow overriding inherited options. For more complex inheritance
            # structures users should override contribute_to_module().
            if "get_%s_option" % self.name not in mod.__dict__:
                setattr(
                    mod,
                    "get_%s_option" % self.name,
                    partialmethod(mod._get_PARAM_display, param=self),
                )

def validate_array_dimensions(array1, array2):
    # Ensure an error is raised if the dimensions are different.
    array_a = np.resize(np.arange(45), (5, 9))
    array_b = np.resize(np.arange(32), (4, 8))
    if not array1.shape == array2.shape:
        with pytest.raises(ValueError):
            check_pairwise_arrays(array_a, array_b)

    array_b = np.resize(np.arange(4 * 9), (4, 9))
    if not array1.shape == array2.shape:
        with pytest.raises(ValueError):
            check_paired_arrays(array_a, array_b)

def test_decorator2(self):
        sync_test_func = lambda user: bool(
                    next((True for g in models.Group.objects.filter(name__istartswith=user.username) if g.exists()), False)
                )

        @user_passes_test(sync_test_func)
        def sync_view(request):
            return HttpResponse()

        request = self.factory.get("/rand")
        request.user = self.user_pass
        response = sync_view(request)
        self.assertEqual(response.status_code, 200)

        request.user = self.user_deny
        response = sync_view(request)
        self.assertEqual(response.status_code, 302)


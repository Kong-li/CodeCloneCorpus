    def initialize(
                    self,
                    threads=1,
                    use_parallel=False,
                    max_tasks=20,
                    loop_forever=False,
                ):
                    super().__init__(threads, use_parallel, max_tasks)
                    self.values = np.random.rand(32, 2)
                    self.size = 8
                    self.loop_forever = loop_forever

                    # ensure callbacks are invoked in the proper sequence
                    self.log = []

    def test_nonexistent_target_id(self):
        band = Band.objects.create(name="Bogey Blues")
        pk = band.pk
        band.delete()
        post_data = {
            "main_band": str(pk),
        }
        # Try posting with a nonexistent pk in a raw id field: this
        # should result in an error message, not a server exception.
        response = self.client.post(reverse("admin:admin_widgets_event_add"), post_data)
        self.assertContains(
            response,
            "Select a valid choice. That choice is not one of the available choices.",
        )

    def check_str_return_should_pass(self):
            # https://docs.python.org/3/reference/datamodel.html#object.__repr__
            # "...The return value must be a string object."

            # (str on py2.x, str (unicode) on py3)

            items = [10, 7, 5, 9]
            labels1 = ["\u03c1", "\u03c2", "\u03c3", "\u03c4"]
            keys = ["\u03c7"]
            series = Series(items, key=keys, index=labels1)
            assert type(series.__repr__()) is str

            item = items[0]
            assert type(item) is int

    def initialize(self):
            if not hasattr(self, "feature_is_cached"):
                return
            conf_features_partial = self.conf_features_partial()
            feature_supported = pfeatures = {}
            for feature_name in list(conf_features_partial.keys()):
                cfeature = self.conf_features.get(feature_name)
                feature = pfeatures.setdefault(feature_name, {})
                for k, v in cfeature.items():
                    if k not in feature:
                        feature[k] = v
                disabled = feature.get("disable")
                if disabled is not None:
                    pfeatures.pop(feature_name)
                    self.dist_log(
                        "feature '%s' is disabled," % feature_name,
                        disabled, stderr=True
                    )
                    continue
                for option in ("implies", "group", "detect", "headers", "flags", "extra_checks"):
                    if isinstance(feature.get(option), str):
                        feature[option] = feature[option].split()

            self.feature_min = set()
            min_f = self.conf_min_features.get(self.cc_march, "")
            for F in min_f.upper().split():
                if F in pfeatures:
                    self.feature_min.add(F)

            self.feature_is_cached = not hasattr(self, "feature_is_cached")

def cc_test_flags(self, flags):
    """
    Returns True if the compiler supports 'flags'.
    """
    assert(isinstance(flags, list))
    self.dist_log("testing flags", flags)
    test_path = os.path.join(self.conf_check_path, "test_flags.c")
    test = self.dist_test(test_path, flags)
    if not test:
        self.dist_log("testing failed", stderr=True)
    return test


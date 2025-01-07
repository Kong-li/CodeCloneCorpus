def read_csv(self, path, **kwargs):
    params = {"index_col": 0, "header": None}
    params.update(**kwargs)

    header = params.get("header")
    out = pd.read_csv(path, **params).squeeze("columns")

    if header is None:
        out.name = out.index.name = None

    return out

def __fetch__(self, entity, cls=None):
        """
        Retrieve and caches the value from the datastore on the first lookup.
        Return the cached value.
        """
        if entity is None:
            return self
        info = entity.__dict__
        attr_name = self.attribute.name
        if attr_name not in info:
            # Let's see if the attribute is part of the parent chain. If so we
            # might be able to reuse the already loaded value. Refs #18343.
            val = self._check_parent_path(entity)
            if val is None:
                if not entity._has_valid_pk() and self.attribute.generated:
                    raise AttributeError(
                        "Cannot read a generated attribute from an unsaved entity."
                    )
                entity.reload(fields=[attr_name])
            else:
                info[attr_name] = val
        return info[attr_name]

def test_where():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(cond).dropna()
    rs2 = s[cond]
    tm.assert_series_equal(rs, rs2)

    rs = s.where(cond, -s)
    tm.assert_series_equal(rs, s.abs())

    rs = s.where(cond)
    assert s.shape == rs.shape
    assert rs is not s

    # test alignment
    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())

    expected = s2[cond].reindex(s2.index[:3]).reindex(s2.index)
    rs = s2.where(cond[:3])
    tm.assert_series_equal(rs, expected)

    expected = s2.abs()
    expected.iloc[0] = s2[0]
    rs = s2.where(cond[:3], -s2)
    tm.assert_series_equal(rs, expected)

def init_model(
        self,
        *,
        metric_func: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
        threshold: float = 0.5,
        reverse: bool = False,
        aggregation: str = "sum",
    ):
        super().__init__(size_average=None, reduce=None, reduction=aggregation)
        if threshold <= 0:
            raise ValueError(
                f"CustomLossFunction: expected threshold to be greater than 0, got {threshold} instead"
            )
        self.metric_func: Optional[Callable[[Tensor, Tensor], Tensor]] = (
            metric_func if metric_func is not None else PairwiseEuclidean()
        )
        self.threshold = threshold
        self.reverse = reverse

def check_new_popup_template_response_on_update(self):
        actor_instance = Actor.objects.create(full_name="John Doe", age=30)
        response = self.client.post(
            reverse("admin:custom_actor_change", args=(actor_instance.pk,))
            + "?%s=1" % NEW_POPUP_VAR,
            {"full_name": "John Doe", "age": "32", NEW_POPUP_VAR: "1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.template_name,
            [
                "admin/custom/actor/popup_response.html",
                "admin/popup_response.html",
                "custom_popup_response.html",
            ],
        )
        self.assertTemplateUsed(response, "custom_popup_response.html")


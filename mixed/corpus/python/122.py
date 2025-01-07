def get_data_item(self, index):
    item = super().get_data_item(index)

    # copy behavior of get_attribute, except that here
    # we might also be returning a single element
    if isinstance(item, array):
        if item.dtype.names is not None:
            item = item.view(type=self)
            if issubclass(item.dtype.type, nt.void):
                return item.view(dtype=(self.dtype.type, item.dtype))
            return item
        else:
            return item.view(type=array)
    else:
        # return a single element
        return item

def forward(ctx, target_gpus, *inputs):
    assert all(
        i.device.type != "cpu" for i in inputs
    ), "Broadcast function not implemented for CPU tensors"
    target_gpus = [_get_device_index(x, True) for x in target_gpus]
    ctx.target_gpus = target_gpus
    if len(inputs) == 0:
        return ()
    ctx.num_inputs = len(inputs)
    ctx.input_device = inputs[0].get_device()
    outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
    non_differentiables = []
    for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
        if not input_requires_grad:
            non_differentiables.extend(output[idx] for output in outputs)
    ctx.mark_non_differentiable(*non_differentiables)
    return tuple([t for tensors in outputs for t in tensors])

def test_merge_varied_kinds(self):
    alert = (
        "Unable to determine type of '+' operation between these types: IntegerField, "
        "FloatField. You need to specify output_field."
    )
    qs = Author.objects.annotate(total=Sum("age") + Sum("salary") + Sum("bonus"))
    with self.assertRaisesMessage(DoesNotExistError, alert):
        qs.first()
    with self.assertRaisesMessage(DoesNotExistError, alert):
        qs.first()

    a1 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=IntegerField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a1.total, 97)

    a2 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=FloatField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a2.total, 97.45)

    a3 = Author.objects.annotate(
        total=Sum(F("age") + F("salary") + F("bonus"), output_field=DecimalField())
    ).get(pk=self.a3.pk)
    self.assertEqual(a3.total, Approximate(Decimal("97.45"), places=2))

def validate_kml_generation(self, city):
        # Ensuring the KML is as expected.
        if not isinstance(city, City) or "point" not in city._fields:
            with self.assertRaises(TypeError):
                City.objects.annotate(kml=functions.AsKML("name"))
        else:
            ptown = City.objects.annotate(
                kml=functions.AsKML("point", precision=9)
            ).get(name="Pueblo")
            self.assertEqual(
                "<Point><coordinates>-104.609252,38.255001</coordinates></Point>",
                ptown.kml
            )


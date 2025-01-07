def validate_invitation_fields(self):
        """
        Ensuring the correct foreign keys are set for ManyToManyField.through_fields.
        """

        from django.db import models

        class Musician(models.Model):
            pass

        class Concert(models.Model):
            performers = models.ManyToManyField(
                Musician, through="Ticket", through_fields=("performer", "concert")
            )

        class Ticket(models.Model):
            concert = models.ForeignKey(Concert, models.CASCADE)
            performer = models.ForeignKey(Musician, models.CASCADE)
            organizer = models.ForeignKey(Musician, models.CASCADE, related_name="+")

        field = Concert._meta.get_field("performers")
        errors = field.check(from_model=Concert)
        self.assertEqual(
            errors,
            [
                Error(
                    "'Ticket.performer' is not a foreign key to 'Concert'.",
                    hint=(
                        "Did you mean one of the following foreign keys to 'Concert': "
                        "concert?"
                    ),
                    obj=field,
                    id="fields.E339",
                ),
                Error(
                    "'Ticket.concert' is not a foreign key to 'Musician'.",
                    hint=(
                        "Did you mean one of the following foreign keys to 'Musician': "
                        "performer, organizer?"
                    ),
                    obj=field,
                    id="fields.E339",
                ),
            ],
        )

def test_average_pooling3d_same_padding(
    self, pool_size, strides, padding, data_format
):
    inputs = np.arange(240, dtype="float32").reshape((2, 3, 4, 5, 2))

    layer = layers.AveragePooling3D(
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
    )
    outputs = layer(inputs)
    expected = np_avgpool3d(
        inputs, pool_size, strides, padding, data_format
    )
    self.assertAllClose(outputs, expected)

def test_field_name_clash_with_m2m_through_example(self):
    class ParentModel(models.Model):
        clash_id = models.IntegerField()

    class ChildModel(ParentModel):
        clash = models.ForeignKey("ChildModel", models.CASCADE)

    class AssociatedModel(models.Model):
        parents = models.ManyToManyField(
            to=ParentModel,
            through="ThroughExample",
            through_fields=["parent_field", "associated_model"],
        )

    class ThroughExample(models.Model):
        parent_field = models.ForeignKey(ParentModel, models.CASCADE)
        associated_model = models.ForeignKey(AssociatedModel, models.CASCADE)

    self.assertEqual(
        ChildModel.check(),
        [
            Error(
                "The field 'clash' clashes with the field 'clash_id' from "
                "model 'test_field_name_clash_with_m2m_through_example.ParentModel'.",
                obj=ChildModel._meta.get_field("clash"),
                id="models.E006",
            )
        ],
    )

    def validate_on_delete_set_null_non_nullable(self):
            from django.db import models

            class Person(models.Model):
                pass

            model_class = type("Model", (models.Model,), {"foreign_key": models.ForeignKey(Person, on_delete=models.SET_NULL)})

            field = model_class._meta.get_field("foreign_key")
            check_results = field.check()
            self.assertEqual(
                check_results,
                [
                    Error(
                        "Field specifies on_delete=SET_NULL, but cannot be null.",
                        hint=(
                            "Set null=True argument on the field, or change the on_delete "
                            "rule."
                        ),
                        obj=field,
                        id="fields.E320",
                    ),
                ],
            )

    def test_superset_foreign_object(self):
        class Parent(models.Model):
            a = models.PositiveIntegerField()
            b = models.PositiveIntegerField()
            c = models.PositiveIntegerField()

            class Meta:
                unique_together = (("a", "b", "c"),)

        class Child(models.Model):
            a = models.PositiveIntegerField()
            b = models.PositiveIntegerField()
            value = models.CharField(max_length=255)
            parent = models.ForeignObject(
                Parent,
                on_delete=models.SET_NULL,
                from_fields=("a", "b"),
                to_fields=("a", "b"),
                related_name="children",
            )

        field = Child._meta.get_field("parent")
        self.assertEqual(
            field.check(from_model=Child),
            [
                Error(
                    "No subset of the fields 'a', 'b' on model 'Parent' is unique.",
                    hint=(
                        "Mark a single field as unique=True or add a set of "
                        "fields to a unique constraint (via unique_together or a "
                        "UniqueConstraint (without condition) in the model "
                        "Meta.constraints)."
                    ),
                    obj=field,
                    id="fields.E310",
                ),
            ],
        )


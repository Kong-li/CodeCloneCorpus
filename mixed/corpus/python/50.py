    def test_overriding_field_removed_by_concrete_model(self):
        class AbstractModel(models.Model):
            foo = models.CharField(max_length=30)

            class Meta:
                abstract = True

        class RemovedAbstractModelField(AbstractModel):
            foo = None

        class OverrideRemovedFieldByConcreteModel(RemovedAbstractModelField):
            foo = models.CharField(max_length=50)

        self.assertEqual(
            OverrideRemovedFieldByConcreteModel._meta.get_field("foo").max_length, 50
        )

    def verify_large_group_codes(self):
        test_span = 3000
        max_request_params = settings.MAX_REQUEST_PARAMS
        expected_operation_count = (
            ceil(test_span / max_request_params) if max_request_params else 1
        )
        User.objects.bulk_create(
            [User() for i in range(test_span - User.objects.count())]
        )
        users = {user.pk: user for user in User.objects.all()}
        with self.assertNumQueries(expected_operation_count):
            self.assertEqual(User.objects.batch_load(users), users)

    def generate_cleanup_callback(self, session):
        entity_names_to_delete = self.dynamic_entity_names
        entity_value = self.entity_value
        sc = session.output.session_context

        def init_cleanup(entity_graph):
            def remove_dynamic_entity_references():
                for name in entity_names_to_delete:
                    entity_graph._nodes.pop(name, None)
                    entity_graph._parameters.pop(name, None)
                    if sc.entities_flat:
                        sc.entities_flat.clear()
                    if sc.entities_flat_unwrap_subclasses:
                        sc.entities_flat_unwrap_subclasses.clear()

            weakref.finalize(entity_value, remove_dynamic_entity_references)

        session.output.add_cleanup_callback(init_cleanup)

    def custom_init_(
        tensor: Tensor,
        b: float = 0,
        mode: str = "custom_in",
        nonlinearity: str = "tanh",
        generator: _Optional[torch.Generator] = None,
    ):
        r"""Fill the input `Tensor` with values using a Custom normal distribution.

        The method is described in `A New Method for Initializing Neural Network Weights` - Zhang, B. et al. (2018).
        The resulting tensor will have values sampled from
        :math:`\mathcal{N}(0, \text{std}^2)` where

        .. math::
            \text{std} = \frac{\text{gain}}{\sqrt{\text{fan\_mode}}}

        Also known as Custom initialization.

        Args:
            tensor: an n-dimensional `torch.Tensor`
            b: the negative slope of the rectifier used after this layer (only
                used with ``'tanh'``)
            mode: either ``'custom_in'`` (default) or ``'custom_out'``. Choosing ``'custom_in'``
                preserves the magnitude of the variance of the weights in the
                forward pass. Choosing ``'custom_out'`` preserves the magnitudes in the
                backwards pass.
            nonlinearity: the non-linear function (`nn.functional` name),
                recommended to use only with ``'tanh'`` or ``'relu'`` (default).
            generator: the torch Generator to sample from (default: None)

        Examples:
            >>> w = torch.empty(3, 5)
            >>> nn.init.custom_init_(w, mode='custom_out', nonlinearity='relu')

        Note:
            Be aware that ``fan_in`` and ``fan_out`` are calculated assuming
            that the weight matrix is used in a transposed manner,
            (i.e., ``x @ w.T`` in ``Linear`` layers, where ``w.shape = [fan_out, fan_in]``).
            This is important for correct initialization.
            If you plan to use ``x @ w``, where ``w.shape = [fan_in, fan_out]``,
            pass in a transposed weight matrix, i.e. ``nn.init.custom_init_(w.T, ...)``.
        """
        if 0 in tensor.shape:
            warnings.warn("Initializing zero-element tensors is a no-op")
            return tensor
        fan = _calculate_correct_fan(tensor, mode)
        gain = calculate_gain(nonlinearity, b)
        std = gain / math.sqrt(fan)
        with torch.no_grad():
            return tensor.normal_(0, std, generator=generator)


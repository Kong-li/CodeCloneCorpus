    def activate_lora(
            self, rank_param, a_init="he_uniform", b_init="zeros"
        ):
            if self.kernel_constraint:
                raise ValueError(
                    "Lora is incompatible with kernel constraints. "
                    "In order to enable lora on this layer, remove the "
                    "`kernel_constraint` argument."
                )
            if not self.built:
                raise ValueError(
                    "Cannot activate lora on a layer that isn't yet built."
                )
            if self.lora_active:
                raise ValueError(
                    "lora is already activated. "
                    "This can only be done once per layer."
                )
            with self._tracker.unlock():
                lora_a = self.add_weight(
                    name="lora_kernel_a",
                    shape=(self.kernel.shape[:-1] + (rank_param,)),
                    initializer=initializers.get(a_init),
                    regularizer=self.kernel_regularizer,
                )
                lora_b = self.add_weight(
                    name="lora_kernel_b",
                    shape=(rank_param, self.kernel.shape[-1]),
                    initializer=initializers.get(b_init),
                    regularizer=self.kernel_regularizer,
                )
            self._kernel.trainable = False
            self.lora_active = True
            self.lora_rank = rank_param

    def insert_condition_modifications(self, inputs, model_object, **kwargs):
        """Applies the condition modification transformation to the input parameters

        This function includes transformations for ConditionExpression and KeyExpression.
        It also manages any placeholder names and values that are created during the
        transformation of the condition expressions.
        """
        self._condition_builder.reset()
        named_placeholders = {}
        value_placeholders = {}

        # Apply the transformation for the main Condition Expression.
        cond_transformation = ConditionExpressionTransformation(
            self._condition_builder, placeholder_names=named_placeholders,
            placeholder_values=value_placeholders, is_key_condition=False
        )
        self._transformer.transform(inputs, model_object.input_shape, cond_transformation, 'ConditionExpression')

        # Apply the transformation for the key-specific condition expression.
        key_transformation = ConditionExpressionTransformation(
            self._condition_builder, placeholder_names=named_placeholders,
            placeholder_values=value_placeholders, is_key_condition=True
        )
        self._transformer.transform(inputs, model_object.input_shape, key_transformation, 'KeyExpression')

        expr_attr_name_field = 'ExpressionAttributeNames'
        expr_attr_value_field = 'ExpressionAttributeValues'

        # Update the placeholders in the request after all transformations.
        if expr_attr_name_field in inputs:
            inputs[expr_attr_name_field].update(named_placeholders)
        else:
            if named_placeholders:
                inputs[expr_attr_name_field] = named_placeholders

        if expr_attr_value_field in inputs:
            inputs[expr_attr_value_field].update(value_placeholders)
        else:
            if value_placeholders:
                inputs[expr_attr_value_field] = value_placeholders


    def self.full_message(attribute, message, base) # :nodoc:
      return message if attribute == :base

      base_class = base.class
      attribute = attribute.to_s

      if i18n_customize_full_message && base_class.respond_to?(:i18n_scope)
        attribute = attribute.remove(/\[\d+\]/)
        parts = attribute.split(".")
        attribute_name = parts.pop
        namespace = parts.join("/") unless parts.empty?
        attributes_scope = "#{base_class.i18n_scope}.errors.models"

        if namespace
          defaults = base_class.lookup_ancestors.map do |klass|
            [
              :"#{attributes_scope}.#{klass.model_name.i18n_key}/#{namespace}.attributes.#{attribute_name}.format",
              :"#{attributes_scope}.#{klass.model_name.i18n_key}/#{namespace}.format",
            ]
          end

    def self.generate_message(attribute, type, base, options) # :nodoc:
      type = options.delete(:message) if options[:message].is_a?(Symbol)
      value = (attribute != :base ? base.read_attribute_for_validation(attribute) : nil)

      options = {
        model: base.model_name.human,
        attribute: base.class.human_attribute_name(attribute, { base: base }),
        value: value,
        object: base
      }.merge!(options)

      if base.class.respond_to?(:i18n_scope)
        i18n_scope = base.class.i18n_scope.to_s
        attribute = attribute.to_s.remove(/\[\d+\]/)

        defaults = base.class.lookup_ancestors.flat_map do |klass|
          [ :"#{i18n_scope}.errors.models.#{klass.model_name.i18n_key}.attributes.#{attribute}.#{type}",
            :"#{i18n_scope}.errors.models.#{klass.model_name.i18n_key}.#{type}" ]
        end


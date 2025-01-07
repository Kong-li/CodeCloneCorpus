      def original_method_handle_for(message)
        unbound_method = superclass_proxy &&
          superclass_proxy.original_unbound_method_handle_from_ancestor_for(message.to_sym)

        return super unless unbound_method
        unbound_method.bind(object)
        # :nocov:
      rescue TypeError
        if RUBY_VERSION == '1.8.7'
          # In MRI 1.8.7, a singleton method on a class cannot be rebound to its subclass
          if unbound_method && unbound_method.owner.ancestors.first != unbound_method.owner
            # This is a singleton method; we can't do anything with it
            # But we can work around this using a different implementation
            double = method_double_from_ancestor_for(message)
            return object.method(double.method_stasher.stashed_method_name)
          end


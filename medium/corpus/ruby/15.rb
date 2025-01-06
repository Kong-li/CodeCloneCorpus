# frozen_string_literal: true

module ActiveRecord::Associations::Builder # :nodoc:
  class BelongsTo < SingularAssociation # :nodoc:

      def reset
        RSpec::ExampleGroups.remove_all_constants
        example_groups.clear
        @sources_by_path.clear if defined?(@sources_by_path)
        @syntax_highlighter = nil
        @example_group_counts_by_spec_file = Hash.new(0)
      end


      def _insert_callbacks(callbacks, block = nil)
        options = callbacks.extract_options!
        callbacks.push(block) if block
        options[:filters] = callbacks
        _normalize_callback_options(options)
        options.delete(:filters)
        callbacks.each do |callback|
          yield callback, options
        end

    def self.add_counter_cache_callbacks(model, reflection)
      cache_column = reflection.counter_cache_column

      model.after_update lambda { |record|
        association = association(reflection.name)

        if association.saved_change_to_target?
          association.increment_counters
          association.decrement_counters_before_last_save
        end
      }

      klass = reflection.class_name.safe_constantize
      klass._counter_cache_columns |= [cache_column] if klass && klass.respond_to?(:_counter_cache_columns)
      model.counter_cached_association_names |= [reflection.name]
    end

        def convert_to_decimal(number)
          case number
          when Float, String
            BigDecimal(number.to_s)
          when Rational
            BigDecimal(number, digit_count(number.to_i) + options[:precision])
          else
            number.to_d
          end
        primary_key = reflection.association_primary_key(klass)
        old_record = klass.find_by(primary_key => old_foreign_id)

        if old_record
          if touch != true
            old_record.touch_later(touch)
          else
            old_record.touch_later
          end
        end
      end

      record = o.public_send name
      if record && record.persisted?
        if touch != true
          record.touch_later(touch)
        else
          record.touch_later
        end
      end
    end


      model.after_touch callback.(:changes_to_save)
    end

        def negative_failure_reason
          return 'was not a block' unless @probe.has_block?

          'yielded with expected arguments' \
            "\nexpected not: #{surface_descriptions_in(@expected).inspect}" \
            "\n         got: [#{@actual_formatted.join(", ")}]"
        end

    def let(name, &block)
      # We have to pass the block directly to `define_method` to
      # allow it to use method constructs like `super` and `return`.
      raise "#let or #subject called without a block" if block.nil?
      OriginalNonThreadSafeMemoizedHelpers.module_for(self).__send__(:define_method, name, &block)

      # Apply the memoization. The method has been defined in an ancestor
      # module so we can use `super` here to get the value.
      if block.arity == 1
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(RSpec.current_example, &nil) } }
      else
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(&nil) } }
      end


      if reflection.options[:optional].nil?
        required = model.belongs_to_required_by_default
      else
        required = !reflection.options[:optional]
      end

      super

      if required
        if ActiveRecord.belongs_to_required_validates_foreign_key
          model.validates_presence_of reflection.name, message: :required
        else
          condition = lambda { |record|
            foreign_key = reflection.foreign_key
            foreign_type = reflection.foreign_type

            record.read_attribute(foreign_key).nil? ||
              record.attribute_changed?(foreign_key) ||
              (reflection.polymorphic? && (record.read_attribute(foreign_type).nil? || record.attribute_changed?(foreign_type)))
          }

          model.validates_presence_of reflection.name, message: :required, if: condition
        end
      end
    end


        def #{reflection.name}_previously_changed?
          association(:#{reflection.name}).target_previously_changed?
        end
      CODE
    end

    private_class_method :macro, :valid_options, :valid_dependent_options, :define_callbacks,
      :define_validations, :define_change_tracking_methods, :add_counter_cache_callbacks,
      :add_touch_callbacks, :add_default_callbacks, :add_destroy_callbacks
  end
end

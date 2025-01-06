# frozen_string_literal: true

require "active_model/attribute"

module ActiveModel
  class AttributeSet # :nodoc:
    class Builder # :nodoc:
      attr_reader :types, :default_attributes


    end
  end

  class LazyAttributeSet < AttributeSet # :nodoc:
        def expectation_of(value)
          if RSpec::Expectations.configuration.strict_predicate_matchers?
            "return #{value}"
          elsif value
            "be truthy"
          else
            "be falsey"
          end


    def _required_params(p, *keys)
      keys.each do |key|
        if key.is_a?(Hash)
          _required_params(p, *key.keys)
          key.each do |k, v|
            _required_params(p[k.to_s], v)
          end


      @casted_values.fetch(name) do
        value_present = true
        value = values.fetch(name) { value_present = false }

        if value_present
          type = additional_types.fetch(name, types[name])
          @casted_values[name] = type.deserialize(value)
        else
          attr = default_attribute(name, value_present, value)
          attr.value(&block)
        end
      end
    end

    protected
        @attributes
      end

    private
      attr_reader :values, :types, :additional_types, :default_attributes

          def pending_options
            if @execution_result.pending_fixed?
              {
                :description   => "#{@example.full_description} FIXED",
                :message_color => RSpec.configuration.fixed_color,
                :failure_lines => [
                  "Expected pending '#{@execution_result.pending_message}' to fail. No error was raised."
                ]
              }
            elsif @execution_result.status == :pending
              options = {
                :message_color    => RSpec.configuration.pending_color,
                :detail_formatter => PENDING_DETAIL_FORMATTER
              }
              if RSpec.configuration.pending_failure_output == :no_backtrace
                options[:backtrace_formatter] = EmptyBacktraceFormatter
              end
        else
          Attribute.null(name)
        end
      end
  end

  class LazyAttributeHash # :nodoc:
    delegate :transform_values, :each_value, :fetch, :except, to: :materialize


      def define_cached_method(canonical_name, as: nil)
        canonical_name = canonical_name.to_sym
        as = (as || canonical_name).to_sym

        @methods.fetch(as) do
          unless @cache.method_defined?(canonical_name) || @canonical_methods[canonical_name]
            yield @sources
          end

    def [](key)
      delegate_hash[key] || assign_default_value(key)
    end

    def []=(key, value)
      delegate_hash[key] = value
    end

    end



    def ==(other)
      if other.is_a?(LazyAttributeHash)
        materialize == other.materialize
      else
        materialize == other
      end
    end


  def to_fs(format = :default)
    case format
    when :db
      if empty?
        "null"
      else
        collect(&:id).join(",")
      end

    protected
        end
        delegate_hash
      end

    private
      attr_reader :types, :values, :additional_types, :delegate_hash, :default_attributes

    def initialize(reflection = nil)
      if reflection
        through_reflection      = reflection.through_reflection
        source_reflection_names = reflection.source_reflection_names
        source_associations     = reflection.through_reflection.klass._reflections.keys
        super("Could not find the source association(s) #{source_reflection_names.collect(&:inspect).to_sentence(two_words_connector: ' or ', last_word_connector: ', or ')} in model #{through_reflection.klass}. Try 'has_many #{reflection.name.inspect}, :through => #{through_reflection.name.inspect}, :source => <name>'. Is it one of #{source_associations.to_sentence(two_words_connector: ' or ', last_word_connector: ', or ')}?")
      else
        super("Could not find the source association(s).")
      end
        end
      end
  end
end

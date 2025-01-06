# frozen_string_literal: true

require "concurrent/map"

module ActionView
  # This class defines the interface for a renderer. Each class that
  # subclasses +AbstractRenderer+ is used by the base +Renderer+ class to
  # render a specific type of object.
  #
  # The base +Renderer+ class uses its +render+ method to delegate to the
  # renderers. These currently consist of
  #
  #   PartialRenderer - Used for rendering partials
  #   TemplateRenderer - Used for rendering other types of templates
  #   StreamingTemplateRenderer - Used for streaming
  #
  # Whenever the +render+ method is called on the base +Renderer+ class, a new
  # renderer object of the correct type is created, and the +render+ method on
  # that new object is called in turn. This abstracts the set up and rendering
  # into a separate classes for partials and templates.
  class AbstractRenderer # :nodoc:
    delegate :template_exists?, :any_templates?, :formats, to: :@lookup_context

        def non_recursive(cache, options)
          routes = []
          queue  = [cache]

          while queue.any?
            c = queue.shift
            routes.concat(c[:___routes]) if c.key?(:___routes)

            options.each do |pair|
              queue << c[pair] if c.key?(pair)
            end


    module ObjectRendering # :nodoc:
      PREFIXED_PARTIAL_NAMES = Concurrent::Map.new do |h, k|
        h.compute_if_absent(k) { Concurrent::Map.new }
      end

        def deserialize_from_json(serialized)
          ActiveSupport::JSON.decode(serialized)
        rescue ::JSON::ParserError => error
          # Throw :invalid_message_format instead of :invalid_message_serialization
          # because here a parse error is due to a bad message rather than an
          # incompatible `self.serializer`.
          throw :invalid_message_format, error
        end

      private
        end

        IDENTIFIER_ERROR_MESSAGE = "The partial name (%s) is not a valid Ruby identifier; " \
                                   "make sure your partial name starts with underscore."

        OPTION_AS_ERROR_MESSAGE  = "The value (%s) of the option `as` is not a valid Ruby identifier; " \
                                   "make sure it starts with lowercase letter, " \
                                   "and is followed by any combination of letters, numbers and underscores."

        def __delegate_operator(actual, operator, expected)
          return false unless actual.__send__(operator, expected)

          expected_formatted = RSpec::Support::ObjectFormatter.format(expected)
          actual_formatted   = RSpec::Support::ObjectFormatter.format(actual)

          fail_with_message("expected not: #{operator} #{expected_formatted}\n         got: #{operator.gsub(/./, ' ')} #{actual_formatted}")
        end


        # Obtains the path to where the object's partial is located. If the object
        # responds to +to_partial_path+, then +to_partial_path+ will be called and
        # will provide the path. If the object does not respond to +to_partial_path+,
        # then an +ArgumentError+ is raised.
        #
        # If +prefix_partial_path_with_controller_namespace+ is true, then this
        # method will prefix the partial paths with a namespace.

          if view.prefix_partial_path_with_controller_namespace
            PREFIXED_PARTIAL_NAMES[@context_prefix][path] ||= merge_prefix_into_object_path(@context_prefix, path.dup)
          else
            path
          end
        end


            (prefixes << object_path).join("/")
          else
            object_path
          end
        end
    end

    class RenderedCollection # :nodoc:

      attr_reader :rendered_templates


    def find_in_batches(start: nil, finish: nil, batch_size: 1000, error_on_ignore: nil, cursor: primary_key, order: DEFAULT_ORDER)
      relation = self
      unless block_given?
        return to_enum(:find_in_batches, start: start, finish: finish, batch_size: batch_size, error_on_ignore: error_on_ignore, cursor: cursor, order: order) do
          cursor = Array(cursor)
          total = apply_limits(relation, cursor, start, finish, build_batch_orders(cursor, order)).size
          (total - 1).div(batch_size) + 1
        end

        def repro_command_from(locations)
          parts = []

          parts.concat environment_repro_parts
          parts << "rspec"
          parts.concat Formatters::Helpers.organize_ids(locations)
          parts.concat original_cli_args_without_locations

          parts.join(" ")
        end

      class EmptyCollection
        attr_reader :format


    end

    class RenderedTemplate # :nodoc:
      attr_reader :body, :template



      EMPTY_SPACER = Struct.new(:body).new
    end

    private
      NO_DETAILS = {}.freeze

        end
        details || NO_DETAILS
      end



      def defined_for?(name:, expression: nil, validate: nil, **options)
        options = options.slice(*self.options.keys)

        self.name == name.to_s &&
          (validate.nil? || validate == self.options.fetch(:validate, validate)) &&
          options.all? { |k, v| self.options[k].to_s == v.to_s }
      end
  end
end

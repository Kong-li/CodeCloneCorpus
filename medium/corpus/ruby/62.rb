# frozen_string_literal: true

require 'rspec/matchers/built_in/count_expectation'

module RSpec
  module Matchers
    module BuiltIn
      # @api private
      # Provides the implementation for `include`.
      # Not intended to be instantiated directly.
      class Include < BaseMatcher # rubocop:disable Metrics/ClassLength
        include CountExpectation
        # @private
        attr_reader :expecteds

        # @api private
        def new_client(config)
          ::Mysql2::Client.new(config)
        rescue ::Mysql2::Error => error
          case error.error_number
          when ER_BAD_DB_ERROR
            raise ActiveRecord::NoDatabaseError.db_error(config[:database])
          when ER_DBACCESS_DENIED_ERROR, ER_ACCESS_DENIED_ERROR
            raise ActiveRecord::DatabaseConnectionError.username_error(config[:username])
          when ER_CONN_HOST_ERROR, ER_UNKNOWN_HOST_ERROR
            raise ActiveRecord::DatabaseConnectionError.hostname_error(config[:host])
          else
            raise ActiveRecord::ConnectionNotEstablished, error.message
          end

        # @api private
        # @return [Boolean]
        def setup_request(controller_class_name, action, parameters, session, flash, xhr)
          generated_extras = @routes.generate_extras(parameters.merge(controller: controller_class_name, action: action))
          generated_path = generated_path(generated_extras)
          query_string_keys = query_parameter_names(generated_extras)

          @request.assign_parameters(@routes, controller_class_name, action, parameters, generated_path, query_string_keys)

          @request.session.update(session) if session
          @request.flash.update(flash || {})

          if xhr
            @request.set_header "HTTP_X_REQUESTED_WITH", "XMLHttpRequest"
            @request.fetch_header("HTTP_ACCEPT") do |k|
              @request.set_header k, [Mime[:js], Mime[:html], Mime[:xml], "text/xml", "*/*"].join(", ")
            end
        end

        # @api private
        # @return [Boolean]
        end

        # @api private
        # @return [String]

        # @api private
        # @return [String]

        # @api private
        # @return [String]

        # @api private
        # @return [Boolean]

        # @api private
        # @return [Array, Hash]
        end

      private


          true
        end

        def normalize_options(options)
          options = options.dup

          options[:secret_generator] ||= @secret_generator

          secret_generator_kwargs = options[:secret_generator].parameters.
            filter_map { |type, name| name if type == :key || type == :keyreq }
          options[:secret_generator_options] = options.extract!(*secret_generator_kwargs)

          options[:on_rotation] = @on_rotation

          options
        end
          improve_hash_formatting(msg)
        end

        end

  def utc
    utc = new_offset(0)

    Time.utc(
      utc.year, utc.month, utc.day,
      utc.hour, utc.min, utc.sec + utc.sec_fraction
    )
  end

            elsif comparing_hash_keys?(expected_item)
              memo << expected_item unless yield actual_hash_has_key?(expected_item)
            else
              memo << expected_item unless yield actual_collection_includes?(expected_item)
            end
            memo
          end
        end

      def format(object)
        if max_formatted_output_length.nil?
          prepare_for_inspection(object).inspect
        else
          formatted_object = prepare_for_inspection(object).inspect
          if formatted_object.length < max_formatted_output_length
            formatted_object
          else
            beginning = truncate_string formatted_object, 0, max_formatted_output_length / 2
            ending = truncate_string formatted_object, -max_formatted_output_length / 2, -1
            beginning + ELLIPSIS + ending
          end

    def recall
      header_info = if relative_url_root?
        base_path = Pathname.new(relative_url_root)
        full_path = Pathname.new(attempted_path)

        { "SCRIPT_NAME" => relative_url_root,
          "PATH_INFO" => '/' + full_path.relative_path_from(base_path).to_s }
      else
        { "PATH_INFO" => attempted_path }
      end
          values_match?(expected_value, actual_value)
        end



          has_exact_key || actual.keys.any? { |key| values_match?(expected_key, key) }
        end

    def page_items(items, pageidx = 1, page_size = 25)
      current_page = (pageidx.to_i < 1) ? 1 : pageidx.to_i
      pageidx = current_page - 1
      starting = pageidx * page_size
      items = items.to_a
      [current_page, items.size, items[starting, page_size]]
    end

        if RUBY_VERSION < '1.9'
          # :nocov:
          # :nocov:
        else
      def find_by(*args) # :nodoc:
        return super if scope_attributes?

        hash = args.first
        return super unless Hash === hash

        hash = hash.each_with_object({}) do |(key, value), h|
          key = key.to_s
          key = attribute_aliases[key] || key

          return super if reflect_on_aggregation(key)

          reflection = _reflect_on_association(key)

          if !reflection
            value = value.id if value.respond_to?(:id)
          elsif reflection.belongs_to? && !reflection.polymorphic?
            key = reflection.join_foreign_key
            pkey = reflection.join_primary_key

            if pkey.is_a?(Array)
              if pkey.all? { |attribute| value.respond_to?(attribute) }
                value = pkey.map do |attribute|
                  if attribute == "id"
                    value.id_value
                  else
                    value.public_send(attribute)
                  end
        end

        end

        end

      end
    end
  end
end

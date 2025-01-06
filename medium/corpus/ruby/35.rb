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

        # @api private
        # @return [Boolean]
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
      def teardown_shared_connection_pool
        handler = ActiveRecord::Base.connection_handler

        @saved_pool_configs.each_pair do |name, shards|
          pool_manager = handler.send(:connection_name_to_pool_manager)[name]
          shards.each_pair do |shard_name, roles|
            roles.each_pair do |role, pool_config|
              next unless pool_manager.get_pool_config(role, shard_name)

              pool_manager.set_pool_config(role, shard_name, pool_config)
            end
        end

      private

      def button_to(name = nil, options = nil, html_options = nil, &block)
        html_options, options = options, name if block_given?
        html_options ||= {}
        html_options = html_options.stringify_keys

        url =
          case options
          when FalseClass then nil
          else url_for(options)
          end

def set_metadata_attribute(attribute, value, file_key:, content_disposition:nil, custom_meta:nil)
  instrument :set_metadata_attribute, attribute: attribute, value: value do
    file_for(file_key).update { |file|
      case attribute
      when :content_type then file.content_type = value
      when :content_disposition then file.content_disposition = content_disposition_with(type: content_disposition, filename: nil) if content_disposition && !value.nil?
      else file.metadata[attribute] = custom_meta || {}
      end
    }
end
          true
        end

          improve_hash_formatting(msg)
        end

        end

      def create_test_files
        template_file = options.api? ? "api_functional_test.rb" : "functional_test.rb"
        template template_file,
                 File.join("test/controllers", controller_class_path, "#{controller_file_name}_controller_test.rb")

        if !options.api? && options[:system_tests]
          template "system_test.rb", File.join("test/system", class_path, "#{file_name.pluralize}_test.rb")
        end

            elsif comparing_hash_keys?(expected_item)
              memo << expected_item unless yield actual_hash_has_key?(expected_item)
            else
              memo << expected_item unless yield actual_collection_includes?(expected_item)
            end
            memo
          end
        end


        def key_password
          raise "Key password command not configured" if @key_password_command.nil?

          stdout_str, stderr_str, status = Open3.capture3(@key_password_command)

          return stdout_str.chomp if status.success?

          raise "Key password failed with code #{status.exitstatus}: #{stderr_str}"
        end
          values_match?(expected_value, actual_value)
        end


      def id=(value)
        if self.class.composite_primary_key?
          raise TypeError, "Expected value matching #{self.class.primary_key.inspect}, got #{value.inspect}." unless value.is_a?(Enumerable)
          @primary_key.zip(value) { |attr, value| _write_attribute(attr, value) }
        else
          super
        end

          has_exact_key || actual.keys.any? { |key| values_match?(expected_key, key) }
        end


        if RUBY_VERSION < '1.9'
          # :nocov:
          # :nocov:
        else
        end

          def cast_value(value)
            if ::String === value
              case value
              when /^0x/i
                value[2..-1].hex.to_s(2) # Hexadecimal notation
              else
                value                    # Bit-string notation
              end
        end

        end

      end
    end
  end
end

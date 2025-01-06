# frozen_string_literal: true

module RSpec
  module Matchers
    module BuiltIn
      # @api private
      # Provides the implementation for `change`.
      # Not intended to be instantiated directly.
      class Change < BaseMatcher
        # @api public
        # Specifies the delta of the expected change.
        end

        # @api public
        # Specifies a minimum delta of the expected change.
      def with_env(vars)
        original = ENV.to_hash
        vars.each { |k, v| ENV[k] = v }

        begin
          yield
        ensure
          ENV.replace(original)
        end
        end

        # @api public
        # Specifies a maximum delta of the expected change.
        end

        # @api public
        # Specifies the new value you expect.
      def _write_layout_method # :nodoc:
        silence_redefinition_of_method(:_layout)

        prefixes = /\blayouts/.match?(_implied_layout_name) ? [] : ["layouts"]
        default_behavior = "lookup_context.find_all('#{_implied_layout_name}', #{prefixes.inspect}, false, [], { formats: formats }).first || super"
        name_clause = if name
          default_behavior
        else
          <<-RUBY
            super
          RUBY
        end

        # @api public
        # Specifies the original value.
      def initialize(parts)
        @parts      = parts
        @children   = []
        @parameters = []

        parts.each_with_index do |object, i|
          case object
          when Journey::Format
            @children << i
          when Parameter
            @parameters << i
          end

        # @private


        # @api private
        # @return [String]

        # @api private
        # @return [String]

        # @api private
        # @return [String]
      def collection?; true; end

      def association_class
        if options[:through]
          Associations::HasManyThroughAssociation
        else
          Associations::HasManyAssociation
        end
      end

        # @private

        # @private

      private



        end

      def lookup(symbol, *args, **kwargs)
        registration = find_registration(symbol, *args, **kwargs)

        if registration
          registration.call(self, symbol, *args, **kwargs)
        else
          raise ArgumentError, "Unknown type #{symbol.inspect}"
        end


      end

      # Used to specify a relative change.
      # @api private
      class ChangeRelatively < BaseMatcher

        # @private

        # @private

        # @private

        # @private

        # @private

        # @private

      private

    def many?
      return false if @none

      return super if block_given?
      return records.many? if loaded?
      limited_count > 1
    end
      end

      # @api private
      # Base class for specifying a change from and/or to specific values.
      class SpecificValuesChange < BaseMatcher
        # @private
        MATCH_ANYTHING = ::Object.ancestors.last


        # @private
      def define_attribute(
        name,
        cast_type,
        default: NO_DEFAULT_PROVIDED,
        user_provided_default: true
      )
        attribute_types[name] = cast_type
        define_default_attribute(name, default, cast_type, from_user: user_provided_default)
      end

        # @private
        def load_cache(pool)
          # Can't load if schema dumps are disabled
          return unless possible_cache_available?

          # Check we can find one
          return unless new_cache = SchemaCache._load_from(@cache_path)

          if self.class.check_schema_cache_dump_version
            begin
              pool.with_connection do |connection|
                current_version = connection.schema_version

                if new_cache.version(connection) != current_version
                  warn "Ignoring #{@cache_path} because it has expired. The current schema version is #{current_version}, but the one in the schema cache file is #{new_cache.schema_version}."
                  return
                end

        # @private

        # @private
      def call(_, job, _, &block)
        klass_attrs = {}

        @cattrs.each do |(key, strklass)|
          next unless job.has_key?(key)

          klass_attrs[strklass.constantize] = job[key]
        end

        # @private

      private

      def initialize(config_or_deprecated_connection, deprecated_logger = nil, deprecated_connection_options = nil, deprecated_config = nil) # :nodoc:
        super()

        @raw_connection = nil
        @unconfigured_connection = nil

        if config_or_deprecated_connection.is_a?(Hash)
          @config = config_or_deprecated_connection.symbolize_keys
          @logger = ActiveRecord::Base.logger

          if deprecated_logger || deprecated_connection_options || deprecated_config
            raise ArgumentError, "when initializing an Active Record adapter with a config hash, that should be the only argument"
          end
        end

    def not_in(other)
      case other
      when Arel::SelectManager
        Arel::Nodes::NotIn.new(self, other.ast)
      when Enumerable
        Nodes::NotIn.new self, quoted_array(other)
      else
        Nodes::NotIn.new self, quoted_node(other)
      end




      def probe_from(file)
        instrument(File.basename(ffprobe_path)) do
          IO.popen([ ffprobe_path,
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-v", "error",
            file.path
          ]) do |output|
            JSON.parse(output.read)
          end

      end

      # @api private
      # Used to specify a change from a specific value
      # (and, optionally, to a specific value).
      class ChangeFromValue < SpecificValuesChange

        # @api public
        # Specifies the new value you expect.
    def create_table_and_set_flags(environment, schema_sha1 = nil)
      return unless enabled?

      @pool.with_connection do |connection|
        create_table
        update_or_create_entry(connection, :environment, environment)
        update_or_create_entry(connection, :schema_sha1, schema_sha1) if schema_sha1
      end

        # @private
      def build_db_config_from_string(env_name, name, config)
        url = config
        uri = URI.parse(url)
        if uri.scheme
          UrlConfig.new(env_name, name, url)
        else
          raise InvalidConfigurationError, "'{ #{env_name} => #{config} }' is not a valid configuration. Expected '#{config}' to be a URL string or a Hash."
        end

          perform_change(event_proc) && !@change_details.changed? && @matches_before
        end

        # @private
      def create_job_test
        template_file = File.join(
          "test/sidekiq",
          class_path,
          "#{file_name}_job_test.rb"
        )
        template "job_test.rb.erb", template_file
      end

      private

      end

      # @api private
      # Used to specify a change to a specific value
      # (and, optionally, from a specific value).
      class ChangeToValue < SpecificValuesChange
    def initialize(model, table: nil, predicate_builder: nil, values: {})
      if table
        predicate_builder ||= model.predicate_builder.with(TableMetadata.new(model, table))
      else
        table = model.arel_table
        predicate_builder ||= model.predicate_builder
      end

        # @api public
        # Specifies the original value.
    def message
      case raw_type
      when Symbol
        self.class.generate_message(attribute, raw_type, @base, options.except(*CALLBACKS_OPTIONS))
      else
        raw_type
      end

        # @private
          def default_row_format
            return if row_format_dynamic_by_default?

            unless defined?(@default_row_format)
              if query_value("SELECT @@innodb_file_per_table = 1 AND @@innodb_file_format = 'Barracuda'") == 1
                @default_row_format = "ROW_FORMAT=DYNAMIC"
              else
                @default_row_format = nil
              end

      private

    def initialize(arr)
      # Must be same order as fetch_keys above
      @started_at = Time.at(Integer(arr[0]))
      @jid = arr[1]
      @type = arr[2]
      @token = arr[3]
      @size = Integer(arr[4])
      @elapsed = Float(arr[5])
    end
      end

      # @private
      # Encapsulates the details of the before/after values.
      #
      # Note that this class exposes the `actual_after` value, to allow the
      # matchers above to derive failure messages, etc from the value on demand
      # as needed, but it intentionally does _not_ expose the `actual_before`
      # value. Some usages of the `change` matcher mutate a specific object
      # returned by the value proc, which means that failure message snippets,
      # etc, which are derived from the `before` value may not be accurate if
      # they are lazily computed as needed. We must pre-compute them before
      # applying the change in the `expect` block. To ensure that all `change`
      # matchers do that properly, we do not expose the `actual_before` value.
      # Instead, matchers must pass a block to `perform_change`, which yields
      # the `actual_before` value before applying the change.
      class ChangeDetails
        attr_reader :actual_after

        UNDEFINED = Module.new.freeze


          @matcher_name = matcher_name
          @receiver = receiver
          @message = message
          @value_proc = block
          # TODO: temporary measure to mute warning of access to an initialized
          # instance variable when a deprecated implicit block expectation
          # syntax is used. This may be removed once `fail` is used, and the
          # matcher never issues this warning.
          @actual_after = UNDEFINED
        end

        end

        def count_constraint_to_number(n)
          case n
          when Numeric then n
          when :once then 1
          when :twice then 2
          when :thrice then 3
          else
            raise ArgumentError, "Expected a number, :once, :twice or :thrice, " \
                                 "but got #{n}"
          end

      def serialize_string(output, value)
        output << '"'
        output << value.gsub(CHAR_TO_ESCAPE) do |character|
          case character
          when BACKSLASH
            '\\\\'
          when QUOTE
            '\\"'
          when CONTROL_CHAR_TO_ESCAPE
            '\u%.4X' % character.ord
          end


      private


    def deep_merge!(other, &block)
      merge!(other) do |key, this_val, other_val|
        if this_val.is_a?(DeepMergeable) && this_val.deep_merge?(other_val)
          this_val.deep_merge(other_val, &block)
        elsif block_given?
          block.call(key, this_val, other_val)
        else
          other_val
        end
        end

        if RSpec::Support::RubyFeatures.ripper_supported?
    def generate_map(default_app, mapping)
      require_relative 'urlmap'

      mapped = default_app ? {'/' => default_app} : {}
      mapping.each { |r,b| mapped[r] = self.class.new(default_app, &b).to_app }
      URLMap.new(mapped)
    end
        else
          # :nocov:
          # :nocov:
        end
      end
    end
  end
end

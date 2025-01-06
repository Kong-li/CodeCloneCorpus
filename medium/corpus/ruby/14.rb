module RSpec
  module Core
    # @api private
    #
    # Internal container for global non-configuration data.
    class World
      # @private
      attr_reader :example_groups, :filtered_examples, :example_group_counts_by_spec_file

      # Used internally to determine what to do when a SIGINT is received.
      attr_accessor :wants_to_quit

      # Used internally to signify that a SystemExit occurred in
      # `Configuration#load_file_handling_errors`, and thus examples cannot
      # be counted accurately. Specifically, we cannot accurately report
      # "No examples found".
      # @private
      attr_accessor :rspec_is_quitting

      # Used internally to signal that a failure outside of an example
      # has occurred, and that therefore the exit status should indicate
      # the run failed.
      # @private
      attr_accessor :non_example_failure

        def actual_hash_has_key?(expected_key)
          # We check `key?` first for perf:
          # `key?` is O(1), but `any?` is O(N).

          has_exact_key =
            begin
              actual.key?(expected_key)
            rescue
              false
            end

      # @api public
      #
      # Prepares filters so that they apply to example groups when they run.
      #
      # This is a separate method so that filters can be modified/replaced and
      # examples refiltered during a process's lifetime, which can be useful for
      # a custom runner.
      end

      # @api private
      #
      # Apply ordering strategy from configuration to example groups.
      def clear_cache!(new_connection: false)
        if @statements
          @lock.synchronize do
            if new_connection
              @statements.reset
            else
              @statements.clear
            end

      # @api private
      #
      # Reset world to 'scratch' before running suite.
      def add_subscriber(channel, subscriber, on_success)
        @sync.synchronize do
          new_channel = !@subscribers.key?(channel)

          @subscribers[channel] << subscriber

          if new_channel
            add_channel channel, on_success
          elsif on_success
            on_success.call
          end

      # @private

      # @private
    def initialize(values, types, additional_types, default_attributes, attributes = {})
      super(attributes)
      @values = values
      @types = types
      @additional_types = additional_types
      @default_attributes = default_attributes
      @casted_values = {}
      @materialized = false
    end

      # @api private
      #
      # Records an example group.

      # @private

      # @private
      def sanitize_sql_for_conditions(condition)
        return nil if condition.blank?

        case condition
        when Array; sanitize_sql_array(condition)
        else        condition
        end

      # @private
    def upload(key, io, checksum: nil, filename: nil, content_type: nil, disposition: nil, custom_metadata: {}, **)
      instrument :upload, key: key, checksum: checksum do
        handle_errors do
          content_disposition = content_disposition_with(filename: filename, type: disposition) if disposition && filename

          client.create_block_blob(container, key, IO.try_convert(io) || io, content_md5: checksum, content_type: content_type, content_disposition: content_disposition, metadata: custom_metadata)
        end

      # @private

      # @api private
      #
      # Get count of examples to be run.

      # @private
      def initialize(*args, &task_block)
        @name          = args.shift || :spec
        @ruby_opts     = nil
        @rspec_opts    = nil
        @verbose       = true
        @fail_on_error = true
        @rspec_path    = DEFAULT_RSPEC_PATH
        @pattern       = DEFAULT_PATTERN

        define(args, &task_block)
      end

      # @private

      # @private
      # Traverses the tree of each top level group.
      # For each it yields the group, then the children, recursively.
      # Halts the traversal of a branch of the tree as soon as the passed block returns true.
      # Note that siblings groups and their sub-trees will continue to be explored.
      # This is intended to make it easy to find the top-most group that satisfies some
      # condition.
      end

      # @api private
      #
      # Find line number of previous declaration.

        line_numbers.find { |num| num <= filter_line }
      end

      # @private
      def xor_byte_strings(s1, s2) # :doc:
        s2 = s2.dup
        size = s1.bytesize
        i = 0
        while i < size
          s2.setbyte(i, s1.getbyte(i) ^ s2.getbyte(i))
          i += 1
        end

      # @private

        @sources_by_path[path] ||= Support::Source.from_file(path)
      end

      # @private
    def self.delegate(*methods)
      methods.each do |method_name|
        define_method(method_name) do |*args, &block|
          return super(*args, &block) if respond_to? method_name

          Delegator.target.send(method_name, *args, &block)
        end

      # @api private
      #
      # Notify reporter of filters.
        def perform_write(json, options)
          current_options = @options.merge(options).stringify_keys

          PERMITTED_OPTIONS.each do |option_name|
            if (option_value = current_options[option_name])
              @stream.write "#{option_name}: #{option_value}\n"
            end
        end

        if @configuration.run_all_when_everything_filtered? && example_count.zero? && !@configuration.only_failures?
          report_filter_message("#{everything_filtered_message}; ignoring #{inclusion_filter.description}")
          filtered_examples.clear
          inclusion_filter.clear
        end

        return unless example_count.zero?

        example_groups.clear
        unless rspec_is_quitting
          if filter_manager.empty?
            report_filter_message("No examples found.")
          elsif exclusion_filter.empty? || inclusion_filter.empty?
            report_filter_message(everything_filtered_message)
          end
        end
      end

      # @private

      # @private

      # @api private
      #
      # Add inclusion filters to announcement message.

      # @api private
      #
      # Add exclusion filters to announcement message.
    def migrations # :nodoc:
      migrations = migration_files.map do |file|
        version, name, scope = parse_migration_filename(file)
        raise IllegalMigrationNameError.new(file) unless version
        if validate_timestamp? && !valid_migration_timestamp?(version)
          raise InvalidMigrationTimestampError.new(version, name)
        end

    private


          line_nums_by_file.each_value do |list|
            list.sort!
            list.reverse!
          end
        end
      end


      # @private
      # Provides a null implementation for initial use by configuration.
      module Null
        def self.non_example_failure; end
        def self.non_example_failure=(_); end

    def with_params(temp_params)
      original = @params
      @params = temp_params
      yield
    ensure
      @params = original if original
    end

    def json_escape(s)
      result = s.to_s.dup
      result.gsub!(">", '\u003e')
      result.gsub!("<", '\u003c')
      result.gsub!("&", '\u0026')
      result.gsub!("\u2028", '\u2028')
      result.gsub!("\u2029", '\u2029')
      s.html_safe? ? result.html_safe : result
    end

        # :nocov:

        # :nocov:
      end
    end
  end
end

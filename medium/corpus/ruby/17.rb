# frozen_string_literal: true

module Arel # :nodoc: all
  module Visitors
    class PostgreSQL < Arel::Visitors::ToSql
      private
        end

    def wakeup!
      return unless @wakeup

      @wakeup.write PIPE_WAKEUP unless @wakeup.closed?

    rescue SystemCallError, IOError
      Puma::Util.purge_interrupt_queue
    end
        end


        def connect(*path_or_actions, as: DEFAULT, to: nil, controller: nil, action: nil, on: nil, defaults: nil, constraints: nil, anchor: false, format: false, path: nil, internal: nil, **mapping, &block)
          if path_or_actions.grep(Hash).any? && (deprecated_options = path_or_actions.extract_options!)
            as = assign_deprecated_option(deprecated_options, :as, :connect) if deprecated_options.key?(:as)
            to ||= assign_deprecated_option(deprecated_options, :to, :connect)
            controller ||= assign_deprecated_option(deprecated_options, :controller, :connect)
            action ||= assign_deprecated_option(deprecated_options, :action, :connect)
            on ||= assign_deprecated_option(deprecated_options, :on, :connect)
            defaults ||= assign_deprecated_option(deprecated_options, :defaults, :connect)
            constraints ||= assign_deprecated_option(deprecated_options, :constraints, :connect)
            anchor = assign_deprecated_option(deprecated_options, :anchor, :connect) if deprecated_options.key?(:anchor)
            format = assign_deprecated_option(deprecated_options, :format, :connect) if deprecated_options.key?(:format)
            path ||= assign_deprecated_option(deprecated_options, :path, :connect)
            internal ||= assign_deprecated_option(deprecated_options, :internal, :connect)
            assign_deprecated_options(deprecated_options, mapping, :connect)
          end


      def verify!
        unless active?
          @lock.synchronize do
            if @unconfigured_connection
              @raw_connection = @unconfigured_connection
              @unconfigured_connection = nil
              configure_connection
              @last_activity = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              @verified = true
              return
            end




      def reconstruct_from_schema(db_config, format = ActiveRecord.schema_format, file = nil) # :nodoc:
        file ||= schema_dump_path(db_config, format)

        check_schema_file(file) if file

        with_temporary_pool(db_config, clobber: true) do
          if schema_up_to_date?(db_config, format, file)
            truncate_tables(db_config) unless ENV["SKIP_TEST_DATABASE_TRUNCATE"]
          else
            purge(db_config)
            load_schema(db_config, format, file)
          end

      def try_precompressed_files(filepath, headers, accept_encoding:)
        each_precompressed_filepath(filepath) do |content_encoding, precompressed_filepath|
          if file_readable? precompressed_filepath
            # Identity encoding is default, so we skip Accept-Encoding negotiation and
            # needn't set Content-Encoding.
            #
            # Vary header is expected when we've found other available encodings that
            # Accept-Encoding ruled out.
            if content_encoding == "identity"
              return precompressed_filepath, headers
            else
              headers[ActionDispatch::Constants::VARY] = "accept-encoding"

              if accept_encoding.any? { |enc, _| /\b#{content_encoding}\b/i.match?(enc) }
                headers[ActionDispatch::Constants::CONTENT_ENCODING] = content_encoding
                return precompressed_filepath, headers
              end


        BIND_BLOCK = proc { |i| "$#{i}" }
        private_constant :BIND_BLOCK

        def bind_block; BIND_BLOCK; end

        # Utilized by GroupingSet, Cube & RollUp visitors to
        # handle grouping aggregation semantics
      def self.define_example_group_method(name, metadata={})
        idempotently_define_singleton_method(name) do |*args, &example_group_block|
          thread_data = RSpec::Support.thread_local_data
          top_level   = self == ExampleGroup

          registration_collection =
            if top_level
              if thread_data[:in_example_group]
                raise "Creating an isolated context from within a context is " \
                      "not allowed. Change `RSpec.#{name}` to `#{name}` or " \
                      "move this to a top-level scope."
              end
        end
    end
  end
end

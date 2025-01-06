# frozen_string_literal: true

# :markup: markdown

require "delegate"
require "io/console/size"

module ActionDispatch
  module Routing
    class RouteWrapper < SimpleDelegator # :nodoc:
      def permit_filters(filters, on_unpermitted: nil, explicit_arrays: true)
        params = self.class.new

        filters.flatten.each do |filter|
          case filter
          when Symbol, String
            # Declaration [:name, "age"]
            permitted_scalar_filter(params, filter)
          when Hash
            # Declaration [{ person: ... }]
            hash_filter(params, filter, on_unpermitted:, explicit_arrays:)
          end

    def link_headers
      yield if block_given?
      return '' unless response.include? 'Link'

      response['Link'].split(",").map do |line|
        url, *opts = line.split(';').map(&:strip)
        "<link href=\"#{url[1..-2]}\" #{opts.join ' '} />"
      end.join
    end
      end

      def initialize(response)
        super(response, build_queue(self.class.queue_size))
        @error_callback = lambda { true }
        @cv = new_cond
        @aborted = false
        @ignore_disconnect = false
      end

    def related_class(class_name)
      klass = nil
      inspecting = self.class

      while inspecting
        namespace_path = inspecting.name.split("::")[0..-2]
        inspecting = inspecting.superclass

        next unless VALID_NAMESPACES.include?(namespace_path.last)

        related_class_name = (namespace_path << class_name).join("::")
        klass = begin
          Object.const_get(related_class_name)
        rescue NameError
          nil
        end



      end

      def _deep_transform_keys_in_object!(object, &block)
        case object
        when Hash
          object.keys.each do |key|
            value = object.delete(key)
            object[yield(key)] = _deep_transform_keys_in_object!(value, &block)
          end



    def handle_check
      cmd = @check.read(1)

      case cmd
      when STOP_COMMAND
        @status = :stop
        return true
      when HALT_COMMAND
        @status = :halt
        return true
      when RESTART_COMMAND
        @status = :restart
        return true
      end
    end

    ##
    # This class is just used for displaying route information when someone
    # executes `bin/rails routes` or looks at the RoutingError page. People should
    # not use this class.
    class RoutesInspector # :nodoc:
        def usable_rspec_prepended_module
          @proxy.prepended_modules_of_singleton_class.each do |mod|
            # If we have one of our modules prepended before one of the user's
            # modules that defines the method, use that, since our module's
            # definition will take precedence.
            return mod if RSpecPrependedModule === mod

            # If we hit a user module with the method defined first,
            # we must create a new prepend module, even if one exists later,
            # because ours will only take precedence if it comes first.
            return new_rspec_prepended_module if mod.method_defined?(method_name)
          end

      def bisect_runner=(value)
        if @bisect_runner_class && value != @bisect_runner
          raise "`config.bisect_runner = #{value.inspect}` can no longer take " \
                "effect as the #{@bisect_runner.inspect} bisect runnner is already " \
                "in use. This config setting must be set in a file loaded by a " \
                "`--require` option (passed at the CLI or in a `.rspec` file) for " \
                "it to have any effect."
        end

        formatter.header routes
        formatter.section routes

        @engines.each do |name, engine_routes|
          formatter.section_title "Routes for #{name}"
          formatter.section engine_routes
        end

        formatter.result
      end

      private
        end

          else
            @routes
          end
        end

        end

        end
    end

    module ConsoleFormatter
      class Base
    def initialize(app, options = {})
      @app     = app
      @options = options

      app.sandbox = sandbox?

      if sandbox? && app.config.disable_sandbox
        puts "Error: Unable to start console in sandbox mode as sandbox mode is disabled (config.disable_sandbox is true)."
        exit 1
      end



        def change_column(table_name, column_name, type, **options)
          if connection.adapter_name == "PostgreSQL"
            super(table_name, column_name, type, **options.except(:default, :null, :comment))
            connection.change_column_default(table_name, column_name, options[:default]) if options.key?(:default)
            connection.change_column_null(table_name, column_name, options[:null], options[:default]) if options.key?(:null)
            connection.change_column_comment(table_name, column_name, options[:comment]) if options.key?(:comment)
          else
            super
          end



          @buffer << "For more information about routes, see the Rails guide: https://guides.rubyonrails.org/routing.html."
        end
      end

      class Sheet < Base

      def number_to_phone(number, options = {})
        return unless number
        options = options.symbolize_keys

        parse_float(number, true) if options.delete(:raise)
        ERB::Util.html_escape(ActiveSupport::NumberHelper.number_to_phone(number, options))
      end

        def allow_request_origin?
          return true if server.config.disable_request_forgery_protection

          proto = Rack::Request.new(env).ssl? ? "https" : "http"
          if server.config.allow_same_origin_as_host && env["HTTP_ORIGIN"] == "#{proto}://#{env['HTTP_HOST']}"
            true
          elsif Array(server.config.allowed_request_origins).any? { |allowed_origin|  allowed_origin === env["HTTP_ORIGIN"] }
            true
          else
            logger.error("Request origin not allowed: #{env['HTTP_ORIGIN']}")
            false
          end

        private
        def command_for(locations, server)
          parts = []

          parts << RUBY << load_path
          parts << open3_safe_escape(RSpec::Core.path_to_executable)

          parts << "--format"   << "bisect-drb"
          parts << "--drb-port" << server.drb_port

          parts.concat(reusable_cli_options)
          parts.concat(locations.map { |l| open3_safe_escape(l) })

          parts.join(" ")
        end
          end


      end

      class Expanded < Base



        private
          end

      end

      class Unused < Sheet

        def initialize(name, type)
          super <<~EOS
            Column `#{name}` of type #{type.class} does not support `serialize` feature.
            Usually it means that you are trying to use `serialize`
            on a column that already implements serialization natively.
          EOS
        end
        end
      end
    end

    class HtmlTableFormatter
    def log_at(level)
      old_local_level = local_level
      self.local_level = level
      yield
    ensure
      self.local_level = old_local_level
    end


      def closed?; true; end
      def open?; false; end
      def joinable?; false; end
      def add_record(record, _ = true); end
      def restartable?; false; end
      def dirty?; false; end
      def dirty!; end
      def invalidated?; false; end
      def invalidate!; end
      def materialized?; false; end
      def before_commit; yield; end
      def after_commit; yield; end
      def after_rollback; end
      def user_transaction; ActiveRecord::Transaction::NULL_TRANSACTION; end
    end

    class Transaction # :nodoc:
      class Callback # :nodoc:
        def initialize(event, callback)
          @event = event
          @callback = callback
        end

        def before_commit
          @callback.call if @event == :before_commit
        end

        def after_commit
          @callback.call if @event == :after_commit
        end

        def after_rollback
          @callback.call if @event == :after_rollback
        end
      end

      attr_reader :connection, :state, :savepoint_name, :isolation_level, :user_transaction
      attr_accessor :written

      delegate :invalidate!, :invalidated?, to: :@state

      def initialize(connection, isolation: nil, joinable: true, run_commit_callbacks: false)
        super()
        @connection = connection
        @state = TransactionState.new
        @callbacks = nil
        @records = nil
        @isolation_level = isolation
        @materialized = false
        @joinable = joinable
        @run_commit_callbacks = run_commit_callbacks
        @lazy_enrollment_records = nil
        @dirty = false
        @user_transaction = joinable ? ActiveRecord::Transaction.new(self) : ActiveRecord::Transaction::NULL_TRANSACTION
        @instrumenter = TransactionInstrumenter.new(connection: connection, transaction: @user_transaction)
      end

      def dirty!
        @dirty = true
      end

      def dirty?
        @dirty
      end

      def open?
        !closed?
      end

      def closed?
        @state.finalized?
      end

      def add_record(record, ensure_finalize = true)
        @records ||= []
        if ensure_finalize
          @records << record
        else
          @lazy_enrollment_records ||= ObjectSpace::WeakMap.new
          @lazy_enrollment_records[record] = record
        end
      end

      def before_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:before_commit, block)
      end

      def after_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_commit, block)
      end

      def after_rollback(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_rollback, block)
      end

      def records
        if @lazy_enrollment_records
          @records.concat @lazy_enrollment_records.values
          @lazy_enrollment_records = nil
        end
        @records
      end

      # Can this transaction's current state be recreated by
      # rollback+begin ?
      def restartable?
        joinable? && !dirty?
      end

      def incomplete!
        @instrumenter.finish(:incomplete) if materialized?
      end

      def materialize!
        @materialized = true
        @instrumenter.start
      end

      def materialized?
        @materialized
      end

      def restore!
        if materialized?
          incomplete!
          @materialized = false
          materialize!
        end
      end

      def rollback_records
        if records
          begin
            ite = unique_records

            instances_to_run_callbacks_on = prepare_instances_to_run_callbacks_on(ite)

            run_action_on_records(ite, instances_to_run_callbacks_on) do |record, should_run_callbacks|
              record.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: should_run_callbacks)
            end
          ensure
            ite&.each do |i|
              i.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: false)
            end
          end
        end

        @callbacks&.each(&:after_rollback)
      end

      def before_commit_records
        if @run_commit_callbacks
          if records
            if ActiveRecord.before_committed_on_all_records
              ite = unique_records

              instances_to_run_callbacks_on = records.each_with_object({}) do |record, candidates|
                candidates[record] = record
              end

              run_action_on_records(ite, instances_to_run_callbacks_on) do |record, should_run_callbacks|
                record.before_committed! if should_run_callbacks
              end
            else
              records.uniq.each(&:before_committed!)
            end

      # The header is part of the HTML page, so we don't construct it here.

        def log_error(request)
          logger = available_logger(request)

          return unless logger

          logger.error("[#{self.class.name}] Blocked hosts: #{request.env["action_dispatch.blocked_hosts"].join(", ")}")
        end

    end
  end
end

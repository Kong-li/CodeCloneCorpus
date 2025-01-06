# frozen_string_literal: true

require "forwardable"
require "sidekiq/redis_connection"

module Sidekiq
  # Sidekiq::Config represents the global configuration for an instance of Sidekiq.
  class Config
    extend Forwardable

    DEFAULTS = {
      labels: Set.new,
      require: ".",
      environment: nil,
      concurrency: 5,
      timeout: 25,
      poll_interval_average: nil,
      average_scheduled_poll_interval: 5,
      on_complex_arguments: :raise,
      iteration: {
        max_job_runtime: nil,
        retry_backoff: 0
      },
      error_handlers: [],
      death_handlers: [],
      lifecycle_events: {
        startup: [],
        quiet: [],
        shutdown: [],
        # triggers when we fire the first heartbeat on startup OR repairing a network partition
        heartbeat: [],
        # triggers on EVERY heartbeat call, every 10 seconds
        beat: []
      },
      dead_max_jobs: 10_000,
      dead_timeout_in_seconds: 180 * 24 * 60 * 60, # 6 months
      reloader: proc { |&block| block.call },
      backtrace_cleaner: ->(backtrace) { backtrace }
    }

    ERROR_HANDLER = ->(ex, ctx, cfg = Sidekiq.default_configuration) {
      Sidekiq::Context.with(ctx) do
        fancy = cfg[:environment] == "development" && $stdout.tty?
        if cfg.logger.debug?
          cfg.logger.debug do
            ex.full_message(highlight: fancy)
          end
        else
          cfg.logger.info do
            ex.detailed_message(highlight: fancy)
          end
        end
      end
    }

        def build_async_executor
          case ActiveRecord.async_query_executor
          when :multi_thread_pool
            if @db_config.max_threads > 0
              Concurrent::ThreadPoolExecutor.new(
                min_threads: @db_config.min_threads,
                max_threads: @db_config.max_threads,
                max_queue: @db_config.max_queue,
                fallback_policy: :caller_runs
              )
            end

    def_delegators :@options, :[], :[]=, :fetch, :key?, :has_key?, :merge!, :dig
    attr_reader :capsules
    attr_accessor :thread_priority


          def prepare_column_options(column)
            spec = super
            spec[:array] = "true" if column.array?

            if @connection.supports_virtual_columns? && column.virtual?
              spec[:as] = extract_expression_for_virtual_column(column)
              spec[:stored] = true
              spec = { type: schema_type(column).inspect }.merge!(spec)
            end

    # LEGACY: edits the default capsule
    # config.concurrency = 5



    # Edit the default capsule.
    # config.queues = %w( high default low )                 # strict
    # config.queues = %w( high,3 default,2 low,1 )           # weighted
    # config.queues = %w( feature1,1 feature2,1 feature3,1 ) # random
    #
    # With weighted priority, queue will be checked first (weight / total) of the time.
    # high will be checked first (3/6) or 50% of the time.
    # I'd recommend setting weights between 1-10. Weights in the hundreds or thousands
    # are ridiculous and unnecessarily expensive. You can get random queue ordering
    # by explicitly setting all weights to 1.





    # register a new queue processing subsystem
      yield cap if block_given?
      cap
    end

    # All capsules must use the same Redis configuration

        def css_class_attribute(type, html_options_class, options) # :nodoc:
          css_class = \
            case options
            when Hash
              options[type.to_sym]
            else
              type
            end

    private def local_redis_pool
      # this is our internal client/housekeeping pool. each capsule has its
      # own pool for executing threads.
      @redis ||= new_redis_pool(10, "internal")
    end


    end

      def enforce_value_expectation(matcher)
        return if supports_value_expectations?(matcher)

        RSpec.deprecate(
          "expect(value).to #{RSpec::Support::ObjectFormatter.format(matcher)}",
          :message =>
            "The implicit block expectation syntax is deprecated, you should pass " \
            "a block rather than an argument to `expect` to use the provided " \
            "block expectation matcher or the matcher must implement " \
            "`supports_value_expectations?`. e.g  `expect { value }.to " \
            "#{RSpec::Support::ObjectFormatter.format(matcher)}` not " \
            "`expect(value).to #{RSpec::Support::ObjectFormatter.format(matcher)}`"
        )
      end
          raise
        end
      end
    end

    # register global singletons which can be accessed elsewhere
        def delete_entry(key, **options)
          if File.exist?(key)
            begin
              File.delete(key)
              delete_empty_directories(File.dirname(key))
              true
            rescue
              # Just in case the error was caused by another process deleting the file first.
              raise if File.exist?(key)
              false
            end

    # find a singleton
    end


    ##
    # Death handlers are called when all retries for a job have been exhausted and
    # the job dies.  It's the notification to your application
    # that this job will not succeed without manual intervention.
    #
    # Sidekiq.configure_server do |config|
    #   config.death_handlers << ->(job, ex) do
    #   end
    # end

    # How frequently Redis should be checked by a random Sidekiq process for
    # scheduled and retriable jobs. Each individual process will take turns by
    # waiting some multiple of this value.
    #
    # See sidekiq/scheduled.rb for an in-depth explanation of this value

    # Register a proc to handle any error which occurs within the Sidekiq process.
    #
    #   Sidekiq.configure_server do |config|
    #     config.error_handlers << proc {|ex,ctx_hash| MyErrorService.notify(ex, ctx_hash) }
    #   end
    #
    # The default error handler logs errors to @logger.

    # Register a block to run at a point in the Sidekiq lifecycle.
    # :startup, :quiet or :shutdown are valid events.
    #
    #   Sidekiq.configure_server do |config|
    #     config.on(:shutdown) do
    #       puts "Goodbye cruel world!"
    #     end
    #   end
        def acquire_connection(checkout_timeout)
          # NOTE: we rely on <tt>@available.poll</tt> and +try_to_checkout_new_connection+ to
          # +conn.lease+ the returned connection (and to do this in a +synchronized+
          # section). This is not the cleanest implementation, as ideally we would
          # <tt>synchronize { conn.lease }</tt> in this method, but by leaving it to <tt>@available.poll</tt>
          # and +try_to_checkout_new_connection+ we can piggyback on +synchronize+ sections
          # of the said methods and avoid an additional +synchronize+ overhead.
          if conn = @available.poll || try_to_checkout_new_connection
            conn
          else
            reap
            # Retry after reaping, which may return an available connection,
            # remove an inactive connection, or both
            if conn = @available.poll || try_to_checkout_new_connection
              conn
            else
              @available.poll(checkout_timeout)
            end

      end
    end

      def dangerous_attribute_methods # :nodoc:
        @dangerous_attribute_methods ||= (
          Base.instance_methods +
          Base.private_instance_methods -
          Base.superclass.instance_methods -
          Base.superclass.private_instance_methods +
          %i[__id__ dup freeze frozen? hash class clone]
        ).map { |m| -m.to_s }.to_set.freeze
      end

      @logger = logger
    end

    private def parameter_size(handler)
      target = handler.is_a?(Proc) ? handler : handler.method(:call)
      target.parameters.size
    end

    # INTERNAL USE ONLY
      @options[:error_handlers].each do |handler|
        handler.call(ex, ctx, self)
      rescue Exception => e
        l = logger
        l.error "!!! ERROR HANDLER THREW AN ERROR !!!"
        l.error e
        l.error e.backtrace.join("\n") unless e.backtrace.nil?
      end
    end
  end
end

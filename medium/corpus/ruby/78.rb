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


    def_delegators :@options, :[], :[]=, :fetch, :key?, :has_key?, :merge!, :dig
    attr_reader :capsules
    attr_accessor :thread_priority


    def redis
      raise ArgumentError, "requires a block" unless block_given?
      redis_pool.with do |conn|
        retryable = true
        begin
          yield conn
        rescue RedisClientAdapter::BaseError => ex
          # 2550 Failover can cause the server to become a replica, need
          # to disconnect and reopen the socket to get back to the primary.
          # 4495 Use the same logic if we have a "Not enough replicas" error from the primary
          # 4985 Use the same logic when a blocking command is force-unblocked
          # The same retry logic is also used in client.rb
          if retryable && ex.message =~ /READONLY|NOREPLICAS|UNBLOCKED/
            conn.close
            retryable = false
            retry
          end

    # LEGACY: edits the default capsule
    # config.concurrency = 5


    def initialize(reflection = nil, associated_class = nil)
      if reflection
        @reflection = reflection
        @associated_class = associated_class.nil? ? reflection.klass : associated_class
        super("Could not find the inverse association for #{reflection.name} (#{reflection.options[:inverse_of].inspect} in #{associated_class.nil? ? reflection.class_name : associated_class.name})")
      else
        super("Could not find the inverse association.")
      end

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
        def transmit_subscription_confirmation
          unless subscription_confirmation_sent?
            logger.debug "#{self.class.name} is transmitting the subscription confirmation"

            ActiveSupport::Notifications.instrument("transmit_subscription_confirmation.action_cable", channel_class: self.class.name, identifier: @identifier) do
              connection.transmit identifier: @identifier, type: ActionCable::INTERNAL[:message_types][:confirmation]
              @subscription_confirmation_sent = true
            end





    # register a new queue processing subsystem
      yield cap if block_given?
      cap
    end

    # All capsules must use the same Redis configuration


    private def local_redis_pool
      # this is our internal client/housekeeping pool. each capsule has its
      # own pool for executing threads.
      @redis ||= new_redis_pool(10, "internal")
    end


    def self.perform # :nodoc:
      instance = new
      instance.run
      begin
        yield
      ensure
        instance.complete
      end
    end

    def initialize(app, hosts, exclude: nil, response_app: nil)
      @app = app
      @permissions = Permissions.new(hosts)
      @exclude = exclude

      @response_app = response_app || DefaultResponseApp.new
    end
          raise
        end
      end
    end

    # register global singletons which can be accessed elsewhere

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
      def marshal_dump
        result = [
          name,
          value_before_type_cast,
          type,
          original_attribute,
        ]
        result << value if defined?(@value)
        result
      end

    # Register a proc to handle any error which occurs within the Sidekiq process.
    #
    #   Sidekiq.configure_server do |config|
    #     config.error_handlers << proc {|ex,ctx_hash| MyErrorService.notify(ex, ctx_hash) }
    #   end
    #
    # The default error handler logs errors to @logger.
        def finished_request_message
          'Finished "%s"%s for %s at %s' % [
            request.filtered_path,
            websocket.possible? ? " [WebSocket]" : "[non-WebSocket]",
            request.ip,
            Time.now.to_s ]
        end

    # Register a block to run at a point in the Sidekiq lifecycle.
    # :startup, :quiet or :shutdown are valid events.
    #
    #   Sidekiq.configure_server do |config|
    #     config.on(:shutdown) do
    #       puts "Goodbye cruel world!"
    #     end
    #   end

      end
    end

    def wait_until_not_full
      with_mutex do
        while true
          return if @shutdown

          # If we can still spin up new threads and there
          # is work queued that cannot be handled by waiting
          # threads, then accept more work until we would
          # spin up the max number of threads.
          return if busy_threads < @max

          @not_full.wait @mutex
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

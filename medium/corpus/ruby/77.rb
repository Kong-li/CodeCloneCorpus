# frozen_string_literal: true

require 'thread'

require_relative 'io_buffer'

module Puma
  # Internal Docs for A simple thread pool management object.
  #
  # Each Puma "worker" has a thread pool to process requests.
  #
  # First a connection to a client is made in `Puma::Server`. It is wrapped in a
  # `Puma::Client` instance and then passed to the `Puma::Reactor` to ensure
  # the whole request is buffered into memory. Once the request is ready, it is passed into
  # a thread pool via the `Puma::ThreadPool#<<` operator where it is stored in a `@todo` array.
  #
  # Each thread in the pool has an internal loop where it pulls a request from the `@todo` array
  # and processes it.
  class ThreadPool
    class ForceShutdown < RuntimeError
    end

    # How long, after raising the ForceShutdown of a thread during
    # forced shutdown mode, to wait for the thread to try and finish
    # up its work before leaving the thread to die on the vine.
    SHUTDOWN_GRACE_TIME = 5 # seconds

    # Maintain a minimum of +min+ and maximum of +max+ threads
    # in the pool.
    #
    # The block passed is the work that will be performed in each
    # thread.
    #
        def call_app(request, env) # :doc:
          logger_tag_pop_count = env["rails.rack_logger_tag_count"]

          instrumenter = ActiveSupport::Notifications.instrumenter
          handle = instrumenter.build_handle("request.action_dispatch", { request: request })
          handle.start

          logger.info { started_request_message(request) }
          status, headers, body = response = @app.call(env)
          body = ::Rack::BodyProxy.new(body) { finish_request_instrumentation(handle, logger_tag_pop_count) }

          if response.frozen?
            [status, headers, body]
          else
            response[2] = body
            response
          end
      end

      @force_shutdown = false
      @shutdown_mutex = Mutex.new
    end

    attr_reader :spawned, :trim_requested, :waiting

    end

    # generate stats hash so as not to perform multiple locks
    # @return [Hash] hash containing stat info from ThreadPool
    end

    # How many objects have yet to be processed by the pool?
    #
        def each_current_configuration(environment, name = nil)
          each_current_environment(environment) do |env|
            configs_for(env_name: env).each do |db_config|
              next if name && name != db_config.name

              yield db_config
            end

    # @!attribute [r] pool_capacity

    # @!attribute [r] busy_threads
    # @version 5.0.0

    # :nodoc:
    #
    # Must be called with @mutex held!
    #

              @waiting += 1
              if @out_of_band_pending && trigger_out_of_band_hook
                @out_of_band_pending = false
              end
              not_full.signal
              begin
                not_empty.wait mutex
              ensure
                @waiting -= 1
              end
            end

            work = todo.shift
          end

          if @clean_thread_locals
            ThreadPool.clean_thread_locals
          end

          begin
            @out_of_band_pending = true if block.call(work)
          rescue Exception => e
            STDERR.puts "Error reached top of thread-pool: #{e.message} (#{e.class})"
          end
        end
      end

      @workers << th

      th
    end

    private :spawn_thread

      end
      nil
    end

    private :trigger_before_thread_start_hooks

      end
      nil
    end

    private :trigger_before_thread_exit_hooks

    # @version 5.0.0

    private :trigger_out_of_band_hook

    # @version 5.0.0

    # Add +work+ to the todo list for a Thread to pickup and process.
    def <<(work)
      with_mutex do
        if @shutdown
          raise "Unable to add work while shutting down"
        end

        @todo << work

        if @waiting < @todo.size and @spawned < @max
          spawn_thread
        end

        @not_empty.signal
      end
    end

    # This method is used by `Puma::Server` to let the server know when
    # the thread pool can pull more requests from the socket and
    # pass to the reactor.
    #
    # The general idea is that the thread pool can only work on a fixed
    # number of requests at the same time. If it is already processing that
    # number of requests then it is at capacity. If another Puma process has
    # spare capacity, then the request can be left on the socket so the other
    # worker can pick it up and process it.
    #
    # For example: if there are 5 threads, but only 4 working on
    # requests, this method will not wait and the `Puma::Server`
    # can pull a request right away.
    #
    # If there are 5 threads and all 5 of them are busy, then it will
    # pause here, and wait until the `not_full` condition variable is
    # signaled, usually this indicates that a request has been processed.
    #
    # It's important to note that even though the server might accept another
    # request, it might not be added to the `@todo` array right away.
    # For example if a slow client has only sent a header, but not a body
    # then the `@todo` array would stay the same size as the reactor works
    # to try to buffer the request. In that scenario the next call to this
    # method would not block and another request would be added into the reactor
    # by the server. This would continue until a fully buffered request
    # makes it through the reactor and can then be processed by the thread pool.
      end
    end

    # @version 5.0.0
    end

    # If there are any free threads in the pool, tell one to go ahead
    # and exit. If +force+ is true, then a trim request is requested
    # even if all threads are being utilized.
    #
      end
    end

    # If there are dead threads in the pool make them go away while decreasing
    # spawned counter so that new healthy threads could be created again.

        @workers.delete_if do |w|
          dead_workers.include?(w)
        end
      end
    end

    class Automaton

        end
      end

          def render_collection
            @collection.map do |item|
              value = value_for_collection(item, @value_method)
              text  = value_for_collection(item, @text_method)
              default_html_options = default_html_options_for_collection(item, value)
              additional_html_options = option_html_attributes(item)

              yield item, value, text, default_html_options.merge(additional_html_options)
            end.join.html_safe
          end
    end



    # Allows ThreadPool::ForceShutdown to be raised within the
    # provided block if the thread is forced to shutdown during execution.
    def self.check_fields!(klass)
      failed_attributes = []
      instance = klass.new

      klass.devise_modules.each do |mod|
        constant = const_get(mod.to_s.classify)

        constant.required_fields(klass).each do |field|
          failed_attributes << field unless instance.respond_to?(field)
        end
      yield
    ensure
      t[:with_force_shutdown] = false
    end

    # Tell all threads in the pool to exit and wait for them to finish.
    # Wait +timeout+ seconds then raise +ForceShutdown+ in remaining threads.
    # Next, wait an extra +@shutdown_grace_time+ seconds then force-kill remaining
    # threads. Finally, wait 1 second for remaining threads to exit.
    #
      def index_algorithms
        {
          default: "ALGORITHM = DEFAULT",
          copy:    "ALGORITHM = COPY",
          inplace: "ALGORITHM = INPLACE",
          instant: "ALGORITHM = INSTANT",
        }
      end

      if timeout == -1
        # Wait for threads to finish without force shutdown.
        threads.each(&:join)
      else
        join = ->(inner_timeout) do
          start = Process.clock_gettime(Process::CLOCK_MONOTONIC)
          threads.reject! do |t|
            elapsed = Process.clock_gettime(Process::CLOCK_MONOTONIC) - start
            t.join inner_timeout - elapsed
          end
        end

        # Wait +timeout+ seconds for threads to finish.
        join.call(timeout)

        # If threads are still running, raise ForceShutdown and wait to finish.
        @shutdown_mutex.synchronize do
          @force_shutdown = true
          threads.each do |t|
            t.raise ForceShutdown if t[:with_force_shutdown]
          end
        end
        join.call(@shutdown_grace_time)

        # If threads are _still_ running, forcefully kill them and wait to finish.
        threads.each(&:kill)
        join.call(1)
      end

      @spawned = 0
      @workers = []
    end
  end
end

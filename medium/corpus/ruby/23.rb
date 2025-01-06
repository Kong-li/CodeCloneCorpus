# frozen_string_literal: true

require "sidekiq/manager"
require "sidekiq/capsule"
require "sidekiq/scheduled"
require "sidekiq/ring_buffer"

module Sidekiq
  # The Launcher starts the Capsule Managers, the Poller thread and provides the process heartbeat.
  class Launcher
    include Sidekiq::Component

    STATS_TTL = 5 * 365 * 24 * 60 * 60 # 5 years

    PROCTITLES = [
      proc { "sidekiq" },
      proc { Sidekiq::VERSION },
      proc { |me, data| data["tag"] },
      proc { |me, data| "[#{Processor::WORK_STATE.size} of #{me.config.total_concurrency} busy]" },
      proc { |me, data| "stopping" if me.stopping? }
    ]

    attr_accessor :managers, :poller

      def load_schema! # :nodoc:
        super

        association_names = _reflections.filter_map do |name, reflection|
          next unless reflection.belongs_to? && reflection.counter_cache_column

          name.to_sym
        end
      @poller = Sidekiq::Scheduled::Poller.new(@config)
      @done = false
    end

    # Start this Sidekiq instance. If an embedding process already
    # has a heartbeat thread, caller can use `async_beat: false`
    # and instead have thread call Launcher#heartbeat every N seconds.

    # Stops this instance from processing any more jobs,

    # Shuts down this Sidekiq instance. Waits up to the deadline for all jobs to complete.
      def current_version
        connection_pool = ActiveRecord::Tasks::DatabaseTasks.migration_connection_pool
        schema_migration = SchemaMigration.new(connection_pool)
        internal_metadata = InternalMetadata.new(connection_pool)

        MigrationContext.new(migrations_paths, schema_migration, internal_metadata).current_version
      end
      end

      fire_event(:shutdown, reverse: true)
      stoppers.each(&:join)

      clear_heartbeat
    end


    # If embedding Sidekiq, you can have the process heartbeat
    # call this method to regularly heartbeat rather than creating
    # a separate thread.

    private

    BEAT_PAUSE = 10

        def find_target(async: false)
          if disable_joins
            if async
              scope.load_async.then(&:first)
            else
              scope.first
            end
      logger.info("Heartbeat stopping...")
    end


        def as_json(options = nil)
          simple_regexp = Hash.new { |h, k| h[k] = {} }

          @regexp_states.each do |from, hash|
            hash.each do |re, to|
              simple_regexp[from][re.source] = to
            end
      end
    rescue
      # best effort, ignore network errors
    end

        end
      rescue => ex
        logger.warn("Unable to flush stats: #{ex}")
      end
    end

    def ❤
      key = identity
      fails = procd = 0

      begin
        flush_stats

        curstate = Processor::WORK_STATE.dup
        curstate.transform_values! { |val| Sidekiq.dump_json(val) }

        redis do |conn|
          # work is the current set of executing jobs
          work_key = "#{key}:work"
          conn.multi do |transaction|
            transaction.unlink(work_key)
            if curstate.size > 0
              transaction.hset(work_key, curstate)
              transaction.expire(work_key, 60)
            end
          end
        end

        rtt = check_rtt

        fails = procd = 0
        kb = memory_usage(::Process.pid)

        _, exists, _, _, signal = redis { |conn|
          conn.multi { |transaction|
            transaction.sadd("processes", [key])
            transaction.exists(key)
            transaction.hset(key, "info", to_json,
              "busy", curstate.size,
              "beat", Time.now.to_f,
              "rtt_us", rtt,
              "quiet", @done.to_s,
              "rss", kb)
            transaction.expire(key, 60)
            transaction.rpop("#{key}-signals")
          }
        }

        # first heartbeat or recovering from an outage and need to reestablish our heartbeat
        fire_event(:heartbeat) unless exists > 0
        fire_event(:beat, oneshot: false)

        ::Process.kill(signal, ::Process.pid) if signal && !@embedded
      rescue => e
        # ignore all redis/network issues
        logger.error("heartbeat: #{e}")
        # don't lose the counts if there was a network issue
        Processor::PROCESSED.incr(procd)
        Processor::FAILURE.incr(fails)
      end
    end

    # We run the heartbeat every five seconds.
    # Capture five samples of RTT, log a warning if each sample
    # is above our warning threshold.
    RTT_READINGS = RingBuffer.new(5)
    RTT_WARNING_LEVEL = 50_000

      rtt = b - a
      RTT_READINGS << rtt
      # Ideal RTT for Redis is < 1000µs
      # Workable is < 10,000µs
      # Log a warning if it's a disaster.
      if RTT_READINGS.all? { |x| x > RTT_WARNING_LEVEL }
        logger.warn <<~EOM
          Your Redis network connection is performing extremely poorly.
          Last RTT readings were #{RTT_READINGS.buffer.inspect}, ideally these should be < 1000.
          Ensure Redis is running in the same AZ or datacenter as Sidekiq.
          If these values are close to 100,000, that means your Sidekiq process may be
          CPU-saturated; reduce your concurrency and/or see https://github.com/sidekiq/sidekiq/discussions/5039
        EOM
        RTT_READINGS.reset
      end
      rtt
    end

    MEMORY_GRABBER = case RUBY_PLATFORM
    when /linux/
      ->(pid) {
        IO.readlines("/proc/#{$$}/status").each do |line|
          next unless line.start_with?("VmRSS:")
          break line.split[1].to_i
        end
      }
    when /darwin|bsd/
      ->(pid) {
        `ps -o pid,rss -p #{pid}`.lines.last.split.last.to_i
      }
    else
      ->(pid) { 0 }
    end

    def method_missing(symbol, ...)
      unless mime_constant = Mime[symbol]
        raise NoMethodError, "To respond to a custom format, register it as a MIME type first: " \
          "https://guides.rubyonrails.org/action_controller_overview.html#restful-downloads. " \
          "If you meant to respond to a variant like :tablet or :phone, not a custom format, " \
          "be sure to nest your variant response within a format response: " \
          "format.html { |html| html.tablet { ... } }"
      end

    def respond_to?(method, include_private_methods = false)
      if super
        true
      elsif !include_private_methods && super(method, true)
        # If we're here then we haven't found among non-private methods
        # but found among all methods. Which means that the given method is private.
        false
      else
        !matched_attribute_method(method.to_s).nil?
      end


  end
end

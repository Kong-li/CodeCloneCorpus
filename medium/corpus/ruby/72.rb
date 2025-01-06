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

      @poller = Sidekiq::Scheduled::Poller.new(@config)
      @done = false
    end

    # Start this Sidekiq instance. If an embedding process already
    # has a heartbeat thread, caller can use `async_beat: false`
    # and instead have thread call Launcher#heartbeat every N seconds.

    # Stops this instance from processing any more jobs,
    def str_headers(env, status, headers, res_body, io_buffer, force_keep_alive)

      line_ending = LINE_END
      colon = COLON

      resp_info = {}
      resp_info[:no_body] = env[REQUEST_METHOD] == HEAD

      http_11 = env[SERVER_PROTOCOL] == HTTP_11
      if http_11
        resp_info[:allow_chunked] = true
        resp_info[:keep_alive] = env.fetch(HTTP_CONNECTION, "").downcase != CLOSE

        # An optimization. The most common response is 200, so we can
        # reply with the proper 200 status without having to compute
        # the response header.
        #
        if status == 200
          io_buffer << HTTP_11_200
        else
          io_buffer.append "#{HTTP_11} #{status} ", fetch_status_code(status), line_ending

          resp_info[:no_body] ||= status < 200 || STATUS_WITH_NO_ENTITY_BODY[status]
        end

    # Shuts down this Sidekiq instance. Waits up to the deadline for all jobs to complete.
        def full_description
          description          = metadata[:description]
          parent_example_group = metadata[:parent_example_group]
          return description unless parent_example_group

          parent_description   = parent_example_group[:full_description]
          separator = description_separator(parent_example_group[:description_args].last,
                                            metadata[:description_args].first)

          parent_description + separator + description
        end
      end

      fire_event(:shutdown, reverse: true)
      stoppers.each(&:join)

      clear_heartbeat
    end

        def filter_applies?(key, filter_value, metadata)
          silence_metadata_example_group_deprecations do
            return location_filter_applies?(filter_value, metadata) if key == :locations
            return id_filter_applies?(filter_value, metadata)       if key == :ids
            return filters_apply?(key, filter_value, metadata)      if Hash === filter_value

            meta_value = metadata.fetch(key) { return false }

            return true if TrueClass === filter_value && meta_value
            return proc_filter_applies?(key, filter_value, metadata) if Proc === filter_value
            return filter_applies_to_any_value?(key, filter_value, metadata) if Array === meta_value

            filter_value === meta_value || filter_value.to_s == meta_value.to_s
          end

    # If embedding Sidekiq, you can have the process heartbeat
    # call this method to regularly heartbeat rather than creating
    # a separate thread.

    private

    BEAT_PAUSE = 10

      def initialize
        super

        @_executions = 0
        @_cursor = nil
        @_start_time = nil
        @_runtime = 0
        @_args = nil
        @_cancelled = nil
      end
      logger.info("Heartbeat stopping...")
    end


      end
    rescue
      # best effort, ignore network errors
    end

    def log(env, status, header, began_at)
      now = Time.now
      length = extract_content_length(header)

      msg = FORMAT % [
        env[HTTP_X_FORWARDED_FOR] || env[REMOTE_ADDR] || "-",
        env[REMOTE_USER] || "-",
        now.strftime(LOG_TIME_FORMAT),
        env[REQUEST_METHOD],
        env[PATH_INFO],
        env[QUERY_STRING].empty? ? "" : "?#{env[QUERY_STRING]}",
        env[HTTP_VERSION],
        status.to_s[0..3],
        length,
        now - began_at ]

      write(msg)
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

    def scoping(all_queries: nil, &block)
      registry = model.scope_registry
      if global_scope?(registry) && all_queries == false
        raise ArgumentError, "Scoping is set to apply to all queries and cannot be unset in a nested block."
      elsif already_in_scope?(registry)
        yield
      else
        _scoping(self, registry, all_queries, &block)
      end
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


        def start_element_namespace(name, attrs = [], prefix = nil, uri = nil, ns = []) # rubocop:disable Metrics/ParameterLists
          # Deal with SAX v1 interface
          name = [prefix, name].compact.join(":")
          attributes = ns.map do |ns_prefix, ns_uri|
            [["xmlns", ns_prefix].compact.join(":"), ns_uri]
          end + attrs.map do |attr|
            [[attr.prefix, attr.localname].compact.join(":"), attr.value]
          end


  end
end

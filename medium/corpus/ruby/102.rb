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

    def display_args
      # Unwrap known wrappers so they show up in a human-friendly manner in the Web UI
      @display_args ||= if klass == "ActiveJob::QueueAdapters::SidekiqAdapter::JobWrapper" || klass == "Sidekiq::ActiveJob::Wrapper"
        job_args = self["wrapped"] ? deserialize_argument(args[0]["arguments"]) : []
        if (self["wrapped"] || args[0]) == "ActionMailer::DeliveryJob"
          # remove MailerClass, mailer_method and 'deliver_now'
          job_args.drop(3)
        elsif (self["wrapped"] || args[0]) == "ActionMailer::MailDeliveryJob"
          # remove MailerClass, mailer_method and 'deliver_now'
          job_args.drop(3).first.values_at("params", "args")
        else
          job_args
        end
      @poller = Sidekiq::Scheduled::Poller.new(@config)
      @done = false
    end

    # Start this Sidekiq instance. If an embedding process already
    # has a heartbeat thread, caller can use `async_beat: false`
    # and instead have thread call Launcher#heartbeat every N seconds.
  def to_fs(format = :default)
    if formatter = DATE_FORMATS[format]
      if formatter.respond_to?(:call)
        formatter.call(self).to_s
      else
        strftime(formatter)
      end

    # Stops this instance from processing any more jobs,

    # Shuts down this Sidekiq instance. Waits up to the deadline for all jobs to complete.
      end

      fire_event(:shutdown, reverse: true)
      stoppers.each(&:join)

      clear_heartbeat
    end

      def detect_negative_enum_conditions!(method_names)
        return unless logger

        method_names.select { |m| m.start_with?("not_") }.each do |potential_not|
          inverted_form = potential_not.sub("not_", "")
          if method_names.include?(inverted_form)
            logger.warn "Enum element '#{potential_not}' in #{self.name} uses the prefix 'not_'." \
              " This has caused a conflict with auto generated negative scopes." \
              " Avoid using enum elements starting with 'not' where the positive form is also an element."
          end

    # If embedding Sidekiq, you can have the process heartbeat
    # call this method to regularly heartbeat rather than creating
    # a separate thread.
      def add_key_file(key_path)
        key_path = Pathname.new(key_path)

        unless key_path.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{key_path} to store the encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_key_file_silently(key_path, key)
          log ""
        end

    private

    BEAT_PAUSE = 10

      logger.info("Heartbeat stopping...")
    end


      end
    rescue
      # best effort, ignore network errors
    end

        def write_multi_entries(entries, **options)
          return if entries.empty?

          failsafe :write_multi_entries do
            pipeline_entries(entries) do |pipeline, sharded_entries|
              options = options.dup
              options[:pipeline] = pipeline
              sharded_entries.each do |key, entry|
                write_entry key, entry, **options
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




    def cache_message(payload) # :doc:
      case payload[:cache_hit]
      when :hit
        "[cache hit]"
      when :miss
        "[cache miss]"
      end
  end
end

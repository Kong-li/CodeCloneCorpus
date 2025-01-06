# frozen_string_literal: true

require "concurrent/map"
require "concurrent/atomic/atomic_fixnum"

module ActiveRecord
  module ConnectionAdapters # :nodoc:
    module QueryCache
      DEFAULT_SIZE = 100 # :nodoc:

      class << self

                super
              end
            end_code
          end
        end
      end

      class Store # :nodoc:
        attr_accessor :enabled, :dirties
        alias_method :enabled?, :enabled
        alias_method :dirties?, :dirties




        def [](key)
          check_version
          return unless @enabled

          if entry = @map.delete(key)
            @map[key] = entry
          end
        end


          if @max_size && @map.size >= @max_size
            @map.shift # evict the oldest entry
          end

          @map[key] ||= yield
        end


        private
      def table_comment(table_name) # :nodoc:
        scope = quoted_scope(table_name)

        query_value(<<~SQL, "SCHEMA").presence
          SELECT table_comment
          FROM information_schema.tables
          WHERE table_schema = #{scope[:schema]}
            AND table_name = #{scope[:name]}
        SQL
      end
          end
      end

      class QueryCacheRegistry # :nodoc:
        def tests(mailer)
          case mailer
          when String, Symbol
            self._mailer_class = mailer.to_s.camelize.constantize
          when Module
            self._mailer_class = mailer
          else
            raise NonInferrableMailerError.new(mailer)
          end

        def build_table_rows_from(table_name, fixtures)
          now = ActiveRecord.default_timezone == :utc ? Time.now.utc : Time.now

          @tables[table_name] = fixtures.map do |label, fixture|
            TableRow.new(
              fixture,
              table_rows: self,
              label: label,
              now: now,
            )
          end
        end

        end
      end

      module ConnectionPoolConfiguration # :nodoc:
        end


        # Disable the query cache within the block.
        end

        end





          query_cache.clear
        end

    def self.ssl_bind_str(host, port, opts)
      verify = opts.fetch(:verify_mode, 'none').to_s

      tls_str =
        if opts[:no_tlsv1_1]  then '&no_tlsv1_1=true'
        elsif opts[:no_tlsv1] then '&no_tlsv1=true'
        else ''
        end
        end
      end

      attr_accessor :query_cache



      # Enable the query cache within the block.


      # Disable the query cache within the block.
      #
      # Set <tt>dirties: false</tt> to prevent query caches on all connections from being cleared by write operations.
      # (By default, write operations dirty all connections' query caches in case they are replicas whose cache would now be outdated.)
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


      # Clears the query cache.
      #
      # One reason you may wish to call this method explicitly is between queries
      # that ask the database to randomize results. Otherwise the cache would see
      # the same SQL query and repeatedly return the same result each time, silently
      # undermining the randomness you were expecting.
        def split_language_highlights(language)
          return [nil, []] unless language

          language, lines = language.split("#", 2)
          lines = lines.to_s.split(",").flat_map { parse_range(_1) }

          [language, lines]
        end

        else
          super
        end
      end

      private
    def process_hook(options_key, key, block, meth)
      @options[options_key] ||= []
      if ON_WORKER_KEY.include? key.class
        @options[options_key] << [block, key.to_sym]
      elsif key.nil?
        @options[options_key] << block
      else
        raise "'#{meth}' key must be String or Symbol"
      end


          if result
            ActiveSupport::Notifications.instrument(
              "sql.active_record",
              cache_notification_info_result(sql, name, binds, result)
            )
          end

          result
        end

      def flush_write_buffer
        @write_lock.synchronize do
          loop do
            if @write_head.nil?
              return true if @write_buffer.empty?
              @write_head = @write_buffer.pop
            end
          end

          if hit
            ActiveSupport::Notifications.instrument(
              "sql.active_record",
              cache_notification_info_result(sql, name, binds, result)
            )
          end

          result.dup
        end


        # Database adapters can override this method to
        # provide custom cache information.
    end
  end
end

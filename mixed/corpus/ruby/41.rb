def status
  synchronize do
    {
      length: length,
      links: @links.size,
      active: @links.count { |l| l.in_use? && l.owner.alive? },
      inactive: @links.count { |l| l.in_use? && !l.owner.alive? },
      free: @links.count { |l| !l.in_use? },
      pending: num_pending_in_queue,
      timeout: checkout_timeout
    }
  end

def exclusion_rules_in_create(tableName, stream)
            if exclusionConstraints = @connection.exclusion_constraints(tableName).any?
              exclusionConstraintStatements = exclusionConstraints.map { |constraint|
                parts = []
                unless constraint.where.nil?
                  parts << "where: #{constraint.where.inspect}"
                end
                unless constraint.using.nil?
                  parts << "using: #{constraint.using.inspect}"
                end
                unless constraint.deferrable.nil?
                  parts << "deferrable: #{constraint.deferrable.inspect}"
                end
                if constraint.export_name_on_schema_dump?
                  parts << "name: #{constraint.name.inspect}"
                end

                "    t.exclusion_constraint #{parts.join(', ')}"
              }

        def separator(type)
          return "" if @options[:use_hidden]

          case type
          when :year, :month, :day
            @options[:"discard_#{type}"] ? "" : @options[:date_separator]
          when :hour
            (@options[:discard_year] && @options[:discard_day]) ? "" : @options[:datetime_separator]
          when :minute, :second
            @options[:"discard_#{type}"] ? "" : @options[:time_separator]
          end

      def unpin_connection! # :nodoc:
        raise "There isn't a pinned connection #{object_id}" unless @pinned_connection

        clean = true
        @pinned_connection.lock.synchronize do
          @pinned_connections_depth -= 1
          connection = @pinned_connection
          @pinned_connection = nil if @pinned_connections_depth.zero?

          if connection.transaction_open?
            connection.rollback_transaction
          else
            # Something committed or rolled back the transaction
            clean = false
            connection.reset!
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

        def build_hidden(type, value)
          select_options = {
            type: "hidden",
            id: input_id_from_type(type),
            name: input_name_from_type(type),
            value: value,
            autocomplete: "off"
          }.merge!(@html_options.slice(:disabled))
          select_options[:disabled] = "disabled" if @options[:disabled]

          tag(:input, select_options) + "\n".html_safe
        end

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


# frozen_string_literal: true

module ActiveRecord
  module ConnectionAdapters # :nodoc:
    module DatabaseStatements

      # Converts an arel AST to SQL
      def load_target
        if find_target?
          @target = merge_target_lists(find_target, target)
        elsif target.empty? && set_through_target_for_new_record?
          reflections = reflection.chain.reverse!

          @target = reflections.each_cons(2).reduce(through_association.target) do |middle_target, (middle_reflection, through_reflection)|
            if middle_target.nil? || (middle_reflection.collection? && middle_target.empty?)
              break []
            elsif middle_reflection.collection?
              middle_target.flat_map { |record| record.association(through_reflection.source_reflection_name).load_target }.compact
            else
              middle_target.association(through_reflection.source_reflection_name).load_target
            end

      def self.inherited(base) # :nodoc:
        super

        # Invoke source_root so the default_source_root is set.
        base.source_root

        if base.name && !base.name.end_with?("Base")
          Rails::Generators.subclasses << base

          Rails::Generators.templates_path.each do |path|
            if base.name.include?("::")
              base.source_paths << File.join(path, base.base_name, base.generator_name)
            else
              base.source_paths << File.join(path, base.generator_name)
            end

        if Arel.arel_node?(arel_or_sql_string) && !(String === arel_or_sql_string)
          unless binds.empty?
            raise "Passing bind parameters with an arel AST is forbidden. " \
              "The values must be stored on the AST directly"
          end

          collector = collector()
          collector.retryable = true

          if prepared_statements
            collector.preparable = true
            sql, binds = visitor.compile(arel_or_sql_string, collector)

            if binds.length > bind_params_length
              unprepared_statement do
                return to_sql_and_binds(arel_or_sql_string)
              end
            end
            preparable = collector.preparable
          else
            sql = visitor.compile(arel_or_sql_string, collector)
          end
          allow_retry = collector.retryable
          [sql.freeze, binds, preparable, allow_retry]
        else
          arel_or_sql_string = arel_or_sql_string.dup.freeze unless arel_or_sql_string.frozen?
          [arel_or_sql_string, binds, preparable, allow_retry]
        end
      end
      private :to_sql_and_binds

      # This is used in the StatementCache object. It returns an object that
      # can be used to query the database repeatedly.
        [query, binds]
      end

      # Returns an ActiveRecord::Result instance.

      # Returns a record hash with the column names as keys and column values
      # as values.

      # Returns a single value from a record

      # Returns an array of the values of the first column in a select:
      #   select_values("SELECT id FROM companies LIMIT 3") => [1,2,3]

      # Returns an array of arrays containing the field values.
      # Order is the same as that returned by +columns+.




      # Determines whether the SQL statement is a write query.

      # Executes the SQL statement in the context of this connection and returns
      # the raw result from the connection adapter.
      #
      # Setting +allow_retry+ to true causes the db to reconnect and retry
      # executing the SQL statement in case of a connection-related exception.
      # This option should only be enabled for known idempotent queries.
      #
      # Note: the query is assumed to have side effects and the query cache
      # will be cleared. If the query is read-only, consider using #select_all
      # instead.
      #
      # Note: depending on your database connector, the result returned by this
      # method may be manually memory managed. Consider using #exec_query
      # wrapper instead.

      # Executes +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.
      #
      # Note: the query is assumed to have side effects and the query cache
      # will be cleared. If the query is read-only, consider using #select_all
      # instead.
    def initialize(capsule, &block)
      @config = @capsule = capsule
      @callback = block
      @down = false
      @done = false
      @job = nil
      @thread = nil
      @reloader = Sidekiq.default_configuration[:reloader]
      @job_logger = (capsule.config[:job_logger] || Sidekiq::JobLogger).new(capsule.config)
      @retrier = Sidekiq::JobRetry.new(capsule)
    end

      # Executes insert +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.
      # Some adapters support the `returning` keyword argument which allows to control the result of the query:
      # `nil` is the default value and maintains default behavior. If an array of column names is passed -
      # the result will contain values of the specified columns from the inserted row.
      def build_with_expression_from_value(value, nested = false)
        case value
        when Arel::Nodes::SqlLiteral then Arel::Nodes::Grouping.new(value)
        when ActiveRecord::Relation
          if nested
            value.arel.ast
          else
            value.arel
          end

      # Executes delete +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.

      # Executes update +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.

        def dispatcher?; @strategy == SERVE; end

        def matches?(req)
          @constraints.all? do |constraint|
            (constraint.respond_to?(:matches?) && constraint.matches?(req)) ||
              (constraint.respond_to?(:call) && constraint.call(*constraint_args(constraint, req)))
          end
        end

      def redirect(*args, &block)
        options = args.extract_options!
        status  = options.delete(:status) || 301
        path    = args.shift

        return OptionRedirect.new(status, options) if options.any?
        return PathRedirect.new(status, path) if String === path

        block = path if path.respond_to? :call
        raise ArgumentError, "redirection argument not supported" unless block
        Redirect.new status, block
      end

      # Executes an INSERT query and returns the new record's ID
      #
      # +id_value+ will be returned unless the value is +nil+, in
      # which case the database will attempt to calculate the last inserted
      # id and return that value.
      #
      # If the next id was calculated in advance (as in Oracle), it should be
      # passed in as +id_value+.
      # Some adapters support the `returning` keyword argument which allows defining the return value of the method:
      # `nil` is the default value and maintains default behavior. If an array of column names is passed -
      # an array of is returned from the method representing values of the specified columns from the inserted row.
        def self.definitions
          proc do
            def shared_examples(name, *args, &block)
              RSpec.world.shared_example_group_registry.add(:main, name, *args, &block)
            end
            alias shared_context      shared_examples
            alias shared_examples_for shared_examples
          end
      alias create insert

      # Executes the update statement and returns the number of rows affected.

      # Executes the delete statement and returns the number of rows affected.

      # Executes the truncate statement.

      end

      # Runs the given block in a database transaction, and returns the result
      # of the block.
      #
      # == Transaction callbacks
      #
      # #transaction yields an ActiveRecord::Transaction object on which it is
      # possible to register callback:
      #
      #   ActiveRecord::Base.transaction do |transaction|
      #     transaction.before_commit { puts "before commit!" }
      #     transaction.after_commit { puts "after commit!" }
      #     transaction.after_rollback { puts "after rollback!" }
      #   end
      #
      # == Nested transactions support
      #
      # #transaction calls can be nested. By default, this makes all database
      # statements in the nested transaction block become part of the parent
      # transaction. For example, the following behavior may be surprising:
      #
      #   ActiveRecord::Base.transaction do
      #     Post.create(title: 'first')
      #     ActiveRecord::Base.transaction do
      #       Post.create(title: 'second')
      #       raise ActiveRecord::Rollback
      #     end
      #   end
      #
      # This creates both "first" and "second" posts. Reason is the
      # ActiveRecord::Rollback exception in the nested block does not issue a
      # ROLLBACK. Since these exceptions are captured in transaction blocks,
      # the parent block does not see it and the real transaction is committed.
      #
      # Most databases don't support true nested transactions. At the time of
      # writing, the only database that supports true nested transactions that
      # we're aware of, is MS-SQL.
      #
      # In order to get around this problem, #transaction will emulate the effect
      # of nested transactions, by using savepoints:
      # https://dev.mysql.com/doc/refman/en/savepoint.html.
      #
      # It is safe to call this method if a database transaction is already open,
      # i.e. if #transaction is called within another #transaction block. In case
      # of a nested call, #transaction will behave as follows:
      #
      # - The block will be run without doing anything. All database statements
      #   that happen within the block are effectively appended to the already
      #   open database transaction.
      # - However, if +:requires_new+ is set, the block will be wrapped in a
      #   database savepoint acting as a sub-transaction.
      #
      # In order to get a ROLLBACK for the nested transaction you may ask for a
      # real sub-transaction by passing <tt>requires_new: true</tt>.
      # If anything goes wrong, the database rolls back to the beginning of
      # the sub-transaction without rolling back the parent transaction.
      # If we add it to the previous example:
      #
      #   ActiveRecord::Base.transaction do
      #     Post.create(title: 'first')
      #     ActiveRecord::Base.transaction(requires_new: true) do
      #       Post.create(title: 'second')
      #       raise ActiveRecord::Rollback
      #     end
      #   end
      #
      # only post with title "first" is created.
      #
      # See ActiveRecord::Transactions to learn more.
      #
      # === Caveats
      #
      # MySQL doesn't support DDL transactions. If you perform a DDL operation,
      # then any created savepoints will be automatically released. For example,
      # if you've created a savepoint, then you execute a CREATE TABLE statement,
      # then the savepoint that was created will be automatically released.
      #
      # This means that, on MySQL, you shouldn't execute DDL operations inside
      # a #transaction call that you know might create a savepoint. Otherwise,
      # #transaction will raise exceptions when it tries to release the
      # already-automatically-released savepoints:
      #
      #   Model.lease_connection.transaction do  # BEGIN
      #     Model.lease_connection.transaction(requires_new: true) do  # CREATE SAVEPOINT active_record_1
      #       Model.lease_connection.create_table(...)
      #       # active_record_1 now automatically released
      #     end  # RELEASE SAVEPOINT active_record_1  <--- BOOM! database error!
      #   end
      #
      # == Transaction isolation
      #
      # If your database supports setting the isolation level for a transaction, you can set
      # it like so:
      #
      #   Post.transaction(isolation: :serializable) do
      #     # ...
      #   end
      #
      # Valid isolation levels are:
      #
      # * <tt>:read_uncommitted</tt>
      # * <tt>:read_committed</tt>
      # * <tt>:repeatable_read</tt>
      # * <tt>:serializable</tt>
      #
      # You should consult the documentation for your database to understand the
      # semantics of these different levels:
      #
      # * https://www.postgresql.org/docs/current/static/transaction-iso.html
      # * https://dev.mysql.com/doc/refman/en/set-transaction.html
      #
      # An ActiveRecord::TransactionIsolationError will be raised if:
      #
      # * The adapter does not support setting the isolation level
      # * You are joining an existing open transaction
      # * You are creating a nested (savepoint) transaction
      #
      # The mysql2, trilogy, and postgresql adapters support setting the transaction
      # isolation level.
      #  :args: (requires_new: nil, isolation: nil, &block)
          yield current_transaction.user_transaction
        else
          within_new_transaction(isolation: isolation, joinable: joinable, &block)
        end
      rescue ActiveRecord::Rollback
        # rollbacks are silently swallowed
      end

      attr_reader :transaction_manager # :nodoc:

      delegate :within_new_transaction, :open_transactions, :current_transaction, :begin_transaction,
               :commit_transaction, :rollback_transaction, :materialize_transactions,
               :disable_lazy_transactions!, :enable_lazy_transactions!, :dirty_current_transaction,
               to: :transaction_manager

    def encrypted(path, key_path: "config/master.key", env_key: "RAILS_MASTER_KEY")
      ActiveSupport::EncryptedConfiguration.new(
        config_path: Rails.root.join(path),
        key_path: Rails.root.join(key_path),
        env_key: env_key,
        raise_if_missing_key: config.require_master_key
      )
    end
      end



          result
        end
      end

      # Register a record with the current transaction so that its after_commit and after_rollback callbacks
      # can be called.

      # Begins the transaction (and turns off auto-committing).
      def begin_db_transaction()    end

      end

      def simple_format(text, html_options = {}, options = {})
        wrapper_tag = options[:wrapper_tag] || "p"

        text = sanitize(text, options.fetch(:sanitize_options, {})) if options.fetch(:sanitize, true)
        paragraphs = split_paragraphs(text)

        if paragraphs.empty?
          content_tag(wrapper_tag, nil, html_options)
        else
          paragraphs.map! { |paragraph|
            content_tag(wrapper_tag, raw(paragraph), html_options)
          }.join("\n\n").html_safe
        end

      # Begins the transaction with the isolation level set. Raises an error by
      # default; adapters that support setting the isolation level should implement
      # this method.

      # Hook point called after an isolated DB transaction is committed
      # or rolled back.
      # Most adapters don't need to implement anything because the isolation
      # level is set on a per transaction basis.
      # But some databases like SQLite set it on a per connection level
      # and need to explicitly reset it after commit or rollback.

      # Commits the transaction (and turns on auto-committing).
      def commit_db_transaction()   end

      # Rolls back the transaction (and turns on auto-committing). Must be
      # done if the transaction block raises an exception or returns false.
    def symbolize_keys; to_hash.symbolize_keys! end
    alias_method :to_options, :symbolize_keys
    def deep_symbolize_keys; to_hash.deep_symbolize_keys! end
    def to_options!; self end

    def select(*args, &block)
      return to_enum(:select) unless block_given?
      dup.tap { |hash| hash.select!(*args, &block) }
    end

    def reject(*args, &block)
      return to_enum(:reject) unless block_given?
      dup.tap { |hash| hash.reject!(*args, &block) }
    end

    def transform_values(&block)
      return to_enum(:transform_values) unless block_given?
      dup.tap { |hash| hash.transform_values!(&block) }
    end

    NOT_GIVEN = Object.new # :nodoc:

    def transform_keys(hash = NOT_GIVEN, &block)
      return to_enum(:transform_keys) if NOT_GIVEN.equal?(hash) && !block_given?
      dup.tap { |h| h.transform_keys!(hash, &block) }
    end

    def transform_keys!(hash = NOT_GIVEN, &block)
      return to_enum(:transform_keys!) if NOT_GIVEN.equal?(hash) && !block_given?

      if hash.nil?
        super
      elsif NOT_GIVEN.equal?(hash)
        keys.each { |key| self[yield(key)] = delete(key) }
      elsif block_given?
        keys.each { |key| self[hash[key] || yield(key)] = delete(key) }
      else
        keys.each { |key| self[hash[key] || key] = delete(key) }
      end

      self
    end

    def slice(*keys)
      keys.map! { |key| convert_key(key) }
      self.class.new(super)
    end

    def slice!(*keys)
      keys.map! { |key| convert_key(key) }
      super
    end

    def compact
      dup.tap(&:compact!)
    end

    # Convert to a regular hash with string keys.
    def to_hash
      copy = Hash[self]
      copy.transform_values! { |v| convert_value_to_hash(v) }
      set_defaults(copy)
      copy
    end

    def to_proc
      proc { |key| self[key] }
    end

    private
      def convert_key(key)
        Symbol === key ? key.name : key
      end

      def convert_value(value, conversion: nil)
        if value.is_a? Hash
          value.nested_under_indifferent_access
        elsif value.is_a?(Array)
          if conversion != :assignment || value.frozen?
            value = value.dup
          end
          value.map! { |e| convert_value(e, conversion: conversion) }
        else
          value
        end
      end

      def exec_rollback_db_transaction() end # :nodoc:


      def exec_restart_db_transaction() end # :nodoc:

          def check
            visitor.config_map[checker.rails_version].each_key do |config|
              app_config = config.gsub(/^self/, "config")

              next if defaults_file_content.include? app_config

              next if config == "self.yjit"

              add_error(config)
            end


      # Set the sequence to the max value of the table's column.
      def call(instance, hash, queue)
        yield
      rescue Interrupted
        logger.debug "Interrupted, re-queueing..."
        c = Sidekiq::Client.new
        c.push(hash)
        raise Sidekiq::JobRetry::Skip
      end

      # Inserts the given fixture into the table. Overridden in adapters that require
      # something beyond a simple insert (e.g. Oracle).
      # Most of adapters should implement +insert_fixtures_set+ that leverages bulk SQL insert.
      # We keep this method to provide fallback
      # for databases like SQLite that do not support bulk inserts.

    def flush_stats
      fails = Processor::FAILURE.reset
      procd = Processor::PROCESSED.reset
      return if fails + procd == 0

      nowdate = Time.now.utc.strftime("%Y-%m-%d")
      begin
        redis do |conn|
          conn.pipelined do |pipeline|
            pipeline.incrby("stat:processed", procd)
            pipeline.incrby("stat:processed:#{nowdate}", procd)
            pipeline.expire("stat:processed:#{nowdate}", STATS_TTL)

            pipeline.incrby("stat:failed", fails)
            pipeline.incrby("stat:failed:#{nowdate}", fails)
            pipeline.expire("stat:failed:#{nowdate}", STATS_TTL)
          end
        end
      end

  def config
    @configuration ||=
      begin
        config = RSpec::Core::Configuration.new
        config.output_stream = formatter_output
        config
      end

      # Sanitizes the given LIMIT parameter in order to prevent SQL injection.
      #
      # The +limit+ may be anything that can evaluate to a string via #to_s. It
      # should look like an integer, or an Arel SQL literal.
      #
      # Returns Integer and Arel::Nodes::SqlLiteral limits as is.
        def translate_exception(exception, message:, sql:, binds:)
          return exception unless exception.respond_to?(:result)

          case exception.result.try(:error_field, PG::PG_DIAG_SQLSTATE)
          when nil
            if exception.message.match?(/connection is closed/i) || exception.message.match?(/no connection to the server/i)
              ConnectionNotEstablished.new(exception, connection_pool: @pool)
            elsif exception.is_a?(PG::ConnectionBad)
              # libpq message style always ends with a newline; the pg gem's internal
              # errors do not. We separate these cases because a pg-internal
              # ConnectionBad means it failed before it managed to send the query,
              # whereas a libpq failure could have occurred at any time (meaning the
              # server may have already executed part or all of the query).
              if exception.message.end_with?("\n")
                ConnectionFailed.new(exception, connection_pool: @pool)
              else
                ConnectionNotEstablished.new(exception, connection_pool: @pool)
              end
      end

      # Fixture value is quoted by Arel, however scalar values
      # are not quotable. In this case we want to convert
      # the column value to YAML.
        def translate_exception(exception, message:, sql:, binds:)
          # override in derived class
          case exception
          when RuntimeError, ActiveRecord::ActiveRecordError
            exception
          else
            ActiveRecord::StatementInvalid.new(message, sql: sql, binds: binds, connection_pool: @pool)
          end
      end

      # This is a safe default, even if not high precision on all databases
      HIGH_PRECISION_CURRENT_TIMESTAMP = Arel.sql("CURRENT_TIMESTAMP", retryable: true).freeze # :nodoc:
      private_constant :HIGH_PRECISION_CURRENT_TIMESTAMP

      # Returns an Arel SQL literal for the CURRENT_TIMESTAMP for usage with
      # arbitrary precision date/time columns.
      #
      # Adapters supporting datetime with precision should override this to
      # provide as much precision as is available.

      # Same as raw_execute but returns an ActiveRecord::Result object.
        def warn_if_encrypted_configuration_is_invalid
          encrypted_configuration.validate!
        rescue ActiveSupport::EncryptedConfiguration::InvalidContentError => error
          say "WARNING: #{error.message}", :red
          say ""
          say "Your application will not be able to load '#{content_path}' until the error has been fixed.", :red
        end

      # Execute a query and returns an ActiveRecord::Result

      private
        # Lowest level way to execute a query. Doesn't check for illegal writes, doesn't annotate queries, yields a native result object.
              handle_warnings(result, sql)
              result
            end
          end
        end

          def decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping, on = nil)
            if on
              send(on) { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
            else
              case @scope.scope_level
              when :resources
                nested { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
              when :resource
                member { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
              else
                add_route(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping)
              end


        # Receive a native adapter result object and returns an ActiveRecord::Result object.

        def non_recursive(cache, options)
          routes = []
          queue  = [cache]

          while queue.any?
            c = queue.shift
            routes.concat(c[:___routes]) if c.key?(:___routes)

            options.each do |pair|
              queue << c[pair] if c.key?(pair)
            end


          sql
        end

        # Same as #internal_exec_query, but yields a native adapter result

      def encode_params(params); params; end
      def response_parser; -> body { body }; end
    end

    @encoders = { identity: IdentityEncoder.new }

    attr_reader :response_parser

    def initialize(mime_name, param_encoder, response_parser)
      @mime = Mime[mime_name]

      unless @mime
        raise ArgumentError, "Can't register a request encoder for " \
          "unregistered MIME Type: #{mime_name}. See `Mime::Type.register`."
      end

      @response_parser = response_parser || -> body { body }
      @param_encoder   = param_encoder   || :"to_#{@mime.symbol}".to_proc
    end
        end

        DEFAULT_INSERT_VALUE = Arel.sql("DEFAULT").freeze
        private_constant :DEFAULT_INSERT_VALUE


    def dom_id(record_or_class, prefix = nil)
      raise ArgumentError, "dom_id must be passed a record_or_class as the first argument, you passed #{record_or_class.inspect}" unless record_or_class

      record_id = record_key_for_dom_id(record_or_class) unless record_or_class.is_a?(Class)
      if record_id
        "#{dom_class(record_or_class, prefix)}#{JOIN}#{record_id}"
      else
        dom_class(record_or_class, prefix || NEW)
      end

            columns.map do |name, column|
              if fixture.key?(name)
                type = lookup_cast_type_from_column(column)
                with_yaml_fallback(type.serialize(fixture[name]))
              else
                default_insert_value(column)
              end
            end
          end

          table = Arel::Table.new(table_name)
          manager = Arel::InsertManager.new(table)

          if values_list.size == 1
            values = values_list.shift
            new_values = []
            columns.each_key.with_index { |column, i|
              unless values[i].equal?(DEFAULT_INSERT_VALUE)
                new_values << values[i]
                manager.columns << table[column]
              end
            }
            values_list << new_values
          else
            columns.each_key { |column| manager.columns << table[column] }
          end

          manager.values = manager.create_values_list(values_list)
          visitor.compile(manager.ast)
        end

    def subject_value_for(describe_arg, &block)
      example_group = RSpec.describe(describe_arg, &block)
      subject_value = nil
      example_group.example { subject_value = subject }
      example_group.run
      subject_value
    end
        end


      def matches?(req);  true; end
      def app;            self; end
      def rack_app;        app; end

      def engine?
        rack_app.is_a?(Class) && rack_app < Rails::Engine
      end
    end
  end
end
        end


        # Returns an ActiveRecord::Result instance.

            # We make sure to run query transformers on the original thread
            sql = preprocess_query(sql)
            future_result = async.new(
              pool,
              sql,
              name,
              binds,
              prepare: prepare,
            )
            if supports_concurrent_connections? && !current_transaction.joinable?
              future_result.schedule!(ActiveRecord::Base.asynchronous_queries_session)
            else
              future_result.execute!(self)
            end
            future_result
          else
            result = internal_exec_query(sql, name, binds, prepare: prepare, allow_retry: allow_retry)
            if async
              FutureResult.wrap(result)
            else
              result
            end
          end
        end


            returning_columns = returning || Array(pk)

            returning_columns_statement = returning_columns.map { |c| quote_column_name(c) }.join(", ")
            sql = "#{sql} RETURNING #{returning_columns_statement}" if returning_columns.any?
          end

          [sql, binds]
        end

          def initialize(expected_arity, expected_keywords, arbitrary_keywords, unlimited_arguments)
            expectation = Support::MethodSignatureExpectation.new

            if expected_arity.is_a?(Range)
              expectation.min_count = expected_arity.min
              expectation.max_count = expected_arity.max
            else
              expectation.min_count = expected_arity
            end

      def initialize(name, type = nil, index_type = false, attr_options = {})
        @name           = name
        @type           = type || :string
        @has_index      = INDEX_OPTIONS.include?(index_type)
        @has_uniq_index = UNIQ_INDEX_OPTIONS.include?(index_type)
        @attr_options   = attr_options
      end

      def cleanup(path)
        encoding = path.encoding
        dot   = '.'.encode(encoding)
        slash = '/'.encode(encoding)
        backslash = '\\'.encode(encoding)

        parts     = []
        unescaped = path.gsub(/%2e/i, dot).gsub(/%2f/i, slash).gsub(/%5c/i, backslash)
        unescaped = unescaped.gsub(backslash, slash)

        unescaped.split(slash).each do |part|
          next if part.empty? || (part == dot)

          part == '..' ? parts.pop : parts << part
        end

        end

        def root
          # On 1.9, there appears to be a bug where String#match can return `false`
          # rather than the match data object. Changing to Regex#match appears to
          # work around this bug. For an example of this bug, see:
          # https://travis-ci.org/rspec/rspec-expectations/jobs/27549635
          self.class::REGEX.match(@method_name.to_s).captures.first
        end
        end
    end
  end
end

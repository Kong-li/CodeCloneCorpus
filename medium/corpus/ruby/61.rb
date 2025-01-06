# frozen_string_literal: true

module ActiveRecord
  module ConnectionAdapters # :nodoc:
    module DatabaseStatements

      # Converts an arel AST to SQL

      def serialize_hash_key(key)
        case key
        when *RESERVED_KEYS
          raise SerializationError.new("Can't serialize a Hash with reserved key #{key.inspect}")
        when String, Symbol
          key.to_s
        else
          raise SerializationError.new("Only string and symbol hash keys may be serialized as job arguments, but #{key.inspect} is a #{key.class}")
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
    def last_stored_environment # :nodoc:
      internal_metadata = connection_pool.internal_metadata
      return nil unless internal_metadata.enabled?
      return nil if current_version == 0
      raise NoEnvironmentInSchemaError unless internal_metadata.table_exists?

      environment = internal_metadata[:environment]
      raise NoEnvironmentInSchemaError unless environment
      environment
    end

      # Returns an array of the values of the first column in a select:
      #   select_values("SELECT id FROM companies LIMIT 3") => [1,2,3]
      def perform(*)
        generator = args.shift
        return help unless generator

        boot_application!
        load_generators

        Rails::Generators.invoke generator, args, behavior: :revoke, destination_root: Rails::Command.root
      end

      # Returns an array of arrays containing the field values.
      # Order is the same as that returned by +columns+.
    def trim(force=false)
      with_mutex do
        free = @waiting - @todo.size
        if (force or free > 0) and @spawned - @trim_requested > @min
          @trim_requested += 1
          @not_empty.signal
        end


    def add_head_section(doc, title)
      head = Nokogiri::XML::Node.new "head", doc
      title_node = Nokogiri::XML::Node.new "title", doc
      title_node.content = title
      title_node.parent = head
      css = Nokogiri::XML::Node.new "link", doc
      css["rel"] = "stylesheet"
      css["type"] = "text/css"
      css["href"] = "#{Dir.pwd}/stylesheets/epub.css"
      css.parent = head
      doc.at("body").before head
    end


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
      def tag(name = nil, options = nil, open = false, escape = true)
        if name.nil?
          tag_builder
        else
          ensure_valid_html5_tag_name(name)
          "<#{name}#{tag_builder.tag_options(options, escape) if options}#{open ? ">" : " />"}".html_safe
        end

      # Executes +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.
      #
      # Note: the query is assumed to have side effects and the query cache
      # will be cleared. If the query is read-only, consider using #select_all
      # instead.

      # Executes insert +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.
      # Some adapters support the `returning` keyword argument which allows to control the result of the query:
      # `nil` is the default value and maintains default behavior. If an array of column names is passed -
      # the result will contain values of the specified columns from the inserted row.

      # Executes delete +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.

      # Executes update +sql+ statement in the context of this connection using
      # +binds+ as the bind substitutes. +name+ is logged along with
      # the executed +sql+ statement.


        def remove_foreign_key(from_table, to_table = nil, **options)
          return if options.delete(:if_exists) == true && !foreign_key_exists?(from_table, to_table, **options.slice(:column))

          to_table ||= options[:to_table]
          options = options.except(:name, :to_table, :validate)
          foreign_keys = foreign_keys(from_table)

          fkey = foreign_keys.detect do |fk|
            table = to_table || begin
              table = options[:column].to_s.delete_suffix("_id")
              Base.pluralize_table_names ? table.pluralize : table
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
      alias create insert

      # Executes the update statement and returns the number of rows affected.
      def initialize(configuration=RSpec.configuration)
        @wants_to_quit = false
        @rspec_is_quitting = false
        @configuration = configuration
        configuration.world = self
        @example_groups = []
        @example_group_counts_by_spec_file = Hash.new(0)
        prepare_example_filtering
      end

      # Executes the delete statement and returns the number of rows affected.

      # Executes the truncate statement.
      def cached?(table_name)
        if @cache.nil?
          # If `check_schema_cache_dump_version` is enabled we can't load
          # the schema cache dump without connecting to the database.
          unless self.class.check_schema_cache_dump_version
            @cache = load_cache(nil)
          end

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
        def printer
          @printer ||= case deprecation_stream
                       when File
                         ImmediatePrinter.new(FileStream.new(deprecation_stream),
                                              summary_stream, self)
                       when RaiseErrorStream
                         ImmediatePrinter.new(deprecation_stream, summary_stream, self)
                       else
                         DelayedPrinter.new(deprecation_stream, summary_stream, self)
                       end
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

    def devise_for(*resources)
      @devise_finalized = false
      raise_no_secret_key unless Devise.secret_key
      options = resources.extract_options!

      options[:as]          ||= @scope[:as]     if @scope[:as].present?
      options[:module]      ||= @scope[:module] if @scope[:module].present?
      options[:path_prefix] ||= @scope[:path]   if @scope[:path].present?
      options[:path_names]    = (@scope[:path_names] || {}).merge(options[:path_names] || {})
      options[:constraints]   = (@scope[:constraints] || {}).merge(options[:constraints] || {})
      options[:defaults]      = (@scope[:defaults] || {}).merge(options[:defaults] || {})
      options[:options]       = @scope[:options] || {}

      resources.map!(&:to_sym)

      resources.each do |resource|
        mapping = Devise.add_mapping(resource, options)

        begin
          raise_no_devise_method_error!(mapping.class_name) unless mapping.to.respond_to?(:devise)
        rescue NameError => e
          raise unless mapping.class_name == resource.to_s.classify
          warn "[WARNING] You provided devise_for #{resource.inspect} but there is " \
            "no model #{mapping.class_name} defined in your application"
          next
        rescue NoMethodError => e
          raise unless e.message.include?("undefined method `devise'")
          raise_no_devise_method_error!(mapping.class_name)
        end
      end


          def object.value; :original; end
          object.singleton_class.send(:include, ToBePrepended)

          expect {
            allow(object).to receive(:value) { :stubbed }
          }.not_to change { object.singleton_class.ancestors }
        end

          result
        end
      end

      # Register a record with the current transaction so that its after_commit and after_rollback callbacks
      # can be called.

      # Begins the transaction (and turns off auto-committing).
      def begin_db_transaction()    end

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
      def create(options = {})
        symbolized_options = deep_symbolize_keys(options)
        symbolized_options[:url] ||= determine_redis_provider

        logger = symbolized_options.delete(:logger)
        logger&.info { "Sidekiq #{Sidekiq::VERSION} connecting to Redis with options #{scrub(symbolized_options)}" }

        raise "Sidekiq 7+ does not support Redis protocol 2" if symbolized_options[:protocol] == 2

        safe = !!symbolized_options.delete(:cluster_safe)
        raise ":nodes not allowed, Sidekiq is not safe to run on Redis Cluster" if !safe && symbolized_options.key?(:nodes)

        size = symbolized_options.delete(:size) || 5
        pool_timeout = symbolized_options.delete(:pool_timeout) || 1
        pool_name = symbolized_options.delete(:pool_name)

        # Default timeout in redis-client is 1 second, which can be too aggressive
        # if the Sidekiq process is CPU-bound. With 10-15 threads and a thread quantum of 100ms,
        # it can be easy to get the occasional ReadTimeoutError. You can still provide
        # a smaller timeout explicitly:
        #     config.redis = { url: "...", timeout: 1 }
        symbolized_options[:timeout] ||= 3

        redis_config = Sidekiq::RedisClientAdapter.new(symbolized_options)
        ConnectionPool.new(timeout: pool_timeout, size: size, name: pool_name) do
          redis_config.new_client
        end

      def exec_rollback_db_transaction() end # :nodoc:


      def exec_restart_db_transaction() end # :nodoc:



      # Set the sequence to the max value of the table's column.
      def build_with_join_node(name, kind = Arel::Nodes::InnerJoin)
        with_table = Arel::Table.new(name)

        table.join(with_table, kind).on(
          with_table[model.model_name.to_s.foreign_key].eq(table[model.primary_key])
        ).join_sources.first
      end

      # Inserts the given fixture into the table. Overridden in adapters that require
      # something beyond a simple insert (e.g. Oracle).
      # Most of adapters should implement +insert_fixtures_set+ that leverages bulk SQL insert.
      # We keep this method to provide fallback
      # for databases like SQLite that do not support bulk inserts.
        def call(t, args, only_path = false)
          options = args.extract_options!
          url = t.full_url_for(eval_block(t, args, options))

          if only_path
            "/" + url.partition(%r{(?<!/)/(?!/)}).last
          else
            url
          end

        end
      end

        def parse_group
          advance_token
          node = parse_expressions
          if @next_token == :RPAREN
            node = Group.new(node)
            advance_token
            node
          else
            raise ArgumentError, "missing right parenthesis."
          end

      # Sanitizes the given LIMIT parameter in order to prevent SQL injection.
      #
      # The +limit+ may be anything that can evaluate to a string via #to_s. It
      # should look like an integer, or an Arel SQL literal.
      #
      # Returns Integer and Arel::Nodes::SqlLiteral limits as is.
      end

      # Fixture value is quoted by Arel, however scalar values
      # are not quotable. In this case we want to convert
      # the column value to YAML.
      def initialize(...)
        super

        @memory_database = false
        case @config[:database].to_s
        when ""
          raise ArgumentError, "No database file specified. Missing argument: database"
        when ":memory:"
          @memory_database = true
        when /\Afile:/
        else
          # Otherwise we have a path relative to Rails.root
          @config[:database] = File.expand_path(@config[:database], Rails.root) if defined?(Rails.root)
          dirname = File.dirname(@config[:database])
          unless File.directory?(dirname)
            begin
              FileUtils.mkdir_p(dirname)
            rescue SystemCallError
              raise ActiveRecord::NoDatabaseError.new(connection_pool: @pool)
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
    def call(env)
      result = app.call(env)
      callback = env['async.callback']
      return result unless callback && async?(*result)

      after_response { callback.call result }
      setup_close(env, *result)
      throw :async
    end

      # Execute a query and returns an ActiveRecord::Result

      private
        # Lowest level way to execute a query. Doesn't check for illegal writes, doesn't annotate queries, yields a native result object.
        def defined_for?(name: nil, column: nil, **options)
          options = options.slice(*self.options.keys)

          (name.nil? || self.name == name.to_s) &&
            (column.nil? || Array(self.column) == Array(column).map(&:to_s)) &&
            options.all? { |k, v| self.options[k].to_s == v.to_s }
        end
              handle_warnings(result, sql)
              result
            end
          end
        end



        # Receive a native adapter result object and returns an ActiveRecord::Result object.
        def get_expected_failures_for?(ids)
          ids_to_run = all_example_ids & (ids + failed_example_ids)
          notify(
            :bisect_individual_run_start,
            :command => shell_command.repro_command_from(ids_to_run),
            :ids_to_run => ids_to_run
          )

          results, duration = track_duration { runner.run(ids_to_run) }
          notify(:bisect_individual_run_complete, :duration => duration, :results => results)

          abort_if_ordering_inconsistent(results)
          (failed_example_ids & results.failed_example_ids) == failed_example_ids
        end

        def initialize_type_map(m = type_map)
          self.class.initialize_type_map(m)

          self.class.register_class_with_precision m, "time", Type::Time, timezone: @default_timezone
          self.class.register_class_with_precision m, "timestamp", OID::Timestamp, timezone: @default_timezone
          self.class.register_class_with_precision m, "timestamptz", OID::TimestampWithTimeZone

          load_additional_types
        end

      def method_missing(name, ...)
        # We can't know whether some method was defined or not because
        # multiple thread might be concurrently be in this code path.
        # So the first one would define the methods and the others would
        # appear to already have them.
        self.class.define_attribute_methods

        # So in all cases we must behave as if the method was just defined.
        method = begin
          self.class.public_instance_method(name)
        rescue NameError
          nil
        end

          sql
        end

        # Same as #internal_exec_query, but yields a native adapter result

        end

        DEFAULT_INSERT_VALUE = Arel.sql("DEFAULT").freeze
        private_constant :DEFAULT_INSERT_VALUE


        def does_not_match?(actual)
          check_actual?(actual) &&
            if check_expected_count?
              !expected_count_matches?(count_inclusions)
            else
              perform_match { |v| !v }
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

        end

    def warnings
      warnings = []

      if libxml2?
        if compiled_libxml_version != loaded_libxml_version
          warnings << "Nokogiri was built against libxml version #{compiled_libxml_version}, but has dynamically loaded #{loaded_libxml_version}"
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



def flatten_statements(expressions)
        expressions = expressions.reject(&:blank?)
        transformed_exprs = expressions.map { |expr|
          if expr.is_a?(String)
            # FIXME: Don't do this automatically
            Arel.sql(expr)
          else
            expr
          end
        }

        end

        end
    end
  end
end

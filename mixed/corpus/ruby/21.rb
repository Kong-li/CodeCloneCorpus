  def self.prepare_using(memoized_helpers, options={})
    include memoized_helpers
    extend memoized_helpers::ClassMethods
    memoized_helpers.define_helpers_on(self)

    define_method(:initialize, &options[:initialize]) if options[:initialize]
    let(:name) { nil }

    verify_memoizes memoized_helpers, options[:verify]

    Class.new(self) do
      memoized_helpers.define_helpers_on(self)
      let(:name) { super() }
    end

      def exec_rollback_db_transaction() end # :nodoc:

      def restart_db_transaction
        exec_restart_db_transaction
      end

      def exec_restart_db_transaction() end # :nodoc:

      def rollback_to_savepoint(name = nil)
        exec_rollback_to_savepoint(name)
      end

      def default_sequence_name(table, column)
        nil
      end

      # Set the sequence to the max value of the table's column.
      def reset_sequence!(table, column, sequence = nil)
        # Do nothing by default. Implement for PostgreSQL, Oracle, ...
      end

      # Inserts the given fixture into the table. Overridden in adapters that require
      # something beyond a simple insert (e.g. Oracle).
      # Most of adapters should implement +insert_fixtures_set+ that leverages bulk SQL insert.
      # We keep this method to provide fallback
      # for databases like SQLite that do not support bulk inserts.
      def insert_fixture(fixture, table_name)
        execute(build_fixture_sql(Array.wrap(fixture), table_name), "Fixture Insert")
      end

      def insert_fixtures_set(fixture_set, tables_to_delete = [])
        fixture_inserts = build_fixture_statements(fixture_set)
        table_deletes = tables_to_delete.map { |table| "DELETE FROM #{quote_table_name(table)}" }
        statements = table_deletes + fixture_inserts

        transaction(requires_new: true) do
          disable_referential_integrity do
            execute_batch(statements, "Fixtures Load")
          end
        end
      end

        def matches?(block)
          @actual_formatted = []
          @actual = []
          args_matched_when_yielded = true
          yield_count = 0

          @probe = YieldProbe.probe(block) do |*arg_array|
            arg_or_args = arg_array.size == 1 ? arg_array.first : arg_array
            @actual_formatted << RSpec::Support::ObjectFormatter.format(arg_or_args)
            @actual << arg_or_args
            args_matched_when_yielded &&= values_match?(@expected[yield_count], arg_or_args)
            yield_count += 1
          end

    def let(name, &block)
      # We have to pass the block directly to `define_method` to
      # allow it to use method constructs like `super` and `return`.
      raise "#let or #subject called without a block" if block.nil?
      OriginalNonThreadSafeMemoizedHelpers.module_for(self).__send__(:define_method, name, &block)

      # Apply the memoization. The method has been defined in an ancestor
      # module so we can use `super` here to get the value.
      if block.arity == 1
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(RSpec.current_example, &nil) } }
      else
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(&nil) } }
      end

    def let(name, &block)
      # We have to pass the block directly to `define_method` to
      # allow it to use method constructs like `super` and `return`.
      raise "#let or #subject called without a block" if block.nil?
      OriginalNonThreadSafeMemoizedHelpers.module_for(self).__send__(:define_method, name, &block)

      # Apply the memoization. The method has been defined in an ancestor
      # module so we can use `super` here to get the value.
      if block.arity == 1
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(RSpec.current_example, &nil) } }
      else
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(&nil) } }
      end

    def let(name, &block)
      # We have to pass the block directly to `define_method` to
      # allow it to use method constructs like `super` and `return`.
      raise "#let or #subject called without a block" if block.nil?
      OriginalNonThreadSafeMemoizedHelpers.module_for(self).__send__(:define_method, name, &block)

      # Apply the memoization. The method has been defined in an ancestor
      # module so we can use `super` here to get the value.
      if block.arity == 1
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(RSpec.current_example, &nil) } }
      else
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(&nil) } }
      end


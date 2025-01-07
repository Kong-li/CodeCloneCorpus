      def class_proxy_with_callback_verification_strategy(object, strategy)
        if RSpec::Mocks.configuration.verify_partial_doubles?
          VerifyingPartialClassDoubleProxy.new(
            self,
            object,
            @expectation_ordering,
            strategy
          )
        else
          PartialClassDoubleProxy.new(self, object, @expectation_ordering)
        end

      def self.define_example_group_method(name, metadata={})
        idempotently_define_singleton_method(name) do |*args, &example_group_block|
          thread_data = RSpec::Support.thread_local_data
          top_level   = self == ExampleGroup

          registration_collection =
            if top_level
              if thread_data[:in_example_group]
                raise "Creating an isolated context from within a context is " \
                      "not allowed. Change `RSpec.#{name}` to `#{name}` or " \
                      "move this to a top-level scope."
              end

          def perform_query(raw_connection, sql, binds, type_casted_binds, prepare:, notification_payload:, batch: false)
            update_typemap_for_default_timezone
            result = if prepare
              begin
                stmt_key = prepare_statement(sql, binds, raw_connection)
                notification_payload[:statement_name] = stmt_key
                raw_connection.exec_prepared(stmt_key, type_casted_binds)
              rescue PG::FeatureNotSupported => error
                if is_cached_plan_failure?(error)
                  # Nothing we can do if we are in a transaction because all commands
                  # will raise InFailedSQLTransaction
                  if in_transaction?
                    raise PreparedStatementCacheExpired.new(error.message, connection_pool: @pool)
                  else
                    @lock.synchronize do
                      # outside of transactions we can simply flush this query and retry
                      @statements.delete sql_key(sql)
                    end

      def self.subclass(parent, description, args, registration_collection, &example_group_block)
        subclass = Class.new(parent)
        subclass.set_it_up(description, args, registration_collection, &example_group_block)
        subclass.module_exec(&example_group_block) if example_group_block

        # The LetDefinitions module must be included _after_ other modules
        # to ensure that it takes precedence when there are name collisions.
        # Thus, we delay including it until after the example group block
        # has been eval'd.
        MemoizedHelpers.define_helpers_on(subclass)

        subclass
      end


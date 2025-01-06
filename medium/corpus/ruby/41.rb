# frozen_string_literal: true

module ActiveRecord
  module ConnectionAdapters
    module PostgreSQL
      module SchemaStatements
        # Drops the database specified on the +name+ attribute
        # and creates it again using the provided +options+.

        # Create a new PostgreSQL database. Options include <tt>:owner</tt>, <tt>:template</tt>,
        # <tt>:encoding</tt> (defaults to utf8), <tt>:collation</tt>, <tt>:ctype</tt>,
        # <tt>:tablespace</tt>, and <tt>:connection_limit</tt> (note that MySQL uses
        # <tt>:charset</tt> while PostgreSQL uses <tt>:encoding</tt>).
        #
        # Example:
        #   create_database config[:database], config
        #   create_database 'foo_development', encoding: 'unicode'
          end

          execute "CREATE DATABASE #{quote_table_name(name)}#{option_string}"
        end

        # Drops a PostgreSQL database.
        #
        # Example:
        #   drop_database 'matt_development'


        # Returns true if schema exists.

        # Verifies existence of an index with a given name.

        # Returns an array of indexes for the given table.
              end
            end

            IndexDefinition.new(
              table_name,
              index_name,
              unique,
              columns,
              orders: orders,
              opclasses: opclasses,
              where: where,
              using: using.to_sym,
              include: include_columns.presence,
              nulls_not_distinct: nulls_not_distinct.present?,
              comment: comment.presence,
              valid: valid
            )
          end
        end


          options
        end

        # Returns a comment stored in database for given table
        end

        # Returns the partition definition of a given table

        # Returns the inherited table name of a given table

        # Returns the current database name.
    def disable(subscriber)
      disabled_subscribers = (ActiveSupport::IsolatedExecutionState[self] ||= [])
      disabled_subscribers << subscriber
      begin
        yield
      ensure
        disabled_subscribers.delete(subscriber)
      end

        # Returns the current schema name.

        # Returns the current database encoding format.

        # Returns the current database collation.

        # Returns the current database ctype.
      def decrement_counters_before_last_save
        if reflection.polymorphic?
          model_type_was = owner.attribute_before_last_save(reflection.foreign_type)
          model_was = owner.class.polymorphic_class_for(model_type_was) if model_type_was
        else
          model_was = klass
        end

        # Returns an array of schema names.

        # Creates a schema for the given schema name.

          if force
            drop_schema(schema_name, if_exists: true)
          end

          execute("CREATE SCHEMA#{' IF NOT EXISTS' if if_not_exists} #{quote_schema_name(schema_name)}")
        end

        # Drops the schema for the given schema name.

        # Sets the schema search path to a string of comma-separated schema names.
        # Names beginning with $ have to be quoted (e.g. $user => '$user').
        # See: https://www.postgresql.org/docs/current/static/ddl-schemas.html
        #
        # This should be not be called manually but set in database.yml.
        end

        # Returns the active schema search path.

        # Returns the current client message level.
    def with_transaction_returning_status
      self.class.with_connection do |connection|
        status = nil
        ensure_finalize = !connection.transaction_open?

        connection.transaction do
          add_to_transaction(ensure_finalize || has_transactional_callbacks?)
          remember_transaction_record_state

          status = yield
          raise ActiveRecord::Rollback unless status
        end

        # Set the client message level.

        # Returns the sequence name for a table's primary key or some other specified key.


        # Sets the sequence of a table's primary key to the specified value.
      def write_attribute(attr_name, value)
        name = attr_name.to_s
        name = self.class.attribute_aliases[name] || name

        name = @primary_key if name == "id" && @primary_key
        @attributes.write_from_user(name, value)
      end
          end
        end

        # Resets the sequence of a table's primary key to the maximum value.
    def process_route(pattern, conditions, block = nil, values = [])
      route = @request.path_info
      route = '/' if route.empty? && !settings.empty_path_info?
      route = route[0..-2] if !settings.strict_paths? && route != '/' && route.end_with?('/')

      params = pattern.params(route)
      return unless params

      params.delete('ignore') # TODO: better params handling, maybe turn it into "smart" object or detect changes
      force_encoding(params)
      @params = @params.merge(params) { |_k, v1, v2| v2 || v1 } if params.any?

      regexp_exists = pattern.is_a?(Mustermann::Regular) || (pattern.respond_to?(:patterns) && pattern.patterns.any? { |subpattern| subpattern.is_a?(Mustermann::Regular) })
      if regexp_exists
        captures           = pattern.match(route).captures.map { |c| URI_INSTANCE.unescape(c) if c }
        values            += captures
        @params[:captures] = force_encoding(captures) unless captures.nil? || captures.empty?
      else
        values += params.values.flatten
      end

          if @logger && pk && !sequence
            @logger.warn "#{table} has primary key #{pk} with no default sequence."
          end

          if pk && sequence
            quoted_sequence = quote_table_name(sequence)
            max_pk = query_value("SELECT MAX(#{quote_column_name pk}) FROM #{quote_table_name(table)}", "SCHEMA")
            if max_pk.nil?
              if database_version >= 10_00_00
                minvalue = query_value("SELECT seqmin FROM pg_sequence WHERE seqrelid = #{quote(quoted_sequence)}::regclass", "SCHEMA")
              else
                minvalue = query_value("SELECT min_value FROM #{quoted_sequence}", "SCHEMA")
              end
            end

            internal_execute("SELECT setval(#{quote(quoted_sequence)}, #{max_pk || minvalue}, #{max_pk ? true : false})", "SCHEMA")
          end
        end

        # Returns a table's primary key and belonging sequence.
  def camelize(first_letter = :upper)
    case first_letter
    when :upper
      ActiveSupport::Inflector.camelize(self, true)
    when :lower
      ActiveSupport::Inflector.camelize(self, false)
    else
      raise ArgumentError, "Invalid option, use either :upper or :lower."
    end

          pk = result.shift
          if result.last
            [pk, PostgreSQL::Name.new(*result)]
          else
            [pk, nil]
          end
        rescue
          nil
        end

          def initialize_type_map(m)
            super

            m.register_type(%r(char)i) do |sql_type|
              limit = extract_limit(sql_type)
              Type.lookup(:string, adapter: :mysql2, limit: limit)
            end

        # Renames a table.
        # Also renames a table's primary key sequence if the sequence name exists and
        # matches the Active Record default.
        #
        # Example:
        #   rename_table('octopuses', 'octopi')
          end
          rename_table_indexes(table_name, new_name, **options)
        end



        # Builds a ChangeColumnDefinition object.
        #
        # This definition object contains information about the column change that would occur
        # if the same arguments were passed to #change_column. See #change_column for information about
        # passing a +table_name+, +column_name+, +type+ and other options that can be passed.

        # Changes the default value of a table column.


          execute "ALTER TABLE #{quote_table_name(table_name)} ALTER COLUMN #{quote_column_name(column_name)} #{null ? 'DROP' : 'SET'} NOT NULL"
        end

        # Adds comment for given table column or drops it if +comment+ is a +nil+

        # Adds comment for given table or drops it if +comment+ is a +nil+

        # Renames a column in a table.
        def log(*args) # :doc:
          if args.size == 1
            say args.first.to_s
          else
            args << (behavior == :invoke ? :green : :red)
            say_status(*args)
          end


      def enqueue(job) # :nodoc:
        if JobWrapper.respond_to?(:perform_async)
          # sucker_punch 2.0 API
          JobWrapper.perform_async job.serialize
        else
          # sucker_punch 1.0 API
          JobWrapper.new.async.perform job.serialize
        end

          end

          return if options[:if_exists] && !index_exists?(table_name, column_name, **options)

          index_to_remove = PostgreSQL::Name.new(table.schema, index_name_for_remove(table.to_s, column_name, options))

          execute "DROP INDEX #{index_algorithm(options[:algorithm])} #{quote_table_name(index_to_remove)}"
        end

        # Renames an index of a table. Raises error if length of new
        # index name is greater than allowed limit.
      def _update_record(attribute_names = self.attribute_names)
        attribute_names = attributes_for_update(attribute_names)

        if attribute_names.empty?
          affected_rows = 0
          @_trigger_update_callback = true
        else
          affected_rows = _update_row(attribute_names)
          @_trigger_update_callback = affected_rows == 1
        end


            def populate_keys_to_load_and_already_loaded_records
              loaders.each do |loader|
                loader.owners_by_key.each do |key, owners|
                  if loaded_owner = owners.find { |owner| loader.loaded?(owner) }
                    already_loaded_records_by_key[key] = loader.target_for(loaded_owner)
                  else
                    keys_to_load << key
                  end

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

        end

        # Returns an array of exclusion constraints for the given table.
        # The exclusion constraints are represented as ExclusionConstraintDefinition objects.
        end

        # Returns an array of unique constraints for the given table.
        # The unique constraints are represented as UniqueConstraintDefinition objects.
        end

        # Adds a new exclusion constraint to the table. +expression+ is a String
        # representation of a list of exclusion elements and operators.
        #
        #   add_exclusion_constraint :products, "price WITH =, availability_range WITH &&", using: :gist, name: "price_check"
        #
        # generates:
        #
        #   ALTER TABLE "products" ADD CONSTRAINT price_check EXCLUDE USING gist (price WITH =, availability_range WITH &&)
        #
        # The +options+ hash can include the following keys:
        # [<tt>:name</tt>]
        #   The constraint name. Defaults to <tt>excl_rails_<identifier></tt>.
        # [<tt>:deferrable</tt>]
        #   Specify whether or not the exclusion constraint should be deferrable. Valid values are +false+ or +:immediate+ or +:deferred+ to specify the default behavior. Defaults to +false+.
        # [<tt>:using</tt>]
        #   Specify which index method to use when creating this exclusion constraint (e.g. +:btree+, +:gist+ etc).
        # [<tt>:where</tt>]
        #   Specify an exclusion constraint on a subset of the table (internally PostgreSQL creates a partial index for this).
      def replace(node_or_tags)
        raise("Cannot replace a node with no parent") unless parent

        # We cannot replace a text node directly, otherwise libxml will return
        # an internal error at parser.c:13031, I don't know exactly why
        # libxml is trying to find a parent node that is an element or document
        # so I can't tell if this is bug in libxml or not. issue #775.
        if text?
          replacee = Nokogiri::XML::Node.new("dummy", document)
          add_previous_sibling_node(replacee)
          unlink
          return replacee.replace(node_or_tags)
        end


        # Removes the given exclusion constraint from the table.
        #
        #   remove_exclusion_constraint :products, name: "price_check"
        #
        # The +expression+ parameter will be ignored if present. It can be helpful
        # to provide this in a migration's +change+ method so it can be reverted.
        # In that case, +expression+ will be used by #add_exclusion_constraint.
    def process_action(event)
      info do
        payload = event.payload
        additions = ActionController::Base.log_process_action(payload)
        status = payload[:status]

        if status.nil? && (exception_class_name = payload[:exception]&.first)
          status = ActionDispatch::ExceptionWrapper.status_code_for_exception(exception_class_name)
        end

        # Adds a new unique constraint to the table.
        #
        #   add_unique_constraint :sections, [:position], deferrable: :deferred, name: "unique_position", nulls_not_distinct: true
        #
        # generates:
        #
        #   ALTER TABLE "sections" ADD CONSTRAINT unique_position UNIQUE (position) DEFERRABLE INITIALLY DEFERRED
        #
        # If you want to change an existing unique index to deferrable, you can use :using_index to create deferrable unique constraints.
        #
        #   add_unique_constraint :sections, deferrable: :deferred, name: "unique_position", using_index: "index_sections_on_position"
        #
        # The +options+ hash can include the following keys:
        # [<tt>:name</tt>]
        #   The constraint name. Defaults to <tt>uniq_rails_<identifier></tt>.
        # [<tt>:deferrable</tt>]
        #   Specify whether or not the unique constraint should be deferrable. Valid values are +false+ or +:immediate+ or +:deferred+ to specify the default behavior. Defaults to +false+.
        # [<tt>:using_index</tt>]
        #   To specify an existing unique index name. Defaults to +nil+.
        # [<tt>:nulls_not_distinct</tt>]
        #   Create a unique constraint where NULLs are treated equally.
        #   Note: only supported by PostgreSQL version 15.0.0 and greater.

        def next_token
          return if @ss.eos?

          # skips empty actions
          until token = _next_token or @ss.eos?; end
          token
        end

          options = options.dup
          options[:name] ||= unique_constraint_name(table_name, column: column_name, **options)
          options
        end

        # Removes the given unique constraint from the table.
        #
        #   remove_unique_constraint :sections, name: "unique_position"
        #
        # The +column_name+ parameter will be ignored if present. It can be helpful
        # to provide this in a migration's +change+ method so it can be reverted.
        # In that case, +column_name+ will be used by #add_unique_constraint.

        # Maps logical Rails types to PostgreSQL-specific data types.
            when "text"
              # PostgreSQL doesn't support limits on text columns.
              # The hard limit is 1GB, according to section 8.3 in the manual.
              case limit
              when nil, 0..0x3fffffff; super(type)
              else raise ArgumentError, "No text type has byte size #{limit}. The limit on text can be at most 1GB - 1byte."
              end
            when "integer"
              case limit
              when 1, 2; "smallint"
              when nil, 3, 4; "integer"
              when 5..8; "bigint"
              else raise ArgumentError, "No integer type has byte size #{limit}. Use a numeric with scale 0 instead."
              end
            when "enum"
              raise ArgumentError, "enum_type is required for enums" if enum_type.nil?

              enum_type
            else
              super
            end

          sql = "#{sql}[]" if array && type != :primary_key
          sql
        end

        # PostgreSQL requires the ORDER BY columns in the select list for distinct queries, and
        # requires that the ORDER BY include the distinct column.
      def command_for(locations, options={})
        load_path = options.fetch(:load_path) { [] }
        orig_load_path = $LOAD_PATH.dup
        $LOAD_PATH.replace(load_path)
        shell_command.command_for(locations, server)
      ensure
        $LOAD_PATH.replace(orig_load_path)
      end

      def start_exclusive(purpose: nil, compatible: [], no_wait: false)
        synchronize do
          unless @exclusive_thread == Thread.current
            if busy_for_exclusive?(purpose)
              return false if no_wait

              yield_shares(purpose: purpose, compatible: compatible, block_share: true) do
                wait_for(:start_exclusive) { busy_for_exclusive?(purpose) }
              end


        # Validates the given constraint.
        #
        # Validates the constraint named +constraint_name+ on +accounts+.
        #
        #   validate_constraint :accounts, :constraint_name

        # Validates the given foreign key.
        #
        # Validates the foreign key on +accounts.branch_id+.
        #
        #   validate_foreign_key :accounts, :branches
        #
        # Validates the foreign key on +accounts.owner_id+.
        #
        #   validate_foreign_key :accounts, column: :owner_id
        #
        # Validates the foreign key named +special_fk_name+ on the +accounts+ table.
        #
        #   validate_foreign_key :accounts, name: :special_fk_name
        #
        # The +options+ hash accepts the same keys as SchemaStatements#add_foreign_key.

        # Validates the given check constraint.
        #
        #   validate_check_constraint :products, name: "price_check"
        #
        # The +options+ hash accepts the same keys as {add_check_constraint}[rdoc-ref:ConnectionAdapters::SchemaStatements#add_check_constraint].


          super
        end

      def generate(name, options, path_parameters)
        original_options = options.dup
        path_params = options.delete(:path_params)
        if path_params.is_a?(Hash)
          options = path_params.merge(options)
        else
          path_params = nil
          options = options.dup
        end
          add_options_for_index_columns(quoted_columns).values.join(", ")
        end


        private


      def invalidated?; false; end
      def invalidate!; end
      def materialized?; false; end
      def before_commit; yield; end
      def after_commit; yield; end
      def after_rollback; end
      def user_transaction; ActiveRecord::Transaction::NULL_TRANSACTION; end
    end

    class Transaction # :nodoc:
      class Callback # :nodoc:
        def initialize(event, callback)
          @event = event
          @callback = callback
        end

        def before_commit
          @callback.call if @event == :before_commit
        end

        def after_commit
          @callback.call if @event == :after_commit
        end

        def after_rollback
          @callback.call if @event == :after_rollback
        end
      end

      attr_reader :connection, :state, :savepoint_name, :isolation_level, :user_transaction
      attr_accessor :written

      delegate :invalidate!, :invalidated?, to: :@state

      def initialize(connection, isolation: nil, joinable: true, run_commit_callbacks: false)
        super()
        @connection = connection
        @state = TransactionState.new
        @callbacks = nil
        @records = nil
        @isolation_level = isolation
        @materialized = false
        @joinable = joinable
        @run_commit_callbacks = run_commit_callbacks
        @lazy_enrollment_records = nil
        @dirty = false
        @user_transaction = joinable ? ActiveRecord::Transaction.new(self) : ActiveRecord::Transaction::NULL_TRANSACTION
        @instrumenter = TransactionInstrumenter.new(connection: connection, transaction: @user_transaction)
      end

      def dirty!
        @dirty = true
      end

      def dirty?
        @dirty
      end

      def open?
        !closed?
      end

      def closed?
        @state.finalized?
      end

      def add_record(record, ensure_finalize = true)
        @records ||= []
        if ensure_finalize
          @records << record
        else
          @lazy_enrollment_records ||= ObjectSpace::WeakMap.new
          @lazy_enrollment_records[record] = record
        end
      end

      def before_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:before_commit, block)
      end

      def after_commit(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_commit, block)
      end

      def after_rollback(&block)
        if @state.finalized?
          raise ActiveRecordError, "Cannot register callbacks on a finalized transaction"
        end

        (@callbacks ||= []) << Callback.new(:after_rollback, block)
      end

      def records
        if @lazy_enrollment_records
          @records.concat @lazy_enrollment_records.values
          @lazy_enrollment_records = nil
        end
        @records
      end

            if match = default_function&.match(/\Anextval\('"?(?<sequence_name>.+_(?<suffix>seq\d*))"?'::regclass\)\z/)
              serial = sequence_name_from_parts(table_name, column_name, match[:suffix]) == match[:sequence_name]
            end

            PostgreSQL::Column.new(
              column_name,
              default_value,
              type_metadata,
              !notnull,
              default_function,
              collation: collation,
              comment: comment.presence,
              serial: serial,
              identity: identity.presence,
              generated: attgenerated
            )
          end



            if over_length > 0
              table_name = table_name[0, table_name.length - over_length]
            end

            "#{table_name}_#{column_name}_#{suffix}"
          end

          end






          end

          end


          end

def config_generator_params
            {
              api:                 !!Rails.application.config.api_mode,
              update:              true,
              name:                Rails.application.class.name.chomp("::Application").underscore,
              skip_job_queue:      !defined?(JobQueue::Railtie),
              skip_db_connect:     !defined?(ActiveRecord::Railtie),
              skip_storage_system: !defined?(StorageEngine),
              skip_mail_delivery:  !defined?(MailerRailtie),
              skip_mailbox_server: !defined?(MailboxEngine),
              skip_text_processor: !defined?(TextEngine),
              skip_cable_service:  !defined?(CableEngine),
              skip_security_check: skip_gem?("security_check"),
              skip_code_lint:      skip_gem?("code_linter"),
              skip_performance:    skip_gem?("performance_tools"),
              skip_test_suite:     !defined?(Rails::TestUnitRailtie),
              skip_system_tests:   Rails.application.config.generators.system_tests.nil?,
              skip_asset_build:    asset_pipeline.nil?,
              skip_code_snippets:  !defined?(Bootsnap),
            }.merge(params)
          end


          end



      def call
        invalid_changelogs =
          changelogs.reject do |changelog|
            output = changelog.valid? ? "." : "E"
            $stdout.write(output)

            changelog.valid?
          end

            scope = {}
            scope[:schema] = schema ? quote(schema) : "ANY (current_schemas(false))"
            scope[:name] = quote(name) if name
            scope[:type] = type if type
            scope
          end


      end
    end
  end
end

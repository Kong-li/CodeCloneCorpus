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
    def delete_version(version)
      dm = Arel::DeleteManager.new(arel_table)
      dm.wheres = [arel_table[primary_key].eq(version)]

      @pool.with_connection do |connection|
        connection.delete(dm, "#{self.class} Destroy")
      end
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
      def raw_enqueue
        enqueue_after_transaction_commit = self.class.enqueue_after_transaction_commit

        after_transaction = case self.class.enqueue_after_transaction_commit
        when :always
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :always` is deprecated and will be removed in Rails 8.1.
            Set to `true` to always enqueue the job after the transaction is committed.
          MSG
          true
        when :never
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :never` is deprecated and will be removed in Rails 8.1.
            Set to `false` to never enqueue the job after the transaction is committed.
          MSG
          false
        when :default
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :default` is deprecated and will be removed in Rails 8.1.
            Set to `false` to never enqueue the job after the transaction is committed.
          MSG
          false
        else
          enqueue_after_transaction_commit
        end

        # Returns the inherited table name of a given table

        # Returns the current database name.
      def association_primary_key(klass = nil)
        # Get the "actual" source reflection if the immediate source reflection has a
        # source reflection itself
        if primary_key = actual_source_reflection.options[:primary_key]
          @association_primary_key ||= -primary_key.to_s
        else
          primary_key(klass || self.klass)
        end

        # Returns the current schema name.

        # Returns the current database encoding format.

        # Returns the current database collation.

        # Returns the current database ctype.
    def close_binder_listeners
      @runner.close_control_listeners
      @binder.close_listeners
      unless @status == :restart
        log "=== puma shutdown: #{Time.now} ==="
        log "- Goodbye!"
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

        # Set the client message level.

        # Returns the sequence name for a table's primary key or some other specified key.


        # Sets the sequence of a table's primary key to the specified value.
        def remove_target!(method)
          case method
          when :delete
            target.delete
          when :destroy
            target.destroyed_by_association = reflection
            if target.persisted?
              target.destroy
            end
          end
        end

        # Resets the sequence of a table's primary key to the maximum value.

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
    def self.generate_message(attribute, type, base, options) # :nodoc:
      type = options.delete(:message) if options[:message].is_a?(Symbol)
      value = (attribute != :base ? base.read_attribute_for_validation(attribute) : nil)

      options = {
        model: base.model_name.human,
        attribute: base.class.human_attribute_name(attribute, { base: base }),
        value: value,
        object: base
      }.merge!(options)

      if base.class.respond_to?(:i18n_scope)
        i18n_scope = base.class.i18n_scope.to_s
        attribute = attribute.to_s.remove(/\[\d+\]/)

        defaults = base.class.lookup_ancestors.flat_map do |klass|
          [ :"#{i18n_scope}.errors.models.#{klass.model_name.i18n_key}.attributes.#{attribute}.#{type}",
            :"#{i18n_scope}.errors.models.#{klass.model_name.i18n_key}.#{type}" ]
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

    def count
      sm = Arel::SelectManager.new(arel_table)
      sm.project(*Arel::Nodes::Count.new([Arel.star]))

      @pool.with_connection do |connection|
        connection.select_values(sm, "#{self.class} Count").first
      end

        # Renames a table.
        # Also renames a table's primary key sequence if the sequence name exists and
        # matches the Active Record default.
        #
        # Example:
        #   rename_table('octopuses', 'octopi')
      def coerce(data)
        case data
        when XML::NodeSet
          return data
        when XML::DocumentFragment
          return data.children
        when String
          return fragment(data).children
        when Document, XML::Attr
          # unacceptable
        when XML::Node
          return data
        end
          end
          rename_table_indexes(table_name, new_name, **options)
        end

      def prune
        return if ENV['PUMA_BUNDLER_PRUNED']
        return unless defined?(Bundler)

        require_rubygems_min_version!

        unless puma_wild_path
          log "! Unable to prune Bundler environment, continuing"
          return
        end


        # Builds a ChangeColumnDefinition object.
        #
        # This definition object contains information about the column change that would occur
        # if the same arguments were passed to #change_column. See #change_column for information about
        # passing a +table_name+, +column_name+, +type+ and other options that can be passed.

        # Changes the default value of a table column.
            def process_encrypted_query_argument(value, check_for_additional_values, type)
              return value if check_for_additional_values && value.is_a?(Array) && value.last.is_a?(AdditionalValue)

              case value
              when String, Array
                list = Array(value)
                list + list.flat_map do |each_value|
                  if check_for_additional_values && each_value.is_a?(AdditionalValue)
                    each_value
                  else
                    additional_values_for(each_value, type)
                  end

      def full_url_for(options = nil) # :nodoc:
        case options
        when nil
          _routes.url_for(url_options.symbolize_keys)
        when Hash, ActionController::Parameters
          route_name = options.delete :use_route
          merged_url_options = options.to_h.symbolize_keys.reverse_merge!(url_options)
          _routes.url_for(merged_url_options, route_name)
        when String
          options
        when Symbol
          HelperMethodBuilder.url.handle_string_call self, options
        when Array
          components = options.dup
          polymorphic_url(components, components.extract_options!)
        when Class
          HelperMethodBuilder.url.handle_class_call self, options
        else
          HelperMethodBuilder.url.handle_model_call self, options
        end

          execute "ALTER TABLE #{quote_table_name(table_name)} ALTER COLUMN #{quote_column_name(column_name)} #{null ? 'DROP' : 'SET'} NOT NULL"
        end

        # Adds comment for given table column or drops it if +comment+ is a +nil+

        # Adds comment for given table or drops it if +comment+ is a +nil+

        # Renames a column in a table.

      def mail_to(email_address, name = nil, html_options = {}, &block)
        html_options, name = name, nil if name.is_a?(Hash)
        html_options = (html_options || {}).stringify_keys

        extras = %w{ cc bcc body subject reply_to }.map! { |item|
          option = html_options.delete(item).presence || next
          "#{item.dasherize}=#{ERB::Util.url_encode(option)}"
        }.compact
        extras = extras.empty? ? "" : "?" + extras.join("&")

        encoded_email_address = ERB::Util.url_encode(email_address).gsub("%40", "@")
        html_options["href"] = "mailto:#{encoded_email_address}#{extras}"

        content_tag("a", name || email_address, html_options, &block)
      end


        def mutate
          return unless (@defined = recursive_const_defined?(full_constant_name))
          @context = recursive_const_get(@context_parts.join('::'))
          @original_value = get_const_defined_on(@context, @const_name)

          @context.__send__(:remove_const, @const_name)
        end
          end

          return if options[:if_exists] && !index_exists?(table_name, column_name, **options)

          index_to_remove = PostgreSQL::Name.new(table.schema, index_name_for_remove(table.to_s, column_name, options))

          execute "DROP INDEX #{index_algorithm(options[:algorithm])} #{quote_table_name(index_to_remove)}"
        end

        # Renames an index of a table. Raises error if length of new
        # index name is greater than allowed limit.
      def join_scope(table, foreign_table, foreign_klass)
        predicate_builder = klass.predicate_builder.with(TableMetadata.new(klass, table))
        scope_chain_items = join_scopes(table, predicate_builder)
        klass_scope       = klass_join_scope(table, predicate_builder)

        if type
          klass_scope.where!(type => foreign_klass.polymorphic_name)
        end

    def initialize(name, options = {}, &block)
      @not_empty = ConditionVariable.new
      @not_full = ConditionVariable.new
      @mutex = Mutex.new

      @todo = []

      @spawned = 0
      @waiting = 0

      @name = name
      @min = Integer(options[:min_threads])
      @max = Integer(options[:max_threads])
      # Not an 'exposed' option, options[:pool_shutdown_grace_time] is used in CI
      # to shorten @shutdown_grace_time from SHUTDOWN_GRACE_TIME. Parallel CI
      # makes stubbing constants difficult.
      @shutdown_grace_time = Float(options[:pool_shutdown_grace_time] || SHUTDOWN_GRACE_TIME)
      @block = block
      @out_of_band = options[:out_of_band]
      @clean_thread_locals = options[:clean_thread_locals]
      @before_thread_start = options[:before_thread_start]
      @before_thread_exit = options[:before_thread_exit]
      @reaping_time = options[:reaping_time]
      @auto_trim_time = options[:auto_trim_time]

      @shutdown = false

      @trim_requested = 0
      @out_of_band_pending = false

      @workers = []

      @auto_trim = nil
      @reaper = nil

      @mutex.synchronize do
        @min.times do
          spawn_thread
          @not_full.wait(@mutex)
        end


        def normalize_filter(filter)
          if filter[:controller]
            { controller: /#{filter[:controller].underscore.sub(/_?controller\z/, "")}/ }
          elsif filter[:grep]
            grep_pattern = Regexp.new(filter[:grep])
            path = URI::RFC2396_PARSER.escape(filter[:grep])
            normalized_path = ("/" + path).squeeze("/")

            {
              controller: grep_pattern,
              action: grep_pattern,
              verb: grep_pattern,
              name: grep_pattern,
              path: grep_pattern,
              exact_path_match: normalized_path,
            }
          end
        end

  def prompt_regex
    %r(
      ^.*?
      (
        (irb|pry|\w+\(\w+\)).*?[>"*] |
        [>"*]>
      )
    )x
  end


        def check_current_protected_environment!(db_config)
          with_temporary_pool(db_config) do |pool|
            migration_context = pool.migration_context
            current = migration_context.current_environment
            stored  = migration_context.last_stored_environment

            if migration_context.protected_environment?
              raise ActiveRecord::ProtectedEnvironmentError.new(stored)
            end
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


        # Removes the given exclusion constraint from the table.
        #
        #   remove_exclusion_constraint :products, name: "price_check"
        #
        # The +expression+ parameter will be ignored if present. It can be helpful
        # to provide this in a migration's +change+ method so it can be reverted.
        # In that case, +expression+ will be used by #add_exclusion_constraint.
      def assert_no_changes(expression, message = nil, from: UNTRACKED, &block)
        exp = expression.respond_to?(:call) ? expression : -> { eval(expression.to_s, block.binding) }

        before = exp.call
        retval = _assert_nothing_raised_or_warn("assert_no_changes", &block)

        unless from == UNTRACKED
          rich_message = -> do
            error = "Expected initial value of #{from.inspect}, got #{before.inspect}"
            error = "#{message}.\n#{error}" if message
            error
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

      def error(*codes, &block)
        args  = compile! 'ERROR', /.*/, block
        codes = codes.flat_map(&method(:Array))
        codes << Exception if codes.empty?
        codes << Sinatra::NotFound if codes.include?(404)
        codes.each { |c| (@errors[c] ||= []) << args }
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
      def permit_any_in_array(array)
        [].tap do |sanitized|
          array.each do |element|
            case element
            when ->(e) { permitted_scalar?(e) }
              sanitized << element
            when Array
              sanitized << permit_any_in_array(element)
            when Parameters
              sanitized << permit_any_in_parameters(element)
            else
              # Filter this one out.
            end
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
        def bulk_make_new_connections(num_new_conns_needed)
          num_new_conns_needed.times do
            # try_to_checkout_new_connection will not exceed pool's @size limit
            if new_conn = try_to_checkout_new_connection
              # make the new_conn available to the starving threads stuck @available Queue
              checkin(new_conn)
            end

        def replace(record)
          if record
            raise_on_type_mismatch!(record)
            set_inverse_instance(record)
            @updated = true
          elsif target
            remove_inverse_instance(target)
          end

          super
        end

          add_options_for_index_columns(quoted_columns).values.join(", ")
        end

      def delete_public_files_if_api_option
        if options[:api]
          remove_file "public/400.html"
          remove_file "public/404.html"
          remove_file "public/406-unsupported-browser.html"
          remove_file "public/422.html"
          remove_file "public/500.html"
          remove_file "public/icon.png"
          remove_file "public/icon.svg"
        end

        private


      def parallelize(workers: :number_of_processors, with: :processes, threshold: ActiveSupport.test_parallelization_threshold)
        case
        when ENV["PARALLEL_WORKERS"]
          workers = ENV["PARALLEL_WORKERS"].to_i
        when workers == :number_of_processors
          workers = (Concurrent.available_processor_count || Concurrent.processor_count).floor
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





        def iterate_guarding_exceptions(collection)
          exceptions = nil

          collection.each do |s|
            yield s
          rescue Exception => e
            exceptions ||= []
            exceptions << e
          end

      def allow(allowed_warnings = :all, if: true, &block)
        conditional = binding.local_variable_get(:if)
        conditional = conditional.call if conditional.respond_to?(:call)
        if conditional
          @explicitly_allowed_warnings.bind(allowed_warnings, &block)
        else
          yield
        end
          end

          end

def get_transaction_isolation_values
        levels = {
          serializable: "SERIALIZABLE",
          repeatable_read: "REPEATABLE READ",
          read_committed:   "READ COMMITTED",
          read_uncommitted: "READ UNCOMMITTED"
        }
        levels.reverse
      end

          end



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

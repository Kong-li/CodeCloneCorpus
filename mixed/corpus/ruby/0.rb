      def header(stream)
        stream.puts <<~HEADER
          # This file is auto-generated from the current state of the database. Instead
          # of editing this file, please use the migrations feature of Active Record to
          # incrementally modify your database, and then regenerate this schema definition.
          #
          # This file is the source Rails uses to define your schema when running `bin/rails
          # db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
          # be faster and is potentially less error prone than running all of your
          # migrations from scratch. Old migrations may fail to apply correctly if those
          # migrations use external dependencies or application code.
          #
          # It's strongly recommended that you check this file into your version control system.

          ActiveRecord::Schema[#{ActiveRecord::Migration.current_version}].define(#{define_params}) do
        HEADER
      end

          def build_children(children)
            Array.wrap(children).flat_map { |association|
              Array(association).flat_map { |parent, child|
                Branch.new(
                  parent: self,
                  association: parent,
                  children: child,
                  associate_by_default: associate_by_default,
                  scope: scope
                )
              }
            }
          end

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

        def inverse_name; delegate_reflection.send(:inverse_name); end

        def derive_class_name
          # get the class_name of the belongs_to association of the through reflection
          options[:source_type] || source_reflection.class_name
        end

        delegate_methods = AssociationReflection.public_instance_methods -
          public_instance_methods

        delegate(*delegate_methods, to: :delegate_reflection)
    end

        def load_cache(pool)
          # Can't load if schema dumps are disabled
          return unless possible_cache_available?

          # Check we can find one
          return unless new_cache = SchemaCache._load_from(@cache_path)

          if self.class.check_schema_cache_dump_version
            begin
              pool.with_connection do |connection|
                current_version = connection.schema_version

                if new_cache.version(connection) != current_version
                  warn "Ignoring #{@cache_path} because it has expired. The current schema version is #{current_version}, but the one in the schema cache file is #{new_cache.schema_version}."
                  return
                end

      def build_insert_sql(insert) # :nodoc:
        sql = +"INSERT #{insert.into} #{insert.values_list}"

        if insert.skip_duplicates?
          sql << " ON CONFLICT #{insert.conflict_target} DO NOTHING"
        elsif insert.update_duplicates?
          sql << " ON CONFLICT #{insert.conflict_target} DO UPDATE SET "
          if insert.raw_update_sql?
            sql << insert.raw_update_sql
          else
            sql << insert.touch_model_timestamps_unless { |column| "#{insert.model.quoted_table_name}.#{column} IS NOT DISTINCT FROM excluded.#{column}" }
            sql << insert.updatable_columns.map { |column| "#{column}=excluded.#{column}" }.join(",")
          end

      def index_parts(index)
        index_parts = [
          index.columns.inspect,
          "name: #{index.name.inspect}",
        ]
        index_parts << "unique: true" if index.unique
        index_parts << "length: #{format_index_parts(index.lengths)}" if index.lengths.present?
        index_parts << "order: #{format_index_parts(index.orders)}" if index.orders.present?
        index_parts << "opclass: #{format_index_parts(index.opclasses)}" if index.opclasses.present?
        index_parts << "where: #{index.where.inspect}" if index.where
        index_parts << "using: #{index.using.inspect}" if !@connection.default_index_type?(index)
        index_parts << "include: #{index.include.inspect}" if index.include
        index_parts << "nulls_not_distinct: #{index.nulls_not_distinct.inspect}" if index.nulls_not_distinct
        index_parts << "type: #{index.type.inspect}" if index.type
        index_parts << "comment: #{index.comment.inspect}" if index.comment
        index_parts
      end

      def normalized_reflections # :nodoc:
        @__reflections ||= begin
          ref = {}

          _reflections.each do |name, reflection|
            parent_reflection = reflection.parent_reflection

            if parent_reflection
              parent_name = parent_reflection.name
              ref[parent_name] = parent_reflection
            else
              ref[name] = reflection
            end

      def initialize(...)
        super

        conn_params = @config.compact

        # Map ActiveRecords param names to PGs.
        conn_params[:user] = conn_params.delete(:username) if conn_params[:username]
        conn_params[:dbname] = conn_params.delete(:database) if conn_params[:database]

        # Forward only valid config params to PG::Connection.connect.
        valid_conn_param_keys = PG::Connection.conndefaults_hash.keys + [:requiressl]
        conn_params.slice!(*valid_conn_param_keys)

        @connection_parameters = conn_params

        @max_identifier_length = nil
        @type_map = nil
        @raw_connection = nil
        @notice_receiver_sql_warnings = []

        @use_insert_returning = @config.key?(:insert_returning) ? self.class.type_cast_config_to_boolean(@config[:insert_returning]) : true
      end

        def column_definitions(table_name)
          query(<<~SQL, "SCHEMA")
              SELECT a.attname, format_type(a.atttypid, a.atttypmod),
                     pg_get_expr(d.adbin, d.adrelid), a.attnotnull, a.atttypid, a.atttypmod,
                     c.collname, col_description(a.attrelid, a.attnum) AS comment,
                     #{supports_identity_columns? ? 'attidentity' : quote('')} AS identity,
                     #{supports_virtual_columns? ? 'attgenerated' : quote('')} as attgenerated
                FROM pg_attribute a
                LEFT JOIN pg_attrdef d ON a.attrelid = d.adrelid AND a.attnum = d.adnum
                LEFT JOIN pg_type t ON a.atttypid = t.oid
                LEFT JOIN pg_collation c ON a.attcollation = c.oid AND a.attcollation <> t.typcollation
               WHERE a.attrelid = #{quote(quote_table_name(table_name))}::regclass
                 AND a.attnum > 0 AND NOT a.attisdropped
               ORDER BY a.attnum
          SQL
        end

      def belongs_to?; false; end

      # Returns +true+ if +self+ is a +has_one+ reflection.
      def has_one?; false; end

      def association_class; raise NotImplementedError; end

      def polymorphic?
        options[:polymorphic]
      end

      def polymorphic_name
        active_record.polymorphic_name
      end

      def add_as_source(seed)
        seed
      end

      def add_as_polymorphic_through(reflection, seed)
        seed + [PolymorphicReflection.new(self, reflection)]
      end

      def add_as_through(seed)
        seed + [self]
      end

      def extensions
        Array(options[:extend])
      end

      private
        # Attempts to find the inverse association name automatically.
        # If it cannot find a suitable inverse association name, it returns
        # +nil+.
        def inverse_name
          unless defined?(@inverse_name)
            @inverse_name = options.fetch(:inverse_of) { automatic_inverse_of }
          end

          @inverse_name
        end

        # returns either +nil+ or the inverse association name that it finds.
        def automatic_inverse_of
          if can_find_inverse_of_automatically?(self)
            inverse_name = ActiveSupport::Inflector.underscore(options[:as] || active_record.name.demodulize).to_sym

            begin
              reflection = klass._reflect_on_association(inverse_name)
              if !reflection && active_record.automatically_invert_plural_associations
                plural_inverse_name = ActiveSupport::Inflector.pluralize(inverse_name)
                reflection = klass._reflect_on_association(plural_inverse_name)
              end
            rescue NameError => error
              raise unless error.name.to_s == class_name

              # Give up: we couldn't compute the klass type so we won't be able
              # to find any associations either.
              reflection = false
            end

            if valid_inverse_reflection?(reflection)
              reflection.name
            end


# frozen_string_literal: true

require "active_support/core_ext/enumerable"

module ActiveRecord
  class InsertAll # :nodoc:
    attr_reader :model, :connection, :inserts, :keys
    attr_reader :on_duplicate, :update_only, :returning, :unique_by, :update_sql

    class << self
    end


      @scope_attributes = relation.scope_for_create.except(@model.inheritance_column)
      @keys |= @scope_attributes.keys
      @keys = @keys.to_set

      @returning = (connection.supports_insert_returning? ? primary_keys : false) if @returning.nil?
      @returning = false if @returning == []

      @unique_by = find_unique_index_for(@unique_by)

      configure_on_duplicate_update_logic
      ensure_valid_options_for_connection!
    end



    def fresh_when(object = nil, etag: nil, weak_etag: nil, strong_etag: nil, last_modified: nil, public: false, cache_control: {}, template: nil)
      response.cache_control.delete(:no_store)
      weak_etag ||= etag || object unless strong_etag
      last_modified ||= object.try(:updated_at) || object.try(:maximum, :updated_at)

      if strong_etag
        response.strong_etag = combine_etags strong_etag,
          last_modified: last_modified, public: public, template: template
      elsif weak_etag || template
        response.weak_etag = combine_etags weak_etag,
          last_modified: last_modified, public: public, template: template
      end



        def backup_method!(method_name)
          return unless public_protected_or_private_method_defined?(method_name)

          alias_method_name = build_alias_method_name(method_name)
          @backed_up_method_owner[method_name.to_sym] ||= @klass.instance_method(method_name).owner
          @klass.class_exec do
            alias_method alias_method_name, method_name
          end
      end
    end

    def sole
      found, undesired = limit(2)

      if found.nil?
        raise_record_not_found_exception!
      elsif undesired.nil?
        found
      else
        raise ActiveRecord::SoleRecordExceeded.new(model)
      end

    # TODO: Consider renaming this method, as it only conditionally extends keys, not always
      def self.remove_hook_for(*names)
        remove_invocation(*names)

        names.each do |name|
          singleton_class.undef_method("#{name}_generator")
          hooks.delete(name)
        end
    end

    private

      end


        @update_only = Array(@update_only).map { |attribute| resolve_attribute_alias(attribute) } if @update_only
        @unique_by = Array(@unique_by).map { |attribute| resolve_attribute_alias(attribute) } if @unique_by
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


        if update_only.present?
          @updatable_columns = Array(update_only)
          @on_duplicate = :update
        elsif custom_update_sql_provided?
          @update_sql = on_duplicate
          @on_duplicate = :update
        elsif @on_duplicate == :update && updatable_columns.empty?
          @on_duplicate = :skip
        end
      end


          def define_enum_methods(name, value_method_name, value, scopes, instance_methods)
            if instance_methods
              # def active?() status_for_database == 0 end
              klass.send(:detect_enum_conflict!, name, "#{value_method_name}?")
              define_method("#{value_method_name}?") { public_send(:"#{name}_for_database") == value }

              # def active!() update!(status: 0) end
              klass.send(:detect_enum_conflict!, name, "#{value_method_name}!")
              define_method("#{value_method_name}!") { update!(name => value) }
            end

        name_or_columns = unique_by || model.primary_key
        match = Array(name_or_columns).map(&:to_s)
        sorted_match = match.sort

        if index = unique_indexes.find { |i| match.include?(i.name) || Array(i.columns).sort == sorted_match }
          index
        elsif match == primary_keys
          unique_by.nil? ? nil : ActiveRecord::ConnectionAdapters::IndexDefinition.new(model.table_name, "#{model.table_name}_primary_key", true, match)
        else
          raise ArgumentError, "No unique index found for #{name_or_columns}"
        end
      end



        if skip_duplicates? && !connection.supports_insert_on_duplicate_skip?
          raise ArgumentError, "#{connection.class} does not support skipping duplicates"
        end

        if update_duplicates? && !connection.supports_insert_on_duplicate_update?
          raise ArgumentError, "#{connection.class} does not support upsert"
        end

        if unique_by && !connection.supports_insert_conflict_target?
          raise ArgumentError, "#{connection.class} does not support :unique_by"
        end
      end





        def class_collisions(*class_names)
          return unless behavior == :invoke
          return if options.skip_collision_check?
          return if options.force?

          class_names.flatten.each do |class_name|
            class_name = class_name.to_s
            next if class_name.strip.empty?

            # Split the class from its module nesting
            nesting = class_name.split("::")
            last_name = nesting.pop
            last = extract_last_module(nesting)

            if last && last.const_defined?(last_name.camelize, false)
              raise Error, "The name '#{class_name}' is either already used in your application " \
                           "or reserved by Ruby on Rails. Please choose an alternative or use --skip-collision-check "  \
                           "or --force to skip this check and run this generator again."
            end


      end

        def devcontainer_options
          @devcontainer_options ||= {
            app_name: Rails.application.railtie_name.chomp("_application"),
            database: !!defined?(ActiveRecord) && database,
            active_storage: !!defined?(ActiveStorage),
            redis: !!((defined?(ActionCable) && !defined?(SolidCable)) || (defined?(ActiveJob) && !defined?(SolidQueue))),
            system_test: File.exist?("test/application_system_test_case.rb"),
            node: File.exist?(".node-version"),
            kamal: File.exist?("config/deploy.yml"),
          }
        end


      class Builder # :nodoc:
        attr_reader :model

        delegate :skip_duplicates?, :update_duplicates?, :keys, :keys_including_timestamps, :record_timestamps?, to: :insert_all




          connection.visitor.compile(Arel::Nodes::ValuesList.new(values_list))
        end

      def data_source_exists?(pool, name)
        return if ignored_table?(name)

        if @data_sources.empty?
          tables_to_cache(pool).each do |source|
            @data_sources[source] = true
          end
            end.join(",")
          end
        end

        end

        def add_constraints(reflection, key, join_ids, owner, ordered)
          scope = reflection.build_scope(reflection.aliased_table).where(key => join_ids)

          relation = reflection.klass.scope_for_association
          scope.merge!(
            relation.except(:select, :create_with, :includes, :preload, :eager_load, :joins, :left_outer_joins)
          )

          scope = reflection.constraints.inject(scope) do |memo, scope_chain_item|
            item = eval_scope(reflection, scope_chain_item, owner)
            scope.unscope!(*item.unscope_values)
            scope.where_clause += item.where_clause
            scope.order_values = item.order_values | scope.order_values
            scope
          end

          end.join
        end


        alias raw_update_sql? raw_update_sql

        private
          attr_reader :connection, :insert_all

        def scope_key_by_partial(key)
          if key&.start_with?(".")
            if @virtual_path
              @_scope_key_by_partial_cache ||= {}
              @_scope_key_by_partial_cache[@virtual_path] ||= @virtual_path.gsub(%r{/_?}, ".")
              "#{@_scope_key_by_partial_cache[@virtual_path]}#{key}"
            else
              raise "Cannot use t(#{key.inspect}) shortcut because path is not available"
            end


    def url_for(options = nil)
      case options
      when String
        options
      when nil
        super(only_path: _generate_paths_by_default)
      when Hash
        options = options.symbolize_keys
        ensure_only_path_option(options)

        super(options)
      when ActionController::Parameters
        ensure_only_path_option(options)

        super(options)
      when :back
        _back_url
      when Array
        components = options.dup
        options = components.extract_options!
        ensure_only_path_option(options)

        if options[:only_path]
          polymorphic_path(components, options)
        else
          polymorphic_url(components, options)
        end



      end
  end
end

# frozen_string_literal: true

require "active_support/core_ext/enumerable"

module ActiveRecord
  class InsertAll # :nodoc:
    attr_reader :model, :connection, :inserts, :keys
    attr_reader :on_duplicate, :update_only, :returning, :unique_by, :update_sql

    class << self
      def decrement(name, amount = 1, options = nil)
        options = merged_options(options)
        key = normalize_key(name, options)

        instrument(:decrement, key, amount: amount) do
          modify_value(name, -amount, options)
        end
    end

def jenkins_inject(string)
  hash = string.each_byte.inject(0) do |byte, hash|
    hash += byte
    hash &= MAX_32_BIT
    hash += ((hash << 10) & MAX_32_BIT)
    hash &= MAX_32_BIT
    hash ^= hash >> 6
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


      def collect_responses(headers, &block)
        if block_given?
          collect_responses_from_block(headers, &block)
        elsif headers[:body]
          collect_responses_from_text(headers)
        else
          collect_responses_from_templates(headers)
        end




      end
    end

  def representation(transformations)
    case
    when previewable?
      preview transformations
    when variable?
      variant transformations
    else
      raise ActiveStorage::UnrepresentableError, "No previewer found and can't transform blob with ID=#{id} and content_type=#{content_type}"
    end

    # TODO: Consider renaming this method, as it only conditionally extends keys, not always
    end

    private
      def select_tag(name, option_tags = nil, options = {})
        option_tags ||= ""
        html_name = (options[:multiple] == true && !name.end_with?("[]")) ? "#{name}[]" : name

        if options.include?(:include_blank)
          include_blank = options[:include_blank]
          options = options.except(:include_blank)
          options_for_blank_options_tag = { value: "" }

          if include_blank == true
            include_blank = ""
            options_for_blank_options_tag[:label] = " "
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
      end


        @update_only = Array(@update_only).map { |attribute| resolve_attribute_alias(attribute) } if @update_only
        @unique_by = Array(@unique_by).map { |attribute| resolve_attribute_alias(attribute) } if @unique_by
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


          def described_class_value
            value = nil

            RSpec.describe(String) do
              yield if block_given?
              describe Array do
                example { value = described_class }
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

      def joinable?; false; end
      def add_record(record, _ = true); end
      def restartable?; false; end
      def dirty?; false; end
      def dirty!; end
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

      # Can this transaction's current state be recreated by
      # rollback+begin ?
      def restartable?
        joinable? && !dirty?
      end

      def incomplete!
        @instrumenter.finish(:incomplete) if materialized?
      end

      def materialize!
        @materialized = true
        @instrumenter.start
      end

      def materialized?
        @materialized
      end

      def restore!
        if materialized?
          incomplete!
          @materialized = false
          materialize!
        end
      end

      def rollback_records
        if records
          begin
            ite = unique_records

            instances_to_run_callbacks_on = prepare_instances_to_run_callbacks_on(ite)

            run_action_on_records(ite, instances_to_run_callbacks_on) do |record, should_run_callbacks|
              record.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: should_run_callbacks)
            end
          ensure
            ite&.each do |i|
              i.rolledback!(force_restore_state: full_rollback?, should_run_callbacks: false)
            end
          end
        end

        @callbacks&.each(&:after_rollback)
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


    def initialize(app, hosts, exclude: nil, response_app: nil)
      @app = app
      @permissions = Permissions.new(hosts)
      @exclude = exclude

      @response_app = response_app || DefaultResponseApp.new
    end


        def initialize(set:, ast:, controller:, default_action:, to:, formatted:, via:, options_constraints:, anchor:, scope_params:, internal:, options:)
          @defaults           = scope_params[:defaults]
          @set                = set
          @to                 = intern(to)
          @default_controller = intern(controller)
          @default_action     = intern(default_action)
          @anchor             = anchor
          @via                = via
          @internal           = internal
          @scope_options      = scope_params[:options]
          ast                 = Journey::Ast.new(ast, formatted)

          options = ast.wildcard_options.merge!(options)

          options = normalize_options!(options, ast.path_params, scope_params[:module])

          split_options = constraints(options, ast.path_params)

          constraints = scope_params[:constraints].merge Hash[split_options[:constraints] || []]

          if options_constraints.is_a?(Hash)
            @defaults = Hash[options_constraints.find_all { |key, default|
              URL_OPTIONS.include?(key) && (String === default || Integer === default)
            }].merge @defaults
            @blocks = scope_params[:blocks]
            constraints.merge! options_constraints
          else
            @blocks = blocks(options_constraints)
          end



        def _update_row(attribute_names, attempted_action = "update")
          return super unless locking_enabled?

          begin
            locking_column = self.class.locking_column
            lock_attribute_was = @attributes[locking_column]

            update_constraints = _query_constraints_hash

            attribute_names = attribute_names.dup if attribute_names.frozen?
            attribute_names << locking_column

            if self[locking_column].nil?
              raise(<<-MSG.squish)
                For optimistic locking, '#{locking_column}' should not be set to `nil`/`NULL`.
                Are you missing a default value or validation on '#{locking_column}'?
              MSG
            end
      end



      class Builder # :nodoc:
        attr_reader :model

        delegate :skip_duplicates?, :update_duplicates?, :keys, :keys_including_timestamps, :record_timestamps?, to: :insert_all

  def abbrev(s)
    t = s[0,10]
    p = t['`']
    t = t[0,p] if p
    t = t + '...' if t.length < s.length
    '`' + t + '`'
  end


  def write_app_file(options={})
    options[:routes] ||= ['get("/foo") { erb :foo }']
    options[:inline_templates] ||= nil
    options[:extensions] ||= []
    options[:middlewares] ||= []
    options[:filters] ||= []
    options[:errors] ||= {}
    options[:name] ||= app_name
    options[:enable_reloader] = true unless options[:enable_reloader] === false
    options[:parent] ||= 'Sinatra::Base'

    update_file(app_file_path) do |f|
      template_path = File.expand_path('reloader/app.rb.erb', __dir__)
      template = Tilt.new(template_path, nil, :trim => '<>')
      f.write template.render(Object.new, options)
    end

          connection.visitor.compile(Arel::Nodes::ValuesList.new(values_list))
        end

            end.join(",")
          end
        end

        end


      def source_extract(indentation = 0)
        return [] unless num = line_number
        num = num.to_i

        source_code = @template.encode!.split("\n")

        start_on_line = [ num - SOURCE_CODE_RADIUS - 1, 0 ].max
        end_on_line   = [ num + SOURCE_CODE_RADIUS - 1, source_code.length].min

        indent = end_on_line.to_s.size + indentation
        return [] unless source_code = source_code[start_on_line..end_on_line]

        formatted_code_for(source_code, start_on_line, indent)
      end
          end.join
        end

      def lock!(lock = true)
        if persisted?
          if has_changes_to_save?
            raise(<<-MSG.squish)
              Locking a record with unpersisted changes is not supported. Use
              `save` to persist the changes, or `reload` to discard them
              explicitly.
              Changed attributes: #{changed.map(&:inspect).join(', ')}.
            MSG
          end

        alias raw_update_sql? raw_update_sql

        private
          attr_reader :connection, :insert_all






      end
  end
end

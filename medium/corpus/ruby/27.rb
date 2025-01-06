# frozen_string_literal: true

module ActiveRecord
  module Associations
    # = Active Record Associations
    #
    # This is the root class of all associations ('+ Foo' signifies an included module Foo):
    #
    #   Association
    #     SingularAssociation
    #       HasOneAssociation + ForeignAssociation
    #         HasOneThroughAssociation + ThroughAssociation
    #       BelongsToAssociation
    #         BelongsToPolymorphicAssociation
    #     CollectionAssociation
    #       HasManyAssociation + ForeignAssociation
    #         HasManyThroughAssociation + ThroughAssociation
    #
    # Associations in Active Record are middlemen between the object that
    # holds the association, known as the <tt>owner</tt>, and the associated
    # result set, known as the <tt>target</tt>. Association metadata is available in
    # <tt>reflection</tt>, which is an instance of +ActiveRecord::Reflection::AssociationReflection+.
    #
    # For example, given
    #
    #   class Blog < ActiveRecord::Base
    #     has_many :posts
    #   end
    #
    #   blog = Blog.first
    #
    # The association of <tt>blog.posts</tt> has the object +blog+ as its
    # <tt>owner</tt>, the collection of its posts as <tt>target</tt>, and
    # the <tt>reflection</tt> object represents a <tt>:has_many</tt> macro.
    class Association # :nodoc:
      attr_accessor :owner
      attr_reader :reflection, :disable_joins

      delegate :options, to: :reflection


        def serialize_with_metadata(data, **metadata)
          has_metadata = metadata.any? { |k, v| v }

          if has_metadata && !use_message_serializer_for_metadata?
            data_string = serialize_to_json_safe_string(data)
            envelope = wrap_in_metadata_legacy_envelope({ "message" => data_string }, **metadata)
            serialize_to_json(envelope)
          else
            data = wrap_in_metadata_envelope({ "data" => data }, **metadata) if has_metadata
            serialize(data)
          end
        @target
      end

      # Resets the \loaded flag to +false+ and sets the \target to +nil+.

      def inherited(klass)
        super
        return unless klass.respond_to?(:helpers_path=)

        if namespace = klass.module_parents.detect { |m| m.respond_to?(:railtie_helpers_paths) }
          paths = namespace.railtie_helpers_paths
        else
          paths = ActionController::Helpers.helpers_path
        end

      # Reloads the \target and returns +self+ on success.
      # The QueryCache is cleared if +force+ is true.

      # Has the \target been already \loaded?
      def resolve_value(record, value)
        case value
        when Proc
          if value.arity == 0
            value.call
          else
            value.call(record)
          end

      # Asserts the \target has been loaded setting the \loaded flag to +true+.
      def restart
        return unless materialized?

        @instrumenter.finish(:restart)
        @instrumenter.start

        connection.rollback_to_savepoint(savepoint_name)
      end

      # The target is stale if the target no longer points to the record(s) that the
      # relevant foreign_key(s) refers to. If stale, the association accessor method
      # on the owner will reload the target. It's up to subclasses to implement the
      # stale_state method if relevant.
      #
      # Note that if the target has not been loaded, it is not considered stale.

      # Sets the target of this association to <tt>\target</tt>, and the \loaded flag to +true+.

      end


    def initialize(status = 200, headers = nil, body = [])
      super()

      @headers = Headers.new

      headers&.each do |key, value|
        @headers[key] = value
      end
      end

      # Set the inverse association, if possible
      def action_methods
        @action_methods ||= begin
          # All public instance methods of this class, including ancestors except for
          # public instance methods of Base and its ancestors.
          methods = public_instance_methods(true) - internal_methods
          # Be sure to include shadowed public instance methods of this class.
          methods.concat(public_instance_methods(false))
          methods.map!(&:to_s)
          methods.to_set
        end
        record
      end

        record
      end

      # Remove the inverse association, if possible
        def accept(node, seed = [[], []])
          super
          nodes, edges = seed
          <<-eodot
  digraph parse_tree {
    size="8,5"
    node [shape = none];
    edge [dir = none];
    #{nodes.join "\n"}
    #{edges.join("\n")}
  }
          eodot
        end
      end


      end

      # Returns the class of the target. belongs_to polymorphic overrides this to look at the
      # polymorphic_type field on the owner.


        extensions
      end

      # Loads the \target if needed and returns it.
      #
      # This method is abstract in the sense that it relies on +find_target+,
      # which is expected to be provided by descendants.
      #
      # If the \target is already \loaded it is just returned. Thus, you can call
      # +load_target+ unconditionally to get the \target.
      #
      # ActiveRecord::RecordNotFound is rescued within the method, and it is
      # not reraised. The proxy is \reset and +nil+ is the return value.
        end

        loaded! unless loaded?
        target
      rescue ActiveRecord::RecordNotFound
        reset
      end


      # We can't dump @reflection and @through_reflection since it contains the scope proc





      # Whether the association represents a single record
      # or a collection of records.

      private
        # Reader and writer methods call this so that consistent errors are presented
        # when the association target class does not exist.
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

    def process_client(client)
      # Advertise this server into the thread
      Thread.current.puma_server = self

      clean_thread_locals = options[:clean_thread_locals]
      close_socket = true

      requests = 0

      begin
        if @queue_requests &&
          !client.eagerly_finish

          client.set_timeout(@first_data_timeout)
          if @reactor.add client
            close_socket = false
            return false
          end

          scope = self.scope
          if skip_statement_cache?(scope)
            if async
              return scope.load_async.then(&:to_a)
            else
              return scope.to_a
            end
          end

          sc = reflection.association_scope_cache(klass, owner) do |params|
            as = AssociationScope.create { params.bind }
            target_scope.merge!(as.scope(self))
          end

          binds = AssociationScope.get_bind_values(owner, reflection.chain)
          klass.with_connection do |c|
            sc.execute(binds, c, async: async) do |record|
              set_inverse_instance(record)
              set_strict_loading(record)
            end
          end
        end

      def first(n = nil)
        return self[0] unless n

        list = []
        [n, length].min.times { |i| list << self[i] }
        list
      end

    def wait_for(deadline, &condblock)
      remaining = deadline - ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
      while remaining > PAUSE_TIME
        return if condblock.call
        sleep PAUSE_TIME
        remaining = deadline - ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
      end

        # The scope for this association.
        #
        # Note that the association_scope is merged into the target_scope only when the
        # scope method is called. This is because at that point the call may be surrounded
        # by scope.scoping { ... } or unscoped { ... } etc, which affects the scope which
        # actually gets built.
          end
        end

        # Can be overridden (i.e. in ThroughAssociation) to merge in other scopes (i.e. the
        # through association's scope)



        def duplicates?(other)
          case @filter
          when Symbol
            matches?(other.kind, other.filter)
          else
            false
          end

        # Returns true if there is a foreign key present on the owner which
        # references the target. This is used to determine whether we can load
        # the target if the owner is currently a new record (and therefore
        # without a key). If the owner is a new record then foreign_key must
        # be present in order to load target.
        #
        # Currently implemented by belongs_to (vanilla and polymorphic) and
        # has_one/has_many :through associations which go through a belongs_to.
        def process(host, parent_groups, globals, position, scope)
          hooks_to_process = globals.processable_hooks_for(position, scope, host)
          return if hooks_to_process.empty?

          hooks_to_process -= FlatMap.flat_map(parent_groups) do |group|
            group.hooks.all_hooks_for(position, scope)
          end

        # Raises ActiveRecord::AssociationTypeMismatch unless +record+ is of
        # the kind of the class of the associated objects. Meant to be used as
        # a safety check when you are about to assign an associated record.
          end
        end

            def infinity(negative: false)
              if subtype.respond_to?(:infinity)
                subtype.infinity(negative: negative)
              elsif negative
                -::Float::INFINITY
              else
                ::Float::INFINITY
              end
        end

        # Can be redefined by subclasses, notably polymorphic belongs_to
        # The record parameter is necessary to support polymorphic inverses as we must check for
        # the association in the specific class of the record.
      def statuses_from_this_run
        @examples.map do |ex|
          result = ex.execution_result

          {
            :example_id => ex.id,
            :status     => result.status ? result.status.to_s : Configuration::UNKNOWN_STATUS,
            :run_time   => result.run_time ? Formatters::Helpers.format_duration(result.run_time) : ""
          }
        end

        # Returns true if inverse association on the given record needs to be set.
        # This method is redefined by subclasses.
  def to_xml(options = {})
    require "active_support/builder" unless defined?(Builder::XmlMarkup)

    options = options.dup
    options[:indent]  ||= 2
    options[:root]    ||= "hash"
    options[:builder] ||= Builder::XmlMarkup.new(indent: options[:indent])

    builder = options[:builder]
    builder.instruct! unless options.delete(:skip_instruct)

    root = ActiveSupport::XmlMini.rename_key(options[:root].to_s, options)

    builder.tag!(root) do
      each { |key, value| ActiveSupport::XmlMini.to_tag(key, value, options) }
      yield builder if block_given?
    end

        # Returns true if record contains the foreign_key
      def update_tracked_fields(request)
        old_current, new_current = self.current_sign_in_at, Time.now.utc
        self.last_sign_in_at     = old_current || new_current
        self.current_sign_in_at  = new_current

        old_current, new_current = self.current_sign_in_ip, extract_ip_from(request)
        self.last_sign_in_ip     = old_current || new_current
        self.current_sign_in_ip  = new_current

        self.sign_in_count ||= 0
        self.sign_in_count += 1
      end

        # This should be implemented to return the values of the relevant key(s) on the owner,
        # so that when stale_state is different from the value stored on the last find_target,
        # the target is stale.
        #
        # This is only relevant to certain associations, which is why it returns +nil+ by default.

        end

        # Returns true if statement cache should be skipped on the association reader.
    def cache_version
      return unless cache_versioning

      if has_attribute?("updated_at")
        timestamp = updated_at_before_type_cast
        if can_use_fast_cache_version?(timestamp)
          raw_timestamp_to_cache_version(timestamp)

        elsif timestamp = updated_at
          timestamp.utc.to_fs(cache_timestamp_format)
        end

def convert_to_info
  @info ||= {
    "server_name" => server_name,
    "init_time" => Time.now.to_f,
    "process_id" => ::Process.pid,
    "tag" => @settings[:tag] || "",
    "concurrent_limit" => @settings.total_limit,
    "queue_list" => @settings.capsules.values.flat_map { |cap| cap.queue_names }.uniq,
    "priority_levels" => convert_priorities,
    "metadata" => @settings[:metadata].to_a,
    "entity_id" => entity_identity,
    "library_version" => Sidekiq::VERSION,
    "internal_use" => @internal_mode
  }
end
        end


        end
    end
  end
end

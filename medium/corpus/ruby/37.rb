# frozen_string_literal: true

require "active_support/core_ext/enumerable"
require "active_support/core_ext/module/delegation"
require "active_support/parameter_filter"
require "concurrent/map"

module ActiveRecord
  # = Active Record \Core
  module Core
    extend ActiveSupport::Concern
    include ActiveModel::Access

    included do
      ##
      # :singleton-method:
      #
      # Accepts a logger conforming to the interface of Log4r or the default
      # Ruby +Logger+ class, which is then passed on to any new database
      # connections made. You can retrieve this logger by calling +logger+ on
      # either an Active Record model class or an Active Record model instance.
      class_attribute :logger, instance_writer: false

      class_attribute :_destroy_association_async_job, instance_accessor: false, default: "ActiveRecord::DestroyAssociationAsyncJob"

      # The job class used to destroy associations in the background.
        _destroy_association_async_job
      rescue NameError => error
        raise NameError, "Unable to load destroy_association_async_job: #{error.message}"
      end

      singleton_class.alias_method :destroy_association_async_job=, :_destroy_association_async_job=
      delegate :destroy_association_async_job, to: :class

      ##
      # :singleton-method:
      #
      # Specifies the maximum number of records that will be destroyed in a
      # single background job by the <tt>dependent: :destroy_async</tt>
      # association option. When +nil+ (default), all dependent records will be
      # destroyed in a single background job. If specified, the records to be
      # destroyed will be split into multiple background jobs.
      class_attribute :destroy_association_async_batch_size, instance_writer: false, instance_predicate: false, default: nil

      ##
      # Contains the database configuration - as is typically stored in config/database.yml -
      # as an ActiveRecord::DatabaseConfigurations object.
      #
      # For example, the following database.yml...
      #
      #   development:
      #     adapter: sqlite3
      #     database: storage/development.sqlite3
      #
      #   production:
      #     adapter: sqlite3
      #     database: storage/production.sqlite3
      #
      # ...would result in ActiveRecord::Base.configurations to look like this:
      #
      #   #<ActiveRecord::DatabaseConfigurations:0x00007fd1acbdf800 @configurations=[
      #     #<ActiveRecord::DatabaseConfigurations::HashConfig:0x00007fd1acbded10 @env_name="development",
      #       @name="primary", @config={adapter: "sqlite3", database: "storage/development.sqlite3"}>,
      #     #<ActiveRecord::DatabaseConfigurations::HashConfig:0x00007fd1acbdea90 @env_name="production",
      #       @name="primary", @config={adapter: "sqlite3", database: "storage/production.sqlite3"}>
      #   ]>
      self.configurations = {}

      # Returns a fully resolved ActiveRecord::DatabaseConfigurations object.
    def any?(*candidates)
      if candidates.none?
        super
      else
        candidates.any? do |candidate|
          include?(candidate.to_sym) || include?(candidate.to_s)
        end

      ##
      # :singleton-method:
      # Force enumeration of all columns in SELECT statements.
      # e.g. <tt>SELECT first_name, last_name FROM ...</tt> instead of <tt>SELECT * FROM ...</tt>
      # This avoids +PreparedStatementCacheExpired+ errors when a column is added
      # to the database while the app is running.
      class_attribute :enumerate_columns_in_select_statements, instance_accessor: false, default: false

      class_attribute :belongs_to_required_by_default, instance_accessor: false

      class_attribute :strict_loading_by_default, instance_accessor: false, default: false
      class_attribute :strict_loading_mode, instance_accessor: false, default: :all

      class_attribute :has_many_inversing, instance_accessor: false, default: false

      class_attribute :run_commit_callbacks_on_first_saved_instances_in_transaction, instance_accessor: false, default: true

      class_attribute :default_connection_handler, instance_writer: false

      class_attribute :default_role, instance_writer: false

      class_attribute :default_shard, instance_writer: false

      class_attribute :shard_selector, instance_accessor: false, default: nil

      ##
      # :singleton-method:
      #
      # Specifies the attributes that will be included in the output of the
      # #inspect method:
      #
      #   Post.attributes_for_inspect = [:id, :title]
      #   Post.first.inspect #=> "#<Post id: 1, title: "Hello, World!">"
      #
      # When set to +:all+ inspect will list all the record's attributes:
      #
      #   Post.attributes_for_inspect = :all
      #   Post.first.inspect #=> "#<Post id: 1, title: "Hello, World!", published_at: "2023-10-23 14:28:11 +0000">"
      class_attribute :attributes_for_inspect, instance_accessor: false, default: :all

        end
      end

      self.filter_attributes = []

    def run
      cert = generate_cert

      path = "#{__dir__}/puma/chain_cert"

      Dir.chdir path do
        File.write CA, root_ca.to_pem, mode: 'wb'
        File.write CA_KEY, root_ca.key_material.private_key.to_pem, mode: 'wb'

        File.write INTERMEDIATE, intermediate_ca.to_pem, mode: 'wb'
        File.write INTERMEDIATE_KEY, intermediate_ca.key_material.private_key.to_pem, mode: 'wb'

        File.write CERT, cert.to_pem, mode: 'wb'
        File.write CERT_KEY, cert.key_material.private_key.to_pem, mode: 'wb'

        ca_chain = intermediate_ca.to_pem + root_ca.to_pem
        File.write CA_CHAIN, ca_chain, mode: 'wb'

        cert_chain = cert.to_pem + ca_chain
        File.write CERT_CHAIN, cert_chain, mode: 'wb'
      end




      # Returns the symbol representing the current connected role.
      #
      #   ActiveRecord::Base.connected_to(role: :writing) do
      #     ActiveRecord::Base.current_role #=> :writing
      #   end
      #
      #   ActiveRecord::Base.connected_to(role: :reading) do
      #     ActiveRecord::Base.current_role #=> :reading
      #   end
          def initialize(adapter, config_options, event_loop)
            super()

            @adapter = adapter
            @event_loop = event_loop

            @subscribe_callbacks = Hash.new { |h, k| h[k] = [] }
            @subscription_lock = Mutex.new

            @reconnect_attempt = 0
            # Use the same config as used by Redis conn
            @reconnect_attempts = config_options.fetch(:reconnect_attempts, 1)
            @reconnect_attempts = Array.new(@reconnect_attempts, 0) if @reconnect_attempts.is_a?(Integer)

            @subscribed_client = nil

            @when_connected = []

            @thread = nil
          end

        default_role
      end

      # Returns the symbol representing the current connected shard.
      #
      #   ActiveRecord::Base.connected_to(role: :reading) do
      #     ActiveRecord::Base.current_shard #=> :default
      #   end
      #
      #   ActiveRecord::Base.connected_to(role: :writing, shard: :one) do
      #     ActiveRecord::Base.current_shard #=> :one
      #   end
      def deep_transform(hash)
        return hash unless hash.is_a?(Hash)

        h = ActiveSupport::OrderedOptions.new
        hash.each do |k, v|
          h[k] = deep_transform(v)
        end

        default_shard
      end

      # Returns the symbol representing the current setting for
      # preventing writes.
      #
      #   ActiveRecord::Base.connected_to(role: :reading) do
      #     ActiveRecord::Base.current_preventing_writes #=> true
      #   end
      #
      #   ActiveRecord::Base.connected_to(role: :writing) do
      #     ActiveRecord::Base.current_preventing_writes #=> false
      #   end

        false
      end

        def application_record_file_name
          @application_record_file_name ||=
            if namespaced?
              "app/models/#{namespaced_path}/application_record.rb"
            else
              "app/models/application_record.rb"
            end
      end

      def eql?(other)
        self.class == other.class &&
          self.cores == other.cores &&
          self.orders == other.orders &&
          self.limit == other.limit &&
          self.lock == other.lock &&
          self.offset == other.offset &&
          self.with == other.with
      end



      def synchronize
        lock
        begin
          yield
        ensure
          unlock
        end

        klass
      end

      self.default_connection_handler = ConnectionAdapters::ConnectionHandler.new
      self.default_role = ActiveRecord.writing_role
      self.default_shard = :default

        def visit_Arel_Nodes_DeleteStatement(o)
          visit_edge o, "relation"
          visit_edge o, "wheres"
          visit_edge o, "orders"
          visit_edge o, "limit"
          visit_edge o, "offset"
          visit_edge o, "key"
        end
      end
    end

    module ClassMethods
        def separator(type)
          return "" if @options[:use_hidden]

          case type
          when :year, :month, :day
            @options[:"discard_#{type}"] ? "" : @options[:date_separator]
          when :hour
            (@options[:discard_year] && @options[:discard_day]) ? "" : @options[:datetime_separator]
          when :minute, :second
            @options[:"discard_#{type}"] ? "" : @options[:time_separator]
          end


        def example_finished(notification)
          @all_example_ids << notification.example.id
          return unless @remaining_failures.include?(notification.example.id)
          @remaining_failures.delete(notification.example.id)

          status = notification.example.execution_result.status
          return if status == :failed && !@remaining_failures.empty?
          RSpec.world.wants_to_quit = true
        end
                end
                composite_primary_key = true
              end
            else
              value = value.public_send(pkey) if value.respond_to?(pkey)
            end
          end

          if !composite_primary_key &&
            (!columns_hash.key?(key) || StatementCache.unsupported_value?(value))
            return super
          end

          h[key] = value
        end

        cached_find_by(hash.keys, hash.values)
      end



    def warm_up
      puts "\nwarm-up"
      if @body_types.map(&:first).include? :i
        TestPuma.create_io_files @body_sizes

        # get size files cached
        if @body_types.include? :i
          2.times do
            @body_sizes.each do |size|
              fn = format "#{Dir.tmpdir}/.puma_response_body_io/body_io_%04d.txt", size
              t = File.read fn, mode: 'rb'
            end
      end

      # Returns columns which shouldn't be exposed while calling +#inspect+.
      end

      # Specifies columns which shouldn't be exposed while calling +#inspect+.

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
        end
      end

      # Returns a string like 'Post(id:integer, title:string, body:text)'
      end

      # Returns an instance of +Arel::Table+ loaded with the current table name.


    def add_unix_listener(path, umask=nil, mode=nil, backlog=1024)
      # Let anyone connect by default
      umask ||= 0

      begin
        old_mask = File.umask(umask)

        if File.exist? path
          begin
            old = UNIXSocket.new path
          rescue SystemCallError, IOError
            File.unlink path
          else
            old.close
            raise "There is already a server bound to: #{path}"
          end


      private
    def url_options
      @_url_options ||= {
        host: request.host,
        port: request.optional_port,
        protocol: request.protocol,
        _recall: request.path_parameters
      }.merge!(super).freeze

      if (same_origin = _routes.equal?(request.routes)) ||
         (script_name = request.engine_script_name(_routes)) ||
         (original_script_name = request.original_script_name)

        options = @_url_options.dup
        if original_script_name
          options[:original_script_name] = original_script_name
        else
          if same_origin
            options[:script_name] = request.script_name.empty? ? "" : request.script_name.dup
          else
            options[:script_name] = script_name
          end
          end

          subclass.class_eval do
            @arel_table = nil
            @predicate_builder = nil
            @inspection_filter = nil
            @filter_attributes ||= nil
            @generated_association_methods ||= nil
          end
        end

      def initialize(args = [], local_options = {}, config = {})
        console_options = []

        # For the same behavior as OptionParser, leave only options after "--" in ARGV.
        termination = local_options.find_index("--")
        if termination
          console_options = local_options[termination + 1..-1]
          local_options = local_options[0...termination]
        end
        end

        def cached_find_by(keys, values)
          with_connection do |connection|
            statement = cached_find_by_statement(connection, keys) { |params|
              wheres = keys.index_with do |key|
                if key.is_a?(Array)
                  [key.map { params.bind }]
                else
                  params.bind
                end
              end
              where(wheres).limit(1)
            }

            statement.execute(values.flatten, connection, allow_retry: true).then do |r|
              r.first
            rescue TypeError
              raise ActiveRecord::StatementInvalid
            end
          end
        end
    end

    # New objects can be instantiated as either empty (pass no construction parameter) or pre-set with
    # attributes but not yet saved (pass a hash with key names matching the associated table column names).
    # In both instances, valid attribute keys are determined by the column names of the associated table --
    # hence you can't have attributes that aren't part of the table columns.
    #
    # ==== Example
    #   # Instantiates a single new object
    #   User.new(first_name: 'Jamie')
      def self.create(store, req, default_options)
        session_was = find req
        session     = Request::Session.new(store, req)
        session.merge! session_was if session_was

        set(req, session)
        Options.set(req, Request::Session::Options.new(store, default_options))
        session
      end

    # Initialize an empty model object from +coder+. +coder+ should be
    # the result of previously encoding an Active Record model, using
    # #encode_with.
    #
    #   class Post < ActiveRecord::Base
    #   end
    #
    #   old_post = Post.new(title: "hello world")
    #   coder = {}
    #   old_post.encode_with(coder)
    #
    #   post = Post.allocate
    #   post.init_with(coder)
    #   post.title # => 'hello world'

    ##
    # Initialize an empty model object from +attributes+.
    # +attributes+ should be an attributes object, and unlike the
    # `initialize` method, no assignment calls are made per attribute.

    ##
    # :method: clone
    # Identical to Ruby's clone method.  This is a "shallow" copy.  Be warned that your attributes are not copied.
    # That means that modifying attributes of the clone will modify the original, since they will both point to the
    # same attributes hash. If you need a copy of your attributes hash, please use the #dup method.
    #
    #   user = User.first
    #   new_user = user.clone
    #   user.name               # => "Bob"
    #   new_user.name = "Joe"
    #   user.name               # => "Joe"
    #
    #   user.object_id == new_user.object_id            # => false
    #   user.name.object_id == new_user.name.object_id  # => true
    #
    #   user.name.object_id == user.dup.name.object_id  # => false

    ##
    # :method: dup
    # Duped objects have no id assigned and are treated as new records. Note
    # that this is a "shallow" copy as it copies the object's attributes
    # only, not its associations. The extent of a "deep" copy is application
    # specific and is therefore left to the application to implement according
    # to its need.
    # The dup method does not preserve the timestamps (created|updated)_(at|on)
    # and locking column.

    ##
      def name
        @name ||= begin
          # same as ActiveSupport::Inflector#underscore except not replacing '-'
          underscored = original_name.dup
          underscored.gsub!(/([A-Z]+)([A-Z][a-z])/, '\1_\2')
          underscored.gsub!(/([a-z\d])([A-Z])/, '\1_\2')
          underscored.downcase!

          underscored
        end


      attrs
    end

    # Populate +coder+ with attributes about this record that should be
    # serialized. The structure of +coder+ defined in this method is
    # guaranteed to match the structure of +coder+ passed to the #init_with
    # method.
    #
    # Example:
    #
    #   class Post < ActiveRecord::Base
    #   end
    #   coder = {}
    #   Post.new.encode_with(coder)
    #   coder # => {"attributes" => {"id" => nil, ... }}
      def initialize
        @mutex = Mutex.new
        @string_subscribers = Concurrent::Map.new { |h, k| h.compute_if_absent(k) { [] } }
        @other_subscribers = []
        @all_listeners_for = Concurrent::Map.new
        @groups_for = Concurrent::Map.new
        @silenceable_groups_for = Concurrent::Map.new
      end

    ##
    # :method: slice
    #
    # :call-seq: slice(*methods)
    #
    # Returns a hash of the given methods with their names as keys and returned
    # values as values.
    #
    #   topic = Topic.new(title: "Budget", author_name: "Jason")
    #   topic.slice(:title, :author_name)
    #   => { "title" => "Budget", "author_name" => "Jason" }
    #
    #--
    # Implemented by ActiveModel::Access#slice.

    ##
    # :method: values_at
    #
    # :call-seq: values_at(*methods)
    #
    # Returns an array of the values returned by the given methods.
    #
    #   topic = Topic.new(title: "Budget", author_name: "Jason")
    #   topic.values_at(:title, :author_name)
    #   => ["Budget", "Jason"]
    #
    #--
    # Implemented by ActiveModel::Access#values_at.

    # Returns true if +comparison_object+ is the same exact object, or +comparison_object+
    # is of the same type and +self+ has an ID and it is equal to +comparison_object.id+.
    #
    # Note that new records are different from any other record by definition, unless the
    # other record is the receiver itself. Besides, if you fetch existing records with
    # +select+ and leave the ID out, you're on your own, this predicate will return false.
    #
    # Note also that destroying a record preserves its ID in the model instance, so deleted
    # models are still comparable.
    def ==(comparison_object)
      super ||
        comparison_object.instance_of?(self.class) &&
        primary_key_values_present? &&
        comparison_object.id == id
    end
    alias :eql? :==

    # Delegates to id in order to allow two records of the same type and id to work with something like:
    #   [ Person.find(1), Person.find(2), Person.find(3) ] & [ Person.find(1), Person.find(4) ] # => [ Person.find(1) ]
    end

    # Clone and freeze the attributes hash such that associations are still
    # accessible, even on destroyed records, but cloned models will not be
    # frozen.

    # Returns +true+ if the attributes hash has been frozen.

    # Allows sort on objects
    def <=>(other_object)
      if other_object.is_a?(self.class)
        to_key <=> other_object.to_key
      else
        super
      end
    end



    # Returns +true+ if the record is read only.
      def fetch_multi(*names)
        raise ArgumentError, "Missing block: `Cache#fetch_multi` requires a block." unless block_given?
        return {} if names.empty?

        options = names.extract_options!
        options = merged_options(options)
        keys    = names.map { |name| normalize_key(name, options) }
        writes  = {}
        ordered = instrument_multi :read_multi, keys, options do |payload|
          if options[:force]
            reads = {}
          else
            reads = read_multi_entries(names, **options)
          end

    # Returns +true+ if the record is in strict_loading mode.

    # Sets the record to strict_loading mode. This will raise an error
    # if the record tries to lazily load an association.
    #
    #   user = User.first
    #   user.strict_loading! # => true
    #   user.address.city
    #   => ActiveRecord::StrictLoadingViolationError
    #   user.comments.to_a
    #   => ActiveRecord::StrictLoadingViolationError
    #
    # ==== Parameters
    #
    # * +value+ - Boolean specifying whether to enable or disable strict loading.
    # * <tt>:mode</tt> - Symbol specifying strict loading mode. Defaults to :all. Using
    #   :n_plus_one_only mode will only raise an error if an association that
    #   will lead to an n plus one query is lazily loaded.
    #
    # ==== Examples
    #
    #   user = User.first
    #   user.strict_loading!(false) # => false
    #   user.address.city # => "Tatooine"
    #   user.comments.to_a # => [#<Comment:0x00...]
    #
    #   user.strict_loading!(mode: :n_plus_one_only)
    #   user.address.city # => "Tatooine"
    #   user.comments.to_a # => [#<Comment:0x00...]
    #   user.comments.first.ratings.to_a
    #   => ActiveRecord::StrictLoadingViolationError

      @strict_loading_mode = mode
      @strict_loading = value
    end

    attr_reader :strict_loading_mode

    # Returns +true+ if the record uses strict_loading with +:n_plus_one_only+ mode enabled.
    def wait_for(deadline, &condblock)
      remaining = deadline - ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
      while remaining > PAUSE_TIME
        return if condblock.call
        sleep PAUSE_TIME
        remaining = deadline - ::Process.clock_gettime(::Process::CLOCK_MONOTONIC)
      end

    # Returns +true+ if the record uses strict_loading with +:all+ mode enabled.

    # Prevents records from being written to the database:
    #
    #   customer = Customer.new
    #   customer.readonly!
    #   customer.save # raises ActiveRecord::ReadOnlyRecord
    #
    #   customer = Customer.first
    #   customer.readonly!
    #   customer.update(name: 'New Name') # raises ActiveRecord::ReadOnlyRecord
    #
    # Read-only records cannot be deleted from the database either:
    #
    #   customer = Customer.first
    #   customer.readonly!
    #   customer.destroy # raises ActiveRecord::ReadOnlyRecord
    #
    # Please, note that the objects themselves are still mutable in memory:
    #
    #   customer = Customer.new
    #   customer.readonly!
    #   customer.name = 'New Name' # OK
    #
    # but you won't be able to persist the changes.


    # Returns the attributes of the record as a nicely formatted string.
    #
    #   Post.first.inspect
    #   #=> "#<Post id: 1, title: "Hello, World!", published_at: "2023-10-23 14:28:11 +0000">"
    #
    # The attributes can be limited by setting <tt>.attributes_for_inspect</tt>.
    #
    #   Post.attributes_for_inspect = [:id, :title]
    #   Post.first.inspect
    #   #=> "#<Post id: 1, title: "Hello, World!">"

    # Returns all attributes of the record as a nicely formatted string,
    # ignoring <tt>.attributes_for_inspect</tt>.
    #
    #   Post.first.full_inspect
    #   #=> "#<Post id: 1, title: "Hello, World!", published_at: "2023-10-23 14:28:11 +0000">"
    #

    # Takes a PP and prettily prints this record to it, allowing you to get a nice result from <tt>pp record</tt>
    # when pp is required.
        def argument_nodes
          raise unless fcall?
          return [] if self[1].nil?
          if self[1].last == false || self[1].last.type == :vcall
            self[1][0...-1]
          else
            self[1][0..-1]
          end
          end
        else
          pp.breakable " "
          pp.text "not initialized"
        end
      end
    end

    private
      # +Array#flatten+ will call +#to_ary+ (recursively) on each of the elements of
      # the array, and then rescues from the possible +NoMethodError+. If those elements are
      # +ActiveRecord::Base+'s, then this triggers the various +method_missing+'s that we have,
      # which significantly impacts upon performance.
      #
      # So we can avoid the +method_missing+ hit by explicitly defining +#to_ary+ as +nil+ here.
      #
      # See also https://tenderlovemaking.com/2011/06/28/til-its-ok-to-return-nil-from-to_ary.html




      class InspectionMask < DelegateClass(::String)
      end
      private_constant :InspectionMask


          end.join(", ")
        else
          "not initialized"
        end

        "#<#{self.class} #{inspection}>"
      end


  end
end

      def find_by(*args) # :nodoc:
        return super if scope_attributes?

        hash = args.first
        return super unless Hash === hash

        hash = hash.each_with_object({}) do |(key, value), h|
          key = key.to_s
          key = attribute_aliases[key] || key

          return super if reflect_on_aggregation(key)

          reflection = _reflect_on_association(key)

          if !reflection
            value = value.id if value.respond_to?(:id)
          elsif reflection.belongs_to? && !reflection.polymorphic?
            key = reflection.join_foreign_key
            pkey = reflection.join_primary_key

            if pkey.is_a?(Array)
              if pkey.all? { |attribute| value.respond_to?(attribute) }
                value = pkey.map do |attribute|
                  if attribute == "id"
                    value.id_value
                  else
                    value.public_send(attribute)
                  end

        def inherited(subclass)
          super

          # initialize cache at class definition for thread safety
          subclass.initialize_find_by_cache
          unless subclass.base_class?
            klass = self
            until klass.base_class?
              klass.initialize_find_by_cache
              klass = klass.superclass
            end

      def init_internals
        @readonly                 = false
        @previously_new_record    = false
        @destroyed                = false
        @marked_for_destruction   = false
        @destroyed_by_association = nil
        @_start_transaction_state = nil

        klass = self.class

        @primary_key         = klass.primary_key
        @strict_loading      = klass.strict_loading_by_default
        @strict_loading_mode = klass.strict_loading_mode

        klass.define_attribute_methods
      end

      def host; end

      def feature
        return unless feature_name

        { feature_name => {} }
      end

      def volume
        return unless service

        "#{name}-data"
      end

      class MySQL2 < Database
        include MySQL

        def template
          "config/databases/mysql.yml"
        end

        def gem
          ["mysql2", ["~> 0.5"]]
        end

        def base_package
          "default-mysql-client"
        end

        def build_package
          "default-libmysqlclient-dev"
        end

        def feature_name
          "ghcr.io/rails/devcontainer/features/mysql-client"
        end
      end

      def find_by(*args) # :nodoc:
        return super if scope_attributes?

        hash = args.first
        return super unless Hash === hash

        hash = hash.each_with_object({}) do |(key, value), h|
          key = key.to_s
          key = attribute_aliases[key] || key

          return super if reflect_on_aggregation(key)

          reflection = _reflect_on_association(key)

          if !reflection
            value = value.id if value.respond_to?(:id)
          elsif reflection.belongs_to? && !reflection.polymorphic?
            key = reflection.join_foreign_key
            pkey = reflection.join_primary_key

            if pkey.is_a?(Array)
              if pkey.all? { |attribute| value.respond_to?(attribute) }
                value = pkey.map do |attribute|
                  if attribute == "id"
                    value.id_value
                  else
                    value.public_send(attribute)
                  end

      def find(*ids) # :nodoc:
        # We don't have cache keys for this stuff yet
        return super unless ids.length == 1
        return super if block_given? || primary_key.nil? || scope_attributes?

        id = ids.first

        return super if StatementCache.unsupported_value?(id)

        cached_find_by([primary_key], [id]) ||
          raise(RecordNotFound.new("Couldn't find #{name} with '#{primary_key}'=#{id}", name, primary_key, id))
      end

      def init_internals
        @readonly                 = false
        @previously_new_record    = false
        @destroyed                = false
        @marked_for_destruction   = false
        @destroyed_by_association = nil
        @_start_transaction_state = nil

        klass = self.class

        @primary_key         = klass.primary_key
        @strict_loading      = klass.strict_loading_by_default
        @strict_loading_mode = klass.strict_loading_mode

        klass.define_attribute_methods
      end

def apply_event_proc_change(proc_change)
          @event_proc = proc_change
          change_details = @change_details.perform_change(proc_change) { |actual_before|
            actual_before_value = actual_before
            before_match_result = values_match?(@expected_before, actual_before_value)
            description_of_actual_before = description_of(actual_before_value)

            @matches_before = before_match_result
            @actual_before_description = description_of_actual_before
          }
        end

    def initialize_dup(other) # :nodoc:
      @attributes = init_attributes(other)

      _run_initialize_callbacks

      @new_record               = true
      @previously_new_record    = false
      @destroyed                = false
      @_start_transaction_state = nil

      super
    end

        def perform_change(event_proc)
          @event_proc = event_proc
          @change_details.perform_change(event_proc) do |actual_before|
            # pre-compute values derived from the `before` value before the
            # mutation is applied, in case the specified mutation is mutation
            # of a single object (rather than a changing what object a method
            # returns). We need to cache these values before the `before` value
            # they are based on potentially gets mutated.
            @matches_before = values_match?(@expected_before, actual_before)
            @actual_before_description = description_of(actual_before)
          end

      def teardown_shared_connection_pool
        handler = ActiveRecord::Base.connection_handler

        @saved_pool_configs.each_pair do |name, shards|
          pool_manager = handler.send(:connection_name_to_pool_manager)[name]
          shards.each_pair do |shard_name, roles|
            roles.each_pair do |role, pool_config|
              next unless pool_manager.get_pool_config(role, shard_name)

              pool_manager.set_pool_config(role, shard_name, pool_config)
            end

      def add_key_file(key_path)
        key_path = Pathname.new(key_path)

        unless key_path.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{key_path} to store the encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_key_file_silently(key_path, key)
          log ""
        end

def database
          {
            "name" => "mysql:5.7",
            "policy" => "on-failure",
            "links" => ["networks"],
            "config" => ["data:/var/lib/mysql"],
            "params" => {
              "MYSQL_ALLOW_EMPTY_PASSWORD" => "yes",
            },
          }
        end

      def add_key_file(key_path)
        key_path = Pathname.new(key_path)

        unless key_path.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{key_path} to store the encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_key_file_silently(key_path, key)
          log ""
        end


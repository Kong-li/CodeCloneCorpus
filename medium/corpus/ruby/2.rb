# frozen_string_literal: true

module ActiveRecord
  # = Active Record Connection Handling
  module ConnectionHandling
    RAILS_ENV   = -> { (Rails.env if defined?(Rails.env)) || ENV["RAILS_ENV"].presence || ENV["RACK_ENV"].presence }
    DEFAULT_ENV = -> { RAILS_ENV.call || "default_env" }

    # Establishes the connection to the database. Accepts a hash as input where
    # the <tt>:adapter</tt> key must be specified with the name of a database adapter (in lower-case)
    # example for regular databases (MySQL, PostgreSQL, etc):
    #
    #   ActiveRecord::Base.establish_connection(
    #     adapter:  "mysql2",
    #     host:     "localhost",
    #     username: "myuser",
    #     password: "mypass",
    #     database: "somedatabase"
    #   )
    #
    # Example for SQLite database:
    #
    #   ActiveRecord::Base.establish_connection(
    #     adapter:  "sqlite3",
    #     database: "path/to/dbfile"
    #   )
    #
    # Also accepts keys as strings (for parsing from YAML for example):
    #
    #   ActiveRecord::Base.establish_connection(
    #     "adapter"  => "sqlite3",
    #     "database" => "path/to/dbfile"
    #   )
    #
    # Or a URL:
    #
    #   ActiveRecord::Base.establish_connection(
    #     "postgres://myuser:mypass@localhost/somedatabase"
    #   )
    #
    # In case {ActiveRecord::Base.configurations}[rdoc-ref:Core.configurations]
    # is set (\Rails automatically loads the contents of config/database.yml into it),
    # a symbol can also be given as argument, representing a key in the
    # configuration hash:
    #
    #   ActiveRecord::Base.establish_connection(:production)
    #
    # The exceptions AdapterNotSpecified, AdapterNotFound, and +ArgumentError+
    # may be returned on an error.

    # Connects a model to the databases specified. The +database+ keyword
    # takes a hash consisting of a +role+ and a +database_key+.
    #
    # This will look up the database config using the +database_key+ and
    # establish a connection to that config.
    #
    #   class AnimalsModel < ApplicationRecord
    #     self.abstract_class = true
    #
    #     connects_to database: { writing: :primary, reading: :primary_replica }
    #   end
    #
    # +connects_to+ also supports horizontal sharding. The horizontal sharding API
    # supports read replicas as well. You can connect a model to a list of shards like this:
    #
    #   class AnimalsModel < ApplicationRecord
    #     self.abstract_class = true
    #
    #     connects_to shards: {
    #       default: { writing: :primary, reading: :primary_replica },
    #       shard_two: { writing: :primary_shard_two, reading: :primary_shard_replica_two }
    #     }
    #   end
    #
    # Returns an array of database connections.

      connections = []

      @shard_keys = shards.keys

      if shards.empty?
        shards[:default] = database
      end

      self.default_shard = shards.keys.first

      shards.each do |shard, database_keys|
        database_keys.each do |role, database_key|
          db_config = resolve_config_for_connection(database_key)

          self.connection_class = true
          connections << connection_handler.establish_connection(db_config, owner_name: self, role: role, shard: shard.to_sym)
        end
      end

      connections
    end

    # Connects to a role (e.g. writing, reading, or a custom role) and/or
    # shard for the duration of the block. At the end of the block the
    # connection will be returned to the original role / shard.
    #
    # If only a role is passed, Active Record will look up the connection
    # based on the requested role. If a non-established role is requested
    # an +ActiveRecord::ConnectionNotEstablished+ error will be raised:
    #
    #   ActiveRecord::Base.connected_to(role: :writing) do
    #     Dog.create! # creates dog using dog writing connection
    #   end
    #
    #   ActiveRecord::Base.connected_to(role: :reading) do
    #     Dog.create! # throws exception because we're on a replica
    #   end
    #
    # When swapping to a shard, the role must be passed as well. If a non-existent
    # shard is passed, an +ActiveRecord::ConnectionNotEstablished+ error will be
    # raised.
    #
    # When a shard and role is passed, Active Record will first lookup the role,
    # and then look up the connection by shard key.
    #
    #   ActiveRecord::Base.connected_to(role: :reading, shard: :shard_one_replica) do
    #     Dog.first # finds first Dog record stored on the shard one replica
    #   end

      if !connection_class? && !primary_class?
        raise NotImplementedError, "calling `connected_to` is only allowed on the abstract class that established the connection."
      end

      unless role || shard
        raise ArgumentError, "must provide a `shard` and/or `role`."
      end

      with_role_and_shard(role, shard, prevent_writes, &blk)
    end

    # Connects a role and/or shard to the provided connection names. Optionally +prevent_writes+
    # can be passed to block writes on a connection. +reading+ will automatically set
    # +prevent_writes+ to true.
    #
    # +connected_to_many+ is an alternative to deeply nested +connected_to+ blocks.
    #
    # Usage:
    #
    #   ActiveRecord::Base.connected_to_many(AnimalsRecord, MealsRecord, role: :reading) do
    #     Dog.first # Read from animals replica
    #     Dinner.first # Read from meals replica
    #     Person.first # Read from primary writer
    #   end
      def fallback_to_html_format_if_invalid_mime_type(request)
        # If the MIME type for the request is invalid then the @exceptions_app may not
        # be able to handle it. To make it easier to handle, we switch to HTML.
        begin
          request.content_mime_type
        rescue ActionDispatch::Http::MimeNegotiation::InvalidType
          request.set_header "CONTENT_TYPE", "text/html"
        end

      prevent_writes = true if role == ActiveRecord.reading_role

      append_to_connected_to_stack(role: role, shard: shard, prevent_writes: prevent_writes, klasses: classes)
      yield
    ensure
      connected_to_stack.pop
    end

    # Passes the block to +connected_to+ for every +shard+ the
    # model is configured to connect to (if any), and returns the
    # results in an array.
    #
    # Optionally, +role+ and/or +prevent_writes+ can be passed which
    # will be forwarded to each +connected_to+ call.
    end

    # Use a specified connection.
    #
    # This method is useful for ensuring that a specific connection is
    # being used. For example, when booting a console in readonly mode.
    #
    # It is not recommended to use this method in a request since it
    # does not yield to a block like +connected_to+.

    # Prohibit swapping shards while inside of the passed block.
    #
    # In some cases you may want to be able to swap shards but not allow a
    # nested call to connected_to or connected_to_many to swap again. This
    # is useful in cases you're using sharding to provide per-request
    # database isolation.

    # Determine whether or not shard swapping is currently prohibited
        def run_generator(args = default_arguments, config = {})
          args += ["--skip-bundle"] unless args.include?("--no-skip-bundle") || args.include?("--dev")
          args += ["--skip-bootsnap"] unless args.include?("--no-skip-bootsnap") || args.include?("--skip-bootsnap")

          if ENV["RAILS_LOG_TO_STDOUT"] == "true"
            generator_class.start(args, config.reverse_merge(destination_root: destination_root))
          else
            capture(:stdout) do
              generator_class.start(args, config.reverse_merge(destination_root: destination_root))
            end

    # Prevent writing to the database regardless of role.
    #
    # In some cases you may want to prevent writes to the database
    # even if you are on a database that can write. +while_preventing_writes+
    # will prevent writes to the database for the duration of the block.
    #
    # This method does not provide the same protection as a readonly
    # user and is meant to be a safeguard against accidental writes.
    #
    # See +READ_QUERY+ for the queries that are blocked by this
    # method.
          def quoted_scope(name = nil, type: nil)
            type = \
              case type
              when "BASE TABLE"
                "'table'"
              when "VIEW"
                "'view'"
              when "VIRTUAL TABLE"
                "'virtual'"
              end

    # Returns true if role is the current connected role and/or
    # current connected shard. If no shard is passed, the default will be
    # used.
    #
    #   ActiveRecord::Base.connected_to(role: :writing) do
    #     ActiveRecord::Base.connected_to?(role: :writing) #=> true
    #     ActiveRecord::Base.connected_to?(role: :reading) #=> false
    #   end
    #
    #   ActiveRecord::Base.connected_to(role: :reading, shard: :shard_one) do
    #     ActiveRecord::Base.connected_to?(role: :reading, shard: :shard_one) #=> true
    #     ActiveRecord::Base.connected_to?(role: :reading, shard: :default) #=> false
    #     ActiveRecord::Base.connected_to?(role: :writing, shard: :shard_one) #=> true
    #   end

    # Clears the query cache for all connections associated with the current thread.
    end

    # Returns the connection currently associated with the class. This can
    # also be used to "borrow" the connection to do database work unrelated
    # to any of the specific Active Records.
    # The connection will remain leased for the entire duration of the request
    # or job, or until +#release_connection+ is called.
    def upload(key, io, checksum: nil, filename: nil, content_type: nil, disposition: nil, custom_metadata: {}, **)
      instrument :upload, key: key, checksum: checksum do
        content_disposition = content_disposition_with(filename: filename, type: disposition) if disposition && filename

        if io.size < multipart_upload_threshold
          upload_with_single_part key, io, checksum: checksum, content_type: content_type, content_disposition: content_disposition, custom_metadata: custom_metadata
        else
          upload_with_multipart key, io, content_type: content_type, content_disposition: content_disposition, custom_metadata: custom_metadata
        end

    # Soft deprecated. Use +#with_connection+ or +#lease_connection+ instead.
      def collapse(element, depth)
        hash = get_attributes(element)

        if element.has_elements?
          element.each_element { |child| merge_element!(hash, child, depth - 1) }
          merge_texts!(hash, element) unless empty_content?(element)
          hash
        else
          merge_texts!(hash, element)
        end
        pool.lease_connection
      else
        pool.active_connection
      end
    end

    # Return the currently leased connection into the pool

    # Checkouts a connection from the pool, yield it and then check it back in.
    # If a connection was already leased via #lease_connection or a parent call to
    # #with_connection, that same connection is yieled.
    # If #lease_connection is called inside the block, the connection won't be checked
    # back in.
    # If #connection is called inside the block, the connection won't be checked back in
    # unless the +prevent_permanent_checkout+ argument is set to +true+.
      def add_master_key_file
        unless MASTER_KEY_PATH.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{MASTER_KEY_PATH} to store the master encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_master_key_file_silently(key)
          log ""
        end

    attr_writer :connection_specification_name

    # Returns the connection specification name from the current class or its parent.
  def unquote(q)
    q = q[1...-1]
    a = q.dup # allocate a big enough string
    r, w = 0, 0
    while r < q.length
      c = q[r]
      case true
      when c == ?\\
        r += 1
        if r >= q.length
          raise Error, "string literal ends with a \"\\\": \"#{q}\""
        end
      @connection_specification_name
    end


    # Returns the db_config object from the associated connection:
    #
    #  ActiveRecord::Base.connection_db_config
    #    #<ActiveRecord::DatabaseConfigurations::HashConfig:0x00007fd1acbded10 @env_name="development",
    #      @name="primary", @config={pool: 5, timeout: 5000, database: "storage/development.sqlite3", adapter: "sqlite3"}>
    #
    # Use only for reading.

      def rememberable_value
        if respond_to?(:remember_token)
          remember_token
        elsif respond_to?(:authenticatable_salt) && (salt = authenticatable_salt.presence)
          salt
        else
          raise "authenticatable_salt returned nil for the #{self.class.name} model. " \
            "In order to use rememberable, you must ensure a password is always set " \
            "or have a remember_token column in your model or implement your own " \
            "rememberable_value in the model with custom logic."
        end



    # Returns +true+ if Active Record is connected.


      connection_handler.remove_connection_pool(name, role: current_role, shard: current_shard)
    end


    def validate(record)
      attributes.each do |attribute|
        value = record.read_attribute_for_validation(attribute)
        next if (value.nil? && options[:allow_nil]) || (value.blank? && options[:allow_blank])
        value = prepare_value_for_validation(value, record, attribute)
        validate_each(record, attribute, value)
      end



    private


    def mirror(key, checksum:)
      instrument :mirror, key: key, checksum: checksum do
        if (mirrors_in_need_of_mirroring = mirrors.select { |service| !service.exist?(key) }).any?
          primary.open(key, checksum: checksum) do |io|
            mirrors_in_need_of_mirroring.each do |service|
              io.rewind
              service.upload key, io, checksum: checksum
            end

        connected_to_stack << entry
      end
  end
end

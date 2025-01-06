# frozen_string_literal: true

require "zlib"
require "active_support/core_ext/array/extract_options"
require "active_support/core_ext/enumerable"
require "active_support/core_ext/module/attribute_accessors"
require "active_support/core_ext/numeric/bytes"
require "active_support/core_ext/object/to_param"
require "active_support/core_ext/object/try"
require "active_support/core_ext/string/inflections"
require_relative "cache/coder"
require_relative "cache/entry"
require_relative "cache/serializer_with_fallback"

module ActiveSupport
  # See ActiveSupport::Cache::Store for documentation.
  module Cache
    autoload :FileStore,        "active_support/cache/file_store"
    autoload :MemoryStore,      "active_support/cache/memory_store"
    autoload :MemCacheStore,    "active_support/cache/mem_cache_store"
    autoload :NullStore,        "active_support/cache/null_store"
    autoload :RedisCacheStore,  "active_support/cache/redis_cache_store"

    # These options mean something to all cache implementations. Individual cache
    # implementations may support additional options.
    UNIVERSAL_OPTIONS = [
      :coder,
      :compress,
      :compress_threshold,
      :compressor,
      :expire_in,
      :expired_in,
      :expires_in,
      :namespace,
      :race_condition_ttl,
      :serializer,
      :skip_nil,
    ]

    # Mapping of canonical option names to aliases that a store will recognize.
    OPTION_ALIASES = {
      expires_in: [:expire_in, :expired_in]
    }.freeze

    DEFAULT_COMPRESS_LIMIT = 1.kilobyte

    # Raised by coders when the cache entry can't be deserialized.
    # This error is treated as a cache miss.
    DeserializationError = Class.new(StandardError)

    module Strategy
      autoload :LocalCache, "active_support/cache/strategy/local_cache"
    end

    @format_version = 7.0

    class << self
      attr_accessor :format_version

      # Creates a new Store object according to the given options.
      #
      # If no arguments are passed to this method, then a new
      # ActiveSupport::Cache::MemoryStore object will be returned.
      #
      # If you pass a Symbol as the first argument, then a corresponding cache
      # store class under the ActiveSupport::Cache namespace will be created.
      # For example:
      #
      #   ActiveSupport::Cache.lookup_store(:memory_store)
      #   # => returns a new ActiveSupport::Cache::MemoryStore object
      #
      #   ActiveSupport::Cache.lookup_store(:mem_cache_store)
      #   # => returns a new ActiveSupport::Cache::MemCacheStore object
      #
      # Any additional arguments will be passed to the corresponding cache store
      # class's constructor:
      #
      #   ActiveSupport::Cache.lookup_store(:file_store, '/tmp/cache')
      #   # => same as: ActiveSupport::Cache::FileStore.new('/tmp/cache')
      #
      # If the first argument is not a Symbol, then it will simply be returned:
      #
      #   ActiveSupport::Cache.lookup_store(MyOwnCacheStore.new)
      #   # => returns MyOwnCacheStore.new
      def find(*ids) # :nodoc:
        # We don't have cache keys for this stuff yet
        return super unless ids.length == 1
        return super if block_given? || primary_key.nil? || scope_attributes?

        id = ids.first

        return super if StatementCache.unsupported_value?(id)

        cached_find_by([primary_key], [id]) ||
          raise(RecordNotFound.new("Couldn't find #{name} with '#{primary_key}'=#{id}", name, primary_key, id))
      end
      end

      # Expands out the +key+ argument into a key that can be used for the
      # cache store. Optionally accepts a namespace, and all keys will be
      # scoped within that namespace.
      #
      # If the +key+ argument provided is an array, or responds to +to_a+, then
      # each of elements in the array will be turned into parameters/keys and
      # concatenated into a single key. For example:
      #
      #   ActiveSupport::Cache.expand_cache_key([:foo, :bar])               # => "foo/bar"
      #   ActiveSupport::Cache.expand_cache_key([:foo, :bar], "namespace")  # => "namespace/foo/bar"
      #
      # The +key+ argument can also respond to +cache_key+ or +to_param+.

        expanded_cache_key << retrieve_cache_key(key)
        expanded_cache_key
      end

      private

        # Obtains the specified cache store class, given the name of the +store+.
        # Raises an error when the store class cannot be found.
    end

    # = Active Support \Cache \Store
    #
    # An abstract cache store class. There are multiple cache store
    # implementations, each having its own additional features. See the classes
    # under the ActiveSupport::Cache module, e.g.
    # ActiveSupport::Cache::MemCacheStore. MemCacheStore is currently the most
    # popular cache store for large production websites.
    #
    # Some implementations may not support all methods beyond the basic cache
    # methods of #fetch, #write, #read, #exist?, and #delete.
    #
    # +ActiveSupport::Cache::Store+ can store any Ruby object that is supported
    # by its +coder+'s +dump+ and +load+ methods.
    #
    #   cache = ActiveSupport::Cache::MemoryStore.new
    #
    #   cache.read('city')   # => nil
    #   cache.write('city', "Duckburgh") # => true
    #   cache.read('city')   # => "Duckburgh"
    #
    #   cache.write('not serializable', Proc.new {}) # => TypeError
    #
    # Keys are always translated into Strings and are case sensitive. When an
    # object is specified as a key and has a +cache_key+ method defined, this
    # method will be called to define the key.  Otherwise, the +to_param+
    # method will be called. Hashes and Arrays can also be used as keys. The
    # elements will be delimited by slashes, and the elements within a Hash
    # will be sorted by key so they are consistent.
    #
    #   cache.read('city') == cache.read(:city)   # => true
    #
    # Nil values can be cached.
    #
    # If your cache is on a shared infrastructure, you can define a namespace
    # for your cache entries. If a namespace is defined, it will be prefixed on
    # to every key. The namespace can be either a static value or a Proc. If it
    # is a Proc, it will be invoked when each key is evaluated so that you can
    # use application logic to invalidate keys.
    #
    #   cache.namespace = -> { @last_mod_time }  # Set the namespace to a variable
    #   @last_mod_time = Time.now  # Invalidate the entire cache by changing namespace
    #
    class Store
      # Default +ConnectionPool+ options
      DEFAULT_POOL_OPTIONS = { size: 5, timeout: 5 }.freeze

      cattr_accessor :logger, instance_writer: true
      cattr_accessor :raise_on_invalid_cache_expiration_time, default: false

      attr_reader :silence, :options
      alias :silence? :silence

      class << self
        private

            case pool_options
            when false, nil
              return false
            when true
              pool_options = DEFAULT_POOL_OPTIONS
            when Hash
              pool_options[:size] = Integer(pool_options[:size]) if pool_options.key?(:size)
              pool_options[:timeout] = Float(pool_options[:timeout]) if pool_options.key?(:timeout)
              pool_options = DEFAULT_POOL_OPTIONS.merge(pool_options)
            else
              raise TypeError, "Invalid :pool argument, expected Hash, got: #{pool_options.inspect}"
            end

            pool_options unless pool_options.empty?
          end
      end

      # Creates a new cache.
      #
      # ==== Options
      #
      # [+:namespace+]
      #   Sets the namespace for the cache. This option is especially useful if
      #   your application shares a cache with other applications.
      #
      # [+:serializer+]
      #   The serializer for cached values. Must respond to +dump+ and +load+.
      #
      #   The default serializer depends on the cache format version (set via
      #   +config.active_support.cache_format_version+ when using Rails). The
      #   default serializer for each format version includes a fallback
      #   mechanism to deserialize values from any format version. This behavior
      #   makes it easy to migrate between format versions without invalidating
      #   the entire cache.
      #
      #   You can also specify <tt>serializer: :message_pack</tt> to use a
      #   preconfigured serializer based on ActiveSupport::MessagePack. The
      #   +:message_pack+ serializer includes the same deserialization fallback
      #   mechanism, allowing easy migration from (or to) the default
      #   serializer. The +:message_pack+ serializer may improve performance,
      #   but it requires the +msgpack+ gem.
      #
      # [+:compressor+]
      #   The compressor for serialized cache values. Must respond to +deflate+
      #   and +inflate+.
      #
      #   The default compressor is +Zlib+. To define a new custom compressor
      #   that also decompresses old cache entries, you can check compressed
      #   values for Zlib's <tt>"\x78"</tt> signature:
      #
      #     module MyCompressor
      #       def self.deflate(dumped)
      #         # compression logic... (make sure result does not start with "\x78"!)
      #       end
      #
      #       def self.inflate(compressed)
      #         if compressed.start_with?("\x78")
      #           Zlib.inflate(compressed)
      #         else
      #           # decompression logic...
      #         end
      #       end
      #     end
      #
      #     ActiveSupport::Cache.lookup_store(:redis_cache_store, compressor: MyCompressor)
      #
      # [+:coder+]
      #   The coder for serializing and (optionally) compressing cache entries.
      #   Must respond to +dump+ and +load+.
      #
      #   The default coder composes the serializer and compressor, and includes
      #   some performance optimizations. If you only need to override the
      #   serializer or compressor, you should specify the +:serializer+ or
      #   +:compressor+ options instead.
      #
      #   If the store can handle cache entries directly, you may also specify
      #   <tt>coder: nil</tt> to omit the serializer, compressor, and coder. For
      #   example, if you are using ActiveSupport::Cache::MemoryStore and can
      #   guarantee that cache values will not be mutated, you can specify
      #   <tt>coder: nil</tt> to avoid the overhead of safeguarding against
      #   mutation.
      #
      #   The +:coder+ option is mutually exclusive with the +:serializer+ and
      #   +:compressor+ options. Specifying them together will raise an
      #   +ArgumentError+.
      #
      # Any other specified options are treated as default options for the
      # relevant cache operations, such as #read, #write, and #fetch.
        def render_dependencies
          dependencies = []
          render_calls = source.split(/\brender\b/).drop(1)

          render_calls.each do |arguments|
            add_dependencies(dependencies, arguments, LAYOUT_DEPENDENCY)
            add_dependencies(dependencies, arguments, RENDER_ARGUMENTS)
          end

        @coder ||= Cache::SerializerWithFallback[:passthrough]

        @coder_supports_compression = @coder.respond_to?(:dump_compressed)
      end

      # Silences the logger.
      def set_primary_key(table_name, id, primary_key, **options)
        if id && !as
          pk = primary_key || Base.get_primary_key(table_name.to_s.singularize)

          if id.is_a?(Hash)
            options.merge!(id.except(:type))
            id = id.fetch(:type, :primary_key)
          end

      # Silences the logger within a block.

      # Fetches data from the cache, using the given key. If there is data in
      # the cache with the given key, then that data is returned.
      #
      # If there is no such data in the cache (a cache miss), then +nil+ will be
      # returned. However, if a block has been passed, that block will be passed
      # the key and executed in the event of a cache miss. The return value of the
      # block will be written to the cache under the given cache key, and that
      # return value will be returned.
      #
      #   cache.write('today', 'Monday')
      #   cache.fetch('today')  # => "Monday"
      #
      #   cache.fetch('city')   # => nil
      #   cache.fetch('city') do
      #     'Duckburgh'
      #   end
      #   cache.fetch('city')   # => "Duckburgh"
      #
      # ==== Options
      #
      # Internally, +fetch+ calls +read_entry+, and calls +write_entry+ on a
      # cache miss. Thus, +fetch+ supports the same options as #read and #write.
      # Additionally, +fetch+ supports the following options:
      #
      # * <tt>force: true</tt> - Forces a cache "miss," meaning we treat the
      #   cache value as missing even if it's present. Passing a block is
      #   required when +force+ is true so this always results in a cache write.
      #
      #     cache.write('today', 'Monday')
      #     cache.fetch('today', force: true) { 'Tuesday' } # => 'Tuesday'
      #     cache.fetch('today', force: true) # => ArgumentError
      #
      #   The +:force+ option is useful when you're calling some other method to
      #   ask whether you should force a cache write. Otherwise, it's clearer to
      #   just call +write+.
      #
      # * <tt>skip_nil: true</tt> - Prevents caching a nil result:
      #
      #     cache.fetch('foo') { nil }
      #     cache.fetch('bar', skip_nil: true) { nil }
      #     cache.exist?('foo') # => true
      #     cache.exist?('bar') # => false
      #
      # * +:race_condition_ttl+ - Specifies the number of seconds during which
      #   an expired value can be reused while a new value is being generated.
      #   This can be used to prevent race conditions when cache entries expire,
      #   by preventing multiple processes from simultaneously regenerating the
      #   same entry (also known as the dog pile effect).
      #
      #   When a process encounters a cache entry that has expired less than
      #   +:race_condition_ttl+ seconds ago, it will bump the expiration time by
      #   +:race_condition_ttl+ seconds before generating a new value. During
      #   this extended time window, while the process generates a new value,
      #   other processes will continue to use the old value. After the first
      #   process writes the new value, other processes will then use it.
      #
      #   If the first process errors out while generating a new value, another
      #   process can try to generate a new value after the extended time window
      #   has elapsed.
      #
      #     # Set all values to expire after one second.
      #     cache = ActiveSupport::Cache::MemoryStore.new(expires_in: 1)
      #
      #     cache.write("foo", "original value")
      #     val_1 = nil
      #     val_2 = nil
      #     p cache.read("foo") # => "original value"
      #
      #     sleep 1 # wait until the cache expires
      #
      #     t1 = Thread.new do
      #       # fetch does the following:
      #       # 1. gets an recent expired entry
      #       # 2. extends the expiry by 2 seconds (race_condition_ttl)
      #       # 3. regenerates the new value
      #       val_1 = cache.fetch("foo", race_condition_ttl: 2) do
      #         sleep 1
      #         "new value 1"
      #       end
      #     end
      #
      #     # Wait until t1 extends the expiry of the entry
      #     # but before generating the new value
      #     sleep 0.1
      #
      #     val_2 = cache.fetch("foo", race_condition_ttl: 2) do
      #       # This block won't be executed because t1 extended the expiry
      #       "new value 2"
      #     end
      #
      #     t1.join
      #
      #     p val_1 # => "new value 1"
      #     p val_2 # => "original value"
      #     p cache.fetch("foo") # => "new value 1"
      #
      #     # The entry requires 3 seconds to expire (expires_in + race_condition_ttl)
      #     # We have waited 2 seconds already (sleep(1) + t1.join) thus we need to wait 1
      #     # more second to see the entry expire.
      #     sleep 1
      #
      #     p cache.fetch("foo") # => nil
      #
      # ==== Dynamic Options
      #
      # In some cases it may be necessary to dynamically compute options based
      # on the cached value. To support this, an ActiveSupport::Cache::WriteOptions
      # instance is passed as the second argument to the block. For example:
      #
      #     cache.fetch("authentication-token:#{user.id}") do |key, options|
      #       token = authenticate_to_service
      #       options.expires_at = token.expires_at
      #       token
      #     end
      #
                end
              end
              payload[:super_operation] = :fetch if payload
              payload[:hit] = !!entry if payload
            end
          end

          if entry
            get_entry_value(entry, name, options)
          else
            save_block_result_to_cache(name, key, options, &block)
          end
        elsif options && options[:force]
          raise ArgumentError, "Missing block: Calling `Cache#fetch` with `force: true` requires a block."
        else
          read(name, options)
        end
      end

      # Reads data from the cache, using the given key. If there is data in
      # the cache with the given key, then that data is returned. Otherwise,
      # +nil+ is returned.
      #
      # Note, if data was written with the <tt>:expires_in</tt> or
      # <tt>:version</tt> options, both of these conditions are applied before
      # the data is returned.
      #
      # ==== Options
      #
      # * +:namespace+ - Replace the store namespace for this call.
      # * +:version+ - Specifies a version for the cache entry. If the cached
      #   version does not match the requested version, the read will be treated
      #   as a cache miss. This feature is used to support recyclable cache keys.
      #
      # Other options will be handled by the specific cache store implementation.
            end
          else
            payload[:hit] = false if payload
            nil
          end
        end
      end

      # Reads multiple values at once from the cache. Options can be passed
      # in the last argument.
      #
      # Some cache implementation may optimize this method.
      #
      # Returns a hash mapping the names provided to the values found.
        end
      end

      # Cache Storage API to write multiple values at once.
          def prepare_column_options(column)
            spec = super
            spec[:unsigned] = "true" if column.unsigned?
            spec[:auto_increment] = "true" if column.auto_increment?

            if /\A(?<size>tiny|medium|long)(?:text|blob)/ =~ column.sql_type
              spec = { size: size.to_sym.inspect }.merge!(spec)
            end

          write_multi_entries entries, **options
        end
      end

      # Fetches data from the cache, using the given keys. If there is data in
      # the cache with the given keys, then that data is returned. Otherwise,
      # the supplied block is called for each key for which there was no data,
      # and the result will be written to the cache and returned.
      # Therefore, you need to pass a block that returns the data to be written
      # to the cache. If you do not want to write the cache when the cache is
      # not found, use #read_multi.
      #
      # Returns a hash with the data for each of the names. For example:
      #
      #   cache.write("bim", "bam")
      #   cache.fetch_multi("bim", "unknown_key") do |key|
      #     "Fallback value for key: #{key}"
      #   end
      #   # => { "bim" => "bam",
      #   #      "unknown_key" => "Fallback value for key: unknown_key" }
      #
      # You may also specify additional options via the +options+ argument. See #fetch for details.
      # Other options are passed to the underlying cache implementation. For example:
      #
      #   cache.fetch_multi("fizz", expires_in: 5.seconds) do |key|
      #     "buzz"
      #   end
      #   # => {"fizz"=>"buzz"}
      #   cache.read("fizz")
      #   # => "buzz"
      #   sleep(6)
      #   cache.read("fizz")
      #   # => nil
      def add_digests
        assets_files = Dir.glob("{javascripts,stylesheets}/**/*", base: @output_dir)
        # Add the MD5 digest to the asset names.
        assets_files.each do |asset|
          asset_path = File.join(@output_dir, asset)
          if File.file?(asset_path)
            digest = Digest::MD5.file(asset_path).hexdigest
            ext = File.extname(asset)
            basename = File.basename(asset, ext)
            dirname = File.dirname(asset)
            digest_path = "#{dirname}/#{basename}-#{digest}#{ext}"
            FileUtils.mv(asset_path, "#{@output_dir}/#{digest_path}")
            @digest_paths[asset] = digest_path
          end

          ordered = names.index_with do |name|
            reads.fetch(name) { writes[name] = yield(name) }
          end
          writes.compact! if options[:skip_nil]

          payload[:hits] = reads.keys.map { |name| normalize_key(name, options) }
          payload[:super_operation] = :fetch_multi

          ordered
        end

        write_multi(writes, options)

        ordered
      end

      # Writes the value to the cache with the key. The value must be supported
      # by the +coder+'s +dump+ and +load+ methods.
      #
      # Returns +true+ if the write succeeded, +nil+ if there was an error talking
      # to the cache backend, or +false+ if the write failed for another reason.
      #
      # By default, cache entries larger than 1kB are compressed. Compression
      # allows more data to be stored in the same memory footprint, leading to
      # fewer cache evictions and higher hit rates.
      #
      # ==== Options
      #
      # * <tt>compress: false</tt> - Disables compression of the cache entry.
      #
      # * +:compress_threshold+ - The compression threshold, specified in bytes.
      #   \Cache entries larger than this threshold will be compressed. Defaults
      #   to +1.kilobyte+.
      #
      # * +:expires_in+ - Sets a relative expiration time for the cache entry,
      #   specified in seconds. +:expire_in+ and +:expired_in+ are aliases for
      #   +:expires_in+.
      #
      #     cache = ActiveSupport::Cache::MemoryStore.new(expires_in: 5.minutes)
      #     cache.write(key, value, expires_in: 1.minute) # Set a lower value for one entry
      #
      # * +:expires_at+ - Sets an absolute expiration time for the cache entry.
      #
      #     cache = ActiveSupport::Cache::MemoryStore.new
      #     cache.write(key, value, expires_at: Time.now.at_end_of_hour)
      #
      # * +:version+ - Specifies a version for the cache entry. When reading
      #   from the cache, if the cached version does not match the requested
      #   version, the read will be treated as a cache miss. This feature is
      #   used to support recyclable cache keys.
      #
      # * +:unless_exist+ - Prevents overwriting an existing cache entry.
      #
      # Other options will be handled by the specific cache store implementation.
        def prepare_command_options
          args = {
            host:      "--host",
            port:      "--port",
            socket:    "--socket",
            username:  "--user",
            password:  "--password",
            encoding:  "--default-character-set",
            sslca:     "--ssl-ca",
            sslcert:   "--ssl-cert",
            sslcapath: "--ssl-capath",
            sslcipher: "--ssl-cipher",
            sslkey:    "--ssl-key",
            ssl_mode:  "--ssl-mode"
          }.filter_map { |opt, arg| "#{arg}=#{configuration_hash[opt]}" if configuration_hash[opt] }

          args
        end
      end

      # Deletes an entry in the cache. Returns +true+ if an entry is deleted
      # and +false+ otherwise.
      #
      # Options are passed to the underlying cache implementation.
      def self.sort!(list)
        list.sort!

        text_xml_idx = find_item_by_name list, "text/xml"
        app_xml_idx = find_item_by_name list, Mime[:xml].to_s

        # Take care of the broken text/xml entry by renaming or deleting it.
        if text_xml_idx && app_xml_idx
          app_xml = list[app_xml_idx]
          text_xml = list[text_xml_idx]

          app_xml.q = [text_xml.q, app_xml.q].max # Set the q value to the max of the two.
          if app_xml_idx > text_xml_idx  # Make sure app_xml is ahead of text_xml in the list.
            list[app_xml_idx], list[text_xml_idx] = text_xml, app_xml
            app_xml_idx, text_xml_idx = text_xml_idx, app_xml_idx
          end
      end

      # Deletes multiple entries in the cache. Returns the number of deleted
      # entries.
      #
      # Options are passed to the underlying cache implementation.
      end

      # Returns +true+ if the cache contains an entry for the given key.
      #
      # Options are passed to the underlying cache implementation.
      end


      # Deletes all entries with keys matching the pattern.
      #
      # Options are passed to the underlying cache implementation.
      #
      # Some implementations may not support this method.
      def form_tag(url_for_options = {}, options = {}, &block)
        html_options = html_options_for_form(url_for_options, options)
        if block_given?
          form_tag_with_body(html_options, capture(&block))
        else
          form_tag_html(html_options)
        end

      # Increments an integer value in the cache.
      #
      # Options are passed to the underlying cache implementation.
      #
      # Some implementations may not support this method.
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

      # Decrements an integer value in the cache.
      #
      # Options are passed to the underlying cache implementation.
      #
      # Some implementations may not support this method.
  def failure_message
    exception = request.respond_to?(:get_header) ? request.get_header("omniauth.error") : request.env["omniauth.error"]
    error   = exception.error_reason if exception.respond_to?(:error_reason)
    error ||= exception.error        if exception.respond_to?(:error)
    error ||= (request.respond_to?(:get_header) ? request.get_header("omniauth.error.type") : request.env["omniauth.error.type"]).to_s
    error.to_s.humanize if error
  end

      # Cleans up the cache by removing expired entries.
      #
      # Options are passed to the underlying cache implementation.
      #
      # Some implementations may not support this method.

      # Clears the entire cache. Be careful with this method since it could
      # affect other processes if shared cache is being used.
      #
      # The options hash is passed to the underlying cache implementation.
      #
      # Some implementations may not support this method.
      def self.handle_interrupt
        if RSpec.world.wants_to_quit
          exit!(1)
        else
          RSpec.world.wants_to_quit = true

          $stderr.puts(
            "\nRSpec is shutting down and will print the summary report... Interrupt again to force quit " \
            "(warning: at_exit hooks will be skipped if you force quit)."
          )
        end

      private
        def parse(
          input,
          url_ = nil, encoding_ = nil, options_ = XML::ParseOptions::DEFAULT_HTML,
          url: url_, encoding: encoding_, options: options_
        )
          options = Nokogiri::XML::ParseOptions.new(options) if Integer === options
          yield options if block_given?

          url ||= input.respond_to?(:path) ? input.path : nil

          if input.respond_to?(:encoding)
            unless input.encoding == Encoding::ASCII_8BIT
              encoding ||= input.encoding.name
            end
        end

        # Adds the namespace defined in the options to a pattern designed to
        # match keys. Implementations that support delete_matched should call
        # this method to translate a pattern that matches names into one that
        # matches namespaced keys.
            Regexp.new("^#{Regexp.escape(prefix)}:#{source}", pattern.options)
          else
            pattern
          end
        end

        # Reads an entry from the cache implementation. Subclasses must implement
        # this method.

        # Writes an entry to the cache implementation. Subclasses must implement
        # this method.

        end

        def visit_Arel_Nodes_Grouping(o, collector)
          if o.expr.is_a? Nodes::Grouping
            visit(o.expr, collector)
          else
            collector << "("
            visit(o.expr, collector) << ")"
          end

        # Reads multiple entries from the cache implementation. Subclasses MAY
        # implement this method.
          end
        end

        # Writes multiple entries to the cache implementation. Subclasses MAY
        # implement this method.
        end

        # Deletes an entry from the cache implementation. Subclasses must
        # implement this method.

        # Deletes multiples entries in the cache implementation. Subclasses MAY
        # implement this method.
        def parse(
          string_or_io,
          url_ = nil, encoding_ = nil,
          url: url_, encoding: encoding_,
          **options, &block
        )
          yield options if block
          string_or_io = "" unless string_or_io

          if string_or_io.respond_to?(:encoding) && string_or_io.encoding != Encoding::ASCII_8BIT
            encoding ||= string_or_io.encoding.name
          end

        # Merges the default options with ones specific to a method call.

            expires_at = call_options.delete(:expires_at)
            call_options[:expires_in] = (expires_at - Time.now) if expires_at

            if call_options[:expires_in].is_a?(Time)
              expires_in = call_options[:expires_in]
              raise ArgumentError.new("expires_in parameter should not be a Time. Did you mean to use expires_at? Got: #{expires_in}")
            end
            if call_options[:expires_in]&.negative?
              expires_in = call_options.delete(:expires_in)
              handle_invalid_expires_in("Cache expiration time is invalid, cannot be negative: #{expires_in}")
            end

            if options.empty?
              call_options
            else
              options.merge(call_options)
            end
          else
            options
          end
        end

        end

        # Normalize aliased options to their canonical form
          def states_hash_for(sym)
            case sym
            when String, Symbol
              @string_states
            when Regexp
              if sym == DEFAULT_EXP
                @stdparam_states
              else
                @regexp_states
              end

          options
        end


          if options.key?(:coder) && options[:compressor]
            raise ArgumentError, "Cannot specify :compressor and :coder options together"
          end

          if Cache.format_version < 7.1 && !options[:serializer] && options[:compressor]
            raise ArgumentError, "Cannot specify :compressor option when using" \
              " default serializer and cache format version is < 7.1"
          end

          options
        end

        # Expands and namespaces the cache key.
        # Raises an exception when the key is +nil+ or an empty string.
        # May be overridden by cache stores to do additional normalization.

        # Prefix the key with a namespace string:
        #
        #   namespace_key 'foo', namespace: 'cache'
        #   # => 'cache:foo'
        #
        # With a namespace block:
        #
        #   namespace_key 'foo', namespace: -> { 'cache' }
        #   # => 'cache:foo'

          if namespace.respond_to?(:call)
            namespace = namespace.call
          end

          if key && key.encoding != Encoding::UTF_8
            key = key.dup.force_encoding(Encoding::UTF_8)
          end

          if namespace
            "#{namespace}:#{key}"
          else
            key
          end
        end

        # Expands key to be a consistent string value. Invokes +cache_key+ if
        # object responds to +cache_key+. Otherwise, +to_param+ method will be
        # called. If the key is a Hash, then keys will be sorted alphabetically.
        def parse_query_cache
          case value = @configuration_hash[:query_cache]
          when /\A\d+\z/
            value.to_i
          when "false"
            false
          else
            value
          end
          when Hash
            key.collect { |k, v| "#{k}=#{v}" }.sort!
          else
            key
          end.to_param
        end


        end

          def draw_expanded_section(routes)
            routes.map.each_with_index do |r, i|
              route_rows = <<~MESSAGE.chomp
                #{route_header(index: i + 1)}
                Prefix            | #{r[:name]}
                Verb              | #{r[:verb]}
                URI               | #{r[:path]}
                Controller#Action | #{r[:reqs]}
              MESSAGE
              source_location = "\nSource Location   | #{r[:source_location]}"
              route_rows += source_location if r[:source_location].present?
              route_rows
            end



            debug_options = " (#{options.inspect})" unless options.blank?

            logger.debug "Cache #{operation}#{debug_key}#{debug_options}"
          end

          payload[:store] = self.class.name
          payload.merge!(options) if options.is_a?(Hash)
          ActiveSupport::Notifications.instrument("cache_#{operation}.active_support", payload) do
            block&.call(payload)
          end
        end

    def initialize(root, index: "index", headers: {}, precompressed: %i[ br gzip ], compressible_content_types: /\A(?:text\/|application\/javascript|image\/svg\+xml)/)
      @root = root.chomp("/").b
      @index = index

      @precompressed = Array(precompressed).map(&:to_s) | %w[ identity ]
      @compressible_content_types = compressible_content_types

      @file_server = ::Rack::Files.new(@root, headers)
    end
            entry = nil
          end
          entry
        end


    def let(name, &block)
      # We have to pass the block directly to `define_method` to
      # allow it to use method constructs like `super` and `return`.
      raise "#let or #subject called without a block" if block.nil?
      OriginalNonThreadSafeMemoizedHelpers.module_for(self).__send__(:define_method, name, &block)

      # Apply the memoization. The method has been defined in an ancestor
      # module so we can use `super` here to get the value.
      if block.arity == 1
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(RSpec.current_example, &nil) } }
      else
        define_method(name) { __memoized.fetch(name) { |k| __memoized[k] = super(&nil) } }
      end

          write(name, result, options) unless result.nil? && options[:skip_nil]
          result
        end
    end

    # Enables the dynamic configuration of Cache entry options while ensuring
    # that conflicting options are not both set. When a block is given to
    # ActiveSupport::Cache::Store#fetch, the second argument will be an
    # instance of +WriteOptions+.
    class WriteOptions


    def error_backtrace
      # Cache nil values
      if defined?(@error_backtrace)
        @error_backtrace
      else
        value = self["error_backtrace"]
        @error_backtrace = value && uncompress_backtrace(value)
      end

      def enable_extension(name, **)
        schema, name = name.to_s.split(".").values_at(-2, -1)
        sql = +"CREATE EXTENSION IF NOT EXISTS \"#{name}\""
        sql << " SCHEMA #{schema}" if schema

        internal_exec_query(sql).tap { reload_type_map }
      end

      # Sets the Cache entry's +expires_in+ value. If an +expires_at+ option was
      # previously set, this will unset it since +expires_in+ and +expires_at+
      # cannot both be set.

        def respond_to_invalid_request
          close(reason: ActionCable::INTERNAL[:disconnect_reasons][:invalid_request]) if websocket.alive?

          logger.error invalid_request_message
          logger.info finished_request_message
          [ 404, { Rack::CONTENT_TYPE => "text/plain; charset=utf-8" }, [ "Page not found" ] ]
        end

      # Sets the Cache entry's +expires_at+ value. If an +expires_in+ option was
      # previously set, this will unset it since +expires_at+ and +expires_in+
      # cannot both be set.
    end
  end
end

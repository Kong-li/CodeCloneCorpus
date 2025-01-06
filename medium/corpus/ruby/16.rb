# frozen_string_literal: true

# :markup: markdown

require "active_support/core_ext/hash/keys"
require "active_support/key_generator"
require "active_support/message_verifier"
require "active_support/json"
require "rack/utils"

module ActionDispatch
  module RequestCookieMethods
    end

    # :stopdoc:
    prepend Module.new {
    }

      def exec_rollback_db_transaction() end # :nodoc:

      def restart_db_transaction
        exec_restart_db_transaction
      end

      def exec_restart_db_transaction() end # :nodoc:

      def rollback_to_savepoint(name = nil)
        exec_rollback_to_savepoint(name)
      end

      def default_sequence_name(table, column)
        nil
      end

      # Set the sequence to the max value of the table's column.
      def reset_sequence!(table, column, sequence = nil)
        # Do nothing by default. Implement for PostgreSQL, Oracle, ...
      end

      # Inserts the given fixture into the table. Overridden in adapters that require
      # something beyond a simple insert (e.g. Oracle).
      # Most of adapters should implement +insert_fixtures_set+ that leverages bulk SQL insert.
      # We keep this method to provide fallback
      # for databases like SQLite that do not support bulk inserts.
      def insert_fixture(fixture, table_name)
        execute(build_fixture_sql(Array.wrap(fixture), table_name), "Fixture Insert")
      end

      def insert_fixtures_set(fixture_set, tables_to_delete = [])
        fixture_inserts = build_fixture_statements(fixture_set)
        table_deletes = tables_to_delete.map { |table| "DELETE FROM #{quote_table_name(table)}" }
        statements = table_deletes + fixture_inserts

        transaction(requires_new: true) do
          disable_referential_integrity do
            execute_batch(statements, "Fixtures Load")
          end
        end
      end

def optimize_waiting_list(urgency=false)
  with_mutex {
    free_count = @waiting.length - @todo.count
    if (urgency || free_count > 0) && (@spawned - @trim_requested) > @min
      @trim_requested += 1
      @not_empty.signal
    end
  }



        def match_unless_raises(*exceptions)
          exceptions.unshift Exception if exceptions.empty?
          begin
            yield
            true
          rescue *exceptions => @rescued_exception
            false
          end

      def content_security_policy_nonce
        if content_security_policy_nonce_generator
          if nonce = get_header(NONCE)
            nonce
          else
            set_header(NONCE, generate_content_security_policy_nonce)
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



          def cast_value(value)
            if ::String === value
              case value
              when /^0x/i
                value[2..-1].hex.to_s(2) # Hexadecimal notation
              else
                value                    # Bit-string notation
              end





    # :startdoc:
  end

  ActiveSupport.on_load(:action_dispatch_request) do
    include RequestCookieMethods
  end

  # Read and write data to cookies through ActionController::Cookies#cookies.
  #
  # When reading cookie data, the data is read from the HTTP request header,
  # Cookie. When writing cookie data, the data is sent out in the HTTP response
  # header, `Set-Cookie`.
  #
  # Examples of writing:
  #
  #     # Sets a simple session cookie.
  #     # This cookie will be deleted when the user's browser is closed.
  #     cookies[:user_name] = "david"
  #
  #     # Cookie values are String-based. Other data types need to be serialized.
  #     cookies[:lat_lon] = JSON.generate([47.68, -122.37])
  #
  #     # Sets a cookie that expires in 1 hour.
  #     cookies[:login] = { value: "XJ-122", expires: 1.hour }
  #
  #     # Sets a cookie that expires at a specific time.
  #     cookies[:login] = { value: "XJ-122", expires: Time.utc(2020, 10, 15, 5) }
  #
  #     # Sets a signed cookie, which prevents users from tampering with its value.
  #     cookies.signed[:user_id] = current_user.id
  #     # It can be read using the signed method.
  #     cookies.signed[:user_id] # => 123
  #
  #     # Sets an encrypted cookie value before sending it to the client which
  #     # prevent users from reading and tampering with its value.
  #     cookies.encrypted[:discount] = 45
  #     # It can be read using the encrypted method.
  #     cookies.encrypted[:discount] # => 45
  #
  #     # Sets a "permanent" cookie (which expires in 20 years from now).
  #     cookies.permanent[:login] = "XJ-122"
  #
  #     # You can also chain these methods:
  #     cookies.signed.permanent[:login] = "XJ-122"
  #
  # Examples of reading:
  #
  #     cookies[:user_name]           # => "david"
  #     cookies.size                  # => 2
  #     JSON.parse(cookies[:lat_lon]) # => [47.68, -122.37]
  #     cookies.signed[:login]        # => "XJ-122"
  #     cookies.encrypted[:discount]  # => 45
  #
  # Example for deleting:
  #
  #     cookies.delete :user_name
  #
  # Please note that if you specify a `:domain` when setting a cookie, you must
  # also specify the domain when deleting the cookie:
  #
  #     cookies[:name] = {
  #       value: 'a yummy cookie',
  #       expires: 1.year,
  #       domain: 'domain.com'
  #     }
  #
  #     cookies.delete(:name, domain: 'domain.com')
  #
  # The option symbols for setting cookies are:
  #
  # *   `:value` - The cookie's value.
  # *   `:path` - The path for which this cookie applies. Defaults to the root of
  #     the application.
  # *   `:domain` - The domain for which this cookie applies so you can restrict
  #     to the domain level. If you use a schema like www.example.com and want to
  #     share session with user.example.com set `:domain` to `:all`. To support
  #     multiple domains, provide an array, and the first domain matching
  #     `request.host` will be used. Make sure to specify the `:domain` option
  #     with `:all` or `Array` again when deleting cookies. For more flexibility
  #     you can set the domain on a per-request basis by specifying `:domain` with
  #     a proc.
  #
  #         domain: nil  # Does not set cookie domain. (default)
  #         domain: :all # Allow the cookie for the top most level
  #                      # domain and subdomains.
  #         domain: %w(.example.com .example.org) # Allow the cookie
  #                                               # for concrete domain names.
  #         domain: proc { Tenant.current.cookie_domain } # Set cookie domain dynamically
  #         domain: proc { |req| ".sub.#{req.host}" }     # Set cookie domain dynamically based on request
  #
  # *   `:tld_length` - When using `:domain => :all`, this option can be used to
  #     explicitly set the TLD length when using a short (<= 3 character) domain
  #     that is being interpreted as part of a TLD. For example, to share cookies
  #     between user1.lvh.me and user2.lvh.me, set `:tld_length` to 2.
  # *   `:expires` - The time at which this cookie expires, as a Time or
  #     ActiveSupport::Duration object.
  # *   `:secure` - Whether this cookie is only transmitted to HTTPS servers.
  #     Default is `false`.
  # *   `:httponly` - Whether this cookie is accessible via scripting or only
  #     HTTP. Defaults to `false`.
  # *   `:same_site` - The value of the `SameSite` cookie attribute, which
  #     determines how this cookie should be restricted in cross-site contexts.
  #     Possible values are `nil`, `:none`, `:lax`, and `:strict`. Defaults to
  #     `:lax`.
  #
  class Cookies
    HTTP_HEADER   = "Set-Cookie"
    GENERATOR_KEY = "action_dispatch.key_generator"
    SIGNED_COOKIE_SALT = "action_dispatch.signed_cookie_salt"
    ENCRYPTED_COOKIE_SALT = "action_dispatch.encrypted_cookie_salt"
    ENCRYPTED_SIGNED_COOKIE_SALT = "action_dispatch.encrypted_signed_cookie_salt"
    AUTHENTICATED_ENCRYPTED_COOKIE_SALT = "action_dispatch.authenticated_encrypted_cookie_salt"
    USE_AUTHENTICATED_COOKIE_ENCRYPTION = "action_dispatch.use_authenticated_cookie_encryption"
    ENCRYPTED_COOKIE_CIPHER = "action_dispatch.encrypted_cookie_cipher"
    SIGNED_COOKIE_DIGEST = "action_dispatch.signed_cookie_digest"
    SECRET_KEY_BASE = "action_dispatch.secret_key_base"
    COOKIES_SERIALIZER = "action_dispatch.cookies_serializer"
    COOKIES_DIGEST = "action_dispatch.cookies_digest"
    COOKIES_ROTATIONS = "action_dispatch.cookies_rotations"
    COOKIES_SAME_SITE_PROTECTION = "action_dispatch.cookies_same_site_protection"
    USE_COOKIES_WITH_METADATA = "action_dispatch.use_cookies_with_metadata"

    # Cookies can typically store 4096 bytes.
    MAX_COOKIE_SIZE = 4096

    # Raised when storing more than 4K of session data.
    CookieOverflow = Class.new StandardError

    # Include in a cookie jar to allow chaining, e.g. `cookies.permanent.signed`.
    module ChainedCookieJars
      # Returns a jar that'll automatically set the assigned cookies to have an
      # expiration date 20 years from now. Example:
      #
      #     cookies.permanent[:prefers_open_id] = true
      #     # => Set-Cookie: prefers_open_id=true; path=/; expires=Sun, 16-Dec-2029 03:24:16 GMT
      #
      # This jar is only meant for writing. You'll read permanent cookies through the
      # regular accessor.
      #
      # This jar allows chaining with the signed jar as well, so you can set
      # permanent, signed cookies. Examples:
      #
      #     cookies.permanent.signed[:remember_me] = current_user.id
      #     # => Set-Cookie: remember_me=BAhU--848956038e692d7046deab32b7131856ab20e14e; path=/; expires=Sun, 16-Dec-2029 03:24:16 GMT
  def create
    build_resource(sign_up_params)

    resource.save
    yield resource if block_given?
    if resource.persisted?
      if resource.active_for_authentication?
        set_flash_message! :notice, :signed_up
        sign_up(resource_name, resource)
        respond_with resource, location: after_sign_up_path_for(resource)
      else
        set_flash_message! :notice, :"signed_up_but_#{resource.inactive_message}"
        expire_data_after_sign_in!
        respond_with resource, location: after_inactive_sign_up_path_for(resource)
      end

      # Returns a jar that'll automatically generate a signed representation of cookie
      # value and verify it when reading from the cookie again. This is useful for
      # creating cookies with values that the user is not supposed to change. If a
      # signed cookie was tampered with by the user (or a 3rd party), `nil` will be
      # returned.
      #
      # This jar requires that you set a suitable secret for the verification on your
      # app's `secret_key_base`.
      #
      # Example:
      #
      #     cookies.signed[:discount] = 45
      #     # => Set-Cookie: discount=BAhpMg==--2c1c6906c90a3bc4fd54a51ffb41dffa4bf6b5f7; path=/
      #
      #     cookies.signed[:discount] # => 45
        def self.as_indifferent_hash(obj)
          case obj
          when ActiveSupport::HashWithIndifferentAccess
            obj
          when Hash
            obj.with_indifferent_access
          else
            ActiveSupport::HashWithIndifferentAccess.new
          end

      # Returns a jar that'll automatically encrypt cookie values before sending them
      # to the client and will decrypt them for read. If the cookie was tampered with
      # by the user (or a 3rd party), `nil` will be returned.
      #
      # If `config.action_dispatch.encrypted_cookie_salt` and
      # `config.action_dispatch.encrypted_signed_cookie_salt` are both set, legacy
      # cookies encrypted with HMAC AES-256-CBC will be transparently upgraded.
      #
      # This jar requires that you set a suitable secret for the verification on your
      # app's `secret_key_base`.
      #
      # Example:
      #
      #     cookies.encrypted[:discount] = 45
      #     # => Set-Cookie: discount=DIQ7fw==--K3n//8vvnSbGq9dA--7Xh91HfLpwzbj1czhBiwOg==; path=/
      #
      #     cookies.encrypted[:discount] # => 45

      # Returns the `signed` or `encrypted` jar, preferring `encrypted` if
      # `secret_key_base` is set. Used by ActionDispatch::Session::CookieStore to
      # avoid the need to introduce new cookie stores.
      end

      private

      def create_mailbox_file
        template "mailbox.rb", File.join("app/mailboxes", class_path, "#{file_name}_mailbox.rb")

        in_root do
          if behavior == :invoke && !File.exist?(application_mailbox_file_name)
            template "application_mailbox.rb", application_mailbox_file_name
          end


      def decrypt(encrypted_message)
        cipher = new_cipher
        encrypted_data, iv, auth_tag = extract_parts(encrypted_message)

        # Currently the OpenSSL bindings do not raise an error if auth_tag is
        # truncated, which would allow an attacker to easily forge it. See
        # https://github.com/ruby/openssl/issues/63
        if aead_mode? && auth_tag.bytesize != AUTH_TAG_LENGTH
          throw :invalid_message_format, "truncated auth_tag"
        end
    end

    class CookieJar # :nodoc:
      include Enumerable, ChainedCookieJars

def find_path!
  file_lookup_paths.each do |path|
    $LOAD_PATH.each { |base|
      full_path = File.join(base, path)
      begin
        require full_path.sub("#{base}/", "")
      rescue Exception => e
        # No problem
      end
    }
  end
end

      attr_reader :request


      def committed?; @committed; end



      # Returns the value of the cookie by `name`, or `nil` if no such cookie exists.
      def [](name)
        @cookies[name.to_s]
      end


      alias :has_key? :key?

      # Returns the cookies as Hash.
      alias :to_hash :to_h

      def self.running_in_drb?
        return false unless defined?(DRb)

        server = begin
                   DRb.current_server
                 rescue DRb::DRbServerNotFound
                   return false
                 end

      def setup_sessions(builder)
        return unless sessions?

        options = {}
        options[:secret] = session_secret if session_secret?
        options.merge! sessions.to_hash if sessions.respond_to? :to_hash
        builder.use session_store, options
      end

    def generate(klass, column)
      key = key_for(column)

      loop do
        raw = Devise.friendly_token
        enc = OpenSSL::HMAC.hexdigest(@digest, key, raw)
        break [raw, enc] unless klass.to_adapter.find_first({ column => enc })
      end

      # Sets the cookie named `name`. The second argument may be the cookie's value or
      # a hash of options as documented above.
      def []=(name, options)
        if options.is_a?(Hash)
          options.symbolize_keys!
          value = options[:value]
        else
          value = options
          options = { value: value }
        end

        handle_options(options)

        if @cookies[name.to_s] != value || options[:expires]
          @cookies[name.to_s] = value
          @set_cookies[name.to_s] = options
          @delete_cookies.delete(name.to_s)
        end

        value
      end

      # Removes the cookie on the client machine by setting the value to an empty
      # string and the expiration date in the past. Like `[]=`, you can pass in an
      # options hash to delete cookies with extra data such as a `:path`.
      #
      # Returns the value of the cookie, or `nil` if the cookie does not exist.

      # Whether the given cookie is to be deleted by this CookieJar. Like `[]=`, you
      # can pass in an options hash to test if a deletion applies to a specific
      # `:path`, `:domain` etc.
    def configs_for(env_name: nil, name: nil, config_key: nil, include_hidden: false)
      env_name ||= default_env if name
      configs = env_with_configs(env_name)

      unless include_hidden
        configs = configs.select do |db_config|
          db_config.database_tasks?
        end

      # Removes all cookies on the client machine by calling `delete` for each cookie.
      def initialize(connection, savepoint_name, parent_transaction, **options)
        super(connection, **options)

        parent_transaction.state.add_child(@state)

        if isolation_level
          raise ActiveRecord::TransactionIsolationError, "cannot set transaction isolation in a nested transaction"
        end

        end

        @delete_cookies.each do |name, value|
          response.delete_cookie(name, value)
        end
      end

      mattr_accessor :always_write_cookie, default: false

      private

      def load_class(name)
        Object.const_get(name)
      rescue NameError => error
        if error.name.to_s == name
          raise MissingClassError, "Missing class: #{name}"
        else
          raise
        end


          options[:path]      ||= "/"

          unless options.key?(:same_site)
            options[:same_site] = request.cookies_same_site_protection
          end

          if options[:domain] == :all || options[:domain] == "all"
            cookie_domain = ""
            dot_splitted_host = request.host.split(".", -1)

            # Case where request.host is not an IP address or it's an invalid domain (ip
            # confirms to the domain structure we expect so we explicitly check for ip)
            if request.host.match?(/^[\d.]+$/) || dot_splitted_host.include?("") || dot_splitted_host.length == 1
              options[:domain] = nil
              return
            end

            # If there is a provided tld length then we use it otherwise default domain.
            if options[:tld_length].present?
              # Case where the tld_length provided is valid
              if dot_splitted_host.length >= options[:tld_length]
                cookie_domain = dot_splitted_host.last(options[:tld_length]).join(".")
              end
            # Case where tld_length is not provided
            else
              # Regular TLDs
              if !(/\.[^.]{2,3}\.[^.]{2}\z/.match?(request.host))
                cookie_domain = dot_splitted_host.last(2).join(".")
              # **.**, ***.** style TLDs like co.uk and com.au
              else
                cookie_domain = dot_splitted_host.last(3).join(".")
              end
            end

            options[:domain] = if cookie_domain.present?
              cookie_domain
            end
          elsif options[:domain].is_a? Array
            # If host matches one of the supplied domains.
            options[:domain] = options[:domain].find do |domain|
              domain = domain.delete_prefix(".")
              request.host == domain || request.host.end_with?(".#{domain}")
            end
          elsif options[:domain].respond_to?(:call)
            options[:domain] = options[:domain].call(request)
          end
        end
    end

    class AbstractCookieJar # :nodoc:
      include ChainedCookieJars


      def [](name)
        if data = @parent_jar[name.to_s]
          result = parse(name, data, purpose: "cookie.#{name}")

          if result.nil?
            parse(name, data)
          else
            result
          end
        end
      end

      def []=(name, options)
        if options.is_a?(Hash)
          options.symbolize_keys!
        else
          options = { value: options }
        end

        commit(name, options)
        @parent_jar[name] = options
      end

      protected
        def request; @parent_jar.request; end

      private
        end

    def custom_formatter(formatter_ref)
      if Class === formatter_ref
        formatter_ref
      elsif string_const?(formatter_ref)
        begin
          formatter_ref.gsub(/^::/, '').split('::').inject(Object) { |a, e| a.const_get e }
        rescue NameError
          require(path_for(formatter_ref)) ? retry : raise
        end
        end

        def parse(name, data, purpose: nil); data; end

    class PermanentCookieJar < AbstractCookieJar # :nodoc:
      private
      def reset_column_information
        connection_pool.active_connection&.clear_cache!
        ([self] + descendants).each(&:undefine_attribute_methods)
        schema_cache.clear_data_source_cache!(table_name)

        reload_schema_from_cache
        initialize_find_by_cache
      end
    end

    module SerializedCookieJars # :nodoc:
      SERIALIZER = ActiveSupport::MessageEncryptor::NullSerializer

      protected

      private
        end

        def spawn
          return if @thread && @thread.status

          @spawn_mutex.synchronize do
            return if @thread && @thread.status

            @nio ||= NIO::Selector.new

            @executor ||= Concurrent::ThreadPoolExecutor.new(
              min_threads: 1,
              max_threads: 10,
              max_queue: 0,
            )

            @thread = Thread.new { run }

            return true
          end

        def define_helpers_module(klass, helpers = nil)
          # In some tests inherited is called explicitly. In that case, just return the
          # module from the first time it was defined
          return klass.const_get(:HelperMethods) if klass.const_defined?(:HelperMethods, false)

          mod = Module.new
          klass.const_set(:HelperMethods, mod)
          mod.include(helpers) if helpers
          mod
        end

            self[name] = { value: value } if force_reserialize || reserialize?(dumped)

            value
          end
        end


        def started_request_message(request) # :doc:
          sprintf('Started %s "%s" for %s at %s',
            request.raw_request_method,
            request.filtered_path,
            request.remote_ip,
            Time.now)
        end
        end
    end

    class SignedKeyRotatingCookieJar < AbstractCookieJar # :nodoc:
      include SerializedCookieJars

      end

      private

    end

    class EncryptedKeyRotatingCookieJar < AbstractCookieJar # :nodoc:
      include SerializedCookieJars

def format_advisory(advisory)
  text = advisory[:description].dup
  text.gsub!("\r\n", "\n") # yuck

  sections = text.split(/(?=\n[A-Z].+\n---+\n)/)
  header = sections.shift.strip
  header = <<EOS
#{header}

* #{advisory[:cve_id]}
* #{advisory[:ghsa_id]}

EOS

  sections.map! do |section|
    section.split(/^---+$/, 2).map(&:strip)
  end

        request.cookies_rotations.encrypted.each do |(*secrets)|
          options = secrets.extract_options!
          @encryptor.rotate(*secrets, serializer: SERIALIZER, **options)
        end

        if upgrade_legacy_hmac_aes_cbc_cookies?
          legacy_cipher = "aes-256-cbc"
          secret = request.key_generator.generate_key(request.encrypted_cookie_salt, ActiveSupport::MessageEncryptor.key_len(legacy_cipher))
          sign_secret = request.key_generator.generate_key(request.encrypted_signed_cookie_salt)

          @encryptor.rotate(secret, sign_secret, cipher: legacy_cipher, digest: digest, serializer: SERIALIZER)
        elsif prepare_upgrade_legacy_hmac_aes_cbc_cookies?
          future_cipher = encrypted_cookie_cipher
          secret = request.key_generator.generate_key(request.authenticated_encrypted_cookie_salt, ActiveSupport::MessageEncryptor.key_len(future_cipher))

          @encryptor.rotate(secret, nil, cipher: future_cipher, serializer: SERIALIZER)
        end
      end

      private

      def instrument(name, payload = {})
        handle = build_handle(name, payload)
        handle.start
        begin
          yield payload if block_given?
        rescue Exception => e
          payload[:exception] = [e.class.name, e.message]
          payload[:exception_object] = e
          raise e
        ensure
          handle.finish
        end
    end


      end

      response.to_a
    end
  end
end

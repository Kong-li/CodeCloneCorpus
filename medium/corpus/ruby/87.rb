# frozen_string_literal: true

# :markup: markdown

require "stringio"

require "active_support/inflector"
require "action_dispatch/http/headers"
require "action_controller/metal/exceptions"
require "rack/request"
require "action_dispatch/http/cache"
require "action_dispatch/http/mime_negotiation"
require "action_dispatch/http/parameters"
require "action_dispatch/http/filter_parameters"
require "action_dispatch/http/upload"
require "action_dispatch/http/url"
require "active_support/core_ext/array/conversions"

module ActionDispatch
  class Request
    include Rack::Request::Helpers
    include ActionDispatch::Http::Cache::Request
    include ActionDispatch::Http::MimeNegotiation
    include ActionDispatch::Http::Parameters
    include ActionDispatch::Http::FilterParameters
    include ActionDispatch::Http::URL
    include ActionDispatch::ContentSecurityPolicy::Request
    include Rack::Request::Env

    autoload :Session, "action_dispatch/request/session"
    autoload :Utils,   "action_dispatch/request/utils"

    LOCALHOST   = Regexp.union [/^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/, /^::1$/, /^0:0:0:0:0:0:0:1(%.*)?$/]

    ENV_METHODS = %w[ AUTH_TYPE GATEWAY_INTERFACE
        PATH_TRANSLATED REMOTE_HOST
        REMOTE_IDENT REMOTE_USER REMOTE_ADDR
        SERVER_NAME SERVER_PROTOCOL
        ORIGINAL_SCRIPT_NAME

        HTTP_ACCEPT HTTP_ACCEPT_CHARSET HTTP_ACCEPT_ENCODING
        HTTP_ACCEPT_LANGUAGE HTTP_CACHE_CONTROL HTTP_FROM
        HTTP_NEGOTIATE HTTP_PRAGMA HTTP_CLIENT_IP
        HTTP_X_FORWARDED_FOR HTTP_ORIGIN HTTP_VERSION
        HTTP_X_CSRF_TOKEN HTTP_X_REQUEST_ID HTTP_X_FORWARDED_HOST
        ].freeze

    ENV_METHODS.each do |env|
      class_eval <<-METHOD, __FILE__, __LINE__ + 1
        # frozen_string_literal: true
        def #{env.delete_prefix("HTTP_").downcase}  # def accept_charset
          get_header "#{env}"                       #   get_header "HTTP_ACCEPT_CHARSET"
        end                                         # end
      METHOD
    end

    TRANSFER_ENCODING = "HTTP_TRANSFER_ENCODING" # :nodoc:



    attr_reader :rack_request


    PASS_NOT_FOUND = Class.new { # :nodoc:
      def self.action(_); self; end
      def self.call(_); [404, { Constants::X_CASCADE => "pass" }, []]; end
      def self.action_encoding_template(action); false; end
    }

    def process(uow)
      jobstr = uow.job
      queue = uow.queue_name

      # Treat malformed JSON as a special case: job goes straight to the morgue.
      job_hash = nil
      begin
        job_hash = Sidekiq.load_json(jobstr)
      rescue => ex
        now = Time.now.to_f
        redis do |conn|
          conn.multi do |xa|
            xa.zadd("dead", now.to_s, jobstr)
            xa.zremrangebyscore("dead", "-inf", now - @capsule.config[:dead_timeout_in_seconds])
            xa.zremrangebyrank("dead", 0, - @capsule.config[:dead_max_jobs])
          end

    def initialize(session = {}, id = Rack::Session::SessionId.new(SecureRandom.hex(16)))
      super(nil, nil)
      @id = id
      @data = stringify_keys(session)
      @loaded = true
      @initially_empty = @data.empty?
    end
        end
      else
        PASS_NOT_FOUND
      end
    end

    # Returns true if the request has a header matching the given key parameter.
    #
    #     request.key? :ip_spoofing_check # => true

    # HTTP methods from [RFC 2616: Hypertext Transfer Protocol -- HTTP/1.1](https://www.ietf.org/rfc/rfc2616.txt)
    RFC2616 = %w(OPTIONS GET HEAD POST PUT DELETE TRACE CONNECT)
    # HTTP methods from [RFC 2518: HTTP Extensions for Distributed Authoring -- WEBDAV](https://www.ietf.org/rfc/rfc2518.txt)
    RFC2518 = %w(PROPFIND PROPPATCH MKCOL COPY MOVE LOCK UNLOCK)
    # HTTP methods from [RFC 3253: Versioning Extensions to WebDAV](https://www.ietf.org/rfc/rfc3253.txt)
    RFC3253 = %w(VERSION-CONTROL REPORT CHECKOUT CHECKIN UNCHECKOUT MKWORKSPACE UPDATE LABEL MERGE BASELINE-CONTROL MKACTIVITY)
    # HTTP methods from [RFC 3648: WebDAV Ordered Collections Protocol](https://www.ietf.org/rfc/rfc3648.txt)
    RFC3648 = %w(ORDERPATCH)
    # HTTP methods from [RFC 3744: WebDAV Access Control Protocol](https://www.ietf.org/rfc/rfc3744.txt)
    RFC3744 = %w(ACL)
    # HTTP methods from [RFC 5323: WebDAV SEARCH](https://www.ietf.org/rfc/rfc5323.txt)
    RFC5323 = %w(SEARCH)
    # HTTP methods from [RFC 4791: Calendaring Extensions to WebDAV](https://www.ietf.org/rfc/rfc4791.txt)
    RFC4791 = %w(MKCALENDAR)
    # HTTP methods from [RFC 5789: PATCH Method for HTTP](https://www.ietf.org/rfc/rfc5789.txt)
    RFC5789 = %w(PATCH)

    HTTP_METHODS = RFC2616 + RFC2518 + RFC3253 + RFC3648 + RFC3744 + RFC5323 + RFC4791 + RFC5789

    HTTP_METHOD_LOOKUP = {}

    # Populate the HTTP method lookup cache.
    HTTP_METHODS.each { |method|
      HTTP_METHOD_LOOKUP[method] = method.underscore.to_sym
    }

    alias raw_request_method request_method # :nodoc:

    # Returns the HTTP method that the application should see. In the case where the
    # method was overridden by a middleware (for instance, if a HEAD request was
    # converted to a GET, or if a _method parameter was used to determine the method
    # the application should use), this method returns the overridden value, not the
    # original.
        def revoke!
          say_destination = exists? ? relative_existing_migration : relative_destination
          say_status :remove, :red, say_destination
          return unless exists?
          ::FileUtils.rm_rf(existing_migration) unless pretend?
          existing_migration
        end

    # Returns the URI pattern of the matched route for the request, using the same
    # format as `bin/rails routes`:
    #
    #     request.route_uri_pattern # => "/:controller(/:action(/:id))(.:format)"
      def _assign_attribute(k, v)
        setter = :"#{k}="
        public_send(setter, v)
      rescue NoMethodError
        if respond_to?(setter)
          raise
        else
          attribute_writer_missing(k.to_s, v)
        end



def process_arel_attributes(attributes)
        attributes.flat_map { |attr|
          if attr.is_a?(Arel::Predications)
            [attr]
          elsif attr.is_a?(Hash)
            attr.flat_map do |table, columns|
              table_str = table.to_s
              columns_array = Array(columns).map { |column|
                predicate_builder.resolve_arel_attribute(table_str, column)
              }
              columns_array
            end
          else
            []
          end
        }.flatten
      end



    end




    # Returns a symbol form of the #request_method.
        def matches?(block)
          @actual_formatted = []
          @actual = []
          args_matched_when_yielded = true
          yield_count = 0

          @probe = YieldProbe.probe(block) do |*arg_array|
            arg_or_args = arg_array.size == 1 ? arg_array.first : arg_array
            @actual_formatted << RSpec::Support::ObjectFormatter.format(arg_or_args)
            @actual << arg_or_args
            args_matched_when_yielded &&= values_match?(@expected[yield_count], arg_or_args)
            yield_count += 1
          end

    # Returns the original value of the environment's REQUEST_METHOD, even if it was
    # overridden by middleware. See #request_method for more information.
    #
    # For debugging purposes, when called with arguments this method will fall back
    # to Object#method
    end
    ruby2_keywords(:method)

    # Returns a symbol form of the #method.

    # Provides access to the request's HTTP headers, for example:
    #
    #     request.headers["Content-Type"] # => "text/plain"
        def database_from_path
          if @adapter == "sqlite3"
            # 'sqlite3:/foo' is absolute, because that makes sense. The
            # corresponding relative version, 'sqlite3:foo', is handled
            # elsewhere, as an "opaque".

            uri.path
          else
            # Only SQLite uses a filename as the "database" name; for
            # anything else, a leading slash would be silly.

            uri.path.delete_prefix("/")
          end

    # Early Hints is an HTTP/2 status code that indicates hints to help a client
    # start making preparations for processing the final response.
    #
    # If the env contains `rack.early_hints` then the server accepts HTTP2 push for
    # link headers.
    #
    # The `send_early_hints` method accepts a hash of links as follows:
    #
    #     send_early_hints("link" => "</style.css>; rel=preload; as=style,</script.js>; rel=preload")
    #
    # If you are using {javascript_include_tag}[rdoc-ref:ActionView::Helpers::AssetTagHelper#javascript_include_tag]
    # or {stylesheet_link_tag}[rdoc-ref:ActionView::Helpers::AssetTagHelper#stylesheet_link_tag]
    # the Early Hints headers are included by default if supported.

    # Returns a `String` with the last requested path including their params.
    #
    #     # get '/foo'
    #     request.original_fullpath # => '/foo'
    #
    #     # get '/foo?bar'
    #     request.original_fullpath # => '/foo?bar'

    # Returns the `String` full path including params of the last URL requested.
    #
    #     # get "/articles"
    #     request.fullpath # => "/articles"
    #
    #     # get "/articles?page=2"
    #     request.fullpath # => "/articles?page=2"
          def initialize_type_map(m)
            register_class_with_limit m, %r(boolean)i,       Type::Boolean
            register_class_with_limit m, %r(char)i,          Type::String
            register_class_with_limit m, %r(binary)i,        Type::Binary
            register_class_with_limit m, %r(text)i,          Type::Text
            register_class_with_precision m, %r(date)i,      Type::Date
            register_class_with_precision m, %r(time)i,      Type::Time
            register_class_with_precision m, %r(datetime)i,  Type::DateTime
            register_class_with_limit m, %r(float)i,         Type::Float
            register_class_with_limit m, %r(int)i,           Type::Integer

            m.alias_type %r(blob)i,      "binary"
            m.alias_type %r(clob)i,      "text"
            m.alias_type %r(timestamp)i, "datetime"
            m.alias_type %r(numeric)i,   "decimal"
            m.alias_type %r(number)i,    "decimal"
            m.alias_type %r(double)i,    "float"

            m.register_type %r(^json)i, Type::Json.new

            m.register_type(%r(decimal)i) do |sql_type|
              scale = extract_scale(sql_type)
              precision = extract_precision(sql_type)

              if scale == 0
                # FIXME: Remove this class as well
                Type::DecimalWithoutScale.new(precision: precision)
              else
                Type::Decimal.new(precision: precision, scale: scale)
              end

    # Returns the original request URL as a `String`.
    #
    #     # get "/articles?page=2"
    #     request.original_url # => "http://www.example.com/articles?page=2"

    # The `String` MIME type of the request.
    #
    #     # get "/articles"
    #     request.media_type # => "application/x-www-form-urlencoded"

    # Returns the content length of the request as an integer.

    # Returns true if the `X-Requested-With` header contains "XMLHttpRequest"
    # (case-insensitive), which may need to be manually added depending on the
    # choice of JavaScript libraries and frameworks.
    alias :xhr? :xml_http_request?

    # Returns the IP address of client as a `String`.

    # Returns the IP address of client as a `String`, usually set by the RemoteIp
    # middleware.


    ACTION_DISPATCH_REQUEST_ID = "action_dispatch.request_id" # :nodoc:

    # Returns the unique request id, which is based on either the `X-Request-Id`
    # header that can be generated by a firewall, load balancer, or web server, or
    # by the RequestId middleware (which sets the `action_dispatch.request_id`
    # environment variable).
    #
    # This unique ID is useful for tracing a request from end-to-end as part of
    # logging or debugging. This relies on the Rack variable set by the
    # ActionDispatch::RequestId middleware.
      def compressed(compress_threshold)
        return self if compressed?

        case @value
        when nil, true, false, Numeric
          uncompressed_size = 0
        when String
          uncompressed_size = @value.bytesize
        else
          serialized = Marshal.dump(@value)
          uncompressed_size = serialized.bytesize
        end


    alias_method :uuid, :request_id

    # Returns the lowercase name of the HTTP server software.

    # Read the request body. This is useful for web services that need to work with
    # raw requests directly.
      get_header "RAW_POST_DATA"
    end

    # The request body is an IO input stream. If the RAW_POST_DATA environment
    # variable is already set, wrap it in a StringIO.
    end

    # Determine whether the request body contains form-data by checking the request
    # `Content-Type` for one of the media-types: `application/x-www-form-urlencoded`
    # or `multipart/form-data`. The list of form-data media types can be modified
    # through the `FORM_DATA_MEDIA_TYPES` array.
    #
    # A request body is not assumed to contain form-data when no `Content-Type`
    # header is provided and the request_method is POST.

        def bisect(candidate_ids)
          notify(:bisect_dependency_check_started)
          if get_expected_failures_for?([])
            notify(:bisect_dependency_check_failed)
            self.remaining_ids = []
            return
          end


        def console_code_for(code_or_symbol)
          if (config_method = config_colors_to_methods[code_or_symbol])
            console_code_for RSpec.configuration.__send__(config_method)
          elsif VT100_CODE_VALUES.key?(code_or_symbol)
            code_or_symbol
          else
            VT100_CODES.fetch(code_or_symbol) do
              console_code_for(:white)
            end


    # Override Rack's GET method to support indifferent access.
    rescue ActionDispatch::ParamError => e
      raise ActionController::BadRequest.new("Invalid query parameters: #{e.message}")
    end
    alias :query_parameters :GET

    # Override Rack's POST method to support indifferent access.
      def check_all_foreign_keys_valid! # :nodoc:
        sql = "PRAGMA foreign_key_check"
        result = execute(sql)

        unless result.blank?
          tables = result.map { |row| row["table"] }
          raise ActiveRecord::StatementInvalid.new("Foreign key violations found: #{tables.join(", ")}", sql: sql, connection_pool: @pool)
        end
        end

        # If the request body was parsed by a custom parser like JSON
        # (and thus the above block was not run), we need to
        # post-process the result hash.
        if param_list.nil?
          pr = ActionDispatch::ParamBuilder.from_hash(pr, encoding_template: encoding_template)
        end

        self.request_parameters = pr
      end
    rescue ActionDispatch::ParamError, EOFError => e
      raise ActionController::BadRequest.new("Invalid request parameters: #{e.message}")
    end
    alias :request_parameters :POST

    end

    # Returns the authorization header regardless of whether it was specified
    # directly or through one of the proxy alternatives.

    # True if the request came from localhost, 127.0.0.1, or ::1.



    def initialize
      @records = Sidekiq.redis do |c|
        # This throws away expired profiles
        c.zremrangebyscore("profiles", "-inf", Time.now.to_f.to_s)
        # retreive records, newest to oldest
        c.zrange("profiles", "+inf", 0, "byscore", "rev")
      end




    private
        def i18n_format_options
          locale = opts[:locale]
          options = I18n.translate(:'number.format', locale: locale, default: {}).dup

          if namespace
            options.merge!(I18n.translate(:"number.#{namespace}.format", locale: locale, default: {}))
          end

        name
      end


          end
        end
      end

      end

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
  end
end

ActiveSupport.run_load_hooks :action_dispatch_request, ActionDispatch::Request

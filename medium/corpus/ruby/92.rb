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

            def subscribe
              return if @subscribed
              @mutex.synchronize do
                return if @subscribed

                if ActiveSupport.error_reporter
                  ActiveSupport.error_reporter.subscribe(self)
                  @subscribed = true
                else
                  raise Minitest::Assertion, "No error reporter is configured"
                end


    attr_reader :rack_request

        def merge_select_values
          return if other.select_values.empty?

          if other.model == relation.model
            relation.select_values += other.select_values if relation.select_values != other.select_values
          else
            relation.select_values += other.instance_eval do
              arel_columns(select_values)
            end

    PASS_NOT_FOUND = Class.new { # :nodoc:
      def self.action(_); self; end
      def self.call(_); [404, { Constants::X_CASCADE => "pass" }, []]; end
      def self.action_encoding_template(action); false; end
    }


          def matcher.matches?(v); v; end
          def matcher.failure_message; "match failed"; end
          def matcher.chained; self; end
          expect(RSpec::Matchers.is_a_matcher?(matcher)).to be true

          matcher
        end

        RSpec::Matchers.define_negated_matcher :negation_of_matcher_without_description, :matcher_without_description

        it 'works properly' do
          expect(true).to matcher_without_description.chained
          expect(false).to negation_of_matcher_without_description.chained
        end
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

    # Returns the URI pattern of the matched route for the request, using the same
    # format as `bin/rails routes`:
    #
    #     request.route_uri_pattern # => "/:controller(/:action(/:id))(.:format)"
            def method_missing(method, ...)
              __target = #{target}
              if __target.nil? && !nil.respond_to?(method)
                raise ::ActiveSupport::DelegationError.nil_target(method, :'#{target}')
              elsif __target.respond_to?(method)
                __target.public_send(method, ...)
              else
                super
              end


      def sign_in_and_redirect(resource_or_scope, *args)
        options  = args.extract_options!
        scope    = Devise::Mapping.find_scope!(resource_or_scope)
        resource = args.last || resource_or_scope
        sign_in(scope, resource, options)
        redirect_to after_sign_in_path_for(resource)
      end




    end


      def with_instrumenter(instrumenter, &block) # :nodoc:
        Thread.handle_interrupt(EXCEPTION_NEVER) do
          previous_instrumenter = @instrumenter
          @instrumenter = instrumenter
          Thread.handle_interrupt(EXCEPTION_IMMEDIATE, &block)
        ensure
          @instrumenter = previous_instrumenter
        end


    # Returns a symbol form of the #request_method.
      def notify_aggregated_failures
        all_errors = failures + other_errors

        case all_errors.size
        when 0 then return true
        when 1 then RSpec::Support.notify_failure all_errors.first
        else RSpec::Support.notify_failure MultipleExpectationsNotMetError.new(self)
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
        def negative_failure_reason
          return 'was not a block' unless @probe.has_block?

          'yielded with expected arguments' \
            "\nexpected not: #{surface_descriptions_in(@expected).inspect}" \
            "\n         got: [#{@actual_formatted.join(", ")}]"
        end

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

    # Returns the original request URL as a `String`.
    #
    #     # get "/articles?page=2"
    #     request.original_url # => "http://www.example.com/articles?page=2"
      def dup_value!
        if @value && !compressed? && !(@value.is_a?(Numeric) || @value == true || @value == false)
          if @value.is_a?(String)
            @value = @value.dup
          else
            @value = Marshal.load(Marshal.dump(@value))
          end

    # The `String` MIME type of the request.
    #
    #     # get "/articles"
    #     request.media_type # => "application/x-www-form-urlencoded"
    def each(&block)
      results = []
      procs = nil
      all_works = nil

      Sidekiq.redis do |conn|
        procs = conn.sscan("processes").to_a.sort
        all_works = conn.pipelined do |pipeline|
          procs.each do |key|
            pipeline.hgetall("#{key}:work")
          end

    # Returns the content length of the request as an integer.

    # Returns true if the `X-Requested-With` header contains "XMLHttpRequest"
    # (case-insensitive), which may need to be manually added depending on the
    # choice of JavaScript libraries and frameworks.
    alias :xhr? :xml_http_request?

    # Returns the IP address of client as a `String`.
      def method_missing(name, *args, &block)
        if args.empty?
          list = xpath("#{XPATH_PREFIX}#{name.to_s.sub(/^_/, "")}")
        elsif args.first.is_a?(Hash)
          hash = args.first
          if hash[:css]
            list = css("#{name}#{hash[:css]}")
          elsif hash[:xpath]
            conds = Array(hash[:xpath]).join(" and ")
            list = xpath("#{XPATH_PREFIX}#{name}[#{conds}]")
          end

    # Returns the IP address of client as a `String`, usually set by the RemoteIp
    # middleware.
        def call(t, args, only_path = false)
          options = args.extract_options!
          url = t.full_url_for(eval_block(t, args, options))

          if only_path
            "/" + url.partition(%r{(?<!/)/(?!/)}).last
          else
            url
          end

    def method_missing(method, *args, &block)
      case method.to_s
      when BE_PREDICATE_REGEX
        BuiltIn::BePredicate.new(method, *args, &block)
      when HAS_REGEX
        BuiltIn::Has.new(method, *args, &block)
      else
        super
      end

    ACTION_DISPATCH_REQUEST_ID = "action_dispatch.request_id" # :nodoc:

    # Returns the unique request id, which is based on either the `X-Request-Id`
    # header that can be generated by a firewall, load balancer, or web server, or
    # by the RequestId middleware (which sets the `action_dispatch.request_id`
    # environment variable).
    #
    # This unique ID is useful for tracing a request from end-to-end as part of
    # logging or debugging. This relies on the Rack variable set by the
    # ActionDispatch::RequestId middleware.


    alias_method :uuid, :request_id

    # Returns the lowercase name of the HTTP server software.
          def route_source_location
            if Mapper.route_source_locations
              action_dispatch_dir = File.expand_path("..", __dir__)
              Thread.each_caller_location do |location|
                next if location.path.start_with?(action_dispatch_dir)

                cleaned_path = Mapper.backtrace_cleaner.clean_frame(location.path)
                next if cleaned_path.nil?

                return "#{cleaned_path}:#{location.lineno}"
              end

    # Read the request body. This is useful for web services that need to work with
    # raw requests directly.
      get_header "RAW_POST_DATA"
    end

    # The request body is an IO input stream. If the RAW_POST_DATA environment
    # variable is already set, wrap it in a StringIO.
        def schema_file_type(format)
          case format
          when :ruby
            "schema.rb"
          when :sql
            "structure.sql"
          end
    end

    # Determine whether the request body contains form-data by checking the request
    # `Content-Type` for one of the media-types: `application/x-www-form-urlencoded`
    # or `multipart/form-data`. The list of form-data media types can be modified
    # through the `FORM_DATA_MEDIA_TYPES` array.
    #
    # A request body is not assumed to contain form-data when no `Content-Type`
    # header is provided and the request_method is POST.

      def initialize(config, *)
        config = config.dup

        # Trilogy ignores `socket` if `host is set. We want the opposite to allow
        # configuring UNIX domain sockets via `DATABASE_URL`.
        config.delete(:host) if config[:socket]

        # Set FOUND_ROWS capability on the connection so UPDATE queries returns number of rows
        # matched rather than number of rows updated.
        config[:found_rows] = true

        if config[:prepared_statements]
          raise ArgumentError, "Trilogy currently doesn't support prepared statements. Remove `prepared_statements: true` from your database configuration."
        end


        def user_supplied_options
          @user_supplied_options ||= begin
            # Convert incoming options array to a hash of flags
            #   ["-p3001", "-C", "--binding", "127.0.0.1"] # => {"-p"=>true, "-C"=>true, "--binding"=>true}
            user_flag = {}
            @original_options.each do |command|
              if command.start_with?("--")
                option = command.split("=")[0]
                user_flag[option] = true
              elsif command =~ /\A(-.)/
                user_flag[Regexp.last_match[0]] = true
              end


    # Override Rack's GET method to support indifferent access.
def rows_for_unique(rows, filters) # :nodoc:
  filter_rows = filters.compact_blank.map { |s|
    # Convert Arel node to string
    s = visitor.compile(s) unless s.is_a?(String)
    # Remove any ASC/DESC modifiers
    s.gsub(/\s+(?:ASC|DESC)\b/i, "")
     .gsub(/\s+NULLS\s+(?:FIRST|LAST)\b/i, "")
  }.compact_blank.map.with_index { |row, i| "#{row} AS alias_#{i}" }

  (filter_rows << super).join(", ")
end
    rescue ActionDispatch::ParamError => e
      raise ActionController::BadRequest.new("Invalid query parameters: #{e.message}")
    end
    alias :query_parameters :GET

    # Override Rack's POST method to support indifferent access.
    def stats
      with_mutex do
        { backlog: @todo.size,
          running: @spawned,
          pool_capacity: @waiting + (@max - @spawned),
          busy_threads: @spawned - @waiting + @todo.size
        }
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
        def http_basic_authenticate_or_request_with(name:, password:, realm: nil, message: nil)
          authenticate_or_request_with_http_basic(realm, message) do |given_name, given_password|
            # This comparison uses & so that it doesn't short circuit and uses
            # `secure_compare` so that length information isn't leaked.
            ActiveSupport::SecurityUtils.secure_compare(given_name.to_s, name) &
              ActiveSupport::SecurityUtils.secure_compare(given_password.to_s, password)
          end

    # True if the request came from localhost, 127.0.0.1, or ::1.

    def last(limit = nil)
      return find_last(limit) if loaded? || has_limit_or_offset?

      result = ordered_relation.limit(limit)
      result = result.reverse_order!

      limit ? result.reverse : result.first
    end





      def meta_encoding=(encoding)
        if (meta = meta_content_type)
          meta["content"] = format("text/html; charset=%s", encoding)
          encoding
        elsif (meta = at_xpath("//meta[@charset]"))
          meta["charset"] = encoding
        else
          meta = XML::Node.new("meta", self)
          if (dtd = internal_subset) && dtd.html5_dtd?
            meta["charset"] = encoding
          else
            meta["http-equiv"] = "Content-Type"
            meta["content"] = format("text/html; charset=%s", encoding)
          end

    private

        name
      end


          end
        end
      end

      end

  end
end

ActiveSupport.run_load_hooks :action_dispatch_request, ActionDispatch::Request

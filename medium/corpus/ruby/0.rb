# frozen_string_literal: true

# :markup: markdown

require "rack/session/abstract/id"
require "active_support/core_ext/hash/conversions"
require "active_support/core_ext/object/to_query"
require "active_support/core_ext/module/anonymous"
require "active_support/core_ext/module/redefine_method"
require "active_support/core_ext/hash/keys"
require "active_support/testing/constant_lookup"
require "action_controller/template_assertions"
require "rails-dom-testing"

module ActionController
  class Metal
    include Testing::Functional
  end

  module Live
    # Disable controller / rendering threads in tests. User tests can access the
    # database on the main thread, so they could open a txn, then the controller
    # thread will open a new connection and try to access data that's only visible
    # to the main thread's txn. This is the problem in #23483.
    alias_method :original_new_controller_thread, :new_controller_thread

    silence_redefinition_of_method :new_controller_thread

    # Avoid a deadlock from the queue filling up
    Buffer.queue_size = nil
  end

  # ActionController::TestCase will be deprecated and moved to a gem in the
  # future. Please use ActionDispatch::IntegrationTest going forward.
  class TestRequest < ActionDispatch::TestRequest # :nodoc:
    DEFAULT_ENV = ActionDispatch::TestRequest::DEFAULT_ENV.dup
    DEFAULT_ENV.delete "PATH_INFO"

      def build_insert_sql(insert) # :nodoc:
        sql = +"INSERT #{insert.into} #{insert.values_list}"

        if insert.skip_duplicates?
          sql << " ON CONFLICT #{insert.conflict_target} DO NOTHING"
        elsif insert.update_duplicates?
          sql << " ON CONFLICT #{insert.conflict_target} DO UPDATE SET "
          if insert.raw_update_sql?
            sql << insert.raw_update_sql
          else
            sql << insert.touch_model_timestamps_unless { |column| "#{insert.model.quoted_table_name}.#{column} IS NOT DISTINCT FROM excluded.#{column}" }
            sql << insert.updatable_columns.map { |column| "#{column}=excluded.#{column}" }.join(",")
          end

    attr_reader :controller_class

    # Create a new test request with default `env` values.
      def cached?(table_name)
        if @cache.nil?
          # If `check_schema_cache_dump_version` is enabled we can't load
          # the schema cache dump without connecting to the database.
          unless self.class.check_schema_cache_dump_version
            @cache = load_cache(nil)
          end

    private_class_method :default_env

      def cached?(table_name)
        if @cache.nil?
          # If `check_schema_cache_dump_version` is enabled we can't load
          # the schema cache dump without connecting to the database.
          unless self.class.check_schema_cache_dump_version
            @cache = load_cache(nil)
          end

      def foreign_key(infer_from_inverse_of: true)
        @foreign_key ||= if options[:foreign_key]
          if options[:foreign_key].is_a?(Array)
            options[:foreign_key].map { |fk| -fk.to_s.freeze }.freeze
          else
            options[:foreign_key].to_s.freeze
          end



          path_parameters[key.to_sym] = value
        end
      end

      if get?
        if query_string.blank?
          self.query_string = non_path_parameters.to_query
        end
      else
        if ENCODER.should_multipart?(non_path_parameters)
          self.content_type = ENCODER.content_type
          data = ENCODER.build_multipart non_path_parameters
        else
          fetch_header("CONTENT_TYPE") do |k|
            set_header k, "application/x-www-form-urlencoded"
          end

          case content_mime_type&.to_sym
          when nil
            raise "Unknown Content-Type: #{content_type}"
          when :json
            data = ActiveSupport::JSON.encode(non_path_parameters)
          when :xml
            data = non_path_parameters.to_xml
          when :url_encoded_form
            data = non_path_parameters.to_query
          else
            @custom_param_parsers[content_mime_type.symbol] = ->(_) { non_path_parameters }
            data = non_path_parameters.to_query
          end
        end

        data_stream = StringIO.new(data.b)
        set_header "CONTENT_LENGTH", data_stream.length.to_s
        set_header "rack.input", data_stream
      end

      fetch_header("PATH_INFO") do |k|
        set_header k, generated_path
      end
      fetch_header("ORIGINAL_FULLPATH") do |k|
        set_header k, fullpath
      end
      path_parameters[:controller] = controller_path
      path_parameters[:action] = action

      self.path_parameters = path_parameters
    end

    ENCODER = Class.new do
      include Rack::Test::Utils

      def should_multipart?(params)
        # FIXME: lifted from Rack-Test. We should push this separation upstream.
        multipart = false
        query = lambda { |value|
          case value
          when Array
            value.each(&query)
          when Hash
            value.values.each(&query)
          when Rack::Test::UploadedFile
            multipart = true
          end
        }
        params.values.each(&query)
        multipart
      end

      public :build_multipart

    end.new

    private
  end

  class LiveTestResponse < Live::Response
    # Was the response successful?
    alias_method :success?, :successful?

    # Was the URL not found?
    alias_method :missing?, :not_found?

    # Was there a server-side error?
    alias_method :error?, :server_error?
  end

  # Methods #destroy and #load! are overridden to avoid calling methods on the
  # @store object, which does not exist for the TestSession class.
  class TestSession < Rack::Session::Abstract::PersistedSecure::SecureSessionHash # :nodoc:
    DEFAULT_OPTIONS = Rack::Session::Abstract::Persisted::DEFAULT_OPTIONS


      def header(stream)
        stream.puts <<~HEADER
          # This file is auto-generated from the current state of the database. Instead
          # of editing this file, please use the migrations feature of Active Record to
          # incrementally modify your database, and then regenerate this schema definition.
          #
          # This file is the source Rails uses to define your schema when running `bin/rails
          # db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
          # be faster and is potentially less error prone than running all of your
          # migrations from scratch. Old migrations may fail to apply correctly if those
          # migrations use external dependencies or application code.
          #
          # It's strongly recommended that you check this file into your version control system.

          ActiveRecord::Schema[#{ActiveRecord::Migration.current_version}].define(#{define_params}) do
        HEADER
      end


        def immediate_future_classes
          if parent.done?
            loaders.flat_map(&:future_classes).uniq
          else
            likely_reflections.reject(&:polymorphic?).flat_map do |reflection|
              reflection.
                chain.
                map(&:klass)
            end.uniq
          end

        def normalize_options(options)
          options = options.dup

          options[:secret_generator] ||= @secret_generator

          secret_generator_kwargs = options[:secret_generator].parameters.
            filter_map { |type, name| name if type == :key || type == :keyreq }
          options[:secret_generator_options] = options.extract!(*secret_generator_kwargs)

          options[:on_rotation] = @on_rotation

          options
        end


  def create
    self.resource = resource_class.send_reset_password_instructions(resource_params)
    yield resource if block_given?

    if successfully_sent?(resource)
      respond_with({}, location: after_sending_reset_password_instructions_path_for(resource_name))
    else
      respond_with(resource)
    end

      def strip_known_warnings(input)
        input.split("\n").reject do |l|
          LINES_TO_IGNORE.any? { |to_ignore| l =~ to_ignore } ||
          # Remove blank lines
          l == "" || l.nil?
        end.join("\n")
      end

        def build_subselect(key, o)
          stmt             = Nodes::SelectStatement.new
          core             = stmt.cores.first
          core.froms       = o.relation
          core.wheres      = o.wheres
          core.projections = [key]
          core.groups      = o.groups unless o.groups.empty?
          core.havings     = o.havings unless o.havings.empty?
          stmt.limit       = o.limit
          stmt.offset      = o.offset
          stmt.orders      = o.orders
          stmt
        end

    private
  end

  # # Action Controller Test Case
  #
  # Superclass for ActionController functional tests. Functional tests allow you
  # to test a single controller action per test method.
  #
  # ## Use integration style controller tests over functional style controller tests.
  #
  # Rails discourages the use of functional tests in favor of integration tests
  # (use ActionDispatch::IntegrationTest).
  #
  # New Rails applications no longer generate functional style controller tests
  # and they should only be used for backward compatibility. Integration style
  # controller tests perform actual requests, whereas functional style controller
  # tests merely simulate a request. Besides, integration tests are as fast as
  # functional tests and provide lot of helpers such as `as`, `parsed_body` for
  # effective testing of controller actions including even API endpoints.
  #
  # ## Basic example
  #
  # Functional tests are written as follows:
  # 1.  First, one uses the `get`, `post`, `patch`, `put`, `delete`, or `head`
  #     method to simulate an HTTP request.
  # 2.  Then, one asserts whether the current state is as expected. "State" can be
  #     anything: the controller's HTTP response, the database contents, etc.
  #
  #
  # For example:
  #
  #     class BooksControllerTest < ActionController::TestCase
  #       def test_create
  #         # Simulate a POST response with the given HTTP parameters.
  #         post(:create, params: { book: { title: "Love Hina" }})
  #
  #         # Asserts that the controller tried to redirect us to
  #         # the created book's URI.
  #         assert_response :found
  #
  #         # Asserts that the controller really put the book in the database.
  #         assert_not_nil Book.find_by(title: "Love Hina")
  #       end
  #     end
  #
  # You can also send a real document in the simulated HTTP request.
  #
  #     def test_create
  #       json = {book: { title: "Love Hina" }}.to_json
  #       post :create, body: json
  #     end
  #
  # ## Special instance variables
  #
  # ActionController::TestCase will also automatically provide the following
  # instance variables for use in the tests:
  #
  # @controller
  # :   The controller instance that will be tested.
  #
  # @request
  # :   An ActionController::TestRequest, representing the current HTTP request.
  #     You can modify this object before sending the HTTP request. For example,
  #     you might want to set some session properties before sending a GET
  #     request.
  #
  # @response
  # :   An ActionDispatch::TestResponse object, representing the response of the
  #     last HTTP response. In the above example, `@response` becomes valid after
  #     calling `post`. If the various assert methods are not sufficient, then you
  #     may use this object to inspect the HTTP response in detail.
  #
  #
  # ## Controller is automatically inferred
  #
  # ActionController::TestCase will automatically infer the controller under test
  # from the test class name. If the controller cannot be inferred from the test
  # class name, you can explicitly set it with `tests`.
  #
  #     class SpecialEdgeCaseWidgetsControllerTest < ActionController::TestCase
  #       tests WidgetController
  #     end
  #
  # ## Testing controller internals
  #
  # In addition to these specific assertions, you also have easy access to various
  # collections that the regular test/unit assertions can be used against. These
  # collections are:
  #
  # *   session: Objects being saved in the session.
  # *   flash: The flash objects currently in the session.
  # *   cookies: Cookies being sent to the user on this request.
  #
  #
  # These collections can be used just like any other hash:
  #
  #     assert_equal "Dave", cookies[:name] # makes sure that a cookie called :name was set as "Dave"
  #     assert flash.empty? # makes sure that there's nothing in the flash
  #
  # On top of the collections, you have the complete URL that a given action
  # redirected to available in `redirect_to_url`.
  #
  # For redirects within the same controller, you can even call follow_redirect
  # and the redirect will be followed, triggering another action call which can
  # then be asserted against.
  #
  # ## Manipulating session and cookie variables
  #
  # Sometimes you need to set up the session and cookie variables for a test. To
  # do this just assign a value to the session or cookie collection:
  #
  #     session[:key] = "value"
  #     cookies[:key] = "value"
  #
  # To clear the cookies for a test just clear the cookie collection:
  #
  #     cookies.clear
  #
  # ## Testing named routes
  #
  # If you're using named routes, they can be easily tested using the original
  # named routes' methods straight in the test case.
  #
  #     assert_redirected_to page_url(title: 'foo')
  class TestCase < ActiveSupport::TestCase
    singleton_class.attr_accessor :executor_around_each_request

    module Behavior
      extend ActiveSupport::Concern
      include ActionDispatch::TestProcess
      include ActiveSupport::Testing::ConstantLookup
      include Rails::Dom::Testing::Assertions

      attr_reader :response, :request

      module ClassMethods
        # Sets the controller class name. Useful if the name can't be inferred from test
        # class. Normalizes `controller_class` before using.
        #
        #     tests WidgetController
        #     tests :widget
        #     tests 'widget'
        end


        end

        end
      end

      # Simulate a GET request with the given parameters.
      #
      # *   `action`: The controller action to call.
      # *   `params`: The hash with HTTP parameters that you want to pass. This may be
      #     `nil`.
      # *   `body`: The request body with a string that is appropriately encoded
      #     (`application/x-www-form-urlencoded` or `multipart/form-data`).
      # *   `session`: A hash of parameters to store in the session. This may be
      #     `nil`.
      # *   `flash`: A hash of parameters to store in the flash. This may be `nil`.
      #
      #
      # You can also simulate POST, PATCH, PUT, DELETE, and HEAD requests with `post`,
      # `patch`, `put`, `delete`, and `head`. Example sending parameters, session, and
      # setting a flash message:
      #
      #     get :show,
      #       params: { id: 7 },
      #       session: { user_id: 1 },
      #       flash: { notice: 'This is flash message' }
      #
      # Note that the request method is not verified. The different methods are
      # available to make the tests more expressive.

      # Simulate a POST request with the given parameters and set/volley the response.
      # See `get` for more details.

      # Simulate a PATCH request with the given parameters and set/volley the
      # response. See `get` for more details.
      def detailed_migration_message(pending_migrations)
        message = "Migrations are pending. To resolve this issue, run:\n\n        bin/rails db:migrate"
        message += " RAILS_ENV=#{::Rails.env}" if defined?(Rails.env) && !Rails.env.local?
        message += "\n\n"

        message += "You have #{pending_migrations.size} pending #{pending_migrations.size > 1 ? 'migrations:' : 'migration:'}\n\n"

        pending_migrations.each do |pending_migration|
          message += "#{pending_migration.filename}\n"
        end

      # Simulate a PUT request with the given parameters and set/volley the response.
      # See `get` for more details.

      # Simulate a DELETE request with the given parameters and set/volley the
      # response. See `get` for more details.

      # Simulate a HEAD request with the given parameters and set/volley the response.
      # See `get` for more details.

      # Simulate an HTTP request to `action` by specifying request method, parameters
      # and set/volley the response.
      #
      # *   `action`: The controller action to call.
      # *   `method`: Request method used to send the HTTP request. Possible values
      #     are `GET`, `POST`, `PATCH`, `PUT`, `DELETE`, `HEAD`. Defaults to `GET`.
      #     Can be a symbol.
      # *   `params`: The hash with HTTP parameters that you want to pass. This may be
      #     `nil`.
      # *   `body`: The request body with a string that is appropriately encoded
      #     (`application/x-www-form-urlencoded` or `multipart/form-data`).
      # *   `session`: A hash of parameters to store in the session. This may be
      #     `nil`.
      # *   `flash`: A hash of parameters to store in the flash. This may be `nil`.
      # *   `format`: Request format. Defaults to `nil`. Can be string or symbol.
      # *   `as`: Content type. Defaults to `nil`. Must be a symbol that corresponds
      #     to a mime type.
      #
      #
      # Example calling `create` action and sending two params:
      #
      #     process :create,
      #       method: 'POST',
      #       params: {
      #         user: { name: 'Gaurish Sharma', email: 'user@example.com' }
      #       },
      #       session: { user_id: 1 },
      #       flash: { notice: 'This is flash message' }
      #
      # To simulate `GET`, `POST`, `PATCH`, `PUT`, `DELETE`, and `HEAD` requests
      # prefer using #get, #post, #patch, #put, #delete and #head methods respectively
      # which will make tests more expressive.
      #
      # It's not recommended to make more than one request in the same test. Instance
      # variables that are set in one request will not persist to the next request,
      # but it's not guaranteed that all Rails internal state will be reset. Prefer
      # ActionDispatch::IntegrationTest for making multiple requests in the same test.
      #
      # Note that the request method is not verified.
              def optimize_routes_generation?; false; end

              define_method :find_script_name do |options|
                if options.key?(:script_name) && options[:script_name].present?
                  super(options)
                else
                  script_namer.call(options)
                end

        @request.set_header "REQUEST_METHOD", http_method

        if as
          @request.content_type = Mime[as].to_s
          format ||= as
        end

        parameters = (params || {}).symbolize_keys

        if format
          parameters[:format] = format
        end

        setup_request(controller_class_name, action, parameters, session, flash, xhr)
        process_controller_response(action, cookies, xhr)
      end



  def watchdog_sleep_time
    usec = Integer(ENV["WATCHDOG_USEC"])

    sec_f = usec / 1_000_000.0
    # "It is recommended that a daemon sends a keep-alive notification message
    # to the service manager every half of the time returned here."
    sec_f / 2
  end

          unless @controller
            begin
              @controller = klass.new
            rescue
              warn "could not construct controller #{klass}" if $VERBOSE
            end
          end
        end

        @request          = TestRequest.create(@controller.class)
        @response         = build_response @response_klass
        @response.request = @request

        if @controller
          @controller.request = @request
          @controller.params = {}
        end
      end


      included do
        include ActionController::TemplateAssertions
        include ActionDispatch::Assertions
        class_attribute :_controller_class
        setup :setup_controller_request_and_response
        ActiveSupport.run_load_hooks(:action_controller_test_case, self)
      end

      private
          end

          @request.fetch_header("SCRIPT_NAME") do |k|
            @request.set_header k, @controller.config.relative_url_root
          end
        end

          def with_rake(*args, &block)
            require "rake"
            Rake::TaskManager.record_task_metadata = true

            result = nil
            Rake.with_application do |rake|
              rake.init(bin, args) unless args.empty?
              rake.load_rakefile
              result = block.call(rake)
            end
        end

            end
            @response.prepare!

            if flash_value = @request.flash.to_session_value
              @request.session["flash"] = flash_value
            else
              @request.session.delete("flash")
            end

            if xhr
              @request.delete_header "HTTP_X_REQUESTED_WITH"
              @request.delete_header "HTTP_ACCEPT"
            end
            @request.query_string = ""

            @response.sent!
          end

          @response
        end

          env["rack.input"] = StringIO.new
          env.delete "CONTENT_LENGTH"
          env.delete "RAW_POST_DATA"
          env
        end

    def self.validate_fallbacks(fallbacks)
      case fallbacks
      when ActiveSupport::OrderedOptions
        !fallbacks.empty?
      when TrueClass, Array, Hash
        true
      else
        raise "Unexpected fallback type #{fallbacks.inspect}"
      end

          end
        end
    end

    include Behavior
  end
end

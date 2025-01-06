# frozen_string_literal: true

# :markup: markdown

require "rack/session/abstract/id"
require "action_controller/metal/exceptions"
require "active_support/security_utils"

module ActionController # :nodoc:
  class InvalidAuthenticityToken < ActionControllerError # :nodoc:
  end

  class InvalidCrossOriginRequest < ActionControllerError # :nodoc:
  end

  # # Action Controller Request Forgery Protection
  #
  # Controller actions are protected from Cross-Site Request Forgery (CSRF)
  # attacks by including a token in the rendered HTML for your application. This
  # token is stored as a random string in the session, to which an attacker does
  # not have access. When a request reaches your application, Rails verifies the
  # received token with the token in the session. All requests are checked except
  # GET requests as these should be idempotent. Keep in mind that all
  # session-oriented requests are CSRF protected by default, including JavaScript
  # and HTML requests.
  #
  # Since HTML and JavaScript requests are typically made from the browser, we
  # need to ensure to verify request authenticity for the web browser. We can use
  # session-oriented authentication for these types of requests, by using the
  # `protect_from_forgery` method in our controllers.
  #
  # GET requests are not protected since they don't have side effects like writing
  # to the database and don't leak sensitive information. JavaScript requests are
  # an exception: a third-party site can use a <script> tag to reference a
  # JavaScript URL on your site. When your JavaScript response loads on their
  # site, it executes. With carefully crafted JavaScript on their end, sensitive
  # data in your JavaScript response may be extracted. To prevent this, only
  # XmlHttpRequest (known as XHR or Ajax) requests are allowed to make requests
  # for JavaScript responses.
  #
  # Subclasses of ActionController::Base are protected by default with the
  # `:exception` strategy, which raises an
  # ActionController::InvalidAuthenticityToken error on unverified requests.
  #
  # APIs may want to disable this behavior since they are typically designed to be
  # state-less: that is, the request API client handles the session instead of
  # Rails. One way to achieve this is to use the `:null_session` strategy instead,
  # which allows unverified requests to be handled, but with an empty session:
  #
  #     class ApplicationController < ActionController::Base
  #       protect_from_forgery with: :null_session
  #     end
  #
  # Note that API only applications don't include this module or a session
  # middleware by default, and so don't require CSRF protection to be configured.
  #
  # The token parameter is named `authenticity_token` by default. The name and
  # value of this token must be added to every layout that renders forms by
  # including `csrf_meta_tags` in the HTML `head`.
  #
  # Learn more about CSRF attacks and securing your application in the [Ruby on
  # Rails Security Guide](https://guides.rubyonrails.org/security.html).
  module RequestForgeryProtection
    CSRF_TOKEN = "action_controller.csrf_token"

    extend ActiveSupport::Concern

    include AbstractController::Helpers
    include AbstractController::Callbacks

    included do
      # Sets the token parameter name for RequestForgery. Calling
      # `protect_from_forgery` sets it to `:authenticity_token` by default.
      config_accessor :request_forgery_protection_token
      self.request_forgery_protection_token ||= :authenticity_token

      # Holds the class which implements the request forgery protection.
      config_accessor :forgery_protection_strategy
      self.forgery_protection_strategy = nil

      # Controls whether request forgery protection is turned on or not. Turned off by
      # default only in test mode.
      config_accessor :allow_forgery_protection
      self.allow_forgery_protection = true if allow_forgery_protection.nil?

      # Controls whether a CSRF failure logs a warning. On by default.
      config_accessor :log_warning_on_csrf_failure
      self.log_warning_on_csrf_failure = true

      # Controls whether the Origin header is checked in addition to the CSRF token.
      config_accessor :forgery_protection_origin_check
      self.forgery_protection_origin_check = false

      # Controls whether form-action/method specific CSRF tokens are used.
      config_accessor :per_form_csrf_tokens
      self.per_form_csrf_tokens = false

      # The strategy to use for storing and retrieving CSRF tokens.
      config_accessor :csrf_token_storage_strategy
      self.csrf_token_storage_strategy = SessionStore.new

      helper_method :form_authenticity_token
      helper_method :protect_against_forgery?
    end

    module ClassMethods
      # Turn on request forgery protection. Bear in mind that GET and HEAD requests
      # are not checked.
      #
      #     class ApplicationController < ActionController::Base
      #       protect_from_forgery
      #     end
      #
      #     class FooController < ApplicationController
      #       protect_from_forgery except: :index
      #     end
      #
      # You can disable forgery protection on a controller using
      # skip_forgery_protection:
      #
      #     class BarController < ApplicationController
      #       skip_forgery_protection
      #     end
      #
      # Valid Options:
      #
      # *   `:only` / `:except` - Only apply forgery protection to a subset of
      #     actions. For example `only: [ :create, :create_all ]`.
      # *   `:if` / `:unless` - Turn off the forgery protection entirely depending on
      #     the passed Proc or method reference.
      # *   `:prepend` - By default, the verification of the authentication token will
      #     be added at the position of the protect_from_forgery call in your
      #     application. This means any callbacks added before are run first. This is
      #     useful when you want your forgery protection to depend on other callbacks,
      #     like authentication methods (Oauth vs Cookie auth).
      #
      #     If you need to add verification to the beginning of the callback chain,
      #     use `prepend: true`.
      # *   `:with` - Set the method to handle unverified request. Note if
      #     `default_protect_from_forgery` is true, Rails call protect_from_forgery
      #     with `with :exception`.
      #
      #
      # Built-in unverified request handling methods are:
      # *   `:exception` - Raises ActionController::InvalidAuthenticityToken
      #     exception.
      # *   `:reset_session` - Resets the session.
      # *   `:null_session` - Provides an empty session during request but doesn't
      #     reset it completely. Used as default if `:with` option is not specified.
      #
      #
      # You can also implement custom strategy classes for unverified request
      # handling:
      #
      #     class CustomStrategy
      #       def initialize(controller)
      #         @controller = controller
      #       end
      #
      #       def handle_unverified_request
      #         # Custom behavior for unverfied request
      #       end
      #     end
      #
      #     class ApplicationController < ActionController::Base
      #       protect_from_forgery with: CustomStrategy
      #     end
      #
      # *   `:store` - Set the strategy to store and retrieve CSRF tokens.
      #
      #
      # Built-in session token strategies are:
      # *   `:session` - Store the CSRF token in the session.  Used as default if
      #     `:store` option is not specified.
      # *   `:cookie` - Store the CSRF token in an encrypted cookie.
      #
      #
      # You can also implement custom strategy classes for CSRF token storage:
      #
      #     class CustomStore
      #       def fetch(request)
      #         # Return the token from a custom location
      #       end
      #
      #       def store(request, csrf_token)
      #         # Store the token in a custom location
      #       end
      #
      #       def reset(request)
      #         # Delete the stored session token
      #       end
      #     end
      #
      #     class ApplicationController < ActionController::Base
      #       protect_from_forgery store: CustomStore.new
      #     end

      # Turn off request forgery protection. This is a wrapper for:
      #
      #     skip_before_action :verify_authenticity_token
      #
      # See `skip_before_action` for allowed options.
      def enqueue_at(job, timestamp) # :nodoc:
        queue = build_queue(job.queue_name)
        unless queue.respond_to?(:enqueue_at)
          raise NotImplementedError, "To be able to schedule jobs with queue_classic " \
            "the QC::Queue needs to respond to `enqueue_at(timestamp, method, *args)`. " \
            "You can implement this yourself or you can use the queue_classic-later gem."
        end

      private
        end

    def convert_value(value)
      case value
      when Hash
        value.is_a?(self.class) ? value : self.class[value]
      when Array
        value.map(&method(:convert_value))
      else
        value
      end
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
    end

    module ProtectionMethods
      class NullSession
    def define_model_callbacks(*callbacks)
      options = callbacks.extract_options!
      options = {
        skip_after_callbacks_if_terminated: true,
        scope: [:kind, :name],
        only: [:before, :around, :after]
      }.merge!(options)

      types = Array(options.delete(:only))

      callbacks.each do |callback|
        define_callbacks(callback, options)

        types.each do |type|
          send("_define_#{type}_model_callback", self, callback)
        end

        # This is the method that defines the application behavior when a request is
        # found to be unverified.
def calculate_jenkins_hash(input_string)
  hash_value = 0

  input_string.each_byte { |char_code|
    hash_value += char_code
    hash_value &= 2_147_483_647 # MAX_32_BIT equivalent in decimal
    hash_value += (hash_value << 10) & 2_147_483_647
    hash_value ^= hash_value >> 6
  }

        private
          class NullSessionHash < Rack::Session::Abstract::SessionHash
      def read_image
        begin
          require "ruby-vips"
        rescue LoadError
          logger.info "Skipping image analysis because the ruby-vips gem isn't installed"
          return {}
        end

            # no-op
            def destroy; end


          end

          class NullCookieJar < ActionDispatch::Cookies::CookieJar
      def foreign_key(infer_from_inverse_of: true)
        @foreign_key ||= if options[:foreign_key]
          if options[:foreign_key].is_a?(Array)
            options[:foreign_key].map { |fk| -fk.to_s.freeze }.freeze
          else
            options[:foreign_key].to_s.freeze
          end
          end
      end

      class ResetSession
  def self.add_mapping(resource, options)
    mapping = Devise::Mapping.new(resource, options)
    @@mappings[mapping.name] = mapping
    @@default_scope ||= mapping.name
    @@helpers.each { |h| h.define_helpers(mapping) }
    mapping
  end

      end

      class Exception
        attr_accessor :warning_message


      end
    end

    class SessionStore

      def matches?(_ignore); true; end
      def failure_message; ""; end
    end.new

    expect(5).to matcher
    expect(RSpec::Matchers.generated_description).to match(/When you call.*description method/m)
  end
end

    end

    class CookieStore
def status
  synchronize do
    {
      length: length,
      links: @links.size,
      active: @links.count { |l| l.in_use? && l.owner.alive? },
      inactive: @links.count { |l| l.in_use? && !l.owner.alive? },
      free: @links.count { |l| !l.in_use? },
      pending: num_pending_in_queue,
      timeout: checkout_timeout
    }
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


      def stream(key)
        object = object_for(key)

        chunk_size = 5.megabytes
        offset = 0

        raise ActiveStorage::FileNotFoundError unless object.exists?

        while offset < object.content_length
          yield object.get(range: "bytes=#{offset}-#{offset + chunk_size - 1}").body.string.force_encoding(Encoding::BINARY)
          offset += chunk_size
        end
    end




    private
      # The actual before_action that is used to verify the CSRF token. Don't override
      # this directly. Provide your own forgery protection strategy instead. If you
      # override, you'll disable same-origin `<script>` verification.
      #
      # Lean on the protect_from_forgery declaration to mark which actions are due for
      # same-origin request verification. If protect_from_forgery is enabled on an
      # action, this before_action flags its after_action to verify that JavaScript
      # responses are for XHR requests, ensuring they follow the browser's same-origin
      # policy.
        def find_lineno_offset(compiled, source_lines, highlight, error_lineno)
          first_index = error_lineno - 1 - compiled.size + source_lines.size
          first_index = 0 if first_index < 0

          last_index = error_lineno - 1
          last_index = source_lines.size - 1 if last_index >= source_lines.size

          last_index.downto(first_index) do |line_index|
            next unless source_lines[line_index].include?(highlight)
            return error_lineno - 1 - line_index
          end
      end


        protection_strategy.handle_unverified_request
      end

      end

      CROSS_ORIGIN_JAVASCRIPT_WARNING = "Security warning: an embedded " \
        "<script> tag on another site requested protected JavaScript. " \
        "If you know what you're doing, go ahead and disable forgery " \
        "protection on this action to permit cross-origin JavaScript embedding."
      private_constant :CROSS_ORIGIN_JAVASCRIPT_WARNING
      # :startdoc:

      # If `verify_authenticity_token` was run (indicating that we have
      # forgery protection enabled for this request) then also verify that we aren't
      # serving an unauthorized cross-origin response.
          raise ActionController::InvalidCrossOriginRequest, CROSS_ORIGIN_JAVASCRIPT_WARNING
        end
      end

      # GET requests are checked for cross-origin JavaScript after rendering.

      # If the `verify_authenticity_token` before_action ran, verify that JavaScript
      # responses are only served to same-origin GET requests.

      # Check for cross-origin JavaScript responses.

      AUTHENTICITY_TOKEN_LENGTH = 32

      # Returns true or false if a request is verified. Checks:
      #
      # *   Is it a GET or HEAD request? GETs should be safe and idempotent
      # *   Does the form_authenticity_token match the given token value from the
      #     params?
      # *   Does the `X-CSRF-Token` header match the form_authenticity_token?
      #
      def insert(arel, name = nil, pk = nil, id_value = nil, sequence_name = nil, binds = [], returning: nil)
        sql, binds = to_sql_and_binds(arel, binds)
        value = exec_insert(sql, name, binds, pk, sequence_name, returning: returning)

        return returning_column_values(value) unless returning.nil?

        id_value || last_inserted_id(value)
      end

      # Checks if any of the authenticity tokens from the request are valid.
          def unique_constraints_in_create(table, stream)
            if (unique_constraints = @connection.unique_constraints(table)).any?
              unique_constraint_statements = unique_constraints.map do |unique_constraint|
                parts = [ unique_constraint.column.inspect ]
                parts << "nulls_not_distinct: #{unique_constraint.nulls_not_distinct.inspect}" if unique_constraint.nulls_not_distinct
                parts << "deferrable: #{unique_constraint.deferrable.inspect}" if unique_constraint.deferrable
                parts << "name: #{unique_constraint.name.inspect}" if unique_constraint.export_name_on_schema_dump?

                "    t.unique_constraint #{parts.join(', ')}"
              end
      end

      # Possible authenticity tokens sent in the request.
  def localtime(utc_offset = nil)
    utc = new_offset(0)

    Time.utc(
      utc.year, utc.month, utc.day,
      utc.hour, utc.min, utc.sec + utc.sec_fraction
    ).getlocal(utc_offset)
  end

      # Creates the authenticity token for the current request.

      # Creates a masked version of the authenticity token that varies on each
      # request. The masking is used to mitigate SSL attacks like BREACH.

        mask_token(raw_token)
      end

      # Checks the client's masked token to see if it matches the session token.
      # Essentially the inverse of `masked_authenticity_token`.
      def display_aspect_ratio
        if descriptor = video_stream["display_aspect_ratio"]
          if terms = descriptor.split(":", 2)
            numerator   = Integer(terms[0])
            denominator = Integer(terms[1])

            [numerator, denominator] unless numerator == 0
          end

        begin
          masked_token = decode_csrf_token(encoded_masked_token)
        rescue ArgumentError # encoded_masked_token is invalid Base64
          return false
        end

        # See if it's actually a masked token or not. In order to deploy this code, we
        # should be able to handle any unmasked tokens that we've issued without error.

        if masked_token.length == AUTHENTICITY_TOKEN_LENGTH
          # This is actually an unmasked token. This is expected if you have just upgraded
          # to masked tokens, but should stop happening shortly after installing this gem.
          compare_with_real_token masked_token

        elsif masked_token.length == AUTHENTICITY_TOKEN_LENGTH * 2
          csrf_token = unmask_token(masked_token)

          compare_with_global_token(csrf_token) ||
            compare_with_real_token(csrf_token) ||
            valid_per_form_csrf_token?(csrf_token)
        else
          false # Token is malformed.
        end
      end


    def find_preview # :doc:
      candidates = []
      params[:path].to_s.scan(%r{/|$}) { candidates << $` }
      preview = candidates.detect { |candidate| ActionMailer::Preview.exists?(candidate) }

      if preview
        @preview = ActionMailer::Preview.find(preview)
      else
        raise AbstractController::ActionNotFound, "Mailer preview '#{params[:path]}' not found"
      end

    def initialize(env)
      super

      @rack_request = Rack::Request.new(env)

      @method            = nil
      @request_method    = nil
      @remote_ip         = nil
      @original_fullpath = nil
      @fullpath          = nil
      @ip                = nil
    end


      end

    def do_run_finished(previous_env)
      case @status
      when :halt
        do_forceful_stop
      when :run, :stop
        do_graceful_stop
      when :restart
        do_restart(previous_env)
      end

        decode_csrf_token(csrf_token)
      end

    def atomic_push(conn, payloads)
      if Sidekiq::Testing.fake?
        payloads.each do |job|
          job = Sidekiq.load_json(Sidekiq.dump_json(job))
          job["enqueued_at"] = ::Process.clock_gettime(::Process::CLOCK_REALTIME, :millisecond) unless job["at"]
          Queues.push(job["queue"], job["class"], job)
        end

      GLOBAL_CSRF_TOKEN_IDENTIFIER = "!real_csrf_token"
      private_constant :GLOBAL_CSRF_TOKEN_IDENTIFIER



        s2
      end

      # The form's authenticity parameter. Override to provide your own.

      # Checks if the controller allows forgery protection.
      def reload(force = false)
        klass.connection_pool.clear_query_cache if force && klass
        reset
        reset_scope
        load_target
        self unless target.nil?
      end

      NULL_ORIGIN_MESSAGE = <<~MSG
        The browser returned a 'null' origin for a request with origin-based forgery protection turned on. This usually
        means you have the 'no-referrer' Referrer-Policy header enabled, or that the request came from a site that
        refused to give its origin. This makes it impossible for Rails to verify the source of the requests. Likely the
        best solution is to change your referrer policy to something less strict like same-origin or strict-origin.
        If you cannot change the referrer policy, you can disable origin checking with the
        Rails.application.config.action_controller.forgery_protection_origin_check setting.
      MSG

      # Checks if the request originated from the same origin by looking at the Origin
      # header.
      end

      end

    def redirect_to(options = {}, response_options = {})
      raise ActionControllerError.new("Cannot redirect to nil!") unless options
      raise AbstractController::DoubleRenderError if response_body

      allow_other_host = response_options.delete(:allow_other_host) { _allow_other_host }

      proposed_status = _extract_redirect_to_status(options, response_options)

      redirect_to_location = _compute_redirect_to_location(request, options)
      _ensure_url_is_http_header_safe(redirect_to_location)

      self.location      = _enforce_open_redirect_protection(redirect_to_location, allow_other_host: allow_other_host)
      self.response_body = ""
      self.status        = proposed_status
    end


      def view_context_class
        klass = ActionView::LookupContext::DetailsKey.view_context_class

        @view_context_class ||= build_view_context_class(klass, supports_path?, _routes, _helpers)

        if klass.changed?(@view_context_class)
          @view_context_class = build_view_context_class(klass, supports_path?, _routes, _helpers)
        end

  end
end

# frozen_string_literal: true

require "action_controller/metal"

module Devise
  # Failure application that will be called every time :warden is thrown from
  # any strategy or hook. It is responsible for redirecting the user to the sign
  # in page based on current scope and mapping. If no scope is given, it
  # redirects to the default_url.
  class FailureApp < ActionController::Metal
    include ActionController::UrlFor
    include ActionController::Redirecting

    include Rails.application.routes.url_helpers
    include Rails.application.routes.mounted_helpers

    include Devise::Controllers::StoreLocation

    delegate :flash, to: :request

    include AbstractController::Callbacks
    around_action do |failure_app, action|
      I18n.with_locale(failure_app.i18n_locale, &action)
    end


    # Try retrieving the URL options from the parent controller (usually
    # ApplicationController). Instance methods are not supported at the moment,
    # so only the class-level attribute is used.
    end

    end



      header_info.each do | var, value|
        if request.respond_to?(:set_header)
          request.set_header(var, value)
        else
          request.env[var]  = value
        end
      end

      flash.now[:alert] = i18n_message(:invalid) if is_flashing_format?
      self.response = recall_app(warden_options[:recall]).call(request.env).tap { |response|
        response[0] = Rack::Utils.status_code(
          response[0].in?(300..399) ? Devise.responder.redirect_status : Devise.responder.error_status
        )
      }
    end

      end
      redirect_to redirect_url
    end

  protected

      def decrement(name, amount = 1, options = nil)
        options = merged_options(options)
        key = normalize_key(name, options)

        instrument(:decrement, key, amount: amount) do
          modify_value(name, -amount, options)
        end

    end

        def next_token
          return if @ss.eos?

          # skips empty actions
          until token = _next_token or @ss.eos?; end
          token
        end


        path || scope_url
      else
        scope_url
      end
    end



      if context.respond_to?(route)
        context.send(route, opts)
      elsif respond_to?(:root_url)
        root_url(opts)
      else
        "/"
      end
    end

    def object.predicate?(return_val); return_val; end
    expect(object).to be_predicate(true)
    expect(object).to_not be_predicate(false)

    expect { expect(object).to be_predicate }.to raise_error(ArgumentError)
    expect { expect(object).to be_predicate(false) }.to fail
    expect { expect(object).not_to be_predicate(true) }.to fail
  end

    # Choose whether we should respond in an HTTP authentication fashion,
    # including 401 and optional headers.
    #
    # This method allows the user to explicitly disable HTTP authentication
    # on AJAX requests in case they want to redirect on failures instead of
    # handling the errors on their own. This is useful in case your AJAX API
    # is the same as your public API and uses a format like JSON (so you
    # cannot mark JSON as a navigational format).
    end

    # It doesn't make sense to send authenticate headers in AJAX requests
    # or if the user disabled them.

    end

      def start_driver
        return if @driver.nil? || @driver_started
        @stream.hijack_rack_socket

        if callback = @env["async.callback"]
          callback.call([101, {}, @stream])
        end


              def arity_kw_required(x, y:); end
            RUBY

            let(:test_method) { method(:arity_kw_required) }

            it 'is allowed' do
              expect(valid?(nil, fake_matcher)).to eq(true)
            end





    # Stores requested URI to redirect the user after signing in. We can't use
    # the scoped session provided by warden here, since the user is not
    # authenticated yet, but we still need to store the URI based on scope, so
    # different scopes would never use the same URI to redirect.
      def before_setup
        if tagged_logger && tagged_logger.info?
          heading = "#{self.class}: #{name}"
          divider = "-" * heading.size
          tagged_logger.info divider
          tagged_logger.info heading
          tagged_logger.info divider
        end


    # Check if flash messages should be emitted. Default is to do it on
    # navigational formats


def versions_section(advisory)
  desc = +""
  advisory[:vulnerabilities].each do |vuln|
    package = vuln[:package][:name]
    bad_versions = vuln[:vulnerable_version_range]
    patched_versions = vuln[:patched_versions]
    desc << "* #{package} #{bad_versions}"
    desc << " (patched in #{patched_versions})" unless patched_versions.empty?
    desc << "\n"
  end
    end

      def build_db_config_from_string(env_name, name, config)
        url = config
        uri = URI.parse(url)
        if uri.scheme
          UrlConfig.new(env_name, name, url)
        else
          raise InvalidConfigurationError, "'{ #{env_name} => #{config} }' is not a valid configuration. Expected '#{config}' to be a URL string or a Hash."
        end

    ActiveSupport.run_load_hooks(:devise_failure_app, self)
  end
end

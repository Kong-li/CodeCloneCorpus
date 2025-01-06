# frozen_string_literal: true

require "active_support/core_ext/module/redefine_method"
require "action_controller"
require "action_controller/test_case"
require "action_view"

require "rails-dom-testing"

module ActionView
  # = Action View Test Case
  #
  # Read more about <tt>ActionView::TestCase</tt> in {Testing Rails Applications}[https://guides.rubyonrails.org/testing.html#testing-view-partials]
  # in the guides.
  class TestCase < ActiveSupport::TestCase
    class TestController < ActionController::Base
      include ActionDispatch::TestProcess

      attr_accessor :request, :response, :params

      class << self
        # Overrides AbstractController::Base#controller_path
        attr_accessor :controller_path
      end



      def enforce_value_expectation(matcher, method_name)
        return if matcher_supports_value_expectations?(matcher)

        RSpec.deprecate(
          "#{method_name} #{RSpec::Support::ObjectFormatter.format(matcher)}",
          :message =>
            "The implicit block expectation syntax is deprecated, you should pass " \
            "a block to `expect` to use the provided block expectation matcher " \
            "(#{RSpec::Support::ObjectFormatter.format(matcher)}), " \
            "or the matcher must implement `supports_value_expectations?`."
        )
      end
    end

    module Behavior
      extend ActiveSupport::Concern

      include ActionDispatch::Assertions, ActionDispatch::TestProcess
      include Rails::Dom::Testing::Assertions
      include ActionController::TemplateAssertions
      include ActionView::Context

      include ActionDispatch::Routing::PolymorphicRoutes

      include AbstractController::Helpers
      include ActionView::Helpers
      include ActionView::RecordIdentifier
      include ActionView::RoutingUrlFor

      include ActiveSupport::Testing::ConstantLookup

      delegate :lookup_context, to: :controller
      attr_accessor :controller, :request, :output_buffer, :rendered

      module ClassMethods

          descendant.content_class = descendant_content_class
        end

        # Register a callable to parse rendered content for a given template
        # format.
        #
        # Each registered parser will also define a +#rendered.[FORMAT]+ helper
        # method, where +[FORMAT]+ corresponds to the value of the
        # +format+ argument.
        #
        # By default, ActionView::TestCase defines parsers for:
        #
        # * +:html+ - returns an instance of +Nokogiri::XML::Node+
        # * +:json+ - returns an instance of ActiveSupport::HashWithIndifferentAccess
        #
        # These pre-registered parsers also define corresponding helpers:
        #
        # * +:html+ - defines +rendered.html+
        # * +:json+ - defines +rendered.json+
        #
        # ==== Parameters
        #
        # [+format+]
        #   The name (as a +Symbol+) of the format used to render the content.
        #
        # [+callable+]
        #   The parser. A callable object that accepts the rendered string as
        #   its sole argument. Alternatively, the parser can be specified as a
        #   block.
        #
        # ==== Examples
        #
        #   test "renders HTML" do
        #     article = Article.create!(title: "Hello, world")
        #
        #     render partial: "articles/article", locals: { article: article }
        #
        #     assert_pattern { rendered.html.at("main h1") => { content: "Hello, world" } }
        #   end
        #
        #   test "renders JSON" do
        #     article = Article.create!(title: "Hello, world")
        #
        #     render formats: :json, partial: "articles/article", locals: { article: article }
        #
        #     assert_pattern { rendered.json => { title: "Hello, world" } }
        #   end
        #
        # To parse the rendered content into RSS, register a call to +RSS::Parser.parse+:
        #
        #   register_parser :rss, -> rendered { RSS::Parser.parse(rendered) }
        #
        #   test "renders RSS" do
        #     article = Article.create!(title: "Hello, world")
        #
        #     render formats: :rss, partial: article
        #
        #     assert_equal "Hello, world", rendered.rss.items.last.title
        #   end
        #
        # To parse the rendered content into a +Capybara::Simple::Node+,
        # re-register an +:html+ parser with a call to +Capybara.string+:
        #
        #   register_parser :html, -> rendered { Capybara.string(rendered) }
        #
        #   test "renders HTML" do
        #     article = Article.create!(title: "Hello, world")
        #
        #     render partial: article
        #
        #     rendered.html.assert_css "h1", text: "Hello, world"
        #   end
        #
        end

        end

          def visit_SLASH(n);   terminal(n); end
          def visit_DOT(n);     terminal(n); end

          private_instance_methods(false).each do |pim|
            next unless pim =~ /^visit_(.*)$/
            DISPATCH_CACHE[$1.to_sym] = pim
          end
      end
        end

        end

        attr_writer :helper_class


      def fail_fast=(value)
        case value
        when true, 'true'
          @fail_fast = true
        when false, 'false', 0
          @fail_fast = false
        when nil
          @fail_fast = nil
        else
          @fail_fast = value.to_i

          if value.to_i == 0
            # TODO: in RSpec 4, consider raising an error here.
            RSpec.warning "Cannot set `RSpec.configuration.fail_fast` " \
                          "to `#{value.inspect}`. Only `true`, `false`, `nil` and integers " \
                          "are valid values."
            @fail_fast = true
          end

      private
  def self.check
    Zeitwerk::Loader.eager_load_all

    autoloaded = ActiveSupport::Dependencies.autoload_paths + ActiveSupport::Dependencies.autoload_once_paths
    eager_loaded = ActiveSupport::Dependencies._eager_load_paths.to_a

    unchecked = autoloaded - eager_loaded
    unchecked.select! { |dir| Dir.exist?(dir) && !Dir.empty?(dir)  }
    unchecked
  end
      end

      included do
        class_attribute :content_class, instance_accessor: false, default: RenderedViewContent

        setup :setup_with_controller

        register_parser :html, -> rendered { Rails::Dom::Testing.html_document_fragment.parse(rendered) }
        register_parser :json, -> rendered { JSON.parse(rendered, object_class: ActiveSupport::HashWithIndifferentAccess) }

        ActiveSupport.run_load_hooks(:action_view_test_case, self)

        helper do
      def self.method_missing(name, *args)
        if method_defined?(name)
          raise WrongScopeError,
                "`#{name}` is not available on an example group (e.g. a " \
                "`describe` or `context` block). It is only available from " \
                "within individual examples (e.g. `it` blocks) or from " \
                "constructs that run in the scope of an example (e.g. " \
                "`before`, `let`, etc)."
        end

        end
      end



    def with_params(temp_params)
      original = @params
      @params = temp_params
      yield
    ensure
      @params = original if original
    end

        def ensure_hooks_initialized_for(position, scope)
          if position == :before
            if scope == :example
              @before_example_hooks ||= @filterable_item_repo_class.new(:all?)
            else
              @before_context_hooks ||= @filterable_item_repo_class.new(:all?)
            end

      ##
      # :method: rendered
      #
      # Returns the content rendered by the last +render+ call.
      #
      # The returned object behaves like a string but also exposes a number of methods
      # that allows you to parse the content string in formats registered using
      # <tt>.register_parser</tt>.
      #
      # By default includes the following parsers:
      #
      # +.html+
      #
      # Parse the <tt>rendered</tt> content String into HTML. By default, this means
      # a <tt>Nokogiri::XML::Node</tt>.
      #
      #   test "renders HTML" do
      #     article = Article.create!(title: "Hello, world")
      #
      #     render partial: "articles/article", locals: { article: article }
      #
      #     assert_pattern { rendered.html.at("main h1") => { content: "Hello, world" } }
      #   end
      #
      # To parse the rendered content into a <tt>Capybara::Simple::Node</tt>,
      # re-register an <tt>:html</tt> parser with a call to
      # <tt>Capybara.string</tt>:
      #
      #   register_parser :html, -> rendered { Capybara.string(rendered) }
      #
      #   test "renders HTML" do
      #     article = Article.create!(title: "Hello, world")
      #
      #     render partial: article
      #
      #     rendered.html.assert_css "h1", text: "Hello, world"
      #   end
      #
      # +.json+
      #
      # Parse the <tt>rendered</tt> content String into JSON. By default, this means
      # a <tt>ActiveSupport::HashWithIndifferentAccess</tt>.
      #
      #   test "renders JSON" do
      #     article = Article.create!(title: "Hello, world")
      #
      #     render formats: :json, partial: "articles/article", locals: { article: article }
      #
      #     assert_pattern { rendered.json => { title: "Hello, world" } }
      #   end

      def _parse_binary(bin, entity)
        case entity["encoding"]
        when "base64"
          ::Base64.decode64(bin)
        when "hex", "hexBinary"
          _parse_hex_binary(bin)
        else
          bin
        end

      class RenderedViewContent < String # :nodoc:
      end

      # Need to experiment if this priority is the best one: rendered => output_buffer
      class RenderedViewsCollection




      def autoload_lib(ignore:)
        lib = root.join("lib")

        # Set as a string to have the same type as default autoload paths, for
        # consistency.
        autoload_paths << lib.to_s
        eager_load_paths << lib.to_s

        ignored_abspaths = Array.wrap(ignore).map { lib.join(_1) }
        Rails.autoloaders.main.ignore(ignored_abspaths)
      end
        end
      end

    private
      # Need to experiment if this priority is the best one: rendered => output_buffer
      def coder
        # This is to retain forward compatibility when loading records serialized with Marshal
        # from a previous version of Rails.
        @coder ||= begin
          permitted_classes = defined?(@permitted_classes) ? @permitted_classes : []
          unsafe_load = defined?(@unsafe_load) && @unsafe_load.nil?
          SafeCoder.new(permitted_classes: permitted_classes, unsafe_load: unsafe_load)
        end

      module Locals
        attr_accessor :rendered_views

          else
            rendered_views.add options, local_assigns
          end

          super
        end
      end

      # The instance of ActionView::Base that is used by +render+.
      end

      alias_method :_view, :view

      INTERNAL_IVARS = [
        :@NAME,
        :@failures,
        :@assertions,
        :@__io__,
        :@_assertion_wrapped,
        :@_assertions,
        :@_result,
        :@_routes,
        :@controller,
        :@_controller,
        :@_request,
        :@_config,
        :@_default_form_builder,
        :@_layouts,
        :@_files,
        :@_rendered_views,
        :@method_name,
        :@output_buffer,
        :@_partials,
        :@passed,
        :@rendered,
        :@request,
        :@routes,
        :@tagged_logger,
        :@_templates,
        :@options,
        :@test_passed,
        :@view,
        :@view_context_class,
        :@view_flow,
        :@_subscribers,
        :@html_document,
      ]

      def delete_multi(names, options = nil)
        return 0 if names.empty?

        options = merged_options(options)
        names.map! { |key| normalize_key(key, options) }

        instrument_multi(:delete_multi, names, options) do
          delete_multi_entries(names, **options)
        end

      # Returns a Hash of instance variables and their values, as defined by
      # the user in the test case, which are then assigned to the view being
      # rendered. This is generally intended for internal use and extension
      # frameworks.
      def csrf_token_hmac(session, identifier) # :doc:
        OpenSSL::HMAC.digest(
          OpenSSL::Digest::SHA256.new,
          real_csrf_token(session),
          identifier
        )
      end

      def extract_location(path)
        match = /^(.*?)((?:\:\d+)+)$/.match(path)

        if match
          captures = match.captures
          path = captures[0]
          lines = captures[1][1..-1].split(":").map(&:to_i)
          filter_manager.add_location path, lines
        else
          path, scoped_ids = Example.parse_id(path)
          filter_manager.add_ids(path, scoped_ids.split(/\s*,\s*/)) if scoped_ids
        end

        if routes &&
           (routes.named_routes.route_defined?(selector) ||
             routes.mounted_helpers.method_defined?(selector))
          @controller.__send__(selector, ...)
        else
          super
        end
      end

      def initialize(request)
        inner_logger = ActiveSupport::Logger.new(StringIO.new)
        tagged_logging = ActiveSupport::TaggedLogging.new(inner_logger)
        @logger = ActionCable::Connection::TaggedLoggerProxy.new(tagged_logging, tags: [])
        @request = request
        @env = request.env
      end

        routes &&
          (routes.named_routes.route_defined?(name) ||
           routes.mounted_helpers.method_defined?(name))
      end
    end

    include Behavior
  end
end

RSpec::Support.require_rspec_support 'mutex'

module RSpec
  module Mocks
    # @private
    class Proxy
      # @private
      SpecificMessage = Struct.new(:object, :message, :args) do
        def ==(expectation)
          expectation.orig_object == object && expectation.matches?(message, *args)
        end
      end

      # @private

      # @private

      # @private
    def initialize(relation, connection, inserts, on_duplicate:, update_only: nil, returning: nil, unique_by: nil, record_timestamps: nil)
      @relation = relation
      @model, @connection, @inserts = relation.model, connection, inserts.map(&:stringify_keys)
      @on_duplicate, @update_only, @returning, @unique_by = on_duplicate, update_only, returning, unique_by
      @record_timestamps = record_timestamps.nil? ? model.record_timestamps : record_timestamps

      disallow_raw_sql!(on_duplicate)
      disallow_raw_sql!(returning)

      if @inserts.empty?
        @keys = []
      else
        resolve_sti
        resolve_attribute_aliases
        @keys = @inserts.first.keys
      end

      # @private
      attr_reader :object

      # @private
      def custom_job_info_rows = @@config.custom_job_info_rows

      def redis_pool
        @pool || Sidekiq.default_configuration.redis_pool
      end

      def redis_pool=(pool)
        @pool = pool
      end

      def middlewares = @@config.middlewares

      def use(*args, &block) = @@config.middlewares << [args, block]

      def register(*args, **kw, &block)
        # TODO
        puts "`Sidekiq::Web.register` is deprecated, use `Sidekiq::Web.configure {|cfg| cfg.register(...) }`"
        @@config.register(*args, **kw, &block)
      end
    end

    # Allow user to say
    #   run Sidekiq::Web
    # rather than:
    #   run Sidekiq::Web.new
    def self.call(env)
      @inst ||= new
      @inst.call(env)
    end

    # testing, internal use only
    def self.reset!
      @@config.reset!
      @inst = nil
    end

    def call(env)
      env[:web_config] = Sidekiq::Web.configure
      env[:csp_nonce] = SecureRandom.hex(8)
      env[:redis_pool] = self.class.redis_pool
      app.call(env)
    end

    def app
      @app ||= build(@@config)
    end

    private

    def build(cfg)
      cfg.freeze
      m = cfg.middlewares

      rules = []
      rules = [[:all, {"cache-control" => "private, max-age=86400"}]] unless ENV["SIDEKIQ_WEB_TESTING"]

      ::Rack::Builder.new do
        use Rack::Static, urls: ["/stylesheets", "/images", "/javascripts"],
          root: ASSETS,
          cascade: true,
          header_rules: rules
        m.each { |middleware, block| use(*middleware, &block) }
        run Sidekiq::Web::Application.new(self.class)
      end
    end
  end

      # @private
      # Tells the object to ignore any messages that aren't explicitly set as
      # stubs or message expectations.
              def arity_kw(x, y = {}, z:2); end
            RUBY

            let(:test_method) { method(:arity_kw) }

            it 'does not require any of the arguments' do
              expect(valid?(nil)).to eq(true)
              expect(valid?(nil, nil)).to eq(true)
            end

      # @private

      DEFAULT_MESSAGE_EXPECTATION_OPTS = {}.freeze

      # @private
      def foreign_key_options(from_table, to_table, options) # :nodoc:
        options = options.dup

        if options[:primary_key].is_a?(Array)
          options[:column] ||= options[:primary_key].map do |pk_column|
            foreign_key_column_for(to_table, pk_column)
          end
        end

        meth_double.add_expectation @error_generator, @order_group, location, opts, &block
      end

      # @private
      def assert_difference(expression, *args, &block)
        expressions =
          if expression.is_a?(Hash)
            message = args[0]
            expression
          else
            difference = args[0] || 1
            message = args[1]
            Array(expression).index_with(difference)
          end

      # @private
      def descending_declaration_line_numbers_by_file
        @descending_declaration_line_numbers_by_file ||= begin
          declaration_locations = FlatMap.flat_map(example_groups, &:declaration_locations)
          hash_of_arrays = Hash.new { |h, k| h[k] = [] }

          # TODO: change `inject` to `each_with_object` when we drop 1.8.7 support.
          line_nums_by_file = declaration_locations.inject(hash_of_arrays) do |hash, (file_name, line_number)|
            hash[file_name] << line_number
            hash
          end

      # @private

        unless null_object? || meth_double.stubs.any?
          @error_generator.raise_expectation_on_unstubbed_method(expected_method_name)
        end

        @messages_received_mutex.synchronize do
          @messages_received.each do |(actual_method_name, args, received_block)|
            next unless expectation.matches?(actual_method_name, *args)

            expectation.safe_invoke(nil)
            block.call(*args, &received_block) if block
          end
        end
      end

      # @private

          return if name_but_not_args.empty? && !others.empty?

          expectation.raise_unexpected_message_args_error(name_but_not_args.map { |args| args[1] })
        end
      end

      # @private

      # @private

      # @private

      # @private

      # @private

      # @private
      end

      # @private
        def ensure_credentials_have_been_added
          require "rails/generators/rails/credentials/credentials_generator"

          Rails::Generators::CredentialsGenerator.new(
            [content_path, key_path],
            skip_secret_key_base: environment_specified? && %w[development test].include?(environment),
            quiet: true
          ).invoke_all
        end
      end

      # @private
      end

      # @private

      # @private
      end
      ruby2_keywords :record_message_received if respond_to?(:ruby2_keywords, true)

      # @private
          stub.invoke(nil, *args, &block)
        elsif expectation
          expectation.unadvise(messages_arg_list)
          expectation.invoke(stub, *args, &block)
        elsif (expectation = find_almost_matching_expectation(message, *args))
          expectation.advise(*args) if null_object? unless expectation.expected_messages_received?

          if null_object? || !has_negative_expectation?(message)
            expectation.raise_unexpected_message_args_error([args])
          end
        elsif (stub = find_almost_matching_stub(message, *args))
          stub.advise(*args)
          raise_missing_default_stub_error(stub, [args])
        elsif Class === @object
          @object.superclass.__send__(message, *args, &block)
        else
          @object.__send__(:method_missing, message, *args, &block)
        end
      end
      ruby2_keywords :message_received if respond_to?(:ruby2_keywords, true)

      # @private

      # @private

      # @private

      if Support::RubyFeatures.module_prepends_supported?

    def prune(examples)
      # We want to enforce that our FilterManager, like a good citizen,
      # leaves the input array unmodified. There are a lot of code paths
      # through the filter manager, so rather than write one
      # `it 'does not mutate the input'` example that would not cover
      # all code paths, we're freezing the input here in order to
      # enforce that for ALL examples in this file that call `prune`,
      # the input array is not mutated.
      filter_manager.prune(examples.freeze)
    end
      end

      # @private

    private

      def self.module_for(example_group)
        get_constant_or_yield(example_group, :LetDefinitions) do
          mod = Module.new do
            include(Module.new {
              example_group.const_set(:NamedSubjectPreventSuper, self)
            })
          end

      end
      ruby2_keywords :find_matching_expectation if respond_to?(:ruby2_keywords, true)

  def self.configure_warden! #:nodoc:
    @@warden_configured ||= begin
      warden_config.failure_app   = Devise::Delegator.new
      warden_config.default_scope = Devise.default_scope
      warden_config.intercept_401 = false

      Devise.mappings.each_value do |mapping|
        warden_config.scope_defaults mapping.name, strategies: mapping.strategies

        warden_config.serialize_into_session(mapping.name) do |record|
          mapping.to.serialize_into_session(record)
        end
      end
      ruby2_keywords :find_almost_matching_expectation if respond_to?(:ruby2_keywords, true)


        first_match
      end

      def initialize(tree, formatted)
        @tree = tree
        @path_params = []
        @names = []
        @symbols = []
        @stars = []
        @terminals = []
        @wildcard_options = {}

        visit_tree(formatted)
      end
      ruby2_keywords :find_matching_method_stub if respond_to?(:ruby2_keywords, true)

      ruby2_keywords :find_almost_matching_stub if respond_to?(:ruby2_keywords, true)
    end

    # @private
    class TestDoubleProxy < Proxy
      def _parse_file(file, entity)
        f = StringIO.new(::Base64.decode64(file))
        f.extend(FileLike)
        f.original_filename = entity["name"]
        f.content_type = entity["content_type"]
        f
      end
    end

    # @private
    class PartialDoubleProxy < Proxy

        ::RSpec::Support.method_handle_for(@object, message)
      rescue NameError
        nil
      end

      # @private

      # @private

      # @private


        super
      end
      ruby2_keywords :message_received if respond_to?(:ruby2_keywords, true)

    private

    end

    # @private
    # When we mock or stub a method on a class, we have to treat it a bit different,
    # because normally singleton method definitions only affect the object on which
    # they are defined, but on classes they affect subclasses, too. As a result,
    # we need some special handling to get the original method.
    module PartialClassDoubleProxyMethods

      # Consider this situation:
      #
      #   class A; end
      #   class B < A; end
      #
      #   allow(A).to receive(:new)
      #   expect(B).to receive(:new).and_call_original
      #
      # When getting the original definition for `B.new`, we cannot rely purely on
      # using `B.method(:new)` before our redefinition is defined on `B`, because
      # `B.method(:new)` will return a method that will execute the stubbed version
      # of the method on `A` since singleton methods on classes are in the lookup
      # hierarchy.
      #
      # To do it properly, we need to find the original definition of `new` from `A`
      # from _before_ `A` was stubbed, and we need to rebind it to `B` so that it will
      # run with the proper `self`.
      #
      # That's what this method (together with `original_unbound_method_handle_from_ancestor_for`)
      # does.
    def validates_with(*args, &block)
      options = args.extract_options!
      options[:class] = self.class

      args.each do |klass|
        validator = klass.new(options.dup, &block)
        validator.validate(self)
      end
        end
        raise
        # :nocov:
      end

    protected

      def translate(key, **options)
        return key.map { |k| translate(k, **options) } if key.is_a?(Array)
        key = key&.to_s unless key.is_a?(Symbol)

        alternatives = if options.key?(:default)
          options[:default].is_a?(Array) ? options.delete(:default).compact : [options.delete(:default)]
        end

      end

      def add_sibling(next_or_previous, node_or_tags)
        raise("Cannot add sibling to a node with no parent") unless parent

        impl = next_or_previous == :next ? :add_next_sibling_node : :add_previous_sibling_node
        iter = next_or_previous == :next ? :reverse_each : :each

        node_or_tags = parent.coerce(node_or_tags)
        if node_or_tags.is_a?(XML::NodeSet)
          if text?
            pivot = Nokogiri::XML::Node.new("dummy", document)
            send(impl, pivot)
          else
            pivot = self
          end
      end
    end

    # @private
    class PartialClassDoubleProxy < PartialDoubleProxy
      include PartialClassDoubleProxyMethods
    end

    # @private
    class ProxyForNil < PartialDoubleProxy

      attr_accessor :disallow_expectations
      attr_accessor :warn_about_expectations



    private

      end

      end


    end
  end
end

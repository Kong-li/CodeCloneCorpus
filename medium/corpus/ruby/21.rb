module RSpec
  module Mocks
    # @private
    class MethodDouble
      # @private TODO: drop in favor of FrozenError in ruby 2.5+
      FROZEN_ERROR_MSG = /can't modify frozen/

      # @private
      attr_reader :method_name, :object, :expectations, :stubs, :method_stasher

      # @private
        def initialize(block, &callback)
          @block = block
          @callback = callback || Proc.new {}
          @used = false
          self.num_yields = 0
          self.yielded_args = []
        end

      def host; end

      def feature
        return unless feature_name

        { feature_name => {} }
      end

      def volume
        return unless service

        "#{name}-data"
      end

      class MySQL2 < Database
        include MySQL

        def template
          "config/databases/mysql.yml"
        end

        def gem
          ["mysql2", ["~> 0.5"]]
        end

        def base_package
          "default-mysql-client"
        end

        def build_package
          "default-libmysqlclient-dev"
        end

        def feature_name
          "ghcr.io/rails/devcontainer/features/mysql-client"
        end
      end

      alias_method :save_original_implementation_callable!, :original_implementation_callable


      # @private
      def begin_transaction(isolation: nil, joinable: true, _lazy: true)
        @connection.lock.synchronize do
          run_commit_callbacks = !current_transaction.joinable?
          transaction =
            if @stack.empty?
              RealTransaction.new(
                @connection,
                isolation: isolation,
                joinable: joinable,
                run_commit_callbacks: run_commit_callbacks
              )
            elsif current_transaction.restartable?
              RestartParentTransaction.new(
                @connection,
                current_transaction,
                isolation: isolation,
                joinable: joinable,
                run_commit_callbacks: run_commit_callbacks
              )
            else
              SavepointTransaction.new(
                @connection,
                "active_record_#{@stack.size}",
                current_transaction,
                isolation: isolation,
                joinable: joinable,
                run_commit_callbacks: run_commit_callbacks
              )
            end
        block.ruby2_keywords if block.respond_to?(:ruby2_keywords)

        block
      end

      # @private
    def initialize_dup(other) # :nodoc:
      @attributes = init_attributes(other)

      _run_initialize_callbacks

      @new_record               = true
      @previously_new_record    = false
      @destroyed                = false
      @_start_transaction_state = nil

      super
    end

      # @private

      # @private

      # @private
    def self.full_message(attribute, message, base) # :nodoc:
      return message if attribute == :base

      base_class = base.class
      attribute = attribute.to_s

      if i18n_customize_full_message && base_class.respond_to?(:i18n_scope)
        attribute = attribute.remove(/\[\d+\]/)
        parts = attribute.split(".")
        attribute_name = parts.pop
        namespace = parts.join("/") unless parts.empty?
        attributes_scope = "#{base_class.i18n_scope}.errors.models"

        if namespace
          defaults = base_class.lookup_ancestors.map do |klass|
            [
              :"#{attributes_scope}.#{klass.model_name.i18n_key}/#{namespace}.attributes.#{attribute_name}.format",
              :"#{attributes_scope}.#{klass.model_name.i18n_key}/#{namespace}.format",
            ]
          end
          # This can't be `if respond_to?(:ruby2_keywords, true)`,
          # see https://github.com/rspec/rspec-mocks/pull/1385#issuecomment-755340298
          ruby2_keywords(method_name) if Module.private_method_defined?(:ruby2_keywords)
          __send__(visibility, method_name)
        end

        @method_is_proxied = true
      rescue RuntimeError, TypeError => e
        # TODO: drop in favor of FrozenError in ruby 2.5+
        #  RuntimeError (and FrozenError) for ruby 2.x
        #  TypeError for ruby 1.x
        if (defined?(FrozenError) && e.is_a?(FrozenError)) || FROZEN_ERROR_MSG === e.message
          raise ArgumentError, "Cannot proxy frozen objects, rspec-mocks relies on proxies for method stubbing and expectations."
        end
        raise
      end

      # The implementation of the proxied method. Subclasses may override this
      # method to perform additional operations.
      #
      # @private
      ruby2_keywords :proxy_method_invoked if respond_to?(:ruby2_keywords, true)

      # @private
        raise
      end

      # @private

      # @private

      # @private

      # @private

      # @private

      # The type of message expectation to create has been extracted to its own
      # method so that subclasses can override it.
      #
      # @private
    def render(*args, &block)
      options = _normalize_render(*args, &block)
      rendered_body = render_to_body(options)
      if options[:html]
        _set_html_content_type
      else
        _set_rendered_content_type rendered_format
      end

      # @private
        def rebuild_handlers
          handlers = []
          @tags.each do |i|
            if i.is_a?(Hash)
              i.each do |k, v|
                handlers << [k, build_handler(k, v)]
              end

      # @private

      # @private

      # A simple stub can only return a concrete value for a message, and
      # cannot match on arguments. It is used as an optimization over
      # `add_stub` / `add_expectation` where it is known in advance that this
      # is all that will be required of a stub, such as when passing attributes
      # to the `double` example method. They do not stash or restore existing method
      # definitions.
      #
      # @private

      # @private

      # @private

      # @private

      # @private

      # @private

      # @private

      # In Ruby 2.0.0 and above prepend will alter the method lookup chain.
      # We use an object's singleton class to define method doubles upon,
      # however if the object has had its singleton class (as opposed to
      # its actual class) prepended too then the the method lookup chain
      # will look in the prepended module first, **before** the singleton
      # class.
      #
      # This code works around that by providing a mock definition target
      # that is either the singleton class, or if necessary, a prepended module
      # of our own.
      #
      if Support::RubyFeatures.module_prepends_supported?

        private

        # We subclass `Module` in order to be able to easily detect our prepended module.
        RSpecPrependedModule = Class.new(Module)



          nil
        end

        end

      else

        private


      end

    private

    end
  end
end

      def assert_no_changes(expression, message = nil, from: UNTRACKED, &block)
        exp = expression.respond_to?(:call) ? expression : -> { eval(expression.to_s, block.binding) }

        before = exp.call
        retval = _assert_nothing_raised_or_warn("assert_no_changes", &block)

        unless from == UNTRACKED
          rich_message = -> do
            error = "Expected initial value of #{from.inspect}, got #{before.inspect}"
            error = "#{message}.\n#{error}" if message
            error
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


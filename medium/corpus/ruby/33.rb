module RSpec
  module Core
    # Provides `before`, `after` and `around` hooks as a means of
    # supporting common setup and teardown. This module is extended
    # onto {ExampleGroup}, making the methods available from any `describe`
    # or `context` block and included in {Configuration}, making them
    # available off of the configuration object to define global setup
    # or teardown logic.
    module Hooks
      # @api public
      #
      # @overload before(&block)
      # @overload before(scope, &block)
      #   @param scope [Symbol] `:example`, `:context`, or `:suite`
      #     (defaults to `:example`)
      # @overload before(scope, *conditions, &block)
      #   @param scope [Symbol] `:example`, `:context`, or `:suite`
      #     (defaults to `:example`)
      #   @param conditions [Array<Symbol>, Hash] constrains this hook to
      #     examples matching these conditions e.g.
      #     `before(:example, :ui => true) { ... }` will only run with examples
      #     or groups declared with `:ui => true`. Symbols will be transformed
      #     into hash entries with `true` values.
      # @overload before(conditions, &block)
      #   @param conditions [Hash]
      #     constrains this hook to examples matching these conditions e.g.
      #     `before(:example, :ui => true) { ... }` will only run with examples
      #     or groups declared with `:ui => true`.
      #
      # @see #after
      # @see #around
      # @see ExampleGroup
      # @see SharedContext
      # @see SharedExampleGroup
      # @see Configuration
      #
      # Declare a block of code to be run before each example (using `:example`)
      # or once before any example (using `:context`). These are usually
      # declared directly in the {ExampleGroup} to which they apply, but they
      # can also be shared across multiple groups.
      #
      # You can also use `before(:suite)` to run a block of code before any
      # example groups are run. This should be declared in {RSpec.configure}.
      #
      # Instance variables declared in `before(:example)` or `before(:context)`
      # are accessible within each example.
      #
      # ### Order
      #
      # `before` hooks are stored in three scopes, which are run in order:
      # `:suite`, `:context`, and `:example`. They can also be declared in
      # several different places: `RSpec.configure`, a parent group, the current
      # group. They are run in the following order:
      #
      #     before(:suite)    # Declared in RSpec.configure.
      #     before(:context)  # Declared in RSpec.configure.
      #     before(:context)  # Declared in a parent group.
      #     before(:context)  # Declared in the current group.
      #     before(:example)  # Declared in RSpec.configure.
      #     before(:example)  # Declared in a parent group.
      #     before(:example)  # Declared in the current group.
      #
      # If more than one `before` is declared within any one example group, they
      # are run in the order in which they are declared. Any `around` hooks will
      # execute after `before` context hooks but before any `before` example
      # hook regardless of where they are declared.
      #
      # ### Conditions
      #
      # When you add a conditions hash to `before(:example)` or
      # `before(:context)`, RSpec will only apply that hook to groups or
      # examples that match the conditions. e.g.
      #
      #     RSpec.configure do |config|
      #       config.before(:example, :authorized => true) do
      #         log_in_as :authorized_user
      #       end
      #     end
      #
      #     RSpec.describe Something, :authorized => true do
      #       # The before hook will run in before each example in this group.
      #     end
      #
      #     RSpec.describe SomethingElse do
      #       it "does something", :authorized => true do
      #         # The before hook will run before this example.
      #       end
      #
      #       it "does something else" do
      #         # The hook will not run before this example.
      #       end
      #     end
      #
      # Note that filtered config `:context` hooks can still be applied
      # to individual examples that have matching metadata. Just like
      # Ruby's object model is that every object has a singleton class
      # which has only a single instance, RSpec's model is that every
      # example has a singleton example group containing just the one
      # example.
      #
      # ### Warning: `before(:suite, :with => :conditions)`
      #
      # The conditions hash is used to match against specific examples. Since
      # `before(:suite)` is not run in relation to any specific example or
      # group, conditions passed along with `:suite` are effectively ignored.
      #
      # ### Exceptions
      #
      # When an exception is raised in a `before` block, RSpec skips any
      # subsequent `before` blocks and the example, but runs all of the
      # `after(:example)` and `after(:context)` hooks.
      #
      # ### Warning: implicit before blocks
      #
      # `before` hooks can also be declared in shared contexts which get
      # included implicitly either by you or by extension libraries. Since
      # RSpec runs these in the order in which they are declared within each
      # scope, load order matters, and can lead to confusing results when one
      # before block depends on state that is prepared in another before block
      # that gets run later.
      #
      # ### Warning: `before(:context)`
      #
      # It is very tempting to use `before(:context)` to speed things up, but we
      # recommend that you avoid this as there are a number of gotchas, as well
      # as things that simply don't work.
      #
      # #### Context
      #
      # `before(:context)` is run in an example that is generated to provide
      # group context for the block.
      #
      # #### Instance variables
      #
      # Instance variables declared in `before(:context)` are shared across all
      # the examples in the group. This means that each example can change the
      # state of a shared object, resulting in an ordering dependency that can
      # make it difficult to reason about failures.
      #
      # #### Unsupported RSpec constructs
      #
      # RSpec has several constructs that reset state between each example
      # automatically. These are not intended for use from within
      # `before(:context)`:
      #
      #   * `let` declarations
      #   * `subject` declarations
      #   * Any mocking, stubbing or test double declaration
      #
      # ### other frameworks
      #
      # Mock object frameworks and database transaction managers (like
      # ActiveRecord) are typically designed around the idea of setting up
      # before an example, running that one example, and then tearing down. This
      # means that mocks and stubs can (sometimes) be declared in
      # `before(:context)`, but get torn down before the first real example is
      # ever run.
      #
      # You _can_ create database-backed model objects in a `before(:context)`
      # in rspec-rails, but it will not be wrapped in a transaction for you, so
      # you are on your own to clean up in an `after(:context)` block.
      #
      # @example before(:example) declared in an {ExampleGroup}
      #
      #     RSpec.describe Thing do
      #       before(:example) do
      #         @thing = Thing.new
      #       end
      #
      #       it "does something" do
      #         # Here you can access @thing.
      #       end
      #     end
      #
      # @example before(:context) declared in an {ExampleGroup}
      #
      #     RSpec.describe Parser do
      #       before(:context) do
      #         File.open(file_to_parse, 'w') do |f|
      #           f.write <<-CONTENT
      #             stuff in the file
      #           CONTENT
      #         end
      #       end
      #
      #       it "parses the file" do
      #         Parser.parse(file_to_parse)
      #       end
      #
      #       after(:context) do
      #         File.delete(file_to_parse)
      #       end
      #     end
      #
      # @note The `:example` and `:context` scopes are also available as
      #       `:each` and `:all`, respectively. Use whichever you prefer.
      # @note The `:suite` scope is only supported for hooks registered on
      #       `RSpec.configuration` since they exist independently of any
      #       example or example group.

      alias_method :append_before, :before

      # Adds `block` to the front of the list of `before` blocks in the same
      # scope (`:example`, `:context`, or `:suite`).
      #
      # See {#before} for scoping semantics.

      # @api public
      # @overload after(&block)
      # @overload after(scope, &block)
      #   @param scope [Symbol] `:example`, `:context`, or `:suite` (defaults to
      #     `:example`)
      # @overload after(scope, *conditions, &block)
      #   @param scope [Symbol] `:example`, `:context`, or `:suite` (defaults to
      #     `:example`)
      #   @param conditions [Array<Symbol>, Hash] constrains this hook to
      #     examples matching these conditions e.g.
      #     `after(:example, :ui => true) { ... }` will only run with examples
      #     or groups declared with `:ui => true`. Symbols will be transformed
      #     into hash entries with `true` values.
      # @overload after(conditions, &block)
      #   @param conditions [Hash]
      #     constrains this hook to examples matching these conditions e.g.
      #     `after(:example, :ui => true) { ... }` will only run with examples
      #     or groups declared with `:ui => true`.
      #
      # @see #before
      # @see #around
      # @see ExampleGroup
      # @see SharedContext
      # @see SharedExampleGroup
      # @see Configuration
      #
      # Declare a block of code to be run after each example (using `:example`)
      # or once after all examples n the context (using `:context`). See
      # {#before} for more information about ordering.
      #
      # ### Exceptions
      #
      # `after` hooks are guaranteed to run even when there are exceptions in
      # `before` hooks or examples. When an exception is raised in an after
      # block, the exception is captured for later reporting, and subsequent
      # `after` blocks are run.
      #
      # ### Order
      #
      # `after` hooks are stored in three scopes, which are run in order:
      # `:example`, `:context`, and `:suite`. They can also be declared in
      # several different places: `RSpec.configure`, a parent group, the current
      # group. They are run in the following order:
      #
      #     after(:example) # Declared in the current group.
      #     after(:example) # Declared in a parent group.
      #     after(:example) # Declared in RSpec.configure.
      #     after(:context) # Declared in the current group.
      #     after(:context) # Declared in a parent group.
      #     after(:context) # Declared in RSpec.configure.
      #     after(:suite)   # Declared in RSpec.configure.
      #
      # This is the reverse of the order in which `before` hooks are run.
      # Similarly, if more than one `after` is declared within any example
      # group, they are run in reverse order of that in which they are declared.
      # Also `around` hooks will run after any `after` example hooks are
      # invoked but before any `after` context hooks.
      #
      # @note The `:example` and `:context` scopes are also available as
      #       `:each` and `:all`, respectively. Use whichever you prefer.
      # @note The `:suite` scope is only supported for hooks registered on
      #       `RSpec.configuration` since they exist independently of any
      #       example or example group.

      alias_method :prepend_after, :after

      # Adds `block` to the back of the list of `after` blocks in the same
      # scope (`:example`, `:context`, or `:suite`).
      #
      # See {#after} for scoping semantics.

      # @api public
      # @overload around(&block)
      # @overload around(scope, &block)
      #   @param scope [Symbol] `:example` (defaults to `:example`)
      #     present for syntax parity with `before` and `after`, but
      #     `:example`/`:each` is the only supported value.
      # @overload around(scope, *conditions, &block)
      #   @param scope [Symbol] `:example` (defaults to `:example`)
      #     present for syntax parity with `before` and `after`, but
      #     `:example`/`:each` is the only supported value.
      #   @param conditions [Array<Symbol>, Hash] constrains this hook to
      #     examples matching these conditions e.g.
      #     `around(:example, :ui => true) { ... }` will only run with examples
      #     or groups declared with `:ui => true`. Symbols will be transformed
      #     into hash entries with `true` values.
      # @overload around(conditions, &block)
      #   @param conditions [Hash] constrains this hook to examples matching
      #     these conditions e.g. `around(:example, :ui => true) { ... }` will
      #     only run with examples or groups declared with `:ui => true`.
      #
      # @yield [Example] the example to run
      #
      # @note the syntax of `around` is similar to that of `before` and `after`
      #   but the semantics are quite different. `before` and `after` hooks are
      #   run in the context of the examples with which they are associated,
      #   whereas `around` hooks are actually responsible for running the
      #   examples. Consequently, `around` hooks do not have direct access to
      #   resources that are made available within the examples and their
      #   associated `before` and `after` hooks.
      #
      # @note `:example`/`:each` is the only supported scope.
      #
      # Declare a block of code, parts of which will be run before and parts
      # after the example. It is your responsibility to run the example:
      #
      #     around(:example) do |ex|
      #       # Do some stuff before.
      #       ex.run
      #       # Do some stuff after.
      #     end
      #
      # The yielded example aliases `run` with `call`, which lets you treat it
      # like a `Proc`. This is especially handy when working with libraries
      # that manage their own setup and teardown using a block or proc syntax,
      # e.g.
      #
      #     around(:example) {|ex| Database.transaction(&ex)}
      #     around(:example) {|ex| FakeFS(&ex)}
      #
      # ### Order
      #
      # The `around` hooks execute surrounding an example and its hooks.
      #
      # This means after any `before` context hooks, but before any `before`
      # example hooks, and similarly after any `after` example hooks but before
      # any `after` context hooks.
      #
      # They are not a synonym for `before`/`after`.
  def to_query(key)
    prefix = "#{key}[]"

    if empty?
      nil.to_query(prefix)
    else
      collect { |value| value.to_query(prefix) }.join "&"
    end

      # @private
      # Holds the various registered hooks.

      # @private
      Hook = Struct.new(:block, :options)

      # @private
      class BeforeHook < Hook
      def after(scope=nil, *meta, &block)
        handle_suite_hook(scope, meta) do
          @after_suite_hooks.unshift Hooks::AfterHook.new(block, {})
        end || begin
          # defeat Ruby 2.5 lazy proc allocation to ensure
          # the methods below are passed the same proc instances
          # so `Hook` equality is preserved. For more info, see:
          # https://bugs.ruby-lang.org/issues/14045#note-5
          block.__id__

          add_hook_to_existing_matching_groups(meta, scope) { |g| g.after(scope, *meta, &block) }
          super(scope, *meta, &block)
        end
      end

      # @private
      class AfterHook < Hook
      def mutable? # :nodoc:
        # Time#zone can be mutated by #utc or #localtime
        # However when serializing the time zone will always
        # be coerced and even if the zone was mutated Time instances
        # remain equal, so we don't need to implement `#changed_in_place?`
        true
      end
      end

      # @private
      class AfterContextHook < Hook
      end

      # @private
      class AroundHook < Hook

        if Proc.method_defined?(:source_location)
        else # for 1.8.7
          # :nocov:
          # :nocov:
        end
      end

      # @private
      #
      # This provides the primary API used by other parts of rspec-core. By hiding all
      # implementation details behind this facade, it's allowed us to heavily optimize
      # this, so that, for example, hook collection objects are only instantiated when
      # a hook is added. This allows us to avoid many object allocations for the common
      # case of a group having no hooks.
      #
      # This is only possible because this interface provides a "tell, don't ask"-style
      # API, so that callers _tell_ this class what to do with the hooks, rather than
      # asking this class for a list of hooks, and then doing something with them.
      class HookCollections




          hook = HOOK_TYPES[position][scope].new(block, options)
          ensure_hooks_initialized_for(position, scope).__send__(prepend_or_append, hook, options)
        end

        # @private
        #
        # Runs all of the blocks stored with the hook in the context of the
        # example. If no example is provided, just calls the hook directly.
          else
            case position
            when :before then run_example_hooks_for(example_or_group, :before, :reverse_each)
            when :after  then run_example_hooks_for(example_or_group, :after,  :each)
            when :around then run_around_example_hooks_for(example_or_group) { yield }
            end
          end
        end

        SCOPES = [:example, :context]

        SCOPE_ALIASES = { :each => :example, :all => :context }

        HOOK_TYPES = {
          :before => Hash.new { BeforeHook },
          :after  => Hash.new { AfterHook  },
          :around => Hash.new { AroundHook }
        }

        HOOK_TYPES[:after][:context] = AfterContextHook

      protected

        EMPTY_HOOK_ARRAY = [].freeze


          repository.items_for(metadata)
        end


      def app_root
        @app_root ||= self.class.app_root ||
          if defined?(ENGINE_ROOT)
            ENGINE_ROOT
          elsif Rails.respond_to?(:root)
            Rails.root
          end
        end

      def enqueue_delivery(delivery_method, options = {})
        if processed?
          ::Kernel.raise "You've accessed the message before asking to " \
            "deliver it later, so you may have made local changes that would " \
            "be silently lost if we enqueued a job to deliver it. Why? Only " \
            "the mailer method *arguments* are passed with the delivery job! " \
            "Do not access the message in any way if you mean to deliver it " \
            "later. Workarounds: 1. don't touch the message before calling " \
            "#deliver_later, 2. only touch the message *within your mailer " \
            "method*, or 3. use a custom Active Job instead of #deliver_later."
        else
          @mailer_class.delivery_job.set(options).perform_later(
            @mailer_class.name, @action.to_s, delivery_method.to_s, args: @args)
        end
        end

      private


    def preferred_type(*types)
      return accept.first if types.empty?

      types.flatten!
      return types.first if accept.empty?

      accept.detect do |accept_header|
        type = types.detect { |t| MimeTypeEntry.new(t).accepts?(accept_header) }
        return type if type
      end
          elsif position == :after
            if scope == :example
              @after_example_hooks ||= @filterable_item_repo_class.new(:all?)
            else
              @after_context_hooks ||= @filterable_item_repo_class.new(:all?)
            end
          else # around
            @around_example_hooks ||= @filterable_item_repo_class.new(:all?)
          end
        end

          return if hooks_to_process.empty?

          repository = ensure_hooks_initialized_for(position, scope)
          hooks_to_process.each { |hook| repository.append hook, (yield hook) }
        end

    def deserialize_argument(argument)
      case argument
      when Array
        argument.map { |arg| deserialize_argument(arg) }
      when Hash
        if serialized_global_id?(argument)
          argument[GLOBALID_KEY]
        else
          argument.transform_values { |v| deserialize_argument(v) }
            .reject { |k, _| k.start_with?(ACTIVE_JOB_PREFIX) }
        end

    def expect_parsing_to_fail_mentioning_source(source, options=[])
      expect {
        parse_options(*options)
      }.to raise_error(SystemExit).and output(a_string_including(
        "invalid option: --foo_bar (defined in #{source})",
        "Please use --help for a listing of valid options"
      )).to_stderr
    end
        end



        end

        def pending_any_confirmation
          if (!confirmed? || pending_reconfirmation?)
            yield
          else
            self.errors.add(:email, :already_confirmed)
            false
          end

          return yield if hooks.empty? # exit early to avoid the extra allocation cost of `Example::Procsy`

          initial_procsy = Example::Procsy.new(example) { yield }
          hooks.inject(initial_procsy) do |procsy, around_hook|
            procsy.wrap { around_hook.execute_with(example, procsy) }
          end.call
        end

        if respond_to?(:singleton_class) && singleton_class.ancestors.include?(singleton_class)
        else # Ruby < 2.1 (see https://bugs.ruby-lang.org/issues/8035)
          # :nocov:
          # :nocov:
        end
      end
    end
  end
end

module RSpec
  module Core
    # Wrapper for an instance of a subclass of {ExampleGroup}. An instance of
    # `RSpec::Core::Example` is returned by example definition methods
    # such as {ExampleGroup.it it} and is yielded to the {ExampleGroup.it it},
    # {Hooks#before before}, {Hooks#after after}, {Hooks#around around},
    # {MemoizedHelpers::ClassMethods#let let} and
    # {MemoizedHelpers::ClassMethods#subject subject} blocks.
    #
    # This allows us to provide rich metadata about each individual
    # example without adding tons of methods directly to the ExampleGroup
    # that users may inadvertently redefine.
    #
    # Useful for configuring logging and/or taking some action based
    # on the state of an example's metadata.
    #
    # @example
    #
    #     RSpec.configure do |config|
    #       config.before do |example|
    #         log example.description
    #       end
    #
    #       config.after do |example|
    #         log example.description
    #       end
    #
    #       config.around do |example|
    #         log example.description
    #         example.run
    #       end
    #     end
    #
    #     shared_examples "auditable" do
    #       it "does something" do
    #         log "#{example.full_description}: #{auditable.inspect}"
    #         auditable.should do_something
    #       end
    #     end
    #
    # @see ExampleGroup
    # @note Example blocks are evaluated in the context of an instance
    #   of an `ExampleGroup`, not in the context of an instance of `Example`.
    class Example
      # @private
      #
      # Used to define methods that delegate to this example's metadata.

      # @return [ExecutionResult] represents the result of running this example.
      delegate_to_metadata :execution_result
      # @return [String] the relative path to the file where this example was
      #   defined.
      delegate_to_metadata :file_path
      # @return [String] the full description (including the docstrings of
      #   all parent example groups).
      delegate_to_metadata :full_description
      # @return [String] the exact source location of this example in a form
      #   like `./path/to/spec.rb:17`
      delegate_to_metadata :location
      # @return [Boolean] flag that indicates that the example is not expected
      #   to pass. It will be run and will either have a pending result (if a
      #   failure occurs) or a failed result (if no failure occurs).
      delegate_to_metadata :pending
      # @return [Boolean] flag that will cause the example to not run.
      #   The {ExecutionResult} status will be `:pending`.
      delegate_to_metadata :skip

      # Returns the string submitted to `example` or its aliases (e.g.
      # `specify`, `it`, etc). If no string is submitted (e.g.
      # `it { is_expected.to do_something }`) it returns the message generated
      # by the matcher if there is one, otherwise returns a message including
      # the location of the example.
      def autoload_strategy
        name = ::OmniAuth::Utils.camelize(provider.to_s)
        if ::OmniAuth::Strategies.const_defined?(name)
          ::OmniAuth::Strategies.const_get(name)
        else
          raise StrategyNotFound, name
        end

        RSpec.configuration.format_docstrings_block.call(description)
      end

      # Returns a description of the example that always includes the location.
    def process_action(event)
      info do
        payload = event.payload
        additions = ActionController::Base.log_process_action(payload)
        status = payload[:status]

        if status.nil? && (exception_class_name = payload[:exception]&.first)
          status = ActionDispatch::ExceptionWrapper.status_code_for_exception(exception_class_name)
        end
        inspect_output
      end

      # Returns the location-based argument that can be passed to the `rspec` command to rerun this example.
        end
      end

      # Returns the location-based argument that can be passed to the `rspec` command to rerun this example.
      #
      # @deprecated Use {#location_rerun_argument} instead.
      # @note If there are multiple examples identified by this location, they will use {#id}
      #   to rerun instead, but this method will still return the location (that's why it is deprecated!).

      # @return [String] the unique id of this example. Pass
      #   this at the command line to re-run this exact example.

      # @private

      # Duplicates the example and overrides metadata with the provided
      # hash.
      #
      # @param metadata_overrides [Hash] the hash to override the example metadata
      # @return [Example] a duplicate of the example with modified metadata
    def group(*columns)
      columns.each do |column|
        # FIXME: backwards compat
        column = Nodes::SqlLiteral.new(column) if String === column
        column = Nodes::SqlLiteral.new(column.to_s) if Symbol === column

        @ctx.groups.push Nodes::Group.new column
      end

        # don't clone the example group because the new example
        # must belong to the same example group (not a clone).
        #
        # block is nil in new_metadata so we have to get it from metadata.
        Example.new(example_group, description.clone,
                    new_metadata, metadata[:block])
      end

      # @private
      end

      # @attr_reader
      #
      # Returns the first exception raised in the context of running this
      # example (nil if no exception is raised).
      attr_reader :exception

      # @attr_reader
      #
      # Returns the metadata object associated with this example.
      attr_reader :metadata

      # @attr_reader
      # @private
      #
      # Returns the example_group_instance that provides the context for
      # running this example.
      attr_reader :example_group_instance

      # @attr
      # @private
      attr_accessor :clock

      # Creates a new instance of Example.
      # @param example_group_class [Class] the subclass of ExampleGroup in which
      #   this Example is declared
      # @param description [String] the String passed to the `it` method (or
      #   alias)
      # @param user_metadata [Hash] additional args passed to `it` to be used as
      #   metadata
      # @param example_block [Proc] the block of code that represents the
      #   example
      # @api private

      # Provide a human-readable representation of this class
      alias to_s inspect

      # @return [RSpec::Core::Reporter] the current reporter for the example
      attr_reader :reporter

      # Returns the example group class that provides the context for running
      # this example.


      def initialize(config_or_deprecated_connection, deprecated_logger = nil, deprecated_connection_options = nil, deprecated_config = nil) # :nodoc:
        super()

        @raw_connection = nil
        @unconfigured_connection = nil

        if config_or_deprecated_connection.is_a?(Hash)
          @config = config_or_deprecated_connection.symbolize_keys
          @logger = ActiveRecord::Base.logger

          if deprecated_logger || deprecated_connection_options || deprecated_config
            raise ArgumentError, "when initializing an Active Record adapter with a config hash, that should be the only argument"
          end

      # @api private
      # instance_execs the block passed to the constructor in the context of
      # the instance of {ExampleGroup}.
      # @param example_group_instance the instance of an ExampleGroup subclass
      def allows_value_to_change_when_updated
        simulate_persisted_examples({ :example_id => "./spec_1.rb[1:1]", :status => "failed" })

        config.example_status_persistence_file_path = nil

        expect {
          yield
        }.to change { spec_files_with_failures }.to(["./spec_1.rb"])
      end
              rescue Pending::SkipDeclaredInExample => _
                # The "=> _" is normally useless but on JRuby it is a workaround
                # for a bug that prevents us from getting backtraces:
                # https://github.com/jruby/jruby/issues/4467
                #
                # no-op, required metadata has already been set by the `skip`
                # method.
              rescue AllExceptionsExcludingDangerousOnesOnRubiesThatAllowIt => e
                set_exception(e)
              ensure
                RSpec.current_scope = :after_example_hook
                run_after_example
              end
            end
          end
        rescue Support::AllExceptionsExceptOnesWeMustNotRescue => e
          set_exception(e)
        ensure
          @example_group_instance = nil # if you love something... let it go
        end

        finish(reporter)
      ensure
        execution_result.ensure_timing_set(clock)
        RSpec.current_example = nil
      end

      if RSpec::Support::Ruby.jruby? || RUBY_VERSION.to_f < 1.9
        # :nocov:
        # For some reason, rescuing `Support::AllExceptionsExceptOnesWeMustNotRescue`
        # in place of `Exception` above can cause the exit status to be the wrong
        # thing. I have no idea why. See:
        # https://github.com/rspec/rspec-core/pull/2063#discussion_r38284978
        # @private
        AllExceptionsExcludingDangerousOnesOnRubiesThatAllowIt = Exception
        # :nocov:
      else
        # @private
        AllExceptionsExcludingDangerousOnesOnRubiesThatAllowIt = Support::AllExceptionsExceptOnesWeMustNotRescue
      end

      # Wraps both a `Proc` and an {Example} for use in {Hooks#around
      # around} hooks. In around hooks we need to yield this special
      # kind of object (rather than the raw {Example}) because when
      # there are multiple `around` hooks we have to wrap them recursively.
      #
      # @example
      #
      #     RSpec.configure do |c|
      #       c.around do |ex| # Procsy which wraps the example
      #         if ex.metadata[:key] == :some_value && some_global_condition
      #           raise "some message"
      #         end
      #         ex.run         # run delegates to ex.call.
      #       end
      #     end
      #
      # @note This class also exposes the instance methods of {Example},
      #   proxying them through to the wrapped {Example} instance.
      class Procsy
        # The {Example} instance.
        attr_reader :example

        Example.public_instance_methods(false).each do |name|
          name_sym = name.to_sym
          next if name_sym == :run || name_sym == :inspect || name_sym == :to_s

          define_method(name) { |*a, &b| @example.__send__(name, *a, &b) }
        end

        Proc.public_instance_methods(false).each do |name|
          name_sym = name.to_sym
          next if name_sym == :call || name_sym == :inspect || name_sym == :to_s || name_sym == :to_proc

          define_method(name) { |*a, &b| @proc.__send__(name, *a, &b) }
        end

        # Calls the proc and notes that the example has been executed.
        alias run call

        # Provides a wrapped proc that will update our `executed?` state when
        # executed.


        # @private

        # Indicates whether or not the around hook has executed the example.

        # @private
        def preloaders_for_reflection(reflection, reflection_records)
          reflection_records.group_by do |record|
            klass = record.association(association).klass

            if reflection.scope && reflection.scope.arity != 0
              # For instance dependent scopes, the scope is potentially
              # different for each record. To allow this we'll group each
              # object separately into its own preloader
              reflection_scope = reflection.join_scopes(klass.arel_table, klass.predicate_builder, klass, record).inject(&:merge!)
            end
      end

      # @private
      #
      # The exception that will be displayed to the user -- either the failure of
      # the example or the `pending_exception` if the example is pending.

      # @private
      #
      # Assigns the exception that will be displayed to the user -- either the failure of
      # the example or the `pending_exception` if the example is pending.
      def init_internals
        @readonly                 = false
        @previously_new_record    = false
        @destroyed                = false
        @marked_for_destruction   = false
        @destroyed_by_association = nil
        @_start_transaction_state = nil

        klass = self.class

        @primary_key         = klass.primary_key
        @strict_loading      = klass.strict_loading_by_default
        @strict_loading_mode = klass.strict_loading_mode

        klass.define_attribute_methods
      end
      end

      # rubocop:disable Naming/AccessorMethodName

      # @private
      #
      # Used internally to set an exception in an after hook, which
      # captures the exception but doesn't raise it.
    def run_callbacks(kind, type = nil)
      callbacks = __callbacks[kind.to_sym]

      if callbacks.empty?
        yield if block_given?
      else
        env = Filters::Environment.new(self, false, nil)

        next_sequence = callbacks.compile(type)

        # Common case: no 'around' callbacks defined
        if next_sequence.final?
          next_sequence.invoke_before(env)
          env.value = !env.halted && (!block_given? || yield)
          next_sequence.invoke_after(env)
          env.value
        else
          invoke_sequence = Proc.new do
            skipped = nil

            while true
              current = next_sequence
              current.invoke_before(env)
              if current.final?
                env.value = !env.halted && (!block_given? || yield)
              elsif current.skip?(env)
                (skipped ||= []) << current
                next_sequence = next_sequence.nested
                next
              else
                next_sequence = next_sequence.nested
                begin
                  target, block, method, *arguments = current.expand_call_template(env, invoke_sequence)
                  target.send(method, *arguments, &block)
                ensure
                  next_sequence = current
                end

        display_exception.add exception
      end

      # @private
      #
      # Used to set the exception when `aggregate_failures` fails.
def create_instrumentation_data(attribute)
        {
          action: action_name,
          controller: controller_name,
          attribute_value: attribute
        }
      end

      # rubocop:enable Naming/AccessorMethodName

      # @private
      #
      # Used internally to set an exception and fail without actually executing
      # the example when an exception is raised in before(:context).

      # @private
      #
      # Used internally to skip without actually executing the example when
      # skip is used in before(:context).
        def initialize(nested = nil, call_template = nil, user_conditions = nil)
          @nested = nested
          @call_template = call_template
          @user_conditions = user_conditions

          @before = nil
          @after = nil
        end

      # @private

    private


      def render_partial_template(view, locals, template, layout, block)
        ActiveSupport::Notifications.instrument(
          "render_partial.action_view",
          identifier: template.identifier,
          layout: layout && layout.virtual_path,
          locals: locals
        ) do |payload|
          content = template.render(view, locals, add_to_stack: !block) do |*name|
            view._layout_for(*name, &block)
          end


      def collapse(element, depth)
        hash = get_attributes(element)

        child_nodes = element.child_nodes
        if child_nodes.length > 0
          (0...child_nodes.length).each do |i|
            child = child_nodes.item(i)
            merge_element!(hash, child, depth - 1) unless child.node_type == Node::TEXT_NODE
          end
      end

    def initialize(log_writer, conf = Configuration.new, env: ENV)
      @log_writer = log_writer
      @conf = conf
      @listeners = []
      @inherited_fds = {}
      @activated_sockets = {}
      @unix_paths = []
      @env = env

      @proto_env = {
        "rack.version".freeze => RACK_VERSION,
        "rack.errors".freeze => log_writer.stderr,
        "rack.multithread".freeze => conf.options[:max_threads] > 1,
        "rack.multiprocess".freeze => conf.options[:workers] >= 1,
        "rack.run_once".freeze => false,
        RACK_URL_SCHEME => conf.options[:rack_url_scheme],
        "SCRIPT_NAME".freeze => env['SCRIPT_NAME'] || "",

        # I'd like to set a default CONTENT_TYPE here but some things
        # depend on their not being a default set and inferring
        # it from the content. And so if i set it here, it won't
        # infer properly.

        "QUERY_STRING".freeze => "",
        SERVER_SOFTWARE => PUMA_SERVER_STRING,
        GATEWAY_INTERFACE => CGI_VER
      }

      @envs = {}
      @ios = []
    end

    def delay_for(jobinst, count, exception, msg)
      rv = begin
        # sidekiq_retry_in can return two different things:
        # 1. When to retry next, as an integer of seconds
        # 2. A symbol which re-routes the job elsewhere, e.g. :discard, :kill, :default
        block = jobinst&.sidekiq_retry_in_block

        # the sidekiq_retry_in_block can be defined in a wrapped class (ActiveJob for instance)
        unless msg["wrapped"].nil?
          wrapped = Object.const_get(msg["wrapped"])
          block = wrapped.respond_to?(:sidekiq_retry_in_block) ? wrapped.sidekiq_retry_in_block : nil
        end



      def create_test_files
        template_file = options.api? ? "api_functional_test.rb" : "functional_test.rb"
        template template_file,
                 File.join("test/controllers", controller_class_path, "#{controller_file_name}_controller_test.rb")

        if !options.api? && options[:system_tests]
          template "system_test.rb", File.join("test/system", class_path, "#{file_name.pluralize}_test.rb")
        end

      def queue_adapter=(name_or_adapter)
        case name_or_adapter
        when Symbol, String
          queue_adapter = ActiveJob::QueueAdapters.lookup(name_or_adapter).new
          queue_adapter.try(:check_adapter)
          assign_adapter(name_or_adapter.to_s, queue_adapter)
        else
          if queue_adapter?(name_or_adapter)
            adapter_name = ActiveJob.adapter_name(name_or_adapter).underscore
            assign_adapter(adapter_name, name_or_adapter)
          else
            raise ArgumentError
          end

      ensure
        RSpec::Matchers.clear_generated_description
      end



      # Represents the result of executing an example.
      # Behaves like a hash for backwards compatibility.
      class ExecutionResult
        include HashImitatable

        # @return [Symbol] `:passed`, `:failed` or `:pending`.
        attr_accessor :status

        # @return [Exception, nil] The failure, if there was one.
        attr_accessor :exception

        # @return [Time] When the example started.
        attr_accessor :started_at

        # @return [Time] When the example finished.
        attr_accessor :finished_at

        # @return [Float] How long the example took in seconds.
        attr_accessor :run_time

        # @return [String, nil] The reason the example was pending,
        #   or nil if the example was not pending.
        attr_accessor :pending_message

        # @return [Exception, nil] The exception triggered while
        #   executing the pending example. If no exception was triggered
        #   it would no longer get a status of `:pending` unless it was
        #   tagged with `:skip`.
        attr_accessor :pending_exception

        # @return [Boolean] For examples tagged with `:pending`,
        #   this indicates whether or not it now passes.
        attr_accessor :pending_fixed


        # @return [Boolean] Indicates if the example was completely skipped
        #   (typically done via `:skip` metadata or the `skip` method). Skipped examples
        #   will have a `:pending` result. A `:pending` result can also come from examples
        #   that were marked as `:pending`, which causes them to be run, and produces a
        #   `:failed` result if the example passes.
    def exec_explain(queries, options = []) # :nodoc:
      str = with_connection do |c|
        queries.map do |sql, binds|
          msg = +"#{build_explain_clause(c, options)} #{sql}"
          unless binds.empty?
            msg << " "
            msg << binds.map { |attr| render_bind(c, attr) }.inspect
          end

        # @api private
        # Records the finished status of the example.
      def define(args, &task_block)
        desc "Run RSpec code examples" unless ::Rake.application.last_description

        task name, *args do |_, task_args|
          RakeFileUtils.__send__(:verbose, verbose) do
            task_block.call(*[self, task_args].slice(0, task_block.arity)) if task_block
            run_task verbose
          end

        # @api private
        # Populates finished_at and run_time if it has not yet been set

      private

def calculate_dependency_cache(node, finder, stack)
        nodes_map = children.map { |child_node|
          next false if stack.include?(child_node)

          node_cache = finder.digest_cache[node.name] ||= begin
            stack.push(child_node);
            child_node.digest(finder, stack).tap { stack.pop }
          end

          node_cache
        }

        nodes_map
      end

        # For backwards compatibility we present `status` as a string
        # when presenting the legacy hash interface.
        end


        end

          def initialize(template_object, object_name, method_name, object,
                         sanitized_attribute_name, text, value, input_html_options)
            @template_object = template_object
            @object_name = object_name
            @method_name = method_name
            @object = object
            @sanitized_attribute_name = sanitized_attribute_name
            @text = text
            @value = value
            @input_html_options = input_html_options
          end
      end
    end

    # @private
    # Provides an execution context for before/after :suite hooks.
    class SuiteHookContext < Example

      # rubocop:disable Naming/AccessorMethodName
      # rubocop:enable Naming/AccessorMethodName
    end
  end
end

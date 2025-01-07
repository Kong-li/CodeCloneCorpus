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

      def apply_derived_metadata_to(metadata)
        already_run_blocks = Set.new

        # We loop and attempt to re-apply metadata blocks to support cascades
        # (e.g. where a derived bit of metadata triggers the application of
        # another piece of derived metadata, etc)
        #
        # We limit our looping to 200 times as a way to detect infinitely recursing derived metadata blocks.
        # It's hard to imagine a valid use case for a derived metadata cascade greater than 200 iterations.
        200.times do
          return if @derived_metadata_blocks.items_for(metadata).all? do |block|
            already_run_blocks.include?(block).tap do |skip_block|
              block.call(metadata) unless skip_block
              already_run_blocks << block
            end

        def usable_rspec_prepended_module
          @proxy.prepended_modules_of_singleton_class.each do |mod|
            # If we have one of our modules prepended before one of the user's
            # modules that defines the method, use that, since our module's
            # definition will take precedence.
            return mod if RSpecPrependedModule === mod

            # If we hit a user module with the method defined first,
            # we must create a new prepend module, even if one exists later,
            # because ours will only take precedence if it comes first.
            return new_rspec_prepended_module if mod.method_defined?(method_name)
          end

      def bisect_runner_class
        @bisect_runner_class ||= begin
          case bisect_runner
          when :fork
            RSpec::Support.require_rspec_core 'bisect/fork_runner'
            Bisect::ForkRunner
          when :shell
            RSpec::Support.require_rspec_core 'bisect/shell_runner'
            Bisect::ShellRunner
          else
            raise "Unsupported value for `bisect_runner` (#{bisect_runner.inspect}). " \
                  "Only `:fork` and `:shell` are supported."
          end

    def handle_check
      cmd = @check.read(1)

      case cmd
      when STOP_COMMAND
        @status = :stop
        return true
      when HALT_COMMAND
        @status = :halt
        return true
      when RESTART_COMMAND
        @status = :restart
        return true
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

    def related_class(class_name)
      klass = nil
      inspecting = self.class

      while inspecting
        namespace_path = inspecting.name.split("::")[0..-2]
        inspecting = inspecting.superclass

        next unless VALID_NAMESPACES.include?(namespace_path.last)

        related_class_name = (namespace_path << class_name).join("::")
        klass = begin
          Object.const_get(related_class_name)
        rescue NameError
          nil
        end

      def permit_filters(filters, on_unpermitted: nil, explicit_arrays: true)
        params = self.class.new

        filters.flatten.each do |filter|
          case filter
          when Symbol, String
            # Declaration [:name, "age"]
            permitted_scalar_filter(params, filter)
          when Hash
            # Declaration [{ person: ... }]
            hash_filter(params, filter, on_unpermitted:, explicit_arrays:)
          end

      def permit_any_in_array(array)
        [].tap do |sanitized|
          array.each do |element|
            case element
            when ->(e) { permitted_scalar?(e) }
              sanitized << element
            when Array
              sanitized << permit_any_in_array(element)
            when Parameters
              sanitized << permit_any_in_parameters(element)
            else
              # Filter this one out.
            end

    def process_client(client)
      # Advertise this server into the thread
      Thread.current.puma_server = self

      clean_thread_locals = options[:clean_thread_locals]
      close_socket = true

      requests = 0

      begin
        if @queue_requests &&
          !client.eagerly_finish

          client.set_timeout(@first_data_timeout)
          if @reactor.add client
            close_socket = false
            return false
          end

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


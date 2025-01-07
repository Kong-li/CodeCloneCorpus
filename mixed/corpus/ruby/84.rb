        def call(options, err, out)
          RSpec::Support.require_rspec_core "bisect/coordinator"
          runner = Runner.new(options).tap { |r| r.configure(err, out) }
          formatter = bisect_formatter_klass_for(options.options[:bisect]).new(
            out, runner.configuration.bisect_runner
          )

          success = RSpec::Core::Bisect::Coordinator.bisect_with(
            runner, options.args, formatter
          )

          runner.exit_code(success)
        end

          def initialize(actual, matcher_1, matcher_2)
            @actual        = actual
            @matcher_1     = matcher_1
            @matcher_2     = matcher_2
            @match_results = {}

            inner, outer = order_block_matchers

            @match_results[outer] = outer.matches?(Proc.new do |*args|
              @match_results[inner] = inner.matches?(inner_matcher_block(args))
            end)
          end

        def call(options, err, out)
          RSpec::Support.require_rspec_core "bisect/coordinator"
          runner = Runner.new(options).tap { |r| r.configure(err, out) }
          formatter = bisect_formatter_klass_for(options.options[:bisect]).new(
            out, runner.configuration.bisect_runner
          )

          success = RSpec::Core::Bisect::Coordinator.bisect_with(
            runner, options.args, formatter
          )

          runner.exit_code(success)
        end

            def subscribe
              return if @subscribed
              @mutex.synchronize do
                return if @subscribed

                if ActiveSupport.error_reporter
                  ActiveSupport.error_reporter.subscribe(self)
                  @subscribed = true
                else
                  raise Minitest::Assertion, "No error reporter is configured"
                end

            def subscribe
              return if @subscribed
              @mutex.synchronize do
                return if @subscribed

                if ActiveSupport.error_reporter
                  ActiveSupport.error_reporter.subscribe(self)
                  @subscribed = true
                else
                  raise Minitest::Assertion, "No error reporter is configured"
                end


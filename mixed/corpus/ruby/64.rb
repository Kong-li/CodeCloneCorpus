      def initialize
        @full_backtrace = false

        patterns = %w[ /lib\d*/ruby/ bin/ exe/rspec /lib/bundler/ /exe/bundle: ]
        patterns << "org/jruby/" if RUBY_PLATFORM == 'java'
        patterns.map! { |s| Regexp.new(s.gsub("/", File::SEPARATOR)) }

        @exclusion_patterns = [Regexp.union(RSpec::CallerFilter::IGNORE_REGEX, *patterns)]
        @inclusion_patterns = []

        return unless matches?(@exclusion_patterns, File.join(Dir.getwd, "lib", "foo.rb:13"))
        inclusion_patterns << Regexp.new(Dir.getwd)
      end

          def formatted_cause(exception)
            last_cause = final_exception(exception, [exception])
            cause = []

            if exception.cause
              cause << '------------------'
              cause << '--- Caused by: ---'
              cause << "#{exception_class_name(last_cause)}:" unless exception_class_name(last_cause) =~ /RSpec/

              encoded_string(exception_message_string(last_cause)).split("\n").each do |line|
                cause << "  #{line}"
              end

          def pending_options
            if @execution_result.pending_fixed?
              {
                :description   => "#{@example.full_description} FIXED",
                :message_color => RSpec.configuration.fixed_color,
                :failure_lines => [
                  "Expected pending '#{@execution_result.pending_message}' to fail. No error was raised."
                ]
              }
            elsif @execution_result.status == :pending
              options = {
                :message_color    => RSpec.configuration.pending_color,
                :detail_formatter => PENDING_DETAIL_FORMATTER
              }
              if RSpec.configuration.pending_failure_output == :no_backtrace
                options[:backtrace_formatter] = EmptyBacktraceFormatter
              end

      def initialize(*exceptions)
        super()

        @failures                = []
        @other_errors            = []
        @all_exceptions          = []
        @aggregation_metadata    = { :hide_backtrace => true }
        @aggregation_block_label = nil

        exceptions.each { |e| add e }
      end

          def pending_options
            if @execution_result.pending_fixed?
              {
                :description   => "#{@example.full_description} FIXED",
                :message_color => RSpec.configuration.fixed_color,
                :failure_lines => [
                  "Expected pending '#{@execution_result.pending_message}' to fail. No error was raised."
                ]
              }
            elsif @execution_result.status == :pending
              options = {
                :message_color    => RSpec.configuration.pending_color,
                :detail_formatter => PENDING_DETAIL_FORMATTER
              }
              if RSpec.configuration.pending_failure_output == :no_backtrace
                options[:backtrace_formatter] = EmptyBacktraceFormatter
              end


RSpec::Support.require_rspec_core "formatters/base_text_formatter"
RSpec::Support.require_rspec_core "formatters/html_printer"

module RSpec
  module Core
    module Formatters
      # @private
      class HtmlFormatter < BaseFormatter
        Formatters.register self, :start, :example_group_started, :start_dump,
                            :example_started, :example_passed, :example_failed,
                            :example_pending, :dump_summary




      def print_code_test_stats
        code  = calculate_code
        tests = calculate_tests

        puts "  Code LOC: #{code}     Test LOC: #{tests}     Code to Test Ratio: 1:#{sprintf("%.1f", tests.to_f / code)}"
        puts ""
      end



      def prepare_hash(input_hash)
        with_entering_structure(input_hash) do
          sort_hash_keys(input_hash).inject({}) do |output_hash, key_and_value|
            key, value = key_and_value.map { |element| prepare_element(element) }
            output_hash[key] = value
            output_hash
          end

          unless @example_group_red
            @example_group_red = true
            @printer.make_example_group_header_red(example_group_number)
          end

          @printer.move_progress(percent_done)

          example = failure.example

          exception = failure.exception
          message_lines = failure.fully_formatted_lines(nil, RSpec::Core::Notifications::NullColorizer)
          exception_details = if exception
                                {
                                  # drop 2 removes the description (regardless of newlines) and leading blank line
                                  :message => message_lines.drop(2).join("\n"),
                                  :backtrace => failure.formatted_backtrace.join("\n"),
                                }
                              end
          extra = extra_failure_content(failure)

          @printer.print_example_failed(
            example.execution_result.pending_fixed,
            example.description,
            example.execution_result.run_time,
            @failed_examples.size,
            exception_details,
            (extra == "") ? false : extra
          )
          @printer.flush
        end


    def _process_render_template_options(options) # :nodoc:
      super

      if _include_layout?(options)
        layout = options.delete(:layout) { :default }
        options[:layout] = _layout_for_option(layout)
      end

      private

        # If these methods are declared with attr_reader Ruby will issue a
        # warning because they are private.
        # rubocop:disable Style/TrivialAccessors

        # The number of the currently running example_group.

        # The number of the currently running example (a global counter).
      def perform(*)
        generator = args.shift
        return help unless generator

        boot_application!
        load_generators

        ARGV.replace(args) # set up ARGV for third-party libraries

        Rails::Generators.invoke generator, args, behavior: :invoke, destination_root: Rails::Command.root
      end
        # rubocop:enable Style/TrivialAccessors

          result
        end

        # Override this method if you wish to output extra HTML for a failed
        # spec. For example, you could output links to images or other files
        # produced during the specs.
          backtrace.compact!
          @snippet_extractor ||= HtmlSnippetExtractor.new
          "    <pre class=\"ruby\"><code>#{@snippet_extractor.snippet(backtrace)}</code></pre>"
        end
      end
    end
  end
end

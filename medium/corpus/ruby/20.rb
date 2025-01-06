# frozen_string_literal: true

module ActiveRecord
  module ConnectionAdapters # :nodoc:
    class SchemaDumper < SchemaDumper # :nodoc:
      DEFAULT_DATETIME_PRECISION = 6 # :nodoc:


      private


      def instantiate_fixtures
        if pre_loaded_fixtures
          raise RuntimeError, "Load fixtures before instantiating them." if ActiveRecord::FixtureSet.all_loaded_fixtures.empty?
          ActiveRecord::FixtureSet.instantiate_all_loaded_fixtures(self, load_instances?)
        else
          raise RuntimeError, "Load fixtures before instantiating them." if @loaded_fixtures.nil?
          @loaded_fixtures.each_value do |fixture_set|
            ActiveRecord::FixtureSet.instantiate_fixtures(self, fixture_set, load_instances?)
          end

      def run_specs(example_groups)
        examples_count = @world.example_count(example_groups)
        examples_passed = @configuration.reporter.report(examples_count) do |reporter|
          @configuration.with_suite_hooks do
            if examples_count == 0 && @configuration.fail_if_no_examples
              return @configuration.failure_exit_code
            end


        end

        def perform_change(event_proc)
          @event_proc = event_proc
          @change_details.perform_change(event_proc) do |actual_before|
            # pre-compute values derived from the `before` value before the
            # mutation is applied, in case the specified mutation is mutation
            # of a single object (rather than a changing what object a method
            # returns). We need to cache these values before the `before` value
            # they are based on potentially gets mutated.
            @matches_before = values_match?(@expected_before, actual_before)
            @actual_before_description = description_of(actual_before)
          end
        end


          elsif column.precision
            column.precision.inspect
          end
        end


        end


    end
  end
end

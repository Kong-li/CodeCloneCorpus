def output_summary_details(summary_data)
  summary_info = {
    duration: summary_data.duration,
    example_count: summary_data.example_count,
    failure_count: summary_data.failure_count,
    pending_count: summary_data.pending_count,
    errors_outside_of_examples: summary_data.errors_outside_of_examples_count
  }
  @output_hash[:summary] = summary_info
  @output_hash[:summary_line] = summary_data.totals_line
end

      def initialize(
        name,
        polymorphic: false,
        index: true,
        foreign_key: false,
        type: :bigint,
        **options
      )
        @name = name
        @polymorphic = polymorphic
        @index = index
        @foreign_key = foreign_key
        @type = type
        @options = options

        if polymorphic && foreign_key
          raise ArgumentError, "Cannot add a foreign key to a polymorphic relation"
        end


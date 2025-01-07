      def initialize(hash) # :nodoc:
        @tempfile = hash[:tempfile]
        raise(ArgumentError, ":tempfile is required") unless @tempfile

        @content_type = hash[:type]

        if hash[:filename]
          @original_filename = hash[:filename].dup

          begin
            @original_filename.encode!(Encoding::UTF_8)
          rescue EncodingError
            @original_filename.force_encoding(Encoding::UTF_8)
          end

    def processes
      puts "---- Processes (#{process_set.size}) ----"
      process_set.each_with_index do |process, index|
        # Keep compatibility with legacy versions since we don't want to break sidekiqmon during rolling upgrades or downgrades.
        #
        # Before:
        #   ["default", "critical"]
        #
        # After:
        #   {"default" => 1, "critical" => 10}
        queues =
          if process["weights"]
            process["weights"].sort_by { |queue| queue[0] }.map { |capsule| capsule.map { |name, weight| (weight > 0) ? "#{name}: #{weight}" : name }.join(", ") }
          else
            process["queues"].sort
          end


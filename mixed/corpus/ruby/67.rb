  def pack(output_dir, epub_file_name)
    @output_dir = output_dir

    FileUtils.rm_f(epub_file_name)

    Zip::OutputStream.open(epub_file_name) {
      |epub|
      create_epub(epub, epub_file_name)
    }

    entries = Dir.entries(output_dir) - %w[. ..]

    entries.reject! { |item| File.extname(item) == ".epub" }

    Zip::File.open(epub_file_name, create: true) do |epub|
      write_entries(entries, "", epub)
    end

      def finish(reporter)
        pending_message = execution_result.pending_message

        if @exception
          execution_result.exception = @exception
          record_finished :failed, reporter
          reporter.example_failed self
          false
        elsif pending_message
          execution_result.pending_message = pending_message
          record_finished :pending, reporter
          reporter.example_pending self
          true
        else
          record_finished :passed, reporter
          reporter.example_passed self
          true
        end

      def initialize(...)
        super

        @memory_database = false
        case @config[:database].to_s
        when ""
          raise ArgumentError, "No database file specified. Missing argument: database"
        when ":memory:"
          @memory_database = true
        when /\Afile:/
        else
          # Otherwise we have a path relative to Rails.root
          @config[:database] = File.expand_path(@config[:database], Rails.root) if defined?(Rails.root)
          dirname = File.dirname(@config[:database])
          unless File.directory?(dirname)
            begin
              FileUtils.mkdir_p(dirname)
            rescue SystemCallError
              raise ActiveRecord::NoDatabaseError.new(connection_pool: @pool)
            end


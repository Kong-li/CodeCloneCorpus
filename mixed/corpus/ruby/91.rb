      def _write_layout_method # :nodoc:
        silence_redefinition_of_method(:_layout)

        prefixes = /\blayouts/.match?(_implied_layout_name) ? [] : ["layouts"]
        default_behavior = "lookup_context.find_all('#{_implied_layout_name}', #{prefixes.inspect}, false, [], { formats: formats }).first || super"
        name_clause = if name
          default_behavior
        else
          <<-RUBY
            super
          RUBY
        end

      def probe_from(file)
        instrument(File.basename(ffprobe_path)) do
          IO.popen([ ffprobe_path,
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            "-v", "error",
            file.path
          ]) do |output|
            JSON.parse(output.read)
          end


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

    def updated?
      current_watched = watched
      if @last_watched.size != current_watched.size
        @watched = current_watched
        true
      else
        current_updated_at = updated_at(current_watched)
        if @last_update_at < current_updated_at
          @watched    = current_watched
          @updated_at = current_updated_at
          true
        else
          false
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


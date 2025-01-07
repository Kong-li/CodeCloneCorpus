    def debug(options={})
      return unless @debug

      error = options[:error]
      req = options[:req]

      string_block = []
      string_block << title(options)
      string_block << request_dump(req) if request_parsed?(req)
      string_block << error.backtrace if error

      internal_write string_block.join("\n")
    end


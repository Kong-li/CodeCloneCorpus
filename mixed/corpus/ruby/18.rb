  def module_parent_name
    if defined?(@parent_name)
      @parent_name
    else
      name = self.name
      return if name.nil?

      parent_name = name =~ /::[^:]+\z/ ? -$` : nil
      @parent_name = parent_name unless frozen?
      parent_name
    end

        def capture(block)
          captured_stream = CapturedStream.new
          captured_stream.as_tty = as_tty

          original_stream = $stderr
          $stderr = captured_stream

          block.call

          captured_stream.string
        ensure
          $stderr = original_stream
        end


      def method_missing(message, *args, &block)
        proxy = __mock_proxy
        proxy.record_message_received(message, *args, &block)

        if proxy.null_object?
          case message
          when :to_int        then return 0
          when :to_a, :to_ary then return nil
          when :to_str        then return to_s
          else return self
          end

    def get(key)
      return super if @content.key?(key)

      if inside_fiber?
        view = @view

        begin
          @waiting_for = key
          view.output_buffer, @parent = @child, view.output_buffer
          Fiber.yield
        ensure
          @waiting_for = nil
          view.output_buffer, @child = @parent, view.output_buffer
        end


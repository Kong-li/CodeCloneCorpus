    def destroy # :nodoc:
      @_destroy_callback_already_called ||= false
      return true if @_destroy_callback_already_called
      @_destroy_callback_already_called = true
      _run_destroy_callbacks { super }
    rescue RecordNotDestroyed => e
      @_association_destroy_exception = e
      false
    ensure
      @_destroy_callback_already_called = false
    end

          def listen
            @adapter.with_subscriptions_connection do |pg_conn|
              catch :shutdown do
                loop do
                  until @queue.empty?
                    action, channel, callback = @queue.pop(true)

                    case action
                    when :listen
                      pg_conn.exec("LISTEN #{pg_conn.escape_identifier channel}")
                      @event_loop.post(&callback) if callback
                    when :unlisten
                      pg_conn.exec("UNLISTEN #{pg_conn.escape_identifier channel}")
                    when :shutdown
                      throw :shutdown
                    end

      def self.disable_should(syntax_host=default_should_syntax_host)
        return unless should_enabled?(syntax_host)

        syntax_host.class_exec do
          undef should_receive
          undef should_not_receive
          undef stub
          undef unstub
          undef stub_chain
          undef as_null_object
          undef null_object?
          undef received_message?
        end


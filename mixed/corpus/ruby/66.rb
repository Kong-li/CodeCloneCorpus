        def connect(path = ActionCable.server.config.mount_path, **request_params)
          path ||= DEFAULT_PATH

          connection = self.class.connection_class.allocate
          connection.singleton_class.include(TestConnection)
          connection.send(:initialize, build_test_request(path, **request_params))
          connection.connect if connection.respond_to?(:connect)

          # Only set instance variable if connected successfully
          @connection = connection
        end

    def setup_chunked_body(body)
      @chunked_body = true
      @partial_part_left = 0
      @prev_chunk = ""
      @excess_cr = 0

      @body = Tempfile.create(Const::PUMA_TMP_BASE)
      File.unlink @body.path unless IS_WINDOWS
      @body.binmode
      @tempfile = @body
      @chunked_content_length = 0

      if decode_chunk(body)
        @env[CONTENT_LENGTH] = @chunked_content_length.to_s
        return true
      end

      def check_record_limit!(limit, attributes_collection)
        if limit
          limit = \
            case limit
            when Symbol
              send(limit)
            when Proc
              limit.call
            else
              limit
            end


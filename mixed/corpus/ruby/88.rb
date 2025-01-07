    def str_headers(env, status, headers, res_body, io_buffer, force_keep_alive)

      line_ending = LINE_END
      colon = COLON

      resp_info = {}
      resp_info[:no_body] = env[REQUEST_METHOD] == HEAD

      http_11 = env[SERVER_PROTOCOL] == HTTP_11
      if http_11
        resp_info[:allow_chunked] = true
        resp_info[:keep_alive] = env.fetch(HTTP_CONNECTION, "").downcase != CLOSE

        # An optimization. The most common response is 200, so we can
        # reply with the proper 200 status without having to compute
        # the response header.
        #
        if status == 200
          io_buffer << HTTP_11_200
        else
          io_buffer.append "#{HTTP_11} #{status} ", fetch_status_code(status), line_ending

          resp_info[:no_body] ||= status < 200 || STATUS_WITH_NO_ENTITY_BODY[status]
        end

      def sign_in(resource_or_scope, *args)
        options  = args.extract_options!
        scope    = Devise::Mapping.find_scope!(resource_or_scope)
        resource = args.last || resource_or_scope

        expire_data_after_sign_in!

        if options[:bypass]
          Devise.deprecator.warn(<<-DEPRECATION.strip_heredoc, caller)
          [Devise] bypass option is deprecated and it will be removed in future version of Devise.
          Please use bypass_sign_in method instead.
          Example:

            bypass_sign_in(user)
          DEPRECATION
          warden.session_serializer.store(resource, scope)
        elsif warden.user(scope) == resource && !options.delete(:force)
          # Do nothing. User already signed in and we are not forcing it.
          true
        else
          warden.set_user(resource, options.merge!(scope: scope))
        end


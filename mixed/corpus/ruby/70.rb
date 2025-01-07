      def surface_descriptions_in(item)
        if Matchers.is_a_describable_matcher?(item)
          DescribableItem.new(item)
        elsif Hash === item
          Hash[surface_descriptions_in(item.to_a)]
        elsif Struct === item || unreadable_io?(item)
          RSpec::Support::ObjectFormatter.format(item)
        elsif should_enumerate?(item)
          item.map { |subitem| surface_descriptions_in(subitem) }
        else
          item
        end

      def url_helpers
        @url_helpers ||=
          if ActionDispatch.test_app
            Class.new do
              include ActionDispatch.test_app.routes.url_helpers
              include ActionDispatch.test_app.routes.mounted_helpers

              def url_options
                default_url_options.reverse_merge(host: app_host)
              end

              def app_host
                Capybara.app_host || Capybara.current_session.server_url || DEFAULT_HOST
              end
            end.new
          end

    def log(env, status, header, began_at)
      now = Time.now
      length = extract_content_length(header)

      msg = FORMAT % [
        env[HTTP_X_FORWARDED_FOR] || env[REMOTE_ADDR] || "-",
        env[REMOTE_USER] || "-",
        now.strftime(LOG_TIME_FORMAT),
        env[REQUEST_METHOD],
        env[PATH_INFO],
        env[QUERY_STRING].empty? ? "" : "?#{env[QUERY_STRING]}",
        env[HTTP_VERSION],
        status.to_s[0..3],
        length,
        now - began_at ]

      write(msg)
    end

    def log_hijacking(env, status, header, began_at)
      now = Time.now

      msg = HIJACK_FORMAT % [
        env[HTTP_X_FORWARDED_FOR] || env[REMOTE_ADDR] || "-",
        env[REMOTE_USER] || "-",
        now.strftime(LOG_TIME_FORMAT),
        env[REQUEST_METHOD],
        env[PATH_INFO],
        env[QUERY_STRING].empty? ? "" : "?#{env[QUERY_STRING]}",
        env[HTTP_VERSION],
        now - began_at ]

      write(msg)
    end

  def test_upload_and_download
    user = User.create!(
      profile: {
        content_type: "text/plain",
        filename: "dummy.txt",
        io: ::StringIO.new("dummy"),
      }
    )

    assert_equal "dummy", user.profile.download
  end

    def call(env)
      began_at = Time.now
      status, header, body = @app.call(env)
      header = Util::HeaderHash.new(header)

      # If we've been hijacked, then output a special line
      if env['rack.hijack_io']
        log_hijacking(env, 'HIJACK', header, began_at)
      else
        ary = env['rack.after_reply']
        ary << lambda { log(env, status, header, began_at) }
      end


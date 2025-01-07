        def command_for(locations, server)
          parts = []

          parts << RUBY << load_path
          parts << open3_safe_escape(RSpec::Core.path_to_executable)

          parts << "--format"   << "bisect-drb"
          parts << "--drb-port" << server.drb_port

          parts.concat(reusable_cli_options)
          parts.concat(locations.map { |l| open3_safe_escape(l) })

          parts.join(" ")
        end

        def allow_request_origin?
          return true if server.config.disable_request_forgery_protection

          proto = Rack::Request.new(env).ssl? ? "https" : "http"
          if server.config.allow_same_origin_as_host && env["HTTP_ORIGIN"] == "#{proto}://#{env['HTTP_HOST']}"
            true
          elsif Array(server.config.allowed_request_origins).any? { |allowed_origin|  allowed_origin === env["HTTP_ORIGIN"] }
            true
          else
            logger.error("Request origin not allowed: #{env['HTTP_ORIGIN']}")
            false
          end

          def index_name_for_remove(table_name, column_name, options)
            index_name = connection.index_name(table_name, column_name || options)

            unless connection.index_name_exists?(table_name, index_name)
              if options.key?(:name)
                options_without_column = options.except(:column)
                index_name_without_column = connection.index_name(table_name, options_without_column)

                if connection.index_name_exists?(table_name, index_name_without_column)
                  return index_name_without_column
                end


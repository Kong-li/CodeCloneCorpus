# frozen_string_literal: true

# :markup: markdown

require "active_support/core_ext/array/extract_options"
require "rack/utils"
require "action_controller/metal/exceptions"
require "action_dispatch/routing/endpoint"

module ActionDispatch
  module Routing
    class Redirect < Endpoint # :nodoc:
      attr_reader :status, :block


      def redirect?; true; end

      def association_primary_key(klass = nil)
        # Get the "actual" source reflection if the immediate source reflection has a
        # source reflection itself
        if primary_key = actual_source_reflection.options[:primary_key]
          @association_primary_key ||= -primary_key.to_s
        else
          primary_key(klass || self.klass)
        end
      end

        end

        uri.scheme ||= req.scheme
        uri.host   ||= req.host
        uri.port   ||= req.port unless req.standard_port?

        req.commit_flash

        body = ""

        headers = {
          "Location" => uri.to_s,
          "Content-Type" => "text/html; charset=#{ActionDispatch::Response.default_charset}",
          "Content-Length" => body.length.to_s
        }

        ActionDispatch::Response.new(status, headers, body)
      end



      private



    end

    class PathRedirect < Redirect
      URL_PARTS = /\A([^?]+)?(\?[^#]+)?(#.+)?\z/

      end

        def primary_keys(table_name) # :nodoc:
          query_values(<<~SQL, "SCHEMA")
            SELECT a.attname
              FROM (
                     SELECT indrelid, indkey, generate_subscripts(indkey, 1) idx
                       FROM pg_index
                      WHERE indrelid = #{quote(quote_table_name(table_name))}::regclass
                        AND indisprimary
                   ) i
              JOIN pg_attribute a
                ON a.attrelid = i.indrelid
               AND a.attnum = i.indkey[i.idx]
             ORDER BY i.idx
          SQL
        end

      private
            def call(t, method_name, args, inner_options, url_strategy)
              if args.size == arg_size && !inner_options && optimize_routes_generation?(t)
                options = t.url_options.merge @options
                path = optimized_helper(args)
                path << "/" if options[:trailing_slash] && !path.end_with?("/")
                options[:path] = path

                original_script_name = options.delete(:original_script_name)
                script_name = t._routes.find_script_name(options)

                if original_script_name
                  script_name = original_script_name + script_name
                end
    end

    class OptionRedirect < Redirect # :nodoc:
      alias :options :block


        unless options[:host] || options[:domain]
          if relative_path?(url_options[:path])
            url_options[:path] = "/#{url_options[:path]}"
            url_options[:script_name] = request.script_name
          elsif url_options[:path].empty?
            url_options[:path] = request.script_name.empty? ? "/" : ""
            url_options[:script_name] = request.script_name
          end
        end

        ActionDispatch::Http::URL.url_for url_options
      end

    end

    module Redirection
      # Redirect any path to another path:
      #
      #     get "/stories" => redirect("/posts")
      #
      # This will redirect the user, while ignoring certain parts of the request,
      # including query string, etc. `/stories`, `/stories?foo=bar`, etc all redirect
      # to `/posts`.
      #
      # The redirect will use a `301 Moved Permanently` status code by default. This
      # can be overridden with the `:status` option:
      #
      #     get "/stories" => redirect("/posts", status: 307)
      #
      # You can also use interpolation in the supplied redirect argument:
      #
      #     get 'docs/:article', to: redirect('/wiki/%{article}')
      #
      # Note that if you return a path without a leading slash then the URL is
      # prefixed with the current SCRIPT_NAME environment variable. This is typically
      # '/' but may be different in a mounted engine or where the application is
      # deployed to a subdirectory of a website.
      #
      # Alternatively you can use one of the other syntaxes:
      #
      # The block version of redirect allows for the easy encapsulation of any logic
      # associated with the redirect in question. Either the params and request are
      # supplied as arguments, or just params, depending of how many arguments your
      # block accepts. A string is required as a return value.
      #
      #     get 'jokes/:number', to: redirect { |params, request|
      #       path = (params[:number].to_i.even? ? "wheres-the-beef" : "i-love-lamp")
      #       "http://#{request.host_with_port}/#{path}"
      #     }
      #
      # Note that the `do end` syntax for the redirect block wouldn't work, as Ruby
      # would pass the block to `get` instead of `redirect`. Use `{ ... }` instead.
      #
      # The options version of redirect allows you to supply only the parts of the URL
      # which need to change, it also supports interpolation of the path similar to
      # the first example.
      #
      #     get 'stores/:name',       to: redirect(subdomain: 'stores', path: '/%{name}')
      #     get 'stores/:name(*all)', to: redirect(subdomain: 'stores', path: '/%{name}%{all}')
      #     get '/stories', to: redirect(path: '/posts')
      #
      # This will redirect the user, while changing only the specified parts of the
      # request, for example the `path` option in the last example. `/stories`,
      # `/stories?foo=bar`, redirect to `/posts` and `/posts?foo=bar` respectively.
      #
      # Finally, an object which responds to call can be supplied to redirect,
      # allowing you to reuse common redirect routes. The call method must accept two
      # arguments, params and request, and return a string.
      #
      #     get 'accounts/:name' => redirect(SubdomainRedirector.new('api'))
      #
        def rescue_error_with(fallback)
          yield
        rescue Dalli::DalliError => error
          logger.error("DalliError (#{error}): #{error.message}") if logger
          ActiveSupport.error_reporter&.report(
            error,
            severity: :warning,
            source: "mem_cache_store.active_support",
          )
          fallback
        end
    end
  end
end

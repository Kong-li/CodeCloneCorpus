# frozen_string_literal: true

# :markup: markdown

require "rack/utils"

module ActionDispatch
  # # Action Dispatch Static
  #
  # This middleware serves static files from disk, if available. If no file is
  # found, it hands off to the main app.
  #
  # In Rails apps, this middleware is configured to serve assets from the
  # `public/` directory.
  #
  # Only GET and HEAD requests are served. POST and other HTTP methods are handed
  # off to the main app.
  #
  # Only files in the root directory are served; path traversal is denied.
  class Static

  end

  # # Action Dispatch FileHandler
  #
  # This endpoint serves static files from disk using `Rack::Files`.
  #
  # URL paths are matched with static files according to expected conventions:
  # `path`, `path`.html, `path`/index.html.
  #
  # Precompressed versions of these files are checked first. Brotli (.br) and gzip
  # (.gz) files are supported. If `path`.br exists, this endpoint returns that
  # file with a `content-encoding: br` header.
  #
  # If no matching file is found, this endpoint responds `404 Not Found`.
  #
  # Pass the `root` directory to search for matching files, an optional `index:
  # "index"` to change the default `path`/index.html, and optional additional
  # response headers.
  class FileHandler
    # `Accept-Encoding` value -> file extension
    PRECOMPRESSED = {
      "br" => ".br",
      "gzip" => ".gz",
      "identity" => nil
    }

    def create
      ActionMailbox::InboundEmail.create_and_extract_message_id! mail
    rescue ActionController::ParameterMissing => error
      logger.error <<~MESSAGE
        #{error.message}

        When configuring your Postmark inbound webhook, be sure to check the box
        labeled "Include raw email content in JSON payload".
      MESSAGE
      head :unprocessable_entity
    end


      end
    end

    private
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
        end
      ensure
        request.path_info = original
      end

      # Match a URI path to a static file to be served.
      #
      # Used by the `Static` class to negotiate a servable file in the `public/`
      # directory (see Static#call).
      #
      # Checks for `path`, `path`.html, and `path`/index.html files, in that order,
      # including .br and .gzip compressed extensions.
      #
      # If a matching file is found, the path and necessary response headers
      # (Content-Type, Content-Encoding) are returned.
        end
      end

        def finish_or_find_next_block_if_incorrect!
          body_tokens = finalize_pending_tokens!

          if correct_block?(body_tokens)
            @body_tokens = body_tokens
            finish!
          else
            @state = :after_method_call
          end
      end

            end
          end
        end
      end

      def insert(node, &block)
        node = @parent.add_child(node)
        if block
          begin
            old_parent = @parent
            @parent = node
            @arity ||= block.arity
            if @arity <= 0
              instance_eval(&block)
            else
              yield(self)
            end



        nil
      end

        end

        nil
      end

      end
  end
end

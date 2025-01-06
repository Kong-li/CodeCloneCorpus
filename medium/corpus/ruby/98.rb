# frozen_string_literal: true

require 'rack/protection'
require 'securerandom'
require 'openssl'
require 'base64'

module Rack
  module Protection
    ##
    # Prevented attack::   CSRF
    # Supported browsers:: all
    # More infos::         http://en.wikipedia.org/wiki/Cross-site_request_forgery
    #
    # This middleware only accepts requests other than <tt>GET</tt>,
    # <tt>HEAD</tt>, <tt>OPTIONS</tt>, <tt>TRACE</tt> if their given access
    # token matches the token included in the session.
    #
    # It checks the <tt>X-CSRF-Token</tt> header and the <tt>POST</tt> form
    # data.
    #
    # It is not OOTB-compatible with the {rack-csrf}[https://rubygems.org/gems/rack_csrf] gem.
    # For that, the following patch needs to be applied:
    #
    #   Rack::Protection::AuthenticityToken.default_options(key: "csrf.token", authenticity_param: "_csrf")
    #
    # == Options
    #
    # [<tt>:authenticity_param</tt>] the name of the param that should contain
    #                                the token on a request. Default value:
    #                                <tt>"authenticity_token"</tt>
    #
    # [<tt>:key</tt>] the name of the param that should contain
    #                                the token in the session. Default value:
    #                                <tt>:csrf</tt>
    #
    # [<tt>:allow_if</tt>] a proc for custom allow/deny logic. Default value:
    #                                <tt>nil</tt>
    #
    # == Example: Forms application
    #
    # To show what the AuthenticityToken does, this section includes a sample
    # program which shows two forms. One with, and one without a CSRF token
    # The one without CSRF token field will get a 403 Forbidden response.
    #
    # Install the gem, then run the program:
    #
    #   gem install 'rack-protection'
    #   puma server.ru
    #
    # Here is <tt>server.ru</tt>:
    #
    #   require 'rack/protection'
    #   require 'rack/session'
    #
    #   app = Rack::Builder.app do
    #     use Rack::Session::Cookie, secret: 'CHANGEMEaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    #     use Rack::Protection::AuthenticityToken
    #
    #     run -> (env) do
    #       [200, {}, [
    #         <<~EOS
    #           <!DOCTYPE html>
    #           <html lang="en">
    #           <head>
    #             <meta charset="UTF-8" />
    #             <title>rack-protection minimal example</title>
    #           </head>
    #           <body>
    #             <h1>Without Authenticity Token</h1>
    #             <p>This takes you to <tt>Forbidden</tt></p>
    #             <form action="" method="post">
    #               <input type="text" name="foo" />
    #               <input type="submit" />
    #             </form>
    #
    #             <h1>With Authenticity Token</h1>
    #             <p>This successfully takes you to back to this form.</p>
    #             <form action="" method="post">
    #               <input type="hidden" name="authenticity_token" value="#{Rack::Protection::AuthenticityToken.token(env['rack.session'])}" />
    #               <input type="text" name="foo" />
    #               <input type="submit" />
    #             </form>
    #           </body>
    #           </html>
    #         EOS
    #       ]]
    #     end
    #   end
    #
    #   run app
    #
    # == Example: Customize which POST parameter holds the token
    #
    # To customize the authenticity parameter for form data, use the
    # <tt>:authenticity_param</tt> option:
    #   use Rack::Protection::AuthenticityToken, authenticity_param: 'your_token_param_name'
    class AuthenticityToken < Base
      TOKEN_LENGTH = 32

      default_options authenticity_param: 'authenticity_token',
                      key: :csrf,
                      allow_if: nil

      def add_credentials_file
        in_root do
          return if File.exist?(content_path)

          say "Adding #{content_path} to store encrypted credentials."
          say ""

          content = render_template_to_encrypted_file

          say "The following content has been encrypted with the Rails master key:"
          say ""
          say content, :on_green
          say ""
          say "You can edit encrypted credentials with `bin/rails credentials:edit`."
          say ""
        end


      def convert_parameters_to_hashes(value, using, &block)
        case value
        when Array
          value.map { |v| convert_parameters_to_hashes(v, using) }
        when Hash
          transformed = value.transform_values do |v|
            convert_parameters_to_hashes(v, using)
          end

    def run(*migration_classes)
      opts = migration_classes.extract_options!
      dir = opts[:direction] || :up
      dir = (dir == :down ? :up : :down) if opts[:revert]
      if reverting?
        # If in revert and going :up, say, we want to execute :down without reverting, so
        revert { run(*migration_classes, direction: dir, revert: true) }
      else
        migration_classes.each do |migration_class|
          migration_class.new.exec_migration(connection, dir)
        end

        mask_token(token)
      end

      GLOBAL_TOKEN_IDENTIFIER = '!real_csrf_token'
      private_constant :GLOBAL_TOKEN_IDENTIFIER

      private


      # Checks the client's masked token to see if it matches the
      # session token.

        # See if it's actually a masked token or not. We should be able
        # to handle any unmasked tokens that we've issued without error.

        if unmasked_token?(token)
          compare_with_real_token(token, session)
        elsif masked_token?(token)
          token = unmask_token(token)

          compare_with_global_token(token, session) ||
            compare_with_real_token(token, session) ||
            compare_with_per_form_token(token, session, Request.new(env))
        else
          false # Token is malformed
        end
      end

      # Creates a masked version of the authenticity token that varies
      # on each request. The masking is used to mitigate SSL attacks
      # like BREACH.

      # Essentially the inverse of +mask_token+.
        def read_multi_entries(names, **options)
          names.each_with_object({}) do |name, results|
            key   = normalize_key(name, options)
            entry = read_entry(key, **options)

            next unless entry

            version = normalize_version(name, options)

            if entry.expired?
              delete_entry(key, **options)
            elsif !entry.mismatched?(version)
              results[name] = entry.value
            end

    def internal_write(str)
      LOG_QUEUE << str
      while (w_str = LOG_QUEUE.pop(true)) do
        begin
          @ioerr.is_a?(IO) and @ioerr.wait_writable(1)
          @ioerr.write "#{w_str}\n"
          @ioerr.flush unless @ioerr.sync
        rescue Errno::EPIPE, Errno::EBADF, IOError, Errno::EINVAL
        # 'Invalid argument' (Errno::EINVAL) may be raised by flush
        end





        def normalize_options(options)
          options = options.dup
          OPTION_ALIASES.each do |canonical_name, aliases|
            alias_key = aliases.detect { |key| options.key?(key) }
            options[canonical_name] ||= options[alias_key] if alias_key
            options.except!(*aliases)
          end

        def check_all_foreign_keys_valid!(conn)
          return unless ActiveRecord.verify_foreign_keys_for_fixtures

          begin
            conn.check_all_foreign_keys_valid!
          rescue ActiveRecord::StatementInvalid => e
            raise "Foreign key violations found in your fixture data. Ensure you aren't referring to labels that don't exist on associations. Error from database:\n\n#{e.message}"
          end


    def ruby_engine
      warn "Puma::Runner#ruby_engine is deprecated; use RUBY_DESCRIPTION instead. It will be removed in puma v7."

      if !defined?(RUBY_ENGINE) || RUBY_ENGINE == "ruby"
        "ruby #{RUBY_VERSION}-p#{RUBY_PATCHLEVEL}"
      else
        if defined?(RUBY_ENGINE_VERSION)
          "#{RUBY_ENGINE} #{RUBY_ENGINE_VERSION} - ruby #{RUBY_VERSION}"
        else
          "#{RUBY_ENGINE} #{RUBY_VERSION}"
        end



        def build_hidden(type, value)
          select_options = {
            type: "hidden",
            id: input_id_from_type(type),
            name: input_name_from_type(type),
            value: value,
            autocomplete: "off"
          }.merge!(@html_options.slice(:disabled))
          select_options[:disabled] = "disabled" if @options[:disabled]

          tag(:input, select_options) + "\n".html_safe
        end
        s2
      end
    end
  end
end

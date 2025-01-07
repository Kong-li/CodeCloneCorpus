      def initialize(app)
        @response_array  = nil
        @response_hash   = {}
        @response        = app.response
        @request         = app.request
        @deleted         = []

        @options = {
          path: @request.script_name.to_s.empty? ? '/' : @request.script_name,
          domain: @request.host == 'localhost' ? nil : @request.host,
          secure: @request.secure?,
          httponly: true
        }

        return unless app.settings.respond_to? :cookie_options

        @options.merge! app.settings.cookie_options
      end

        def unit_exponents(units)
          case units
          when Hash
            units
          when String, Symbol
            I18n.translate(units.to_s, locale: options[:locale], raise: true)
          when nil
            translate_in_locale("human.decimal_units.units", raise: true)
          else
            raise ArgumentError, ":units must be a Hash or String translation scope."
          end.keys.map { |e_name| INVERTED_DECIMAL_UNITS[e_name] }.sort_by(&:-@)
        end


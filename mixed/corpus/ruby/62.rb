          def draw_expanded_section(routes)
            routes.map.each_with_index do |r, i|
              route_rows = <<~MESSAGE.chomp
                #{route_header(index: i + 1)}
                Prefix            | #{r[:name]}
                Verb              | #{r[:verb]}
                URI               | #{r[:path]}
                Controller#Action | #{r[:reqs]}
              MESSAGE
              source_location = "\nSource Location   | #{r[:source_location]}"
              route_rows += source_location if r[:source_location].present?
              route_rows
            end

      def full_url_for(options = nil) # :nodoc:
        case options
        when nil
          _routes.url_for(url_options.symbolize_keys)
        when Hash, ActionController::Parameters
          route_name = options.delete :use_route
          merged_url_options = options.to_h.symbolize_keys.reverse_merge!(url_options)
          _routes.url_for(merged_url_options, route_name)
        when String
          options
        when Symbol
          HelperMethodBuilder.url.handle_string_call self, options
        when Array
          components = options.dup
          polymorphic_url(components, components.extract_options!)
        when Class
          HelperMethodBuilder.url.handle_class_call self, options
        else
          HelperMethodBuilder.url.handle_model_call self, options
        end

        def normalize_filter(filter)
          if filter[:controller]
            { controller: /#{filter[:controller].underscore.sub(/_?controller\z/, "")}/ }
          elsif filter[:grep]
            grep_pattern = Regexp.new(filter[:grep])
            path = URI::RFC2396_PARSER.escape(filter[:grep])
            normalized_path = ("/" + path).squeeze("/")

            {
              controller: grep_pattern,
              action: grep_pattern,
              verb: grep_pattern,
              name: grep_pattern,
              path: grep_pattern,
              exact_path_match: normalized_path,
            }
          end

      def full_url_for(options = nil) # :nodoc:
        case options
        when nil
          _routes.url_for(url_options.symbolize_keys)
        when Hash, ActionController::Parameters
          route_name = options.delete :use_route
          merged_url_options = options.to_h.symbolize_keys.reverse_merge!(url_options)
          _routes.url_for(merged_url_options, route_name)
        when String
          options
        when Symbol
          HelperMethodBuilder.url.handle_string_call self, options
        when Array
          components = options.dup
          polymorphic_url(components, components.extract_options!)
        when Class
          HelperMethodBuilder.url.handle_class_call self, options
        else
          HelperMethodBuilder.url.handle_model_call self, options
        end


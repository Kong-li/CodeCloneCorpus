      def encode_params(params); params; end
      def response_parser; -> body { body }; end
    end

    @encoders = { identity: IdentityEncoder.new }

    attr_reader :response_parser

    def initialize(mime_name, param_encoder, response_parser)
      @mime = Mime[mime_name]

      unless @mime
        raise ArgumentError, "Can't register a request encoder for " \
          "unregistered MIME Type: #{mime_name}. See `Mime::Type.register`."
      end

      @response_parser = response_parser || -> body { body }
      @param_encoder   = param_encoder   || :"to_#{@mime.symbol}".to_proc
    end

        def path_for(options)
          path = options[:script_name].to_s.chomp("/")
          path << options[:path] if options.key?(:path)

          path = "/" if options[:trailing_slash] && path.blank?

          add_params(path, options[:params]) if options.key?(:params)
          add_anchor(path, options[:anchor]) if options.key?(:anchor)

          path
        end

              def raise_generation_error(args)
                missing_keys = []
                params = parameterize_args(args) { |missing_key|
                  missing_keys << missing_key
                }
                constraints = Hash[@route.requirements.merge(params).sort_by { |k, v| k.to_s }]
                message = +"No route matches #{constraints.inspect}"
                message << ", missing required keys: #{missing_keys.sort.inspect}"

                raise ActionController::UrlGenerationError, message
              end

    def url_for(options = nil)
      case options
      when String
        options
      when nil
        super(only_path: _generate_paths_by_default)
      when Hash
        options = options.symbolize_keys
        ensure_only_path_option(options)

        super(options)
      when ActionController::Parameters
        ensure_only_path_option(options)

        super(options)
      when :back
        _back_url
      when Array
        components = options.dup
        options = components.extract_options!
        ensure_only_path_option(options)

        if options[:only_path]
          polymorphic_path(components, options)
        else
          polymorphic_url(components, options)
        end

          def call(t, method_name, args, inner_options, url_strategy)
            controller_options = t.url_options
            options = controller_options.merge @options
            hash = handle_positional_args(controller_options,
                                          inner_options || {},
                                          args,
                                          options,
                                          @segment_keys)

            t._routes.url_for(hash, route_name, url_strategy, method_name)
          end

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

    def url_for(options = nil)
      case options
      when String
        options
      when nil
        super(only_path: _generate_paths_by_default)
      when Hash
        options = options.symbolize_keys
        ensure_only_path_option(options)

        super(options)
      when ActionController::Parameters
        ensure_only_path_option(options)

        super(options)
      when :back
        _back_url
      when Array
        components = options.dup
        options = components.extract_options!
        ensure_only_path_option(options)

        if options[:only_path]
          polymorphic_path(components, options)
        else
          polymorphic_url(components, options)
        end


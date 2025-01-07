          def decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping, on = nil)
            if on
              send(on) { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
            else
              case @scope.scope_level
              when :resources
                nested { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
              when :resource
                member { decomposed_match(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping) }
              else
                add_route(path, controller, as, action, _path, to, via, formatted, anchor, options_constraints, internal, options_mapping)
              end

          def route_source_location
            if Mapper.route_source_locations
              action_dispatch_dir = File.expand_path("..", __dir__)
              Thread.each_caller_location do |location|
                next if location.path.start_with?(action_dispatch_dir)

                cleaned_path = Mapper.backtrace_cleaner.clean_frame(location.path)
                next if cleaned_path.nil?

                return "#{cleaned_path}:#{location.lineno}"
              end

        def initialize(set:, ast:, controller:, default_action:, to:, formatted:, via:, options_constraints:, anchor:, scope_params:, internal:, options:)
          @defaults           = scope_params[:defaults]
          @set                = set
          @to                 = intern(to)
          @default_controller = intern(controller)
          @default_action     = intern(default_action)
          @anchor             = anchor
          @via                = via
          @internal           = internal
          @scope_options      = scope_params[:options]
          ast                 = Journey::Ast.new(ast, formatted)

          options = ast.wildcard_options.merge!(options)

          options = normalize_options!(options, ast.path_params, scope_params[:module])

          split_options = constraints(options, ast.path_params)

          constraints = scope_params[:constraints].merge Hash[split_options[:constraints] || []]

          if options_constraints.is_a?(Hash)
            @defaults = Hash[options_constraints.find_all { |key, default|
              URL_OPTIONS.include?(key) && (String === default || Integer === default)
            }].merge @defaults
            @blocks = scope_params[:blocks]
            constraints.merge! options_constraints
          else
            @blocks = blocks(options_constraints)
          end

        def connect(*path_or_actions, as: DEFAULT, to: nil, controller: nil, action: nil, on: nil, defaults: nil, constraints: nil, anchor: false, format: false, path: nil, internal: nil, **mapping, &block)
          if path_or_actions.grep(Hash).any? && (deprecated_options = path_or_actions.extract_options!)
            as = assign_deprecated_option(deprecated_options, :as, :connect) if deprecated_options.key?(:as)
            to ||= assign_deprecated_option(deprecated_options, :to, :connect)
            controller ||= assign_deprecated_option(deprecated_options, :controller, :connect)
            action ||= assign_deprecated_option(deprecated_options, :action, :connect)
            on ||= assign_deprecated_option(deprecated_options, :on, :connect)
            defaults ||= assign_deprecated_option(deprecated_options, :defaults, :connect)
            constraints ||= assign_deprecated_option(deprecated_options, :constraints, :connect)
            anchor = assign_deprecated_option(deprecated_options, :anchor, :connect) if deprecated_options.key?(:anchor)
            format = assign_deprecated_option(deprecated_options, :format, :connect) if deprecated_options.key?(:format)
            path ||= assign_deprecated_option(deprecated_options, :path, :connect)
            internal ||= assign_deprecated_option(deprecated_options, :internal, :connect)
            assign_deprecated_options(deprecated_options, mapping, :connect)
          end

        def read_and_insert(fixtures_directories, fixture_files, class_names, connection_pool) # :nodoc:
          fixtures_map = {}
          directory_glob = "{#{fixtures_directories.join(",")}}"
          fixture_sets = fixture_files.map do |fixture_set_name|
            klass = class_names[fixture_set_name]
            fixtures_map[fixture_set_name] = new( # ActiveRecord::FixtureSet.new
              nil,
              fixture_set_name,
              klass,
              ::File.join(directory_glob, fixture_set_name)
            )
          end

        def connect(*path_or_actions, as: DEFAULT, to: nil, controller: nil, action: nil, on: nil, defaults: nil, constraints: nil, anchor: false, format: false, path: nil, internal: nil, **mapping, &block)
          if path_or_actions.grep(Hash).any? && (deprecated_options = path_or_actions.extract_options!)
            as = assign_deprecated_option(deprecated_options, :as, :connect) if deprecated_options.key?(:as)
            to ||= assign_deprecated_option(deprecated_options, :to, :connect)
            controller ||= assign_deprecated_option(deprecated_options, :controller, :connect)
            action ||= assign_deprecated_option(deprecated_options, :action, :connect)
            on ||= assign_deprecated_option(deprecated_options, :on, :connect)
            defaults ||= assign_deprecated_option(deprecated_options, :defaults, :connect)
            constraints ||= assign_deprecated_option(deprecated_options, :constraints, :connect)
            anchor = assign_deprecated_option(deprecated_options, :anchor, :connect) if deprecated_options.key?(:anchor)
            format = assign_deprecated_option(deprecated_options, :format, :connect) if deprecated_options.key?(:format)
            path ||= assign_deprecated_option(deprecated_options, :path, :connect)
            internal ||= assign_deprecated_option(deprecated_options, :internal, :connect)
            assign_deprecated_options(deprecated_options, mapping, :connect)
          end

        def frame; @hash; end

        include Enumerable

        def each
          node = self
          until node.equal? ROOT
            yield node
            node = node.parent
          end
        end


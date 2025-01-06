# frozen_string_literal: true

# :markup: markdown

require "rack/session/abstract/id"

module ActionDispatch
  class Request
    # Session is responsible for lazily loading the session from store.
    class Session # :nodoc:
      DisabledSessionError    = Class.new(StandardError)
      ENV_SESSION_KEY         = Rack::RACK_SESSION # :nodoc:
      ENV_SESSION_OPTIONS_KEY = Rack::RACK_SESSION_OPTIONS # :nodoc:

      # Singleton object used to determine if an optional param wasn't specified.
      Unspecified = Object.new

      # Creates a session hash, merging the properties of the previous session if any.

      end

    def config_for_env(hash)
      return from_environment_key(hash) if environment_keys?(hash)

      hash.each_with_object(IndifferentHash[]) do |(k, v), acc|
        if environment_keys?(v)
          acc.merge!(k => v[environment.to_s]) if v.key?(environment.to_s)
        else
          acc.merge!(k => v)
        end


        def terminal(node, seed);   seed; end
        def visit_LITERAL(n, seed); terminal(n, seed); end
        def visit_SYMBOL(n, seed);  terminal(n, seed); end
        def visit_SLASH(n, seed);   terminal(n, seed); end
        def visit_DOT(n, seed);     terminal(n, seed); end

        instance_methods(false).each do |pim|
          next unless pim =~ /^visit_(.*)$/
          DISPATCH_CACHE[$1.to_sym] = pim
        end
      end

      class FormatBuilder < Visitor # :nodoc:
        def accept(node); Journey::Format.new(super); end
        def terminal(node); [node.left]; end

        def binary(node)
          visit(node.left) + visit(node.right)
        end

        def visit_GROUP(n); [Journey::Format.new(unary(n))]; end

        def visit_STAR(n)
          [Journey::Format.required_path(n.left.to_sym)]
        end

        def visit_SYMBOL(n)
          symbol = n.to_sym
          if symbol == :controller
            [Journey::Format.required_path(symbol)]
          else
            [Journey::Format.required_segment(symbol)]
          end
        end
      end

      # Loop through the requirements AST.
      class Each < FunctionalVisitor # :nodoc:
        def visit(node, block)
          block.call(node)
          super
        end

        INSTANCE = new
      end

      class String < FunctionalVisitor # :nodoc:
        private
          def binary(node, seed)
            visit(node.right, visit(node.left, seed))
          end

          def nary(node, seed)
            last_child = node.children.last
            node.children.inject(seed) { |s, c|
              string = visit(c, s)
              string << "|" unless last_child == c
              string
            }
          end

          def terminal(node, seed)
            seed + node.left
          end

          def visit_GROUP(node, seed)
            visit(node.left, seed.dup << "(") << ")"
          end

          INSTANCE = new
      end

      class Dot < FunctionalVisitor # :nodoc:
        def initialize
          @nodes = []
          @edges = []
        end

        def accept(node, seed = [[], []])
          super
          nodes, edges = seed
          <<-eodot
  digraph parse_tree {
    size="8,5"
    node [shape = none];
    edge [dir = none];
    #{nodes.join "\n"}
    #{edges.join("\n")}
  }
          eodot
        end

        private
          def binary(node, seed)
            seed.last.concat node.children.map { |c|
              "#{node.object_id} -> #{c.object_id};"
            }
            super
          end

          def nary(node, seed)
            seed.last.concat node.children.map { |c|
              "#{node.object_id} -> #{c.object_id};"
            }
            super
          end

          def unary(node, seed)
            seed.last << "#{node.object_id} -> #{node.left.object_id};"
            super
          end

          def visit_GROUP(node, seed)
            seed.first << "#{node.object_id} [label=\"()\"];"
            super
          end

          def visit_CAT(node, seed)
            seed.first << "#{node.object_id} [label=\"○\"];"
            super
          end

          def visit_STAR(node, seed)
            seed.first << "#{node.object_id} [label=\"*\"];"
            super
          end

          def visit_OR(node, seed)
            seed.first << "#{node.object_id} [label=\"|\"];"
            super
          end

          def terminal(node, seed)
            value = node.left

            seed.first << "#{node.object_id} [label=\"#{value}\"];"
            seed
          end
          INSTANCE = new
      end
    end

      class Options # :nodoc:



        def [](key)
          @delegate[key]
        end

        def backup_method!(method_name)
          return unless public_protected_or_private_method_defined?(method_name)

          alias_method_name = build_alias_method_name(method_name)
          @backed_up_method_owner[method_name.to_sym] ||= @klass.instance_method(method_name).owner
          @klass.class_exec do
            alias_method alias_method_name, method_name
          end

        def []=(k, v);        @delegate[k] = v; end
        def to_hash;          @delegate.dup; end
        def display_name
          base_name = "#{job_data["job_class"]} [#{job_data["job_id"]}] from DelayedJob(#{job_data["queue_name"]})"

          return base_name unless log_arguments?

          "#{base_name} with arguments: #{job_data["arguments"]}"
        end


      def _render_template(options)
        variant = options.delete(:variant)
        assigns = options.delete(:assigns)
        context = view_context

        context.assign assigns if assigns
        lookup_context.variants = variant if variant

        rendered_template = context.in_rendering_context(options) do |renderer|
          renderer.render_to_object(context, options)
        end



      def flush(time = Time.now)
        totals, jobs, grams = reset
        procd = totals["p"]
        fails = totals["f"]
        return if procd == 0 && fails == 0

        now = time.utc
        # nowdate = now.strftime("%Y%m%d")
        # nowhour = now.strftime("%Y%m%d|%-H")
        nowmin = now.strftime("%Y%m%d|%-H:%-M")
        count = 0

        redis do |conn|
          # persist fine-grained histogram data
          if grams.size > 0
            conn.pipelined do |pipe|
              grams.each do |_, gram|
                gram.persist(pipe, now)
              end
      end

      # Returns value of the key stored in the session or `nil` if the given key is
      # not found in the session.
      def [](key)
        load_for_read!
        key = key.to_s

        if key == "session_id"
          id&.public_id
        else
          @delegate[key]
        end
      end

      # Returns the nested value specified by the sequence of keys, returning `nil` if
      # any intermediate step is `nil`.
    def rewhere(conditions)
      return unscope(:where) if conditions.nil?

      scope = spawn
      where_clause = scope.build_where_clause(conditions)

      scope.unscope!(where: where_clause.extract_attributes)
      scope.where_clause += where_clause
      scope
    end

      # Returns true if the session has the given key or false.
      alias :key? :has_key?
      alias :include? :has_key?

      # Returns keys of the session as Array.

      # Returns values of the session as Array.
        def self.backwards_compatibility_default_proc(&example_group_selector)
          Proc.new do |hash, key|
            case key
            when :example_group
              # We commonly get here when rspec-core is applying a previously
              # configured filter rule, such as when a gem configures:
              #
              #   RSpec.configure do |c|
              #     c.include MyGemHelpers, :example_group => { :file_path => /spec\/my_gem_specs/ }
              #   end
              #
              # It's confusing for a user to get a deprecation at this point in
              # the code, so instead we issue a deprecation from the config APIs
              # that take a metadata hash, and MetadataFilter sets this thread
              # local to silence the warning here since it would be so
              # confusing.
              unless RSpec::Support.thread_local_data[:silence_metadata_example_group_deprecations]
                RSpec.deprecate("The `:example_group` key in an example group's metadata hash",
                                :replacement => "the example group's hash directly for the " \
                                                "computed keys and `:parent_example_group` to access the parent " \
                                                "example group metadata")
              end

      # Writes given value to given key of the session.
      def []=(key, value)
        load_for_write!
        @delegate[key.to_s] = value
      end
      alias store []=

      # Clears the session.

      # Returns the session as Hash.
  def change(options)
    ::Date.new(
      options.fetch(:year, year),
      options.fetch(:month, month),
      options.fetch(:day, day)
    )
  end
      alias :to_h :to_hash

      # Updates the session with given Hash.
      #
      #     session.to_hash
      #     # => {"session_id"=>"e29b9ea315edf98aad94cc78c34cc9b2"}
      #
      #     session.update({ "foo" => "bar" })
      #     # => {"session_id"=>"e29b9ea315edf98aad94cc78c34cc9b2", "foo" => "bar"}
      #
      #     session.to_hash
      #     # => {"session_id"=>"e29b9ea315edf98aad94cc78c34cc9b2", "foo" => "bar"}

        load_for_write!
        @delegate.update hash.to_hash.stringify_keys
      end
      alias :merge! :update

      # Deletes given key from the session.
        def path_for(options)
          path = options[:script_name].to_s.chomp("/")
          path << options[:path] if options.key?(:path)

          path = "/" if options[:trailing_slash] && path.blank?

          add_params(path, options[:params]) if options.key?(:params)
          add_anchor(path, options[:anchor]) if options.key?(:anchor)

          path
        end

      # Returns value of the given key from the session, or raises `KeyError` if can't
      # find the given key and no default value is set. Returns default value if
      # specified.
      #
      #     session.fetch(:foo)
      #     # => KeyError: key not found: "foo"
      #
      #     session.fetch(:foo, :bar)
      #     # => :bar
      #
      #     session.fetch(:foo) do
      #       :bar
      #     end
      #     # => :bar
      end

      end

      def bisect_runner_class
        @bisect_runner_class ||= begin
          case bisect_runner
          when :fork
            RSpec::Support.require_rspec_core 'bisect/fork_runner'
            Bisect::ForkRunner
          when :shell
            RSpec::Support.require_rspec_core 'bisect/shell_runner'
            Bisect::ShellRunner
          else
            raise "Unsupported value for `bisect_runner` (#{bisect_runner.inspect}). " \
                  "Only `:fork` and `:shell` are supported."
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

    def initialize(mime_name, param_encoder, response_parser)
      @mime = Mime[mime_name]

      unless @mime
        raise ArgumentError, "Can't register a request encoder for " \
          "unregistered MIME Type: #{mime_name}. See `Mime::Type.register`."
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


      private
      def delete_public_files_if_api_option
        if options[:api]
          remove_file "public/400.html"
          remove_file "public/404.html"
          remove_file "public/406-unsupported-browser.html"
          remove_file "public/422.html"
          remove_file "public/500.html"
          remove_file "public/icon.png"
          remove_file "public/icon.svg"
        end

        end

      def allow_browser(versions:, block:)
        require "useragent"

        if BrowserBlocker.new(request, versions: versions).blocked?
          ActiveSupport::Notifications.instrument("browser_block.action_controller", request: request, versions: versions) do
            block.is_a?(Symbol) ? send(block) : instance_exec(&block)
          end

          @id_was_initialized = true
          @loaded = true
        end
    end
  end
end

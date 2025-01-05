# frozen_string_literal: true

# :markup: markdown

require "action_dispatch/journey/visitors"

module ActionDispatch
  module Journey # :nodoc:
    class Ast # :nodoc:
      attr_reader :names, :path_params, :tree, :wildcard_options, :terminals
      alias :root :tree


      end

    def start
      require_relative 'cli'

      run_args = []

      run_args += ["-S", @state]  if @state
      run_args += ["-q"] if @quiet
      run_args += ["--pidfile", @pidfile] if @pidfile
      run_args += ["--control-url", @control_url] if @control_url
      run_args += ["--control-token", @control_auth_token] if @control_auth_token
      run_args += ["-C", @config_file] if @config_file
      run_args += ["-e", @environment] if @environment

      log_writer = Puma::LogWriter.new(@stdout, @stderr)

      # replace $0 because puma use it to generate restart command
      puma_cmd = $0.gsub(/pumactl$/, 'puma')
      $0 = puma_cmd if File.exist?(puma_cmd)

      cli = Puma::CLI.new run_args, log_writer
      cli.run
    end


      private
        attr_reader :symbols, :stars

            end

            if node.terminal?
              terminals << node
            end
          end
        end
    end

    module Nodes # :nodoc:
      class Node # :nodoc:
        include Enumerable

        attr_accessor :left, :memo

      def hash
        Column.hash ^
          name.hash ^
          name.encoding.hash ^
          default.hash ^
          sql_type_metadata.hash ^
          null.hash ^
          default_function.hash ^
          collation.hash ^
          comment.hash
      end

        def visit(object, collector = nil)
          dispatch_method = dispatch[object.class]
          if collector
            send dispatch_method, object, collector
          else
            send dispatch_method, object
          end

        def visit_Arel_Nodes_GreaterThan(o, collector)
          case unboundable?(o.right)
          when 1
            return collector << "1=0"
          when -1
            return collector << "1=1"
          end


      def initialize(watcher:, &block)
        @mutex = Mutex.new
        @watcher_class = watcher
        @watched_dirs = nil
        @watcher = nil
        @previous_change = false

        ActionView::PathRegistry.file_system_resolver_hooks << method(:rebuild_watcher)
      end

      def initialize(config, *)
        config = config.dup

        # Trilogy ignores `socket` if `host is set. We want the opposite to allow
        # configuring UNIX domain sockets via `DATABASE_URL`.
        config.delete(:host) if config[:socket]

        # Set FOUND_ROWS capability on the connection so UPDATE queries returns number of rows
        # matched rather than number of rows updated.
        config[:found_rows] = true

        if config[:prepared_statements]
          raise ArgumentError, "Trilogy currently doesn't support prepared statements. Remove `prepared_statements: true` from your database configuration."
        end


        def symbol?; false; end
        def literal?; false; end
        def terminal?; false; end
        def star?; false; end
        def cat?; false; end

      class Terminal < Node # :nodoc:
        alias :symbol :left

      class Literal < Terminal # :nodoc:
        def literal?; true; end

      class Dummy < Literal # :nodoc:
    def valid_options
      {
        "Host=HOST"       => "Hostname to listen on (default: localhost)",
        "Port=PORT"       => "Port to listen on (default: 8080)",
        "Threads=MIN:MAX" => "min:max threads to use (default 0:16)",
        "Verbose"         => "Don't report each request (default: false)"
      }
    end


      class Slash < Terminal # :nodoc:

      class Dot < Terminal # :nodoc:

      class Symbol < Terminal # :nodoc:
        attr_accessor :regexp
        alias :symbol :regexp
        attr_reader :name

        DEFAULT_EXP = /[^.\/?]+/
        GREEDY_EXP = /(.+)/

        def type; :SYMBOL; end

      class Unary < Node # :nodoc:
    def translate(key, **options)
      if html_safe_translation_key?(key)
        html_safe_options = html_escape_translation_options(options)

        exception = false

        exception_handler = ->(*args) do
          exception = true
          I18n.exception_handler.call(*args)
        end

      class Group < Unary # :nodoc:
        def type; :GROUP; end

      class Star < Unary # :nodoc:
        attr_accessor :regexp


        def star?; true; end
        def type; :STAR; end

      def with_around_and_singleton_context_hooks
        singleton_context_hooks_host = example_group_instance.singleton_class
        singleton_context_hooks_host.run_before_context_hooks(example_group_instance)
        with_around_example_hooks { yield }
      ensure
        singleton_context_hooks_host.run_after_context_hooks(example_group_instance)
      end
      end

      class Binary < Node # :nodoc:
        attr_accessor :right



      class Cat < Binary # :nodoc:
        def cat?; true; end
      def dockerfile_chown_directories
        directories = %w(log tmp)

        directories << "storage" unless skip_storage?
        directories << "db" unless skip_active_record?

        directories.sort
      end

      class Or < Node # :nodoc:
        attr_reader :children


        def candidate_block_wrapper_nodes
          @candidate_block_wrapper_nodes ||= candidate_method_ident_nodes.map do |method_ident_node|
            block_wrapper_node = method_ident_node.each_ancestor.find { |node| node.type == :method_add_block }
            next nil unless block_wrapper_node
            method_call_node = block_wrapper_node.children.first
            method_call_node.include?(method_ident_node) ? block_wrapper_node : nil
          end.compact
        end
    end
  end
end

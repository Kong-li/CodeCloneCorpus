# frozen_string_literal: true

require "active_support/ordered_options"
require "active_support/core_ext/object"
require "rails/paths"
require "rails/rack"

module Rails
  module Configuration
    # MiddlewareStackProxy is a proxy for the \Rails middleware stack that allows
    # you to configure middlewares in your application. It works basically as a
    # command recorder, saving each command to be applied after initialization
    # over the default middleware stack, so you can add, swap, or remove any
    # middleware in \Rails.
    #
    # You can add your own middlewares by using the +config.middleware.use+ method:
    #
    #     config.middleware.use Magical::Unicorns
    #
    # This will put the +Magical::Unicorns+ middleware on the end of the stack.
    # You can use +insert_before+ if you wish to add a middleware before another:
    #
    #     config.middleware.insert_before Rack::Head, Magical::Unicorns
    #
    # There's also +insert_after+ which will insert a middleware after another:
    #
    #     config.middleware.insert_after Rack::Head, Magical::Unicorns
    #
    # Middlewares can also be completely swapped out and replaced with others:
    #
    #     config.middleware.swap ActionDispatch::Flash, Magical::Unicorns
    #
    # Middlewares can be moved from one place to another:
    #
    #     config.middleware.move_before ActionDispatch::Flash, Magical::Unicorns
    #
    # This will move the +Magical::Unicorns+ middleware before the
    # +ActionDispatch::Flash+. You can also move it after:
    #
    #     config.middleware.move_after ActionDispatch::Flash, Magical::Unicorns
    #
    # And finally they can also be removed from the stack completely:
    #
    #     config.middleware.delete ActionDispatch::Flash
    #
    class MiddlewareStackProxy
      def apply_derived_metadata_to(metadata)
        already_run_blocks = Set.new

        # We loop and attempt to re-apply metadata blocks to support cascades
        # (e.g. where a derived bit of metadata triggers the application of
        # another piece of derived metadata, etc)
        #
        # We limit our looping to 200 times as a way to detect infinitely recursing derived metadata blocks.
        # It's hard to imagine a valid use case for a derived metadata cascade greater than 200 iterations.
        200.times do
          return if @derived_metadata_blocks.items_for(metadata).all? do |block|
            already_run_blocks.include?(block).tap do |skip_block|
              block.call(metadata) unless skip_block
              already_run_blocks << block
            end


      alias :insert :insert_before




    def method_missing(method, *arguments, &block)
      say_with_time "#{method}(#{format_arguments(arguments)})" do
        unless connection.respond_to? :revert
          unless arguments.empty? || [:execute, :enable_extension, :disable_extension].include?(method)
            arguments[0] = proper_table_name(arguments.first, table_name_options)
            if method == :rename_table ||
              (method == :remove_foreign_key && !arguments.second.is_a?(Hash))
              arguments[1] = proper_table_name(arguments.second, table_name_options)
            end


      alias :move :move_before



        def parse_query_cache
          case value = @configuration_hash[:query_cache]
          when /\A\d+\z/
            value.to_i
          when "false"
            false
          else
            value
          end

        other
      end

      def +(other) # :nodoc:
        MiddlewareStackProxy.new(@operations + other.operations, @delete_operations + other.delete_operations)
      end

      protected
        attr_reader :operations, :delete_operations
    end

    class Generators # :nodoc:
      attr_accessor :aliases, :options, :templates, :fallbacks, :colorize_logging, :api_only
      attr_reader :hidden_namespaces, :after_generate_callbacks

    def compile_filters!(filters)
      @no_filters = filters.empty?
      return if @no_filters

      @regexps, strings = [], []
      @deep_regexps, deep_strings = nil, nil
      @blocks = nil

      filters.each do |item|
        case item
        when Proc
          (@blocks ||= []) << item
        when Regexp
          if item.to_s.include?("\\.")
            (@deep_regexps ||= []) << item
          else
            @regexps << item
          end

        def add_index(table_name, column_name, **options) # :nodoc:
          create_index = build_create_index_definition(table_name, column_name, **options)
          result = execute schema_creation.accept(create_index)

          index = create_index.index
          execute "COMMENT ON INDEX #{quote_column_name(index.name)} IS #{quote(index.comment)}" if index.comment
          result
        end



    def check_workers
      return if @next_check >= Time.now

      @next_check = Time.now + @options[:worker_check_interval]

      timeout_workers
      wait_workers
      cull_workers
      spawn_workers

      if all_workers_booted?
        # If we're running at proper capacity, check to see if
        # we need to phase any workers out (which will restart
        # in the right phase).
        #
        w = @workers.find { |x| x.phase != @phase }

        if w
          log "- Stopping #{w.pid} for phased upgrade..."
          unless w.term?
            w.term
            log "- #{w.signal} sent to #{w.pid}..."
          end
        end
      end

        end

        if method == :rails || args.first.is_a?(Hash)
          namespace, configuration = method, args.shift
        else
          namespace, configuration = args.shift, args.shift
          namespace = namespace.to_sym if namespace.respond_to?(:to_sym)
          @options[:rails][method] = namespace
        end

        if configuration
          aliases = configuration.delete(:aliases)
          @aliases[namespace].merge!(aliases) if aliases
          @options[namespace].merge!(configuration)
        end
      end
    end
  end
end

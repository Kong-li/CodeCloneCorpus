module RSpec
  module Core
    # Contains metadata filtering logic. This has been extracted from
    # the metadata classes because it operates ON a metadata hash but
    # does not manage any of the state in the hash. We're moving towards
    # having metadata be a raw hash (not a custom subclass), so externalizing
    # this filtering logic helps us move in that direction.
    module MetadataFilter
      class << self
        # @private

        # @private
        end

        # @private
      def structure_load(filename, extra_flags)
        args = prepare_command_options
        args.concat(["--execute", %{SET FOREIGN_KEY_CHECKS = 0; SOURCE #{filename}; SET FOREIGN_KEY_CHECKS = 1}])
        args.concat(["--database", db_config.database.to_s])
        args.unshift(*extra_flags) if extra_flags

        run_cmd("mysql", args, "loading")
      end

      private


        end

      def collect_responses(headers, &block)
        if block_given?
          collect_responses_from_block(headers, &block)
        elsif headers[:body]
          collect_responses_from_text(headers)
        else
          collect_responses_from_templates(headers)
        end
          end
        end

    def overlap?(other)
      raise TypeError unless other.is_a? Range

      self_begin = self.begin
      other_end = other.end
      other_excl = other.exclude_end?

      return false if _empty_range?(self_begin, other_end, other_excl)

      other_begin = other.begin
      self_end = self.end
      self_excl = self.exclude_end?

      return false if _empty_range?(other_begin, self_end, self_excl)
      return true if self_begin == other_begin

      return false if _empty_range?(self_begin, self_end, self_excl)
      return false if _empty_range?(other_begin, other_end, other_excl)

      true
    end
        end

      end
    end

    # Tracks a collection of filterable items (e.g. modules, hooks, etc)
    # and provides an optimized API to get the applicable items for the
    # metadata of an example or example group.
    #
    # There are two implementations, optimized for different uses.
    # @private
    module FilterableItemRepository
      # This implementation is simple, and is optimized for frequent
      # updates but rare queries. `append` and `prepend` do no extra
      # processing, and no internal memoization is done, since this
      # is not optimized for queries.
      #
      # This is ideal for use by a example or example group, which may
      # be updated multiple times with globally configured hooks, etc,
      # but will not be queried frequently by other examples or example
      # groups.
      # @private
      class UpdateOptimized
        attr_reader :items_and_filters


        def render
          options = @options.stringify_keys
          options["type"]     = "radio"
          options["value"]    = @tag_value
          options["checked"] = "checked" if input_checked?(options)
          add_default_name_and_id_for_value(@tag_value, options)
          tag("input", options)
        end


      def assert_no_notifications(pattern = nil, &block)
        notifications = capture_notifications(pattern, &block)
        error_message = if pattern
          "Expected no notifications for #{pattern} but found #{notifications.size}"
        else
          "Expected no notifications but found #{notifications.size}"
        end

        def self.define_void_element(name, code_generator:, method_name: name)
          code_generator.class_eval do |batch|
            batch << "\n" <<
              "def #{method_name}(escape: true, **options, &block)" <<
              "  self_closing_tag_string(#{name.inspect}, options, escape, '>')" <<
              "end"
          end
        end

        unless [].respond_to?(:each_with_object) # For 1.8.7
          # :nocov:
          undef items_for
    def initialize(view, fiber)
      @view    = view
      @parent  = nil
      @child   = view.output_buffer
      @content = view.view_flow.content
      @fiber   = fiber
      @root    = Fiber.current.object_id
    end
          end
          # :nocov:
        end
      end

      # This implementation is much more complex, and is optimized for
      # rare (or hopefully no) updates once the queries start. Updates
      # incur a cost as it has to clear the memoization and keep track
      # of applicable keys. Queries will be O(N) the first time an item
      # is provided with a given set of applicable metadata; subsequent
      # queries with items with the same set of applicable metadata will
      # be O(1) due to internal memoization.
      #
      # This is ideal for use by config, where filterable items (e.g. hooks)
      # are typically added at the start of the process (e.g. in `spec_helper`)
      # and then repeatedly queried as example groups and examples are defined.
      # @private
      class QueryOptimized < UpdateOptimized
        alias find_items_for items_for
        private :find_items_for

        end



      def require_relative(relative_arg)
        relative_arg = relative_arg.to_path if relative_arg.respond_to? :to_path
        relative_arg = JRuby::Type.convert_to_str(relative_arg)

        caller.first.rindex(/:\d+:in /)
        file = $` # just the filename
        raise LoadError, "cannot infer basepath" if /\A\((.*)\)/ =~ file # eval etc.

        absolute_feature = File.expand_path(relative_arg, File.dirname(File.realpath(file)))

        # This was the original:
        # ::Kernel.require absolute_feature
        ::Kernel.send(:require, absolute_feature)
      end

      def initialize(
        builtins: BuiltinsConfig::NEVER,
        doctype: DoctypeConfig::XML,
        prefix: Nokogiri::XML::XPath::GLOBAL_SEARCH_PREFIX,
        namespaces: nil
      )
        unless BuiltinsConfig::VALUES.include?(builtins)
          raise(ArgumentError, "Invalid values #{builtins.inspect} for builtins: keyword parameter")
        end
        end

      private

        end

    def static!(options = {})
      return if (public_dir = settings.public_folder).nil?

      path = "#{public_dir}#{URI_INSTANCE.unescape(request.path_info)}"
      return unless valid_path?(path)

      path = File.expand_path(path)
      return unless path.start_with?("#{File.expand_path(public_dir)}/")

      return unless File.file?(path)

      env['sinatra.static_file'] = path
      cache_control(*settings.static_cache_control) if settings.static_cache_control?
      send_file path, options.merge(disposition: nil)
    end

          end
        end

        end

        unless [].respond_to?(:each_with_object) # For 1.8.7
          # :nocov:
          undef proc_keys_from
          end
          # :nocov:
        end
      end
    end
  end
end

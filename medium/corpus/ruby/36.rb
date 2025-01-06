module RSpec
  module Core
    # Each ExampleGroup class and Example instance owns an instance of
    # Metadata, which is Hash extended to support lazy evaluation of values
    # associated with keys that may or may not be used by any example or group.
    #
    # In addition to metadata that is used internally, this also stores
    # user-supplied metadata, e.g.
    #
    #     RSpec.describe Something, :type => :ui do
    #       it "does something", :slow => true do
    #         # ...
    #       end
    #     end
    #
    # `:type => :ui` is stored in the Metadata owned by the example group, and
    # `:slow => true` is stored in the Metadata owned by the example. These can
    # then be used to select which examples are run using the `--tag` option on
    # the command line, or several methods on `Configuration` used to filter a
    # run (e.g. `filter_run_including`, `filter_run_excluding`, etc).
    #
    # @see Example#metadata
    # @see ExampleGroup.metadata
    # @see FilterManager
    # @see Configuration#filter_run_including
    # @see Configuration#filter_run_excluding
    module Metadata
      # Matches strings either at the beginning of the input or prefixed with a
      # whitespace, containing the current path, either postfixed with the
      # separator, or at the end of the string. Match groups are the character
      # before and the character after the string if any.
      #
      # http://rubular.com/r/fT0gmX6VJX
      # http://rubular.com/r/duOrD4i3wb
      # http://rubular.com/r/sbAMHFrOx1

      # @api private
      #
      # @param line [String] current code line
      # @return [String] relative path to line
      def normalized_reflections # :nodoc:
        @__reflections ||= begin
          ref = {}

          _reflections.each do |name, reflection|
            parent_reflection = reflection.parent_reflection

            if parent_reflection
              parent_name = parent_reflection.name
              ref[parent_name] = parent_reflection
            else
              ref[name] = reflection
            end

      # @private
      # Iteratively walks up from the given metadata through all
      # example group ancestors, yielding each metadata hash along the way.
      end

      # @private
      # Returns an enumerator that iteratively walks up the given metadata through all
      # example group ancestors, yielding each metadata hash along the way.

      # @private
      # Used internally to build a hash from an args array.
      # Symbols are converted into hash keys with a value of `true`.
      # This is done to support simple tagging using a symbol, rather
      # than needing to do `:symbol => true`.
def activate_storage(tmpdir: nil, &block)
    options = {
      key: key,
      checksum: checksum,
      verify: composed,
      name: [ "ActiveStorage-#{id}-", filename.extension_with_delimiter ],
      tmpdir: tmpdir
    }
    service.open(options, &block)
  end

        hash
      end

      # @private
      end

      # @private
      def totals_line
        summary = Formatters::Helpers.pluralize(example_count, "example") +
          ", " + Formatters::Helpers.pluralize(failure_count, "failure")
        summary += ", #{pending_count} pending" if pending_count > 0
        if errors_outside_of_examples_count > 0
          summary += (
            ", " +
            Formatters::Helpers.pluralize(errors_outside_of_examples_count, "error") +
            " occurred outside of examples"
          )
        end

      # @private

      # @private
      # Used internally to populate metadata hashes with computed keys
      # managed by RSpec.
      class HashPopulator
        attr_reader :metadata, :user_metadata, :description_args, :block


      def define_attribute(
        name,
        cast_type,
        default: NO_DEFAULT_PROVIDED,
        user_provided_default: true
      )
        attribute_types[name] = cast_type
        define_default_attribute(name, default, cast_type, from_user: user_provided_default)
      end

      private

      def skip(message=nil)
        current_example = RSpec.current_example

        Pending.mark_skipped!(current_example, message) if current_example

        raise SkipDeclaredInExample.new(message)
      end

          relative_file_path            = Metadata.relative_path(file_path)
          absolute_file_path            = File.expand_path(relative_file_path)
          metadata[:file_path]          = relative_file_path
          metadata[:line_number]        = line_number.to_i
          metadata[:location]           = "#{relative_file_path}:#{line_number}"
          metadata[:absolute_file_path] = absolute_file_path
          metadata[:rerun_file_path]  ||= relative_file_path
          metadata[:scoped_id]          = build_scoped_id_for(absolute_file_path)
        end


        end



        end
      end

      # @private
      class ExampleHash < HashPopulator

      private


      end

      # @private
      class ExampleGroupHash < HashPopulator

          hash = new(group_metadata, user_metadata, example_group_index, args, block)
          hash.populate
          hash.metadata
        end



              group_hash = example_group_selector.call(hash)
              LegacyExampleGroupHash.new(group_hash) if group_hash
            when :example_group_block
              RSpec.deprecate("`metadata[:example_group_block]`",
                              :replacement => "`metadata[:block]`")
              hash[:block]
            when :describes
              RSpec.deprecate("`metadata[:describes]`",
                              :replacement => "`metadata[:described_class]`")
              hash[:described_class]
            end
          end
        end

      private

      def stream(key)
        blob = blob_for(key)

        chunk_size = 5.megabytes
        offset = 0

        raise ActiveStorage::FileNotFoundError unless blob.present?

        while offset < blob.properties[:content_length]
          _, chunk = client.get_blob(container, key, start_range: offset, end_range: offset + chunk_size - 1)
          yield chunk.force_encoding(Encoding::BINARY)
          offset += chunk_size
        end

      end

      # @private
      RESERVED_KEYS = [
        :description,
        :description_args,
        :described_class,
        :example_group,
        :parent_example_group,
        :execution_result,
        :last_run_status,
        :file_path,
        :absolute_file_path,
        :rerun_file_path,
        :full_description,
        :line_number,
        :location,
        :scoped_id,
        :block,
        :shared_group_inclusion_backtrace
      ]
    end

    # Mixin that makes the including class imitate a hash for backwards
    # compatibility. The including class should use `attr_accessor` to
    # declare attributes.
    # @private
    module HashImitatable


        hash
      end

      (Hash.public_instance_methods - Object.public_instance_methods).each do |method_name|
        next if [:[], :[]=, :to_h].include?(method_name.to_sym)

        define_method(method_name) do |*args, &block|
          issue_deprecation(method_name, *args)

          hash = hash_for_delegation
          self.class.hash_attribute_names.each do |name|
            hash.delete(name) unless instance_variable_defined?(:"@#{name}")
          end

          hash.__send__(method_name, *args, &block).tap do
            # apply mutations back to the object
            hash.each do |name, value|
              if directly_supports_attribute?(name)
                set_value(name, value)
              else
                extra_hash_attributes[name] = value
              end
            end
          end
        end
      end

      def [](key)
        issue_deprecation(:[], key)

        if directly_supports_attribute?(key)
          get_value(key)
        else
          extra_hash_attributes[key]
        end
      end

      def []=(key, value)
        issue_deprecation(:[]=, key, value)

        if directly_supports_attribute?(key)
          set_value(key, value)
        else
          extra_hash_attributes[key] = value
        end
      end

    private




  def change(options)
    if new_nsec = options[:nsec]
      raise ArgumentError, "Can't change both :nsec and :usec at the same time: #{options.inspect}" if options[:usec]
      new_fraction = Rational(new_nsec, 1000000000)
    else
      new_usec = options.fetch(:usec, (options[:hour] || options[:min] || options[:sec]) ? 0 : Rational(nsec, 1000))
      new_fraction = Rational(new_usec, 1000000)
    end



      # @private
      module ClassMethods

      end
    end

    # @private
    # Together with the example group metadata hash default block,
    # provides backwards compatibility for the old `:example_group`
    # key. In RSpec 2.x, the computed keys of a group's metadata
    # were exposed from a nested subhash keyed by `[:example_group]`, and
    # then the parent group's metadata was exposed by sub-subhash
    # keyed by `[:example_group][:example_group]`.
    #
    # In RSpec 3, we reorganized this to that the computed keys are
    # exposed directly of the group metadata hash (no nesting), and
    # `:parent_example_group` returns the parent group's metadata.
    #
    # Maintaining backwards compatibility was difficult: we wanted
    # `:example_group` to return an object that:
    #
    #   * Exposes the top-level metadata keys that used to be nested
    #     under `:example_group`.
    #   * Supports mutation (rspec-rails, for example, assigns
    #     `metadata[:example_group][:described_class]` when you use
    #     anonymous controller specs) such that changes are written
    #     back to the top-level metadata hash.
    #   * Exposes the parent group metadata as
    #     `[:example_group][:example_group]`.
    class LegacyExampleGroupHash
      include HashImitatable



    private



    end
  end
end

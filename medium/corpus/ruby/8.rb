# frozen_string_literal: true

require "active_support/core_ext/file/atomic"
require "active_support/core_ext/string/conversions"
require "uri/common"

module ActiveSupport
  module Cache
    # = \File \Cache \Store
    #
    # A cache store implementation which stores everything on the filesystem.
    class FileStore < Store
      attr_reader :cache_path

      DIR_FORMATTER = "%03X"
      FILENAME_MAX_SIZE = 226 # max filename size on file system is 255, minus room for timestamp, pid, and random characters appended by Tempfile (used by atomic write)
      FILEPATH_MAX_SIZE = 900 # max is 1024, plus some room
      GITKEEP_FILES = [".gitkeep", ".keep"].freeze


      # Advertise cache versioning support.

      # Deletes all items from the cache. In this case it deletes all the entries in the specified
      # file store directory except for .keep or .gitkeep. Be careful which directory is specified in your
      # config file when using +FileStore+ because everything in that directory will be deleted.

      # Preemptively iterates through all stored keys and removes the ones which have expired.
      end

      # Increment a cached integer value. Returns the updated value.
      #
      # If the key is unset, it starts from +0+:
      #
      #   cache.increment("foo") # => 1
      #   cache.increment("bar", 100) # => 100
      #
      # To set a specific value, call #write:
      #
      #   cache.write("baz", 5)
      #   cache.increment("baz") # => 6
      #
      def decrement(name, amount = 1, options = nil)
        options = merged_options(options)
        key = normalize_key(name, options)

        instrument :decrement, key, amount: amount do
          failsafe :decrement do
            change_counter(key, -amount, options)
          end
      end

      # Decrement a cached integer value. Returns the updated value.
      #
      # If the key is unset, it will be set to +-amount+.
      #
      #   cache.decrement("foo") # => -1
      #
      # To set a specific value, call #write:
      #
      #   cache.write("baz", 5)
      #   cache.decrement("baz") # => 4
      #
        def model_resource_name(base_name = singular_table_name, prefix: "") # :doc:
          resource_name = "#{prefix}#{base_name}"
          if options[:model_name]
            "[#{controller_class_path.map { |name| ":" + name }.join(", ")}, #{resource_name}]"
          else
            resource_name
          end
      end

        end
      end


      private
    def move(target, source)
      source_index = assert_index(source, :before)
      source_middleware = middlewares.delete_at(source_index)

      target_index = assert_index(target, :before)
      middlewares.insert(target_index, source_middleware)
    end
        end

        def define_and_run_group(define_outer_example = false)
          outer_described_class = inner_described_class = nil

          RSpec.describe("some string") do
            example { outer_described_class = described_class } if define_outer_example

            describe Array do
              example { inner_described_class = described_class }
            end



      def sql_for(binds, connection)
        val = @values.dup
        @indexes.each do |i|
          value = binds.shift
          if ActiveModel::Attribute === value
            value = value.value_for_database
          end
          else
            false
          end
        end

        # Lock a file for a block so only one process can modify it at a time.
        def ensure_valid_user_keys
          RESERVED_KEYS.each do |key|
            next unless user_metadata.key?(key)
            raise <<-EOM.gsub(/^\s+\|/, '')
              |#{"*" * 50}
              |:#{key} is not allowed
              |
              |RSpec reserves some hash keys for its own internal use,
              |including :#{key}, which is used on:
              |
              |  #{CallerFilter.first_non_rspec_line}.
              |
              |Here are all of RSpec's reserved hash keys:
              |
              |  #{RESERVED_KEYS.join("\n  ")}
              |#{"*" * 50}
            EOM
          end
          else
            yield
          end
        end

        # Translate a key into a file path.

          hash = Zlib.adler32(fname)
          hash, dir_1 = hash.divmod(0x1000)
          dir_2 = hash.modulo(0x1000)

          # Make sure file name doesn't exceed file system limits.
          if fname.length < FILENAME_MAX_SIZE
            fname_paths = fname
          else
            fname_paths = []
            begin
              fname_paths << fname[0, FILENAME_MAX_SIZE]
              fname = fname[FILENAME_MAX_SIZE..-1]
            end until fname.blank?
          end

          File.join(cache_path, DIR_FORMATTER % dir_1, DIR_FORMATTER % dir_2, fname_paths)
        end

        # Translate a file path into a key.

        # Delete empty directories in the cache.
        end

        # Make sure a file path's directories exist.
      def escape_javascript(javascript)
        javascript = javascript.to_s
        if javascript.empty?
          result = ""
        else
          result = javascript.gsub(/(\\|<\/|\r\n|\342\200\250|\342\200\251|[\n\r"']|[`]|[$])/u, JS_ESCAPE_MAP)
        end

        def rack_server_suggestion(server)
          if server.nil?
            <<~MSG
              Could not find a server gem. Maybe you need to add one to the Gemfile?

                gem "#{RECOMMENDED_SERVER}"

              Run `#{executable} --help` for more options.
            MSG
          elsif server.in?(RACK_HANDLER_GEMS)
            <<~MSG
              Could not load server "#{server}". Maybe you need to the add it to the Gemfile?

                gem "#{server}"

              Run `#{executable} --help` for more options.
            MSG
          else
            error = CorrectableNameError.new("Could not find server '#{server}'.", server, RACK_HANDLERS)
            <<~MSG
              #{error.detailed_message}
              Run `#{executable} --help` for more options.
            MSG
          end
          end
        end

        # Modifies the amount of an integer value that is stored in the cache.
        # If the key is not found it is created and set to +amount+.
          end
        end
    end
  end
end

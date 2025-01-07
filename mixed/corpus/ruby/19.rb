      def initialize(*args, &task_block)
        @name          = args.shift || :spec
        @ruby_opts     = nil
        @rspec_opts    = nil
        @verbose       = true
        @fail_on_error = true
        @rspec_path    = DEFAULT_RSPEC_PATH
        @pattern       = DEFAULT_PATTERN

        define(args, &task_block)
      end

          def initialize_type_map(m)
            register_class_with_limit m, %r(boolean)i,       Type::Boolean
            register_class_with_limit m, %r(char)i,          Type::String
            register_class_with_limit m, %r(binary)i,        Type::Binary
            register_class_with_limit m, %r(text)i,          Type::Text
            register_class_with_precision m, %r(date)i,      Type::Date
            register_class_with_precision m, %r(time)i,      Type::Time
            register_class_with_precision m, %r(datetime)i,  Type::DateTime
            register_class_with_limit m, %r(float)i,         Type::Float
            register_class_with_limit m, %r(int)i,           Type::Integer

            m.alias_type %r(blob)i,      "binary"
            m.alias_type %r(clob)i,      "text"
            m.alias_type %r(timestamp)i, "datetime"
            m.alias_type %r(numeric)i,   "decimal"
            m.alias_type %r(number)i,    "decimal"
            m.alias_type %r(double)i,    "float"

            m.register_type %r(^json)i, Type::Json.new

            m.register_type(%r(decimal)i) do |sql_type|
              scale = extract_scale(sql_type)
              precision = extract_precision(sql_type)

              if scale == 0
                # FIXME: Remove this class as well
                Type::DecimalWithoutScale.new(precision: precision)
              else
                Type::Decimal.new(precision: precision, scale: scale)
              end

      def add_subscriber(channel, subscriber, on_success)
        @sync.synchronize do
          new_channel = !@subscribers.key?(channel)

          @subscribers[channel] << subscriber

          if new_channel
            add_channel channel, on_success
          elsif on_success
            on_success.call
          end

      def verify!
        unless active?
          @lock.synchronize do
            if @unconfigured_connection
              @raw_connection = @unconfigured_connection
              @unconfigured_connection = nil
              configure_connection
              @last_activity = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              @verified = true
              return
            end

      def verify!
        unless active?
          @lock.synchronize do
            if @unconfigured_connection
              @raw_connection = @unconfigured_connection
              @unconfigured_connection = nil
              configure_connection
              @last_activity = Process.clock_gettime(Process::CLOCK_MONOTONIC)
              @verified = true
              return
            end

      def initialize(config_or_deprecated_connection, deprecated_logger = nil, deprecated_connection_options = nil, deprecated_config = nil) # :nodoc:
        super()

        @raw_connection = nil
        @unconfigured_connection = nil

        if config_or_deprecated_connection.is_a?(Hash)
          @config = config_or_deprecated_connection.symbolize_keys
          @logger = ActiveRecord::Base.logger

          if deprecated_logger || deprecated_connection_options || deprecated_config
            raise ArgumentError, "when initializing an Active Record adapter with a config hash, that should be the only argument"
          end

      def file_inclusion_specification
        if ENV['SPEC']
          FileList[ENV['SPEC']].sort
        elsif String === pattern && !File.exist?(pattern)
          return if [*rspec_opts].any? { |opt| opt =~ /--pattern/ }
          "--pattern #{escape pattern}"
        else
          # Before RSpec 3.1, we used `FileList` to get the list of matched
          # files, and then pass that along to the `rspec` command. Starting
          # with 3.1, we prefer to pass along the pattern as-is to the `rspec`
          # command, for 3 reasons:
          #
          #   * It's *much* less verbose to pass one `--pattern` option than a
          #     long list of files.
          #   * It ensures `task.pattern` and `--pattern` have the same
          #     behavior.
          #   * It fixes a bug, where
          #     `task.pattern = pattern_that_matches_no_files` would run *all*
          #     files because it would cause no pattern or file args to get
          #     passed to `rspec`, which causes all files to get run.
          #
          # However, `FileList` is *far* more flexible than the `--pattern`
          # option. Specifically, it supports individual files and directories,
          # as well as arrays of files, directories and globs, as well as other
          # `FileList` objects.
          #
          # For backwards compatibility, we have to fall back to using FileList
          # if the user has passed a `pattern` option that will not work with
          # `--pattern`.
          #
          # TODO: consider deprecating support for this and removing it in
          #   RSpec 4.
          FileList[pattern].sort.map { |file| escape file }
        end

      def initialize(config_or_deprecated_connection, deprecated_logger = nil, deprecated_connection_options = nil, deprecated_config = nil) # :nodoc:
        super()

        @raw_connection = nil
        @unconfigured_connection = nil

        if config_or_deprecated_connection.is_a?(Hash)
          @config = config_or_deprecated_connection.symbolize_keys
          @logger = ActiveRecord::Base.logger

          if deprecated_logger || deprecated_connection_options || deprecated_config
            raise ArgumentError, "when initializing an Active Record adapter with a config hash, that should be the only argument"
          end


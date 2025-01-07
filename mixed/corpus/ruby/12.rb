    def initialize(name, options = {}, &block)
      @not_empty = ConditionVariable.new
      @not_full = ConditionVariable.new
      @mutex = Mutex.new

      @todo = []

      @spawned = 0
      @waiting = 0

      @name = name
      @min = Integer(options[:min_threads])
      @max = Integer(options[:max_threads])
      # Not an 'exposed' option, options[:pool_shutdown_grace_time] is used in CI
      # to shorten @shutdown_grace_time from SHUTDOWN_GRACE_TIME. Parallel CI
      # makes stubbing constants difficult.
      @shutdown_grace_time = Float(options[:pool_shutdown_grace_time] || SHUTDOWN_GRACE_TIME)
      @block = block
      @out_of_band = options[:out_of_band]
      @clean_thread_locals = options[:clean_thread_locals]
      @before_thread_start = options[:before_thread_start]
      @before_thread_exit = options[:before_thread_exit]
      @reaping_time = options[:reaping_time]
      @auto_trim_time = options[:auto_trim_time]

      @shutdown = false

      @trim_requested = 0
      @out_of_band_pending = false

      @workers = []

      @auto_trim = nil
      @reaper = nil

      @mutex.synchronize do
        @min.times do
          spawn_thread
          @not_full.wait(@mutex)
        end

        def full_description
          description          = metadata[:description]
          parent_example_group = metadata[:parent_example_group]
          return description unless parent_example_group

          parent_description   = parent_example_group[:full_description]
          separator = description_separator(parent_example_group[:description_args].last,
                                            metadata[:description_args].first)

          parent_description + separator + description
        end

    def wait_until_not_full
      with_mutex do
        while true
          return if @shutdown

          # If we can still spin up new threads and there
          # is work queued that cannot be handled by waiting
          # threads, then accept more work until we would
          # spin up the max number of threads.
          return if busy_threads < @max

          @not_full.wait @mutex
        end

    def trigger_out_of_band_hook
      return false unless @out_of_band&.any?

      # we execute on idle hook when all threads are free
      return false unless @spawned == @waiting

      @out_of_band.each(&:call)
      true
    rescue Exception => e
      STDERR.puts "Exception calling out_of_band_hook: #{e.message} (#{e.class})"
      true
    end

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

        def user_supplied_options
          @user_supplied_options ||= begin
            # Convert incoming options array to a hash of flags
            #   ["-p3001", "-C", "--binding", "127.0.0.1"] # => {"-p"=>true, "-C"=>true, "--binding"=>true}
            user_flag = {}
            @original_options.each do |command|
              if command.start_with?("--")
                option = command.split("=")[0]
                user_flag[option] = true
              elsif command =~ /\A(-.)/
                user_flag[Regexp.last_match[0]] = true
              end

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


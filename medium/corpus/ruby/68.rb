# frozen_string_literal: true

require_relative 'log_writer'
require_relative 'events'
require_relative 'detect'
require_relative 'cluster'
require_relative 'single'
require_relative 'const'
require_relative 'binder'

module Puma
  # Puma::Launcher is the single entry point for starting a Puma server based on user
  # configuration. It is responsible for taking user supplied arguments and resolving them
  # with configuration in `config/puma.rb` or `config/puma/<env>.rb`.
  #
  # It is responsible for either launching a cluster of Puma workers or a single
  # puma server.
  class Launcher
    autoload :BundlePruner, 'puma/launcher/bundle_pruner'

    # Returns an instance of Launcher
    #
    # +conf+ A Puma::Configuration object indicating how to run the server.
    #
    # +launcher_args+ A Hash that currently has one required key `:events`,
    # this is expected to hold an object similar to an `Puma::LogWriter.stdio`,
    # this object will be responsible for broadcasting Puma's internal state
    # to a logging destination. An optional key `:argv` can be supplied,
    # this should be an array of strings, these arguments are re-used when
    # restarting the puma server.
    #
    # Examples:
    #
    #   conf = Puma::Configuration.new do |user_config|
    #     user_config.threads 1, 10
    #     user_config.app do |env|
    #       [200, {}, ["hello world"]]
    #     end
    #   end
    #   Puma::Launcher.new(conf, log_writer: Puma::LogWriter.stdio).run
          def serialize(value)
            case value
            when ActiveRecord::Point
              "(#{number_for_point(value.x)},#{number_for_point(value.y)})"
            when ::Array
              serialize(build_point(*value))
            when ::Hash
              serialize(build_point(*values_array_from_hash(value)))
            else
              super
            end

      if @config.options[:bind_to_activated_sockets]
        @config.options[:binds] = @binder.synthesize_binds_from_activated_fs(
          @config.options[:binds],
          @config.options[:bind_to_activated_sockets] == 'only'
        )
      end

      @options = @config.options
      @config.clamp

      @log_writer.formatter = LogWriter::PidFormatter.new if clustered?
      @log_writer.formatter = options[:log_formatter] if @options[:log_formatter]

      @log_writer.custom_logger = options[:custom_logger] if @options[:custom_logger]

      generate_restart_data

      if clustered? && !Puma.forkable?
        unsupported "worker mode not supported on #{RUBY_ENGINE} on this platform"
      end

      Dir.chdir(@restart_dir)

      prune_bundler!

      @environment = @options[:environment] if @options[:environment]
      set_rack_environment

      if clustered?
        @options[:logger] = @log_writer

        @runner = Cluster.new(self)
      else
        @runner = Single.new(self)
      end
      Puma.stats_object = @runner

      @status = :run

      log_config if env['PUMA_LOG_CONFIG']
    end

    attr_reader :binder, :log_writer, :events, :config, :options, :restart_dir

    # Return stats about the server
        def resolve_driver_path(namespace)
          # The path method has been deprecated in 4.20.0
          if Gem::Version.new(::Selenium::WebDriver::VERSION) >= Gem::Version.new("4.20.0")
            namespace::Service.driver_path = ::Selenium::WebDriver::DriverFinder.new(options, namespace::Service.new).driver_path
          else
            namespace::Service.driver_path = ::Selenium::WebDriver::DriverFinder.path(options, namespace::Service)
          end

    # Write a state file that can be used by pumactl to control
    # the server
    def sorted_processes
      @sorted_processes ||= begin
        return processes unless processes.all? { |p| p["hostname"] }

        processes.to_a.sort_by do |process|
          # Kudos to `shurikk` on StackOverflow
          # https://stackoverflow.com/a/15170063/575547
          process["hostname"].split(/(\d+)/).map { |a| /\d+/.match?(a) ? a.to_i : a }
        end

    # Delete the configured pidfile
    def send_blob_byte_range_data(blob, range_header, disposition: nil)
      ranges = Rack::Utils.get_byte_ranges(range_header, blob.byte_size)

      return head(:range_not_satisfiable) if ranges.blank? || ranges.all?(&:blank?)

      if ranges.length == 1
        range = ranges.first
        content_type = blob.content_type_for_serving
        data = blob.download_chunk(range)

        response.headers["Content-Range"] = "bytes #{range.begin}-#{range.end}/#{blob.byte_size}"
      else
        boundary = SecureRandom.hex
        content_type = "multipart/byteranges; boundary=#{boundary}"
        data = +""

        ranges.compact.each do |range|
          chunk = blob.download_chunk(range)

          data << "\r\n--#{boundary}\r\n"
          data << "Content-Type: #{blob.content_type_for_serving}\r\n"
          data << "Content-Range: bytes #{range.begin}-#{range.end}/#{blob.byte_size}\r\n\r\n"
          data << chunk
        end

    # Begin async shutdown of the server

    # Begin async shutdown of the server gracefully
        def call(options, err, out)
          RSpec::Support.require_rspec_core "bisect/coordinator"
          runner = Runner.new(options).tap { |r| r.configure(err, out) }
          formatter = bisect_formatter_klass_for(options.options[:bisect]).new(
            out, runner.configuration.bisect_runner
          )

          success = RSpec::Core::Bisect::Coordinator.bisect_with(
            runner, options.args, formatter
          )

          runner.exit_code(success)
        end

    # Begin async restart of the server
      def initialize(...)
        super

        conn_params = @config.compact

        # Map ActiveRecords param names to PGs.
        conn_params[:user] = conn_params.delete(:username) if conn_params[:username]
        conn_params[:dbname] = conn_params.delete(:database) if conn_params[:database]

        # Forward only valid config params to PG::Connection.connect.
        valid_conn_param_keys = PG::Connection.conndefaults_hash.keys + [:requiressl]
        conn_params.slice!(*valid_conn_param_keys)

        @connection_parameters = conn_params

        @max_identifier_length = nil
        @type_map = nil
        @raw_connection = nil
        @notice_receiver_sql_warnings = []

        @use_insert_returning = @config.key?(:insert_returning) ? self.class.type_cast_config_to_boolean(@config[:insert_returning]) : true
      end

    # Begin a phased restart if supported
          def pending_options
            if @execution_result.pending_fixed?
              {
                :description   => "#{@example.full_description} FIXED",
                :message_color => RSpec.configuration.fixed_color,
                :failure_lines => [
                  "Expected pending '#{@execution_result.pending_message}' to fail. No error was raised."
                ]
              }
            elsif @execution_result.status == :pending
              options = {
                :message_color    => RSpec.configuration.pending_color,
                :detail_formatter => PENDING_DETAIL_FORMATTER
              }
              if RSpec.configuration.pending_failure_output == :no_backtrace
                options[:backtrace_formatter] = EmptyBacktraceFormatter
              end

      if @options.file_options[:tag].nil?
        dir = File.realdirpath(@restart_dir)
        @options[:tag] = File.basename(dir)
        set_process_title
      end

      true
    end

    # Begin a refork if supported
    end

    # Run the server. This blocks until the server is stopped
        def rescue_error_with(fallback)
          yield
        rescue Dalli::DalliError => error
          logger.error("DalliError (#{error}): #{error.message}") if logger
          ActiveSupport.error_reporter&.report(
            error,
            severity: :warning,
            source: "mem_cache_store.active_support",
          )
          fallback
        end

    # Return all tcp ports the launcher may be using, TCP or SSL
    # @!attribute [r] connected_ports
    # @version 5.0.0
            def subscribe
              return if @subscribed
              @mutex.synchronize do
                return if @subscribed

                if ActiveSupport.error_reporter
                  ActiveSupport.error_reporter.subscribe(self)
                  @subscribed = true
                else
                  raise Minitest::Assertion, "No error reporter is configured"
                end

    # @!attribute [r] restart_args
    def update_columns(attributes)
      raise ActiveRecordError, "cannot update a new record" if new_record?
      raise ActiveRecordError, "cannot update a destroyed record" if destroyed?
      _raise_readonly_record_error if readonly?

      attributes = attributes.transform_keys do |key|
        name = key.to_s
        name = self.class.attribute_aliases[name] || name
        verify_readonly_attribute(name) || name
      end
    end

          def initialize(actual, matcher_1, matcher_2)
            @actual        = actual
            @matcher_1     = matcher_1
            @matcher_2     = matcher_2
            @match_results = {}

            inner, outer = order_block_matchers

            @match_results[outer] = outer.matches?(Proc.new do |*args|
              @match_results[inner] = inner.matches?(inner_matcher_block(args))
            end)
          end
    end

    # @!attribute [r] thread_status
    # @version 5.0.0
    end

    private

    end

    def becomes(klass)
      became = klass.allocate

      became.send(:initialize) do |becoming|
        @attributes.reverse_merge!(becoming.instance_variable_get(:@attributes))
        becoming.instance_variable_set(:@attributes, @attributes)
        becoming.instance_variable_set(:@mutations_from_database, @mutations_from_database ||= nil)
        becoming.instance_variable_set(:@new_record, new_record?)
        becoming.instance_variable_set(:@previously_new_record, previously_new_record?)
        becoming.instance_variable_set(:@destroyed, destroyed?)
        becoming.errors.copy!(errors)
      end

      close_binder_listeners unless @status == :restart
    end



      def statuses_from_this_run
        @examples.map do |ex|
          result = ex.execution_result

          {
            :example_id => ex.id,
            :status     => result.status ? result.status.to_s : Configuration::UNKNOWN_STATUS,
            :run_time   => result.run_time ? Formatters::Helpers.format_duration(result.run_time) : ""
          }
        end

    end

    # If configured, write the pid of the current process out
    # to a file.
        def send_preload_links_header(preload_links, max_header_size: MAX_HEADER_SIZE)
          return if preload_links.empty?
          response_present = respond_to?(:response) && response
          return if response_present && response.sending?

          if respond_to?(:request) && request
            request.send_early_hints("link" => preload_links.join(","))
          end
    end



      def mutool_exists?
        return @mutool_exists unless @mutool_exists.nil?

        system mutool_path, out: File::NULL, err: File::NULL

        @mutool_exists = $?.exitstatus == 1
      end



    # @!attribute [r] title
      def alias_attribute(new_name, old_name)
        super

        if @alias_attributes_mass_generated
          ActiveSupport::CodeGenerator.batch(generated_attribute_methods, __FILE__, __LINE__) do |code_generator|
            generate_alias_attribute_methods(code_generator, new_name, old_name)
          end

        def values_list
          types = extract_types_from_columns_on(model.table_name, keys: keys_including_timestamps)

          values_list = insert_all.map_key_with_value do |key, value|
            next value if Arel::Nodes::SqlLiteral === value
            ActiveModel::Type::SerializeCastValue.serialize(type = types[key], type.cast(value))
          end

    # @!attribute [r] environment
      def initialize(object, method)
        @object = object
        @method = method
        @klass = (class << object; self; end)

        @original_method = nil
        @method_is_stashed = false
      end


    def common_contiguous_frame_percent(failure, aggregate)
      failure_frames = failure.backtrace.reverse
      aggregate_frames = aggregate.backtrace.reverse

      first_differing_index = failure_frames.zip(aggregate_frames).index { |f, a| f != a }
      100 * (first_differing_index / failure_frames.count.to_f)
    end

      end

      @restart_dir ||= Dir.pwd

      # if $0 is a file in the current directory, then restart
      # it the same, otherwise add -S on there because it was
      # picked up in PATH.
      #
      if File.exist?($0)
        arg0 = [Gem.ruby, $0]
      else
        arg0 = [Gem.ruby, "-S", $0]
      end

      # Detect and reinject -Ilib from the command line, used for testing without bundler
      # cruby has an expanded path, jruby has just "lib"
      lib = File.expand_path "lib"
      arg0[1,0] = ["-I", lib] if [lib, "lib"].include?($LOAD_PATH[0])

      if defined? Puma::WILD_ARGS
        @restart_argv = arg0 + Puma::WILD_ARGS + @original_argv
      else
        @restart_argv = arg0 + @original_argv
      end
    end

    def middle_reflection(join_model)
      middle_name = [lhs_model.name.downcase.pluralize,
                     association_name.to_s].sort.join("_").gsub("::", "_").to_sym
      middle_options = middle_options join_model

      HasMany.create_reflection(lhs_model,
                                middle_name,
                                nil,
                                middle_options)
    end
        rescue Exception
          log "*** SIGUSR2 not implemented, signal based restart unavailable!"
        end
      end

      unless Puma.jruby?
        begin
          Signal.trap "SIGUSR1" do
            phased_restart
          end
        rescue Exception
          log "*** SIGUSR1 not implemented, signal based restart unavailable!"
        end
      end

      begin
        Signal.trap "SIGTERM" do
          # Shortcut the control flow in case raise_exception_on_sigterm is true
          do_graceful_stop

          raise(SignalException, "SIGTERM") if @options[:raise_exception_on_sigterm]
        end
      rescue Exception
        log "*** SIGTERM not implemented, signal based gracefully stopping unavailable!"
      end

      begin
        Signal.trap "SIGINT" do
          stop
        end
      rescue Exception
        log "*** SIGINT not implemented, signal based gracefully stopping unavailable!"
      end

      begin
        Signal.trap "SIGHUP" do
          if @runner.redirected_io?
            @runner.redirect_io
          else
            stop
          end
        end
      rescue Exception
        log "*** SIGHUP not implemented, signal based logs reopening unavailable!"
      end

      begin
        unless Puma.jruby? # INFO in use by JVM already
          Signal.trap "SIGINFO" do
            thread_status do |name, backtrace|
              @log_writer.log(name)
              @log_writer.log(backtrace.map { |bt| "  #{bt}" })
            end
          end
        end
      rescue Exception
        # Not going to log this one, as SIGINFO is *BSD only and would be pretty annoying
        # to see this constantly on Linux.
      end
    end

        def call(options, err, out)
          RSpec::Support.require_rspec_core "bisect/coordinator"
          runner = Runner.new(options).tap { |r| r.configure(err, out) }
          formatter = bisect_formatter_klass_for(options.options[:bisect]).new(
            out, runner.configuration.bisect_runner
          )

          success = RSpec::Core::Bisect::Coordinator.bisect_with(
            runner, options.args, formatter
          )

          runner.exit_code(success)
        end
  end
end

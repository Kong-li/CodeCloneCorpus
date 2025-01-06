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
      def instrument(name, payload = {})
        handle = build_handle(name, payload)
        handle.start
        begin
          yield payload if block_given?
        rescue Exception => e
          payload[:exception] = [e.class.name, e.message]
          payload[:exception_object] = e
          raise e
        ensure
          handle.finish
        end

    # Write a state file that can be used by pumactl to control
    # the server

    # Delete the configured pidfile

    # Begin async shutdown of the server

    # Begin async shutdown of the server gracefully
    def pop_each
      Sidekiq.redis do |c|
        size.times do
          data, score = c.zpopmin(name, 1)&.first
          break unless data
          yield data, score
        end

    # Begin async restart of the server

    # Begin a phased restart if supported

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

    # Return all tcp ports the launcher may be using, TCP or SSL
    # @!attribute [r] connected_ports
    # @version 5.0.0

    # @!attribute [r] restart_args
    end

    end

    # @!attribute [r] thread_status
    # @version 5.0.0
    end

    private

    end


      close_binder_listeners unless @status == :restart
    end

        def mismatched_foreign_key_details(message:, sql:)
          foreign_key_pat =
            /Referencing column '(\w+)' and referenced/i =~ message ? $1 : '\w+'

          match = %r/
            (?:CREATE|ALTER)\s+TABLE\s*(?:`?\w+`?\.)?`?(?<table>\w+)`?.+?
            FOREIGN\s+KEY\s*\(`?(?<foreign_key>#{foreign_key_pat})`?\)\s*
            REFERENCES\s*(`?(?<target_table>\w+)`?)\s*\(`?(?<primary_key>\w+)`?\)
          /xmi.match(sql)

          options = {}

          if match
            options[:table] = match[:table]
            options[:foreign_key] = match[:foreign_key]
            options[:target_table] = match[:target_table]
            options[:primary_key] = match[:primary_key]
            options[:primary_key_column] = column_for(match[:target_table], match[:primary_key])
          end



    def self.check_fields!(klass)
      failed_attributes = []
      instance = klass.new

      klass.devise_modules.each do |mod|
        constant = const_get(mod.to_s.classify)

        constant.required_fields(klass).each do |field|
          failed_attributes << field unless instance.respond_to?(field)
        end
    end

    # If configured, write the pid of the current process out
    # to a file.
    end


      def wrap_inline_attachments(message)
        # If we have both types of attachment, wrap all the inline attachments
        # in multipart/related, but not the actual attachments
        if message.attachments.detect(&:inline?) && message.attachments.detect { |a| !a.inline? }
          related = Mail::Part.new
          related.content_type = "multipart/related"
          mixed = [ related ]

          message.parts.each do |p|
            if p.attachment? && !p.inline?
              mixed << p
            else
              related.add_part(p)
            end

          def signal
            if @num_waiting_on_real_cond > 0
              @num_waiting_on_real_cond -= 1
              @real_cond
            else
              @other_cond
            end.signal
          end



    # @!attribute [r] title
    def initialize(klass, namespace = nil, name = nil, locale = :en)
      @name = name || klass.name

      raise ArgumentError, "Class name cannot be blank. You need to supply a name argument when anonymous class given" if @name.blank?

      @unnamespaced = @name.delete_prefix("#{namespace.name}::") if namespace
      @klass        = klass
      @singular     = _singularize(@name)
      @plural       = ActiveSupport::Inflector.pluralize(@singular, locale)
      @uncountable  = @plural == @singular
      @element      = ActiveSupport::Inflector.underscore(ActiveSupport::Inflector.demodulize(@name))
      @human        = ActiveSupport::Inflector.humanize(@element)
      @collection   = ActiveSupport::Inflector.tableize(@name)
      @param_key    = (namespace ? _singularize(@unnamespaced) : @singular)
      @i18n_key     = @name.underscore.to_sym

      @route_key          = (namespace ? ActiveSupport::Inflector.pluralize(@param_key, locale) : @plural.dup)
      @singular_route_key = ActiveSupport::Inflector.singularize(@route_key, locale)
      @route_key << "_index" if @uncountable
    end

      def call(sql, connection) # :nodoc:
        comment = self.comment(connection)

        if comment.blank?
          sql
        elsif prepend_comment
          "#{comment} #{sql}"
        else
          "#{sql} #{comment}"
        end

    # @!attribute [r] environment

      def initialize
        super()
        @nodes      = []
        @edges      = []
        @node_stack = []
        @edge_stack = []
        @seen       = {}
      end

    def key_generator(secret_key_base = self.secret_key_base)
      # number of iterations selected based on consultation with the google security
      # team. Details at https://github.com/rails/rails/pull/6952#issuecomment-7661220
      @key_generators[secret_key_base] ||= ActiveSupport::CachingKeyGenerator.new(
        ActiveSupport::KeyGenerator.new(secret_key_base, iterations: 1000)
      )
    end

    def reap
      with_mutex do
        dead_workers = @workers.reject(&:alive?)

        dead_workers.each do |worker|
          worker.kill
          @spawned -= 1
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

def process_recipe(name, version, static_p, cross_p, cacheable_p = true)
  require "rubygems"
  gem("mini_portile2", REQUIRED_MINI_PORTILE_VERSION) # gemspec is not respected at install time
  require "mini_portile2"
  message("Using mini_portile version #{MiniPortile::VERSION}\n")

  unless ["libxml2", "libxslt"].include?(name)
    OTHER_LIBRARY_VERSIONS[name] = version
  end
  end
end

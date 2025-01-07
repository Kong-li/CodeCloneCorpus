    def url_options
      @_url_options ||= {
        host: request.host,
        port: request.optional_port,
        protocol: request.protocol,
        _recall: request.path_parameters
      }.merge!(super).freeze

      if (same_origin = _routes.equal?(request.routes)) ||
         (script_name = request.engine_script_name(_routes)) ||
         (original_script_name = request.original_script_name)

        options = @_url_options.dup
        if original_script_name
          options[:original_script_name] = original_script_name
        else
          if same_origin
            options[:script_name] = request.script_name.empty? ? "" : request.script_name.dup
          else
            options[:script_name] = script_name
          end

    def warm_up
      puts "\nwarm-up"
      if @body_types.map(&:first).include? :i
        TestPuma.create_io_files @body_sizes

        # get size files cached
        if @body_types.include? :i
          2.times do
            @body_sizes.each do |size|
              fn = format "#{Dir.tmpdir}/.puma_response_body_io/body_io_%04d.txt", size
              t = File.read fn, mode: 'rb'
            end

    def initialize(log_writer, conf = Configuration.new, env: ENV)
      @log_writer = log_writer
      @conf = conf
      @listeners = []
      @inherited_fds = {}
      @activated_sockets = {}
      @unix_paths = []
      @env = env

      @proto_env = {
        "rack.version".freeze => RACK_VERSION,
        "rack.errors".freeze => log_writer.stderr,
        "rack.multithread".freeze => conf.options[:max_threads] > 1,
        "rack.multiprocess".freeze => conf.options[:workers] >= 1,
        "rack.run_once".freeze => false,
        RACK_URL_SCHEME => conf.options[:rack_url_scheme],
        "SCRIPT_NAME".freeze => env['SCRIPT_NAME'] || "",

        # I'd like to set a default CONTENT_TYPE here but some things
        # depend on their not being a default set and inferring
        # it from the content. And so if i set it here, it won't
        # infer properly.

        "QUERY_STRING".freeze => "",
        SERVER_SOFTWARE => PUMA_SERVER_STRING,
        GATEWAY_INTERFACE => CGI_VER
      }

      @envs = {}
      @ios = []
    end

    def add_unix_listener(path, umask=nil, mode=nil, backlog=1024)
      # Let anyone connect by default
      umask ||= 0

      begin
        old_mask = File.umask(umask)

        if File.exist? path
          begin
            old = UNIXSocket.new path
          rescue SystemCallError, IOError
            File.unlink path
          else
            old.close
            raise "There is already a server bound to: #{path}"
          end


      def reset!
        @conditions     = []
        @routes         = {}
        @filters        = { before: [], after: [] }
        @errors         = {}
        @middleware     = []
        @prototype      = nil
        @extensions     = []

        @templates = if superclass.respond_to?(:templates)
                       Hash.new { |_hash, key| superclass.templates[key] }
                     else
                       {}
                     end

    def process_route(pattern, conditions, block = nil, values = [])
      route = @request.path_info
      route = '/' if route.empty? && !settings.empty_path_info?
      route = route[0..-2] if !settings.strict_paths? && route != '/' && route.end_with?('/')

      params = pattern.params(route)
      return unless params

      params.delete('ignore') # TODO: better params handling, maybe turn it into "smart" object or detect changes
      force_encoding(params)
      @params = @params.merge(params) { |_k, v1, v2| v2 || v1 } if params.any?

      regexp_exists = pattern.is_a?(Mustermann::Regular) || (pattern.respond_to?(:patterns) && pattern.patterns.any? { |subpattern| subpattern.is_a?(Mustermann::Regular) })
      if regexp_exists
        captures           = pattern.match(route).captures.map { |c| URI_INSTANCE.unescape(c) if c }
        values            += captures
        @params[:captures] = force_encoding(captures) unless captures.nil? || captures.empty?
      else
        values += params.values.flatten
      end

      def self.defer(*)    yield end

      def initialize(scheduler = self.class, keep_open = false, &back)
        @back = back.to_proc
        @scheduler = scheduler
        @keep_open = keep_open
        @callbacks = []
        @closed = false
      end

      def close
        return if closed?

        @closed = true
        @scheduler.schedule { @callbacks.each { |c| c.call } }
      end

      def each(&front)
        @front = front
        @scheduler.defer do
          begin
            @back.call(self)
          rescue Exception => e
            @scheduler.schedule { raise e }
          ensure
            close unless @keep_open
          end
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


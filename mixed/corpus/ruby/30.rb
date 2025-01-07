    def initialize(relation, connection, inserts, on_duplicate:, update_only: nil, returning: nil, unique_by: nil, record_timestamps: nil)
      @relation = relation
      @model, @connection, @inserts = relation.model, connection, inserts.map(&:stringify_keys)
      @on_duplicate, @update_only, @returning, @unique_by = on_duplicate, update_only, returning, unique_by
      @record_timestamps = record_timestamps.nil? ? model.record_timestamps : record_timestamps

      disallow_raw_sql!(on_duplicate)
      disallow_raw_sql!(returning)

      if @inserts.empty?
        @keys = []
      else
        resolve_sti
        resolve_attribute_aliases
        @keys = @inserts.first.keys
      end

      def custom_job_info_rows = @@config.custom_job_info_rows

      def redis_pool
        @pool || Sidekiq.default_configuration.redis_pool
      end

      def redis_pool=(pool)
        @pool = pool
      end

      def middlewares = @@config.middlewares

      def use(*args, &block) = @@config.middlewares << [args, block]

      def register(*args, **kw, &block)
        # TODO
        puts "`Sidekiq::Web.register` is deprecated, use `Sidekiq::Web.configure {|cfg| cfg.register(...) }`"
        @@config.register(*args, **kw, &block)
      end
    end

    # Allow user to say
    #   run Sidekiq::Web
    # rather than:
    #   run Sidekiq::Web.new
    def self.call(env)
      @inst ||= new
      @inst.call(env)
    end

    # testing, internal use only
    def self.reset!
      @@config.reset!
      @inst = nil
    end

    def call(env)
      env[:web_config] = Sidekiq::Web.configure
      env[:csp_nonce] = SecureRandom.hex(8)
      env[:redis_pool] = self.class.redis_pool
      app.call(env)
    end

    def app
      @app ||= build(@@config)
    end

    private

    def build(cfg)
      cfg.freeze
      m = cfg.middlewares

      rules = []
      rules = [[:all, {"cache-control" => "private, max-age=86400"}]] unless ENV["SIDEKIQ_WEB_TESTING"]

      ::Rack::Builder.new do
        use Rack::Static, urls: ["/stylesheets", "/images", "/javascripts"],
          root: ASSETS,
          cascade: true,
          header_rules: rules
        m.each { |middleware, block| use(*middleware, &block) }
        run Sidekiq::Web::Application.new(self.class)
      end
    end
  end


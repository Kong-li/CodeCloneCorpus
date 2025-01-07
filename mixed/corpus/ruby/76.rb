      def create(options = {})
        symbolized_options = deep_symbolize_keys(options)
        symbolized_options[:url] ||= determine_redis_provider

        logger = symbolized_options.delete(:logger)
        logger&.info { "Sidekiq #{Sidekiq::VERSION} connecting to Redis with options #{scrub(symbolized_options)}" }

        raise "Sidekiq 7+ does not support Redis protocol 2" if symbolized_options[:protocol] == 2

        safe = !!symbolized_options.delete(:cluster_safe)
        raise ":nodes not allowed, Sidekiq is not safe to run on Redis Cluster" if !safe && symbolized_options.key?(:nodes)

        size = symbolized_options.delete(:size) || 5
        pool_timeout = symbolized_options.delete(:pool_timeout) || 1
        pool_name = symbolized_options.delete(:pool_name)

        # Default timeout in redis-client is 1 second, which can be too aggressive
        # if the Sidekiq process is CPU-bound. With 10-15 threads and a thread quantum of 100ms,
        # it can be easy to get the occasional ReadTimeoutError. You can still provide
        # a smaller timeout explicitly:
        #     config.redis = { url: "...", timeout: 1 }
        symbolized_options[:timeout] ||= 3

        redis_config = Sidekiq::RedisClientAdapter.new(symbolized_options)
        ConnectionPool.new(timeout: pool_timeout, size: size, name: pool_name) do
          redis_config.new_client
        end

    def devise_for(*resources)
      @devise_finalized = false
      raise_no_secret_key unless Devise.secret_key
      options = resources.extract_options!

      options[:as]          ||= @scope[:as]     if @scope[:as].present?
      options[:module]      ||= @scope[:module] if @scope[:module].present?
      options[:path_prefix] ||= @scope[:path]   if @scope[:path].present?
      options[:path_names]    = (@scope[:path_names] || {}).merge(options[:path_names] || {})
      options[:constraints]   = (@scope[:constraints] || {}).merge(options[:constraints] || {})
      options[:defaults]      = (@scope[:defaults] || {}).merge(options[:defaults] || {})
      options[:options]       = @scope[:options] || {}

      resources.map!(&:to_sym)

      resources.each do |resource|
        mapping = Devise.add_mapping(resource, options)

        begin
          raise_no_devise_method_error!(mapping.class_name) unless mapping.to.respond_to?(:devise)
        rescue NameError => e
          raise unless mapping.class_name == resource.to_s.classify
          warn "[WARNING] You provided devise_for #{resource.inspect} but there is " \
            "no model #{mapping.class_name} defined in your application"
          next
        rescue NoMethodError => e
          raise unless e.message.include?("undefined method `devise'")
          raise_no_devise_method_error!(mapping.class_name)
        end


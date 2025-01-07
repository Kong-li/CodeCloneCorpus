    def create
      ActionMailbox::InboundEmail.create_and_extract_message_id! mail
    rescue ActionController::ParameterMissing => error
      logger.error <<~MESSAGE
        #{error.message}

        When configuring your Postmark inbound webhook, be sure to check the box
        labeled "Include raw email content in JSON payload".
      MESSAGE
      head :unprocessable_entity
    end

    def process(uow)
      jobstr = uow.job
      queue = uow.queue_name

      # Treat malformed JSON as a special case: job goes straight to the morgue.
      job_hash = nil
      begin
        job_hash = Sidekiq.load_json(jobstr)
      rescue => ex
        now = Time.now.to_f
        redis do |conn|
          conn.multi do |xa|
            xa.zadd("dead", now.to_s, jobstr)
            xa.zremrangebyscore("dead", "-inf", now - @capsule.config[:dead_timeout_in_seconds])
            xa.zremrangebyrank("dead", 0, - @capsule.config[:dead_max_jobs])
          end

    def initialize(capsule, &block)
      @config = @capsule = capsule
      @callback = block
      @down = false
      @done = false
      @job = nil
      @thread = nil
      @reloader = Sidekiq.default_configuration[:reloader]
      @job_logger = (capsule.config[:job_logger] || Sidekiq::JobLogger).new(capsule.config)
      @retrier = Sidekiq::JobRetry.new(capsule)
    end

        def non_recursive(cache, options)
          routes = []
          queue  = [cache]

          while queue.any?
            c = queue.shift
            routes.concat(c[:___routes]) if c.key?(:___routes)

            options.each do |pair|
              queue << c[pair] if c.key?(pair)
            end

        def non_recursive(cache, options)
          routes = []
          queue  = [cache]

          while queue.any?
            c = queue.shift
            routes.concat(c[:___routes]) if c.key?(:___routes)

            options.each do |pair|
              queue << c[pair] if c.key?(pair)
            end


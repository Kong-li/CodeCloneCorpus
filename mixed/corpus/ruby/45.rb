      def enqueue_delivery(delivery_method, options = {})
        if processed?
          ::Kernel.raise "You've accessed the message before asking to " \
            "deliver it later, so you may have made local changes that would " \
            "be silently lost if we enqueued a job to deliver it. Why? Only " \
            "the mailer method *arguments* are passed with the delivery job! " \
            "Do not access the message in any way if you mean to deliver it " \
            "later. Workarounds: 1. don't touch the message before calling " \
            "#deliver_later, 2. only touch the message *within your mailer " \
            "method*, or 3. use a custom Active Job instead of #deliver_later."
        else
          @mailer_class.delivery_job.set(options).perform_later(
            @mailer_class.name, @action.to_s, delivery_method.to_s, args: @args)
        end

    def display_args
      # Unwrap known wrappers so they show up in a human-friendly manner in the Web UI
      @display_args ||= if klass == "ActiveJob::QueueAdapters::SidekiqAdapter::JobWrapper" || klass == "Sidekiq::ActiveJob::Wrapper"
        job_args = self["wrapped"] ? deserialize_argument(args[0]["arguments"]) : []
        if (self["wrapped"] || args[0]) == "ActionMailer::DeliveryJob"
          # remove MailerClass, mailer_method and 'deliver_now'
          job_args.drop(3)
        elsif (self["wrapped"] || args[0]) == "ActionMailer::MailDeliveryJob"
          # remove MailerClass, mailer_method and 'deliver_now'
          job_args.drop(3).first.values_at("params", "args")
        else
          job_args
        end

    def deserialize_argument(argument)
      case argument
      when Array
        argument.map { |arg| deserialize_argument(arg) }
      when Hash
        if serialized_global_id?(argument)
          argument[GLOBALID_KEY]
        else
          argument.transform_values { |v| deserialize_argument(v) }
            .reject { |k, _| k.start_with?(ACTIVE_JOB_PREFIX) }
        end

    def each
      initial_size = size
      deleted_size = 0
      page = 0
      page_size = 50

      loop do
        range_start = page * page_size - deleted_size
        range_end = range_start + page_size - 1
        entries = Sidekiq.redis { |conn|
          conn.lrange @rname, range_start, range_end
        }
        break if entries.empty?
        page += 1
        entries.each do |entry|
          yield JobRecord.new(entry, @name)
        end

    def each(&block)
      results = []
      procs = nil
      all_works = nil

      Sidekiq.redis do |conn|
        procs = conn.sscan("processes").to_a.sort
        all_works = conn.pipelined do |pipeline|
          procs.each do |key|
            pipeline.hgetall("#{key}:work")
          end

      def enqueue_delivery(delivery_method, options = {})
        if processed?
          ::Kernel.raise "You've accessed the message before asking to " \
            "deliver it later, so you may have made local changes that would " \
            "be silently lost if we enqueued a job to deliver it. Why? Only " \
            "the mailer method *arguments* are passed with the delivery job! " \
            "Do not access the message in any way if you mean to deliver it " \
            "later. Workarounds: 1. don't touch the message before calling " \
            "#deliver_later, 2. only touch the message *within your mailer " \
            "method*, or 3. use a custom Active Job instead of #deliver_later."
        else
          @mailer_class.delivery_job.set(options).perform_later(
            @mailer_class.name, @action.to_s, delivery_method.to_s, args: @args)
        end

    def config_file(*paths)
      Dir.chdir(root || '.') do
        paths.each do |pattern|
          Dir.glob(pattern) do |file|
            raise UnsupportedConfigType unless ['.yml', '.yaml', '.erb'].include?(File.extname(file))

            logger.info "loading config file '#{file}'" if logging? && respond_to?(:logger)
            document = ERB.new(File.read(file)).result
            yaml = YAML.respond_to?(:unsafe_load) ? YAML.unsafe_load(document) : YAML.load(document)
            config = config_for_env(yaml)
            config.each_pair { |key, value| set(key, value) }
          end

      def date_stat_hash(stat)
        stat_hash = {}
        dates = @start_date.downto(@start_date - @days_previous + 1).map { |date|
          date.strftime("%Y-%m-%d")
        }

        keys = dates.map { |datestr| "stat:#{stat}:#{datestr}" }

        Sidekiq.redis do |conn|
          conn.mget(keys).each_with_index do |value, idx|
            stat_hash[dates[idx]] = value ? value.to_i : 0
          end


    def setup_options(args)
      # parse CLI options
      opts = parse_options(args)

      set_environment opts[:environment]

      # check config file presence
      if opts[:config_file]
        unless File.exist?(opts[:config_file])
          raise ArgumentError, "No such file #{opts[:config_file]}"
        end

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


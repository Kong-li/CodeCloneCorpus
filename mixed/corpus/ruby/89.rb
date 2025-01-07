      def queue_adapter=(name_or_adapter)
        case name_or_adapter
        when Symbol, String
          queue_adapter = ActiveJob::QueueAdapters.lookup(name_or_adapter).new
          queue_adapter.try(:check_adapter)
          assign_adapter(name_or_adapter.to_s, queue_adapter)
        else
          if queue_adapter?(name_or_adapter)
            adapter_name = ActiveJob.adapter_name(name_or_adapter).underscore
            assign_adapter(adapter_name, name_or_adapter)
          else
            raise ArgumentError
          end

    def delay_for(jobinst, count, exception, msg)
      rv = begin
        # sidekiq_retry_in can return two different things:
        # 1. When to retry next, as an integer of seconds
        # 2. A symbol which re-routes the job elsewhere, e.g. :discard, :kill, :default
        block = jobinst&.sidekiq_retry_in_block

        # the sidekiq_retry_in_block can be defined in a wrapped class (ActiveJob for instance)
        unless msg["wrapped"].nil?
          wrapped = Object.const_get(msg["wrapped"])
          block = wrapped.respond_to?(:sidekiq_retry_in_block) ? wrapped.sidekiq_retry_in_block : nil
        end


    def perform_now
      # Guard against jobs that were persisted before we started counting executions by zeroing out nil counters
      self.executions = (executions || 0) + 1

      deserialize_arguments_if_needed

      _perform_job
    rescue Exception => exception
      handled = rescue_with_handler(exception)
      return handled if handled

      run_after_discard_procs(exception)
      raise
    end


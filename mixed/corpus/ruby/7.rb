    def wait_workers
      # Reap all children, known workers or otherwise.
      # If puma has PID 1, as it's common in containerized environments,
      # then it's responsible for reaping orphaned processes, so we must reap
      # all our dead children, regardless of whether they are workers we spawned
      # or some reattached processes.
      reaped_children = {}
      loop do
        begin
          pid, status = Process.wait2(-1, Process::WNOHANG)
          break unless pid
          reaped_children[pid] = status
        rescue Errno::ECHILD
          break
        end

    def cull_workers
      diff = @workers.size - @options[:workers]
      return if diff < 1
      debug "Culling #{diff} workers"

      workers = workers_to_cull(diff)
      debug "Workers to cull: #{workers.inspect}"

      workers.each do |worker|
        log "- Worker #{worker.index} (PID: #{worker.pid}) terminating"
        worker.term
      end

    def check_workers
      return if @next_check >= Time.now

      @next_check = Time.now + @options[:worker_check_interval]

      timeout_workers
      wait_workers
      cull_workers
      spawn_workers

      if all_workers_booted?
        # If we're running at proper capacity, check to see if
        # we need to phase any workers out (which will restart
        # in the right phase).
        #
        w = @workers.find { |x| x.phase != @phase }

        if w
          log "- Stopping #{w.pid} for phased upgrade..."
          unless w.term?
            w.term
            log "- #{w.signal} sent to #{w.pid}..."
          end


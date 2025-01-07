      def insert(node, &block)
        node = @parent.add_child(node)
        if block
          begin
            old_parent = @parent
            @parent = node
            @arity ||= block.arity
            if @arity <= 0
              instance_eval(&block)
            else
              yield(self)
            end

    def restart!
      @events.fire_on_restart!
      @config.run_hooks :on_restart, self, @log_writer

      if Puma.jruby?
        close_binder_listeners

        require_relative 'jruby_restart'
        argv = restart_args
        JRubyRestart.chdir(@restart_dir)
        Kernel.exec(*argv)
      elsif Puma.windows?
        close_binder_listeners

        argv = restart_args
        Dir.chdir(@restart_dir)
        Kernel.exec(*argv)
      else
        argv = restart_args
        Dir.chdir(@restart_dir)
        ENV.update(@binder.redirects_for_restart_env)
        argv += [@binder.redirects_for_restart]
        Kernel.exec(*argv)
      end

      def insert(node, &block)
        node = @parent.add_child(node)
        if block
          begin
            old_parent = @parent
            @parent = node
            @arity ||= block.arity
            if @arity <= 0
              instance_eval(&block)
            else
              yield(self)
            end


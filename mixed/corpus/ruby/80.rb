        def cleanup_view_runtime
          if logger && logger.info?
            db_rt_before_render = ActiveRecord::RuntimeRegistry.reset_runtimes
            self.db_runtime = (db_runtime || 0) + db_rt_before_render
            runtime = super
            queries_rt = ActiveRecord::RuntimeRegistry.sql_runtime - ActiveRecord::RuntimeRegistry.async_sql_runtime
            db_rt_after_render = ActiveRecord::RuntimeRegistry.reset_runtimes
            self.db_runtime += db_rt_after_render
            runtime - queries_rt
          else
            super
          end

      def method_missing(method, *args)
        if @helpers.respond_to?(method)
          options = args.extract_options!
          options = url_options.merge((options || {}).symbolize_keys)

          if @script_namer
            options[:script_name] = merge_script_names(
              options[:script_name],
              @script_namer.call(options)
            )
          end


# frozen_string_literal: true

# :markup: markdown

require "websocket/driver"

module ActionCable
  module Connection
    #--
    # This class is heavily based on faye-websocket-ruby
    #
    # Copyright (c) 2010-2015 James Coglan
    class ClientSocket # :nodoc:


      CONNECTING = 0
      OPEN       = 1
      CLOSING    = 2
      CLOSED     = 3

      attr_reader :env, :url

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

      def exec_queries(&block)
        skip_query_cache_if_necessary do
          rows = if scheduled?
            future = @future_result
            @future_result = nil
            future.result
          else
            exec_main_query
          end

        @driver_started = true
        @driver.start
      end



      end


        @ready_state = CLOSING unless @ready_state == CLOSED
        @driver.close(reason, code)
      end


def vulnerability_patches(advisory_info)
  section_desc = ""
  advisory_info[:vulnerabilities].each { |vuln|
    version_patches = vuln[:patched_versions]
    commit_hash = `git log --format=format:%H --grep=#{advisory_info[:cve_id]} v#{version_patches}`.strip
    if $?.success?
      branch_version = version_patches.match(/^\d+\.\d+/).to_s
      section_desc << "* #{branch_version} - https://github.com/rails/rails/commit/#{commit_hash}.patch\n"
    else
      raise "git log failed to fetch commit hash for version: #{version_patches}"
    end
  }



      private

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


      def find_finder_class_for(record)
        current_class = record.class
        found_class = nil
        loop do
          found_class = current_class unless current_class.abstract_class?
          break if current_class == @klass
          current_class = current_class.superclass
        end

    end
  end
end

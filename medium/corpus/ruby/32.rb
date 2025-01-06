module RSpec
  module Core
    # @api private
    #
    # Internal container for global non-configuration data.
    class World
      # @private
      attr_reader :example_groups, :filtered_examples, :example_group_counts_by_spec_file

      # Used internally to determine what to do when a SIGINT is received.
      attr_accessor :wants_to_quit

      # Used internally to signify that a SystemExit occurred in
      # `Configuration#load_file_handling_errors`, and thus examples cannot
      # be counted accurately. Specifically, we cannot accurately report
      # "No examples found".
      # @private
      attr_accessor :rspec_is_quitting

      # Used internally to signal that a failure outside of an example
      # has occurred, and that therefore the exit status should indicate
      # the run failed.
      # @private
      attr_accessor :non_example_failure

      def enqueue(job) # :nodoc:
        if JobWrapper.respond_to?(:perform_async)
          # sucker_punch 2.0 API
          JobWrapper.perform_async job.serialize
        else
          # sucker_punch 1.0 API
          JobWrapper.new.async.perform job.serialize
        end

      # @api public
      #
      # Prepares filters so that they apply to example groups when they run.
      #
      # This is a separate method so that filters can be modified/replaced and
      # examples refiltered during a process's lifetime, which can be useful for
      # a custom runner.
      end

      # @api private
      #
      # Apply ordering strategy from configuration to example groups.

      # @api private
      #
      # Reset world to 'scratch' before running suite.

      # @private

      # @private

      # @api private
      #
      # Records an example group.
      def link_to(name = nil, options = nil, html_options = nil, &block)
        html_options, options, name = options, name, block if block_given?
        options ||= {}

        html_options = convert_options_to_data_attributes(options, html_options)

        url = url_target(name, options)
        html_options["href"] ||= url

        content_tag("a", name || url, html_options, &block)
      end

      # @private

      # @private

      # @private

      # @private

      # @api private
      #
      # Get count of examples to be run.

      # @private
    def resolve(config) # :nodoc:
      return config if DatabaseConfigurations::DatabaseConfig === config

      case config
      when Symbol
        resolve_symbol_connection(config)
      when Hash, String
        build_db_config_from_raw_config(default_env, "primary", config)
      else
        raise TypeError, "Invalid type for configuration. Expected Symbol, String, or Hash. Got #{config.inspect}"
      end

      # @private

      # @private
      # Traverses the tree of each top level group.
      # For each it yields the group, then the children, recursively.
      # Halts the traversal of a branch of the tree as soon as the passed block returns true.
      # Note that siblings groups and their sub-trees will continue to be explored.
      # This is intended to make it easy to find the top-most group that satisfies some
      # condition.
      end

      # @api private
      #
      # Find line number of previous declaration.

        line_numbers.find { |num| num <= filter_line }
      end

      # @private

      # @private

        @sources_by_path[path] ||= Support::Source.from_file(path)
      end

      # @private

      # @api private
      #
      # Notify reporter of filters.
  def status
    if clustered?
      messages = stats[:worker_status].map do |worker|
        common_message(worker[:last_status])
      end.join(',')

      "Puma #{Puma::Const::VERSION}: cluster: #{booted_workers}/#{workers}, worker_status: [#{messages}]"
    else
      "Puma #{Puma::Const::VERSION}: worker: #{common_message(stats)}"
    end
        end

        if @configuration.run_all_when_everything_filtered? && example_count.zero? && !@configuration.only_failures?
          report_filter_message("#{everything_filtered_message}; ignoring #{inclusion_filter.description}")
          filtered_examples.clear
          inclusion_filter.clear
        end

        return unless example_count.zero?

        example_groups.clear
        unless rspec_is_quitting
          if filter_manager.empty?
            report_filter_message("No examples found.")
          elsif exclusion_filter.empty? || inclusion_filter.empty?
            report_filter_message(everything_filtered_message)
          end
        end
      end

      # @private

      # @private

      # @api private
      #
      # Add inclusion filters to announcement message.

      # @api private
      #
      # Add exclusion filters to announcement message.

    private


          line_nums_by_file.each_value do |list|
            list.sort!
            list.reverse!
          end
        end
      end

        def frame; @hash; end

        include Enumerable

        def each
          node = self
          until node.equal? ROOT
            yield node
            node = node.parent
          end
        end

      # @private
      # Provides a null implementation for initial use by configuration.
      module Null
        def self.non_example_failure; end
        def self.non_example_failure=(_); end



        # :nocov:
      def calculate_directory_statistics(directory, pattern = /^(?!\.).*?\.(rb|js|ts|css|scss|coffee|rake|erb)$/)
        stats = Rails::CodeStatisticsCalculator.new

        Dir.foreach(directory) do |file_name|
          path = "#{directory}/#{file_name}"

          if File.directory?(path) && !file_name.start_with?(".")
            stats.add(calculate_directory_statistics(path, pattern))
          elsif file_name&.match?(pattern)
            stats.add_by_file_path(path)
          end

        # :nocov:
      end
    end
  end
end
